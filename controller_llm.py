import json
import os
import tempfile
import wave
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

load_dotenv()

# Try to import Vosk, but continue if not available
try:
    from vosk import Model, KaldiRecognizer

    vosk_available = True
except ImportError:
    vosk_available = False
    print("Warning: Vosk is not available. Install it with 'pip install vosk'")

app = FastAPI()

# Set up CORS to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files if directory exists
if Path("tts_recordings").exists():
    app.mount("/tts_recordings", StaticFiles(directory="tts_recordings"), name="frontend")

# Load Vosk model if available
model = None
if vosk_available:
    model_path = "model/vosk-model-small-de-zamia-0.3"
    if os.path.exists(model_path):
        model = Model(model_path)
        print(f"Loaded Vosk model from {model_path}")
    else:
        print(f"Warning: Model path does not exist: {model_path}")

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
BACKEND_API_URL = os.environ.get("BACKEND_API_URL", "https://your-backend-api.com/api/weather")


# Simple connection manager for WebSockets
async def send_message(message, websocket):
    await websocket.send_json(message)


class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket):
        self.active_connections.remove(websocket)
        print(f"Client disconnected. Remaining connections: {len(self.active_connections)}")


manager = ConnectionManager()


# Function to analyze text with Ollama
def analyze_text_with_ollama(text):
    print(f"Analyzing text: {text}")

    prompt = f"""
    Extract the weather query information from this German text: '{text}'

    Return ONLY a valid JSON object with this structure:
    {{
        "is_weather_query": true/false,
        "location": "city_name",
        "time_period": "today/tomorrow/week"
    }}

    If it's not a weather query, set is_weather_query to false.
    """

    try:
        # Send request to Ollama
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        )

        if response.status_code != 200:
            print(f"Ollama error: {response.status_code}")
            return {"is_weather_query": False}

        # Extract JSON from response
        result = response.json()
        response_text = result.get('response', '{}')

        # Find JSON in the response
        import re
        json_match = re.search(r'({.*})', response_text, re.DOTALL)
        if json_match:
            try:
                weather_data = json.loads(json_match.group(1))
                return weather_data
            except:
                return {"is_weather_query": False}
        else:
            return {"is_weather_query": False}

    except Exception as e:
        print(f"Error with Ollama: {str(e)}")
        return {"is_weather_query": False}


# Function to get weather data from backend
def get_weather_from_backend(weather_data):
    if not weather_data.get("is_weather_query", False):
        return {"success": False, "message": "Not a weather query"}

    try:
        # Prepare data for backend
        payload = {
            "query_type": "weather",
            "location": weather_data.get("location", ""),
            "time_period": weather_data.get("time_period", "today"),
            "original_query": weather_data.get("original_query", ""),
            "language": "de"
        }

        # Send request to backend
        response = requests.post(
            BACKEND_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            return {
                "success": True,
                "data": response.json()
            }
        else:
            return {
                "success": False,
                "message": f"Backend error: {response.status_code}"
            }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error: {str(e)}"
        }


# WebSocket endpoint for audio processing
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive()

            # Process audio data if received
            if "bytes" in data:
                # Check if Vosk is available
                if not vosk_available or model is None:
                    await send_message(
                        {"type": "error", "message": "Speech recognition not available"},
                        websocket
                    )
                    continue

                # Save audio to temporary file
                audio_data = data["bytes"]
                temp_file_path = None

                try:
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                        temp_file_path = temp_file.name
                        temp_file.write(audio_data)

                    # Process audio with Vosk
                    wf = wave.open(temp_file_path, "rb")
                    rec = KaldiRecognizer(model, wf.getframerate())

                    # Transcribe audio
                    result = ""
                    while True:
                        data_chunk = wf.readframes(4000)
                        if len(data_chunk) == 0:
                            break
                        if rec.AcceptWaveform(data_chunk):
                            part_result = json.loads(rec.Result())
                            result += part_result.get("text", "") + " "

                    final_result = json.loads(rec.FinalResult())
                    result += final_result.get("text", "")
                    transcribed_text = result.strip()

                    # Send transcription to client
                    await send_message(
                        {"type": "transcription", "text": transcribed_text},
                        websocket
                    )

                    # If no text was recognized, send error
                    if not transcribed_text:
                        await send_message(
                            {"type": "transcription", "text": "Keine Sprache erkannt. Bitte versuchen Sie es erneut."},
                            websocket
                        )
                        continue

                    # Process text with Ollama
                    weather_data = analyze_text_with_ollama(transcribed_text)
                    weather_data["original_query"] = transcribed_text

                    # Get weather from backend
                    backend_response = get_weather_from_backend(weather_data)

                    # Prepare response
                    if backend_response["success"]:
                        response_text = backend_response["data"].get("response", "Keine Antwort vom Backend")
                    else:
                        response_text = f"Fehler: {backend_response['message']}"

                    # Send enhanced response to client
                    result_text = f"{transcribed_text}\n\nWetterabfrage: {weather_data.get('is_weather_query')}\nOrt: {weather_data.get('location', 'nicht erkannt')}\nZeitraum: {weather_data.get('time_period', 'heute')}\n\nAntwort: {response_text}"

                    await send_message(
                        {"type": "transcription", "text": result_text},
                        websocket
                    )

                except Exception as e:
                    await send_message(
                        {"type": "error", "message": f"Error: {str(e)}"},
                        websocket
                    )
                finally:
                    # Clean up temporary file
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket)


# Basic routes
@app.get("/")
async def root():
    return {"message": "Voice Weather API is running. Open /tts_recordings/frontend.html to use the app."}


@app.get("/health")
async def health_check():
    # Check Ollama status
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            ollama_status = "available"
        else:
            ollama_status = f"error: {response.status_code}"
    except Exception as e:
        ollama_status = f"error: {str(e)}"

    # Check backend status
    try:
        backend_url = BACKEND_API_URL.split('/api/')[0] + "/health"
        response = requests.get(backend_url, timeout=5)
        if response.status_code == 200:
            backend_status = "available"
        else:
            backend_status = f"error: {response.status_code}"
    except Exception as e:
        backend_status = f"error: {str(e)}"

    return {
        "status": "ok",
        "vosk_available": vosk_available,
        "model_loaded": model is not None,
        "ollama_status": ollama_status,
        "backend_status": backend_status
    }
