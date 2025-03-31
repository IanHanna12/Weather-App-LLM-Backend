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

try:
    from vosk import Model, KaldiRecognizer

    vosk_available = True
except ImportError:
    vosk_available = False
    print("Warning: Vosk is not available. Install it with 'pip install vosk'")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if Path("tts_recordings").exists():
    app.mount("/tts_recordings", StaticFiles(directory="tts_recordings"), name="frontend")

model = None
if vosk_available:
    model_path = "model/vosk-model-small-de-zamia-0.3"
    if os.path.exists(model_path):
        model = Model(model_path)
        print(f"Loaded Vosk model from {model_path}")
    else:
        print(f"Warning: Model path does not exist: {model_path}")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:12b")

BACKEND_API_URL = os.environ.get("BACKEND_API_URL", "https://your-backend-api.com/api/weather")


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"Client disconnected. Remaining connections: {len(self.active_connections)}")

    async def send_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)


manager = ConnectionManager()


def process_with_ollama(text):
    try:
        print(f"\nOLLAMA RECEIVED: '{text}'")

        prompt = f"""
        Extract the weather query information from this German text: '{text}'

        Return ONLY a valid JSON object with this structure:
        {{
            "is_weather_query": true/false,
            "location": "city_name",
            "time_period": "today/tomorrow/week"
        }}

        If it's not a weather query, set is_weather_query to false.
        Examples of weather queries:
        - "Wie ist das Wetter in Berlin heute?"
        - "Wird es morgen in München regnen?"
        - "Wetterbericht für Hamburg"
        """

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            }
        )

        if response.status_code == 200:
            result = response.json()
            response_text = result.get('response', '{}')
            print(f"OLLAMA RESPONSE: {response_text}")

            import re
            json_match = re.search(r'({.*})', response_text, re.DOTALL)
            if json_match:
                try:
                    weather_data = json.loads(json_match.group(1))
                    print(f"EXTRACTED DATA: {weather_data}")
                    return weather_data
                except json.JSONDecodeError:
                    print(f"JSON PARSE ERROR: {response_text}")
                    return {"is_weather_query": False}
            else:
                print(f"NO JSON FOUND: {response_text}")
                return {"is_weather_query": False}
        else:
            print(f"OLLAMA ERROR: {response.status_code}")
            return {"is_weather_query": False, "error": "Ollama request failed"}
    except Exception as e:
        print(f"EXCEPTION: {str(e)}")
        return {"is_weather_query": False, "error": str(e)}


def forward_to_backend(weather_data):
    try:
        print(f"\nBACKEND RECEIVED: {weather_data}")

        if not weather_data.get("is_weather_query", False):
            print("NOT A WEATHER QUERY")
            return {
                "success": False,
                "message": "Not a weather query",
                "data": None
            }

        payload = {
            "query_type": "weather",
            "location": weather_data.get("location", ""),
            "time_period": weather_data.get("time_period", "today"),
            "original_query": weather_data.get("original_query", ""),
            "language": "de"
        }

        headers = {
            "Content-Type": "application/json"
        }

        print(f"BACKEND REQUEST: {payload}")
        response = requests.post(
            BACKEND_API_URL,
            json=payload,
            headers=headers
        )

        if response.status_code == 200:
            backend_response = response.json()
            print(f"BACKEND RESPONSE: {backend_response}")
            return {
                "success": True,
                "message": "Successfully retrieved from backend",
                "data": backend_response
            }
        else:
            error_message = f"Backend request failed: {response.status_code}"
            print(f"BACKEND ERROR: {error_message}")
            return {
                "success": False,
                "message": error_message,
                "data": None
            }

    except Exception as e:
        error_message = f"Error forwarding to backend: {str(e)}"
        print(f"EXCEPTION: {str(e)}")
        return {
            "success": False,
            "message": error_message,
            "data": None
        }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data:
                if not vosk_available or model is None:
                    await manager.send_message(
                        {"type": "error", "message": "Vosk model not available"},
                        websocket
                    )
                    continue

                audio_data = data["bytes"]

                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                    temp_file_path = temp_file.name
                    temp_file.write(audio_data)

                try:
                    wf = wave.open(temp_file_path, "rb")

                    print(f"Audio file: channels={wf.getnchannels()}, width={wf.getsampwidth()}, "
                          f"rate={wf.getframerate()}, frames={wf.getnframes()}")

                    rec = KaldiRecognizer(model, wf.getframerate())

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

                    await manager.send_message(
                        {"type": "transcription", "text": transcribed_text},
                        websocket
                    )

                    if not transcribed_text:
                        await manager.send_message(
                            {"type": "transcription", "text": "Keine Sprache erkannt. Bitte versuchen Sie es erneut."},
                            websocket
                        )
                        continue

                    weather_data = process_with_ollama(transcribed_text)

                    weather_data["original_query"] = transcribed_text

                    backend_response = forward_to_backend(weather_data)

                    if backend_response["success"]:
                        response_text = backend_response["data"].get("response", "Keine Antwort vom Backend")
                    else:
                        response_text = f"Fehler: {backend_response['message']}"

                    enhanced_transcription = f"{transcribed_text}\n\nWetterabfrage: {weather_data.get('is_weather_query')}\nOrt: {weather_data.get('location', 'nicht erkannt')}\nZeitraum: {weather_data.get('time_period', 'heute')}\n\nAntwort: {response_text}"

                    await manager.send_message(
                        {"type": "transcription", "text": enhanced_transcription},
                        websocket
                    )

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    await manager.send_message(
                        {"type": "error", "message": f"Error processing audio: {str(e)}"},
                        websocket
                    )
                finally:
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        try:
            await manager.send_message(
                {"type": "error", "message": f"WebSocket error: {str(e)}"},
                websocket
            )
        except:
            pass
        manager.disconnect(websocket)


@app.get("/")
async def root():
    return {"message": "Voice Weather API is running. Open /tts_recordings/frontend.html to use the app."}


@app.get("/health")
async def health_check():
    ollama_status = "unknown"
    backend_status = "unknown"

    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model.get("name") for model in models]
            if OLLAMA_MODEL in model_names or any(name.startswith("gemma") for name in model_names):
                ollama_status = "available (Gemma model found)"
            else:
                ollama_status = f"available (Gemma model not found, available models: {', '.join(model_names)})"
        else:
            ollama_status = f"error: {response.status_code}"
    except Exception as e:
        ollama_status = f"error: {str(e)}"

    try:
        response = requests.get(BACKEND_API_URL.split('/api/')[0] + "/health", timeout=5)
        if response.status_code == 200:
            backend_status = "available"
        else:
            backend_status = f"error: {response.status_code}"
    except Exception as e:
        backend_status = f"error: {str(e)}"

    return {
        "status": "ok",
        "vosk_available": vosk_available,
        "model_loaded": True,
        "ollama_status": ollama_status,
        "backend_status": backend_status
    }


