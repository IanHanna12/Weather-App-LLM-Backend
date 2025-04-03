import os
import tempfile
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from vosk_service import VoskService
from weather_service import WeatherService
from extractorService import WeatherExtractor

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

vosk_service = VoskService()
weather_extractor = WeatherExtractor()
weather_service = WeatherService(api_url=os.environ.get("BACKEND_API_URL", "http://localhost:8080/api/weather/"))


class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message, websocket):
        await websocket.send_json(message)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data:
                if not vosk_service.is_available():
                    await manager.send_message(
                        {"type": "error", "message": "Speech recognition not available"},
                        websocket
                    )
                    continue

                audio_data = data["bytes"]
                temp_file_path = None

                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                        temp_file_path = temp_file.name
                        temp_file.write(audio_data)

                    transcription_result = vosk_service.transcribe_audio(temp_file_path)

                    if not transcription_result["success"]:
                        await manager.send_message(
                            {"type": "error", "message": transcription_result["error"]},
                            websocket
                        )
                        continue

                    transcribed_text = transcription_result["text"]

                    await manager.send_message(
                        {"type": "transcription", "text": transcribed_text},
                        websocket
                    )

                    if not transcribed_text:
                        await manager.send_message(
                            {"type": "transcription", "text": "Keine Sprache erkannt."},
                            websocket
                        )
                        continue

                    # Use our simple rule-based extractor
                    weather_data = weather_extractor.extract(transcribed_text)
                    weather_data["original_query"] = transcribed_text

                    backend_response = weather_service.get_weather(weather_data)

                    if backend_response["success"]:
                        response_text = backend_response["data"].get("response", "Keine Antwort vom Backend")
                    else:
                        response_text = f"Fehler: {backend_response['message']}"

                    result_text = f"{transcribed_text}\n\nWetterabfrage: {weather_data.get('is_weather_query')}\nOrt: {weather_data.get('location', 'nicht erkannt')}\nZeitraum: {weather_data.get('time_period', 'heute')}\n\nAntwort: {response_text}"

                    await manager.send_message(
                        {"type": "transcription", "text": result_text},
                        websocket
                    )

                except Exception as e:
                    await manager.send_message(
                        {"type": "error", "message": f"Error: {str(e)}"},
                        websocket
                    )
                finally:
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)

            elif "text" in data:
                # Handle text input directly (for testing without audio)
                try:
                    text_query = data["text"]
                    weather_data = weather_extractor.extract(text_query)
                    weather_data["original_query"] = text_query

                    backend_response = weather_service.get_weather(weather_data)

                    if backend_response["success"]:
                        response_text = backend_response["data"].get("response", "Keine Antwort vom Backend")
                    else:
                        response_text = f"Fehler: {backend_response['message']}"

                    result_text = f"{text_query}\n\nWetterabfrage: {weather_data.get('is_weather_query')}\nOrt: {weather_data.get('location', 'nicht erkannt')}\nZeitraum: {weather_data.get('time_period', 'heute')}\n\nAntwort: {response_text}"

                    await manager.send_message(
                        {"type": "transcription", "text": result_text},
                        websocket
                    )

                except Exception as e:
                    await manager.send_message(
                        {"type": "error", "message": f"Error: {str(e)}"},
                        websocket
                    )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


@app.get("/")
async def root():
    return {"message": "Voice Weather API is running"}

@app.websocket("/weather")
async def weather_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Weather WebSocket client connected")

    try:
        while True:
            data = await websocket.receive_json()
            print(f"Received weather request: {data}")

            try:
                if data.get("type") == "get_weather":
                    city = data.get("city", "Heilbronn")
                    days = int(data.get("days", 5))

                    weather_data = {
                        "location": city,
                        "time_period": f"{days} days",
                        "is_weather_query": True
                    }

                    backend_response = weather_service.get_weather(weather_data)

                    if backend_response["success"]:
                        await websocket.send_json({
                            "type": "weather_data",
                            "data": backend_response["data"]
                        })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": backend_response["message"]
                        })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Unknown request type"
                    })
            except Exception as e:
                print(f"Error processing weather request: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Server error: {str(e)}"
                })
    except WebSocketDisconnect:
        print("Weather WebSocket client disconnected")
    except Exception as e:
        print(f"Unexpected error in weather WebSocket: {e}")
