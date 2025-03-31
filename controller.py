import json
import os
import tempfile
import wave

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

try:
    from vosk import Model, KaldiRecognizer

    vosk_available = True
except ImportError:
    vosk_available = False
    print("Warning: Vosk is not available. Install it with 'pip install vosk'")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = None
if vosk_available:
    model_path = "model/vosk-model-small-de-zamia-0.3"
    if os.path.exists(model_path):
        model = Model(model_path)
        print(f"Loaded Vosk model from {model_path}")
    else:
        print(f"Warning: Model path does not exist: {model_path}")


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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        while True:
            # Receive data from the client
            data = await websocket.receive()

            # Check if it's binary data (audio)
            if "bytes" in data:
                if not vosk_available or model is None:
                    await manager.send_message(
                        {"type": "error", "message": "Vosk model not available"},
                        websocket
                    )
                    continue

                # Process the audio data with Vosk
                audio_data = data["bytes"]

                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                    temp_file_path = temp_file.name
                    temp_file.write(audio_data)

                try:
                    # Process with Vosk
                    wf = wave.open(temp_file_path, "rb")

                    # Log file details for debugging
                    print(f"Audio file: channels={wf.getnchannels()}, width={wf.getsampwidth()}, "
                          f"rate={wf.getframerate()}, frames={wf.getnframes()}")

                    # Create a recognizer for the audio sample rate
                    rec = KaldiRecognizer(model, wf.getframerate())

                    # Process the audio in chunks
                    result = ""
                    while True:
                        data_chunk = wf.readframes(4000)
                        if len(data_chunk) == 0:
                            break
                        if rec.AcceptWaveform(data_chunk):
                            part_result = json.loads(rec.Result())
                            result += part_result.get("text", "") + " "

                    # Get the final result
                    final_result = json.loads(rec.FinalResult())
                    result += final_result.get("text", "")

                    # Send the transcription back to the client
                    await manager.send_message(
                        {"type": "transcription", "text": result.strip()},
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
                    # Clean up the temporary file
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


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "vosk_available": vosk_available,
        "model_loaded": model is not None
    }
