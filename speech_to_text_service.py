import asyncio
import json
import os
import wave
import numpy as np
import pyaudio
import websockets
from datetime import datetime
from typing import Dict, Any

from extractorService import WeatherExtractor
from vosk_service import VoskService
from weather_service import WeatherService

# Create directories
RECORDINGS_DIR = "recordings"
TRANSCRIBE_TEXT_DIR = "transcribe_text"
os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(TRANSCRIBE_TEXT_DIR, exist_ok=True)

# Global variables
is_speech = False
finish_result = True

# WebSocket server configuration
WS_HOST = "localhost"
WS_PORT = 8765

# Active WebSocket connections
active_connections = set()


class SpeechToTextService:
    def __init__(self, trigger_word="hey", model_path="model/vosk-model-de-0.6-900K"):
        print("Initializing Speech-to-Text Service")
        self.vosk_service = VoskService(model_path)
        self.weather_extractor = WeatherExtractor()
        self.weather_service = WeatherService(
            api_url=os.environ.get("BACKEND_API_URL", "https://your-backend-api.com/api/weather"))

        # Audio configuration
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1024
        self.THRESHOLD = 500  # Adjust based on your microphone sensitivity
        self.INPUT_DEVICE_INDEX = None  # Use default input device

        # Initialize PyAudio
        self.p = pyaudio.PyAudio()

        # Recording state
        self.frames = []
        self.is_recording = False
        self.triggered = False
        self.trigger_word = trigger_word.lower()

        # Configure silence detection
        self.silence_threshold_seconds = 1.5
        self.frames_per_second = int(self.RATE / self.CHUNK)
        self.silence_frames_threshold = int(self.silence_threshold_seconds * self.frames_per_second)

        print(f"Ready. Listening for trigger word: '{self.trigger_word}'")

    def initialize_stream(self):
        """Initialize audio stream for recording"""
        return self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            input_device_index=self.INPUT_DEVICE_INDEX,
        )

    def is_voice(self, data):
        """Detect if audio chunk contains voice"""
        audio_data = np.frombuffer(data, dtype=np.int16)
        return np.abs(audio_data).mean() > self.THRESHOLD

    def start_recording(self):
        """Start recording audio"""
        self.frames = []
        self.is_recording = True
        print("Recording started")

    def stop_recording(self):
        """Stop recording audio"""
        self.is_recording = False
        print(f"Recording stopped. Captured {len(self.frames)} frames")

    def save_audio(self, filename):
        """Save recorded audio to file"""
        if not self.frames:
            print("No audio to save")
            return False

        try:
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            return True
        except Exception as e:
            print(f"Error saving audio: {e}")
            return False

    def save_frames_to_file(self, frames, filename):
        """Save a list of audio frames to a WAV file"""
        try:
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            return True
        except Exception as e:
            print(f"Error saving frames to file: {e}")
            return False

    async def detect_speech(self, stream):
        """Listen for speech and record when detected"""
        global is_speech
        silence_frames = 0
        is_recording = False
        buffer = []
        buffer_size = 30  # About 1 second of audio

        print("Listening for trigger word...")

        while True:
            # Read audio from microphone
            data = await asyncio.to_thread(stream.read, self.CHUNK, exception_on_overflow=False)

            # Add to buffer
            buffer.append(data)
            if len(buffer) > buffer_size:
                buffer.pop(0)

            # Check if this is speech
            is_speech = self.is_voice(data)

            # If not triggered yet, check for trigger word
            if not self.triggered:
                if is_speech and len(buffer) >= 15:  # Check after collecting some audio
                    # Save buffer to temporary file for transcription
                    temp_file = os.path.join(RECORDINGS_DIR, "temp_trigger.wav")
                    self.save_frames_to_file(buffer, temp_file)

                    # Check for trigger word
                    result = self.vosk_service.transcribe_audio(temp_file)
                    if os.path.exists(temp_file):
                        os.remove(temp_file)

                    if result.get("success") and result.get("text"):
                        text = result.get("text", "").lower()
                        if self.trigger_word in text:
                            print(f"Trigger word detected: '{self.trigger_word}'")
                            self.triggered = True
                            is_recording = True
                            self.start_recording()
                            # Add buffer to recording
                            for frame in buffer:
                                self.frames.append(frame)

                            # Notify frontend that recording has started
                            await self.broadcast_to_clients({
                                "type": "status",
                                "message": "Recording started"
                            })

            # If triggered, handle recording
            elif self.triggered:
                # Add current frame to recording
                if self.is_recording:
                    self.frames.append(data)

                if is_speech:
                    # Reset silence counter when speech detected
                    silence_frames = 0
                else:
                    # Count silence frames
                    silence_frames += 1
                    if silence_frames % 10 == 0:
                        print(f"Silence: {silence_frames}/{self.silence_frames_threshold}")

                # Stop after silence threshold reached
                if silence_frames >= self.silence_frames_threshold:
                    print("Silence threshold reached, stopping recording")
                    self.stop_recording()
                    self.triggered = False

                    # Notify frontend that recording has stopped
                    await self.broadcast_to_clients({
                        "type": "status",
                        "message": "Recording stopped"
                    })

                    return

    async def broadcast_to_clients(self, message):
        """Send a message to all connected WebSocket clients"""
        if active_connections:
            message_json = json.dumps(message)
            await asyncio.gather(
                *[connection.send(message_json) for connection in active_connections]
            )
            print(f"Broadcasted message to {len(active_connections)} clients")
        else:
            print("No clients connected to broadcast message")

    async def handle_websocket_connection(self, websocket, path):
        """Handle a WebSocket connection from a client"""
        print(f"New client connected: {websocket.remote_address}")
        active_connections.add(websocket)

        try:
            await websocket.send(json.dumps({
                "type": "status",
                "message": "Connected to Speech-to-Text Service"
            }))

            # Keep the connection open and handle any messages from the client
            async for message in websocket:
                try:
                    data = json.loads(message)
                    print(f"Received message from client: {data}")

                    # Handle client commands if needed
                    if data.get("command") == "set_city":
                        city = data.get("city", "")
                        print(f"City set to: {city}")

                except json.JSONDecodeError:
                    print(f"Received non-JSON message: {message}")

        except websockets.exceptions.ConnectionClosed:
            print(f"Client disconnected: {websocket.remote_address}")
        finally:
            active_connections.remove(websocket)

    async def start_websocket_server(self):
        """Start the WebSocket server"""
        server = await websockets.serve(
            self.handle_websocket_connection,
            WS_HOST,
            WS_PORT
        )
        print(f"WebSocket server started at ws://{WS_HOST}:{WS_PORT}")
        return server

    async def createText(self):
        """Main function: record speech, convert to text, extract keywords, and send to frontend"""
        global finish_result
        finish_result = True

        try:
            # Initialize microphone
            stream = self.initialize_stream()

            try:
                # Listen for speech
                await self.detect_speech(stream)

                # Save audio file
                audio_file = None
                if self.frames:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    audio_file = os.path.join(RECORDINGS_DIR, f"audio_{timestamp}.wav")
                    self.save_audio(audio_file)
                    print(f"Audio saved to: {audio_file}")

                    # Notify frontend that audio is being processed
                    await self.broadcast_to_clients({
                        "type": "status",
                        "message": "Processing audio"
                    })

                # Transcribe audio
                if audio_file and os.path.exists(audio_file):
                    print(f"Transcribing audio: {audio_file}")
                    result = self.vosk_service.transcribe_audio(audio_file)

                    if result.get("success") and result.get("text"):
                        # Save transcription to file
                        text = result["text"]
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filepath = os.path.join(TRANSCRIBE_TEXT_DIR, f"transcription_{timestamp}.text")

                        with open(filepath, "w", encoding="utf-8") as file:
                            file.write(text)
                        print(f"Transcription: '{text}'")

                        # Send transcription to frontend
                        await self.broadcast_to_clients({
                            "type": "transcription",
                            "text": text
                        })

                        # Extract keywords
                        weather_data = self.weather_extractor.extract(text)
                        weather_data["original_query"] = text
                        print(f"Extracted data: {weather_data}")

                        # If it's a weather query, send the extracted location to frontend
                        if weather_data["is_weather_query"]:
                            location = weather_data.get("location", "")
                            if location:
                                await self.broadcast_to_clients({
                                    "type": "city",
                                    "city": location
                                })

                                # Also send a message to update the UI
                                await self.broadcast_to_clients({
                                    "type": "message",
                                    "text": f"Stadt auf {location} gesetzt. Klicken Sie auf 'Aktualisieren', um die Wetterdaten zu laden."
                                })

                        finish_result = False
                        return filepath
                    else:
                        error_message = "Transcription failed or empty"
                        print(error_message)
                        await self.broadcast_to_clients({
                            "type": "error",
                            "message": error_message
                        })

                return None

            finally:
                # Clean up
                stream.stop_stream()
                stream.close()

        except Exception as e:
            error_message = f"Error: {str(e)}"
            print(error_message)
            await self.broadcast_to_clients({
                "type": "error",
                "message": error_message
            })
            return None


async def main():
    """Main function to start the service"""
    service = SpeechToTextService()

    # Start WebSocket server
    server = await service.start_websocket_server()

    try:
        # Main loop - continuously listen for speech
        while True:
            await service.createText()
            # Small delay before starting the next listening cycle
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Service stopped by user")
    finally:
        server.close()
        await server.wait_closed()
        print("WebSocket server closed")


if __name__ == "__main__":
    asyncio.run(main())
