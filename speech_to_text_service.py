import asyncio
import json
import os
import wave
import numpy as np
import pyaudio
import websockets
from datetime import datetime

RECORDINGS_DIR = "recordings"
TRANSCRIBE_TEXT_DIR = "transcribe_text"
os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(TRANSCRIBE_TEXT_DIR, exist_ok=True)

# Server settings
WS_HOST = "localhost"
WS_PORT = 8765

# List of connected clients
active_connections = set()


class SpeechToTextService:
    def __init__(self, trigger_word="wetter", model_path="model/vosk-model-de-0.6-900K"):
        print("Starting Speech-to-Text Service")

        # Import services here to avoid circular imports
        from vosk_service import VoskService
        from extractorService import WeatherExtractor
        from weather_service import WeatherService

        # Initialize services
        self.vosk_service = VoskService(model_path)
        self.weather_extractor = WeatherExtractor()
        self.weather_service = WeatherService(
            api_url=os.environ.get("BACKEND_API_URL", "http://localhost:8080/api/weather/"))

        # Audio settings
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1024
        self.THRESHOLD = 500

        # Initialize audio system
        self.p = pyaudio.PyAudio()


        self.frames = []
        self.is_recording = False
        self.triggered = False
        self.trigger_word = trigger_word.lower()

        self.silence_seconds = 1.5
        self.frames_per_second = int(self.RATE / self.CHUNK)
        self.silence_frames = int(self.silence_seconds * self.frames_per_second)

        print(f"Ready Listening for trigger word: '{self.trigger_word}'")

    def setup_microphone(self):
        """Set up the microphone for recording"""
        return self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )

    def is_voice(self, data):
        """Check if audio contains voice"""
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
        """Save audio frames to a file"""
        try:
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            return True
        except Exception as e:
            print(f"Error saving frames: {e}")
            return False

    async def detect_speech(self, stream):
        """Listen for speech and record when detected"""
        silence_count = 0
        buffer = []
        buffer_size = 30  # About 1 second of audio

        print("Listening for trigger word...")

        while True:
            # Read audio from microphone
            data = await asyncio.to_thread(stream.read, self.CHUNK)

            # Add to buffer
            buffer.append(data)
            if len(buffer) > buffer_size:
                buffer.pop(0)

            # Check if this is speech
            is_speech = self.is_voice(data)

            # If not triggered yet, check for trigger word
            if not self.triggered:
                if is_speech and len(buffer) >= 15:
                    # Save buffer to temporary file
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
                            self.start_recording()

                            # Add buffer to recording
                            for frame in buffer:
                                self.frames.append(frame)

                            # Tell clients recording started
                            await self.send_to_all_clients({
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
                    silence_count = 0
                else:
                    # Count silence frames
                    silence_count += 1
                    if silence_count % 10 == 0:
                        print(f"Silence: {silence_count}/{self.silence_frames}")

                # Stop after silence threshold reached
                if silence_count >= self.silence_frames:
                    print("Silence detected, stopping recording")
                    self.stop_recording()
                    self.triggered = False

                    # Tell clients recording stopped
                    await self.send_to_all_clients({
                        "type": "status",
                        "message": "Recording stopped"
                    })

                    return

    async def send_to_all_clients(self, message):
        """Send a message to all connected clients"""
        if active_connections:
            message_json = json.dumps(message)
            await asyncio.gather(
                *[connection.send(message_json) for connection in active_connections]
            )
            print(f"Sent message to {len(active_connections)} clients")
        else:
            print("No clients connected")

    async def handle_client_connection(self, websocket, path):
        """Handle a client connection"""
        print(f"New client connected: {websocket.remote_address}")
        active_connections.add(websocket)

        try:
            # Send welcome message
            await websocket.send(json.dumps({
                "type": "status",
                "message": "Connected to Speech-to-Text Service"
            }))

            # Handle messages from client
            async for message in websocket:
                try:
                    data = json.loads(message)
                    print(f"Received from client: {data}")

                    # Handle client commands if needed
                    if data.get("command") == "set_city":
                        city = data.get("city", "")
                        print(f"City set to: {city}")

                except json.JSONDecodeError:
                    print(f"Received invalid message: {message}")

        except websockets.exceptions.ConnectionClosed:
            print(f"Client disconnected: {websocket.remote_address}")
        finally:
            active_connections.remove(websocket)

    async def start_websocket_server(self):
        """Start the WebSocket server"""
        server = await websockets.serve(
            self.handle_client_connection,
            WS_HOST,
            WS_PORT
        )
        print(f"WebSocket server started at ws://{WS_HOST}:{WS_PORT}")
        return server

    async def process_speech(self):
        """Main function: record speech, convert to text, extract info"""
        try:
            # Initialize microphone
            stream = self.setup_microphone()

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

                    # Tell clients we're processing
                    await self.send_to_all_clients({
                        "type": "status",
                        "message": "Processing audio"
                    })

                # Transcribe audio
                if audio_file and os.path.exists(audio_file):
                    print(f"Transcribing audio: {audio_file}")
                    result = self.vosk_service.transcribe_audio(audio_file)

                    if result.get("success") and result.get("text"):
                        # Save transcription
                        text = result["text"]
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filepath = os.path.join(TRANSCRIBE_TEXT_DIR, f"transcription_{timestamp}.text")

                        with open(filepath, "w", encoding="utf-8") as file:
                            file.write(text)
                        print(f"Transcription: '{text}'")

                        # Send transcription to clients
                        await self.send_to_all_clients({
                            "type": "transcription",
                            "text": text
                        })

                        # Extract weather info
                        weather_data = self.weather_extractor.extract(text)
                        weather_data["original_query"] = text
                        print(f"Extracted data: {weather_data}")

                        # If it's a weather query, send the location
                        if weather_data["is_weather_query"]:
                            location = weather_data.get("location", "")
                            if location:
                                await self.send_to_all_clients({
                                    "type": "city",
                                    "city": location
                                })

                                # Update the UI
                                await self.send_to_all_clients({
                                    "type": "message",
                                    "text": f"Stadt auf {location} gesetzt. Klicken Sie auf 'Aktualisieren', um die Wetterdaten zu laden."
                                })

                        return filepath
                    else:
                        error_message = "Transcription failed or empty"
                        print(error_message)
                        await self.send_to_all_clients({
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
            await self.send_to_all_clients({
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
            await service.process_speech()
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
