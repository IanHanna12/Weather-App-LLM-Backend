import logging
import os
import queue
import threading
import warnings
import wave
import json

import pyaudio
import numpy as np
import webrtcvad
from vosk import Model, KaldiRecognizer

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")
warnings.filterwarnings("ignore", category=UserWarning)

logging.getLogger("torch").setLevel(logging.ERROR)

SAMPLE_RATE = 16000
FRAME_DURATION = 30  # in Milliseconds
CHUNK = int(SAMPLE_RATE * FRAME_DURATION / 1000)


RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
#RECORDINGS_DIR = r'C:\Users\FurkanAydin\Documents\lws_pbv\pbv\recordings'

if not os.path.exists(RECORDINGS_DIR):
    os.makedirs(RECORDINGS_DIR)

TRANSCRIBE_TEXT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcribe_text")
if not os.path.exists(TRANSCRIBE_TEXT_DIR):
    os.makedirs(TRANSCRIBE_TEXT_DIR)

# Lade das Modell
model_path = "model/vosk-model-de-tuda-0.6-900k"  # Passe diesen Pfad an, falls nötig
model = Model(model_path)
recognizer = KaldiRecognizer(model, 16000)

class AudioRecorder:
    def __init__(self):
        self.CHUNK = CHUNK
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = SAMPLE_RATE
        self.INPUT_DEVICE_INDEX = 3
        self.p = pyaudio.PyAudio()
        self.frames = []
        self.is_recording = False
        self.recording_thread = None
        self.vad = webrtcvad.Vad(3)
        self.audio_queue = queue.Queue()
        self.model = model
        self.recognizer = recognizer

    def audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def record(self):
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            input_device_index=self.INPUT_DEVICE_INDEX,
            stream_callback=self.audio_callback
        )

        self.stream.start_stream()

        while self.is_recording:
            try:
                data = self.audio_queue.get(timeout=1.0)
                self.frames.append(data)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Fehler bei der Aufnahme: {e}")
                break

        self.stream.stop_stream()
        self.stream.close()

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.frames = []
            self.audio_queue = queue.Queue()
            self.recording_thread = threading.Thread(target=self.record)
            self.recording_thread.start()
            print("Aufnahme gestartet...")

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join()
            print("Aufnahme beendet.")

    def save_audio(self, filename):
        if not self.frames:
            print("Keine Audiodaten zum Speichern vorhanden.")
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
            print(f"Fehler beim Speichern der Audiodatei: {e}")
            return False

    def close(self):
        self.p.terminate()

    def transcribe(self, audio_path):
        full_text = ""  # Sammle den gesamten Text
        with wave.open(audio_path, "rb") as wf:
            while True:
                data = wf.readframes(64000)
                if len(data) == 0:
                    break
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    full_text += result["text"] + " "  # Sammeln statt sofort return

        final_result = json.loads(self.recognizer.FinalResult())
        full_text += final_result["text"]
        result_text = full_text.strip()
        return {"text": result_text}

    def increase_volume(self, input_file, output_file, factor):
        # Öffne die WAV-Datei zum Lesen
        with wave.open(input_file, "rb") as wf:
            params = wf.getparams()  # Speichert die Audio-Parameter
            frames = wf.readframes(wf.getnframes())  # Lies alle Frames als Byte-String

        # Umwandlung in ein NumPy-Array für die Bearbeitung
        audio_data = np.frombuffer(frames, dtype=np.int16)  # 16-bit PCM-Daten

        # Skalierung der Amplitude (Lautstärke erhöhen)
        audio_data = np.clip(audio_data * factor, -32768, 32767).astype(np.int16)

        # Speichere das veränderte Audio in einer neuen Datei
        with wave.open(output_file, "wb") as wf:
            wf.setparams(params)  # Setzt dieselben Parameter wie das Original
            wf.writeframes(audio_data.tobytes())  # Konvertiert das NumPy-Array zurück in Bytes

        print(f"Lautstärke erhöht und gespeichert als {output_file}")