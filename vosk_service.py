import json
import os
import wave
from typing import Dict, Any

try:
    from vosk import Model, KaldiRecognizer, SetLogLevel

    vosk_available = True
except ImportError:
    vosk_available = False
    print("Warning: Vosk not available")

_vosk_instance = None


class VoskService:
    def __init__(self, model_path="model/vosk-model-small-de-0.15"):
        global _vosk_instance
        if _vosk_instance is None:
            self.model_path = model_path
            self.model = None
            self.initialize_model()
            _vosk_instance = self
        else:
            self.model_path = _vosk_instance.model_path
            self.model = _vosk_instance.model

    def initialize_model(self) -> bool:
        if not vosk_available:
            return False
        if os.path.exists(self.model_path):
            try:
                # Reduce console output
                SetLogLevel(-1)
                self.model = Model(self.model_path)
                print(f"Vosk model loaded from: {self.model_path}")
                return True
            except Exception as e:
                print(f"Error loading Vosk model: {e}")
                return False
        return False

    def is_available(self) -> bool:
        return vosk_available and self.model is not None

    def transcribe_audio(self, audio_file_path: str) -> Dict[str, Any]:
        if not self.is_available():
            return {"success": False, "error": "Speech recognition not available"}

        try:
            wf = wave.open(audio_file_path, "rb")
            rec = KaldiRecognizer(self.model, wf.getframerate())

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

            return {
                "success": True,
                "text": result.strip()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
