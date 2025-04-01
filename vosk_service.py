import json
import os
import wave
from typing import Dict, Any

try:
    from vosk import Model, KaldiRecognizer

    vosk_available = True
except ImportError:
    vosk_available = False
    print("Warning: Vosk not available")


class VoskService:
    def __init__(self, model_path: str = "model/vosk-model-small-de-0.15"):
        self.model = None
        self.model_path = model_path
        self.initialize_model()

    def initialize_model(self) -> bool:
        if not vosk_available:
            return False
        if os.path.exists(self.model_path):
            self.model = Model(self.model_path)
            return True
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
