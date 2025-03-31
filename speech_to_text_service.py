import asyncio
import os
import time
from datetime import datetime

from starlette.responses import JSONResponse

from speech_to_text_vosk import AudioRecorder, RECORDINGS_DIR, TRANSCRIBE_TEXT_DIR

#Globale Variable
global finish_result

class SpeechToTextService:
    def __init__(self):
        self.recorder = AudioRecorder()
        self.silence_threshold_seconds = 2.0
        self.frames_per_second = int(self.recorder.RATE / self.recorder.CHUNK)
        self.silence_frames_threshold = int(self.silence_threshold_seconds * self.frames_per_second)

    def initialize_stream(self):
        # Initialisiert den Audio-Stream.

        return self.recorder.p.open(
            format=self.recorder.FORMAT,
            channels=self.recorder.CHANNELS,
            rate=self.recorder.RATE,
            input=True,
            frames_per_buffer=self.recorder.CHUNK,
            input_device_index=self.recorder.INPUT_DEVICE_INDEX,
        )

    async def detect_speech(self, stream):
        global is_speech
        silence_frames = 0
        is_speaking = False

        while True:
            is_speech = True
            data = await asyncio.to_thread(stream.read, self.recorder.CHUNK, exception_on_overflow=False)
            is_speech = self.recorder.vad.is_speech(data, self.recorder.RATE)

            print("Nach der is_spech-Variable: ", is_speech, is_speaking)

            # is_speech = true == Sprache wird erkannt
            # is_speaking = false  == es wird gerade nicht gesprochen
            if is_speech and not is_speaking:
                # es wird aktuell geredet
                print("Sprache erkannt aber wird nicht geredet: ", is_speech, is_speaking)
                is_speaking = True
                self.recorder.start_recording()

            # Wenn Sprache erkannt und gesprochen wird
            # reseten silence_frames = 0
            if is_speaking and is_speech:
                print("Es wird gerade geredet: ", is_speaking, is_speech)
                silence_frames = 0

            # es wird aktuell keine Sprache erkannt
            # es wurde gesprochen
            # wenn in den nächsten 66 Frames keine Sprache erkannt wird, wird die Aufnahme gestoppt
            elif is_speaking and not is_speech:
                silence_frames += 1
                print("Stille erkannt: ", is_speaking, is_speech)
                if silence_frames >= self.silence_frames_threshold:
                    self.recorder.stop_recording()
                    return


    def save_audio_file(self):
        if self.recorder.frames:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_file = os.path.join(RECORDINGS_DIR, f"audio_{timestamp}.wav")
            if self.recorder.save_audio(audio_file):
                print(f"Audio gespeichert in: {audio_file}")
                louder_audio_file = os.path.join(RECORDINGS_DIR, f"audio_{timestamp}_louder.wav")
                self.recorder.increase_volume(audio_file, louder_audio_file, factor=2.0)
                if os.path.exists(louder_audio_file):
                    os.remove(audio_file)
                return louder_audio_file
        return None

    def transcribe_audio_file(self, audio_file):
        global finish_result
        #Transkribiert eine gespeicherte Audiodatei.

        if not audio_file:
            return None

        transcription = self.recorder.transcribe(audio_file)

        if transcription and transcription.get("text").strip():
            print(f"Transkription: {transcription}")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transcription_{timestamp}.text"
            filepath = os.path.join(TRANSCRIBE_TEXT_DIR, filename)

            with open(filepath, "w", encoding="utf-8") as file:
                file.write(transcription["text"])

            finish_result = False
            return filepath
        else:
            print("Leerer Audio-File und Text-File")
            os.remove(audio_file)
            finish_result = True
            return



    async def createText(self):
        global finish_result
        start_time = time.time()  # Startzeit erfassen
        timeout = 30  # 30 Sekunden

        finish_result = True
        try:
            stream = self.initialize_stream()
            while finish_result and (time.time() - start_time < timeout):
                print("Warte auf Sprache...")
                await self.detect_speech(stream)
                audio_file = self.save_audio_file()
                transcription = self.transcribe_audio_file(audio_file)
                print("Finish Result: ", finish_result)
            return transcription
        except Exception as e:
            error_message = {
                "error": str(e),
                "message": "Ein Fehler ist aufgetreten, entweder in der Initialisierung, Spracherkennung, speichern von der Audio-Datei oder in der Transcribierung."
            }

            return JSONResponse(
                status_code=500,
                content=error_message
            )
        #finally:
            #self.recorder.close()
         #   if 'stream' in locals():
          #      stream.stop_stream()
           #     stream.close()
            #print("Aufnahme und Transkription abgeschlossen.")
