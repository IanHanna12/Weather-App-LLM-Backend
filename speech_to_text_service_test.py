import numpy as np

from speech_to_text_service import SpeechToTextService


class MockVosk:
    def transcribe_audio(self, file_path):
        return {"success": True, "text": "wetter in berlin"}


class MockExtractor:
    def extract(self, text):
        return {
            "is_weather_query": "wetter" in text.lower(),
            "location": "berlin",
            "time_period": "today"
        }


class MockWeather:
    def get_weather(self, location, period):
        return {"temperature": 22, "condition": "sunny"}

class TestSpeechToText:

    def setup_method(self):
        self.service = SpeechToTextService()
        self.service.vosk_service = MockVosk()
        self.service.weather_extractor = MockExtractor()
        self.service.weather_service = MockWeather()

    def test_init(self):
        assert self.service.trigger_word == "wetter"
        assert self.service.is_recording == False
        assert self.service.triggered == False
        assert self.service.frames == []

    def test_recording(self):
        self.service.start_recording()
        assert self.service.is_recording == True
        assert self.service.frames == []

        self.service.frames = [b'test1', b'test2']

        self.service.stop_recording()
        assert self.service.is_recording == False
        assert len(self.service.frames) == 2

    def test_save_audio(self, monkeypatch):
        saved_data = {}

        def mock_wave_open(filename, mode):
            class MockWaveFile:
                def setnchannels(self, channels):
                    saved_data['channels'] = channels

                def setsampwidth(self, width):
                    saved_data['width'] = width

                def setframerate(self, rate):
                    saved_data['rate'] = rate

                def writeframes(self, frames):
                    saved_data['frames'] = frames

                def close(self):
                    pass

            saved_data['filename'] = filename
            return MockWaveFile()

        monkeypatch.setattr('wave.open', mock_wave_open)

        # Test saving with frames
        self.service.frames = [b'data1', b'data2']
        result = self.service.save_audio("test.wav")

        # Check result
        assert result == True
        assert saved_data['filename'] == "test.wav"
        assert saved_data['frames'] == b'data1data2'

        # Test saving with no frames
        self.service.frames = []
        result = self.service.save_audio("empty.wav")
        assert result == False

    def test_transcribe(self, monkeypatch):
        self.service.save_audio = lambda filename: True

        self.service.frames = [b'test_audio']

        text = self.service.vosk_service.transcribe_audio("fake.wav")

        # Check results
        assert text["success"] == True
        assert "wetter" in text["text"]

        # Test extraction
        result = self.service.weather_extractor.extract(text["text"])
        assert result["is_weather_query"] == True
        assert result["location"] == "berlin"
