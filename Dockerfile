FROM python:3.12-slim

WORKDIR /app

# Install all dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model/vosk-model-small-de-zamia-0.3/ ./model/vosk-model-de-tuda-0.6-900k/

COPY *.py .
COPY recordings/ ./recordings/
COPY transcribe_text/ ./transcribe_text/

CMD ["python", "speech_to_text_service.py"]
