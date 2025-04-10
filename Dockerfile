FROM python:3.9-slim

WORKDIR /app

# Install all dependencies including build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from parent directory
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code from parent directory
COPY .. .

# Create directories
RUN mkdir -p recordings transcribe_text

CMD ["python", "speech_to_text_service.py"]
