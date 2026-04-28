FROM python:3.11-slim

# Install system dependencies: ffmpeg, curl, weasyprint deps, tesseract OCR, poppler
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg curl libmagic1 \
    libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf-2.0-0 libffi-dev shared-mime-info \
    tesseract-ocr tesseract-ocr-eng poppler-utils fonts-liberation \
    libreoffice-writer libreoffice-impress libreoffice-calc && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI (for OAuth token support via Agent SDK)
RUN npm install -g @anthropic-ai/claude-code

WORKDIR /app
ENV PYTHONPATH="/app"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download Piper voice model for streaming TTS
RUN mkdir -p /models && \
    curl -L -o /models/en_US-lessac-medium.onnx \
      "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx" && \
    curl -L -o /models/en_US-lessac-medium.onnx.json \
      "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"

ENV PIPER_MODEL_PATH=/models/en_US-lessac-medium.onnx

# Pre-download fastembed embedding model so it's baked into the image
ENV FASTEMBED_CACHE_PATH=/models/fastembed
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='/models/fastembed')"

COPY models.py db_schema.py llm.py document_processing.py canvas_client.py ./
COPY config.py dependencies.py middleware.py ./
COPY core/ core/
COPY static/ static/
COPY services/ services/
COPY routers/ routers/
COPY main_live.py main_live.py

# Non-root user
RUN groupadd -r atlas && useradd -r -g atlas -d /app -s /sbin/nologin atlas && \
    chown -R atlas:atlas /app /models
USER atlas

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live')"]

CMD ["gunicorn", "main_live:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--timeout", "120", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-"]
