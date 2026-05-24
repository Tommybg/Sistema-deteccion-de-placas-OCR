FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System packages for OpenCV (matches the existing root Dockerfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0t64 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first to keep this layer cached
COPY apps/api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Application code
COPY apps/api/app /app/app
COPY apps/api/tests /app/tests

# Inference modules reused as-is
COPY scripts/vehicle_detector.py /app/scripts/vehicle_detector.py
COPY scripts/color_classifier.py /app/scripts/color_classifier.py
COPY scripts/brand_detector.py /app/scripts/brand_detector.py
COPY scripts/yolo11n.pt /app/scripts/yolo11n.pt

# Model artifacts
COPY models/ /app/models/
COPY samples/ /app/samples/

# Allow `from scripts...` and `from app...` to resolve
ENV PYTHONPATH=/app
# config.py walks parents[3] from app/config.py — point REPO_ROOT explicitly
ENV REPO_ROOT=/app
ENV MODEL_DIR=/app/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request, os, sys; r=urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/api/health',timeout=5); sys.exit(0 if r.status==200 else 1)"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
