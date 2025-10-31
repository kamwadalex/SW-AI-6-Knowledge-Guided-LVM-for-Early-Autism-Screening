FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1 \
	DEVICE=cpu

WORKDIR /app

# Install system dependencies for OpenCV (headless) and MediaPipe
# opencv-python-headless needs minimal dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgthread-2.0-0 \
        libavcodec58 \
        libavformat58 \
        libswscale5 \
        libavutil56 \
        libv4l-0 \
        libjpeg62-turbo \
        libpng16-16 \
        libtiff5 \
        python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Install pip and CPU-only torch/torchvision first
RUN pip install --upgrade pip setuptools wheel && \
	pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision

# Install remaining requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')" && \
    python -c "import mediapipe; print('MediaPipe imported successfully')"

# Copy application code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "App.main:app", "--host", "0.0.0.0", "--port", "8000"]


