FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1 \
	DEVICE=cpu

WORKDIR /app

# Install system dependencies for OpenCV (headless) and MediaPipe
# opencv-python-headless is pre-built but needs runtime libraries
# ffmpeg will pull in most video codec libraries needed
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1-mesa-glx \
        ffmpeg \
        libv4l-0 \
        libv4lconvert0 \
        libjpeg62-turbo \
        libpng16-16 \
        libtiff5 \
        libwebp6 \
        python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Install pip and upgrade first
RUN pip install --upgrade pip setuptools wheel

# Install opencv-python-headless BEFORE torch/torchvision to avoid conflicts
COPY requirements.txt ./
RUN pip install --no-cache-dir opencv-python-headless==4.10.0.84 && \
    python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"

# Install CPU-only torch/torchvision (may pull in opencv, but we've already installed headless)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision

# Install remaining requirements (excluding opencv-python-headless as it's already installed)
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "import cv2; print(f'OpenCV version verified: {cv2.__version__}')" && \
    python -c "import mediapipe; print('MediaPipe imported successfully')"

# Copy application code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "App.main:app", "--host", "0.0.0.0", "--port", "8000"]


