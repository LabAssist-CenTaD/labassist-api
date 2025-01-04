FROM nvidia/cuda:11.8.0-base-ubuntu22.04

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY requirements.txt ./

# Install build dependencies and FFmpeg libraries
RUN apt update -y && \
  apt-get update -y && \
  apt install curl git python3-pip -y && \
  apt-get upgrade -y

RUN apt-get install -y --no-install-recommends gcc python3-dev libssl-dev libgl1-mesa-glx && \
  pip install --no-cache-dir --upgrade pip && \
  apt install -y pkg-config ffmpeg libavformat-dev libavcodec-dev libswscale-dev libopencv-dev && \
  pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
  pip install --no-cache-dir -r requirements.txt && \
  apt-get install -y --reinstall systemd

# Install CUDA container toolkit
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && \
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && \
  apt-get update && \
  apt-get install -y nvidia-container-toolkit && \
  nvidia-ctk runtime configure --runtime=docker

# Clean up
RUN apt-get remove -y gcc python3-dev libssl-dev && \
  apt-get autoremove -y

# Set PKG_CONFIG_PATH
ENV PKG_CONFIG_PATH=/usr/lib/pkgconfig:/usr/local/lib/pkgconfig

# Copy the app folder and run.py to the container
COPY ./app /app/app
COPY ./run.py /app

# Command to run when the container starts
CMD ["python3", "/app/run.py"]