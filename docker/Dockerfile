FROM python:3.10-slim

# ffmpeg for encoding videos, git for wandb run metadata,
# build-essential + swig to build box2d (LunarLander) when no prebuilt wheel exists for the platform
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg build-essential swig \
    && rm -rf /var/lib/apt/lists/*

# SDL_VIDEODRIVER=dummy lets pygame render rgb_array frames headless (video capture)
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SDL_VIDEODRIVER=dummy

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

# Non-root user matching the host UID/GID so files written to the mounted repo (runs/, videos/, wandb/) stay yours
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} rl && useradd -m -u ${UID} -g ${GID} rl
USER rl

WORKDIR /workspace
ENV PYTHONPATH=/workspace
CMD ["bash"]
