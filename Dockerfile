FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    pciutils \
    ripgrep \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/stable-diffusion-webui-codex

COPY . .

ARG CODEX_TORCH_MODE=cuda
ARG CODEX_CUDA_VARIANT=12.8
ARG CODEX_TORCH_BACKEND=
ARG CODEX_NODE_VERSION=24.15.0
ARG CODEX_FFMPEG_VERSION=7.0.2

ENV CODEX_ROOT=/opt/stable-diffusion-webui-codex \
    CODEX_TORCH_MODE=${CODEX_TORCH_MODE} \
    CODEX_CUDA_VARIANT=${CODEX_CUDA_VARIANT} \
    CODEX_TORCH_BACKEND=${CODEX_TORCH_BACKEND} \
    CODEX_NODE_VERSION=${CODEX_NODE_VERSION} \
    CODEX_FFMPEG_VERSION=${CODEX_FFMPEG_VERSION} \
    CODEX_MAIN_DEVICE=cuda \
    CODEX_MOUNT_DEVICE=cuda \
    CODEX_OFFLOAD_DEVICE=cpu \
    CODEX_CORE_DEVICE=cuda \
    CODEX_TE_DEVICE=cuda \
    CODEX_VAE_DEVICE=cuda \
    CODEX_ATTENTION_BACKEND=pytorch \
    CODEX_ATTENTION_SDPA_POLICY=flash \
    CODEX_LORA_APPLY_MODE=online \
    CODEX_LORA_ONLINE_MATH=weight_merge \
    CODEX_LORA_MERGE_MODE=fast \
    CODEX_LORA_REFRESH_SIGNATURE=content_sha256 \
    CODEX_WAN22_IMG2VID_CHUNK_BUFFER_MODE=ram+hd \
    CODEX_CFG_BATCH_MODE=fused \
    CODEX_ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF=0 \
    CODEX_CUDA_MALLOC=0 \
    PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync \
    CODEX_SINGLE_FLIGHT=1 \
    CODEX_SAFE_WEIGHTS=0 \
    CODEX_TASK_CANCEL_DEFAULT_MODE=immediate \
    CODEX_TASK_EVENT_BUFFER_MAX_EVENTS=5000 \
    CODEX_TASK_EVENT_BUFFER_MAX_MB=64 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/stable-diffusion-webui-codex/.nodeenv/bin:${PATH}

RUN bash install-webui.sh --reinstall-deps

EXPOSE 7850 7860

ENTRYPOINT ["./run-webui-docker.sh"]
