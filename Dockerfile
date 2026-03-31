# LMDeploy + CUDA 12.8 base image
FROM openmmlab/lmdeploy:v0.12.2-cu12.8

WORKDIR /app

# Install RunPod and HF utilities
RUN pip install --no-cache-dir \
    runpod \
    requests \
    pillow \
    huggingface_hub \
    hf_transfer

# Copy source files
COPY handler.py download_model.py test_input.json ./

# Model download build args (optional — bake model into image)
ARG MODEL_PATH=""
ARG MODEL_REVISION=""
ARG HF_TOKEN=""

ENV MODEL_PATH=${MODEL_PATH} \
    MODEL_REVISION=${MODEL_REVISION} \
    # Point HF cache to a volume-friendly path
    BASE_PATH="/runpod-volume" \
    HF_HOME="/runpod-volume/huggingface-cache/hub" \
    HUGGINGFACE_HUB_CACHE="/runpod-volume/huggingface-cache/hub" \
    HF_DATASETS_CACHE="/runpod-volume/huggingface-cache/datasets" \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1

# Optionally bake model into the image at build time
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
        export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_PATH" ]; then \
        MODEL_NAME=$MODEL_PATH python3 download_model.py; \
    fi

CMD ["python3", "-u", "handler.py"]
