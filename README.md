# GmVLM

A [Modal.com](https://modal.com) serverless endpoint for Vision-Language Model inference using [vLLM](https://github.com/vllm-project/vllm).

## Model

Defaults to `GitMylo/nsfwcaption-qwen3-vl-8b-v3-safetensors`. Override with the `MODEL_PATH` env var or by editing the default in `handler.py` and redeploying.

## Infrastructure

- **Runtime**: Modal serverless (GPU: L4)
- **Container image**: `vllm/vllm-openai:v0.10.2`
- **Model cache**: Modal Volume `gmvlm-hf-cache` mounted at `/hf-cache` — model is downloaded once and reused across cold starts
- **Cold start**: GPU memory snapshot enabled (`enable_gpu_snapshot`) — after the first boot creates a snapshot, subsequent cold starts restore much faster than a full model load
- **Inference API**: in-process vLLM `LLM.chat`, using OpenAI-compatible multimodal `messages`

## Deploy

```bash
pip install modal
modal setup        # authenticate once
modal deploy handler.py
```

The endpoint URL is printed after deploy:
```
https://<your-username>--gmvlm-vlmmodel-handler.modal.run
```

## Test

```bash
# Using the test script
python scripts/test_custom_vl_endpoint.py --image-url "https://example.com/image.jpg"

# FL2V (two images)
python scripts/test_custom_vl_endpoint.py --image-url "https://..." --last-image-url "https://..."

# Override endpoint URL
python scripts/test_custom_vl_endpoint.py --image-url "https://..." --endpoint-url "https://..."

# Using modal run (invokes GPU container directly)
modal run handler.py
```

Set `MODAL_VL_ENDPOINT_URL` env var to avoid passing `--endpoint-url` every time.

## Input Format

Accepts OpenAI-compatible chat messages (bare, no `input` wrapper):

```json
{
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    {
      "role": "user",
      "content": [
        { "type": "image_url", "image_url": { "url": "https://..." } },
        { "type": "text", "text": "Describe this image." }
      ]
    }
  ],
  "temperature": 0.1,
  "max_tokens": 800
}
```

Also accepts a simpler format:

```json
{
  "image_url": "https://...",
  "prompt": "Describe this image."
}
```

The RunPod-style `{"input": {...}}` wrapper is also accepted for backwards compatibility.

## Output

Returns an OpenAI-compatible response shape:

```json
{
  "choices": [
    { "message": { "role": "assistant", "content": "..." } }
  ],
  "usage": {
    "prompt_tokens": 120,
    "completion_tokens": 80
  }
}
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `GitMylo/nsfwcaption-qwen3-vl-8b-v3-safetensors` | HuggingFace model ID |
| `MAX_MODEL_LEN` | `4096` | vLLM context length |
| `MAX_NUM_SEQS` | `2` | Concurrent sequences allowed in the engine |
| `GPU_MEMORY_UTILIZATION` | `0.9` | Fraction of GPU memory vLLM may reserve |
| `MAX_IMAGES_PER_PROMPT` | `4` | Max image parts accepted in one request |
| `MAX_IMAGE_PIXELS` | `1003520` | Upper bound passed to the multimodal processor |
| `TRUST_REMOTE_CODE` | `0` | Set to `1` for models that require custom HF code |
| `HF_HOME` | `/hf-cache` | HF cache dir (Modal Volume) |

## Volume Management

```bash
# List cached models
modal volume ls gmvlm-hf-cache /hub

# Delete a specific model
modal volume rm gmvlm-hf-cache /hub/models--org--modelname -r

# Wipe entire cache
modal volume delete gmvlm-hf-cache
```

