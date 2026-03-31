# GmVLM

A RunPod serverless endpoint for Vision-Language Model inference using [LMDeploy](https://github.com/InternLM/lmdeploy).

## Model

Defaults to `Qwen/Qwen2.5-VL-7B-Instruct`. Override with the `MODEL_PATH` environment variable.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `Qwen/Qwen2.5-VL-7B-Instruct` | HuggingFace model ID or local path |
| `MODEL_REVISION` | _(latest)_ | HuggingFace revision/commit |
| `CACHE_MAX_ENTRY_COUNT` | `0.5` | KV cache fraction (lower on 16GB GPUs) |
| `HF_TOKEN` | _(none)_ | HuggingFace token for gated models |
| `HF_HOME` | `/runpod-volume/huggingface-cache/hub` | HF cache dir — mount a Network Volume here |

## Input Format

Accepts OpenAI-compatible chat messages:

```json
{
  "input": {
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
}
```

Also accepts a simpler format:

```json
{
  "input": {
    "image_url": "https://...",
    "prompt": "Describe this image."
  }
}
```

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

## Docker Build

```bash
# Runtime model loading via Network Volume (recommended)
docker build -t gmvlm .

# Bake model into image at build time
docker build \
  --build-arg MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct \
  --secret id=HF_TOKEN \
  -t gmvlm .
```
