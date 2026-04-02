import os
import modal

# ---------------------------------------------------------------------------
# Modal app & container image
# ---------------------------------------------------------------------------

app = modal.App("gmvlm")

image = (
    modal.Image.from_registry(
        "openmmlab/lmdeploy:v0.12.2-cu12.8",
        add_python="3.11",
    ).pip_install(
        "pillow",
        "huggingface_hub",
        "hf_transfer",
    ).env({
        "HF_HOME": "/hf-cache",
        "HUGGINGFACE_HUB_CACHE": "/hf-cache",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_XET_HIGH_PERFORMANCE": "1",
    })
)

# Persistent volume for HuggingFace model cache
hf_cache_vol = modal.Volume.from_name("gmvlm-hf-cache", create_if_missing=True)
HF_CACHE_PATH = "/hf-cache"


# ---------------------------------------------------------------------------
# Message parsing helpers
# ---------------------------------------------------------------------------

def _parse_messages(job_input: dict):
    """
    Accept two input formats:

    Format A – OpenAI chat messages (what the test client / GmAnimato sends):
      {
        "messages": [
          {"role": "system", "content": "..."},
          {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "..."}},
            {"type": "text",      "text": "..."}
          ]}
        ]
      }

    Format B – Simple:
      {"image_url": "...", "prompt": "..."}

    Returns (prompt_text, images, gen_config).
    """
    # Lazy imports so this function runs correctly inside the Modal container
    from lmdeploy import GenerationConfig
    from lmdeploy.vl import load_image

    temperature = float(job_input.get("temperature", 0.1))
    max_tokens = int(job_input.get("max_tokens", 800))
    repetition_penalty = float(job_input.get("repetition_penalty", 1.05))
    gen_config = GenerationConfig(temperature=temperature, max_new_tokens=max_tokens, repetition_penalty=repetition_penalty)

    # --- Format A ---
    if "messages" in job_input:
        system_text = ""
        user_text = ""
        image_urls = []

        for msg in job_input["messages"]:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                system_text = content if isinstance(content, str) else ""

            elif role == "user":
                if isinstance(content, str):
                    user_text = content
                elif isinstance(content, list):
                    for part in content:
                        if part.get("type") == "text":
                            user_text = part.get("text", "")
                        elif part.get("type") == "image_url":
                            url = part.get("image_url", {}).get("url", "")
                            if url:
                                image_urls.append(url)

        full_prompt = f"{system_text}\n\n{user_text}".strip() if system_text else user_text
        images = [load_image(u) for u in image_urls]
        return full_prompt, images, gen_config

    # --- Format B ---
    image_url = job_input.get("image_url")
    prompt = job_input.get("prompt", "Analyze this image.")
    images = [load_image(image_url)] if image_url else []
    return prompt, images, gen_config


# ---------------------------------------------------------------------------
# Model class — one container, model loaded once, endpoint reused per request
# ---------------------------------------------------------------------------

@app.cls(
    gpu="L4",
    image=image,
    volumes={HF_CACHE_PATH: hf_cache_vol},
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    scaledown_window=30,
    timeout=300,
)
@modal.concurrent(max_inputs=3, target_inputs=2)
class VLMModel:
    @modal.enter(snap=True)
    def load_model(self):
        from lmdeploy import pipeline, TurbomindEngineConfig

        model_path = os.getenv("MODEL_PATH", "GitMylo/nsfwcaption-qwen3-vl-8b-v3-safetensors")
        backend_config = TurbomindEngineConfig(
            session_len=8192,
            cache_max_entry_count=float(os.getenv("CACHE_MAX_ENTRY_COUNT", "0.5")),
        )
        print(f"Loading model from {model_path}...")
        self.pipe = pipeline(model_path, backend_config=backend_config)
        print("Model loaded. Running warmup pass before snapshot...")
        # Warmup: ensures CUDA kernels are JIT-compiled before the snapshot is taken,
        # so cold-start restores don't pay that cost on the first real request.
        self.pipe("Describe this image briefly.", gen_config=None)
        print("Warmup done. Ready to snapshot.")

    @modal.method()
    def run_inference(self, item: dict) -> dict:
        # Accept both RunPod-style {"input": {...}} and bare {...}
        job_input = item.get("input", item)

        try:
            prompt, images, gen_config = _parse_messages(job_input)
        except Exception as e:
            return {"error": f"Failed to parse input: {str(e)}", "type": type(e).__name__}

        if not prompt:
            return {"error": "No prompt text found in input."}

        try:
            if images:
                # Single image: (prompt, image) — multiple: (prompt, [img1, img2, ...])
                model_input = (prompt, images[0]) if len(images) == 1 else (prompt, images)
            else:
                model_input = prompt

            response = self.pipe(model_input, gen_config=gen_config)
        except Exception as e:
            return {"error": f"Inference failed: {str(e)}", "type": type(e).__name__}

        # OpenAI-compatible response shape
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response.text,
                    }
                }
            ],
            "usage": {
                "prompt_tokens": getattr(response, "input_token_len", None),
                "completion_tokens": getattr(response, "generate_token_len", None),
            },
        }

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def handler(self, item: dict) -> dict:
        return self.run_inference.local(item)


# ---------------------------------------------------------------------------
# Local entrypoint — modal run handler.py
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    import json

    test_input = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert image analyzer. Return ONLY valid JSON with fields: "
                    "suggested_prompts (array of 2 short prompts, first English and second Chinese), "
                    "tags (array of booru-style tags), is_photo_realistic (boolean), is_nsfw (boolean)."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://image.acg.lol/file/2026/03/28/yejiang260328.1.gif"},
                    },
                    {"type": "text", "text": "Analyze this image for image-to-video prompt generation."},
                ],
            },
        ],
        "temperature": 0.1,
        "max_tokens": 800,
    }

    model = VLMModel()
    result = model.run_inference.remote(test_input)
    print(json.dumps(result, ensure_ascii=False, indent=2))
