import importlib
import os
import modal

# ---------------------------------------------------------------------------
# Modal app & container image
# ---------------------------------------------------------------------------

app = modal.App("gmvlm")

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "curl")
    .uv_pip_install(
        "vllm",
        "transformers>=4.57.0,<4.57.7",
        "fastapi[standard]",
        "huggingface_hub",
        "hf_transfer",
    ).env({
        "HF_HOME": "/hf-cache",
        "HUGGINGFACE_HUB_CACHE": "/hf-cache",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_XET_HIGH_PERFORMANCE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
)

# Persistent volume for HuggingFace model cache
hf_cache_vol = modal.Volume.from_name("gmvlm-hf-cache", create_if_missing=True)
HF_CACHE_PATH = "/hf-cache"


# ---------------------------------------------------------------------------
# Message parsing helpers
# ---------------------------------------------------------------------------

def _normalize_messages(job_input: dict):
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

    Returns OpenAI-compatible chat messages for vLLM.
    """
    def normalize_content(content):
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""

        normalized_parts = []
        for part in content:
            if not isinstance(part, dict):
                continue

            part_type = part.get("type")
            if part_type == "text":
                normalized_parts.append({"type": "text", "text": str(part.get("text", ""))})
                continue

            if part_type == "image_url":
                image_url = part.get("image_url")
                if isinstance(image_url, dict):
                    image_url = image_url.get("url")
                if image_url:
                    normalized_parts.append({
                        "type": "image_url",
                        "image_url": {"url": str(image_url)},
                    })
                continue

            normalized_parts.append(part)

        return normalized_parts

    # --- Format A ---
    if "messages" in job_input:
        messages = job_input["messages"]
        if not isinstance(messages, list):
            raise TypeError("messages must be a list")

        normalized_messages = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            normalized_messages.append({
                "role": str(message.get("role", "user")),
                "content": normalize_content(message.get("content", "")),
            })

        return normalized_messages

    # --- Format B ---
    image_url = job_input.get("image_url")
    prompt = str(job_input.get("prompt", "Analyze this image."))
    content = []

    if image_url:
        if isinstance(image_url, dict):
            image_url = image_url.get("url")
        content.append({"type": "image_url", "image_url": {"url": str(image_url)}})

    if prompt:
        content.append({"type": "text", "text": prompt})

    return [{"role": "user", "content": content if content else prompt}]


def _build_sampling_params(job_input: dict):
    SamplingParams = importlib.import_module("vllm").SamplingParams

    params = {
        "temperature": float(job_input.get("temperature", 0.1)),
        "max_tokens": int(job_input.get("max_tokens", 800)),
        "repetition_penalty": float(job_input.get("repetition_penalty", 1.05)),
    }

    if "top_p" in job_input:
        params["top_p"] = float(job_input["top_p"])
    if "top_k" in job_input:
        params["top_k"] = int(job_input["top_k"])
    if "presence_penalty" in job_input:
        params["presence_penalty"] = float(job_input["presence_penalty"])
    if "frequency_penalty" in job_input:
        params["frequency_penalty"] = float(job_input["frequency_penalty"])
    if "stop" in job_input:
        params["stop"] = job_input["stop"]
    if "seed" in job_input:
        params["seed"] = int(job_input["seed"])

    return SamplingParams(**params)


# ---------------------------------------------------------------------------
# Model class — one container, model loaded once, endpoint reused per request
# ---------------------------------------------------------------------------

@app.cls(
    gpu="L4",
    image=image,
    volumes={HF_CACHE_PATH: hf_cache_vol},
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    scaledown_window=300,
    timeout=600,
)
class VLMModel:
    @modal.enter(snap=True)
    def load_model(self):
        vllm = importlib.import_module("vllm")
        LLM = vllm.LLM
        SamplingParams = vllm.SamplingParams

        model_path = os.getenv("MODEL_PATH", "GitMylo/nsfwcaption-qwen3-vl-8b-v3-safetensors")
        max_images_per_prompt = int(os.getenv("MAX_IMAGES_PER_PROMPT", "4"))
        self.mm_processor_kwargs = {
            "min_pixels": 28 * 28,
            "max_pixels": int(os.getenv("MAX_IMAGE_PIXELS", str(1280 * 28 * 28))),
            "fps": 1,
        }

        print(f"Loading model from {model_path}...")
        self.llm = LLM(
            model=model_path,
            trust_remote_code=os.getenv("TRUST_REMOTE_CODE", "0") == "1",
            max_model_len=int(os.getenv("MAX_MODEL_LEN", "4096")),
            max_num_seqs=int(os.getenv("MAX_NUM_SEQS", "2")),
            gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9")),
            limit_mm_per_prompt={"image": max_images_per_prompt},
            mm_processor_kwargs=self.mm_processor_kwargs,
        )
        print("Model loaded. Running warmup pass before snapshot...")
        # Warmup: ensures CUDA kernels are JIT-compiled before the snapshot is taken,
        # so cold-start restores don't pay that cost on the first real request.
        self.llm.chat(
            [{"role": "user", "content": "Say ready."}],
            sampling_params=SamplingParams(temperature=0.0, max_tokens=8),
            use_tqdm=False,
        )
        print("Warmup done. Ready to snapshot.")

    @modal.method()
    def run_inference(self, item: dict) -> dict:
        # Accept both RunPod-style {"input": {...}} and bare {...}
        job_input = item.get("input", item)

        try:
            messages = _normalize_messages(job_input)
            sampling_params = _build_sampling_params(job_input)
        except Exception as e:
            return {"error": f"Failed to parse input: {str(e)}", "type": type(e).__name__}

        if not messages:
            return {"error": "No valid messages found in input."}

        try:
            outputs = self.llm.chat(
                messages=messages,
                sampling_params=sampling_params,
                use_tqdm=False,
                mm_processor_kwargs=self.mm_processor_kwargs,
            )
        except Exception as e:
            return {"error": f"Inference failed: {str(e)}", "type": type(e).__name__}

        response = outputs[0]
        first_output = response.outputs[0] if response.outputs else None
        prompt_token_ids = getattr(response, "prompt_token_ids", None)
        completion_token_ids = getattr(first_output, "token_ids", None) if first_output else None

        # OpenAI-compatible response shape
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": first_output.text if first_output else "",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_token_ids) if prompt_token_ids is not None else None,
                "completion_tokens": len(completion_token_ids) if completion_token_ids is not None else None,
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
                        "image_url": {"url": "https://www.google.com/logos/doodles/2026/artemis-ii-spaceflight-around-the-moon-6753651837111248.3-law.gif"},
                    },
                    {"type": "text", "text": "Analyze this image for image-to-video prompt generation."},
                ],
            },
        ],
        "temperature": 0.5,
        "max_tokens": 800,
    }

    model = VLMModel()
    result = model.run_inference.remote(test_input)
    print(json.dumps(result, ensure_ascii=False, indent=2))
