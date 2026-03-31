import os
import runpod
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image

# ---------------------------------------------------------------------------
# Startup: load model once, reuse across jobs
# ---------------------------------------------------------------------------
model_path = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct")

backend_config = TurbomindEngineConfig(
    session_len=8192,
    cache_max_entry_count=float(os.getenv("CACHE_MAX_ENTRY_COUNT", "0.5")),
)

print(f"Loading model from {model_path}...")
pipe = pipeline(model_path, backend_config=backend_config)
print("Model loaded and ready.")


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

    Format B – Simple (from the TODO.md draft):
      {"image_url": "...", "prompt": "..."}

    Returns (prompt_text, images, gen_config).
    """
    temperature = float(job_input.get("temperature", 0.1))
    max_tokens = int(job_input.get("max_tokens", 800))
    gen_config = GenerationConfig(temperature=temperature, max_new_tokens=max_tokens)

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

        # Prepend system text if present
        full_prompt = f"{system_text}\n\n{user_text}".strip() if system_text else user_text
        images = [load_image(u) for u in image_urls]
        return full_prompt, images, gen_config

    # --- Format B ---
    image_url = job_input.get("image_url")
    prompt = job_input.get("prompt", "Analyze this image.")
    images = [load_image(image_url)] if image_url else []
    return prompt, images, gen_config


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(job):
    job_input = job.get("input", {})

    try:
        prompt, images, gen_config = _parse_messages(job_input)
    except Exception as e:
        return {"error": f"Failed to parse input: {str(e)}", "type": type(e).__name__}

    if not prompt:
        return {"error": "No prompt text found in input."}

    try:
        if images:
            # Single image: (prompt, image) — multiple images: (prompt, [img1, img2, ...])
            model_input = (prompt, images[0]) if len(images) == 1 else (prompt, images)
        else:
            model_input = prompt

        response = pipe(model_input, gen_config=gen_config)
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}", "type": type(e).__name__}

    # Return an OpenAI-compatible shape so the existing test client's extract_text() works
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


runpod.serverless.start({"handler": handler})
