"""
Check, without a key, which API models exist and what run_games.py would send
them: from each API's public model list, and on OpenRouter also how many of a
model's providers take a structured-output schema. NVIDIA does not report
structured-output or reasoning support; its reasoning fields come from
llm_backends.NVIDIA_REASONING.

    python analysis/check_api_models.py llm:openrouter/qwen/qwen-plus llm:nvidia/z-ai/glm-5.3-flash
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm_backends import (PROVIDER_DEFAULTS, _ollama, api_model_info,  # noqa: E402
                          api_reasoning, parse_spec)


def check(spec):
    provider, model, _ = parse_spec(spec)
    try:
        info = api_model_info(provider, model)
    except ValueError:
        return f"{spec}: not listed"
    fields, reasons = api_reasoning(provider, info)
    structured = "not reported"
    if provider == "openrouter":
        endpoints = _ollama(PROVIDER_DEFAULTS[provider]["base_url"],
                            f"/models/{model}/endpoints", timeout=30)["data"]["endpoints"]
        able = sum("structured_outputs" in e["supported_parameters"] for e in endpoints)
        structured = f"{able} of {len(endpoints)} providers"
    reasoning = {False: "off", True: "cannot be off", None: "not reported"}[reasons]
    return (f"{spec}: structured outputs {structured}; reasoning {reasoning} {fields or ''}; "
            f"temperature {'not taken' if info['temperature'] is False else 0}")


if __name__ == "__main__":
    for spec in sys.argv[1:]:
        print(check(spec))
