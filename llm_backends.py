"""
Model-parameterised LLM backends.

real_llm.py hardcodes a single global MODEL, so two LLMAgents in the same game
are always the same model. This module builds a fresh call-function per model
so different models can play each other.

A backend is specified as "<provider>:<model>":
    ollama:llama3.2
    ollama:qwen2.5:7b                 (model names may contain colons)
    ollama_native:qwen3.5:9b          (native /api/chat; needed for think=False)
    openrouter:qwen/qwen-2.5-7b-instruct
    dummy                             (offline stub, no API)

Usage:
    from llm_backends import make_llm
    fn = make_llm("ollama:llama3.2")
    reply = fn("some prompt")
"""

import os
import time

DEFAULT_SYSTEM_PROMPT = "You are an expert Ludo player."

PROVIDER_DEFAULTS = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",          # Ollama ignores it but the client requires one
    },
    "ollama_native": {
        "base_url": "http://localhost:11434",   # native /api/chat, honours "think"
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": None,              # taken from OPENAI_API_KEY / OPENROUTER_API_KEY
    },
}


DEFAULT_TIMEOUT = 600      # seconds per request


def _timed_out(call):
    """
    A request that runs past the timeout (e.g. a thinking model looping at
    temperature 0) is recorded as an empty, invalid reply and not retried: a
    retry would most likely loop again, and one such move must not end the run.
    """
    call.last_reasoning = ""
    call.last_finish_reason = "timeout"
    return ""


def make_llm(spec,
             max_tokens=60,
             stop=("\n",),
             temperature=0,
             system_prompt=DEFAULT_SYSTEM_PROMPT,
             think=None,
             retries=3,
             retry_wait=5,
             timeout=DEFAULT_TIMEOUT):
    """
    Build a callable prompt -> response string for the given "provider:model" spec.

    max_tokens / stop / system_prompt are exposed deliberately: the released code
    pins them to 60 and ["\\n"] with an undocumented system prompt, which blocks
    multi-step reasoning. Pass max_tokens=None and stop=None to let a reasoning
    model think, and system_prompt=None to match what the paper actually claims.

    think=False disables a thinking model's reasoning phase, so the model answers
    directly. It only takes effect with the ollama_native provider: Ollama's /v1
    endpoint (provider "ollama") ignores it. Leave it None to use the model's
    default.

    timeout is the per-request limit in seconds. A timed-out request returns ""
    (an invalid reply) without retrying. Other failures, such as a refused
    connection, are retried and then raise, which stops the run.

    After each call, the returned function carries:
      last_reasoning      the model's reasoning text, for display only
      last_finish_reason  why the reply ended: "stop", "length" (ran out of
                          tokens), "timeout", or None if the provider gave none
    """
    if spec == "dummy":
        from dummy_llm import dummy_llm
        return dummy_llm

    if ":" not in spec:
        raise ValueError(
            f"backend spec must be 'provider:model' or 'dummy', got {spec!r}")

    provider, model = spec.split(":", 1)      # split once: model may contain ':'
    if provider not in PROVIDER_DEFAULTS:
        raise ValueError(f"unknown provider {provider!r}; "
                         f"expected one of {sorted(PROVIDER_DEFAULTS)} or 'dummy'")

    if provider == "ollama_native":
        return _make_ollama_native(spec, model, max_tokens, stop, temperature,
                                   system_prompt, think, retries, retry_wait,
                                   timeout)

    cfg = PROVIDER_DEFAULTS[provider]
    from openai import OpenAI, APITimeoutError

    api_key = cfg["api_key"] or os.environ.get("OPENROUTER_API_KEY") \
        or os.environ.get("OPENAI_API_KEY")
    # max_retries=0: the client would otherwise retry timeouts on its own.
    # Retries are handled in call() below.
    client = OpenAI(base_url=cfg["base_url"], api_key=api_key, timeout=timeout,
                    max_retries=0)

    def call(prompt):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {"model": model, "messages": messages, "temperature": temperature}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if stop:
            kwargs["stop"] = list(stop)
        if think is not None:
            # Ollama's /v1 endpoint ignores this (use ollama_native instead);
            # other providers ignore unknown extra fields.
            kwargs["extra_body"] = {"think": think}

        call.last_finish_reason = None
        last_err = None
        for attempt in range(retries):
            try:
                resp = client.chat.completions.create(**kwargs)
                choice = resp.choices[0]
                call.last_finish_reason = choice.finish_reason
                msg = choice.message
                # Thinking models put their chain of thought in a separate field.
                # Keep it for display only: it is never the answer, so an empty
                # content is returned as "" and recorded as an invalid reply.
                call.last_reasoning = ""
                for field in ("reasoning", "reasoning_content", "thinking"):
                    alt = getattr(msg, field, None)
                    if alt:
                        call.last_reasoning = str(alt).strip()
                        break
                return (msg.content or "").strip()
            except APITimeoutError:
                return _timed_out(call)
            except Exception as e:                    # rate limits, transient 5xx
                last_err = e
                if attempt < retries - 1:
                    time.sleep(retry_wait * (attempt + 1))
        raise RuntimeError(f"{spec} failed after {retries} attempts: {last_err}")

    call.spec = spec
    call.last_reasoning = ""
    call.last_finish_reason = None
    return call


def _make_ollama_native(spec, model, max_tokens, stop, temperature, system_prompt,
                        think, retries, retry_wait, timeout):
    """
    Call Ollama's native /api/chat. Its /v1 OpenAI-compatible endpoint ignores
    "think", so this is the only way to switch a thinking model's reasoning off.
    Uses urllib so it needs no extra dependency.
    """
    import json
    import urllib.error
    import urllib.request

    url = PROVIDER_DEFAULTS["ollama_native"]["base_url"] + "/api/chat"

    def call(prompt):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        options = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if stop:
            options["stop"] = list(stop)
        payload = {"model": model, "messages": messages, "stream": False,
                   "options": options}
        if think is not None:
            payload["think"] = think

        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        call.last_finish_reason = None
        last_err = None
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                call.last_finish_reason = body.get("done_reason")
                msg = body.get("message", {})
                # Same contract as the /v1 path: reasoning is display-only.
                call.last_reasoning = (msg.get("thinking") or "").strip()
                return (msg.get("content") or "").strip()
            except urllib.error.HTTPError as e:
                # Ollama explains the failure in the body, e.g. "model not found".
                detail = e.read().decode("utf-8", errors="replace").strip()
                last_err = RuntimeError(f"HTTP {e.code}: {detail}")
                if 400 <= e.code < 500:
                    break                             # bad request: retrying won't help
                if attempt < retries - 1:
                    time.sleep(retry_wait * (attempt + 1))
            except TimeoutError:                      # read timed out (socket.timeout)
                return _timed_out(call)
            except Exception as e:
                # urllib wraps a connect timeout in URLError; a refused
                # connection also arrives as URLError but is retried, then raised.
                if isinstance(getattr(e, "reason", None), TimeoutError):
                    return _timed_out(call)
                last_err = e
                if attempt < retries - 1:
                    time.sleep(retry_wait * (attempt + 1))
        raise RuntimeError(f"{spec} failed after {attempt + 1} attempt(s): {last_err}")

    call.spec = spec
    call.last_reasoning = ""
    call.last_finish_reason = None
    return call