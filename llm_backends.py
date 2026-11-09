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

ollama_native can also name another machine's Ollama server:
    ollama_native://<host>:11434/qwen2.5vl:3b
    ollama_native://<host>/qwen2.5vl:3b        (port defaults to 11434)

Usage:
    from llm_backends import make_llm
    fn = make_llm("ollama:llama3.2")
    reply = fn("some prompt")
"""

import json
import os
import time

DEFAULT_SYSTEM_PROMPT = "You are an expert Ludo player."

OLLAMA_PORT = 11434

# 127.0.0.1, not "localhost": on Windows "localhost" tries IPv6 ::1 first, and
# Ollama listens on IPv4 only by default, so each new connection waited ~2s.
PROVIDER_DEFAULTS = {
    "ollama": {
        "base_url": f"http://127.0.0.1:{OLLAMA_PORT}/v1",
        "api_key": "ollama",          # Ollama ignores it but the client requires one
    },
    "ollama_native": {
        "base_url": f"http://127.0.0.1:{OLLAMA_PORT}",  # native /api/chat, honours "think"
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": None,              # taken from OPENAI_API_KEY / OPENROUTER_API_KEY
    },
}

DEFAULT_TIMEOUT = 600      # seconds per request

# Output-token ceilings (Ollama's num_predict; thinking counts towards it).
ANSWER_CEILING = 1024
THINKING_CEILING = 8192    # provisional; set from analysis/measure_thinking.py

# Thinking settings Ollama does not report, by model name without its tag. For
# these models /api/show lists "thinking" among the capabilities but gives no
# "thinking" values. Each was checked with a one-line question sent with "think"
# false, true and "high":
# - deepseek-r1:8b (2026-11-09, Ollama 0.34.4): "think": false still produced
#   991 output tokens (2,900 characters of thinking), so it cannot be switched
#   off; "high" was accepted and ignored.
# - granite4.2:8b (2026-11-10, Ollama 0.40.16): "think": false gave 7 output
#   tokens and no thinking, true about 330 tokens with 836 characters of
#   thinking; "high" acted as true. Without a "think" field it thinks.
THINKING_NOT_REPORTED = {"deepseek-r1": [True], "granite4.2": [False, True]}

# Ollama writes a structured reply's fields in the schema's order, and that
# order changes what the model answers. With "piece" first and thinking off,
# the model must commit to a piece before it has reasoned: qwen3:4b then wrote
# piece 0 in 83 of 83 replies, even where its own reason named another token.
# With "reason" first it picked a legal piece in 9 of 10 of those positions.
SCHEMA_ORDERS = {"reason_first": ("reason", "piece"),
                 "piece_first": ("piece", "reason")}
DEFAULT_SCHEMA_ORDER = "reason_first"
REASON_MAX_CHARS = 500     # reason limit of the JSON reply unless --original-rules


def move_schema(order=DEFAULT_SCHEMA_ORDER, reason_max_chars=None):
    """
    JSON schema passed as Ollama's "format" when structured output is on.

    reason_max_chars caps the reason's length inside the schema. A token limit
    just stops the reply wherever it is, which with the reason first leaves no
    piece; a length in the schema makes the model close the reason and then
    write the piece.
    """
    props = {"piece": {"type": "integer", "minimum": 0, "maximum": 3},
             "reason": {"type": "string"}}
    if reason_max_chars is not None:
        props["reason"]["maxLength"] = reason_max_chars
    fields = SCHEMA_ORDERS[order]
    return {"type": "object",
            "properties": {name: props[name] for name in fields},
            "required": list(fields)}


def parse_spec(spec):
    """
    Split "provider:model" or "ollama_native://host[:port]/model" into
    (provider, model, base_url). Model names may contain ':' and '/'.
    """
    if ":" not in spec:
        raise ValueError(
            f"backend spec must be 'provider:model' or 'dummy', got {spec!r}")

    provider, rest = spec.split(":", 1)       # split once: model may contain ':'
    if provider not in PROVIDER_DEFAULTS:
        raise ValueError(f"unknown provider {provider!r}; "
                         f"expected one of {sorted(PROVIDER_DEFAULTS)} or 'dummy'")

    if not rest.startswith("//"):
        return provider, rest, PROVIDER_DEFAULTS[provider]["base_url"]

    if provider != "ollama_native":
        raise ValueError(f"a remote host is only supported for ollama_native, "
                         f"got {spec!r}")
    hostport, _, model = rest[2:].partition("/")   # model keeps any later '/' or ':'
    host, sep, port = hostport.partition(":")
    if not host or not model:
        raise ValueError(f"expected ollama_native://host[:port]/model, got {spec!r}")
    if sep and not port.isdigit():
        raise ValueError(f"bad port {port!r} in {spec!r}")
    return provider, model, f"http://{host}:{port or OLLAMA_PORT}"


def check_ollama(base_url, timeout=5):
    """
    Confirm an Ollama server answers at base_url before any game starts.
    Returns its version string; raises RuntimeError naming host and port if not.
    """
    import urllib.request
    from urllib.parse import urlsplit

    root = base_url[:-3] if base_url.endswith("/v1") else base_url
    parts = urlsplit(root)
    try:
        with urllib.request.urlopen(root + "/api/version", timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")).get("version", "?")
    except Exception as e:
        reason = getattr(e, "reason", e)
        raise RuntimeError(
            f"cannot reach Ollama at {parts.hostname}:{parts.port} ({reason}). "
            f"Check that Ollama is running there. For another machine, start it with "
            f"OLLAMA_HOST=0.0.0.0 and allow port {parts.port} through its firewall."
        ) from None


def _ollama(base_url, path, body=None, timeout=10):
    """GET (body None) or POST an Ollama API path; returns the decoded JSON."""
    import urllib.request
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(base_url + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def model_info(base_url, model):
    """
    What the Ollama server reports about a model: its digest, quantisation,
    capabilities and the values its "think" field accepts (None if neither
    Ollama nor THINKING_NOT_REPORTED says).
    """
    show = _ollama(base_url, "/api/show", {"model": model})
    tags = _ollama(base_url, "/api/tags")["models"]
    digest = next((m["digest"] for m in tags if model in (m["name"], m["model"])), None)
    capabilities = show.get("capabilities", [])
    values = (show.get("thinking") or {}).get("values")
    if values is None and "thinking" in capabilities:
        values = THINKING_NOT_REPORTED.get(model.split(":")[0])
    return {"digest": digest, "quantization": show["details"].get("quantization_level"),
            "parameter_size": show["details"].get("parameter_size"),
            "capabilities": capabilities, "thinking_values": values}


def thinking_setting(model, info, think=False, effort=None):
    """
    (value for Ollama's "think" field, or None to leave it out; whether the
    model will think). By default thinking is off where the model can switch it
    off; think=True turns it on; effort sends a level such as "high".
    """
    if "thinking" not in info["capabilities"]:
        if think or effort:
            raise ValueError(f"{model} has no thinking mode")
        return None, False
    values = info["thinking_values"]
    if values is None:
        raise ValueError(f"Ollama does not report whether {model} can switch thinking "
                         f"off; check it and add it to THINKING_NOT_REPORTED")
    if effort:
        if effort not in values:
            raise ValueError(f"{model} takes no thinking effort levels "
                             f"(its think values: {values})")
        return effort, True
    if think:
        return (True if True in values else None), True
    return (False, False) if False in values else (None, True)


def _timed_out(call):
    """
    A request the server accepted but did not answer in time (e.g. a thinking
    model looping at temperature 0) is recorded as an empty, invalid reply and
    not retried: a retry would most likely loop again, and one such move must
    not end the run.
    """
    call.last_reasoning = ""
    call.last_content = ""
    call.last_prompt_tokens = call.last_completion_tokens = None
    call.last_thinking_tokens = None
    call.last_finish_reason = "timeout"
    return ""


def parse_move_json(content):
    """
    Read {"reason": str, "piece": int} from a JSON reply.
    Returns (piece, reason), or None if the reply is not that shape.
    """
    try:
        obj = json.loads(content)
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    piece = obj.get("piece")
    if not isinstance(piece, int) or isinstance(piece, bool):
        return None
    reason = obj.get("reason")
    if not isinstance(reason, str):
        return None
    return piece, " ".join(reason.split())             # one line


def reason_first(content):
    """Whether "reason" comes before "piece" in a valid JSON reply; None if not one."""
    if parse_move_json(content) is None:
        return None
    keys = list(json.loads(content))
    return keys.index("reason") < keys.index("piece")


def make_llm(spec,
             max_tokens=60,
             stop=("\n",),
             temperature=0,
             system_prompt=DEFAULT_SYSTEM_PROMPT,
             think=None,
             retries=3,
             retry_wait=5,
             timeout=DEFAULT_TIMEOUT,
             structured=False,
             schema_order=DEFAULT_SCHEMA_ORDER,
             reason_max_chars=None):
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

    timeout is the per-request limit in seconds. A request the server accepted
    but did not answer in time returns "" (an invalid reply) without retrying.
    Failing to connect at all, e.g. a refused connection or an unreachable host,
    is retried and then raises, which stops the run.

    structured=True (ollama_native only) constrains the reply to move_schema()
    via Ollama's "format" parameter, with its two fields in schema_order
    ("reason_first" or "piece_first") and, if reason_max_chars is set, the reason
    limited to that many characters. The reply is returned as received;
    LLMAgent reads the JSON.

    After each call, the returned function carries:
      last_reasoning      the model's reasoning text, for display only
      last_finish_reason  why the reply ended: "stop", "length" (ran out of
                          tokens), "timeout", or None if the provider gave none
      last_content        the raw reply text
      last_prompt_tokens / last_completion_tokens
                          token counts the server reported, or None; the
                          completion (output) tokens include any thinking
      last_thinking_tokens
                          ollama_native: tokens streamed as thinking (one
                          chunk per token); None for other providers
    and, fixed: spec, base_url.
    """
    if spec == "dummy":
        from dummy_llm import dummy_llm
        if not structured:
            return dummy_llm
        return lambda prompt: json.dumps({"reason": "dummy", "piece": int(dummy_llm(prompt))})

    provider, model, base_url = parse_spec(spec)

    if schema_order not in SCHEMA_ORDERS:
        raise ValueError(f"schema_order must be one of {sorted(SCHEMA_ORDERS)}, "
                         f"got {schema_order!r}")
    if reason_max_chars is not None and (not structured or reason_max_chars < 1):
        raise ValueError("reason_max_chars needs structured=True and a positive value")
    if structured and provider != "ollama_native":
        raise ValueError(f"structured output is only supported for ollama_native, "
                         f"got {spec!r}")

    if provider == "ollama_native":
        return _make_ollama_native(spec, model, base_url, max_tokens, stop,
                                   temperature, system_prompt, think, retries,
                                   retry_wait, timeout,
                                   move_schema(schema_order, reason_max_chars)
                                   if structured else None)

    cfg = PROVIDER_DEFAULTS[provider]
    import httpx
    from openai import OpenAI, APITimeoutError

    api_key = cfg["api_key"] or os.environ.get("OPENROUTER_API_KEY") \
        or os.environ.get("OPENAI_API_KEY")
    # max_retries=0: the client would otherwise retry timeouts on its own.
    # Retries are handled in call() below.
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout,
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
        call.last_content = ""
        call.last_prompt_tokens = call.last_completion_tokens = None
        call.last_thinking_tokens = None
        last_err = None
        for attempt in range(retries):
            try:
                resp = client.chat.completions.create(**kwargs)
                choice = resp.choices[0]
                call.last_finish_reason = choice.finish_reason
                if resp.usage is not None:
                    call.last_prompt_tokens = resp.usage.prompt_tokens
                    call.last_completion_tokens = resp.usage.completion_tokens
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
                call.last_content = (msg.content or "").strip()
                return call.last_content
            except APITimeoutError as e:
                # Only a slow answer counts against the model; failing to
                # connect is a network problem, retried below like any other.
                if not isinstance(e.__cause__, httpx.ConnectTimeout):
                    return _timed_out(call)
                last_err = e
                if attempt < retries - 1:
                    time.sleep(retry_wait * (attempt + 1))
            except Exception as e:                    # rate limits, transient 5xx
                last_err = e
                if attempt < retries - 1:
                    time.sleep(retry_wait * (attempt + 1))
        raise RuntimeError(f"{spec} failed after {retries} attempts "
                           f"at {base_url}: {last_err}")

    call.spec = spec
    call.base_url = base_url
    call.last_reasoning = ""
    call.last_finish_reason = None
    call.last_content = ""
    call.last_prompt_tokens = call.last_completion_tokens = None
    call.last_thinking_tokens = None
    return call


def _make_ollama_native(spec, model, base_url, max_tokens, stop, temperature,
                        system_prompt, think, retries, retry_wait, timeout,
                        schema):
    """
    Call Ollama's native /api/chat. Its /v1 OpenAI-compatible endpoint ignores
    "think", so this is the only way to switch a thinking model's reasoning off.
    The reply is streamed, one token per chunk, to count the thinking tokens;
    the timeout bounds the whole reply. Uses urllib (no extra dependency).
    """
    import urllib.error
    import urllib.request

    url = base_url + "/api/chat"
    structured = schema is not None             # a schema means structured output

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
        payload = {"model": model, "messages": messages, "stream": True,
                   "options": options}
        if think is not None:
            payload["think"] = think
        if structured:
            payload["format"] = schema

        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        call.last_finish_reason = None
        call.last_content = ""
        call.last_prompt_tokens = call.last_completion_tokens = None
        call.last_thinking_tokens = None
        last_err = None
        for attempt in range(retries):
            try:
                deadline = time.monotonic() + timeout
                thinking, content, done = [], [], {}
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    for line in resp:
                        chunk = json.loads(line)
                        if "error" in chunk:
                            raise RuntimeError(chunk["error"])
                        msg = chunk.get("message", {})
                        if msg.get("thinking"):
                            thinking.append(msg["thinking"])
                        content.append(msg.get("content") or "")
                        if chunk.get("done"):
                            done = chunk
                        elif time.monotonic() > deadline:
                            return _timed_out(call)   # closing the stream stops the model
                call.last_finish_reason = done.get("done_reason")
                call.last_prompt_tokens = done.get("prompt_eval_count")
                call.last_completion_tokens = done.get("eval_count")
                call.last_thinking_tokens = len(thinking)
                # Same contract as the /v1 path: reasoning is display-only.
                call.last_reasoning = "".join(thinking).strip()
                call.last_content = "".join(content).strip()
                return call.last_content
            except urllib.error.HTTPError as e:
                # Ollama explains the failure in the body, e.g. "model not found".
                detail = e.read().decode("utf-8", errors="replace").strip()
                last_err = RuntimeError(f"HTTP {e.code}: {detail}")
                if 400 <= e.code < 500:
                    break                             # bad request: retrying won't help
                if attempt < retries - 1:
                    time.sleep(retry_wait * (attempt + 1))
            except TimeoutError:
                # Connected, but no answer in time: the model's fault, not the
                # network's. (A connect timeout arrives wrapped in URLError and
                # is handled below.)
                return _timed_out(call)
            except Exception as e:                    # refused, unreachable, 5xx
                last_err = getattr(e, "reason", e)
                if attempt < retries - 1:
                    time.sleep(retry_wait * (attempt + 1))
        raise RuntimeError(f"{spec} failed after {attempt + 1} attempt(s) "
                           f"at {base_url}: {last_err}")

    call.spec = spec
    call.base_url = base_url
    call.last_reasoning = ""
    call.last_finish_reason = None
    call.last_content = ""
    call.last_prompt_tokens = call.last_completion_tokens = None
    call.last_thinking_tokens = None
    return call
