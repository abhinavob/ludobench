"""Ollama backend: model info, thinking settings, streamed replies, JSON parsing (against a fake server)."""
import itertools
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

import llm_backends
from llm_backends import (THINKING_NOT_REPORTED, make_llm, model_info, parse_move_json,
                          reason_first, thinking_setting)


class FakeOllama(BaseHTTPRequestHandler):
    """Answers /api/tags, /api/show and a streamed /api/chat; keeps the last chat request."""
    models = {
        "think:1b": {"capabilities": ["completion", "thinking"],
                     "thinking": {"values": [False, True], "default": True}},
        "plain:1b": {"capabilities": ["completion"]},
        "deepseek-r1:8b": {"capabilities": ["completion", "thinking"]},
        "granite4.2:8b": {"capabilities": ["completion", "thinking"]},
        "levels:1b": {"capabilities": ["completion", "thinking"],
                      "thinking": {"values": ["low", "medium", "high"], "default": "medium"}},
    }
    # With logprobs, one token per chunk; a whitespace token has an empty message.
    chunks = [{"message": {"thinking": "Piece"}}, {"message": {"content": ""}},
              {"message": {"thinking": "two"}}, {"message": {"thinking": " captures."}},
              {"message": {"content": '{"reason": "capture",'}}, {"message": {"content": ""}},
              {"message": {"content": '"piece": 2}'}}]
    chunks = [dict(c, logprobs=[{"token": "t", "logprob": 0}]) for c in chunks] + [
        {"message": {}, "done": True, "done_reason": "stop", "eval_count": 8,
         "prompt_eval_count": 40}]
    last_chat = None

    def log_message(self, *args):
        pass

    def reply(self, lines):
        self.send_response(200)
        self.end_headers()
        try:
            for line in lines:
                self.wfile.write((json.dumps(line) + "\n").encode())
        except ConnectionError:                       # the client stopped reading
            pass

    def do_GET(self):
        self.reply([{"models": [{"name": m, "model": m, "digest": f"sha-{m}"}
                                for m in self.models]}])

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/show":
            show = dict(self.models[body["model"]],
                        details={"quantization_level": "Q4_K_M", "parameter_size": "1B"})
            self.reply([show])
        else:
            type(self).last_chat = body
            self.reply(self.chunks)


@pytest.fixture(scope="module")
def ollama():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_model_info_reports_digest_quantisation_and_thinking(ollama):
    info = model_info(ollama, "think:1b")
    assert info["digest"] == "sha-think:1b" and info["quantization"] == "Q4_K_M"
    assert info["thinking_values"] == [False, True]


def test_unreported_thinking_values_come_from_the_table(ollama):
    assert model_info(ollama, "deepseek-r1:8b")["thinking_values"] == THINKING_NOT_REPORTED["deepseek-r1"]


@pytest.mark.parametrize("model, kw, expected", [
    ("think:1b", {}, (False, False)),                 # off where it can be switched off
    ("think:1b", {"think": True}, (True, True)),
    ("plain:1b", {}, (None, False)),
    ("deepseek-r1:8b", {}, (True, True)),             # cannot be switched off
    ("deepseek-r1:8b", {"think": True}, (True, True)),
    ("granite4.2:8b", {}, (False, False)),           # switch not reported, but it works
    ("granite4.2:8b", {"think": True}, (True, True)),
    ("levels:1b", {}, ("low", True)),                 # no off: its lowest level
    ("levels:1b", {"effort": "high"}, ("high", True)),
])
def test_thinking_setting(ollama, model, kw, expected):
    assert thinking_setting(model, model_info(ollama, model), **kw) == expected


@pytest.mark.parametrize("model, kw", [("plain:1b", {"think": True}), ("plain:1b", {"effort": "high"}),
                                       ("think:1b", {"effort": "high"}),
                                       ("deepseek-r1:8b", {"effort": "high"})])
def test_thinking_switch_the_model_cannot_take_is_an_error(ollama, model, kw):
    with pytest.raises(ValueError):
        thinking_setting(model, model_info(ollama, model), **kw)


def test_thinking_model_missing_from_the_table_is_an_error():
    info = {"capabilities": ["thinking"], "thinking_values": None}
    with pytest.raises(ValueError, match="THINKING_NOT_REPORTED"):
        thinking_setting("new-model:1b", info)


def test_streamed_reply_counts_thinking_tokens_and_sends_the_settings(ollama):
    host = ollama.split("//")[1]
    call = make_llm(f"ollama_native://{host}/think:1b", max_tokens=16384, stop=None,
                    think=True, num_ctx=20480, structured=True, reason_max_chars=500)
    assert call("prompt") == '{"reason": "capture","piece": 2}'
    assert call.last_thinking_tokens == 4 and call.last_reasoning == "Piecetwo captures."
    assert call.last_completion_tokens == 8 and call.last_finish_reason == "stop"
    sent = FakeOllama.last_chat
    assert sent["stream"] and sent["logprobs"] and sent["think"] is True
    assert sent["options"]["num_predict"] == 16384 and sent["options"]["num_ctx"] == 20480
    assert list(sent["format"]["properties"]) == ["reason", "piece"]
    assert sent["format"]["properties"]["reason"]["maxLength"] == 500


TOKEN = [{"token": "t", "logprob": 0}]


@pytest.mark.parametrize("chunks, thinking", [
    ([{"message": {"content": ""}, "logprobs": TOKEN},
      {"message": {"content": "{}"}, "logprobs": TOKEN}], 0),               # did not think
    ([{"message": {"thinking": "a"}, "logprobs": TOKEN},
      {"message": {"content": ""}, "logprobs": TOKEN},
      {"message": {"thinking": "b"}, "logprobs": TOKEN}], 3),               # cut off thinking
    ([{"message": {"thinking": "a"}}, {"message": {"content": "{}"}}], None),  # no logprobs
])
def test_thinking_tokens_are_every_token_before_the_answer(ollama, monkeypatch, chunks, thinking):
    monkeypatch.setattr(FakeOllama, "chunks", chunks + [{"message": {}, "done": True}])
    call = make_llm(f"ollama_native://{ollama.split('//')[1]}/think:1b", stop=None, think=True)
    call("prompt")
    assert call.last_thinking_tokens == thinking


@pytest.mark.parametrize("think, logprobs", [(True, True), ("low", True), (False, None),
                                              (None, None)])
def test_logprobs_are_asked_for_only_when_thinking_is_on(ollama, think, logprobs):
    make_llm(f"ollama_native://{ollama.split('//')[1]}/think:1b", stop=None, think=think)("p")
    assert FakeOllama.last_chat.get("logprobs") is logprobs


def thinking_model(ollama):
    return make_llm(f"ollama_native://{ollama.split('//')[1]}/think:1b", max_tokens=8192,
                    stop=None, think=True, structured=True, retry_wait=0)


def test_timeouts_are_retried_twice(ollama, failing_urlopen):
    failing_urlopen(TimeoutError(), TimeoutError())
    call = thinking_model(ollama)
    assert call("prompt") == '{"reason": "capture","piece": 2}' and call.last_retries == 2
    assert call.last_timeouts == 2
    call("prompt")
    assert (call.last_retries, call.last_timeouts) == (0, 0)


def test_reply_still_streaming_at_the_timeout_stops_the_run_after_two_retries(ollama, monkeypatch):
    clock = itertools.count(0, 1000)                  # every chunk arrives 1000 s later
    monkeypatch.setattr(llm_backends, "time", SimpleNamespace(monotonic=lambda: next(clock),
                                                              sleep=lambda s: None))
    with pytest.raises(RuntimeError, match=r"after 3 attempt\(s\).*no reply within 600s"):
        thinking_model(ollama)("prompt")


class Reply:
    """Backend stand-in that gives a fixed reply with the attributes Ollama's backend sets."""

    def __init__(self, content, finish="stop", thinking_tokens=120, output_tokens=130):
        self.spec, self.base_url = "llm:fake", None
        self.content, self.finish = content, finish
        self.thinking_tokens, self.output_tokens = thinking_tokens, output_tokens

    def __call__(self, prompt):
        self.last_content, self.last_finish_reason = self.content, self.finish
        self.last_thinking_tokens = self.thinking_tokens
        self.last_prompt_tokens, self.last_completion_tokens = 900, self.output_tokens
        self.last_reasoning = "thinking"
        return self.content


def decide(reply, **kw):
    from run_games import TrackedLLMAgent
    from spot_evaluation import SpotEnv
    env = SpotEnv([0, 2], {0: [5, -1, -1, -1], 2: [-1] * 4})
    agent = TrackedLLMAgent(reply, **kw)
    agent.select_move(env, 0, 6, [0, 1, 2, 3])
    return agent.last_decision


def test_decision_log_records_reply_size_thinking_and_time():
    d = decide(Reply('{"reason": "bring  one\\nout", "piece": 1}'))
    assert d["parsed_reason"] == "bring one out"
    assert d["reason_chars"] == len("bring  one\nout")         # as written
    assert d["completion_tokens"] == 130 and (d["retries"], d["timeouts"]) == (0, 0)
    assert d["reason_first"] is True
    assert d["thinking_tokens"] == 120 and d["hit_ceiling"] is False
    assert d["latency_ms"] is not None and not d["was_fallback"]


@pytest.mark.parametrize("reply, kw, shown, position", [
    ('{"reason": "x", "piece": 1}', {}, [0, 1, 2, 3], 1),
    ('{"reason": "x", "piece": 1}', {"legal_order": "reversed"}, [3, 2, 1, 0], 2),
    ('{"reason": "x", "piece": 7}', {"legal_order": "reversed"}, [3, 2, 1, 0], None),
    ('{"reason": "x", "piece": 1}', {"include_legal_moves_in_prompt": False}, None, None),
])
def test_decision_log_records_the_shown_list_and_chosen_position(reply, kw, shown, position):
    d = decide(Reply(reply), **kw)
    assert d["shown_legal_moves"] == shown and d["chosen_list_position"] == position


def test_reply_cut_off_at_the_ceiling_is_invalid_and_keeps_its_thinking_length():
    d = decide(Reply("", finish="length", thinking_tokens=16384, output_tokens=16384))
    assert d["was_fallback"] and d["hit_ceiling"] and d["thinking_tokens"] == 16384
    assert d["reason_chars"] is None and d["reason_first"] is None


@pytest.mark.parametrize("reply, first", [
    ('{"reason": "x", "piece": 2}', True), ('{"piece": 2, "reason": "x"}', False),
    ('{"piece": 2}', None), ("2 | x", None), ("", None),
])
def test_field_order_of_the_raw_reply_is_recorded(reply, first):
    assert reason_first(reply) is first


@pytest.mark.parametrize("reply, parsed", [
    ('{"reason": "a\\n b", "piece": 2}', (2, "a b")),
    ('{"piece": 2, "reason": "x"}', (2, "x")),
    ('{"piece": 2}', None), ('{"reason": "x", "piece": true}', None),
    ('{"reason": "x", "piece": "2"}', None), ("2 | x", None), ('{"reason": "x", "pie', None),
])
def test_json_reply_must_have_a_string_reason_and_an_integer_piece(reply, parsed):
    assert parse_move_json(reply) == parsed
