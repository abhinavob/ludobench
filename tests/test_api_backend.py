"""OpenAI-compatible API backend (OpenRouter, NVIDIA), against a fake server: no network, no key."""
import csv
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import llm_backends
import run_games
from analysis.check_api_models import check
from llm_backends import api_model_info, api_reasoning, make_llm, parse_spec

KEY = "sk-test-secret-123"
OK = {"choices": [{"message": {"content": '{"reason": "safe square", "piece": 0}',
                               "reasoning": "short"}, "finish_reason": "stop"}],
      "usage": {"prompt_tokens": 900, "completion_tokens": 20, "cost": 0.0012,
                "completion_tokens_details": {"reasoning_tokens": 5}},
      "provider": "DeepInfra"}


class FakeAPI(BaseHTTPRequestHandler):
    """/models lists the models; /chat/completions answers from `queue`, then OK."""
    models = [
        {"id": "plain/model", "supported_parameters": ["structured_outputs", "temperature"],
         "reasoning": None},
        {"id": "off/model", "supported_parameters": ["structured_outputs"],
         "reasoning": {"mandatory": False, "supported_efforts": ["none", "low", "high"]}},
        {"id": "always/model", "supported_parameters": ["structured_outputs", "temperature"],
         "reasoning": {"mandatory": True, "supported_efforts": ["high", "low", "minimal"]}},
        {"id": "loose/model", "supported_parameters": ["temperature"], "reasoning": None},
        {"id": "moonshotai/kimi-k3"}, {"id": "nv/other"}, {"id": "openai/gpt-oss-20b"},
        {"id": "mistralai/mistral-large-4-1t-a52b"},
        {"id": "nvidia/nemotron-3.5-super-vl-120b-a12b"},
    ]
    queue, requests = [], []

    def log_message(self, *args):
        pass

    def send(self, status, body, headers=()):
        self.send_response(status)
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else json.dumps(body).encode())

    def do_GET(self):
        if self.path.endswith("/endpoints"):
            self.send(200, {"data": {"endpoints": [
                {"supported_parameters": ["structured_outputs"]}, {"supported_parameters": []}]}})
        else:
            self.send(200, {"data": self.models})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append({"body": body, "auth": self.headers["Authorization"]})
        status, reply, headers = self.queue.pop(0) if self.queue else (200, OK, ())
        self.send(status, reply, headers)


@pytest.fixture
def api(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeAPI)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"
    for provider in ("openrouter", "nvidia"):
        monkeypatch.setitem(llm_backends.PROVIDER_DEFAULTS[provider], "base_url", url)
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    monkeypatch.setenv("NVIDIA_API_KEY", KEY)
    monkeypatch.setattr(llm_backends, "_next_request", {})
    FakeAPI.queue, FakeAPI.requests = [], []
    yield url
    server.shutdown()


def standard(spec, **kw):
    """make_llm as run_games sets up an API player under our rules."""
    return make_llm(spec, max_tokens=1024, stop=None, structured=True, reason_max_chars=500,
                    retries=4, retry_wait=0, **kw)


def test_api_models_are_named_after_llm():
    assert parse_spec("llm:openrouter/openai/gpt-6-luna")[:2] == ("openrouter", "openai/gpt-6-luna")
    assert parse_spec("llm:nvidia/moonshotai/kimi-k2.6")[:2] == ("nvidia", "moonshotai/kimi-k2.6")
    assert parse_spec("llm:hf.co/user/model:q4")[:2] == ("ollama_native", "hf.co/user/model:q4")
    assert run_games.folder_name("llm:nvidia/moonshotai/kimi-k2.6") == "nvidia-moonshotai-kimi-k2.6"


@pytest.mark.parametrize("provider, model, fields, reasons", [
    ("openrouter", "plain/model", {}, False),
    ("openrouter", "off/model", {"reasoning": {"enabled": False}}, False),
    ("openrouter", "always/model", {"reasoning": {"effort": "minimal"}}, True),
    ("nvidia", "nvidia/nemotron-3.5-super-vl-120b-a12b",
     {"chat_template_kwargs": {"enable_thinking": False}}, False),
    ("nvidia", "moonshotai/kimi-k3", {"reasoning_effort": "low"}, True),       # always thinks
    ("nvidia", "mistralai/mistral-large-4-1t-a52b", {"reasoning_effort": "high"}, True),
    ("nvidia", "openai/gpt-oss-20b", {"reasoning_effort": "low"}, True),
    ("nvidia", "nv/other", {}, None),                   # NVIDIA does not say
])
def test_reasoning_is_off_where_the_api_allows(api, provider, model, fields, reasons):
    assert api_reasoning(provider, api_model_info(provider, model)) == (fields, reasons)


@pytest.mark.parametrize("provider, model, effort, fields", [
    ("openrouter", "always/model", "high", {"reasoning": {"effort": "high"}}),
    ("openrouter", "off/model", "low", {"reasoning": {"effort": "low"}}),
    ("nvidia", "openai/gpt-oss-20b", "high", {"reasoning_effort": "high"}),
])
def test_effort_is_sent_as_the_api_s_reasoning_effort(api, provider, model, effort, fields):
    assert api_reasoning(provider, api_model_info(provider, model), effort) == (fields, True)


@pytest.mark.parametrize("provider, model", [("openrouter", "always/model"),
                                             ("openrouter", "plain/model"),
                                             ("nvidia", "nv/other")])
def test_effort_the_model_does_not_list_is_an_error(api, provider, model):
    with pytest.raises(ValueError, match="no reasoning effort 'medium'"):
        api_reasoning(provider, api_model_info(provider, model), "medium")


def test_model_check_reports_structured_outputs_and_reasoning(api):
    assert check("llm:openrouter/off/model") == (
        "llm:openrouter/off/model: structured outputs 1 of 2 providers; reasoning off "
        "{'reasoning': {'enabled': False}}; temperature not taken")
    assert check("llm:nvidia/nv/other").startswith(
        "llm:nvidia/nv/other: structured outputs not reported; reasoning not reported")
    assert check("llm:nvidia/no/such-model") == "llm:nvidia/no/such-model: not listed"


def test_model_the_api_does_not_list_is_an_error(api):
    with pytest.raises(ValueError, match="does not list"):
        api_model_info("openrouter", "no/such-model")


def test_openrouter_request_sends_the_schema_and_routes_only_to_providers_that_take_it(api):
    call = standard("llm:openrouter/off/model", temperature=0,
                    extra_body={"reasoning": {"enabled": False}})
    assert call("prompt") == '{"reason": "safe square", "piece": 0}'
    sent = FakeAPI.requests[0]["body"]
    schema = sent["response_format"]["json_schema"]
    assert sent["response_format"]["type"] == "json_schema" and schema["strict"] is True
    assert list(schema["schema"]["properties"]) == ["reason", "piece"]
    assert schema["schema"]["properties"]["reason"]["maxLength"] == 500
    assert schema["schema"]["additionalProperties"] is False
    assert sent["provider"] == {"require_parameters": True}
    assert sent["max_tokens"] == 1024 and sent["temperature"] == 0
    assert sent["reasoning"] == {"enabled": False}
    assert call.last_cost == 0.0012 and call.last_served_by == "DeepInfra"
    assert call.last_thinking_tokens == 5 and call.last_completion_tokens == 20
    assert call.last_retries == 0


def test_nvidia_request_has_no_openrouter_fields_and_costs_nothing(api):
    call = standard("llm:nvidia/nv/other", temperature=None)
    call("prompt")
    sent = FakeAPI.requests[0]["body"]
    assert "provider" not in sent and "usage" not in sent and "temperature" not in sent
    assert call.last_cost == 0


def test_rate_limit_and_server_errors_are_retried_and_not_invalid(api, capsys):
    FakeAPI.queue = [(429, {"error": "slow down"}, [("Retry-After", "0")]),
                     (503, {"error": "busy"}, ()),
                     (200, {"error": {"code": 502, "message": "upstream failed"}}, ())]
    call = standard("llm:openrouter/plain/model")
    assert call("prompt") == '{"reason": "safe square", "piece": 0}'
    assert call.last_retries == 3 and len(FakeAPI.requests) == 4
    err = capsys.readouterr().err
    assert "HTTP 429" in err and "HTTP 503" in err and "upstream failed" in err
    assert KEY not in err
    FakeAPI.queue = [(503, {"error": "busy"}, ())]
    call("prompt")                                      # the count is per request
    assert call.last_retries == 1
    call("prompt")
    assert call.last_retries == 0


def test_timeouts_and_cut_off_or_broken_bodies_are_retried_and_not_invalid(api, failing_urlopen,
                                                                          capsys):
    failing_urlopen(TimeoutError())
    FakeAPI.queue = [(200, b'{"choices": [', [("Content-Length", "500")]),
                     (200, b"<html>bad gateway</html>", ())]
    call = standard("llm:openrouter/plain/model")
    assert call("prompt") == '{"reason": "safe square", "piece": 0}'
    assert call.last_retries == 3 and len(FakeAPI.requests) == 3
    assert call.last_timeouts == 1
    err = capsys.readouterr().err
    assert "no reply within" in err and "IncompleteRead" in err and "Expecting value" in err


def test_third_timeout_stops_the_run(api, failing_urlopen):
    failing_urlopen(*[TimeoutError()] * 3)
    with pytest.raises(RuntimeError, match="after 3 attempts: no reply within 600s"):
        standard("llm:openrouter/plain/model")("prompt")       # though 4 attempts are allowed


def test_other_errors_stop_at_once_without_the_key(api):
    FakeAPI.queue = [(400, {"error": "bad schema"}, ())]
    call = standard("llm:nvidia/nv/other")
    with pytest.raises(RuntimeError, match="HTTP 400") as e:
        call("prompt")
    assert len(FakeAPI.requests) == 1 and KEY not in str(e.value)
    assert FakeAPI.requests[0]["auth"] == f"Bearer {KEY}"


def test_missing_key_is_an_error(api, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY")
    with pytest.raises(RuntimeError, match="set NVIDIA_API_KEY"):
        standard("llm:nvidia/nv/other")("prompt")


def test_requests_are_paced_per_api(monkeypatch):
    clock, slept = [100.0], []
    monkeypatch.setattr(llm_backends.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(llm_backends.time, "sleep", lambda s: slept.append(s) or
                        clock.__setitem__(0, clock[0] + s))
    monkeypatch.setattr(llm_backends, "_next_request", {})
    for _ in range(3):
        llm_backends._pace("https://a", 40)
    llm_backends._pace("https://b", 40)                  # another API is not held up
    assert slept == [pytest.approx(1.5), pytest.approx(1.5)]


def test_run_with_an_api_player_logs_provider_tokens_and_cost(api, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_games, "RESULTS", str(tmp_path / "results"))
    monkeypatch.setattr(sys, "argv", ["run_games.py", "--players", "llm:openrouter/always/model",
                                      "heuristic", "--max-steps", "40"])
    run_games.main()
    folder = tmp_path / "results" / "sandbox" / "openrouter-always-model_heuristic"
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    player = run["players"]["0"]
    assert player["reasoning"] == {"reasoning": {"effort": "minimal"}} and player["thinks"]
    assert player["ceiling"] == 16384 and player["temperature"] == 0     # cannot reason less
    assert all(r["body"]["max_tokens"] == 16384 for r in FakeAPI.requests)
    plies = [json.loads(line) for line in (folder / "decisions.jsonl").open(encoding="utf-8")]
    calls = [p for p in plies if p["provider"] == "openrouter"]
    assert calls and all(p["served_by"] == "DeepInfra" and p["cost_usd"] == 0.0012
                         and p["thinking_tokens"] == 5 and p["reason_first"]
                         and p["reason_chars"] == len("safe square") and p["retries"] == 0
                         and p["timeouts"] == 0 and p["hit_ceiling"] is False for p in calls)
    assert run["cost_usd"] == pytest.approx(0.0012 * len(calls))
    row, = csv.DictReader(open(tmp_path / "results" / "cells.csv", encoding="utf-8"))
    assert float(row["cost_usd"]) == pytest.approx(0.0012 * len(calls), abs=1e-4)
    for f in folder.iterdir():
        assert KEY not in f.read_text(encoding="utf-8")


def test_every_api_model_gets_the_thinking_ceiling_and_its_effort_is_recorded(api, tmp_path):
    for spec, effort in (("llm:openrouter/plain/model", None),
                         ("llm:nvidia/openai/gpt-oss-20b", "high")):
        player = run_games.player_info([spec, "heuristic"], [0, 2])[0]
        kwargs, record = run_games.llm_setup(player, False, False, effort, run_games.RELEASED)
        assert kwargs["max_tokens"] == record["ceiling"] == 16384 and record["effort"] == effort
        assert record["reasoning"] == kwargs["extra_body"]


def test_api_player_needs_its_key_and_structured_outputs(api, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_games, "RESULTS", str(tmp_path / "results"))
    monkeypatch.setattr(sys, "argv", ["run_games.py", "--players", "llm:openrouter/loose/model",
                                      "heuristic"])
    with pytest.raises(SystemExit, match="no structured-output support"):
        run_games.main()
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(SystemExit):
        run_games.main()
