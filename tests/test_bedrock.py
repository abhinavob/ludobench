"""Amazon Bedrock backend (Chat Completions and Converse), against a fake server: no network, no key."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import llm_backends
import run_games
import run_spots
from analysis import bedrock_models
from analysis.check_api_models import check
from llm_backends import api_model_info, api_reasoning, bedrock_model, make_llm, parse_spec

KEY = "bedrock-test-secret-456"
KIMI = "global.moonshotai.kimi-k3.1"
HAIKU = "au.anthropic.claude-haiku-5-5"
GPT_OSS = "openai.gpt-oss-20b-1:0"
MOVE = '{"reason": "safe square", "piece": 0}'
CHAT = {"choices": [{"message": {"content": MOVE, "reasoning_content": "brief"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 900, "completion_tokens": 20,
                  "completion_tokens_details": {"reasoning_tokens": 7}}}
CONVERSE = {"output": {"message": {"role": "assistant", "content": [
                {"reasoningContent": {"reasoningText": {"text": "think", "signature": "s"}}},
                {"text": MOVE}]}},
            "stopReason": "end_turn", "usage": {"inputTokens": 800, "outputTokens": 30,
                                                "totalTokens": 830}, "metrics": {"latencyMs": 5}}


class FakeBedrock(BaseHTTPRequestHandler):
    """Runtime: chat completions and converse; control plane: model and profile lists."""
    queue, requests = [], []

    def log_message(self, *args):
        pass

    def send(self, status, body, headers=()):
        self.send_response(status)
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def do_GET(self):
        type(self).requests.append({"path": self.path, "auth": self.headers["Authorization"]})
        if self.path.startswith("/foundation-models"):
            self.send(200, {"modelSummaries": [
                {"modelId": "anthropic.claude-haiku-5-5", "providerName": "Anthropic",
                 "inferenceTypesSupported": ["INFERENCE_PROFILE"],
                 "modelLifecycle": {"status": "ACTIVE"}}]})
        elif "nextToken=abc" in self.path:
            self.send(200, {"inferenceProfileSummaries": [{"inferenceProfileId": HAIKU,
                                                           "status": "ACTIVE"}]})
        else:
            self.send(200, {"inferenceProfileSummaries": [{"inferenceProfileId": KIMI,
                                                           "status": "ACTIVE"}],
                            "nextToken": "abc"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append({"path": self.path, "body": body,
                                    "auth": self.headers["Authorization"]})
        if self.queue:
            self.send(*self.queue.pop(0))
        else:
            self.send(200, CHAT if self.path.endswith("/chat/completions") else CONVERSE)


@pytest.fixture
def bedrock(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeBedrock)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"
    for field in ("base_url", "control_url"):
        monkeypatch.setitem(llm_backends.PROVIDER_DEFAULTS["bedrock"], field, url)
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", KEY)
    monkeypatch.setattr(llm_backends, "_next_request", {})
    FakeBedrock.queue, FakeBedrock.requests = [], []
    yield url
    server.shutdown()


def standard(model, effort=None):
    """make_llm as run_games sets up a Bedrock player, with no wait between retries."""
    player = run_games.player_info([f"llm:bedrock/{model}"], [0])[0]
    kwargs, record = run_games.llm_setup(player, False, False, effort, run_games.RELEASED)
    return make_llm(f"llm:bedrock/{model}", **dict(kwargs, retry_wait=0)), record


def test_model_ids_and_au_apac_global_profiles_are_accepted():
    assert parse_spec(f"llm:bedrock/{KIMI}")[:2] == ("bedrock", KIMI)
    for model in ("moonshotai.kimi-k3.1", KIMI, "au.moonshotai.kimi-k3.1",
                  "apac.moonshotai.kimi-k3.1"):
        assert bedrock_model(model)["api"] == "chat"
    assert bedrock_model(HAIKU) == {"api": "converse", "structured_outputs": None,
                                    "reasoning": None}
    with pytest.raises(ValueError, match="another geography"):
        bedrock_model("us.moonshotai.kimi-k3.1")
    assert run_games.folder_name(f"llm:bedrock/{KIMI}") == "bedrock-global.moonshotai.kimi-k3.1"


@pytest.mark.parametrize("model, effort, fields, reasons", [
    (KIMI, None, {"reasoning_effort": "none"}, False),          # off where the model allows
    (KIMI, "low", {"reasoning_effort": "low"}, True),
    (KIMI, "high", {"reasoning_effort": "high"}, True),
    (GPT_OSS, None, {"reasoning_effort": "low"}, True),       # always reasons: its lowest level
    (GPT_OSS, "high", {"reasoning_effort": "high"}, True),
    (HAIKU, None, {}, None),                                    # nothing known: nothing sent
])
def test_reasoning_off_and_effort_mapped_to_the_model_s_levels(model, effort, fields, reasons):
    assert api_reasoning("bedrock", api_model_info("bedrock", model), effort) == (fields, reasons)


def test_effort_a_model_does_not_list_is_an_error():
    with pytest.raises(ValueError, match="no reasoning effort 'medium'"):
        api_reasoning("bedrock", api_model_info("bedrock", HAIKU), "medium")


def test_kimi_goes_through_chat_completions_with_reasoning_off(bedrock):
    call, record = standard(KIMI)
    assert call("prompt") == MOVE
    sent = FakeBedrock.requests[0]
    assert sent["path"] == "/openai/v1/chat/completions" and sent["auth"] == f"Bearer {KEY}"
    body = sent["body"]
    assert body["model"] == KIMI and body["reasoning_effort"] == "none"
    assert body["max_tokens"] == 16384 and body["temperature"] == 0
    assert list(body["response_format"]["json_schema"]["schema"]["properties"]) == ["reason", "piece"]
    assert "provider" not in body and "usage" not in body
    assert (call.provider, call.last_prompt_tokens, call.last_completion_tokens,
            call.last_thinking_tokens, call.last_cost) == ("bedrock_chat_completions", 900, 20, 7, None)
    assert call.last_reasoning == "brief" and call.last_finish_reason == "stop"
    assert record == {"reasoning": {"reasoning_effort": "none"}, "effort": None, "thinks": False,
                      "ceiling": 16384, "temperature": 0, "requests_per_minute": None,
                      "api": "chat", "schema_sent": True}


def test_other_models_go_through_converse_without_schema_or_reasoning_fields(bedrock):
    call, record = standard(HAIKU)
    assert call("prompt") == MOVE
    sent = FakeBedrock.requests[0]
    assert sent["path"] == f"/model/{HAIKU}/converse"
    assert sent["body"] == {"messages": [{"role": "user", "content": [{"text": "prompt"}]}],
                            "inferenceConfig": {"maxTokens": 16384, "temperature": 0},
                            "system": [{"text": llm_backends.DEFAULT_SYSTEM_PROMPT}]}
    assert (call.provider, call.last_prompt_tokens, call.last_completion_tokens,
            call.last_thinking_tokens, call.last_cost) == ("bedrock_converse", 800, 30, None, None)
    assert call.last_reasoning == "think" and call.last_finish_reason == "stop"
    assert record["api"] == "converse" and record["schema_sent"] is False
    FakeBedrock.queue = [(200, dict(CONVERSE, stopReason="max_tokens"))]
    call("prompt")
    assert call.last_finish_reason == "length"                 # logged as hitting the ceiling
    make_llm("llm:bedrock/anthropic.claude-x-v1:0", stop=None)("prompt")
    assert FakeBedrock.requests[-1]["path"] == "/model/anthropic.claude-x-v1%3A0/converse"


def test_converse_sends_the_schema_and_effort_of_a_listed_model(bedrock, monkeypatch):
    monkeypatch.setitem(llm_backends.BEDROCK_MODELS, "maker.model", {
        "api": "converse", "structured_outputs": True, "reasoning": {"efforts": ["low", "high"]}})
    call, record = standard("apac.maker.model", effort="high")
    call("prompt")
    body = FakeBedrock.requests[0]["body"]
    assert body["additionalModelRequestFields"] == {"reasoning_effort": "high"}
    fmt = body["outputConfig"]["textFormat"]
    schema = json.loads(fmt["structure"]["jsonSchema"]["schema"])     # sent as a string
    assert fmt["type"] == "json_schema" and schema["additionalProperties"] is False
    assert list(schema["properties"]) == ["reason", "piece"]
    assert schema["properties"]["reason"]["maxLength"] == 500
    assert record["thinks"] is True and record["effort"] == "high"


def test_gpt_oss_effort_goes_through_converse_and_into_run_json(bedrock, tmp_path):
    records = run_spots.run_spots(f"llm:bedrock/{GPT_OSS}", str(tmp_path / "run"),
                                  legal=("shown",), per_category=1, effort="high")
    assert len(records) == 13 and all(r["config"]["effort"] == "high" for r in records)
    assert all(r["path"] == "/model/openai.gpt-oss-20b-1%3A0/converse" and
               r["body"]["additionalModelRequestFields"] == {"reasoning_effort": "high"}
               and "outputConfig" not in r["body"] for r in FakeBedrock.requests)
    run = json.loads((tmp_path / "run" / "run.json").read_text(encoding="utf-8"))
    assert run["settings"]["effort"] == "high"
    assert {k: run["players"]["0"][k] for k in ("reasoning", "effort", "thinks", "api")} == {
        "reasoning": {"reasoning_effort": "high"}, "effort": "high", "thinks": True, "api": "converse"}


def test_throttling_and_server_errors_are_retried_access_denied_stops(bedrock, capsys):
    FakeBedrock.queue = [(429, {"message": "Too many tokens"}, [("Retry-After", "0")]),
                         (503, {"message": "Service unavailable"})]
    call, _ = standard(HAIKU)
    assert call("prompt") == MOVE and call.last_retries == 2
    FakeBedrock.queue = [(403, {"message": "You don't have access to the model"})]
    with pytest.raises(RuntimeError, match="HTTP 403: .*have access") as e:
        call("prompt")
    err = capsys.readouterr().err
    assert "HTTP 429" in err and "HTTP 503" in err
    assert KEY not in err and KEY not in str(e.value)


def test_a_game_logs_the_api_and_input_output_and_reasoning_tokens(bedrock, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_games, "RESULTS", str(tmp_path / "results"))
    monkeypatch.setattr(sys, "argv", ["run_games.py", "--players", f"llm:bedrock/{KIMI}",
                                      "heuristic", "--max-steps", "40"])
    run_games.main()
    folder = tmp_path / "results" / "sandbox" / "bedrock-global.moonshotai.kimi-k3.1_heuristic"
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    assert run["players"]["0"]["api"] == "chat" and run["players"]["0"]["schema_sent"]
    assert run["settings"]["llm_endpoints"]["0"] == bedrock
    plies = [json.loads(line) for line in (folder / "decisions.jsonl").open(encoding="utf-8")]
    calls = [p for p in plies if p["provider"]]
    assert calls and all(p["provider"] == "bedrock_chat_completions" and p["prompt_tokens"] == 900
                         and p["completion_tokens"] == 20 and p["thinking_tokens"] == 7
                         and p["reasoning_text"] == "brief" and p["cost_usd"] is None
                         for p in calls)
    assert run["cost_usd"] == 0
    for f in folder.iterdir():
        assert KEY not in f.read_text(encoding="utf-8")


def test_spots_run_through_converse(bedrock, tmp_path):
    records = run_spots.run_spots(f"llm:bedrock/{HAIKU}", str(tmp_path / "run"),
                                  legal=("shown",), per_category=1)
    assert len(records) == 13 and all(r["provider"] == "bedrock_converse" for r in records)
    assert all(r["prompt_tokens"] == 800 and r["completion_tokens"] == 30
               and r["thinking_tokens"] is None and r["reasoning_text"] == "think" for r in records)
    assert all(r["path"].endswith("/converse") for r in FakeBedrock.requests)
    players = json.loads((tmp_path / "run" / "run.json").read_text(encoding="utf-8"))["players"]
    assert players["0"]["api"] == "converse" and players["0"]["ceiling"] == 16384


def test_a_bedrock_player_needs_the_key(bedrock, tmp_path, monkeypatch):
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK")
    monkeypatch.setattr(sys, "argv", ["run_games.py", "--players", f"llm:bedrock/{KIMI}",
                                      "heuristic"])
    with pytest.raises(SystemExit):
        run_games.main()


def test_listing_and_test_request(bedrock):
    lines = bedrock_models.listing()
    assert any("anthropic.claude-haiku-5-5  Anthropic  INFERENCE_PROFILE  ACTIVE" in l for l in lines)
    assert sum(KIMI in l or HAIKU in l for l in lines) == 2              # both pages read
    assert all(r["auth"] == f"Bearer {KEY}" for r in FakeBedrock.requests)
    FakeBedrock.queue = [(403, {"message": "AccessDeniedException: not authorized"})]
    out = bedrock_models.test(HAIKU)
    assert "converse" in out[0] and "refused or failed" in out[1] and "HTTP 403" in out[1]
    assert KEY not in "\n".join(out)
    out = bedrock_models.test(KIMI)
    assert out[-1] == f"reply: {MOVE}" and "reasoning 7" in out[1]
    assert check(f"llm:bedrock/{KIMI}").startswith(
        f"llm:bedrock/{KIMI}: structured outputs sent (chat); reasoning off")
