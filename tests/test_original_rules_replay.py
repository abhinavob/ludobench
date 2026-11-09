"""
With original_rules=True, the games logged in llm_game_results/ (the released
rules) replay exactly. Runs under _excluded/ are skipped. The logs are not in
the repository, so these tests skip when the folder is missing.
"""
import glob
import json
import os

import pytest

import decision_log
import ludo_rules as R
import run_games
from llm_backends import DEFAULT_SCHEMA_ORDER, _parse_move_json, parse_spec

LOGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "llm_game_results")
RUNS = sorted(os.path.dirname(p) for p in
              glob.glob(os.path.join(LOGS, "**", "decisions.jsonl"), recursive=True)
              if "_excluded" not in os.path.relpath(p, LOGS).split(os.sep))
VOLATILE = {"run_id", "game_id", "latency_ms", "config"}   # config is compared without num_ctx

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not RUNS, reason="no logged games in llm_game_results/")]


def plies(run):
    with open(os.path.join(run, "decisions.jsonl"), encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def meta(run):
    with open(os.path.join(run, "run.json"), encoding="utf-8") as f:
        return json.load(f)


def board(rec, key):
    return {int(p): t for p, t in rec[key].items()}


def replay_run(run):
    """Check every ply of a run; return (plies, games)."""
    seats = meta(run)["engine"]["player_ids"]
    prev, n, games = None, 0, 0
    for rec in plies(run):
        where = f"{os.path.basename(run)} {rec['game_id']} ply {rec['ply']}"
        before, p, dice = board(rec, "tokens_before"), rec["player_id"], rec["dice"]
        legal = R.legal_moves(before, p, dice, original=True)
        assert legal == rec["legal_moves"], where
        if rec["skipped"]:
            assert not legal and rec["tokens_after"] == rec["tokens_before"], where
        else:
            assert rec["chosen_index"] in legal, where
            after, _ = R.apply_move(before, p, rec["chosen_index"], dice, original=True)
            assert after == board(rec, "tokens_after"), where
        if prev is not None and prev["game_id"] == rec["game_id"]:
            assert rec["tokens_before"] == prev["tokens_after"], where
            i = seats.index(prev["player_id"])
            expected = prev["player_id"] if prev["dice"] == 6 else seats[(i + 1) % len(seats)]
            assert p == expected, where
        else:
            assert p == seats[0] and rec["ply"] == 0, where      # seat 0 always starts
            games += 1
        prev, n = rec, n + 1
    return n, games


@pytest.mark.parametrize("run", RUNS, ids=lambda r: os.path.relpath(r, LOGS))
def test_every_logged_ply_replays(run):
    n, games = replay_run(run)
    assert n > 0 and games == meta(run)["games"]["completed"]


def replay_backend(logged_calls):
    """make_llm stand-in that answers each call with the next logged reply."""
    queue = {}
    for rec in logged_calls:
        queue.setdefault(rec["model_spec"], []).append(rec)
    queue = {spec: iter(recs) for spec, recs in queue.items()}

    def make_llm(spec, structured=False, **_):
        def call(prompt):
            rec = next(queue[spec])
            call.last_content = rec["raw_reply"] or ""
            call.last_finish_reason = rec["finish_reason"]
            call.last_reasoning = rec["reasoning_text"] or ""
            call.last_prompt_tokens = rec["prompt_tokens"]
            call.last_completion_tokens = rec["completion_tokens"]
            if rec["finish_reason"] == "timeout":
                return ""
            if not structured:
                return call.last_content
            parsed = _parse_move_json(call.last_content)
            return "" if parsed is None else f"{parsed[0]} | {parsed[1]}"
        call.spec, call.base_url = spec, parse_spec(spec)[2]
        call.last_content, call.last_finish_reason, call.last_reasoning = "", None, ""
        call.last_prompt_tokens = call.last_completion_tokens = None
        return call
    return make_llm


def rerun(run, game_index, tmp_path, monkeypatch):
    """Play one logged game again from its seed; return (logged, replayed) plies."""
    m = meta(run)
    s = m["settings"]
    seed = s["seed0"] + game_index
    tag = f'"seed": {seed},'
    with open(os.path.join(run, "decisions.jsonl"), encoding="utf-8") as f:
        logged = [r for r in map(json.loads, (line for line in f if tag in line))
                  if r["seed"] == seed]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_games, "make_llm",
                        replay_backend(r for r in logged if r["prompt_hash"]))
    monkeypatch.setattr(decision_log, "ollama_num_ctx", lambda *a, **k: None)
    out = tmp_path / f"g{game_index}"
    run_games.run_games(
        s["specs"], games=1, seed0=seed, include_legal_moves=s["include_legal_moves"],
        uncapped=s["uncapped"], out_dir=str(out), run_id="rerun",
        max_steps=s["max_steps"], no_think=s["no_think"],
        max_tokens=s["max_tokens"] if s["uncapped"] else None,
        structured=s["structured_output"],
        last_move=s.get("last_moves") or int(s.get("last_move_context", False)),
        schema_order=s.get("schema_order") or DEFAULT_SCHEMA_ORDER,
        reason_max_chars=s.get("reason_max_chars"),
        move_outcomes=s.get("move_outcomes", False),
        legal_order=s.get("legal_order") or "ascending", original_rules=True)
    replayed = [json.loads(line) for line in open(out / "decisions.jsonl", encoding="utf-8")]
    return logged, replayed


def assert_same_plies(logged, replayed, where):
    assert len(replayed) == len(logged), where
    for old, new in zip(logged, replayed):
        for key, value in old.items():
            if key not in VOLATILE:
                assert new[key] == value, f"{where} ply {old['ply']}: {key}"
        config = {k: v for k, v in old["config"].items() if k != "num_ctx"}
        assert {k: new["config"][k] for k in config} == config, f"{where} ply {old['ply']}"


BOT_RUNS = [r for r in RUNS
            if not any(run_games.is_llm_spec(s) for s in meta(r)["settings"]["specs"])
            and not any(s.startswith("gt:") and s != "gt:2" for s in meta(r)["settings"]["specs"])]
LLM_RUNS = [r for r in RUNS if r not in BOT_RUNS
            and any(run_games.is_llm_spec(s) for s in meta(r)["settings"]["specs"])]


@pytest.mark.parametrize("run", BOT_RUNS, ids=lambda r: os.path.relpath(r, LOGS))
def test_bot_games_rerun_identically_from_their_seeds(run, tmp_path, monkeypatch):
    last = meta(run)["games"]["completed"] - 1
    for g in sorted({0, last // 2, last}):
        assert_same_plies(*rerun(run, g, tmp_path, monkeypatch), f"{run} game {g}")


@pytest.mark.parametrize("run", LLM_RUNS, ids=lambda r: os.path.relpath(r, LOGS))
def test_llm_games_rerun_identically_with_logged_replies(run, tmp_path, monkeypatch):
    """Same prompts (by hash), parses, fallbacks and boards when each logged reply is given back."""
    assert_same_plies(*rerun(run, 0, tmp_path, monkeypatch), f"{run} game 0")
