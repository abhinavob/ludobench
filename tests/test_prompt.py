"""The LLM prompt: released text with original rules, our rules and the JSON reply otherwise."""
import hashlib
import json
import random
import sys

import pytest

import run_games
from llm_agent import LLMAgent
from spot_evaluation import SpotEnv

TOKENS = {0: [5, -1, 54, 57], 1: [18, 30, -1, -1], 2: [-1, 27, 64, -1]}


def prompt(original, reply="0 | x", persona=None, history=None, legal=True, **agent_kw):
    """(prompt, chosen move, env) for player 0 rolling a 6 with legal moves [0, 1, 3]."""
    env = SpotEnv([0, 1, 2], {p: list(t) for p, t in TOKENS.items()}, original_rules=original)
    env.rng = {"fallback": random.Random(0)}
    if persona:
        env.persona_text = persona
    if history:
        env.history_text = history
    got = []
    agent = LLMAgent(lambda p: got.append(p) or reply, include_legal_moves_in_prompt=legal,
                     **agent_kw)
    move = agent.select_move(env, 0, 6, [0, 1, 3])
    return got[0], move, env


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@pytest.mark.parametrize("kw, released", [
    ({}, "8d304e2f86ab3ee0"),
    ({"persona": "Play safely."}, "019a5476104f4007"),
    ({"history": "P1 captured you.", "legal": False}, "9e90cf7e7347e7d7"),
])
def test_original_rules_prompt_is_the_released_prompt(kw, released):
    # Hashes of the prompts the released llm_agent.py builds for the same inputs.
    assert sha(prompt(True, **kw)[0]) == released


def test_our_prompt_states_our_rules():
    text = prompt(False)[0]
    for phrase in ("57 pips from START", "A token never lands on or passes its own START again",
                   "any roll that does not pass HOME_END", "Capturing or finishing a token does not",
                   "(after a 6 you roll again)", "a roll-off chose who started",
                   "Your own tokens may share safe squares, home path squares and HOME_END",
                   "no square is ever blocked"):
        assert phrase in text, phrase
    for released in ("52 steps", "<int> | <one line reason>", "must land EXACTLY on HOME_END"):
        assert released not in text, released


def test_our_prompt_names_pieces_only_by_index():
    assert "token1" not in prompt(False)[0]
    assert "0→token1" in prompt(True)[0]


def test_our_prompt_asks_for_the_json_reply_reason_first():
    text = prompt(False)[0]
    assert ('{"reason": "<why you chose this move, at most 500 characters>", '
            '"piece": <token index>}') in text


def test_json_reply_format_follows_the_schema_order():
    text = prompt(False, schema_order="piece_first", reason_max_chars=None)[0]
    assert '{"piece": <token index>, "reason": "<why you chose this move>"}' in text


def test_json_reply_is_read_by_its_piece_field():
    reply = json.dumps({"reason": "token 3 is safer", "piece": 1})
    _, move, env = prompt(False, reply=reply)
    assert move == 1 and env.last_llm_reason == "token 3 is safer" and not env.last_llm_fallback


@pytest.mark.parametrize("reply", ["1 | piece one", "1", '{"piece": 1}',
                                   '{"reason": "x", "piece": "1"}', '{"reason": "cut off'])
def test_reply_that_is_not_the_json_object_is_invalid(reply):
    _, _, env = prompt(False, reply=reply)
    assert env.last_llm_fallback


def test_bad_json_reply_falls_back_on_the_fallback_stream():
    _, move, env = prompt(False, reply='{"reason": "no piece"}')
    assert env.last_llm_fallback and move == random.Random(0).choice([0, 1, 3])


def test_original_rules_read_the_bar_reply():
    _, move, env = prompt(True, reply="3 | finish the lap")
    assert move == 3 and env.last_llm_reason == "finish the lap"


def run_cli(tmp_path, monkeypatch, *args):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_games, "RESULTS", str(tmp_path / "results"))
    monkeypatch.setattr(sys, "argv", ["run_games.py", "--players", "dummy", "heuristic",
                                      *args])
    run_games.main()
    run = tmp_path / "results" / "sandbox" / "dummy" / "run.json"
    return json.loads(run.read_text(encoding="utf-8"))["settings"]


def test_runs_use_the_json_reply_by_default(tmp_path, monkeypatch):
    s = run_cli(tmp_path, monkeypatch)
    assert s["structured_output"] and s["schema_order"] == "reason_first"
    assert s["reason_max_chars"] == 500 and not s["original_rules"]


def test_original_rules_runs_keep_the_released_reply_format(tmp_path, monkeypatch):
    s = run_cli(tmp_path, monkeypatch, "--original-rules")
    assert not s["structured_output"] and s["reason_max_chars"] is None and s["original_rules"]
    assert s["seats"] == [0, 1] and s["first_player"] == "seat 0"
