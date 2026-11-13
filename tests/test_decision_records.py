"""Decision records: games log the same records as before, with home-entry availability added."""
import hashlib
import json
import re
import time
import types
from pathlib import Path

import decision_log
import ludo_rules as R
import run_games

REFERENCE = Path(__file__).parent / "data" / "scripted_game_record_hashes.txt"
SPECS = ["llm:scripted:1b", "heuristic", "random", "first"]


def scripted_llm(spec, **_):
    """A model stand-in: it moves the legal pieces in turn, and every 9th reply is not JSON."""
    def call(prompt):
        call.n += 1
        legal = json.loads(re.search(r"legal to move this turn:\n(\[.*\])", prompt).group(1))
        call.last_content = ("no move" if call.n % 9 == 0 else
                             json.dumps({"reason": f"reply {call.n}", "piece": legal[call.n % len(legal)]}))
        return call.last_content
    call.n, call.base_url, call.last_content = 0, None, None
    return call


def offline(mp, make_llm):
    """No server is asked anything, and every model call takes 0 ms."""
    mp.setattr(run_games, "make_llm", make_llm)
    mp.setattr(run_games, "model_info", lambda *a: {"capabilities": ["completion", "thinking"],
                                                    "thinking_values": [False, True]})
    mp.setattr(run_games, "api_model_info", lambda *a: {"structured_outputs": True,
                                                          "temperature": True, "reasoning": None})
    mp.setattr(decision_log, "ollama_num_ctx", lambda *a, **k: None)
    mp.setattr(run_games, "time", types.SimpleNamespace(**{**vars(time), "perf_counter": lambda: 0.0}))


def read_run(out):
    decisions = [json.loads(line) for line in (out / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    games = [json.loads(line) for line in (out / "games.jsonl").read_text(encoding="utf-8").splitlines()]
    return decisions, games


def record_hashes(decisions, games):
    """SHA-256 of each record as written, without home-entry availability and wall-clock times."""
    out = []
    for rec in decisions:
        rec = dict(rec, availability={k: v for k, v in rec["availability"].items() if k != "home_entry"})
        out.append(rec)
    for rec in games:
        rec = {k: v for k, v in rec.items() if k != "timings"}
        rec["per_player"] = {pid: {k: v for k, v in p.items() if not k.startswith("home_entries_")}
                             for pid, p in rec["per_player"].items()}
        out.append(rec)
    return [hashlib.sha256(json.dumps(r).encode("utf-8")).hexdigest() for r in out]


def scripted_game(out, mp):
    offline(mp, scripted_llm)
    run_games.run_games(SPECS, games=1, seed0=7, out_dir=str(out))
    return read_run(out)


def test_game_records_are_unchanged_apart_from_home_entry(tmp_path, monkeypatch):
    decisions, games = scripted_game(tmp_path / "scripted", monkeypatch)
    reference = REFERENCE.read_text(encoding="utf-8").splitlines()[1:]
    assert record_hashes(decisions, games) == reference


def test_home_entry_is_a_move_from_the_track_into_the_home_path(tmp_path, monkeypatch):
    decisions, (game,) = scripted_game(tmp_path / "scripted", monkeypatch)
    available = dict.fromkeys(game["per_player"], 0)
    taken = dict(available)
    for d in decisions:
        before = {int(p): t for p, t in d["tokens_before"].items()}
        p = d["player_id"]
        entering = [t for t in d["legal_moves"]
                    if 0 <= before[p][t] < 52 <= R.destination(p, before[p][t], d["dice"])]
        assert d["availability"]["home_entry"] == {"any": bool(entering), "tokens": entering}
        if d["chosen_index"] is not None and entering:
            available[str(p)] += 1
            taken[str(p)] += d["chosen_index"] in entering
    assert sum(available.values()) > 10 and 0 < sum(taken.values()) < sum(available.values())
    for pid, stats in game["per_player"].items():
        assert (stats["home_entries_available"], stats["home_entries_taken"]) == (available[pid], taken[pid])
