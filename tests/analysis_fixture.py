"""
Small results folders for the analysis tests, made by run_games.py and run_spots.py
with a fake model: "alpha" always takes the first-listed move, "beta" and
"gamma" (a thinking model) the last (without a list, pieces 0, 1, 2, 3 in
turn). alpha's game 1 is marked repaired and replayed, as MiniMax's was, and
alpha also plays with move outcomes.
"""
import json
import re

import pytest

import run_games
import run_spots
from test_decision_records import offline

LIST = re.compile(r"legal to move this turn:\n(\[.*\])")


def fake_make_llm(spec, **_):
    def call(prompt):
        call.n += 1
        found = LIST.search(prompt)
        listed = json.loads(found.group(1)) if found else None
        piece = (call.n % 4 if listed is None else listed[0] if "alpha" in spec else listed[-1])
        call.last_content = json.dumps({"reason": "fixture", "piece": piece})
        return call.last_content
    call.n, call.base_url, call.last_content = 0, None, None
    return call


def build(tmp):
    """{label: folder} of two results roots: games ("main") and spots ("work")."""
    main, work = tmp / "main" / "results", tmp / "work" / "results"
    mp = pytest.MonkeyPatch()
    mp.chdir(tmp)
    offline(mp, fake_make_llm)
    game = lambda specs, exp, n=4, seed0=0, **kw: run_games.run_games(
        specs, games=n, seed0=seed0, out_dir=str(main / exp / run_games.run_folder(specs)), **kw)
    for model in ("alpha", "beta"):
        game([f"llm:{model}:1b", "heuristic"], "main")
    game(["llm:gamma:1b", "heuristic"], "main", think=True)
    game(["llm:alpha:1b", "heuristic"], "outcomes", move_outcomes=True)
    for bot in ("first", "random"):
        game([bot, "heuristic"], "bots")
    game(["llm:alpha:1b", "heuristic"], "hidden", include_legal_moves=False)
    game(["llm:alpha:1b", "heuristic"], "main-replay-s1", n=1, seed0=1)
    path = main / "main" / "alpha-1b_heuristic" / "run.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["repairs"] = [{"games_affected": [{"game_id": "g001-s1", "seed": 1}],
                        "use": "superseded by main-replay-s1"}]
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    spots = lambda exp, legal=("hidden", "shown"), **kw: run_spots.run_spots(
        "llm:alpha:1b", str(work / exp / "alpha-1b"), legal=legal, per_category=1, **kw)
    spots("spots")
    spots("spots-shuffle", ("shown",), legal_order="shuffled")
    spots("spots-outcomes", ("shown",), move_outcomes=True)
    spots("spots-think", ("shown",), think=True)
    run_spots.run_spots("llm:beta:1b", str(work / "spots" / "beta-1b"), per_category=1)
    mp.undo()
    return {"main": str(main), "work": str(work)}
