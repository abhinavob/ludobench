"""Analysis part 6: reason themes, claims checked with the engine, and their tables."""
import csv
import json
import os

import pytest

from analysis import reasons as R
from analysis.common import find_runs, lines
from analysis.positions import complete, measures
from analysis_fixture import build

# a two-player board: player 0's token 0 on 17, token 1 on 41; player 2's tokens on 20 and 12
TOKENS = {"0": [17, 41, -1, -1], "2": [20, 12, -1, -1]}


def facts(token, dice):
    return R.move_facts({"tokens_before": TOKENS, "player_id": 0, "dice": dice, "chosen_index": token})


def test_move_facts_come_from_the_engine():
    m = facts(0, 3)                                   # 17 -> 20 lands on player 2's token
    assert (m["from"], m["to"], m["capture"], m["safe_to"]) == (17, 20, True, False)
    assert m["exposed_before"] and not m["exposed_after"]   # player 2's 12 reaches 17, not 20
    m = facts(1, 6)                                   # 41 -> 47, a safe square
    assert (m["to"], m["capture"], m["safe_to"], m["exposed_after"]) == (47, False, True, False)


@pytest.mark.parametrize("reason, themes", [
    ("Moving token 0 from 17 to 20 captures the opponent's token, sending it back to base.",
     {"capture"}),
    ("Token 1 moves to 47, a safe square, so it cannot be captured by Player 2.", {"danger"}),
    ("Move token 0 while avoiding opponent capture zones.", {"danger"}),
    ("Rolling a 6 lets me bring a new token out of base.", {"bring_out"}),
    ("Advance token 0 closer to the home path.", {"progress", "home"}),
    ("This blocks the opponent.", {"blocking"}),
])
def test_themes(reason, themes):
    assert {t for t, v in R.themes(reason).items() if v} == themes


@pytest.mark.parametrize("reason, token, dice, expected", [
    ("Moving token 0 from 17 to 20 captures the opponent's token.", 0, 3, {"capture": True}),
    ("Moving token 1 to 47 captures an opponent token.", 1, 6, {"capture": False}),
    ("Move token 1 from 41 to safe square 47.", 1, 6, {"safe_square": True}),      # the square after
    ("Token 0 lands on 20, a safe square.", 0, 3, {"safe_square": False}),          # the square before
    ("Token 0 is on 17 (safe square) and moves on.", 0, 3, {"safe_square": False}),  # the start square
    ("Token 1 moves to 47, avoiding capture.", 1, 6, {"out_of_reach": True}),
    ("Token 0 moves to 18, avoiding capture.", 0, 1, {"out_of_reach": False}),     # 12 + 6
    ("Token 0 escapes the threat.", 0, 3, {"threatened": True}),
    ("Token 1 enters the home path.", 1, 6, {"home_entry": False}),
    # not read: another token, a hypothetical, a negation, on the way there
    ("Moving token 1 is fine; token 0 would capture the opponent.", 1, 6, {}),
    ("Token 1 moves to 47 without landing on a safe square occupied by others.", 1, 6, {}),
    ("Token 1 moves to 47, closest to entering the home path.", 1, 6, {}),
    ("Token 1 moves toward the safe square at 13.", 1, 6, {}),
    ("Token 1 moves to 47 and keeps it out of capture range.", 1, 6, {"out_of_reach": True}),
    ("Token 1 moves to 47, away from opponent captures.", 1, 6, {}),
    ("Moving to 59 is illegal (overshooting HOME_END 57).", 1, 6, {}),
    ("Token 1 moves to 47, the home end.", 1, 6, {"finish": False}),
])
def test_claims_about_the_chosen_move(reason, token, dice, expected):
    assert R.claims(reason, token, facts(token, dice)) == expected


@pytest.fixture(scope="session")
def runs(tmp_path_factory):
    return find_runs(build(tmp_path_factory.mktemp("fixture")))


def table(out, name):
    with open(os.path.join(out, f"{name}.csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def logged(run, player=None):
    """The run's valid replies on decisions with a real choice, as the runner logged them."""
    ds = [d for d in lines(os.path.join(run.path, "decisions.jsonl")) if d["parsed_reason"]
          and not d["was_fallback"] and (player is None or d["player_id"] == player)]
    return [d for d in ds if measures(complete(d))["choice"]]


def test_claim_verdicts_match_what_the_runs_logged(runs):
    rows = R.reason_rows(runs)
    assert {(r["model"], r["experiment"]) for r in rows} >= {
        ("alpha-1b", "main"), ("alpha-1b", "hidden"), ("alpha-1b", "outcomes"), ("alpha-1b", "spots"),
        ("alpha-1b", "spots-shuffle"), ("alpha-1b", "spots-outcomes"), ("alpha-1b", "spots-think")}
    # alpha says its move captures, beta that it lands on a safe square: the false claims
    # are the moves the runner logged as no capture, or not on a safe square or the home path
    alpha = next(r for r in runs if r.key == "work:spots/alpha-1b")
    mine = [r for r in rows if r["run"] == alpha.key]
    assert all(set(r["claims"]) == {"capture"} for r in mine)
    assert sum(not r["claims"]["capture"] for r in mine) == sum(not d["was_capture"] for d in logged(alpha))
    beta = next(r for r in runs if r.key == "main:main/beta-1b_heuristic")
    mine = [r for r in rows if r["run"] == beta.key]
    assert mine and all(set(r["claims"]) == {"safe_square"} for r in mine)
    assert sum(not r["claims"]["safe_square"] for r in mine) == sum(
        not (d["landed_on_safe"] or d["landed_in_home_path"]) for d in logged(beta, beta.pid))


def test_tables_and_sample_are_written(runs, tmp_path):
    summary = R.run(runs, str(tmp_path))
    assert summary[0].startswith("reasons: ")
    themes = table(tmp_path, "reasons_themes")
    assert all(float(r["capture"]) == 1 for r in themes if r["model"] == "alpha-1b")
    claims = {(r["model"], r["experiment"], r["claim"]): r for r in table(tmp_path, "reasons_claims")}
    c = claims[("alpha-1b", "spots", "capture")]
    assert int(c["claims"]) == int(c["reasons"]) and 0 < int(c["false"]) <= int(c["claims"])
    for r in table(tmp_path, "reasons_false_vs_disagreement"):
        assert -1 <= float(r["diff"]) <= 1 and float(r["p_holm"]) >= float(r["p"])
    sample = table(tmp_path, "reasons_sample")
    assert 0 < len(sample) <= R.SAMPLE and all(r["themes"] for r in sample)
    assert os.path.exists(tmp_path / "reasons_themes.png")
