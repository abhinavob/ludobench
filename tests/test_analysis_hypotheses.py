"""Analysis part 3: the preregistered hypotheses on fixture runs whose answers are known."""
import csv
import os

import numpy as np
import pytest

from analysis import hypotheses
from analysis.common import find_runs
from analysis_fixture import build


@pytest.fixture(scope="session")
def rows(tmp_path_factory):
    roots = build(tmp_path_factory.mktemp("fixture"))
    out = tmp_path_factory.mktemp("out")
    hypotheses.run(find_runs(roots), str(out))
    with open(out / "hypotheses.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f)), out


def row(rows, model, test, data="main games"):
    return next(r for r in rows if (r["model"], r["test"], r["data"]) == (model, test, data))


def test_h1_first_listed_against_chance_and_the_first_listed_bot(rows):
    rows, _ = rows
    alpha = row(rows, "alpha-1b", "first-listed move vs chance")
    assert float(alpha["rate"]) == 1 and float(alpha["lo"]) > 0 and int(alpha["clusters"]) == 4
    beta = row(rows, "beta-1b", "first-listed move vs chance")
    assert float(beta["rate"]) < float(beta["reference"])           # the last-listed picker avoids it
    bot = row(rows, "alpha-1b", "agreement vs first-listed bot")
    assert float(bot["diff"]) == 0 and bot["result"] == "not refuted"  # alpha is that bot
    assert not any(r["model"] == "gamma-1b" and r["hypothesis"] == "H1" for r in rows)  # it thinks


def test_h1_chance_is_one_over_the_distinct_moves():
    ds = [{"m": {"first": 1, "chance": 1 / 2}, "game": "a"}, {"m": {"first": 0, "chance": 1 / 3}, "game": "b"}]
    r = hypotheses.against(ds, lambda d: d["m"]["first"], lambda d: d["m"]["chance"])
    assert r["reference"] == pytest.approx(5 / 12) and r["rate"] == 0.5


def test_h2_only_sixes_offering_a_bring_out_and_another_move(rows):
    rows, _ = rows
    for ref in ("chance", "search player"):
        r = row(rows, "gamma-1b", f"bring-out rate vs {ref}")      # all models, thinking or not
        assert int(r["n"]) > 0 and np.isclose(float(r["diff"]), float(r["rate"]) - float(r["reference"]))
    d = {"dice": 6, "m": {"open": {"offered": True, "taken": True, "chance": 0.5, "search": True}},
         "game": "g"}
    assert [r["n"] for r in hypotheses.h2([d, dict(d, dice=5)])] == [1, 1]


def test_h3_a_first_listed_picker_follows_the_position(rows):
    rows, _ = rows
    r = row(rows, "alpha-1b", "same position vs same piece", "spots vs spots-shuffle")
    assert float(r["rate"]) == 1 and r["only_b"] == "0" and r["result"] in ("supported", "not significant")


def test_h4_on_spots_and_games(rows):
    rows, _ = rows
    spots = row(rows, "alpha-1b", "agreement with outcomes vs without", "spots vs spots-outcomes")
    assert float(spots["diff"]) == 0 and float(spots["changed_choices"]) == 0
    games = row(rows, "alpha-1b", "agreement with outcomes vs without", "outcomes vs main games")
    assert int(games["clusters"]) == 4 and games["family"] == "preregistered"


def test_exploratory_picks_and_holm_across_the_four_hypotheses(rows):
    rows, out = rows
    ex = [r for r in rows if r["family"] == "exploratory"]
    assert ex and all(r["test"].startswith("listed first vs lowest number") for r in ex)
    pre = [r for r in rows if r["family"] == "preregistered"]
    assert {r["hypothesis"] for r in pre} == {"H1", "H2", "H3", "H4"}
    ps = [float(r["p"]) for r in pre]
    assert [float(r["p_holm"]) for r in pre] == pytest.approx(hypotheses.holm(ps))
    assert os.path.exists(out / "hypotheses.png")


def test_results_read_the_direction():
    r = {"n": 5, "diff": -0.2, "p_holm": 0.01}
    assert hypotheses.result({**r, "expected": "-"}) == "supported"
    assert hypotheses.result({**r, "expected": "+"}) == "refuted"
    assert hypotheses.result({**r, "diff": 0.2, "expected": "not +"}) == "refuted"
    assert hypotheses.result({**r, "p_holm": 0.2, "expected": "+"}) == "not significant"
