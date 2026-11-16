"""Analysis part 3: the four hypotheses on fixture runs whose answers are known."""
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


def row(rows, model, test, data):
    return next(r for r in rows if (r["model"], r["test"], r["data"]) == (model, test, data))


def test_first_listed_rates_against_chance(rows):
    rows, _ = rows
    alpha = row(rows, "alpha-1b", "first-listed move vs chance", "main games")
    beta = row(rows, "beta-1b", "first-listed move vs chance", "main games")
    assert float(alpha["rate"]) == 1 and float(alpha["diff"]) > 0 and float(alpha["lo"]) > 0
    assert float(beta["rate"]) < float(beta["reference"])         # the last-listed picker avoids it
    assert int(alpha["clusters"]) == 4                              # games, the replay included


def test_win_rate_against_the_first_listed_bot_is_paired_on_seeds(rows):
    rows, _ = rows
    r = row(rows, "alpha-1b", "win rate vs first-listed bot", "main games vs first-listed bot")
    assert r["n"] == "4" and r["family"] == "primary"


def test_bring_out_is_compared_with_chance_and_the_search_player(rows):
    rows, _ = rows
    for ref in ("random policy", "search player"):
        r = row(rows, "alpha-1b", f"bring-out rate vs {ref}", "main games")
        assert int(r["n"]) > 0 and 0 <= float(r["reference"]) <= 1
        assert np.isclose(float(r["diff"]), float(r["rate"]) - float(r["reference"]))


def test_shuffled_order_first_listed_picker_keeps_the_position(rows):
    rows, _ = rows
    r = row(rows, "alpha-1b", "same list position vs same move", "spots vs spots-shuffle")
    assert float(r["rate"]) == 1 and r["only_move"] == "0"


def test_outcome_lines_change_nothing_for_a_first_listed_picker(rows):
    rows, _ = rows
    r = row(rows, "alpha-1b", "agreement with outcomes vs without", "spots vs spots-outcomes")
    assert float(r["diff"]) == 0 and float(r["p"]) == 1


def test_holm_within_each_family_and_figure(rows):
    rows, out = rows
    assert {r["family"] for r in rows} == {"primary", "secondary"}
    assert all(float(r["p_holm"]) >= float(r["p"]) for r in rows if r["p"])
    assert os.path.exists(out / "hypotheses.png") and os.path.exists(out / "hypotheses.pdf")
