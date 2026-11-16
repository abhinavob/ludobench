"""Analysis part 4: hidden-list games against main games, thinking against not on the same spots."""
import os

import pytest

from analysis import side_experiments
from analysis.common import find_runs
from analysis_fixture import build


@pytest.fixture(scope="session")
def runs(tmp_path_factory):
    return find_runs(build(tmp_path_factory.mktemp("fixture")))


def test_hidden_games_are_paired_with_main_games_on_seeds(runs):
    (row,) = side_experiments.hidden_vs_main(runs)
    assert row["model"] == "alpha-1b" and row["seeds"] == 4
    assert row["invalid_main"] == 0 < row["invalid_hidden"]     # without a list it guesses pieces
    assert row["invalid_diff"] == pytest.approx(row["invalid_hidden"] - row["invalid_main"])
    assert row["win_p_holm"] >= row["win_p"]


def test_thinking_is_paired_with_the_same_spots_without_thinking(runs):
    (row,) = side_experiments.thinking(runs)
    assert row["model"] == "alpha-1b" and row["spots_asked"] == 13
    assert 0 < row["spots_paired"] <= 13
    assert row["diff"] == 0 and row["p"] == 1                   # it takes the first-listed move either way


def test_figures_and_tables_are_written(runs, tmp_path):
    side_experiments.run(runs, str(tmp_path))
    for name in ("side_hidden_vs_main", "side_thinking"):
        assert os.path.exists(tmp_path / f"{name}.csv") and os.path.exists(tmp_path / f"{name}.png")
