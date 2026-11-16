"""Analysis part 1 and 5: finding runs, the repaired-game exclusion, main win rates and paired comparisons."""
import csv
import json
import os

import pytest

from analysis import main_games, report
from analysis.common import decisions, find_runs, games, lines
from analysis.stats import bootstrap, holm, mcnemar, paired, wilson
from analysis_fixture import build


@pytest.fixture(scope="session")
def roots(tmp_path_factory):
    return build(tmp_path_factory.mktemp("fixture"))


def table(out, name):
    with open(os.path.join(out, f"{name}.csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_named(runs, key):
    return next(r for r in runs if r.key == key)


def test_runs_are_found_with_their_kind_model_and_player(roots):
    runs = find_runs(roots)
    keys = {r.key for r in runs}
    assert {"main:main/alpha-1b_heuristic", "main:bots/first_heuristic",
            "work:spots/alpha-1b", "work:spots-think/alpha-1b"} <= keys
    alpha = run_named(runs, "main:main/alpha-1b_heuristic")
    assert (alpha.kind, alpha.model, alpha.pid) == ("games", "alpha-1b", 0)
    assert run_named(runs, "main:bots/first_heuristic").model == "first"
    assert run_named(runs, "work:spots/alpha-1b").kind == "spots"


def test_a_repaired_game_is_replaced_by_its_replay(roots):
    runs = find_runs(roots)
    alpha = run_named(runs, "main:main/alpha-1b_heuristic")
    recs, notes = games(alpha, runs)
    assert [g["seed"] for g in recs] == [0, 1, 2, 3]
    assert recs[1]["_run"] == "main:main-replay-s1/alpha-1b_heuristic"
    assert notes == {"excluded": ["g001-s1"], "replaced_by": ["main:main-replay-s1/alpha-1b_heuristic"]}
    ds = decisions(alpha, recs)
    assert not any(d["game"] == "main:main/alpha-1b_heuristic#g001-s1" for d in ds)
    replayed = [d for d in ds if d["game"].startswith("main:main-replay-s1/")]
    logged = [d for d in lines(os.path.join(roots["main"], "main-replay-s1", "alpha-1b_heuristic",
                                            "decisions.jsonl")) if d["player_id"] == 0 and d["legal_moves"]]
    assert len(replayed) == len(logged) > 0


def test_main_tables_count_wins_and_pair_shared_seeds(roots, tmp_path):
    runs = find_runs(roots)
    main_games.run(runs, str(tmp_path))
    rows = {r["model"]: r for r in table(tmp_path, "main_win_rates")}
    beta = [json.loads(l) for l in open(os.path.join(roots["main"], "main", "beta-1b_heuristic",
                                                     "games.jsonl"), encoding="utf-8")]
    assert int(rows["beta-1b"]["wins"]) == sum(g["winner"] == 0 for g in beta)
    assert rows["alpha-1b"]["excluded_games"] == "g001-s1" and rows["alpha-1b"]["games"] == "4"
    assert float(rows["alpha-1b"]["invalid_rate"]) == 0
    assert {r["model"] for r in table(tmp_path, "bot_baselines")} == {"first", "random"}
    pairs = table(tmp_path, "main_paired")
    assert [(r["family"], r["n"]) for r in pairs if r["family"] == "model vs model"] == \
        [("model vs model", "4")]
    assert sum(r["family"] == "model vs bot" for r in pairs) == 4 and all(r["p_holm"] for r in pairs)
    for name in ("main_win_rates", "main_invalid_rates", "main_paired"):
        assert os.path.exists(tmp_path / f"{name}.png") and os.path.exists(tmp_path / f"{name}.pdf")


def test_report_writes_the_inventory_and_summary(roots, tmp_path):
    summary = report.build(roots, str(tmp_path), ["main"])
    assert summary[0].startswith("main runs: 2 models, 8 games")
    assert len(table(tmp_path, "runs")) == len(find_runs(roots))


def test_statistics():
    lo, hi = wilson(5, 10)
    assert (round(lo, 4), round(hi, 4)) == (0.2366, 0.7634)
    assert mcnemar(0, 5) == 0.0625 and mcnemar(3, 3) == 1.0 and mcnemar(0, 0) == 1.0
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    est, lo, hi, p = bootstrap([[1, 2], [0, 2], [2, 2]], lambda s: s[..., 0] / s[..., 1])
    assert est == 0.5 and lo <= est <= hi
    r = paired([1, 1, 1, 0], [0, 0, 1, 0])
    assert (r["only_a"], r["only_b"], r["diff"], r["p_mcnemar"]) == (2, 0, 0.5, 0.5)
