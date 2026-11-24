"""Analysis part 2: spot agreement and choice rates, with moves grouped and real choices only."""
import copy
import csv
import os

import pytest

from analysis import spot_tables
from analysis.common import find_runs
from analysis.positions import complete
from analysis_fixture import build
from decision_log import choice_rates


@pytest.fixture(scope="session")
def roots(tmp_path_factory):
    return build(tmp_path_factory.mktemp("fixture"))


@pytest.fixture(scope="session")
def alpha(roots):
    run = next(r for r in find_runs(roots) if r.key == "work:spots/alpha-1b")
    return spot_tables.spot_decisions(run)


def table(out, name):
    with open(os.path.join(out, f"{name}.csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_groups_availability_and_search_computed_as_logged(alpha):
    for d in alpha:
        bare = complete({**copy.deepcopy(d), "move_groups": None, "availability": None, "search_move": None})
        assert (bare["move_groups"], bare["availability"], bare["search_move"]) == \
            (d["move_groups"], d["availability"], d["search_move"])


def test_agreement_counts_valid_replies_on_real_choices(alpha):
    shown = [d for d in alpha if d["shown"]]
    real = [d for d in shown if len(d["move_groups"]) >= 2]
    row = spot_tables.agreement_row(shown)
    assert row["decisions"] == len(real) and len(shown) == 13
    agree = sum(any(d["chosen_index"] in g and d["search_move"] in g for g in d["move_groups"])
                for d in real if not d["was_fallback"])
    assert row["agree"] == agree and row["invalid"] == 0
    assert 0 < row["chance"] < 1


def test_choice_rates_match_the_logged_definition(alpha):
    real = [d for d in alpha if len(d["move_groups"]) >= 2]
    ours = {r["kind"]: (r["offered"], r["taken"]) for r in spot_tables.choice_rows(real)}
    assert ours == {k: (v["available"], v["taken"]) for k, v in choice_rates(real).items()}


def test_tables_cover_conditions_spot_sets_and_categories(roots, tmp_path):
    spot_tables.run(find_runs(roots), str(tmp_path))
    rows = table(tmp_path, "spots_agreement")
    keys = {(r["model"], r["condition"], r["spot_set"]) for r in rows}
    assert {("alpha-1b", c, s) for c in ("hidden", "shown") for s in ("all", "unchanged")} <= keys
    alpha_all = [r for r in rows if r["model"] == "alpha-1b" and r["condition"] == "shown"
                 and r["category"] == "all"]
    full, kept = (int(next(r for r in alpha_all if r["spot_set"] == s)["decisions"]) for s in ("all", "unchanged"))
    assert kept <= full
    assert len({r["category"] for r in rows}) == 13                 # 12 categories and "all"
    assert {r["kind"] for r in table(tmp_path, "spots_choice_rates")} == \
        {"capture", "safe", "open", "home_entry", "finish"}
    hvs = table(tmp_path, "spots_hidden_vs_shown")
    assert {r["model"] for r in hvs} == {"alpha-1b", "beta-1b"} and all(r["p_holm"] for r in hvs)
    for name in ("spots_agreement", "spots_conditions", "spots_choice_rates"):
        assert os.path.exists(tmp_path / f"{name}.png")
