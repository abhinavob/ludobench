"""The final analysis, stage 2 (analysis/final): ranking, tests and the original paper's scoring."""
import json
from math import comb

import numpy as np
import pandas as pd
import pytest

from analysis.final import m_h2h, m_paper, m_seat, m_side, m_spots


def test_bradley_terry_orders_by_wins_and_ties_equal_records():
    p = m_h2h.bradley_terry([[0, 30, 30], [20, 0, 25], [20, 25, 0]])
    assert p[0] > p[1] == pytest.approx(p[2]) and p.sum() == pytest.approx(1)


def test_chance_test_is_the_binomial_when_every_spot_has_the_same_chance():
    n, k = 20, 15
    p = m_spots.poisson_binomial_p(k, [0.5] * n)
    pmf = [comb(n, i) / 2 ** n for i in range(n + 1)]
    assert p == pytest.approx(sum(q for q in pmf if q <= pmf[k] * (1 + 1e-7)))


def test_chi_square_survival_with_3_degrees_of_freedom():
    assert m_seat.chi2_sf_3(7.8147) == pytest.approx(0.05, abs=1e-4)


def test_repeat_answers_compare_distinct_moves_and_invalid_replies():
    groups = json.dumps([[0, 1], [2]])
    row = lambda valid, chosen: pd.Series({"valid": valid, "chosen": chosen, "move_groups": groups})
    assert m_side.same_answer(row(True, 0), row(True, 1))
    assert not m_side.same_answer(row(True, 0), row(True, 2))
    assert m_side.same_answer(row(False, 0), row(False, 2))
    assert not m_side.same_answer(row(False, 0), row(True, 0))


def test_paper_block_rate_and_grudge_effect_on_known_answers():
    d = pd.DataFrame([{"model": "m", "category": "blocked", "valid": True, "chosen": c, "heuristic_move": 1,
                       "attempt": ""} for c in (1, 1, 2, 1)])
    rows = {r["metric"]: r for r in m_paper.metric_rows(d, ["model", "category"])}
    assert (rows["block_rate"]["k"], rows["block_rate"]["n"]) == (3, 4)
    pairs = []
    for i in range(4):
        for hist, cap in (("grudge", 2.0 if i < 3 else np.nan), ("neutral", "2;3" if i == 0 else "")):
            pairs.append({"model": "m", "category": "grudge", "pair_id": f"p{i}", "history": hist, "valid": True,
                          "chosen": 0 if hist == "grudge" else i % 2, "target": 2.0, "captured_players": cap,
                          "reviewed": False})
    g = m_paper.grudge(pd.DataFrame(pairs), ["model"])[0]
    assert (g["retaliation_grudge"], g["retaliation_neutral"]) == (0.75, 0.25)
    assert g["grudge_effect"] == pytest.approx(0.5) and g["change_rate"] == 0.5
    assert np.isfinite(g["grudge_effect_lo"])


def test_short_dot_plot_names_the_top_rows_and_keeps_every_row(tmp_path):
    from analysis.final import style
    data = pd.DataFrame({"display": ["Nemotron 3.5 Super 120B A12B", "b", "c", "d"], "v": [0.8, 0.7, 0.6, 0.5],
                         "lo": [0.7, 0.6, 0.5, 0.4], "hi": [0.9, 0.8, 0.7, 0.6],
                         "reasoning_group": ["reasoning", "non-reasoning", "reasoning", "non-reasoning"]})
    style.short_dot_plot(str(tmp_path), "fig", data, 2, "v", "lo", "hi", "x", refs=[(0.55, "ref", "--", "left")],
                         band=(0.85, 0.9, "band"))
    saved = pd.read_csv(tmp_path / "figures" / "fig.csv")
    assert list(saved["label"]) == ["Nemotron 3.5 Super", "b", "c", "d"]
    assert list(saved["named"]) == [True, True, False, False]
    assert (tmp_path / "figures" / "fig.png").exists() and (tmp_path / "figures" / "fig.pdf").exists()


def test_short_scatter_labels_every_model_with_both_measures(tmp_path):
    from types import SimpleNamespace
    from analysis.final import m_cross
    t = pd.DataFrame({"display": ["Nemotron 3.5 Super 120B A12B", "b", "c"], "agreement_shown": [0.6, 0.5, 0.4],
                      "win_rate": [0.3, 0.2, np.nan], "group_games": ["non-reasoning", "reasoning", np.nan]})
    corr = {"pearson": (1, 1, 1), "spearman": (1, 1, 1)}
    m_cross.plot_spots_vs_games_short(SimpleNamespace(out=str(tmp_path)), t, corr)
    saved = pd.read_csv(tmp_path / "figures" / "cross_spots_vs_games_short.csv")
    assert list(saved["label"]) == ["Nemotron 3.5 Super", "b"]
    assert (tmp_path / "figures" / "cross_spots_vs_games_short.pdf").exists()


def test_blog_bar_chart_sorts_rows_and_keeps_the_hatching_with_its_row(tmp_path):
    from analysis.final import style
    data = pd.DataFrame({"display": ["a", "b", "c"], "v": [0.2, 0.6, 0.4],
                         "reasoning_group": ["non-reasoning", "reasoning", "non-reasoning"]})
    style.bar_chart(str(tmp_path), "fig", data, "v", "x", "title", refs=[(0.5, "ref", "left")],
                    hatched=[True, False, False], hatch_label="only some")
    saved = pd.read_csv(tmp_path / "figures" / "fig.csv")
    assert list(saved["display"]) == ["b", "c", "a"] and list(saved["hatched"]) == [False, False, True]
    assert (tmp_path / "figures" / "fig.png").exists() and not (tmp_path / "figures" / "fig.pdf").exists()


def test_reasoning_setting_labels_name_the_spots_run_by_what_it_sent():
    assert m_side.setting_label("spots run (off or lowest)", '{"reasoning_effort": "none"}') == "off"
    assert m_side.setting_label("spots run (off or lowest)", '{"think": false}') == "off"
    assert m_side.setting_label("spots run (off or lowest)", '{"reasoning_effort": "low"}') == "low"
    assert m_side.setting_label("effort-medium", '{"reasoning_effort": "medium"}') == "medium"
    assert m_side.setting_label("think", '{"think": true}') == "think"


def test_size_blog_plots_the_game_models_with_a_size(tmp_path):
    from types import SimpleNamespace
    from analysis.final import m_cross
    (tmp_path / "tables").mkdir()
    pd.DataFrame({"experiment": ["bots"], "model": ["heuristic"], "win_rate": [0.483]}).to_csv(
        tmp_path / "tables" / "games_summary.csv", index=False)
    t = pd.DataFrame({"display": ["gpt-oss-20b", "b", "c"], "size_b": [20, np.nan, 7], "win_rate": [0.4, 0.3, np.nan],
                      "group_games": ["reasoning", "non-reasoning", np.nan]})
    m_cross.plot_size_blog(SimpleNamespace(out=str(tmp_path), tables=str(tmp_path / "tables")), t)
    assert list(pd.read_csv(tmp_path / "figures" / "size_blog.csv")["display"]) == ["gpt-oss-20b"]
    assert not (tmp_path / "figures" / "size_blog.pdf").exists()


def test_size_families_blog_keeps_only_the_two_families_in_size_order(tmp_path):
    from types import SimpleNamespace
    from analysis.final import m_cross
    (tmp_path / "tables").mkdir()
    pd.DataFrame({"experiment": ["bots"], "model": ["heuristic"], "win_rate": [0.483]}).to_csv(
        tmp_path / "tables" / "games_summary.csv", index=False)
    t = pd.DataFrame({"display": ["Qwen3.5 9B", "Gemma 4 E2B", "Qwen3.5 2B", "Llama 3.1 8B", "Gemma 4 31B"],
                      "family": ["Qwen3.5", "Gemma 4", "Qwen3.5", "Llama 3.1", "Gemma 4"],
                      "size_b": [9, 2, 2, 8, 31], "win_rate": [0.19, 0.25, 0.15, 0.21, np.nan]})
    m_cross.plot_size_families_blog(SimpleNamespace(out=str(tmp_path), tables=str(tmp_path / "tables")), t)
    saved = pd.read_csv(tmp_path / "figures" / "size_families_blog.csv")
    assert list(saved["display"]) == ["Gemma 4 E2B", "Qwen3.5 2B", "Qwen3.5 9B"]
