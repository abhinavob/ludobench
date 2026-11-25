"""
Stage 3: checks of the output against the raw files and of the figures against
the tables, into OUT/checks.

  raw_spot_checks.csv  for a seeded sample of datasets, the win rate recomputed
                       from the raw games.jsonl and the spot agreement
                       recomputed from the complete spot file, against the tables
  totals.csv           per dataset: games and decisions in the tidy tables
                       against the raw files (kept games; the player's
                       decisions with a legal move)
  figures.csv          every figure's plotted data against its table
"""
import json
import os
import random

import numpy as np
import pandas as pd

from analysis.common import games as kept_games
from analysis.common import lines
from analysis.final import catalog as C
from analysis.final.data import file_name

FIGURES = [  # figure, table, filter on the table, key, value columns (figure, table)
    ("main_win_rates", "games_summary", {"experiment": "main"}, "display", [("win_rate", "win_rate")]),
    ("main_win_rates_short", "games_summary", {"experiment": "main"}, "display", [("win_rate", "win_rate")]),
    ("main_win_rates_blog", "games_summary", {"experiment": "main"}, "display", [("win_rate", "win_rate")]),
    ("spots_frontier_130_blog", "spots_frontier_130", {}, "display", [("agreement", "agreement")]),
    ("side_thinking_blog", "side_thinking", {}, ["model", "setting"], [("agreement", "agreement")]),
    ("size_blog", "cross_models", {}, "display", [("win_rate", "win_rate"), ("size_b", "size_b")]),
    ("size_families_blog", "cross_models", {}, "display", [("win_rate", "win_rate"), ("size_b", "size_b")]),
    ("spots_agreement_shown_short", "spots_agreement", {"experiment": "spots", "condition": "shown",
                                                        "subset": "all"}, "display", [("agreement", "agreement")]),
    ("main_invalid_rates", "games_summary", {"experiment": "main"}, "display", [("invalid_rate", "invalid_rate")]),
    ("spots_agreement_shown", "spots_agreement", {"experiment": "spots", "condition": "shown", "subset": "all"},
     "display", [("agreement", "agreement"), ("chance", "chance")]),
    ("spots_agreement_hidden", "spots_agreement", {"experiment": "spots", "condition": "hidden", "subset": "all"},
     "display", [("agreement", "agreement")]),
    ("h2h_bradley_terry", "h2h_2p_bradley_terry", {}, "display", [("strength", "strength")]),
    ("spots_frontier_130", "spots_frontier_130", {}, "display", [("agreement", "agreement")]),
    ("reasons_false_claims", "reasons_themes", {"source": "spots, shown"}, "display",
     [("false_claim_rate", "false_claim_rate")]),
    ("side_hidden_vs_shown", "side_hidden_vs_shown_spots", {}, "display", [("diff", "diff")]),
    ("cross_spots_vs_games", "cross_models", {}, "display", [("win_rate", "win_rate"),
                                                            ("agreement_shown", "agreement_shown")]),
    ("cross_spots_vs_games_short", "cross_models", {}, "display", [("win_rate", "win_rate"),
                                                                  ("agreement_shown", "agreement_shown")]),
    ("spots_category_heatmap_shown", "spots_by_category", {"experiment": "spots", "condition": "shown",
                                                           "subset": "all"}, ["model", "category"],
     [("agreement", "agreement")]),
    ("spots_category_heatmap_hidden", "spots_by_category", {"experiment": "spots", "condition": "hidden",
                                                            "subset": "all"}, ["model", "category"],
     [("agreement", "agreement")]),
    ("h2h_2p_matrix", "h2h_2p_matrix", {}, ["model", "opponent"], [("win_rate", "win_rate")]),
    ("h2h_4p_wins", "h2h_4p_wins", {}, ["experiment", "model"], [("win_rate", "win_rate")]),
    ("paper_tradeoffs_hidden", "paper_tradeoffs", {}, ["model", "condition", "tradeoff"],
     [("preference", "preference")]),
    ("side_thinking", "side_thinking", {}, ["model", "setting"], [("agreement", "agreement")]),
    ("reasoning_groups", "reasoning_groups", {}, ["measure", "group"], [("mean_over_models", "mean_over_models")]),
    ("hypotheses", "hypotheses", {"family": "preregistered"}, ["model", "hypothesis", "test", "data"],
     [("diff", "diff")]),
]


def raw_checks(ctx_out, roots, datasets, runs, rng):
    gs = pd.read_csv(os.path.join(ctx_out, "tables", "games_summary.csv"))
    sp = pd.read_csv(os.path.join(ctx_out, "tables", "spots_agreement.csv"))
    rows = []
    games_ds = [d for d in datasets if d.kind == "games" and d.experiment in ("main", "hidden", "bots")]
    for d in rng.sample(games_ds, min(10, len(games_ds))):
        recs, _ = kept_games(d.runs[0], runs)
        won = np.mean([g["winner"] == d.runs[0].pid for g in recs])
        t = gs[(gs["experiment"] == d.experiment) & (gs["model"] == d.model)]["win_rate"].iloc[0]
        rows.append({"check": f"win rate {d.key}", "raw": won, "table": t, "ok": abs(won - t) < 1e-9})
    spot_ds = [d for d in datasets if d.kind == "spots" and d.experiment == "spots"]
    for d in rng.sample(spot_ds, min(10, len(spot_ds))):
        path = os.path.join(ctx_out, "data", "spots", d.experiment, file_name(d) + ".jsonl")
        for cond in (True, False):
            recs = [r for r in lines(path) if r["config"]["legal_moves_shown"] == cond and len(r["move_groups"]) > 1]
            if not recs:
                continue
            agree = [(not r["was_fallback"]) and any(r["chosen_index"] in g and r["search_move"] in g
                                                     for g in r["move_groups"]) for r in recs]
            t = sp[(sp["experiment"] == "spots") & (sp["model"] == d.model) &
                   (sp["condition"] == ("shown" if cond else "hidden"))]["agreement"].iloc[0]
            rows.append({"check": f"agreement {d.key} {'shown' if cond else 'hidden'}", "raw": np.mean(agree),
                         "table": t, "ok": abs(np.mean(agree) - t) < 1e-9})
    return rows


def totals(ctx_out, datasets, runs):
    g = pd.read_csv(os.path.join(ctx_out, "data", "games.csv"), usecols=["experiment", "model", "game"])
    gd = pd.read_csv(os.path.join(ctx_out, "data", "game_decisions.csv"), usecols=["experiment", "model"])
    sd = pd.read_csv(os.path.join(ctx_out, "data", "spot_decisions.csv"), usecols=["experiment", "model", "setting"])
    rows = []
    for d in datasets:
        if d.experiment.startswith("h2h"):
            continue
        if d.kind == "games":
            recs, _ = kept_games(d.runs[0], runs)
            pid = d.runs[0].pid
            keep = {}
            for r in recs:
                keep.setdefault(r["_dir"], set()).add(r["game_id"])
            raw_dec = sum(1 for folder, ids in keep.items() for x in lines(os.path.join(folder, "decisions.jsonl"))
                          if x["player_id"] == pid and x["game_id"] in ids and x["legal_moves"])
            tg = int(((g["experiment"] == d.experiment) & (g["model"] == d.model)).sum())
            td = int(((gd["experiment"] == d.experiment) & (gd["model"] == d.model)).sum())
            rows.append({"dataset": d.key, "games_raw": len(recs), "games_tidy": tg, "decisions_raw": raw_dec,
                         "decisions_tidy": td, "ok": tg == len(recs) and td == raw_dec})
        else:
            raw = sum(1 for _ in lines(os.path.join(ctx_out, "data", "spots", d.experiment, file_name(d) + ".jsonl")))
            td = int(((sd["experiment"] == d.experiment) & (sd["model"] == d.model) &
                      (sd["setting"].fillna("") == d.setting)).sum())
            rows.append({"dataset": d.key, "decisions_raw": raw, "decisions_tidy": td, "ok": raw == td})
    return rows


def figure_checks(out):
    rows = []
    for fig, table, flt, key, cols in FIGURES:
        fpath = os.path.join(out, "figures", f"{fig}.csv")
        if not os.path.exists(fpath):
            rows.append({"figure": fig, "ok": False, "detail": "no plotted data"})
            continue
        f = pd.read_csv(fpath)
        t = pd.read_csv(os.path.join(out, "tables", f"{table}.csv"))
        for k, v in flt.items():
            t = t[t[k] == v]
        keys = [key] if isinstance(key, str) else key
        t = t.drop_duplicates(keys).set_index(keys)
        bad = 0
        for _, r in f.iterrows():
            k = r[keys[0]] if len(keys) == 1 else tuple(r[keys])
            for fc, tc in cols:
                if pd.isna(r[fc]) and k not in t.index:
                    continue
                if k not in t.index or not np.isclose(r[fc], t.loc[k, tc], equal_nan=True):
                    bad += 1
        rows.append({"figure": fig, "table": table, "points": len(f), "mismatches": bad, "ok": bad == 0})
    return rows


def run(out, roots, log):
    folder = os.path.join(out, "checks")
    os.makedirs(folder, exist_ok=True)
    datasets, _, runs = C.build(roots)
    rng = random.Random(0)
    for name, rows in (("raw_spot_checks", raw_checks(out, roots, datasets, runs, rng)),
                       ("totals", totals(out, datasets, runs)), ("figures", figure_checks(out))):
        t = pd.DataFrame(rows)
        t.to_csv(os.path.join(folder, f"{name}.csv"), index=False)
        log(f"checks {name}: {int(t['ok'].sum())} of {len(t)} ok")
    json.dump({"done": True}, open(os.path.join(folder, "done.json"), "w"))
