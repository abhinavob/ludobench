"""
Part 2: spots (experiment "spots": every spot asked with the legal-move list
hidden and shown).

Only decisions with at least 2 distinct moves count (equivalent moves
grouped, positions.py), and each table comes twice: all spots, and only the
spots left unchanged by repair_spots.py (the robustness check).

Tables:
  spots_agreement.csv     per model, condition, spot set and category (and
                          "all"): decisions, invalid replies, agreement with
                          the search player among valid replies (Wilson
                          interval), and the chance level on the same
                          decisions: 1 / number of distinct moves, averaged
  spots_choice_rates.csv  per model, condition, spot set and category, per
                          kind (capture, safe and not a bring-out, open = a
                          bring-out, home entry, finish): decisions offering
                          it, the model's rate (Wilson), its chance rate (the
                          share of distinct moves of that kind, averaged) and
                          the search player's rate there
  spots_hidden_vs_shown.csv
                          per model and spot set: agreement with the list
                          hidden and shown on the same spots, paired
                          (bootstrap over spots, exact McNemar, Holm)
Figures: spots_agreement (shown, all spots), spots_conditions, spots_choice_rates.
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

from analysis.common import REPO, decisions, figure, save_figure, write_csv
from analysis.positions import complete, measures
from analysis.stats import holm, paired, wilson
from decision_log import RATE_KINDS

CONDITIONS = {False: "hidden", True: "shown"}
SPOT_SETS = {"all": lambda d: True, "unchanged": lambda d: not d["spot_changed"]}


def categories(folder):
    """spot id -> category (its scenario) for a spot folder of the repo."""
    return {s["id"]: s["scenario"] for f in sorted(glob.glob(os.path.join(REPO, folder, "spots_*.json")))
            for s in json.load(open(f, encoding="utf-8"))}


def spot_decisions(run):
    """The run's decisions, completed, with their category and measures."""
    cats = categories(run.settings["spots_folder"])
    out = []
    for d in decisions(run):
        complete(d)
        d["category"], d["m"] = cats[d["spot_id"]], measures(d)
        out.append(d)
    return out


def rate(k, n):
    lo, hi = wilson(k, n)
    return {"rate": k / n if n else None, "lo": lo, "hi": hi}


def agreement_row(ds):
    """Counts and rates of decisions with a real choice."""
    ds = [d for d in ds if d["m"]["choice"]]
    valid = [d for d in ds if d["m"]["valid"]]
    agree = sum(d["m"]["agree"] for d in valid)
    r = rate(agree, len(valid))
    return {"decisions": len(ds), "invalid": len(ds) - len(valid),
            "invalid_rate": (len(ds) - len(valid)) / len(ds) if ds else None,
            "valid": len(valid), "agree": agree, "agreement": r["rate"], "agreement_lo": r["lo"],
            "agreement_hi": r["hi"],
            "chance": float(np.mean([d["m"]["chance"] for d in valid])) if valid else None}


def choice_rows(ds):
    valid = [d for d in ds if d["m"]["choice"] and d["m"]["valid"]]
    rows = []
    for kind in RATE_KINDS:
        offered = [d["m"][kind] for d in valid if d["m"][kind]["offered"]]
        taken = sum(o["taken"] for o in offered)
        r = rate(taken, len(offered))
        rows.append({"kind": kind, "offered": len(offered), "taken": taken, "rate": r["rate"],
                     "rate_lo": r["lo"], "rate_hi": r["hi"],
                     "chance_rate": float(np.mean([o["chance"] for o in offered])) if offered else None,
                     "search_rate": float(np.mean([o["search"] for o in offered])) if offered else None})
    return rows


def cells(by_model):
    """(model, condition, spot set, category, decisions) for every table cell."""
    for model, ds in sorted(by_model.items()):
        for shown, cond in CONDITIONS.items():
            for set_name, keep in SPOT_SETS.items():
                sub = [d for d in ds if d["shown"] is shown and keep(d)]
                if not sub:
                    continue
                by_cat = defaultdict(list)
                for d in sub:
                    by_cat[d["category"]].append(d)
                for cat, cds in [("all", sub), *sorted(by_cat.items())]:
                    yield {"model": model, "condition": cond, "spot_set": set_name, "category": cat}, cds


def hidden_vs_shown(by_model):
    rows = []
    for model, ds in sorted(by_model.items()):
        for set_name, keep in SPOT_SETS.items():
            pairs = defaultdict(dict)
            for d in ds:
                if d["m"]["choice"] and d["m"]["valid"] and keep(d):
                    pairs[d["spot_id"]][d["shown"]] = d["m"]["agree"]
            both = [p for p in pairs.values() if len(p) == 2]
            if both:
                rows.append({"model": model, "spot_set": set_name,
                             **paired([p[True] for p in both], [p[False] for p in both])})
    for r, p in zip(rows, holm([r["p_mcnemar"] for r in rows])):
        r["p_holm"] = p
    return rows      # rate_a: shown, rate_b: hidden


def plot(agree, choice, out):
    sel = [r for r in agree if r["condition"] == "shown" and r["spot_set"] == "all"]
    models = sorted({r["model"] for r in sel})
    cats = ["all"] + sorted({r["category"] for r in sel} - {"all"})
    if not models:
        return
    val = {(r["model"], r["category"]): r for r in sel}
    mat = np.array([[val[(m, c)]["agreement"] if (m, c) in val and val[(m, c)]["agreement"] is not None
                     else np.nan for c in cats] for m in models])
    chance = [np.nanmean([val[(m, c)]["chance"] or np.nan for m in models if (m, c) in val]) for c in cats]
    fig, ax = figure(figsize=(1 + 0.7 * len(cats), 1.2 + 0.4 * (len(models) + 1)))
    im = ax.imshow(np.vstack([mat, chance]), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for i, row in enumerate(np.vstack([mat, chance])):
        for j, v in enumerate(row):
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v < 0.5 else "black")
    ax.set_xticks(range(len(cats)), cats, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(models) + 1), models + ["chance (1 / distinct moves)"], fontsize=7)
    fig.colorbar(im, label="agreement with the search player (legal moves shown)")
    save_figure(fig, out, "spots_agreement")

    fig, ax = figure(figsize=(7, 0.4 * len(models) + 1.5))
    y = np.arange(len(models))
    for k, (cond, color) in enumerate((("hidden", "#DD8452"), ("shown", "#4C72B0"))):
        rows = {r["model"]: r for r in agree if r["condition"] == cond and r["spot_set"] == "all"
                and r["category"] == "all"}
        ax.barh(y + 0.4 * k - 0.2, [rows[m]["agreement"] if m in rows else 0 for m in models], 0.4,
                color=color, label=f"list {cond}")
        ax.scatter([rows[m]["chance"] if m in rows else np.nan for m in models], y + 0.4 * k - 0.2,
                   marker="|", color="k", s=60, zorder=3)
    ax.set_yticks(y, models, fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("agreement with the search player (| = chance level)")
    ax.legend(fontsize=7)
    save_figure(fig, out, "spots_conditions")

    sel = [r for r in choice if r["condition"] == "shown" and r["spot_set"] == "all" and r["category"] == "all"]
    kinds = list(RATE_KINDS)
    fig, ax = figure(figsize=(8, 4))
    width = 0.8 / (len(models) + 2)
    for i, m in enumerate(models + ["chance", "search"]):
        key = {"chance": "chance_rate", "search": "search_rate"}.get(m, "rate")
        # chance and search: their rates on each model's decisions, averaged over models
        vals = [np.nanmean([r[key] if r[key] is not None else np.nan for r in sel
                            if r["kind"] == kind and (r["model"] == m or key != "rate")] or [np.nan])
                for kind in kinds]
        ax.bar(np.arange(len(kinds)) + i * width, vals, width,
               label={"chance": "chance (distinct moves)", "search": "search player"}.get(m, m),
               color={"chance": "0.6", "search": "k"}.get(m))
    ax.set_xticks(np.arange(len(kinds)) + 0.4, ["capture", "safe (not bring-out)", "open (bring-out)",
                                               "home entry", "finish"], fontsize=8)
    ax.set_ylabel("share of decisions offering it that take it")
    ax.legend(fontsize=6, ncol=3)
    save_figure(fig, out, "spots_choice_rates")


def run(runs, out):
    by_model = {r.model: spot_decisions(r) for r in runs if r.kind == "spots" and r.experiment == "spots"}
    agree, choice = [], []
    for keys, ds in cells(by_model):
        agree.append({**keys, **agreement_row(ds)})
        choice += [{**keys, **row} for row in choice_rows(ds)]
    hvs = hidden_vs_shown(by_model)
    write_csv(out, "spots_agreement", agree)
    write_csv(out, "spots_choice_rates", choice)
    write_csv(out, "spots_hidden_vs_shown", hvs)
    plot(agree, choice, out)
    shown = [r for r in agree if (r["condition"], r["spot_set"], r["category"]) == ("shown", "all", "all")]
    return [f"spots: {len(by_model)} models; agreement with the search player (list shown, all spots) "
            + ", ".join(f"{r['model']} {r['agreement']:.2f}" for r in shown)
            + (f"; chance {shown[0]['chance']:.2f}" if shown else "")]
