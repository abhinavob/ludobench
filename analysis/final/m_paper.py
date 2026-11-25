"""
The original paper's spot metrics, per model, category and condition, scored as
its scripts score them (run_all_spots.py, analysis_scripts/compare_personas.py,
plot_tradeoff_preference_hierarchies.py, plot_gt_behavioral_radar.py,
plot_archetype_clustering.py) but on the moves as they are under our rules:

  invalid rate          invalid replies / all prompts of the category
  capture, safe, home entry, home finish, open (bring-out), bring-out and
  move-existing rates   share of valid replies whose move does it; "safe" is a
                        safe square, bring-outs included (the original's
                        definition); "home entry" ends in the home path
  block rate            blocked: the move equals the heuristic bot's
  overshoot rate        overshoot: a valid move past the home end, which no
                        legal move can be (0 by construction); added here: the
                        share of all replies naming a piece that would overshoot
  grudge                pairs with both halves valid: change rate (the piece
                        differs), retaliation rate in the grudge half and in the
                        neutral half (the move captures the player the grudge
                        names), grudge effect = their difference (bootstrap
                        over pairs); with and without the 6 pairs the review
                        changed
The original scoring bug: in safe_vs_openexisting the original action label
tests "safe" before "bring-out", and a bring-out lands on the start square,
which is safe, so every bring-out was labelled safe and its open rate was 0.
The table gives, per model, the share of bring-outs that labelling hid.

Reference rows: the search player and the heuristic bot on the same spots.
Tables: paper_metrics.csv, paper_grudge.csv, paper_tradeoffs.csv,
paper_open_bug.csv. Figures: paper_tradeoffs_<cond>,
paper_builder_finisher_<cond>, paper_radar_<cond>.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from analysis.common import REPO
from analysis.final import style as S
from analysis.final.metrics import cluster_mean, rate
from analysis.final.tidy import effects

METRICS = {
    "capture": [("capture_rate", "eff_capture")],
    "capture_vs_home": [("capture_rate", "eff_capture"), ("home_entry_rate", "eff_in_home")],
    "capture_vs_home_finish": [("capture_rate", "eff_capture"), ("home_finish_rate", "eff_finish")],
    "capture_vs_openexisting": [("capture_rate", "eff_capture"), ("open_rate", "eff_bring_out")],
    "capture_vs_safe": [("capture_rate", "eff_capture"), ("safe_rate", "eff_safe_square")],
    "safe": [("safe_rate", "eff_safe_square")],
    "safe_vs_openexisting": [("safe_rate", "eff_safe_square"), ("open_rate", "eff_bring_out")],
    "extra_turn": [("bring_out_rate", "eff_bring_out")],
    "home_entry": [("home_entry_rate", "eff_in_home")],
}
TRADEOFFS = [("capture_vs_safe", "safe", "eff_safe_square"), ("capture_vs_home", "home entry", "eff_in_home"),
             ("capture_vs_home_finish", "home finish", "eff_finish"),
             ("capture_vs_openexisting", "open", "eff_bring_out")]
RADAR = [("capture", "capture_rate"), ("safe", "safe_rate"), ("home_entry", "home_entry_rate"),
         ("extra_turn", "bring_out_rate"), ("capture_vs_home_finish", "home_finish_rate"),
         ("blocked", "block_rate"), ("overshoot", "overshoot_avoid_rate")]


def spot_dicts():
    """spot id -> the spot of spots_40_final/."""
    ref = {}
    for p in sorted(glob.glob(os.path.join(REPO, "spots_40_final", "spots_*.json"))):
        for s in json.load(open(p, encoding="utf-8")):
            ref[s["id"]] = s
    return ref


def references(ctx):
    spots = spot_dicts()
    r = ctx.frame("spot_reference")
    rows = []
    for col, name in (("search_move", "Search player (depth 2)"), ("heuristic", "Heuristic bot")):
        for _, x in r.iterrows():
            s = spots[x["spot_id"]]
            d = {"tokens_before": s["tokens"], "player_id": s["current_player"], "chosen_index": int(x[col]),
                 "dice": s["dice"]}
            rows.append({"experiment": "spots", "model": name, "display": name, "condition": "reference",
                         "spot_id": x["spot_id"], "category": x["category"], "valid": True, "chosen": int(x[col]),
                         "heuristic_move": int(x["heuristic"]), "pair_id": x["pair_id"], "history": x["history"],
                         "target": x["target"], "reviewed": str(x["reviewed"]) == "True", "attempt": "",
                         **effects(d)})
    return pd.DataFrame(rows)


def metric_rows(d, keys):
    out = []
    for k, x in d.groupby(keys, dropna=False):
        base = dict(zip(keys, k))
        cat = base["category"]
        v = x[x["valid"] == True]
        out.append({**base, "metric": "invalid_rate", "n": len(x), "k": int((x["valid"] != True).sum()),
                    **_rate((x["valid"] != True).sum(), len(x))})
        for name, col in METRICS.get(cat, []):
            out.append({**base, "metric": name, "n": len(v), "k": int(v[col].sum()), **_rate(v[col].sum(), len(v))})
        if cat == "extra_turn":
            out.append({**base, "metric": "move_existing_rate", "n": len(v), "k": int((~v["eff_bring_out"].astype(bool)).sum()),
                        **_rate((~v["eff_bring_out"].astype(bool)).sum(), len(v))})
        if cat == "safe_vs_openexisting":
            out.append({**base, "metric": "safe_rate_bring_outs_excluded", "n": len(v), "k": int(v["eff_safe"].sum()),
                        **_rate(v["eff_safe"].sum(), len(v))})
        if cat == "blocked":
            same = (v["chosen"] == v["heuristic_move"]).sum()
            out.append({**base, "metric": "block_rate", "n": len(v), "k": int(same), **_rate(same, len(v))})
        if cat == "overshoot":
            out.append({**base, "metric": "overshoot_rate", "n": len(v), "k": 0, **_rate(0, len(v)),
                        "note": "0 by construction: a legal move never passes the home end"})
            att = (x["attempt"] == "overshoot").sum()
            out.append({**base, "metric": "overshoot_attempt_rate", "n": len(x), "k": int(att), **_rate(att, len(x))})
            out.append({**base, "metric": "overshoot_avoid_rate", "n": len(x), "k": int(len(x) - att),
                        **_rate(len(x) - att, len(x))})
    return out


def _rate(k, n):
    p, lo, hi = rate(k, n)
    return {"rate": p, "lo": lo, "hi": hi}


def grudge(d, keys):
    g = d[d["category"] == "grudge"]
    rows = []
    for k, x in g.groupby(keys, dropna=False):
        base = dict(zip(keys, k))
        for subset, sel in (("all pairs", x), ("unchanged pairs", x[~x["pair_id"].isin(
                set(x[x["reviewed"] == True]["pair_id"]))])):
            pairs = []
            for pid, p in sel.groupby("pair_id"):
                a, b = p[p["history"] == "grudge"], p[p["history"] == "neutral"]
                if len(a) != 1 or len(b) != 1:
                    continue
                a, b = a.iloc[0], b.iloc[0]
                hit = lambda r: (r["valid"] == True and pd.notna(r["target"])
                                 and int(r["target"]) in captured(r.get("captured_players")))
                pairs.append({"pair": pid, "both_valid": a["valid"] == True and b["valid"] == True,
                              "changed": a["chosen"] != b["chosen"], "ret_grudge": hit(a), "ret_neutral": hit(b),
                              "valid_grudge": a["valid"] == True, "valid_neutral": b["valid"] == True})
            if not pairs:
                continue
            P = pd.DataFrame(pairs)
            both = P[P["both_valid"]]
            ch = rate(both["changed"].sum(), len(both))
            rg = rate(P.loc[P["valid_grudge"], "ret_grudge"].sum(), P["valid_grudge"].sum())
            rn = rate(P.loc[P["valid_neutral"], "ret_neutral"].sum(), P["valid_neutral"].sum())
            eff = cluster_mean((both["ret_grudge"].astype(int) - both["ret_neutral"].astype(int)), both["pair"], b=2000) \
                if len(both) else (np.nan,) * 4
            rows.append({**base, "pairs": subset, "n_pairs": len(P), "both_valid": len(both),
                         "change_rate": ch[0], "change_lo": ch[1], "change_hi": ch[2],
                         "retaliation_grudge": rg[0], "retaliation_grudge_lo": rg[1], "retaliation_grudge_hi": rg[2],
                         "retaliation_neutral": rn[0], "retaliation_neutral_lo": rn[1],
                         "retaliation_neutral_hi": rn[2], "grudge_effect": eff[0], "grudge_effect_lo": eff[1],
                         "grudge_effect_hi": eff[2]})
    return rows


def captured(v):
    """The players a move captured, from the tidy column ("2;3", or a number once read back from CSV)."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or v == "":
        return set()
    return {int(float(x)) for x in str(v).split(";") if x}


def tradeoffs(d, keys):
    rows = []
    for k, x in d[d["valid"] == True].groupby(keys, dropna=False):
        base = dict(zip(keys, k))
        for cat, right, col in TRADEOFFS:
            v = x[x["category"] == cat]
            left_n = int(v["eff_capture"].sum())
            right_n = int((~v["eff_capture"].astype(bool) & v[col].astype(bool)).sum())
            two = left_n + right_n
            rows.append({**base, "tradeoff": f"capture vs {right}", "category": cat, "valid": len(v),
                         "capture": left_n, "other_side": right_n, "neither": len(v) - two,
                         "preference": (right_n - left_n) / two if two else np.nan})
    return rows


def open_bug(d):
    v = d[(d["category"] == "safe_vs_openexisting") & (d["valid"] == True)]
    rows = []
    for (model, cond), x in v.groupby(["model", "condition"]):
        k = int(x["eff_bring_out"].sum())
        p, lo, hi = rate(k, len(x))
        rows.append({"model": model, "condition": cond, "valid": len(x), "bring_outs": k,
                     "open_rate_original_scoring": 0.0, "open_rate": p, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def plot_tradeoffs(ctx, tr, cond):
    x = tr[(tr["condition"].isin([cond, "reference"]))]
    if x.empty:
        return
    labels = [m for m in x.drop_duplicates("model")["model"]]
    tlist = [t for t in dict.fromkeys(x["tradeoff"])]
    fig, axes = S.figure(ncols=len(tlist), figsize=(2.6 * len(tlist) + 1.5, 0.2 * len(labels) + 1.6), sharey=True)
    disp = dict(zip(x["model"], x["display"]))
    order = sorted(labels, key=lambda m: (m not in ("Search player (depth 2)", "Heuristic bot"), disp[m]))
    y = np.arange(len(order))[::-1]
    for ax, t in zip(axes, tlist):
        v = x[x["tradeoff"] == t].set_index("model").reindex(order)
        vals = v["preference"].to_numpy()
        colors = [S.DIVERGING(0.15) if a < 0 else S.DIVERGING(0.85) for a in np.nan_to_num(vals)]
        ax.barh(y, vals, color=colors, height=0.66, edgecolor="white", linewidth=0.6)
        ax.axvline(0, color=S.MUTED, linewidth=0.8)
        ax.set_xlim(-1, 1)
        ax.set_title(t, fontsize=8)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [disp[m] for m in order])
    fig.supxlabel("preference: -1 always capture, +1 always the other move (among the two)", fontsize=8)
    fig.suptitle(f"Trade-offs, legal moves {cond} (the original paper's figure)", fontsize=10)
    S.save(fig, ctx.out, f"paper_tradeoffs_{cond}", x)


def plot_builder_finisher(ctx, pm, cond):
    x = pm[pm["condition"].isin([cond, "reference"])]
    bo = x[(x["category"] == "extra_turn") & (x["metric"] == "bring_out_rate")].set_index("model")["rate"]
    hf = x[(x["category"] == "capture_vs_home_finish") & (x["metric"] == "home_finish_rate")].set_index("model")["rate"]
    both = pd.DataFrame({"bring_out_rate": bo, "home_finish_rate": hf}).dropna()
    if both.empty:
        return
    disp = dict(zip(x["model"], x["display"]))
    fig, ax = S.figure(figsize=(6.2, 5.2))
    for m, r in both.iterrows():
        refm = m in ("Search player (depth 2)", "Heuristic bot")
        ax.plot(r["bring_out_rate"], r["home_finish_rate"], "D" if refm else "o",
                color=S.SERIES[1] if refm else S.SERIES[0], markersize=7 if refm else 5,
                markeredgecolor="white", markeredgewidth=0.8)
    S.pct(ax, "x")
    S.pct(ax, "y")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    S.note_left_out(fig, S.label_points(ax, both["bring_out_rate"], both["home_finish_rate"],
                                        [disp[m] for m in both.index]))
    ax.set_xlabel("bring-out rate on extra-turn spots (development)")
    ax.set_ylabel("home-finish rate on capture-vs-finish spots (completion)")
    ax.set_title(f"Builder vs finisher, legal moves {cond}")
    S.save(fig, ctx.out, f"paper_builder_finisher_{cond}", both.reset_index().rename(columns={"index": "model"}))


def plot_radar(ctx, pm, cond):
    x = pm[pm["condition"].isin([cond, "reference"])]
    vals = {}
    for cat, metric in RADAR:
        s = x[(x["category"] == cat) & (x["metric"] == metric)].set_index("model")["rate"]
        for m, v in s.items():
            vals.setdefault(m, {})[metric] = v
    models = [m for m, v in vals.items() if len(v) == len(RADAR) and m != "Search player (depth 2)"]
    if not models or "Search player (depth 2)" not in vals:
        return
    disp = dict(zip(x["model"], x["display"]))
    angles = np.linspace(0, 2 * np.pi, len(RADAR), endpoint=False).tolist()
    cols = 6
    rows_n = int(np.ceil(len(models) / cols))
    fig, axes = S.figure(nrows=rows_n, ncols=cols, figsize=(2.1 * cols, 2.3 * rows_n),
                         subplot_kw={"projection": "polar"}, squeeze=False)
    gt = [vals["Search player (depth 2)"][m] for _, m in RADAR]
    for ax, m in zip(axes.flat, sorted(models, key=lambda m: disp[m])):
        v = [vals[m][k] for _, k in RADAR]
        ax.plot(angles + angles[:1], gt + gt[:1], color=S.MUTED, linewidth=1, linestyle="--")
        ax.fill(angles + angles[:1], v + v[:1], color=S.SERIES[0], alpha=0.25)
        ax.plot(angles + angles[:1], v + v[:1], color=S.SERIES[0], linewidth=1.2)
        ax.set_ylim(0, 1)
        ax.set_xticks(angles, ["capture", "safe", "home\nentry", "bring\nout", "finish", "block", "no\novershoot"],
                      fontsize=5.5)
        ax.set_yticks([0.5, 1.0], ["", ""])
        ax.set_title(disp[m], fontsize=7, pad=8)
    for ax in axes.flat[len(models):]:
        ax.axis("off")
    fig.suptitle(f"Behaviour profiles, legal moves {cond} (dashed: the search player)", fontsize=10)
    data = pd.DataFrame([{"model": m, **vals[m]} for m in vals])
    S.save(fig, ctx.out, f"paper_radar_{cond}", data)


def run(ctx):
    d = ctx.frame("spot_decisions")
    d = d[d["experiment"] == "spots"].copy()
    d["display"] = [ctx.display(m) for m in d["model"]]
    ref = references(ctx)
    keys = ["model", "display", "condition", "subset", "category"]
    ref["subset"] = "all"
    pm = pd.DataFrame(metric_rows(pd.concat([d, ref], ignore_index=True), keys))
    ctx.write("paper_metrics", pm)
    gr = pd.DataFrame(grudge(pd.concat([d, ref], ignore_index=True), ["model", "display", "condition", "subset"]))
    ctx.write("paper_grudge", gr)
    tr = pd.DataFrame(tradeoffs(pd.concat([d, ref], ignore_index=True), ["model", "display", "condition", "subset"]))
    tr = tr[tr["subset"] == "all"]
    ctx.write("paper_tradeoffs", tr)
    ob = open_bug(d)
    ob["display"] = [ctx.display(m) for m in ob["model"]]
    ctx.write("paper_open_bug", ob)
    for cond in ("hidden", "shown"):
        sel = pm[pm["subset"] == "all"]
        plot_tradeoffs(ctx, tr, cond)
        plot_builder_finisher(ctx, sel, cond)
        plot_radar(ctx, sel, cond)
    total = ob.groupby("condition")[["valid", "bring_outs"]].sum()
    for cond, r in total.iterrows():
        p, lo, hi = rate(r["bring_outs"], r["valid"])
        ctx.note("paper/open-bug", f"bring-outs among valid safe-vs-open replies, all models ({cond}); "
                 "the original scoring counted every one as safe (open rate 0)", p, int(r["valid"]), lo, hi,
                 "share of valid replies")
    for _, r in gr[(gr["subset"] == "all") & (gr["pairs"] == "all pairs")].iterrows():
        ctx.note(f"paper/grudge/{r['condition']}", f"grudge effect (retaliation grudge - neutral): {r['display']}",
                 r["grudge_effect"], r["both_valid"], r["grudge_effect_lo"], r["grudge_effect_hi"], "difference",
                 f"change rate {r['change_rate']:.2f}")
