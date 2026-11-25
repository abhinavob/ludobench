"""
Spots: agreement with the search player (GameTheoryMultiplayerAgent, depth 2)
on decisions with at least 2 distinct moves (equivalent moves grouped).

Primary: over all such decisions, an invalid reply counted as wrong. Secondary:
among valid replies. Chance: what a uniformly random choice among the distinct
moves would score, 1 / number of distinct moves, averaged (its expectation;
the exact test against it uses the Poisson-binomial distribution of the
agreements). Ceiling: how often versions of the search player itself agree
(search_validation: depth 2 against depths 3 and 4, and against reweighted
evaluators). Reference players on the same spots: random (= chance), the
first-listed move (the lowest legal piece, as the list is shown in piece
order), the heuristic bot and the search player at depth 1. Value loss
(exploratory): the search player's value of its best move minus the value of
the chosen move, on valid replies.

Tables: spots_agreement.csv, spots_by_category.csv, spots_reference.csv,
spots_value_loss.csv, spots_frontier_130.csv, spots_ceiling.csv.
Figures: spots_agreement_shown, spots_agreement_hidden,
spots_category_heatmap_shown, spots_category_heatmap_hidden,
spots_value_loss, spots_frontier_130.
"""
import json
import os

import numpy as np
import pandas as pd

from analysis.final import style as S
from analysis.final.metrics import cluster_mean, rate

CONDITIONS = ("shown", "hidden")


def poisson_binomial_p(k, probs):
    """Two-sided exact p of k successes under independent success probabilities probs."""
    dist = np.zeros(len(probs) + 1)
    dist[0] = 1.0
    for q in probs:
        dist[1:] = dist[1:] * (1 - q) + dist[:-1] * q
        dist[0] *= 1 - q
    pk = dist[int(k)]
    return float(min(1.0, dist[dist <= pk * (1 + 1e-7)].sum()))


def choice_rows(ctx):
    d = ctx.frame("spot_decisions")
    return d[d["choice"] == True].copy()


def agreement_table(ctx, d):
    meta = ctx.datasets.set_index("dataset")
    rows = []
    for (exp, model, setting, subset, cond), x in d.groupby(
            ["experiment", "model", "setting", "subset", "condition"], dropna=False):
        setting = setting if isinstance(setting, str) else ""
        key = f"{exp}/{model}" + (f"/{setting}" if setting else "")
        m = meta.loc[key]
        n, valid, agree = len(x), int(x["valid"].sum()), int(x["agree"].sum())
        p1, lo1, hi1 = rate(agree, n)
        p2, lo2, hi2 = rate(agree, valid)
        chance = x["chance"].mean()
        rows.append({"experiment": exp, "model": model, "display": m["display"], "setting": setting,
                     "subset": subset, "condition": cond, "provider": m["provider"], "family": m["family"],
                     "size_b": m["size_b"], "reasoning_group": m["reasoning_group"], "decisions": n,
                     "valid": valid, "invalid": n - valid, "invalid_rate": 1 - valid / n, "agree": agree,
                     "agreement": p1, "agreement_lo": lo1, "agreement_hi": hi1,
                     "agreement_valid": p2, "agreement_valid_lo": lo2, "agreement_valid_hi": hi2,
                     "chance": chance, "above_chance": p1 - chance,
                     "p_vs_chance": poisson_binomial_p(agree, x["chance"].to_numpy()),
                     "cost_usd": m.get("cost_usd"), "cost_status": m.get("cost_status")})
    return pd.DataFrame(rows)


def category_table(d):
    rows = []
    for keys, x in d.groupby(["experiment", "model", "setting", "subset", "condition", "category"], dropna=False):
        p, lo, hi = rate(x["agree"].sum(), len(x))
        rows.append(dict(zip(["experiment", "model", "setting", "subset", "condition", "category"], keys),
                         decisions=len(x), agreement=p, lo=lo, hi=hi,
                         agreement_valid=x["agree"].sum() / max(1, x["valid"].sum()),
                         invalid_rate=1 - x["valid"].mean(), chance=x["chance"].mean()))
    return pd.DataFrame(rows)


def reference_table(ref):
    r = ref[ref["n_groups"] >= 2]
    rows = []
    for cat, x in [("all", r)] + list(r.groupby("category")):
        row = {"category": cat, "spots": len(x), "random (chance)": x["chance"].mean()}
        for col, name in (("lowest_agree", "first-listed (lowest piece)"), ("heuristic_agree", "heuristic bot"),
                          ("d1_agree", "search player, depth 1")):
            row[name] = x[col].astype(str).eq("True").mean()
        rows.append(row)
    return pd.DataFrame(rows)


def ceiling(ctx):
    path = os.path.join(ctx.out, "search_validation", "search_spots_agreement.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    v = pd.read_csv(path)
    v = v[(v["a"] == "depth 2") | (v["b"] == "depth 2")]
    return v[v["category"] == "all"]


def value_loss(ctx, d, ref):
    values = {r["spot_id"]: {int(k): v for k, v in json.loads(r["values"]).items()} for _, r in ref.iterrows()}
    v = d[(d["valid"] == True)].copy()
    v["loss"] = [max(values[s].values()) - values[s][int(c)] for s, c in zip(v["spot_id"], v["chosen"])]
    rows = []
    for keys, x in v.groupby(["experiment", "model", "setting", "subset", "condition"], dropna=False):
        est, lo, hi, _ = cluster_mean(x["loss"], x["spot_id"], b=2000)
        rows.append(dict(zip(["experiment", "model", "setting", "subset", "condition"], keys),
                         decisions=len(x), value_loss=est, lo=lo, hi=hi))
    r = ref[ref["n_groups"] >= 2]
    refs = []
    for col, name in (("search_d1", "search player, depth 1"), ("heuristic", "heuristic bot"),
                      ("lowest", "first-listed (lowest piece)")):
        loss = [max(values[s].values()) - values[s][int(m)] for s, m in zip(r["spot_id"], r[col])]
        refs.append({"reference": name, "spots": len(r), "value_loss": float(np.mean(loss))})
    loss = [max(values[s].values()) - np.mean(list(values[s].values())) for s in r["spot_id"]]
    refs.append({"reference": "random (expected over legal pieces)", "spots": len(r),
                 "value_loss": float(np.mean(loss))})
    return pd.DataFrame(rows), pd.DataFrame(refs)


def ceiling_band(ceil):
    """Lowest and highest agreement of the depth-2 search player with depths 3 and 4."""
    deeper = ceil["b"].isin(["depth 3", "depth 4"]) | ceil["a"].isin(["depth 3", "depth 4"])
    return tuple(ceil[deeper]["agreement"].agg(["min", "max"]))


def plot_agreement_short(ctx, t, refs, ceil):
    """One-column version of spots_agreement_shown: the top 10 and every reasoning model named, the rest as one row."""
    x = t[(t["experiment"] == "spots") & (t["condition"] == "shown") & (t["subset"] == "all")]
    a = refs[refs["category"] == "all"].iloc[0]
    S.short_dot_plot(ctx.out, "spots_agreement_shown_short", x.sort_values("agreement", ascending=False)[
        ["display", "decisions", "agreement", "agreement_lo", "agreement_hi", "agreement_valid", "chance",
         "reasoning_group"]], 10, "agreement", "agreement_lo", "agreement_hi", "agreement with the search player",
        refs=[(a["random (chance)"], "chance", "--", "center"),
              (a["first-listed (lowest piece)"], "lowest piece", ":", "center")],
        band=(*ceiling_band(ceil), "ceiling"), named_groups=("reasoning",))


def plot_agreement(ctx, t, refs, ceil, cond):
    x = t[(t["experiment"] == "spots") & (t["condition"] == cond) & (t["subset"] == "all")]
    x = x.sort_values("agreement", ascending=False)
    if x.empty:
        return
    fig, ax = S.figure(figsize=(6.6, 0.22 * len(x) + 1.6))
    colors = [S.REASONING.get(g, S.MUTED) for g in x["reasoning_group"]]
    S.dot_plot(ax, list(x["display"]), x["agreement"], x["agreement_lo"], x["agreement_hi"], colors,
               "agreement with the search player, invalid replies counted as wrong (95% Wilson interval)")
    a = refs[refs["category"] == "all"].iloc[0]
    S.reference_line(ax, x=a["random (chance)"], label="chance")
    S.reference_line(ax, x=a["first-listed (lowest piece)"], label="first-listed (lowest piece)", style=":",
                     where="bottom")
    if not ceil.empty:
        lo, hi = ceiling_band(ceil)
        ax.axvspan(lo, hi, color=S.GRID, alpha=0.8, zorder=0)
        ax.annotate("search versions", ((lo + hi) / 2, 1), xycoords=("data", "axes fraction"), xytext=(0, -3),
                    textcoords="offset points", ha="center", va="top", rotation=90, fontsize=7, color=S.INK2)
    S.pct(ax)
    ax.set_xlim(0, 1)
    for g in ("non-reasoning", "reasoning"):
        ax.plot([], [], "o", color=S.REASONING[g], label=g.replace("-", " "))
    ax.legend(loc="lower left")
    ax.set_title("Agreement with the search player on the spots")
    S.save(fig, ctx.out, f"spots_agreement_{cond}",
           x[["display", "decisions", "agreement", "agreement_lo", "agreement_hi", "agreement_valid", "chance",
              "reasoning_group"]])


def plot_heatmap(ctx, cats, t, cond):
    x = cats[(cats["experiment"] == "spots") & (cats["condition"] == cond) & (cats["subset"] == "all")]
    order = t[(t["experiment"] == "spots") & (t["condition"] == cond) & (t["subset"] == "all")].sort_values(
        "agreement", ascending=False)
    if order.empty:
        return
    piv = x.pivot_table(index="model", columns="category", values="agreement").loc[order["model"]]
    piv = piv[sorted(piv.columns)]
    fig, ax = S.figure(figsize=(0.62 * piv.shape[1] + 3.2, 0.24 * len(piv) + 1.8))
    im = ax.imshow(piv.to_numpy(), cmap=S.SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v > 0.6 else S.INK)
    ax.set_xticks(range(piv.shape[1]), [c.replace("_", " ") for c in piv.columns], rotation=40, ha="right")
    ax.set_yticks(range(len(piv)), [ctx.display(m) for m in piv.index])
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.03, format="{x:.0%}").set_label("agreement (invalid = wrong)")
    ax.set_title(f"Agreement with the search player by category, legal moves {cond} (%)")
    data = piv.reset_index().melt(id_vars="model", value_name="agreement")
    S.save(fig, ctx.out, f"spots_category_heatmap_{cond}", data)


def plot_value_loss(ctx, vl, refs):
    x = vl[(vl["experiment"] == "spots") & (vl["condition"] == "shown") & (vl["subset"] == "all")]
    x = x.assign(display=[ctx.display(m) for m in x["model"]]).sort_values("value_loss")
    fig, ax = S.figure(figsize=(6.4, 0.22 * len(x) + 1.6))
    S.dot_plot(ax, list(x["display"]), x["value_loss"], x["lo"], x["hi"], [S.SERIES[0]] * len(x),
               "mean value loss against the search player's best move (search value units; lower is better)")
    for _, r in refs.iterrows():
        if r["reference"] in ("random (expected over legal pieces)", "heuristic bot"):
            S.reference_line(ax, x=r["value_loss"], label=r["reference"].split(" (")[0])
    ax.set_title("Spots, legal moves shown: value loss (exploratory)")
    S.save(fig, ctx.out, "spots_value_loss", x)


def frontier(ctx, d):
    ids = set(d[(d["subset"] == "130")]["spot_id"])
    x = d[(d["experiment"] == "spots") & (d["condition"] == "shown") & d["spot_id"].isin(ids)]
    rows = []
    for (model, subset), g in x.groupby(["model", "subset"]):
        p, lo, hi = rate(g["agree"].sum(), len(g))
        rows.append({"model": model, "display": ctx.display(model), "subset": subset, "decisions": len(g),
                     "agreement": p, "lo": lo, "hi": hi, "agreement_valid": g["agree"].sum() / max(1, g["valid"].sum()),
                     "chance": g["chance"].mean()})
    f = pd.DataFrame(rows).sort_values("agreement", ascending=False)
    fig, ax = S.figure(figsize=(6.4, 0.22 * len(f) + 1.885))  # 0.285 in for the legend below the axes
    meta = ctx.datasets[ctx.datasets["experiment"] == "spots"].set_index("model")["reasoning_group"]
    colors = [S.REASONING.get(meta.get(m), S.MUTED) for m in f["model"]]
    markers = ["D" if s == "130" else "o" for s in f["subset"]]
    S.dot_plot(ax, list(f["display"]), f["agreement"], f["lo"], f["hi"], colors,
               "agreement with the search player on the shared 130 spots (legal moves shown)", markers)
    S.reference_line(ax, x=f["chance"].mean(), label="chance")
    S.pct(ax)
    ax.set_xlim(0, 1)
    for g in ("non-reasoning", "reasoning"):
        ax.plot([], [], "o", color=S.REASONING[g], label=g.replace("-", " "))
    ax.plot([], [], "D", color=S.MUTED, label="asked only these 130")
    fig.legend(loc="outside lower center", ncol=3)
    ax.set_title(f"The {len(ids)} spots (frontier models asked only these)")
    S.save(fig, ctx.out, "spots_frontier_130", f)
    S.bar_chart(ctx.out, "spots_frontier_130_blog", f.assign(reasoning_group=[meta.get(m) for m in f["model"]]),
                "agreement", "agreement with the search player (legal moves shown)",
                f"Agreement on the {len(ids)} spots every model answered", hatched=f["subset"] == "130",
                hatch_label=f"answered only these {len(ids)}")
    return f


def run(ctx):
    d = choice_rows(ctx)
    ref = ctx.frame("spot_reference")
    t = agreement_table(ctx, d)
    ctx.write("spots_agreement", t)
    cats = category_table(d)
    ctx.write("spots_by_category", cats)
    refs = reference_table(ref)
    ctx.write("spots_reference", refs)
    ceil = ceiling(ctx)
    ctx.write("spots_ceiling", ceil)
    vl, vrefs = value_loss(ctx, d, ref)
    ctx.write("spots_value_loss", vl)
    ctx.write("spots_value_loss_reference", vrefs)
    for cond in CONDITIONS:
        plot_agreement(ctx, t, refs, ceil, cond)
        plot_heatmap(ctx, cats, t, cond)
    plot_agreement_short(ctx, t, refs, ceil)
    plot_value_loss(ctx, vl, vrefs)
    f = frontier(ctx, d)
    ctx.write("spots_frontier_130", f)
    for _, r in t[t["experiment"] == "spots"].iterrows():
        sub = " (130 spots)" if r["subset"] == "130" else ""
        ctx.note(f"spots/{r['condition']}", f"agreement (invalid = wrong): {r['display']}{sub}", r["agreement"],
                 r["decisions"], r["agreement_lo"], r["agreement_hi"], "share of decisions",
                 f"chance {r['chance']:.3f}; exact p vs chance {r['p_vs_chance']:.2g}")
        ctx.note(f"spots/{r['condition']}", f"agreement among valid replies: {r['display']}{sub}",
                 r["agreement_valid"], r["valid"], r["agreement_valid_lo"], r["agreement_valid_hi"],
                 "share of valid replies")
    a = refs[refs["category"] == "all"].iloc[0]
    for name in ("random (chance)", "first-listed (lowest piece)", "heuristic bot", "search player, depth 1"):
        ctx.note("spots/reference", f"agreement with the depth-2 search player: {name}", a[name], a["spots"],
                 detail="same 519 spots, no interval (deterministic)")
    for _, r in ceil.iterrows():
        ctx.note("spots/ceiling", f"search versions agree: {r['a']} vs {r['b']}", r["agreement"], r["spots"],
                 r.get("lo"), r.get("hi"), "share of spots")
    for _, r in f.iterrows():
        ctx.note("spots/frontier-130", f"agreement on the shared 130 spots: {r['display']}", r["agreement"],
                 r["decisions"], r["lo"], r["hi"], "share of decisions")
