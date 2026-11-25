"""
Reasons (the "reason" of valid replies on decisions with a real choice), with
the theme and claim rules of analysis/reasons.py, in the spots (both
conditions) and the main games.

  themes              share of reasons mentioning each theme
  false claims        among reasons with at least one checked claim about the
                      chosen move, the share with a false one (Wilson); per
                      claim type, checked and false
  correct-reason agreement
                      agreement with the search player when every checked
                      claim is true and when one is false (bootstrap over
                      spots / games for the difference)
  matches the move    among reasons naming one of the player's pieces, the
                      share naming the chosen one; reasons naming only others
  same move, different reasons
                      spots where two models chose the same distinct move: the
                      share of model pairs whose reasons share no theme, and
                      the mean Jaccard similarity of their theme sets
  similar win rates   pairs of models whose main win rates differ by at most 3
                      points but whose false-claim rates in the main games
                      differ by 20 points or more, side by side (e.g. GLM 5.3
                      Flash and Lightning)
By category (spots) and reasoning group.

Tables: reasons_themes.csv, reasons_claims.csv, reasons_claim_types.csv,
reasons_correct_agreement.csv, reasons_by_category.csv, reasons_by_group.csv,
reasons_same_move.csv, reasons_similar_pairs.csv, reasons_sample.csv.
Figures: reasons_themes, reasons_false_claims.
"""
import itertools

import numpy as np
import pandas as pd

from analysis.final import style as S
from analysis.final.metrics import rate
from analysis.stats import bootstrap
from analysis.final.tidy import CLAIM_NAMES, THEME_NAMES


def reasons(ctx):
    d = ctx.frame("spot_decisions")
    d = d[(d["experiment"] == "spots") & (d["has_reason"] == True)].copy()
    d["source"] = "spots, " + d["condition"]
    d["cluster"] = d["spot_id"]
    g = ctx.frame("game_decisions")
    g = g[(g["experiment"] == "main") & (g["has_reason"] == True)].copy()
    g["source"] = "main games"
    g["cluster"] = g["model"] + "#" + g["game_id"].astype(str)
    cols = ["model", "source", "cluster", "agree", "any_claim", "any_false", "names_chosen", "names_only_other",
            *[f"theme_{t}" for t in THEME_NAMES], *[f"claim_{c}" for c in CLAIM_NAMES]]
    out = pd.concat([d[cols + ["category", "spot_id", "chosen", "move_groups", "condition", "reason"]], g[cols]],
                    ignore_index=True)
    groups = ctx.datasets[ctx.datasets["experiment"].isin(["spots", "main"])].drop_duplicates("model")
    out["reasoning_group"] = out["model"].map(dict(zip(groups["model"], groups["reasoning_group"])))
    out["display"] = [ctx.display(m) for m in out["model"]]
    return out


def theme_table(r, keys):
    rows = []
    for k, x in r.groupby(keys, dropna=False):
        row = dict(zip(keys, k if isinstance(k, tuple) else (k,)), reasons=len(x))
        for t in THEME_NAMES:
            row[t] = x[f"theme_{t}"].astype(bool).mean()
        claimed = x[x["any_claim"] == True]
        p, lo, hi = rate(claimed["any_false"].sum(), len(claimed))
        row.update(with_claim=len(claimed), false_claim_rate=p, false_lo=lo, false_hi=hi)
        named = x[x["names_chosen"].notna()]
        row.update(naming_a_piece=len(named), names_chosen=named["names_chosen"].astype(bool).mean() if len(named) else np.nan,
                   names_only_other=x["names_only_other"].astype(bool).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def claim_types(r):
    rows = []
    for (model, source), x in r.groupby(["model", "source"]):
        for c in CLAIM_NAMES:
            v = x[f"claim_{c}"].dropna()
            v = v.map({True: True, False: False, "True": True, "False": False})
            if len(v):
                p, lo, hi = rate((v == False).sum(), len(v))
                rows.append({"model": model, "display": x["display"].iloc[0], "source": source, "claim": c,
                             "checked": len(v), "false": int((v == False).sum()), "false_rate": p, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def correct_agreement(r):
    rows = []
    for (model, source), x in r[r["any_claim"] == True].groupby(["model", "source"]):
        x = x.assign(false=x["any_false"].astype(bool), agree=x["agree"].astype(bool))
        ok, bad = x[~x["false"]], x[x["false"]]
        if len(ok) < 5 or len(bad) < 5:
            continue
        a, b = ok["agree"].mean(), bad["agree"].mean()
        sums = x.groupby("cluster").apply(lambda c: pd.Series({"ok_agree": c.loc[~c["false"], "agree"].sum(),
                                                               "ok": (~c["false"]).sum(),
                                                               "bad_agree": c.loc[c["false"], "agree"].sum(),
                                                               "bad": c["false"].sum()}), include_groups=False)
        est, lo, hi, p = bootstrap(sums.to_numpy(float), lambda t: t[..., 0] / t[..., 1] - t[..., 2] / t[..., 3], b=2000)
        rows.append({"model": model, "display": x["display"].iloc[0], "source": source, "true_claims": len(ok),
                     "false_claims": len(bad), "agreement_true": a, "agreement_false": b, "difference": est,
                     "lo": lo, "hi": hi, "p": p})
    return pd.DataFrame(rows)


def same_move(r):
    s = r[r["source"] == "spots, shown"].copy()
    s["themes"] = [frozenset(t for t in THEME_NAMES if row[f"theme_{t}"]) for _, row in s.iterrows()]
    s["group"] = [next(i for i, g in enumerate(eval(mg)) if int(c) in g) for mg, c in zip(s["move_groups"], s["chosen"])]
    pairs, jac, disjoint = 0, [], 0
    for _, x in s.groupby(["spot_id", "group"]):
        th = list(x["themes"])
        for a, b in itertools.combinations(th, 2):
            pairs += 1
            u = a | b
            j = len(a & b) / len(u) if u else 1.0
            jac.append(j)
            disjoint += not (a & b) and bool(u)
    return pd.DataFrame([{"model_pairs_same_move": pairs, "mean_jaccard": float(np.mean(jac)) if jac else np.nan,
                          "share_no_shared_theme": disjoint / pairs if pairs else np.nan,
                          "share_identical_themes": float(np.mean([j == 1 for j in jac])) if jac else np.nan}])


def similar_pairs(ctx, r):
    gs = ctx.frame("games")
    main = gs[gs["experiment"] == "main"].groupby("model")["won"].mean()
    th = theme_table(r[r["source"] == "main games"], ["model"]).set_index("model")
    dec = ctx.frame("game_decisions")
    dec = dec[(dec["experiment"] == "main") & (dec["replied"] == True)]
    tok = dec.groupby("model").agg(output_tokens=("completion_tokens", "mean"), thinking=("thinking_tokens", "mean"),
                                   seconds=("latency_ms", lambda v: v.mean() / 1000),
                                   reasoning_share=("reasoned", lambda v: v.astype(bool).mean()))
    rows = []
    for a, b in itertools.combinations(main.index, 2):
        if (abs(main[a] - main[b]) <= 0.03 and a in th.index and b in th.index
                and abs(th.loc[a, "false_claim_rate"] - th.loc[b, "false_claim_rate"]) >= 0.2):
            for m in (a, b):
                rows.append({"pair": f"{ctx.display(a)} / {ctx.display(b)}", "model": ctx.display(m),
                             "win_rate": main[m], **th.loc[m][["reasons", "false_claim_rate", *THEME_NAMES]].to_dict(),
                             **(tok.loc[m].to_dict() if m in tok.index else {})})
    return pd.DataFrame(rows)


def plot_themes(ctx, t):
    x = t[t["source"] == "spots, shown"].sort_values("false_claim_rate")
    piv = x.set_index("display")[list(THEME_NAMES)]
    fig, ax = S.figure(figsize=(0.75 * len(THEME_NAMES) + 3, 0.22 * len(piv) + 1.6))
    im = ax.imshow(piv.to_numpy(float), cmap=S.SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=6.5, color="white" if v > 0.6 else S.INK)
    ax.set_xticks(range(piv.shape[1]), [c.replace("_", " ") for c in piv.columns])
    ax.set_yticks(range(len(piv)), piv.index)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.03, format="{x:.0%}").set_label("share of reasons")
    ax.set_title("What reasons mention (spots, legal moves shown), %")
    S.save(fig, ctx.out, "reasons_themes", piv.reset_index())


def plot_false(ctx, t):
    x = t[(t["source"] == "spots, shown") & (t["with_claim"] >= 20)].sort_values("false_claim_rate")
    fig, ax = S.figure(figsize=(6.4, 0.22 * len(x) + 1.885))  # 0.285 in for the legend below the axes
    colors = [S.REASONING.get(g, S.MUTED) for g in x["reasoning_group"]]
    S.dot_plot(ax, list(x["display"]), x["false_claim_rate"], x["false_lo"], x["false_hi"], colors,
               "share of reasons with a false claim about the chosen move (95% Wilson interval)")
    S.pct(ax)
    for g in ("non-reasoning", "reasoning"):
        ax.plot([], [], "o", color=S.REASONING[g], label=g.replace("-", " "))
    fig.legend(loc="outside lower center", ncol=2)
    ax.set_title("False claims in reasons (spots, legal moves shown; models with 20+ checked reasons)")
    S.save(fig, ctx.out, "reasons_false_claims", x[["display", "with_claim", "false_claim_rate", "false_lo",
                                                    "false_hi", "reasoning_group"]])


def run(ctx):
    r = reasons(ctx)
    t = theme_table(r, ["model", "display", "source", "reasoning_group"])
    ctx.write("reasons_themes", t)
    ctx.write("reasons_claim_types", claim_types(r))
    ca = correct_agreement(r)
    ctx.write("reasons_correct_agreement", ca)
    ctx.write("reasons_by_category", theme_table(r[r["source"] != "main games"], ["source", "category"]))
    bg = theme_table(r, ["source", "reasoning_group"])
    ctx.write("reasons_by_group", bg)
    sm = same_move(r)
    ctx.write("reasons_same_move", sm)
    sp = similar_pairs(ctx, r)
    ctx.write("reasons_similar_pairs", sp)
    sample = r[r["source"] == "spots, shown"].sample(n=min(60, len(r)), random_state=0)
    ctx.write("reasons_sample", sample[["display", "spot_id", "category", "chosen", "reason", "agree", "any_false",
                                        *[f"theme_{x}" for x in THEME_NAMES], *[f"claim_{c}" for c in CLAIM_NAMES]]])
    plot_themes(ctx, t)
    t["reasoning_group"] = t["reasoning_group"].fillna("")
    plot_false(ctx, t)
    for _, x in t.iterrows():
        if x["with_claim"]:
            ctx.note(f"reasons/{x['source']}", f"false-claim rate: {x['display']}", x["false_claim_rate"],
                     x["with_claim"], x["false_lo"], x["false_hi"], "share of reasons with a checked claim")
    for _, x in bg.iterrows():
        if x["with_claim"]:
            ctx.note(f"reasons/{x['source']}", f"false-claim rate, {x['reasoning_group']} models (pooled)",
                     x["false_claim_rate"], x["with_claim"], x["false_lo"], x["false_hi"], "share")
    for _, x in ca.iterrows():
        ctx.note(f"reasons/{x['source']}", f"agreement when claims true minus when false: {x['display']}",
                 x["difference"], x["true_claims"] + x["false_claims"], x["lo"], x["hi"], "difference")
    s = sm.iloc[0]
    ctx.note("reasons/same-move", "model pairs choosing the same move whose reasons share no theme",
             s["share_no_shared_theme"], s["model_pairs_same_move"], unit="share of pairs",
             detail=f"mean Jaccard {s['mean_jaccard']:.2f}")
