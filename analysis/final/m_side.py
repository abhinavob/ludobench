"""
Side experiments, the preregistered hypotheses, and the reasoning groups.

hidden vs shown   spots: agreement (invalid = wrong) with the list hidden and
                  shown on the same spots, paired (bootstrap over spots, exact
                  McNemar, Holm); games: "hidden" against "main" on shared
                  seeds (win rate, McNemar; invalid rate, seed bootstrap)
H1-H4             exactly as analysis/hypotheses.py (preregistered 11 Nov 2026):
                  H1 and H2 on main games (H1 only for models that do not
                  reason), H3 spots vs spots-shuffle, H4 spots vs
                  spots-outcomes and main-outcomes vs main games on shared
                  seeds; Holm across every preregistered test
thinking          the same 130 spots with reasoning off or at its lowest
                  (from the model's spots run) and at each higher setting
history           main-history against main on shared seeds
repeatability     spots-repeat against the model's spots run on the same
                  spot and condition: share of identical answers (both
                  invalid, or the same distinct move) and of identical pieces
                  named, after checking prompt hashes and Ollama digests
reasoning groups  main win rate and spot agreement of the models that reason
                  during play and those that do not (mean over models with a
                  bootstrap over models, and pooled)

Tables: side_hidden_vs_shown_spots.csv, side_hidden_vs_main_games.csv,
hypotheses.csv, side_thinking.csv, side_history.csv, side_repeatability.csv,
reasoning_groups.csv. Figures: side_hidden_vs_shown, hypotheses,
side_thinking, reasoning_groups.
"""
import json
import os

import numpy as np
import pandas as pd

from analysis import hypotheses as H
from analysis.final import style as S
from analysis.final.metrics import rate
from analysis.stats import bootstrap, holm, paired

KINDS = ("capture", "safe", "open", "home_entry", "finish")


def m_of(r):
    m = {"valid": bool(r["valid"]), "choice": bool(r["choice"]), "agree": bool(r["agree"]),
         "chance": float(r["chance"]), "first": bool(r["first"]) if pd.notna(r.get("first")) else False,
         "first_agree": bool(r["first_agree"]) if pd.notna(r.get("first_agree")) else False}
    for k in KINDS:
        m[k] = {f: bool(r[f"{k}_{f}"]) if f != "chance" else float(r[f"{k}_chance"])
                for f in ("offered", "taken", "chance", "search")}
    return m


def game_ds(g, exp, model):
    x = g[(g["experiment"] == exp) & (g["model"] == model) & (g["shown"] == True) & (g["choice"] == True)
          & (g["valid"] == True)]
    return [{"m": m_of(r), "game": f"{model}#{r['game_id']}", "dice": r["dice"], "seed": r["seed"]}
            for _, r in x.iterrows()]


def spot_ds(d, exp, model, setting=""):
    x = d[(d["experiment"] == exp) & (d["model"] == model) & (d["condition"] == "shown") & (d["choice"] == True)
          & (d["valid"] == True) & (d["setting"].fillna("") == setting)]
    return {r["spot_id"]: {"m": m_of(r), "spot_id": r["spot_id"], "chosen_index": int(r["chosen"]),
                           "shown_legal_moves": json.loads(r["shown_legal_moves"]),
                           "move_groups": json.loads(r["move_groups"]), "shown": True}
            for _, r in x.iterrows()}


def hypotheses(ctx):
    g, d = ctx.frame("game_decisions"), ctx.frame("spot_decisions")
    meta = ctx.datasets.set_index("dataset")
    rows = []
    for model in sorted(g[g["experiment"] == "main"]["model"].unique()):
        ds = game_ds(g, "main", model)
        group = meta.loc[f"main/{model}", "reasoning_group"]
        tests = (H.h1(ds) if group == "non-reasoning" else []) + H.h2(ds)
        rows += [dict(r, model=model, data="main games") for r in tests]
        if ((g["experiment"] == "main-outcomes") & (g["model"] == model)).any():
            rows.append(dict(H.h4_games(game_ds(g, "main-outcomes", model), ds), model=model,
                             data="main-outcomes vs main games"))
    for model in sorted(d[d["experiment"] == "spots"]["model"].unique()):
        base = spot_ds(d, "spots", model)
        for exp, f, label in (("spots-shuffle", H.h3, "spots vs spots-shuffle"),
                              ("spots-outcomes", H.h4_spots, "spots vs spots-outcomes")):
            other = spot_ds(d, exp, model)
            if other:
                rows.append(dict(f(base, other), model=model, data=label))
                if exp == "spots-shuffle":
                    rows.append(dict(H.equivalent_picks(other), model=model, data="spots-shuffle"))
    for r in rows:
        r["family"] = "exploratory" if r["hypothesis"] == "exploratory" else "preregistered"
    for family in ("preregistered", "exploratory"):
        sel = [r for r in rows if r["family"] == family]
        for r, p in zip(sel, holm([r["p"] if r["n"] else np.nan for r in sel])):
            r["p_holm"] = p
    for r in rows:
        r["result"] = H.result(r)
        r["display"] = ctx.display(r["model"])
    return pd.DataFrame(rows)


def hidden_vs_shown(ctx):
    d = ctx.frame("spot_decisions")
    d = d[(d["experiment"] == "spots") & (d["subset"] == "all") & (d["choice"] == True)]
    rows = []
    for model, x in d.groupby("model"):
        s = x[x["condition"] == "shown"].set_index("spot_id")
        h = x[x["condition"] == "hidden"].set_index("spot_id")
        ids = s.index.intersection(h.index)
        if not len(ids):
            continue
        r = paired(s.loc[ids, "agree"].astype(int), h.loc[ids, "agree"].astype(int))
        rows.append({"model": model, "display": ctx.display(model), "spots": len(ids), "agreement_shown": r["rate_a"],
                     "agreement_hidden": r["rate_b"], "diff": r["diff"], "lo": r["diff_lo"], "hi": r["diff_hi"],
                     "p": r["p_mcnemar"], "invalid_shown": 1 - s.loc[ids, "valid"].astype(bool).mean(),
                     "invalid_hidden": 1 - h.loc[ids, "valid"].astype(bool).mean()})
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t["p"])
    return t


def hidden_vs_main(ctx):
    gs = ctx.frame("games")
    rows = []
    for model in sorted(gs[gs["experiment"] == "hidden"]["model"].unique()):
        a = gs[(gs["experiment"] == "hidden") & (gs["model"] == model)].set_index("seed")
        b = gs[(gs["experiment"] == "main") & (gs["model"] == model)].set_index("seed")
        seeds = a.index.intersection(b.index)
        w = paired(a.loc[seeds, "won"], b.loc[seeds, "won"])
        inv = bootstrap(np.c_[a.loc[seeds, ["invalid", "calls"]].to_numpy(), b.loc[seeds, ["invalid", "calls"]].to_numpy()],
                        lambda s: s[..., 0] / s[..., 1] - s[..., 2] / s[..., 3], b=4000)
        rows.append({"model": model, "display": ctx.display(model), "seeds": len(seeds), "win_hidden": w["rate_a"],
                     "win_main": w["rate_b"], "win_diff": w["diff"], "win_lo": w["diff_lo"], "win_hi": w["diff_hi"],
                     "win_p": w["p_mcnemar"], "invalid_hidden": a.loc[seeds, "invalid"].sum() / a.loc[seeds, "calls"].sum(),
                     "invalid_main": b.loc[seeds, "invalid"].sum() / b.loc[seeds, "calls"].sum(),
                     "invalid_diff": inv[0], "invalid_lo": inv[1], "invalid_hi": inv[2], "invalid_p": inv[3]})
    t = pd.DataFrame(rows)
    if len(t):
        t["win_p_holm"], t["invalid_p_holm"] = holm(t["win_p"]), holm(t["invalid_p"])
    return t


def thinking(ctx):
    d = ctx.frame("spot_decisions")
    meta = ctx.datasets.set_index("dataset")
    rows = []
    think = d[d["experiment"] == "spots-think"]
    for model, x in think.groupby("model"):
        ids = set(x["spot_id"])
        base = d[(d["experiment"] == "spots") & (d["model"] == model) & (d["condition"] == "shown")
                 & d["spot_id"].isin(ids)].assign(setting="spots run (off or lowest)")
        curve = pd.concat([base, x])
        off = base.set_index("spot_id")
        for setting, y in curve.groupby("setting", sort=False):
            c = y[y["choice"] == True]
            p, lo, hi = rate(c["agree"].sum(), len(c))
            yy = y.set_index("spot_id")
            both = yy.index.intersection(off.index)
            pr = paired(yy.loc[both, "agree"].astype(int), off.loc[both, "agree"].astype(int)) \
                if setting != "spots run (off or lowest)" else None
            m = meta.loc[f"spots-think/{model}/{setting}"] if f"spots-think/{model}/{setting}" in meta.index else \
                meta.loc[f"spots/{model}"]
            rows.append({"model": model, "display": ctx.display(model), "setting": setting, "spots": len(c),
                         "agreement": p, "lo": lo, "hi": hi,
                         "agreement_valid": c["agree"].sum() / max(1, c["valid"].sum()),
                         "invalid_rate": 1 - y["valid"].astype(bool).mean(),
                         "output_tokens": y["completion_tokens"].mean(), "thinking_tokens": y["thinking_tokens"].mean(),
                         "seconds": y["latency_ms"].mean() / 1000, "reasoning_share": y["reasoned"].astype(bool).mean(),
                         "sent": m.get("sent_reasoning"),
                         "diff_vs_spots_run": pr["diff"] if pr else np.nan, "diff_lo": pr["diff_lo"] if pr else np.nan,
                         "diff_hi": pr["diff_hi"] if pr else np.nan, "p_mcnemar": pr["p_mcnemar"] if pr else np.nan})
    t = pd.DataFrame(rows)
    order = {"spots run (off or lowest)": 0, "think": 1, "effort-low": 1, "effort-medium": 2, "effort-high": 3}
    t["step"] = t["setting"].map(order)
    return t.sort_values(["model", "step"])


def history(ctx):
    gs, gd = ctx.frame("games"), ctx.frame("game_decisions")
    rows = []
    for model in sorted(gs[gs["experiment"] == "main-history"]["model"].unique()):
        a = gs[(gs["experiment"] == "main-history") & (gs["model"] == model)].set_index("seed")
        b = gs[(gs["experiment"] == "main") & (gs["model"] == model)].set_index("seed")
        seeds = a.index.intersection(b.index)
        w = paired(a.loc[seeds, "won"], b.loc[seeds, "won"])
        agree = {}
        for exp in ("main-history", "main"):
            x = gd[(gd["experiment"] == exp) & (gd["model"] == model) & (gd["choice"] == True) & gd["seed"].isin(seeds)]
            agree[exp] = x.groupby("seed")["agree"].agg(["sum", "count"])
        s = agree["main-history"].join(agree["main"], lsuffix="_h", rsuffix="_m", how="inner").to_numpy(float)
        est, lo, hi, p = bootstrap(s, lambda t: t[..., 0] / t[..., 1] - t[..., 2] / t[..., 3], b=4000)
        rows.append({"model": model, "display": ctx.display(model), "seeds": len(seeds), "win_history": w["rate_a"],
                     "win_main": w["rate_b"], "win_diff": w["diff"], "win_lo": w["diff_lo"], "win_hi": w["diff_hi"],
                     "win_p": w["p_mcnemar"], "agreement_history": s[:, 0].sum() / s[:, 1].sum(),
                     "agreement_main": s[:, 2].sum() / s[:, 3].sum(), "agreement_diff": est, "agreement_lo": lo,
                     "agreement_hi": hi, "agreement_p": p})
    return pd.DataFrame(rows)


def run_meta(ctx, key):
    """run.json of a run key (root:experiment/name), read from the roots."""
    root, rel = key.split(":", 1)
    with open(os.path.join(ctx.roots[root], rel, "run.json"), encoding="utf-8") as f:
        return json.load(f)


def same_answer(a, b):
    """The same answer: both invalid, or both valid with moves of one distinct move (a's grouping, same board)."""
    if bool(a["valid"]) != bool(b["valid"]):
        return False
    if not a["valid"]:
        return True
    return any(int(a["chosen"]) in g and int(b["chosen"]) in g for g in json.loads(a["move_groups"]))


def repeatability(ctx):
    d = ctx.frame("spot_decisions")
    ds = ctx.datasets.set_index("dataset")
    rows = []
    for model, r in d[d["experiment"] == "spots-repeat"].groupby("model"):
        o = d[(d["experiment"] == "spots") & (d["model"] == model)].set_index(["spot_id", "condition"])
        r = r.set_index(["spot_id", "condition"])
        both = r.index.intersection(o.index)
        same_prompt = (r.loc[both, "prompt_hash"] == o.loc[both, "prompt_hash"]).mean()
        a, b = r.loc[both], o.loc[both]
        k = sum(same_answer(x, y) for (_, x), (_, y) in zip(a.iterrows(), b.iterrows()))
        same_piece = (a["parsed_index"].fillna(-1) == b["parsed_index"].fillna(-1)).mean()
        p, lo, hi = rate(k, len(both))
        dig = []
        for key in (f"spots-repeat/{model}", f"spots/{model}"):
            run = ds.loc[key, "runs"].split(";")[0]
            dig.append(run_meta(ctx, run)["players"]["0"].get("digest"))
        rows.append({"model": model, "display": ctx.display(model), "prompts": len(both),
                     "same_prompt_hash": same_prompt, "digest_checked": dig[0] is not None,
                     "same_digest": dig[0] == dig[1] if dig[0] else None, "same_move": p, "lo": lo, "hi": hi,
                     "same_piece": same_piece,
                     "agreement_original": o.loc[both, "agree"].astype(bool).mean(),
                     "agreement_repeat": r.loc[both, "agree"].astype(bool).mean()})
    return pd.DataFrame(rows)


def reasoning_groups(ctx):
    gs = pd.read_csv(os.path.join(ctx.tables, "games_summary.csv"))
    sp = pd.read_csv(os.path.join(ctx.tables, "spots_agreement.csv"))
    rows = []
    for what, t, col, n in (("main games: win rate", gs[gs["experiment"] == "main"], "win_rate", "games"),
                            ("spots shown: agreement", sp[(sp["experiment"] == "spots") & (sp["condition"] == "shown")
                                                          & (sp["subset"] == "all")], "agreement", "decisions"),
                            ("spots hidden: agreement", sp[(sp["experiment"] == "spots") & (sp["condition"] == "hidden")
                                                           & (sp["subset"] == "all")], "agreement", "decisions")):
        for g, x in t.groupby("reasoning_group"):
            v = x[col].to_numpy()
            est, lo, hi, _ = bootstrap(np.c_[v, np.ones(len(v))], lambda s: s[..., 0] / s[..., 1], b=4000)
            rows.append({"measure": what, "group": g, "models": len(x), "mean_over_models": est, "lo": lo, "hi": hi,
                         "pooled": (x[col] * x[n]).sum() / x[n].sum(), "models_list": "; ".join(x["display"])})
    return pd.DataFrame(rows)


def plot_hidden_shown(ctx, t):
    t = t.sort_values("diff")
    fig, ax = S.figure(figsize=(6.4, 0.22 * len(t) + 1.6))
    S.dot_plot(ax, list(t["display"]), t["diff"], t["lo"], t["hi"],
               [S.SERIES[1] if p < 0.05 else S.SERIES[0] for p in t["p_holm"]],
               "agreement with the list shown minus hidden (same spots; 95% bootstrap interval)")
    S.reference_line(ax, x=0)
    ax.plot([], [], "o", color=S.SERIES[1], label="Holm p < 0.05")
    ax.plot([], [], "o", color=S.SERIES[0], label="not significant")
    ax.legend(loc="lower right")
    ax.set_title("Showing the legal moves: change in agreement with the search player")
    S.save(fig, ctx.out, "side_hidden_vs_shown", t)


def plot_hypotheses(ctx, t):
    pre = t[(t["family"] == "preregistered") & (t["n"] > 0)]
    tests = list(dict.fromkeys(zip(pre["hypothesis"], pre["test"], pre["data"])))
    fig, axes = S.figure(ncols=len(tests), figsize=(2.9 * len(tests), 0.22 * pre["model"].nunique() + 1.8),
                         squeeze=False)
    for ax, (h, test, data) in zip(axes[0], tests):
        x = pre[(pre["hypothesis"] == h) & (pre["test"] == test) & (pre["data"] == data)].sort_values("display")
        S.dot_plot(ax, list(x["display"]), x["diff"], x["lo"], x["hi"],
                   [S.SERIES[1] if p < 0.05 else S.SERIES[0] for p in x["p_holm"]], "difference")
        S.reference_line(ax, x=0)
        ax.set_title(f"{h}: {test}\n({data})", fontsize=7.5)
    fig.suptitle("Preregistered hypotheses (orange: Holm p < 0.05 across all tests)", fontsize=10)
    S.save(fig, ctx.out, "hypotheses", pre)


def setting_label(setting, sent):
    """Tick label of a reasoning setting; the spots run is named by what it sent: "off", or its effort."""
    if setting != "spots run (off or lowest)":
        return setting.replace("effort-", "")
    value = next(iter(json.loads(sent).values()))
    return "off" if value in ("none", False) else value


def plot_thinking(ctx, t, blog=False):
    """Agreement by reasoning setting; blog: the web version (large type, no error bars, PNG only)."""
    models = list(dict.fromkeys(t["model"]))
    with S.plt.rc_context(S.BLOG if blog else {}):
        fig, axes = S.figure(ncols=len(models), figsize=((3.4 if blog else 2.4) * len(models), 4.6 if blog else 2.8),
                             sharey=True, squeeze=False)
        for ax, m in zip(axes[0], models):
            x = t[t["model"] == m]
            if blog:
                ax.plot(range(len(x)), x["agreement"], "o-", color=S.SERIES[0], markersize=9, linewidth=3)
            else:
                ax.errorbar(range(len(x)), x["agreement"], yerr=[x["agreement"] - x["lo"], x["hi"] - x["agreement"]],
                            fmt="o-", color=S.SERIES[0], capsize=2, markersize=5)
            ax.set_xticks(range(len(x)), [setting_label(s, sent) for s, sent in zip(x["setting"], x["sent"])],
                          rotation=20)
            ax.set_title(x["display"].iloc[0], fontsize=16 if blog else 8)
            ax.set_ylim(0, 1)
            S.pct(ax, "y")
        axes[0][0].set_ylabel("agreement (130 spots, list shown)")
        fig.suptitle("Reasoning effort and agreement with the search player", fontsize=18 if blog else 10)
        S.save(fig, ctx.out, "side_thinking_blog" if blog else "side_thinking", t,
               exts=("png",) if blog else ("png", "pdf"))


def plot_groups(ctx, t):
    fig, ax = S.figure(figsize=(6.4, 2.6))
    labels = []
    for i, (_, r) in enumerate(t.iterrows()):
        c = S.REASONING.get(r["group"], S.MUTED)
        ax.plot([r["lo"], r["hi"]], [i, i], color=c, linewidth=1.5, alpha=0.6)
        ax.plot(r["mean_over_models"], i, "o", color=c)
        labels.append(f"{r['measure']}, {r['group']} ({r['models']} models)")
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    S.pct(ax)
    ax.set_xlabel("mean over models (95% bootstrap interval over models)")
    ax.set_title("Models that reason during play and those that do not")
    S.save(fig, ctx.out, "reasoning_groups", t)


def run(ctx):
    hs = hidden_vs_shown(ctx)
    ctx.write("side_hidden_vs_shown_spots", hs)
    plot_hidden_shown(ctx, hs)
    hm = hidden_vs_main(ctx)
    ctx.write("side_hidden_vs_main_games", hm)
    hy = hypotheses(ctx)
    ctx.write("hypotheses", hy)
    plot_hypotheses(ctx, hy)
    th = thinking(ctx)
    ctx.write("side_thinking", th)
    plot_thinking(ctx, th)
    plot_thinking(ctx, th, blog=True)
    hi = history(ctx)
    ctx.write("side_history", hi)
    rp = repeatability(ctx)
    ctx.write("side_repeatability", rp)
    rg = reasoning_groups(ctx)
    ctx.write("reasoning_groups", rg)
    plot_groups(ctx, rg)
    for _, r in hs.iterrows():
        ctx.note("side/hidden-vs-shown spots", f"agreement shown - hidden: {r['display']}", r["diff"], r["spots"],
                 r["lo"], r["hi"], "difference", f"Holm p {r['p_holm']:.2g}")
    for _, r in hm.iterrows():
        ctx.note("side/hidden-vs-main games", f"win rate hidden - main: {r['display']}", r["win_diff"], r["seeds"],
                 r["win_lo"], r["win_hi"], "difference", f"Holm p {r['win_p_holm']:.2g}")
        ctx.note("side/hidden-vs-main games", f"invalid rate hidden - main: {r['display']}", r["invalid_diff"],
                 r["seeds"], r["invalid_lo"], r["invalid_hi"], "difference", f"Holm p {r['invalid_p_holm']:.2g}")
    for _, r in hy.iterrows():
        ctx.note(f"hypotheses/{r['hypothesis']}", f"{r['test']} ({r['data']}): {r['display']}", r["diff"], r["n"],
                 r["lo"], r["hi"], "difference", f"{r['result']}; Holm p {r['p_holm']:.2g}")
    for _, r in th.iterrows():
        ctx.note("side/thinking", f"agreement at {r['setting']}: {r['display']}", r["agreement"], r["spots"], r["lo"],
                 r["hi"], "share", f"output tokens {r['output_tokens']:.0f}")
    for _, r in hi.iterrows():
        ctx.note("side/history", f"win rate main-history - main: {r['display']}", r["win_diff"], r["seeds"],
                 r["win_lo"], r["win_hi"], "difference")
    for _, r in rp.iterrows():
        ctx.note("side/repeatability", f"same move when asked again: {r['display']}", r["same_move"], r["prompts"],
                 r["lo"], r["hi"], "share", f"prompt hashes equal {r['same_prompt_hash']:.0%}; digest same {r['same_digest']}")
    for _, r in rg.iterrows():
        ctx.note("reasoning-groups", f"{r['measure']}, {r['group']} ({r['models']} models)", r["mean_over_models"],
                 r["models"], r["lo"], r["hi"], "mean over models", f"pooled {r['pooled']:.3f}")
