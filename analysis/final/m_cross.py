"""
Cross-cutting comparisons, over models.

  spot agreement vs win rate   models with a spots run (legal moves shown, all
                               520 spots) and main games: Pearson and Spearman
                               correlations with bootstrap intervals over models
  quality per dollar           agreement against the spots run's cost per 1,000
                               prompts, and main win rate against cost per game;
                               local and free models at $0; unknown costs left out
  size and family              agreement and win rate against the size stated in
                               the model's name, by family (e.g. the Qwen3.5 curve)

Tables: cross_models.csv, cross_correlation.csv. Figures: cross_spots_vs_games
(and _short, one column wide), cross_cost, cross_size (and size_blog,
size_families_blog).
"""
import numpy as np
import pandas as pd

from analysis.final import style as S


def models_table(ctx):
    import os
    gs = pd.read_csv(os.path.join(ctx.tables, "games_summary.csv"))
    sp = pd.read_csv(os.path.join(ctx.tables, "spots_agreement.csv"))
    main = gs[gs["experiment"] == "main"].set_index("model")
    shown = sp[(sp["experiment"] == "spots") & (sp["condition"] == "shown") & (sp["subset"] == "all")].set_index("model")
    hidden = sp[(sp["experiment"] == "spots") & (sp["condition"] == "hidden") & (sp["subset"] == "all")].set_index("model")
    ds = ctx.datasets
    spot_cost = ds[(ds["experiment"] == "spots") & (ds["subset"] == "all")].set_index("model")
    models = sorted(set(main.index) | set(shown.index))
    info = ds.drop_duplicates("model").set_index("model")
    rows = []
    for m in models:
        r = {"model": m, "display": info.loc[m, "display"], "family": info.loc[m, "family"],
             "provider": info.loc[m, "provider"], "size_b": info.loc[m, "size_b"],
             "win_rate": main["win_rate"].get(m), "games": main["games"].get(m),
             "agreement_shown": shown["agreement"].get(m), "agreement_hidden": hidden["agreement"].get(m),
             "decisions_shown": shown["decisions"].get(m),
             "group_games": main["reasoning_group"].get(m), "group_spots": shown["reasoning_group"].get(m),
             "cost_per_game_usd": main["cost_per_game_usd"].get(m) if "cost_per_game_usd" in main else np.nan}
        if m in spot_cost.index and pd.notna(spot_cost.loc[m, "cost_usd"]):
            r["cost_per_1000_prompts_usd"] = spot_cost.loc[m, "cost_usd"] / spot_cost.loc[m, "records"] * 1000
        rows.append(r)
    return pd.DataFrame(rows)


def correlation(x, y, b=4000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rank = lambda v: pd.Series(v).rank().to_numpy()
    stats = {"pearson": lambda a, c: np.corrcoef(a, c)[0, 1], "spearman": lambda a, c: np.corrcoef(rank(a), rank(c))[0, 1]}
    rng = np.random.default_rng(seed)
    out = {}
    for name, f in stats.items():
        boot = []
        for _ in range(b):
            i = rng.integers(0, len(x), len(x))
            if len(set(i)) > 2:
                boot.append(f(x[i], y[i]))
        boot = np.array(boot)
        boot = boot[~np.isnan(boot)]
        out[name] = (f(x, y), *np.percentile(boot, [2.5, 97.5]))
    return out


def plot_spots_vs_games(ctx, t, corr):
    x = t.dropna(subset=["win_rate", "agreement_shown"])
    fig, ax = S.figure(figsize=(6.2, 4.8))
    for _, r in x.iterrows():
        c = S.REASONING.get(r["group_games"], S.MUTED)
        ax.plot(r["agreement_shown"], r["win_rate"], "o", color=c, markeredgecolor="white", markeredgewidth=0.8)
    S.note_left_out(fig, S.label_points(ax, x["agreement_shown"], x["win_rate"], x["display"],
                                        first={"Nova 2 Lite": 0.03, "Qwen3.5 27B": 0.03, "Llama 3.1 8B": -0.03,
                                               "Qwen3.5 0.8B": -0.03, "Gemma 4 E4B": 0.03,
                                               "Nemotron 3.5 Super 120B A12B": -0.03}))
    S.pct(ax, "x")
    S.pct(ax, "y")
    p, s = corr["pearson"], corr["spearman"]
    ax.set_xlabel("spot agreement with the search player (legal moves shown)")
    ax.set_ylabel("win rate against the heuristic bot (main games)")
    ax.set_title(f"Spots and games ({len(x)} models): Pearson r {p[0]:.2f} [{p[1]:.2f}, {p[2]:.2f}], "
                 f"Spearman {s[0]:.2f} [{s[1]:.2f}, {s[2]:.2f}]", fontsize=8.5)
    for g in ("non-reasoning", "reasoning"):
        ax.plot([], [], "o", color=S.REASONING[g], label=g.replace("-", " "))
    ax.legend(loc="upper left")
    S.save(fig, ctx.out, "cross_spots_vs_games", x)


# Label offsets (points) and alignment for the one-column scatter, where the default (right of the point) collides.
SHORT_PLACE = {"gpt-oss-120b": (-4, 0, "right"), "GLM 5.3 Flash": (-3, -4, "right"), "Nova 2 Lite": (-4, 0, "right"),
               "DeepSeek V4.1 Flash": (-3, 3, "right"), "Gemma 4 12B": (-3, -3, "right"),
               "Qwen3.5 0.8B": (-4, 0, "right"), "Qwen3.5 4B": (0, 5, "center"), "Qwen2.5 7B": (0, -5, "center"),
               "Nemotron 3.5 Super": (0, -5, "center"), "Qwen3.5 27B": (-4, 0, "right"),
               "Llama 3.1 8B": (-3, -4, "right"), "Mistral Large 3": (3, 6, "left"), "Gemma 4 E4B": (0, 5, "center"),
               "OLMo 3 7B": (3, 5, "left"), "Gemma 3 12B": (0, -6, "center"), "Qwen3.5 9B": (-1, -5, "right"),
               "Granite 4.2 8B": (-4, 0, "right")}


def plot_spots_vs_games_short(ctx, t, corr, place=SHORT_PLACE):
    """The same scatter at one-column width (3 in), every model labelled with its short name."""
    x = t.dropna(subset=["win_rate", "agreement_shown"])
    x = x.assign(label=[S.SHORT_NAMES.get(d, d) for d in x["display"]])
    with S.plt.rc_context(S.COLUMN):
        fig, ax = S.figure(figsize=(3.0, 3.4))
        for _, r in x.iterrows():
            ax.plot(r["agreement_shown"], r["win_rate"], "o", color=S.REASONING.get(r["group_games"], S.MUTED),
                    markersize=3.5, markeredgecolor="white", markeredgewidth=0.5)
            dx, dy, ha = place.get(r["label"], (4, 0, "left"))
            ax.annotate(r["label"], (r["agreement_shown"], r["win_rate"]), xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va="center", fontsize=5, color=S.INK2)
        ax.margins(x=0.08)
        S.pct(ax, "x")
        S.pct(ax, "y")
        ax.set_xlabel("spot agreement with the search player")
        ax.set_ylabel("win rate against the heuristic bot")
        p, s = corr["pearson"], corr["spearman"]
        ax.text(0.02, 0.98, f"{len(x)} models\nPearson r {p[0]:.2f} [{p[1]:.2f}, {p[2]:.2f}]\n"
                f"Spearman {s[0]:.2f} [{s[1]:.2f}, {s[2]:.2f}]", transform=ax.transAxes, va="top", fontsize=6,
                color=S.INK2)
        for g in ("non-reasoning", "reasoning"):
            ax.plot([], [], "o", markersize=3.5, color=S.REASONING[g], label=g.replace("-", " "))
        ax.legend(loc="upper left", bbox_to_anchor=(0, 0.84), fontsize=6, handletextpad=0.1, borderpad=0.2)
        S.save(fig, ctx.out, "cross_spots_vs_games_short", x, pad=0.02)


def plot_cost(ctx, t):
    fig, axes = S.figure(ncols=2, figsize=(11, 4.6))
    for ax, (cost, q, xl, yl) in zip(axes, (
            ("cost_per_1000_prompts_usd", "agreement_shown", "US$ per 1,000 spot prompts (symlog; 0 = free or local)",
             "spot agreement (shown)"),
            ("cost_per_game_usd", "win_rate", "US$ per game (symlog; 0 = free or local)", "win rate (main games)"))):
        x = t.dropna(subset=[cost, q])
        for _, r in x.iterrows():
            ax.plot(r[cost], r[q], "o", color=S.SERIES[0] if r[cost] > 0 else S.MUTED, markeredgecolor="white")
        ax.set_xscale("symlog", linthresh=0.01)
        S.note_left_out(fig, S.label_points(ax, x[cost], x[q], x["display"]))
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        S.pct(ax, "y")
    fig.suptitle("Quality and cost (OpenRouter: reported; Bedrock: list price; unknown prices left out)", fontsize=10)
    S.save(fig, ctx.out, "cross_cost", t)


def plot_size(ctx, t):
    x = t.dropna(subset=["size_b"])
    fams = [f for f, g in x.groupby("family") if len(g) >= 2]
    fig, axes = S.figure(ncols=2, figsize=(11, 4.2))
    for ax, q, yl in zip(axes, ("agreement_shown", "win_rate"), ("spot agreement (shown)", "win rate (main games)")):
        for i, f in enumerate(sorted(fams, key=lambda f: -len(x[x["family"] == f]))):
            g = x[x["family"] == f].dropna(subset=[q]).sort_values("size_b")
            if len(g) < 2:
                continue
            ax.plot(g["size_b"], g[q], "o-", color=S.SERIES[i % 4], label=f, linewidth=1.5)
        other = x[~x["family"].isin(fams)].dropna(subset=[q])
        ax.plot(other["size_b"], other[q], "o", color=S.MUTED, label="other families")
        ax.set_xscale("log")
        lab = x.dropna(subset=[q])
        S.note_left_out(fig, S.label_points(ax, lab["size_b"], lab[q], lab["display"], fontsize=5.5))
        ax.set_xlabel("parameters (billions, from the model name; log scale)")
        ax.set_ylabel(yl)
        S.pct(ax, "y")
        ax.legend(fontsize=7)
    fig.suptitle("Size and family", fontsize=10)
    S.save(fig, ctx.out, "cross_size", x)


# Labelled points of size_blog: label offset (points) and alignment.
SIZE_BLOG_LABELS = {"Qwen3.5 0.8B": (0, 18, "center"), "Qwen3.5 122B": (10, 8, "left"),
                    "Gemma 4 12B": (-10, 0, "right"), "Gemma 4 31B": (10, 0, "left"),
                    "Nemotron 3.5 Lightning": (-10, -10, "right"), "Nemotron 3.5 Super": (0, -18, "center"),
                    "gpt-oss-20b": (-10, 0, "right"), "gpt-oss-120b": (0, 16, "center")}


def plot_size_blog(ctx, t, labels=SIZE_BLOG_LABELS):
    """Web version of cross_size's win-rate panel: win rate against size (log), by reasoning group, PNG only."""
    import os
    from matplotlib.ticker import FuncFormatter
    x = t.dropna(subset=["size_b", "win_rate"])
    x = x.assign(label=[S.SHORT_NAMES.get(d, d) for d in x["display"]])
    gs = pd.read_csv(os.path.join(ctx.tables, "games_summary.csv"))
    heuristic = gs[(gs["experiment"] == "bots") & (gs["model"] == "heuristic")]["win_rate"].iloc[0]
    with S.plt.rc_context(S.BLOG):
        fig, ax = S.figure(figsize=(11, 7))
        ax.scatter(x["size_b"], x["win_rate"], s=130, c=[S.REASONING.get(g, S.MUTED) for g in x["group_games"]],
                   edgecolors="white", linewidths=1.2, zorder=3)
        for _, r in x[x["label"].isin(labels)].iterrows():
            dx, dy, ha = labels[r["label"]]
            ax.annotate(r["label"], (r["size_b"], r["win_rate"]), xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va="center", fontsize=14, color=S.INK2)
        ax.axhline(heuristic, color=S.INK2, linewidth=1.5, linestyle="--", zorder=2)
        ax.annotate(f"heuristic bot vs itself ({heuristic:.0%})", (0.01, heuristic), xycoords=("axes fraction", "data"),
                    xytext=(0, 5), textcoords="offset points", fontsize=14, color=S.INK2, va="bottom")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}B"))
        ax.set_ylim(0.1, 0.56)
        S.pct(ax, "y")
        ax.set_xlabel("parameters (billions, from the model name; log scale)")
        ax.set_ylabel("win rate against the heuristic bot")
        ax.set_title("Win rate against the heuristic bot by model size", loc="left")
        for g in ("reasoning", "non-reasoning"):
            ax.plot([], [], "o", markersize=11, color=S.REASONING[g], label=g.replace("-", " "))
        fig.legend(loc="outside lower center", ncol=2)
        S.save(fig, ctx.out, "size_blog", x[["display", "size_b", "win_rate", "group_games"]], exts=("png",))


def plot_size_families_blog(ctx, t, families=("Qwen3.5", "Gemma 4")):
    """
    Web version of size against win rate for the two families with a size
    ladder: one line per family, each point labelled with its win rate (below
    it where the other family is higher at the same size), PNG only.
    """
    import os
    from matplotlib.ticker import FixedLocator, NullLocator
    x = t[t["family"].isin(families)].dropna(subset=["size_b", "win_rate"]).sort_values("size_b")
    gs = pd.read_csv(os.path.join(ctx.tables, "games_summary.csv"))
    heuristic = gs[(gs["experiment"] == "bots") & (gs["model"] == "heuristic")]["win_rate"].iloc[0]
    with S.plt.rc_context(S.BLOG):
        fig, ax = S.figure(figsize=(11, 7))
        for f, color in zip(families, (S.SERIES[2], S.SERIES[6])):
            g = x[x["family"] == f]
            ax.plot(g["size_b"], g["win_rate"], "o-", color=color, linewidth=3, markersize=11, label=f,
                    markeredgecolor="white", markeredgewidth=1.2, zorder=3)
            for _, r in g.iterrows():
                below = (x[(x["size_b"] == r["size_b"]) & (x["family"] != f)]["win_rate"] > r["win_rate"]).any()
                ax.annotate(f"{r['win_rate']:.0%}", (r["size_b"], r["win_rate"]), xytext=(0, -16 if below else 16),
                            textcoords="offset points", ha="center", va="center", fontsize=14, color=S.INK2)
            last = g.iloc[-1]
            ax.annotate(f, (last["size_b"], last["win_rate"]), xytext=(14, 0), textcoords="offset points",
                        va="center", fontsize=15, color=color, fontweight="bold")
        ax.axhline(heuristic, color=S.INK2, linewidth=1.5, linestyle="--", zorder=2)
        ax.annotate(f"heuristic bot vs itself ({heuristic:.0%})", (0.01, heuristic), xycoords=("axes fraction", "data"),
                    xytext=(0, 5), textcoords="offset points", fontsize=14, color=S.INK2, va="bottom")
        ax.set_xscale("log")
        sizes = sorted(set(x["size_b"]))
        ax.xaxis.set_major_locator(FixedLocator(sizes))
        ax.xaxis.set_minor_locator(NullLocator())
        ticks = ax.set_xticklabels([f"{v:g}B" for v in sizes])
        for a, b, size_a, size_b in zip(ticks, ticks[1:], sizes, sizes[1:]):
            if size_b / size_a < 1.3:                       # close sizes: labels lean apart
                a.set_horizontalalignment("right")
                b.set_horizontalalignment("left")
        ax.set_xlim(sizes[0] / 1.4, sizes[-1] * 2.6)
        ax.set_ylim(0.1, 0.55)
        S.pct(ax, "y")
        ax.set_xlabel("parameters (billions, from the model name; log scale)")
        ax.set_ylabel("win rate against the heuristic bot")
        ax.set_title("Win rate by model size within two families", loc="left")
        fig.legend(loc="outside lower center", ncol=len(families))
        S.save(fig, ctx.out, "size_families_blog", x[["display", "family", "size_b", "win_rate"]], exts=("png",))


def run(ctx):
    t = models_table(ctx)
    ctx.write("cross_models", t)
    both = t.dropna(subset=["win_rate", "agreement_shown"])
    corr = correlation(both["agreement_shown"], both["win_rate"])
    ctx.write("cross_correlation", [{"measure": k, "estimate": v[0], "lo": v[1], "hi": v[2], "models": len(both)}
                                    for k, v in corr.items()])
    plot_spots_vs_games(ctx, t, corr)
    plot_spots_vs_games_short(ctx, t, corr)
    plot_cost(ctx, t)
    plot_size(ctx, t)
    plot_size_blog(ctx, t)
    plot_size_families_blog(ctx, t)
    for k, v in corr.items():
        ctx.note("cross", f"{k} correlation of spot agreement (shown) and main win rate", v[0], len(both), v[1], v[2],
                 "correlation", "bootstrap over models")
