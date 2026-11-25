"""
Games against the heuristic bot (main, hidden, main-outcomes, main-history)
and the bot baselines (bots): per dataset the win rate (Wilson), the invalid
rate overall and by cause (bootstrap over games), captures made and suffered
per game, game length, time, tokens and cost per move.

Tables: games_summary.csv. Figures: main_win_rates (and _short, _blog), main_invalid_rates.
"""
import numpy as np
import pandas as pd

from analysis.final import style as S
from analysis.final.metrics import rate, ratio

EXPERIMENTS = ("main", "hidden", "main-outcomes", "main-history", "bots")


def summarise(ctx):
    g, dec = ctx.frame("games"), ctx.frame("game_decisions")
    meta = ctx.datasets.set_index("dataset")
    think = dec[dec["replied"] == True].groupby(["experiment", "model"])["thinking_tokens"].mean()
    rows = []
    for (exp, model, setting), x in g.groupby(["experiment", "model", "setting"], dropna=False):
        key = f"{exp}/{model}" + (f"/{setting}" if isinstance(setting, str) and setting else "")
        m = meta.loc[key]
        n, wins = len(x), int(x["won"].sum())
        r, lo, hi = rate(wins, n)
        calls = x["calls"].sum()
        row = {"experiment": exp, "model": model, "display": m["display"], "provider": m["provider"],
               "family": m["family"], "size_b": m["size_b"], "reasoning_group": m.get("reasoning_group"),
               "games": n, "wins": wins, "win_rate": r, "win_lo": lo, "win_hi": hi,
               "no_winner": int(x["no_winner"].sum()), "calls": int(calls)}
        if calls:
            row["invalid_rate"], row["invalid_lo"], row["invalid_hi"] = ratio(x["invalid"], x["calls"], x["game"])
            for c in ("empty", "unparseable", "illegal"):
                row[f"invalid_{c}_rate"] = x[f"invalid_{c}"].sum() / calls
            row["seconds_per_move"] = x["latency_ms"].sum() / calls / 1000
            row["output_tokens_per_move"] = x["completion_tokens"].sum() / calls
            row["thinking_tokens_per_move"] = think.get((exp, model), np.nan)
            row["cost_usd"] = m.get("cost_usd")
            row["cost_per_game_usd"] = m.get("cost_usd") / n if pd.notna(m.get("cost_usd")) else np.nan
            row["cost_status"] = m.get("cost_status")
        for c in ("captures_made", "times_captured", "dice_rolls", "turns", "decisions"):
            row[f"{c}_per_game"] = x[c].mean()
            row[f"{c}_sd"] = x[c].std()
        rows.append(row)
    return pd.DataFrame(rows)


def plot_win_rates(ctx, t):
    main = t[t["experiment"] == "main"].sort_values("win_rate", ascending=False)
    bots = t[t["experiment"] == "bots"].set_index("model")
    fig, ax = S.figure(figsize=(6.4, 0.24 * len(main) + 1.4))
    colors = [S.REASONING.get(g, S.MUTED) for g in main["reasoning_group"]]
    S.dot_plot(ax, list(main["display"]), main["win_rate"], main["win_lo"], main["win_hi"], colors,
               "win rate against the heuristic bot (95% Wilson interval)")
    for model, label, where in (("heuristic", "heuristic vs itself", "bottom"), ("random", "random bot", "top"),
                                ("gt", "search player", "top")):
        if model in bots.index:
            S.reference_line(ax, x=bots.loc[model, "win_rate"], label=label, where=where)
    S.pct(ax)
    ax.set_xlim(0, 1)
    for g, c in S.REASONING.items():
        if g != "bot":
            ax.plot([], [], "o", color=c, label=g.replace("-", " "))
    ax.legend(loc="lower right")
    ax.set_title("Main games: each model against the heuristic bot (100 games; Kimi K3.1: 50)")
    S.save(fig, ctx.out, "main_win_rates", main[["display", "games", "win_rate", "win_lo", "win_hi",
                                                  "reasoning_group"]])


def plot_win_rates_blog(ctx, t):
    """Web version of main_win_rates: sorted bars, no intervals, the three reference players as dashed lines."""
    main = t[t["experiment"] == "main"]
    bots = t[t["experiment"] == "bots"].set_index("model")["win_rate"]
    S.bar_chart(ctx.out, "main_win_rates_blog", main[["display", "games", "win_rate", "reasoning_group"]], "win_rate",
                "win rate against the heuristic bot", "Win rate against the heuristic bot (100 games; Kimi K3.1: 50)",
                refs=[(bots["random"], "random bot", "center"), (bots["heuristic"], "heuristic vs itself", "right"),
                      (bots["gt"], "search player", "left")])


def plot_win_rates_short(ctx, t):
    """One-column version of main_win_rates: the top 8 named, the rest as one row."""
    main = t[t["experiment"] == "main"].sort_values("win_rate", ascending=False)
    bots = t[t["experiment"] == "bots"].set_index("model")["win_rate"]
    S.short_dot_plot(ctx.out, "main_win_rates_short", main[["display", "games", "win_rate", "win_lo", "win_hi",
                                                            "reasoning_group"]], 8, "win_rate", "win_lo", "win_hi",
                     "win rate against the heuristic bot",
                     refs=[(bots[m], label, "--", ha) for m, label, ha in
                           (("random", "random", "center"), ("heuristic", "heuristic", "right"),
                            ("gt", "search", "left")) if m in bots.index])


def plot_invalid(ctx, t):
    main = t[t["experiment"] == "main"].sort_values("invalid_rate", ascending=False)
    fig, ax = S.figure(figsize=(6.4, 0.24 * len(main) + 1.4))
    left = np.zeros(len(main))
    y = np.arange(len(main))[::-1]
    for c, color in zip(("illegal", "unparseable", "empty"), (S.SERIES[0], S.SERIES[1], S.SERIES[6])):
        v = main[f"invalid_{c}_rate"].to_numpy()
        ax.barh(y, v, left=left, height=0.62, color=color, label=c, edgecolor="white", linewidth=0.8)
        left += v
    ax.set_yticks(y, main["display"])
    ax.set_xlabel("invalid replies per move (main games)")
    ax.grid(axis="y", visible=False)
    S.pct(ax)
    ax.legend(title="cause", loc="lower right")
    S.save(fig, ctx.out, "main_invalid_rates",
           main[["display", "calls", "invalid_rate", "invalid_illegal_rate", "invalid_unparseable_rate",
                 "invalid_empty_rate"]])


def run(ctx):
    t = summarise(ctx)
    ctx.write("games_summary", t)
    plot_win_rates(ctx, t)
    plot_win_rates_short(ctx, t)
    plot_win_rates_blog(ctx, t)
    plot_invalid(ctx, t)
    for _, r in t.iterrows():
        ctx.note(f"games/{r['experiment']}", f"win rate: {r['display']}", r["win_rate"], r["games"],
                 r["win_lo"], r["win_hi"], "share of games")
        if r["calls"]:
            ctx.note(f"games/{r['experiment']}", f"invalid rate: {r['display']}", r["invalid_rate"], r["calls"],
                     r["invalid_lo"], r["invalid_hi"], "share of moves", "interval: bootstrap over games")
    return t
