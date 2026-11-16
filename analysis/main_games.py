"""
Part 1: main runs (experiment "main", each model against the heuristic).

Tables:
  main_win_rates.csv   per model: games, wins, win rate with a 95% Wilson
                       interval, invalid-reply rate with a game-bootstrap
                       interval, excluded and replayed games
  bot_baselines.csv    the same for the bots (experiment "bots": random,
                       first-listed, heuristic, search player) against the
                       heuristic
  main_paired.csv      every pair of models, and every model against every
                       bot, on their shared seeds (the same dice and first
                       player): win rates, difference with a seed-bootstrap
                       interval, exact McNemar p, Holm-adjusted within each
                       family (model vs model; model vs bot)
Figures: main_win_rates, main_invalid_rates, main_paired (PNG and PDF).
"""
from itertools import combinations

import numpy as np

from analysis.common import figure, game_rows, games, save_figure, write_csv
from analysis.stats import bootstrap, holm, paired, wilson

BOT_LABELS = {"random": "random", "first": "first-listed", "heuristic": "heuristic",
              "gt": "search player"}


def run_table(run_list, runs):
    """(one row per run, {model: {seed: won}}) for game runs."""
    rows, by_seed = [], {}
    for run in run_list:
        recs, notes = games(run, runs)
        g = game_rows(run, recs)
        if not g:
            continue
        wins, n = sum(r["won"] for r in g), len(g)
        calls = sum(r["calls"] for r in g)
        inv = (bootstrap([[r["invalid"], r["calls"]] for r in g], lambda s: s[..., 0] / s[..., 1])
               if calls else (None,) * 4)
        lo, hi = wilson(wins, n)
        rows.append({"model": run.model, "run": run.key, "games": n, "wins": wins,
                     "win_rate": wins / n, "win_lo": lo, "win_hi": hi,
                     "no_winner": sum(r["no_winner"] for r in g),
                     "seeds": f"{g[0]['seed']}-{g[-1]['seed']}", "calls": calls,
                     "invalid": sum(r["invalid"] for r in g), "invalid_rate": inv[0],
                     "invalid_lo": inv[1], "invalid_hi": inv[2],
                     "excluded_games": " ".join(notes["excluded"]),
                     "replaced_by": " ".join(notes["replaced_by"])})
        by_seed[run.model] = {r["seed"]: r["won"] for r in g}
    return rows, by_seed


def paired_rows(by_seed, bots):
    """Paired win comparisons on shared seeds, Holm-adjusted per family."""
    rows = []
    families = [("model vs model", list(combinations(sorted(by_seed), 2))),
                ("model vs bot", [(m, b) for m in sorted(by_seed) for b in sorted(bots)])]
    for family, pairs in families:
        start = len(rows)
        for a, b in pairs:
            wa, wb = by_seed[a], {**by_seed, **bots}[b]
            shared = sorted(set(wa) & set(wb))
            if shared:
                rows.append({"family": family, "a": a, "b": b,
                             **paired([wa[s] for s in shared], [wb[s] for s in shared])})
        for r, p in zip(rows[start:], holm([r["p_mcnemar"] for r in rows[start:]])):
            r["p_holm"] = p
    return rows


def plot_rates(rows, bots, out):
    rows = sorted(rows, key=lambda r: r["win_rate"])
    fig, ax = figure(figsize=(7, 0.35 * len(rows) + 1.5))
    y = np.arange(len(rows))
    ax.barh(y, [r["win_rate"] for r in rows], color="#4C72B0",
            xerr=[[r["win_rate"] - r["win_lo"] for r in rows], [r["win_hi"] - r["win_rate"] for r in rows]],
            capsize=2)
    ax.set_yticks(y, [f"{r['model']} (n={r['games']})" for r in rows])
    for (name, b), color in zip(sorted(bots.items()), ("#8172B3", "#937860", "#DA8BC3", "#8C8C8C")):
        ax.axvline(b["win_rate"], color=color, linestyle="--", linewidth=1.2,
                   label=f"{BOT_LABELS.get(name, name)} {b['win_rate']:.2f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("win rate against the heuristic (95% Wilson interval)")
    ax.legend(title="bot against the heuristic", fontsize=7, loc="lower right")
    save_figure(fig, out, "main_win_rates")

    rows = [r for r in rows if r["invalid_rate"] is not None]
    fig, ax = figure(figsize=(7, 0.35 * len(rows) + 1.2))
    y = np.arange(len(rows))
    ax.barh(y, [r["invalid_rate"] for r in rows], color="#C44E52",
            xerr=[[r["invalid_rate"] - r["invalid_lo"] for r in rows],
                  [r["invalid_hi"] - r["invalid_rate"] for r in rows]], capsize=2)
    ax.set_yticks(y, [r["model"] for r in rows])
    ax.set_xlabel("invalid-reply rate (95% game-bootstrap interval)")
    save_figure(fig, out, "main_invalid_rates")


def plot_paired(rows, order, out):
    pairs = [r for r in rows if r["family"] == "model vs model"]
    if not pairs:
        return
    idx = {m: i for i, m in enumerate(order)}
    mat = np.full((len(order), len(order)), np.nan)
    stars = {}
    for r in pairs:
        i, j = idx[r["a"]], idx[r["b"]]
        mat[i, j], mat[j, i] = r["diff"], -r["diff"]
        stars[(i, j)] = stars[(j, i)] = r["p_holm"] < 0.05
    fig, ax = figure(figsize=(0.45 * len(order) + 3, 0.45 * len(order) + 2))
    im = ax.imshow(mat, cmap="RdBu", vmin=-1, vmax=1)
    for (i, j), s in stars.items():
        if s:
            ax.text(j, i, "*", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize=7)
    ax.set_yticks(range(len(order)), order, fontsize=7)
    fig.colorbar(im, label="win rate of row minus column, shared seeds (* Holm p < 0.05)")
    save_figure(fig, out, "main_paired")


def run(runs, out):
    main = [r for r in runs if r.kind == "games" and r.experiment == "main"]
    bot_runs = [r for r in runs if r.kind == "games" and r.experiment == "bots"]
    rows, by_seed = run_table(main, runs)
    bot_rows, bot_seeds = run_table(bot_runs, runs)
    bots = {r["model"]: r for r in bot_rows}
    pairs = paired_rows(by_seed, bot_seeds)
    write_csv(out, "main_win_rates", rows)
    write_csv(out, "bot_baselines", bot_rows)
    write_csv(out, "main_paired", pairs)
    if rows:
        plot_rates(rows, bots, out)
        plot_paired(pairs, [r["model"] for r in sorted(rows, key=lambda r: -r["win_rate"])], out)
    mm = [r for r in pairs if r["family"] == "model vs model"]
    return [f"main runs: {len(rows)} models, {sum(r['games'] for r in rows)} games; "
            f"{sum(r['p_holm'] < 0.05 for r in mm)} of {len(mm)} model pairs differ (Holm p < 0.05)"]
