"""
Head-to-head games between models.

h2h-2p: every pair of four Bedrock models, 50 games each: the win matrix
(row model's win rate against the column model, Wilson) and a Bradley-Terry
ranking (strengths by the MM algorithm, log scale centred on 0; 95% intervals
from 2,000 resamples of the games within each pair).
h2h-4p and h2h-4p-heuristic: four-player games over the six cyclic seatings:
wins per model overall (Wilson), per seating and per seat position, and the
first player's share of wins.

Tables: h2h_2p_matrix.csv, h2h_2p_bradley_terry.csv, h2h_4p_wins.csv,
h2h_4p_by_seating.csv, h2h_4p_by_seat.csv. Figures: h2h_2p_matrix,
h2h_bradley_terry, h2h_4p_wins.
"""
import numpy as np
import pandas as pd

from analysis.final import style as S
from analysis.final.metrics import rate


def bradley_terry(wins, iters=2000, tol=1e-10):
    """Strengths p (sum 1) from wins[i][j] = games i beat j, by the MM algorithm (Hunter 2004)."""
    w = np.asarray(wins, float)
    n = w + w.T
    p = np.ones(len(w)) / len(w)
    for _ in range(iters):
        denom = np.array([sum(n[i, j] / (p[i] + p[j]) for j in range(len(w)) if j != i and n[i, j])
                          for i in range(len(w))])
        new = w.sum(1) / denom
        new /= new.sum()
        if np.max(np.abs(new - p)) < tol:
            return new
        p = new
    return p


def names(h):
    return dict(zip(h["model"], h["display"]))


def two_player(ctx, h):
    disp = names(h)
    x = h[h["experiment"] == "h2h-2p"]
    models = sorted(x["model"].unique())
    idx = {m: i for i, m in enumerate(models)}
    wins = np.zeros((len(models), len(models)))
    games = []
    for _, g in x.groupby("game"):
        a, b = g.iloc[0], g.iloc[1]
        winner, loser = (a, b) if a["won"] else (b, a)
        if a["won"] or b["won"]:
            wins[idx[winner["model"]], idx[loser["model"]]] += 1
            games.append((idx[winner["model"]], idx[loser["model"]]))
    matrix = []
    for m in models:
        for o in models:
            if m != o:
                k, n = wins[idx[m], idx[o]], wins[idx[m], idx[o]] + wins[idx[o], idx[m]]
                r, lo, hi = rate(k, n)
                matrix.append({"model": m, "display": disp[m], "opponent": o,
                               "opponent_display": disp[o], "games": int(n), "wins": int(k),
                               "win_rate": r, "lo": lo, "hi": hi})
    p = bradley_terry(wins)
    rng = np.random.default_rng(0)
    pairs = {}
    for wi, li in games:
        pairs.setdefault(tuple(sorted((wi, li))), []).append((wi, li))
    boot = []
    for _ in range(2000):
        bw = np.zeros_like(wins)
        for lst in pairs.values():
            for k in rng.integers(0, len(lst), len(lst)):
                bw[lst[k]] += 1
        boot.append(np.log(bradley_terry(bw, iters=500)))
    boot = np.array(boot)
    boot -= boot.mean(1, keepdims=True)
    logp = np.log(p) - np.log(p).mean()
    bt = pd.DataFrame({"model": models, "display": [disp[m] for m in models], "strength": logp,
                       "lo": np.percentile(boot, 2.5, 0), "hi": np.percentile(boot, 97.5, 0),
                       "wins": wins.sum(1).astype(int), "games": (wins + wins.T).sum(1).astype(int)})
    bt["win_prob_vs_average"] = 1 / (1 + np.exp(-bt["strength"]))
    bt = bt.sort_values("strength", ascending=False)
    bt["rank"] = range(1, len(bt) + 1)
    return pd.DataFrame(matrix), bt


def four_player(ctx, h, exp):
    x = h[h["experiment"] == exp]
    n_games = x["game"].nunique()
    rows, seating, seat = [], [], []
    for m, g in x.groupby("model"):
        r, lo, hi = rate(g["won"].sum(), len(g))
        rows.append({"experiment": exp, "model": m, "display": names(h)[m],
                     "games": len(g), "wins": int(g["won"].sum()), "win_rate": r, "lo": lo, "hi": hi,
                     "mean_place": g["place"].mean(), "captures_per_game": g["captures_made"].mean(),
                     "invalid_rate": g["invalid"].sum() / g["calls"].sum() if g["calls"].sum() else np.nan})
        for k, s in g.groupby("seating"):
            seating.append({"experiment": exp, "model": m, "seating": k, "games": len(s),
                            "wins": int(s["won"].sum()), "win_rate": s["won"].mean()})
        for k, s in g.groupby("seat_position"):
            seat.append({"experiment": exp, "model": m, "seat_position": k, "games": len(s),
                         "wins": int(s["won"].sum()), "win_rate": s["won"].mean()})
    first = x[x["first_player"] == 1]
    r, lo, hi = rate(first["won"].sum(), len(first))
    return pd.DataFrame(rows).sort_values("win_rate", ascending=False), seating, seat, \
        {"experiment": exp, "games": n_games, "first_player_wins": int(first["won"].sum()), "rate": r,
         "lo": lo, "hi": hi, "expected": 1 / x["n_players"].iloc[0]}


def plot_matrix(ctx, matrix, bt):
    order = list(bt["model"])
    labels = list(bt["display"])
    grid = np.full((len(order), len(order)), np.nan)
    for _, r in matrix.iterrows():
        grid[order.index(r["model"]), order.index(r["opponent"])] = r["win_rate"]
    fig, ax = S.figure(figsize=(5.2, 4.2))
    im = ax.imshow(grid, cmap=S.DIVERGING.reversed(), vmin=0, vmax=1)
    for i in range(len(order)):
        for j in range(len(order)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if abs(grid[i, j] - 0.5) > 0.3 else S.INK)
    ax.set_xticks(range(len(order)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(order)), labels)
    ax.grid(False)
    ax.set_xlabel("opponent")
    ax.set_title("Two-player head-to-head: row model's win rate (50 games per pair)")
    fig.colorbar(im, ax=ax, fraction=0.046, format="{x:.0%}").set_label("win rate")
    S.save(fig, ctx.out, "h2h_2p_matrix", matrix)
    fig, ax = S.figure(figsize=(5.2, 0.4 * len(bt) + 1.2))
    S.dot_plot(ax, list(bt["display"]), bt["strength"], bt["lo"], bt["hi"], [S.SERIES[0]] * len(bt),
               "Bradley-Terry strength (log odds against an average model; 95% interval)")
    S.reference_line(ax, x=0, label="average")
    ax.set_title("Two-player ranking")
    S.save(fig, ctx.out, "h2h_bradley_terry", bt)


def plot_4p(ctx, tables):
    fig, axes = S.figure(ncols=len(tables), figsize=(5.2 * len(tables), 2.6), squeeze=False)
    for ax, (exp, t) in zip(axes[0], tables.items()):
        S.dot_plot(ax, list(t["display"]), t["win_rate"], t["lo"], t["hi"], [S.SERIES[0]] * len(t), "win rate")
        S.reference_line(ax, x=0.25, label="1 in 4")
        S.pct(ax)
        ax.set_xlim(0, 1)
        ax.set_title(f"{exp}: {int(t['games'].iloc[0])} four-player games")
    S.save(fig, ctx.out, "h2h_4p_wins", pd.concat(tables.values()))


def run(ctx):
    h = ctx.frame("h2h_players")
    matrix, bt = two_player(ctx, h)
    ctx.write("h2h_2p_matrix", matrix)
    ctx.write("h2h_2p_bradley_terry", bt)
    plot_matrix(ctx, matrix, bt)
    for _, r in bt.iterrows():
        ctx.note("h2h/2p", f"Bradley-Terry strength: {r['display']} (rank {r['rank']})", r["strength"], r["games"],
                 r["lo"], r["hi"], "log odds vs average")
    for _, r in matrix.iterrows():
        if r["model"] < r["opponent"]:
            ctx.note("h2h/2p", f"{r['display']} vs {r['opponent_display']}: win rate", r["win_rate"], r["games"],
                     r["lo"], r["hi"], "share of games")
    tables, seatings, seats, firsts = {}, [], [], []
    for exp in ("h2h-4p", "h2h-4p-heuristic"):
        t, se, st, first = four_player(ctx, h, exp)
        tables[exp] = t
        seatings += se
        seats += st
        firsts.append(first)
        for _, r in t.iterrows():
            ctx.note(f"h2h/{exp}", f"win rate: {r['display']}", r["win_rate"], r["games"], r["lo"], r["hi"],
                     "share of games", "chance 25%")
        ctx.note(f"h2h/{exp}", "first player's share of wins", first["rate"], first["games"], first["lo"],
                 first["hi"], "share of games", "chance 25%")
    ctx.write("h2h_4p_wins", pd.concat(tables.values()))
    ctx.write("h2h_4p_by_seating", seatings)
    ctx.write("h2h_4p_by_seat", seats)
    ctx.write("h2h_4p_first_player", firsts)
    plot_4p(ctx, tables)
