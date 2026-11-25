"""
The seat-effect check: four heuristic bots, 1,000 games, read from the two
games files of the check (884 games, then 116 in a second run whose decisions
file is damaged; the games file is intact). Wins by seat position and of the
first player (chosen by the roll-off), with Wilson intervals and a chi-square
test of equal seats (3 degrees of freedom, exact survival function).

Tables: seat_check.csv, seat_check_tests.csv.
"""
import glob
import json
import math
import os

import pandas as pd

from analysis.final.metrics import rate


def chi2_sf_3(x):
    """P(chi-square with 3 degrees of freedom > x)."""
    return math.erfc(math.sqrt(x / 2)) + math.sqrt(2 * x / math.pi) * math.exp(-x / 2)


def games(ctx):
    out, files = [], []
    for root in ctx.roots.values():
        for p in sorted(glob.glob(os.path.join(root, "seat-check*", "*", "games.jsonl"))):
            files.append(p)
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.endswith("\n"):
                        out.append(json.loads(line))
    return out, files


def run(ctx):
    gs, files = games(ctx)
    if not gs:
        return
    seeds = [g["seed"] for g in gs]
    seat_wins = pd.Series([g["seat_assignment"][str(g["winner"])] for g in gs if g["winner"] is not None])
    first_wins = sum(g["winner"] == g["first_player"] for g in gs)
    n = len(seat_wins)
    rows = []
    for seat in range(4):
        k = int((seat_wins == seat).sum())
        p, lo, hi = rate(k, n)
        rows.append({"seat_position": seat, "games": n, "wins": k, "win_rate": p, "lo": lo, "hi": hi})
    chi = sum((r["wins"] - n / 4) ** 2 / (n / 4) for r in rows)
    p_first, lo_f, hi_f = rate(first_wins, len(gs))
    first_seat = pd.Series([g["seat_assignment"][str(g["first_player"])] for g in gs]).value_counts().to_dict()
    tests = [{"games": len(gs), "files": len(files), "unique_seeds": len(set(seeds)), "seed_min": min(seeds),
              "seed_max": max(seeds), "no_winner": sum(g["winner"] is None for g in gs), "chi2_seats": chi,
              "p_equal_seats": chi2_sf_3(chi), "first_player_wins": first_wins, "first_player_rate": p_first,
              "first_lo": lo_f, "first_hi": hi_f, "first_player_seat_counts": json.dumps(first_seat)}]
    ctx.write("seat_check", rows)
    ctx.write("seat_check_tests", tests)
    for r in rows:
        ctx.note("seat-check", f"wins of seat {r['seat_position']} (4 heuristic bots)", r["win_rate"], r["games"],
                 r["lo"], r["hi"], "share of games", "chance 25%")
    ctx.note("seat-check", "chi-square p, all seats equal", tests[0]["p_equal_seats"], len(gs),
             detail=f"chi-square {chi:.2f}, 3 df")
    ctx.note("seat-check", "first player's share of wins (roll-off)", p_first, len(gs), lo_f, hi_f, "share of games",
             "chance 25%")
