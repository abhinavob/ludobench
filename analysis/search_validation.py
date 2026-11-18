"""
Search-player validation (CPU only, no models): how much the search player's
choices depend on its depth and on its evaluator weights.

Variants: GameTheoryMultiplayerAgent at depth 1, 2 (the reference: the search
player the spot runs log), 3 and 4 (on the spots only, not in games); and at
depth 2 with each evaluator weight multiplied by an independent factor drawn
uniformly from [1 - s, 1 + s], three draws each for s = 0.25 and s = 0.5
(seeded).

  search_variants.csv         the variants and their weights
  search_spots_agreement.csv  per pair of variants (depths with each other,
                              every weighted variant with depth 2) and
                              category: spots with at least 2 distinct moves
                              (spots_40_repaired), and the share where both
                              choose the same distinct move (Wilson interval)
  search_games.csv            head-to-head 2-player games per pairing on seeds
                              0..N-1, each seed played twice: "as run" puts the
                              first variant in the first seat on even seeds,
                              "swapped" in the other seat, "both" adds them up.
                              Wins of each, the first's win rate with a Wilson
                              interval, and an exact binomial p against 0.5
                              (Holm across pairings within each of the three):
                              over games for one direction, for "both" over
                              seeds one variant won in both seats (dice luck
                              cancels). Depth 2 against itself is the control:
                              its "both" must come out exactly even
Figures: search_spots_agreement, search_games.

    python analysis/search_validation.py [--games 200] [--workers 4] [--out DIR]

The weights are literals in the authors' game_theory_agent.py and
game_theory_multiplayer_agent.py, which stay unchanged; the weighted variants
are subclasses that restate those evaluators with the weights as parameters
(tests/test_search_validation.py checks they score and choose exactly as the
originals with the original weights).
"""
import argparse
import glob
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ludo_rules as rules  # noqa: E402
from analysis.common import REPO, figure, save_figure, write_csv  # noqa: E402
from analysis.stats import holm, mcnemar, wilson  # noqa: E402
from board_render import group_moves  # noqa: E402
from game_theory_agent import GameTheoryAgent  # noqa: E402
from game_theory_multiplayer_agent import GameTheoryMultiplayerAgent  # noqa: E402
from ludo_env import LudoEnv  # noqa: E402
from spot_evaluation import SpotEnv  # noqa: E402

WEIGHTS = {"progress": 0.0025, "finished": 0.20, "base": -0.05, "safe": 0.01}   # as in both evaluators
SPREADS, DRAWS = (0.25, 0.5), 3
REFERENCE = "depth 2"


def terms(state, pid, original):
    """(progress, pieces in base, finished, on safe squares) of a player, as both evaluators count them."""
    hs, he = rules.home_range(pid)
    st = rules.START_POSITIONS[pid]
    prog, base, fin, safe = 0.0, 0, 0, 0
    for pos in state[pid]:
        if pos == -1:
            base += 1
            continue
        if pos == he:
            fin += 1
        if 0 <= pos < rules.BOARD_SIZE:
            prog += (pos - st) % rules.BOARD_SIZE + 1
            safe += pos in rules.SAFE_SQUARES
        elif pos >= hs:
            prog += rules.track_length(original) + (pos - hs + 1)
    return prog, base, fin, safe


def score(w, prog, base, fin, safe):
    """The evaluators' weighted sum, in their order of terms."""
    s = 0.0
    s += w["progress"] * prog
    s += w["finished"] * fin
    s += w["base"] * base
    s += w["safe"] * safe
    return s


class WeightedTwoPlayer(GameTheoryAgent):
    def __init__(self, search_depth, weights):
        super().__init__(search_depth)
        self.weights = weights

    def _evaluate_state(self, state):
        me, op = self._root_player, self._other_player(self._root_player)
        a, b = terms(state, me, self._original), terms(state, op, self._original)
        return max(-0.999, min(0.999, score(self.weights, *(x - y for x, y in zip(a, b)))))


class WeightedSearch(GameTheoryMultiplayerAgent):
    """The search player with evaluator weights as a parameter (2 players: its 2-player agent)."""

    def __init__(self, search_depth, weights):
        super().__init__(search_depth)
        self.weights = weights
        self._two_player_agent = WeightedTwoPlayer(search_depth, weights)

    def _evaluate_state(self, state):
        raw = [score(self.weights, *terms(state, pid, self._original)) for pid in self._player_ids]
        mean_raw = sum(raw) / max(1, len(raw))
        return tuple(max(-0.999, min(0.999, s - mean_raw)) for s in raw)


def variants():
    """name -> (depth, weights or None for the original evaluator)."""
    out = {f"depth {d}": (d, None) for d in (1, 2, 3, 4)}
    rng = np.random.default_rng(0)
    for s in SPREADS:
        for k in range(1, DRAWS + 1):
            out[f"weights ±{s:.0%} #{k}"] = (2, {name: w * rng.uniform(1 - s, 1 + s)
                                                for name, w in WEIGHTS.items()})
    return out


def agent(spec):
    depth, weights = spec
    return GameTheoryMultiplayerAgent(depth) if weights is None else WeightedSearch(depth, weights)


def load_spots(folder="spots_40_repaired"):
    return [s for f in sorted(glob.glob(os.path.join(REPO, folder, "spots_*.json")))
            for s in json.load(open(f, encoding="utf-8"))]


def spot_choices(specs, spots):
    """Per spot with at least 2 distinct moves: its category, its groups and each variant's choice."""
    agents = {name: agent(spec) for name, spec in specs.items()}
    out = []
    for s in spots:
        env = SpotEnv(s["players"], {int(p): list(t) for p, t in s["tokens"].items()})
        p, d = s["current_player"], s["dice"]
        legal = rules.legal_moves(env.tokens, p, d)
        groups = group_moves(env, p, d, legal) if legal else []
        if len(groups) >= 2:
            out.append({"category": s["scenario"], "group": {t: i for i, g in enumerate(groups) for t in g},
                        "choice": {n: a.select_move(env, p, d, legal) for n, a in agents.items()}})
    return out


def spot_pairs(names):
    depths = [n for n in names if n.startswith("depth")]
    return ([(a, b) for i, a in enumerate(depths) for b in depths[i + 1:]]
            + [(REFERENCE, n) for n in names if not n.startswith("depth")])


def spot_agreement(choices, pairs):
    rows = []
    for a, b in pairs:
        for cat in ["all"] + sorted({c["category"] for c in choices}):
            sel = [c for c in choices if cat in ("all", c["category"])]
            same = sum(c["group"][c["choice"][a]] == c["group"][c["choice"][b]] for c in sel)
            lo, hi = wilson(same, len(sel))
            rows.append({"a": a, "b": b, "category": cat, "spots": len(sel), "same": same,
                         "agreement": same / len(sel), "lo": lo, "hi": hi})
    return rows


def _play(job):
    """
    One game: (pairing index, seed, swapped, 1 if the first variant won, 0 if
    the second, None if no winner). The first variant sits first on even seeds,
    or on odd seeds when swapped.
    """
    i, seed, swapped, spec_a, spec_b = job
    env = LudoEnv(seed=seed, num_players=2)
    first, second = env.player_ids
    pid_a, pid_b = (first, second) if (seed % 2 == 0) != swapped else (second, first)
    winner, _ = env.play_game({pid_a: agent(spec_a), pid_b: agent(spec_b)}, max_steps=1000)
    return i, seed, swapped, None if winner is None else int(winner == pid_a)


def _quiet_cwd():
    """The engine writes log files to its cwd: give each process a scratch folder."""
    os.chdir(tempfile.mkdtemp(prefix="search_validation_"))


def head_to_head(specs, pairings, games, workers=1):
    jobs = [(i, seed, swapped, specs[a], specs[b]) for i, (a, b) in enumerate(pairings)
            for seed in range(games) for swapped in (False, True)]
    if workers > 1:
        with ProcessPoolExecutor(workers, initializer=_quiet_cwd) as pool:
            results = list(pool.map(_play, jobs, chunksize=4))
    else:
        home = os.getcwd()
        _quiet_cwd()
        try:
            results = [_play(j) for j in jobs]
        finally:
            os.chdir(home)
    rows = []
    for seats in ("as run", "swapped", "both"):
        part = []
        for i, (a, b) in enumerate(pairings):
            won = {(seed, sw): r for k, seed, sw, r in results
                   if k == i and seats in ("both", "swapped" if sw else "as run")}
            wa, wb = sum(r == 1 for r in won.values()), sum(r == 0 for r in won.values())
            lo, hi = wilson(wa, wa + wb)
            row = {"a": a, "b": b, "seats": seats, "games": len(won), "wins_a": wa, "wins_b": wb,
                   "no_winner": sum(r is None for r in won.values()),
                   "win_rate_a": wa / max(1, wa + wb), "lo": lo, "hi": hi, "p": mcnemar(wa, wb)}
            if seats == "both":            # per seed: which variant won in both seats
                pairs = [(won[(s, False)], won[(s, True)]) for s in range(games)]
                row.update(seeds_a_both=pairs.count((1, 1)), seeds_b_both=pairs.count((0, 0)),
                           p=mcnemar(pairs.count((1, 1)), pairs.count((0, 0))))
            part.append(row)
        for r, p in zip(part, holm([r["p"] for r in part])):
            r["p_holm"] = p
        rows += part
    return rows


def plot(agree, games, out):
    pairs = list(dict.fromkeys((r["a"], r["b"]) for r in agree))
    cats = list(dict.fromkeys(r["category"] for r in agree))
    val = {(r["a"], r["b"], r["category"]): r["agreement"] for r in agree}
    mat = np.array([[val[(a, b, c)] for c in cats] for a, b in pairs])
    fig, ax = figure(figsize=(1 + 0.65 * len(cats), 1 + 0.4 * len(pairs)))
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for i, row in enumerate(mat):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if v < 0.5 else "black")
    ax.set_xticks(range(len(cats)), cats, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(pairs)), [f"{a} vs {b}" for a, b in pairs], fontsize=7)
    fig.colorbar(im, label="share of spots where both choose the same distinct move")
    save_figure(fig, out, "search_spots_agreement")

    pairings = list(dict.fromkeys((r["a"], r["b"]) for r in games))
    fig, ax = figure(figsize=(6, 0.45 * len(pairings) + 1.2))
    for k, (seats, color) in enumerate([("as run", "#DD8452"), ("swapped", "#55A868"),
                                        ("both", "#4C72B0")]):
        rs = [r for r in games if r["seats"] == seats]
        y = np.arange(len(rs)) + (k - 1) * 0.22
        ax.errorbar([r["win_rate_a"] for r in rs], y, fmt="o", capsize=2, color=color, markersize=4,
                    label=seats, xerr=[[r["win_rate_a"] - r["lo"] for r in rs],
                                       [r["hi"] - r["win_rate_a"] for r in rs]])
    ax.axvline(0.5, color="0.5", linewidth=0.8)
    n = max(r["games"] for r in games if r["seats"] == "both")
    ax.set_yticks(range(len(pairings)), [f"{a} vs {b}" for a, b in pairings], fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"win rate of the first variant (95% Wilson interval; both = {n} games)")
    ax.legend(fontsize=7, loc="lower right")
    save_figure(fig, out, "search_games")


def run(out, games=200, workers=1, spots=None):
    specs = variants()
    write_csv(out, "search_variants", [{"variant": n, "depth": d, **(w or WEIGHTS)}
                                       for n, (d, w) in specs.items()])
    agree = spot_agreement(spot_choices(specs, spots or load_spots()), spot_pairs(list(specs)))
    # depth 2 against itself is the control: identical play, so its win rate is the
    # seat and dice split of these seeds that the other pairings depart from.
    pairings = [(REFERENCE, REFERENCE), ("depth 2", "depth 1"), ("depth 3", "depth 2"),
                ("depth 3", "depth 1")] + [(REFERENCE, n) for n in specs if not n.startswith("depth")]
    games_rows = head_to_head(specs, pairings, games, workers)
    write_csv(out, "search_spots_agreement", agree)
    write_csv(out, "search_games", games_rows)
    plot(agree, games_rows, out)
    return agree, games_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("--games", type=int, default=200, help="games per pairing (default 200)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--out", default=os.path.join(REPO, "results", "analysis"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    agree, games = run(os.path.abspath(args.out), args.games, args.workers)
    for r in agree:
        if r["category"] == "all":
            print(f"{r['a']} vs {r['b']}: same move on {r['agreement']:.2f} of {r['spots']} spots")
    for r in games:
        print(f"{r['a']} vs {r['b']} ({r['seats']}): {r['wins_a']}-{r['wins_b']} "
              f"(win rate {r['win_rate_a']:.2f}, Holm p {r['p_holm']:.3g})")


if __name__ == "__main__":
    main()
