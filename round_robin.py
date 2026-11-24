"""
Round robins between any players (LLMs from any backend, or bots). Every group
of --players from --pool plays --games games on seeds 0..games-1; game g uses
the (g mod k)-th of the group's k cyclic seatings (its first player in seat 0,
the others in every order), so each seating gets games/k games and every group
the same schedule. Each game's first player is chosen by the roll-off.

    python round_robin.py --players 2 --games 100 --experiment h2h-2p --pool A B C
    python round_robin.py --players 4 --games 48 --experiment h2h-4p --pool A B C D E

A pair is one run, seeds 0..games-1, in results/<experiment>/<A>_<B>/. Seating
k of a larger group is a run on seeds k, k + 6, ... (4 players) in
results/<experiment>/without-<players left out>/seating-<k>/. Every run.json
has the seat order (specs) and seeds (seed0, seed_step); games.jsonl has each
game's players by seat, the roll-off and the first player. --effort goes to
every LLM player whose model takes it, as in run_games.py (run.json records each
player's). --dry-run lists the runs without playing.
"""
import argparse
import itertools
import os
import sys

import run_games as rg


def schedule(pool, n, games):
    """[(folder in the experiment, specs in seat order, seed0, seed_step, games)] of every group."""
    runs = []
    for group in itertools.combinations(range(len(pool)), n):
        seatings = [(group[0], *rest) for rest in itertools.permutations(group[1:])]
        k = len(seatings)
        if games % k:
            raise ValueError(f"--games must be a multiple of {k}, the seatings of {n} players")
        if k == 1:
            runs.append((rg.run_folder([pool[i] for i in group]), [pool[i] for i in group],
                         0, 1, games))
            continue
        left = [rg.folder_name(p) for i, p in enumerate(pool) if i not in group]
        name = "without-" + "_".join(left) if left else "all"
        runs += [(os.path.join(name, f"seating-{s}"), [pool[i] for i in seating], s, k, games // k)
                 for s, seating in enumerate(seatings)]
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("--pool", nargs="+", required=True, metavar="PLAYER",
                    help="players as in run_games.py --players, each at most once")
    ap.add_argument("--players", type=int, choices=(2, 3, 4), default=2,
                    help="players per game (default 2)")
    ap.add_argument("--games", type=int, required=True, help="games per pair or group")
    ap.add_argument("--experiment", required=True, help="results go to results/<experiment>/")
    ap.add_argument("--effort", choices=["low", "medium", "high"],
                    help="reasoning effort of the LLM players that take one, as in run_games.py")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true", help="list the runs, play nothing")
    args = ap.parse_args()
    if len(set(args.pool)) < len(args.pool) or len(args.pool) < args.players:
        ap.error(f"--pool needs at least {args.players} different players")
    try:
        runs = schedule(args.pool, args.players, args.games)
    except ValueError as e:
        ap.error(str(e))
    for folder, specs, seed0, step, n in runs:
        print(f"{folder}: {'  vs  '.join(specs)}; seeds {seed0}"
              + (f", {seed0 + step}, ..., {seed0 + (n - 1) * step}" if n > 1 else ""))
    if args.dry_run:
        return

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    ollama_versions = rg.check_llm_players(ap, dict(enumerate(args.pool)))
    if rg.git_state()["dirty"]:
        print("WARNING: tracked files have uncommitted changes (recorded in run.json)")
    for folder, specs, seed0, step, n in runs:
        out = os.path.join(rg.RESULTS, args.experiment, folder)
        print(f"\n{folder}: writing to {out}")
        try:
            records = rg.run_games(specs, games=n, seed0=seed0, seed_step=step, out_dir=out,
                                   effort=args.effort, max_steps=args.max_steps,
                                   ollama_versions=ollama_versions)
        except ValueError as e:                      # e.g. an effort a model cannot take
            raise SystemExit(str(e))
        rg.summarise(records, specs)


if __name__ == "__main__":
    main()
