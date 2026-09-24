"""
Play Ludo yourself against the GameTheory agent (or Heuristic/Random).

Usage (from the repo root):
    python play_vs_gt.py                  # you = P0, GT depth 2, 2 players
    python play_vs_gt.py --depth 3        # stronger GT
    python play_vs_gt.py --opponent heuristic
    python play_vs_gt.py --players 4      # you + 3 GT agents
    python play_vs_gt.py --you 1          # you play as P1 instead
    python play_vs_gt.py --plain          # no colour codes

Positions follow the repo's internal scheme:
    -1      = in base
    0..51   = main track
    >= 52   = that player's private home path (home_end = finished)
"""

import argparse
import sys

from ludo_env import LudoEnv
from agents import BaseAgent, RandomAgent, HeuristicAgent
from game_theory_multiplayer_agent import GameTheoryMultiplayerAgent


# --------------------------------------------------------------------------
# Board geometry: map the 52 main-track squares onto a 15x15 cross
# --------------------------------------------------------------------------

def _build_track():
    """Return 52 (row, col) coords; index i = main-track square i."""
    coords = []
    coords += [(6, c) for c in range(1, 6)]        # 0-4
    coords += [(r, 6) for r in range(5, -1, -1)]   # 5-10
    coords += [(0, 7)]                             # 11
    coords += [(r, 8) for r in range(0, 6)]        # 12-17
    coords += [(6, c) for c in range(9, 15)]       # 18-23
    coords += [(7, 14)]                            # 24
    coords += [(8, c) for c in range(14, 8, -1)]   # 25-30
    coords += [(r, 8) for r in range(9, 15)]       # 31-36
    coords += [(14, 7)]                            # 37
    coords += [(r, 6) for r in range(14, 8, -1)]   # 38-43
    coords += [(8, c) for c in range(5, -1, -1)]   # 44-49
    coords += [(7, 0)]                             # 50
    coords += [(6, 0)]                             # 51
    return coords


TRACK = _build_track()

HOME_COORDS = {
    0: [(7, c) for c in range(1, 7)],
    1: [(r, 7) for r in range(1, 7)],
    2: [(7, c) for c in range(13, 7, -1)],
    3: [(r, 7) for r in range(13, 7, -1)],
}

BASE_ANCHOR = {0: (10, 2), 1: (2, 2), 2: (2, 10), 3: (10, 10)}

COLOURS = {0: "\033[91m", 1: "\033[92m", 2: "\033[93m", 3: "\033[94m"}
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def render_board(env, you, use_colour=True):
    def paint(text, code):
        return f"{code}{text}{RESET}" if use_colour else text

    grid = [["   " for _ in range(15)] for _ in range(15)]

    for sq, (r, c) in enumerate(TRACK):
        grid[r][c] = paint(" ::", DIM) if sq in env.SAFE_SQUARES else paint("  .", DIM)

    for pid in env.player_ids:
        for r, c in HOME_COORDS[pid]:
            grid[r][c] = paint("  -", COLOURS[pid])
    grid[7][7] = paint(" ()", BOLD)

    occupants = {}
    for pid in env.player_ids:
        home_start = env.HOME_START + pid * env.HOME_LEN
        for t_idx, pos in enumerate(env.tokens[pid]):
            if pos == -1:
                continue
            if 0 <= pos < env.BOARD_SIZE:
                cell = TRACK[pos]
            else:
                step = pos - home_start
                cell = HOME_COORDS[pid][step] if step < env.HOME_LEN else (7, 7)
            occupants.setdefault(cell, []).append((pid, t_idx))

    for (r, c), here in occupants.items():
        pid, t_idx = here[0]
        label = f"{len(here)}x{pid}" if len(here) > 1 else f" {pid}{t_idx}"
        grid[r][c] = paint(label, COLOURS[pid] + BOLD)

    for pid in env.player_ids:
        r0, c0 = BASE_ANCHOR[pid]
        in_base = [i for i, p in enumerate(env.tokens[pid]) if p == -1]
        grid[r0][c0] = paint(f" P{pid}", COLOURS[pid] + BOLD)
        grid[r0 + 1][c0] = paint(f" x{len(in_base)}", COLOURS[pid])

    print()
    for row in grid:
        print("   " + "".join(row))
    print()

    for pid in env.player_ids:
        home_start = env.HOME_START + pid * env.HOME_LEN
        home_end = home_start + env.HOME_LEN - 1
        parts = []
        for pos in env.tokens[pid]:
            if pos == -1:
                parts.append("base")
            elif pos == home_end:
                parts.append("DONE")
            elif pos >= env.HOME_START:
                parts.append(f"h{pos - home_start}")
            else:
                parts.append(f"sq{pos}")
        tag = " (you)" if pid == you else ""
        line = f"   P{pid}{tag}: " + "  ".join(
            f"t{i}={v}" for i, v in enumerate(parts))
        print(paint(line, COLOURS[pid]))
    print(paint("\n   :: safe    - home path    () finish    nxP stacked", DIM))
    print()


def move_outcome(env, player, dice, token_idx):
    """Predict landing square. Mirrors ludo_env.apply_move arithmetic."""
    start = env.START_POSITIONS[player]
    home_start = env.HOME_START + player * env.HOME_LEN
    home_end = home_start + env.HOME_LEN - 1
    pos = env.tokens[player][token_idx]

    if pos == -1:
        return start, "leave base", home_end
    if 0 <= pos < env.BOARD_SIZE:
        rel = (pos - start) % env.BOARD_SIZE
        new_rel = rel + dice
        if new_rel < env.BOARD_SIZE:
            new_pos = (start + new_rel) % env.BOARD_SIZE
            return new_pos, f"sq{pos} -> sq{new_pos}", home_end
        step = new_rel - env.BOARD_SIZE
        new_pos = home_start + step - 1
        return new_pos, f"sq{pos} -> home+{new_pos - home_start}", home_end
    new_pos = pos + dice
    return new_pos, f"home+{pos - home_start} -> finish", home_end


def describe_move(env, player, dice, token_idx):
    new_pos, label, home_end = move_outcome(env, player, dice, token_idx)
    extras = []
    if new_pos == home_end:
        extras.append("FINISHES this token")
    if 0 <= new_pos < env.BOARD_SIZE:
        if new_pos in env.SAFE_SQUARES:
            extras.append("safe square")
        else:
            for opp in env.player_ids:
                if opp != player and new_pos in env.tokens[opp]:
                    extras.append(f"CAPTURES a P{opp} token")
    text = f"token {token_idx}: {label}"
    return text + ("   [" + ", ".join(extras) + "]" if extras else "")


def group_moves(env, player, dice, legal_moves):
    """Group moves with identical outcomes (e.g. any base token leaving)."""
    groups = {}
    for t in legal_moves:
        new_pos, _, _ = move_outcome(env, player, dice, t)
        key = (env.tokens[player][t] == -1, new_pos)
        groups.setdefault(key, []).append(t)
    return list(groups.values())


# --------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------

class HumanAgent(BaseAgent):
    def __init__(self, use_colour=True):
        self.use_colour = use_colour

    def select_move(self, env, player, dice, legal_moves):
        c = COLOURS[player] if self.use_colour else ""
        b = BOLD if self.use_colour else ""
        r = RESET if self.use_colour else ""
        print("\n" + "=" * 54)
        print(f"  {c}{b}YOUR TURN  (P{player}){r}     dice rolled: {dice}")
        render_board(env, player, self.use_colour)

        groups = group_moves(env, player, dice, legal_moves)
        print("  Your options:")
        for g in groups:
            primary = g[0]
            same = (f"   (tokens {','.join(map(str, g))} do the same thing)"
                    if len(g) > 1 else "")
            print(f"    [{primary}]  {describe_move(env, player, dice, primary)}{same}")
        print()

        while True:
            raw = input(f"  Choose token {legal_moves} (or 'q' to quit): ").strip()
            if raw.lower() in ("q", "quit", "exit"):
                print("\nBye.")
                sys.exit(0)
            try:
                choice = int(raw)
            except ValueError:
                print("  Enter a number.")
                continue
            if choice in legal_moves:
                return choice
            print(f"  {choice} is not legal here. Legal: {legal_moves}")


class VerboseAgent(BaseAgent):
    """Wraps any agent and narrates its move."""

    def __init__(self, inner, label, use_colour=True):
        self.inner = inner
        self.label = label
        self.use_colour = use_colour

    def select_move(self, env, player, dice, legal_moves):
        choice = self.inner.select_move(env, player, dice, legal_moves)
        c = COLOURS[player] if self.use_colour else ""
        r = RESET if self.use_colour else ""
        print(f"  {c}P{player} ({self.label}){r} rolled {dice}: "
              f"{describe_move(env, player, dice, choice)}")
        return choice


def make_opponent(name, depth):
    if name == "gt":
        return GameTheoryMultiplayerAgent(search_depth=depth), f"GT d={depth}"
    if name == "heuristic":
        return HeuristicAgent(), "heuristic"
    if name == "random":
        return RandomAgent(), "random"
    raise ValueError(f"unknown opponent: {name}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Play Ludo against the GT agent.")
    ap.add_argument("--opponent", default="gt", choices=["gt", "heuristic", "random"])
    ap.add_argument("--depth", type=int, default=2, help="GT search depth")
    ap.add_argument("--players", type=int, default=2, choices=[2, 3, 4])
    ap.add_argument("--you", type=int, default=0, help="which player id you control")
    ap.add_argument("--seed", type=int, default=42, help="dice seed")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--plain", action="store_true", help="disable ANSI colour")
    args = ap.parse_args()

    if args.you >= args.players:
        raise SystemExit(f"--you must be < --players ({args.players})")

    use_colour = not args.plain
    env = LudoEnv(seed=args.seed, num_players=args.players)
    _, label = make_opponent(args.opponent, args.depth)

    agents = {}
    for pid in env.player_ids:
        if pid == args.you:
            agents[pid] = HumanAgent(use_colour)
        else:
            opp, opp_label = make_opponent(args.opponent, args.depth)
            agents[pid] = VerboseAgent(opp, opp_label, use_colour)

    print(f"\nYou are Player {args.you}. Opponent(s): {label}. "
          f"{args.players}-player, seed {args.seed}.")
    print("Leave base only on a 6. Land exactly on the finish.\n")

    winner, steps = env.play_game(agents, max_steps=args.max_steps)

    print("=" * 54)
    render_board(env, args.you, use_colour)
    if winner is None:
        print(f"No winner after {steps} steps.")
    elif winner == args.you:
        print(f"You win, in {steps} steps.")
    else:
        print(f"P{winner} ({label}) wins, in {steps} steps.")


if __name__ == "__main__":
    main()
