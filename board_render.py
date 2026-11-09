"""
Shared terminal board renderer for Ludo.

Used by run_games.py, for human play and --watch spectator mode, so there is
exactly one copy of the board geometry.

The rules (ludo_rules.py) have no notion of geometry: they store positions as
integers (-1 base, 0-51 main track, >=52 home path). Everything here is
presentation layered on top of that.
"""
import ludo_rules as rules


def _build_track():
    """52 (row, col) coords on a 15x15 grid; index i = main-track square i."""
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

# Each player owns one arm of the cross.
HOME_COORDS = {
    0: [(7, c) for c in range(1, 7)],
    1: [(r, 7) for r in range(1, 7)],
    2: [(7, c) for c in range(13, 7, -1)],
    3: [(r, 7) for r in range(13, 7, -1)],
}

# Base corner sits beside the arm that player enters on.
BASE_ANCHOR = {0: (2, 2), 1: (2, 10), 2: (10, 10), 3: (10, 2)}

COLOURS = {0: "\033[91m", 1: "\033[92m", 2: "\033[93m", 3: "\033[94m"}
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def render_board(env, highlight=None, use_colour=True, labels=None):
    """
    Print the board. `highlight` is a player id to mark "(you)" / "<--".
    `labels` optionally maps player id -> a name shown on the status line.
    """
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
        home_start = rules.home_range(pid)[0]
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
        n_base = sum(1 for p in env.tokens[pid] if p == -1)
        grid[r0][c0] = paint(f" P{pid}", COLOURS[pid] + BOLD)
        grid[r0 + 1][c0] = paint(f" x{n_base}", COLOURS[pid])

    print()
    for row in grid:
        print("   " + "".join(row))
    print()

    for pid in env.player_ids:
        home_start, home_end = rules.home_range(pid)
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
        name = f" [{labels[pid]}]" if labels and pid in labels else ""
        tag = " <--" if pid == highlight else ""
        line = (f"   P{pid}{name}: "
                + "  ".join(f"t{i}={v}" for i, v in enumerate(parts)) + tag)
        print(paint(line, COLOURS[pid]))
    print(paint("\n   :: safe    - home path    () finish    nxP stacked", DIM))
    print()


def move_outcome(env, player, dice, token_idx):
    """Landing position of a legal move and a short label for it."""
    pos = env.tokens[player][token_idx]
    new_pos = rules.destination(player, pos, dice, env.original_rules)
    home_start, home_end = rules.home_range(player)

    def where(p):
        if p == home_end:
            return "finish"
        return f"sq{p}" if p < rules.BOARD_SIZE else f"home+{p - home_start}"
    return new_pos, ("leave base" if pos == -1 else f"{where(pos)} -> {where(new_pos)}")


def describe_move(env, player, dice, token_idx):
    _, label = move_outcome(env, player, dice, token_idx)
    out = rules.move_outcome(env.tokens, player, token_idx, dice, env.original_rules)
    extras = []
    if out["home_finish"]:
        extras.append("FINISHES this token")
    if out["safe"]:
        extras.append("safe square")
    for opp in dict.fromkeys(opp for opp, _ in out["captures"]):
        extras.append(f"CAPTURES a P{opp} token")
    text = f"token {token_idx}: {label}"
    return text + ("   [" + ", ".join(extras) + "]" if extras else "")


def group_moves(env, player, dice, legal_moves):
    """Group legal moves with identical outcomes (e.g. any base token leaving)."""
    groups = {}
    for t in legal_moves:
        new_pos, _ = move_outcome(env, player, dice, t)
        key = (env.tokens[player][t] == -1, new_pos)
        groups.setdefault(key, []).append(t)
    return list(groups.values())
