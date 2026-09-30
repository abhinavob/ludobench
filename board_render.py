"""
Shared terminal board renderer for Ludo.

Used by run_games.py, for human play and --watch spectator mode, so there is
exactly one copy of the board geometry.

The engine (ludo_env.py) has no notion of geometry: it stores positions as
integers (-1 base, 0-51 main track, >=52 home path) and does modular arithmetic.
Everything here is presentation layered on top of that.
"""


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
        n_base = sum(1 for p in env.tokens[pid] if p == -1)
        grid[r0][c0] = paint(f" P{pid}", COLOURS[pid] + BOLD)
        grid[r0 + 1][c0] = paint(f" x{n_base}", COLOURS[pid])

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
        name = f" [{labels[pid]}]" if labels and pid in labels else ""
        tag = " <--" if pid == highlight else ""
        line = (f"   P{pid}{name}: "
                + "  ".join(f"t{i}={v}" for i, v in enumerate(parts)) + tag)
        print(paint(line, COLOURS[pid]))
    print(paint("\n   :: safe    - home path    () finish    nxP stacked", DIM))
    print()


def move_outcome(env, player, dice, token_idx):
    """Predict landing position. Mirrors ludo_env.apply_move arithmetic."""
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
    """Group legal moves with identical outcomes (e.g. any base token leaving)."""
    groups = {}
    for t in legal_moves:
        new_pos, _, _ = move_outcome(env, player, dice, t)
        key = (env.tokens[player][t] == -1, new_pos)
        groups.setdefault(key, []).append(t)
    return list(groups.values())
