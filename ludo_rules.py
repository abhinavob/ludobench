"""
Ludo rules: board constants, movement, legal moves, captures and move outcomes.

Every component (engine, agents, search players, prompt, logs, board renderer,
spot evaluation) takes its rules from here.

Positions: -1 is the base, 0-51 the shared track, and player p's home path is
52 + 6p ... 57 + 6p, whose last cell is that player's home end.

A piece's route is its relative squares 0-50 (its start square, then on to the
square beside its home path on the standard board), then home cells 1-6, so
relative 50 + d is home cell d and the home end is 56 pips from the start. A
piece never stops on its own relative 51 (the square just behind its start),
which other players' pieces pass through.

Each function takes original=False. original=True gives the movement of the
released LudoBench code: after its lap a piece passes its start square once
more before the home path (58 pips from start to home end, and a 1 from
relative 51 is unusable), and inside the home path it moves only by the exact
roll to the home end.
"""

BOARD_SIZE = 52
HOME_START = 52
HOME_LEN = 6
TOKENS_PER_PLAYER = 4
BASE = -1
START_POSITIONS = (0, 13, 26, 39)           # by player id
SAFE_SQUARES = frozenset({0, 8, 13, 21, 26, 34, 39, 47})


def seats(num_players, original=False):
    """Player ids in a game, in turn order. Two players sit opposite each other."""
    if num_players == 2 and not original:
        return [0, 2]
    return list(range(num_players))


def home_range(player):
    """(first home-path cell, home end) of a player."""
    home_start = HOME_START + player * HOME_LEN
    return home_start, home_start + HOME_LEN - 1


def track_length(original=False):
    """Track squares on a piece's route: relative 0-50, or 0-51 with original rules."""
    return BOARD_SIZE if original else BOARD_SIZE - 1


def _entry(original):
    """Relative step at which home cell k is reached is _entry + k."""
    return BOARD_SIZE if original else BOARD_SIZE - 2


def route_index(player, pos, original=False):
    """Pips from the player's start square to pos (None in the base)."""
    if pos == BASE:
        return None
    if pos < BOARD_SIZE:
        return (pos - START_POSITIONS[player]) % BOARD_SIZE
    return _entry(original) + pos - home_range(player)[0] + 1


def destination(player, pos, dice, original=False):
    """Where a piece of `player` on `pos` lands with `dice`, or None if it cannot move."""
    home_start, home_end = home_range(player)
    if pos == BASE:
        return START_POSITIONS[player] if dice == 6 else None
    if pos < BOARD_SIZE:
        step = (pos - START_POSITIONS[player]) % BOARD_SIZE + dice
        if step < track_length(original):
            return (START_POSITIONS[player] + step) % BOARD_SIZE
        cell = step - _entry(original)              # home cell 1..HOME_LEN
        return home_start + cell - 1 if 1 <= cell <= HOME_LEN else None
    new = pos + dice
    if new > home_end or (original and new != home_end):
        return None
    return new


def captures(tokens, player, square):
    """(opponent, token index) pairs a piece landing on `square` sends to base."""
    if square is None or not 0 <= square < BOARD_SIZE or square in SAFE_SQUARES:
        return []
    return [(opp, i) for opp, toks in tokens.items() if opp != player
            for i, pos in enumerate(toks) if pos == square]


def legal_moves(tokens, player, dice, original=False):
    """Indices of the player's pieces that can use this roll, ascending."""
    own = tokens[player]
    legal = []
    for i, pos in enumerate(own):
        dest = destination(player, pos, dice, original)
        if dest is None:
            continue
        if dest < BOARD_SIZE and dest not in SAFE_SQUARES and dest in own:
            continue                                # own piece on a non-safe track square
        legal.append(i)
    return legal


def move_outcome(tokens, player, token_idx, dice, original=False):
    """What moving this piece does, read from the position before the move."""
    old = tokens[player][token_idx]
    to = destination(player, old, dice, original)
    return {"from": old, "to": to, "captures": captures(tokens, player, to),
            "bring_out": old == BASE, "safe": to in SAFE_SQUARES,
            "in_home": to >= BOARD_SIZE, "home_finish": to == home_range(player)[1]}


def apply_move(tokens, player, token_idx, dice, original=False):
    """Board after the move (a new dict of lists) and the captured (opponent, index) pairs."""
    to = destination(player, tokens[player][token_idx], dice, original)
    caught = captures(tokens, player, to)
    after = {p: list(t) for p, t in tokens.items()}
    for opp, i in caught:
        after[opp][i] = BASE
    after[player][token_idx] = to
    return after, caught


def has_won(tokens, player):
    """All of the player's pieces are on its home end."""
    return all(pos == home_range(player)[1] for pos in tokens[player])
