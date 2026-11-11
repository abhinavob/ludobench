"""
Independent reference rules for the tests.

Each player's route is written out as an explicit list of cells, and a piece is
walked along it one pip at a time. Nothing here is shared with ludo_rules.
"""
from functools import lru_cache

SAFE = {0, 8, 13, 21, 26, 34, 39, 47}
STARTS = {0: 0, 1: 13, 2: 26, 3: 39}
PASS = "start square passed again"          # original rules: a cell where no piece may stop


@lru_cache(maxsize=None)
def route(player, original=False):
    """
    Cells from the start square to the home end, in walking order. On the
    standard board a piece turns into its home column from the square two
    before its start; the original rules go round to the square just before
    it and pass the start square once more.
    """
    start = STARTS[player]
    track = [(start + i) % 52 for i in range(52 if original else 51)]
    home = [52 + 6 * player + k for k in range(6)]
    return tuple(track + ([PASS] if original else []) + home)


def walk(player, pos, dice, original=False):
    """Destination of one piece, or None if the roll cannot be used."""
    cells = route(player, original)
    if pos == -1:
        return cells[0] if dice == 6 else None
    i = cells.index(pos)
    for _ in range(dice):
        i += 1
        if i == len(cells):
            return None
    if cells[i] == PASS:
        return None
    if original and pos >= 52 and cells[i] != cells[-1]:
        return None
    return cells[i]


def legal(tokens, player, dice, original=False):
    out = []
    for i, pos in enumerate(tokens[player]):
        dest = walk(player, pos, dice, original)
        if dest is None:
            continue
        others = tokens[player][:i] + tokens[player][i + 1:]
        if dest < 52 and dest not in SAFE and dest in others:
            continue
        out.append(i)
    return out


def play(tokens, player, idx, dice, original=False):
    """Board after the move and the set of captured (player, index) pairs."""
    dest = walk(player, tokens[player][idx], dice, original)
    after = {p: list(t) for p, t in tokens.items()}
    caught = set()
    if dest < 52 and dest not in SAFE:
        for p in after:
            for j, pos in enumerate(after[p]):
                if p != player and pos == dest:
                    after[p][j] = -1
                    caught.add((p, j))
    after[player][idx] = dest
    return after, caught


def random_state(rng, players, original=False):
    """
    A board that rules allow: every piece on its own route, no own pieces
    together and no two players on one non-safe square.
    """
    while True:
        tokens = {}
        for p in players:
            track = route(p, original)[:52 if original else 51]
            tokens[p] = []
            for _ in range(4):
                r = rng.random()
                if r < 0.25:
                    tokens[p].append(-1)
                elif r < 0.4:
                    tokens[p].append(52 + 6 * p + rng.randrange(6))
                else:
                    tokens[p].append(rng.choice(track))
        crowded = [x for t in tokens.values() for x in t if 0 <= x < 52 and x not in SAFE]
        if len(crowded) == len(set(crowded)):
            return tokens


def seat_sets(rng):
    return rng.choice([[0, 1], [0, 2], [0, 1, 2], [0, 1, 2, 3]])

