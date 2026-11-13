"""
Make spots_40_repaired/ from spots_40/: the spots as they mean under our rules.

A spot is kept as it is when each of the mover's four pieces does the same
under our rules as in the original spot under the original rules: the same
pieces can move, each move has the same flags (bring out, safe, capture, home
entry, finish) and leaves the piece exposed to capture next turn or not alike,
and each piece that cannot move cannot for the same reason. A spot that fails,
or has a piece where our route never stops, is repaired by its cause: the
pieces involved (the off-route piece, a move that changed, the pieces it
captures or that can reach it) are moved back the fewest squares, together so
their distances stay the same, until the check passes. A spot with no such fix
is left out. spots_40/ is never changed.

    python repair_spots.py        # writes spots_40_repaired/ and its changes.json

changes.json lists every repair and exclusion (cause, fix, check result) and
the spots kept as they are but flagged: a single legal move, or pieces of two
players on one non-safe square.
"""
import argparse
import copy
import glob
import itertools
import json
import os
from collections import Counter

import ludo_rules as rules
from decision_log import is_exposed
from spot_evaluation import SpotEnv

FLAGS = ("bring_out", "safe", "capture", "home_entry", "home_finish")
MAX_PIECES, MAX_SQUARES = 3, 2           # a fix moves at most 3 pieces, each 1 or 2 squares back


def board(spot):
    return {int(p): list(t) for p, t in spot["tokens"].items()}


def where(player, pos):
    if pos == rules.BASE:
        return "base"
    if pos < rules.BOARD_SIZE:
        return f"square {pos}"
    home_start, home_end = rules.home_range(player)
    return "home end" if pos == home_end else f"home cell {pos - home_start + 1}"


def cannot_move(tokens, player, i, dice, original):
    """Why a piece cannot use the roll."""
    pos, end = tokens[player][i], rules.home_range(player)[1]
    if pos == end:
        return "finished"
    if pos == rules.BASE:
        return "in base"
    if rules.destination(player, pos, dice, original) is None:
        over = rules.route_index(player, pos, original) + dice > rules.route_index(player, end, original)
        return "overshoot" if over else "no move"
    return "blocked by own piece"


def pieces(tokens, players, player, dice, original):
    """What each of the mover's pieces does with the roll."""
    legal = rules.legal_moves(tokens, player, dice, original)
    out = []
    for i in range(len(tokens[player])):
        if i not in legal:
            out.append({"cannot_move": cannot_move(tokens, player, i, dice, original)})
            continue
        e = rules.move_outcome(tokens, player, i, dice, original)
        e["capture"] = bool(e["captures"])
        after, _ = rules.apply_move(tokens, player, i, dice, original)
        out.append({"to": e["to"], "flags": [f for f in FLAGS if e[f]],
                    "exposed": is_exposed(SpotEnv(players, after, original), player, e["to"])})
    return out


def same(a, b):
    """Two pieces() entries mean the same (where a move lands may differ)."""
    return {k: v for k, v in a.items() if k != "to"} == {k: v for k, v in b.items() if k != "to"}


def describe(player, p):
    if "cannot_move" in p:
        return f"cannot move ({p['cannot_move']})"
    kinds = ", ".join(p["flags"]) or "plain move"
    return f"to {where(player, p['to'])}: {kinds}" + (", exposed" if p["exposed"] else "")


def off_route(tokens):
    return [(q, i) for q, t in tokens.items() for i, pos in enumerate(t) if not rules.on_route(q, pos)]


def shared_squares(tokens):
    """Non-safe track squares holding more than one piece, which play never makes."""
    n = Counter(pos for t in tokens.values() for pos in t
                if 0 <= pos < rules.BOARD_SIZE and pos not in rules.SAFE_SQUARES)
    return {s for s, k in n.items() if k > 1}


def check(spot, tokens):
    """(passes, our pieces(), reasons it fails) for the spot with these tokens."""
    p, d, players = spot["current_player"], spot["dice"], spot["players"]
    ref = pieces(board(spot), players, p, d, True)
    ours = pieces(tokens, players, p, d, False)
    reasons = [f"P{q} piece {i} is off our route" for q, i in off_route(tokens)]
    if not shared_squares(tokens) <= shared_squares(board(spot)):
        reasons.append("two pieces on one non-safe square")
    reasons += [f"piece {i} differs" for i, (a, b) in enumerate(zip(ref, ours)) if not same(a, b)]
    return not reasons, ours, reasons


def cause(spot):
    """What makes the spot fail under our rules: sentences, and the pieces involved."""
    tokens, p, d, players = board(spot), spot["current_player"], spot["dice"], spot["players"]
    ref = pieces(tokens, players, p, d, True)
    ours = pieces(tokens, players, p, d, False)
    text, involved = [], set()
    for q, i in off_route(tokens):
        pos = tokens[q][i]
        text.append(f"P{q} piece {i} is on square {pos}, just behind its start "
                    f"({rules.START_POSITIONS[q]}): our pieces turn into the home path "
                    f"one square earlier, so they never stop there.")
        involved.add((q, i))
    for i, (a, b) in enumerate(zip(ref, ours)):
        if not same(a, b):
            text.append(f"Piece {i}: {describe(p, a)} under the original rules; "
                        f"{describe(p, b)} under ours.")
            involved.add((p, i))
    # Pieces whose squares matter to those: what they land on or capture, and
    # opponents that can reach the involved squares in one roll.
    squares = {tokens[q][i] for q, i in involved}
    squares |= {x.get("to") for q, i in involved if q == p for x in (ref[i], ours[i])}
    squares = {s for s in squares if s is not None and 0 <= s < rules.BOARD_SIZE}
    for q in players:
        for i, pos in enumerate(tokens[q]):
            lands = {rules.destination(q, pos, r, o) for r in range(1, 7) for o in (True, False)}
            if 0 <= pos < rules.BOARD_SIZE and (pos in squares or lands & squares):
                involved.add((q, i))
    return text, sorted(involved)


def fix(spot):
    """The smallest move of the involved pieces back along their routes that passes the check."""
    tokens = board(spot)
    _, involved = cause(spot)
    off = set(off_route(tokens))
    movable = [(q, i) for q, i in involved if 0 <= tokens[q][i] < rules.BOARD_SIZE]
    tries = []
    for n in range(1, MAX_PIECES + 1):
        for group in itertools.combinations(movable, n):
            if off <= set(group):
                tries += [(n * k, n, k, group) for k in range(1, MAX_SQUARES + 1)]
    for _, n, k, group in sorted(tries):
        new = copy.deepcopy(tokens)
        for q, i in group:
            if rules.route_index(q, new[q][i]) < k:
                break                                   # would go back past its start
            new[q][i] = (new[q][i] - k) % rules.BOARD_SIZE
        else:
            if check(spot, new)[0]:
                moved = [{"player": q, "piece": i, "from": tokens[q][i], "to": new[q][i]}
                         for q, i in group]
                text = "; ".join(f"P{m['player']} piece {m['piece']}: square {m['from']} -> "
                                 f"{m['to']}" for m in moved)
                text += f" ({k} square{'s' * (k > 1)} back"
                text += ", together so the distances between them stay the same)" if n > 1 else ")"
                return new, {"moved": moved, "squares_back": k, "text": text}
    return None, None


def review(spot):
    """The change-log entry and the spot to write (None if left out)."""
    tokens, p, d = board(spot), spot["current_player"], spot["dice"]
    ref = pieces(tokens, spot["players"], p, d, True)
    passes, ours, reasons = check(spot, tokens)
    flags = []
    if len(rules.legal_moves(tokens, p, d)) == 1:
        flags.append("single legal move")
    if shared_squares(tokens):
        flags.append("pieces of two players on one non-safe square")
    entry = {"id": spot["id"], "category": spot["scenario"], "status": "unchanged",
             "flags": flags, "cause": None, "fix": None,
             "check": {"passed": passes, "failed_on": reasons,
                       "original_rules": [describe(p, x) for x in ref],
                       "ours_before_fix": [describe(p, x) for x in ours], "ours_after_fix": None},
             "original_tokens": spot["tokens"], "repaired_tokens": None}
    if passes:
        return entry, spot
    entry["cause"], _ = cause(spot)
    new, how = fix(spot)
    if new is None:
        entry["status"] = "excluded"
        return entry, None
    repaired = dict(spot, tokens={str(q): t for q, t in new.items()})
    entry.update(status="repaired", fix=how, repaired_tokens=repaired["tokens"])
    entry["check"].update(passed=True, ours_after_fix=[describe(p, x) for x in check(spot, new)[1]])
    return entry, repaired


def build(src, out):
    """Write the repaired copy of src to out; return the change-log entries."""
    entries, files = [], {}
    for path in sorted(glob.glob(os.path.join(src, "spots_*.json"))):
        with open(path, encoding="utf-8", newline="") as f:
            raw = f.read()
        newline = "\r\n" if "\r\n" in raw else "\n"
        spots = json.loads(raw)
        assert json.dumps(spots, indent=2).replace("\n", newline) == raw, f"{path}: unexpected layout"
        name, kept = os.path.basename(path), []
        for spot in spots:
            entry, spot = review(spot)
            entries.append({"id": entry["id"], "file": name, **entry})
            kept.append(spot)
        files[name] = (kept, newline)
    # The two halves of a grudge pair share a board, so they must fare alike.
    by_id = {e["id"]: e for e in entries}
    for e in entries:
        if e["category"] == "grudge":
            twin = by_id[e["id"][:-1] + ("b" if e["id"].endswith("a") else "a")]
            assert (twin["status"], twin["fix"]) == (e["status"], e["fix"]), e["id"]
    os.makedirs(out, exist_ok=True)
    for name, (kept, newline) in files.items():
        with open(os.path.join(out, name), "w", encoding="utf-8", newline="") as f:
            f.write(json.dumps([s for s in kept if s is not None], indent=2).replace("\n", newline))
    counts = {}
    for e in entries:
        counts.setdefault(e["category"], dict.fromkeys(("unchanged", "repaired", "excluded"), 0))
        counts[e["category"]][e["status"]] += 1
    log = {"about": f"Every spot of {os.path.basename(os.path.normpath(src))}/ that was repaired, "
                    f"left out or flagged; see repair_spots.py.",
           "counts": dict(sorted(counts.items())),
           "spots": [e for e in entries if e["status"] != "unchanged" or e["flags"]]}
    with open(os.path.join(out, "changes.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(write_log(log))
    return entries, log


def write_log(log):
    """The change log as JSON with one category or one spot per line."""
    rows = lambda items: ",\n".join(f"  {x}" for x in items)
    return (f'{{\n "about": {json.dumps(log["about"])},\n "counts": {{\n'
            + rows(f"{json.dumps(k)}: {json.dumps(v)}" for k, v in log["counts"].items())
            + '\n },\n "spots": [\n' + rows(json.dumps(e) for e in log["spots"]) + "\n ]\n}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("--src", default="spots_40")
    ap.add_argument("--out", default="spots_40_repaired")
    args = ap.parse_args()
    _, log = build(args.src, args.out)
    total = Counter()
    print(f"{'category':<26}{'unchanged':>10}{'repaired':>10}{'excluded':>10}")
    for cat, n in log["counts"].items():
        total.update(n)
        print(f"{cat:<26}{n['unchanged']:>10}{n['repaired']:>10}{n['excluded']:>10}")
    print(f"{'total':<26}{total['unchanged']:>10}{total['repaired']:>10}{total['excluded']:>10}")


if __name__ == "__main__":
    main()
