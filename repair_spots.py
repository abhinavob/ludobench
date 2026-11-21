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
    python repair_spots.py --review spots_40_final/review.json --out spots_40_final
    python repair_spots.py --check spots_40_final

changes.json lists every repair and exclusion (cause, fix, check result) and
the spots kept as they are but flagged: a single legal move, or pieces of two
players on one non-safe square.

--review builds the reviewed copy: each spot as repaired here, except that a
spot whose repair the review rejected gets the review's fix (pieces moved and
dice set, from the original spot) instead. Its changes.json logs every spot
that differs from the original, with the full check and what run_spots.py
derives from the position. --check runs the full check (audit) on every spot
of a folder against the original spots and lists those that fail.
"""
import argparse
import copy
import glob
import itertools
import json
import os
from collections import Counter

import ludo_rules as rules
import run_spots
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


def attackers(tokens, players, player, i, original=False):
    """The (opponent, piece) pairs whose next roll can capture this piece, by the game's rules."""
    pos = tokens[player][i]
    return {(q, k) for q in players if q != player for d in range(1, 7)
            for k in rules.legal_moves(tokens, q, d, original)
            if rules.destination(q, tokens[q][k], d, original) == pos
            and (player, i) in rules.captures(tokens, q, pos)}


def exposure(tokens, players, player, dice, original=False):
    """Legal move -> whether each of the mover's pieces can be captured next turn after it."""
    out = {}
    for m in rules.legal_moves(tokens, player, dice, original):
        after, _ = rules.apply_move(tokens, player, m, dice, original)
        out[m] = [bool(attackers(after, players, player, k, original)) for k in range(len(after[player]))]
    return out


def threats(tokens, players, original=False):
    """(attacker, target) pairs of (player, piece) in the position as given."""
    return {(a, (q, i)) for q in players for i in range(len(tokens[q]))
            for a in attackers(tokens, players, q, i, original)}


def shared_by_colours(tokens):
    """Non-safe track squares holding pieces of more than one player."""
    owners = {}
    for q, t in tokens.items():
        for pos in t:
            if 0 <= pos < rules.BOARD_SIZE and pos not in rules.SAFE_SQUARES:
                owners.setdefault(pos, set()).add(q)
    return sorted(s for s, qs in owners.items() if len(qs) > 1)


def audit(original, spot):
    """
    The full check: why `spot` under our rules does not mean what `original`
    meant under the original rules ([] if it does). Every piece of the mover
    keeps its outcome type (as in check()); after each legal move, each of the
    mover's pieces, moved or not, can be captured next turn as before; no
    threat between two pieces flips direction; no piece starts on a safe square
    unless it did originally; no two players share a non-safe square.
    """
    p, players = spot["current_player"], spot["players"]
    a, b = board(original), board(spot)
    why = [f"P{q} piece {i} is off our route" for q, i in off_route(b)]
    ref, ours = pieces(a, players, p, original["dice"], True), pieces(b, players, p, spot["dice"], False)
    why += [f"piece {i}: {describe(p, x)} originally, {describe(p, y)} now"
            for i, (x, y) in enumerate(zip(ref, ours)) if not same(x, y)]
    before, now = exposure(a, players, p, original["dice"], True), exposure(b, players, p, spot["dice"], False)
    why += [f"after piece {m} moves, piece {k} {'can' if y else 'cannot'} be captured next turn "
            f"({'could' if x else 'could not'} originally)"
            for m in sorted(before.keys() & now.keys())
            for k, (x, y) in enumerate(zip(before[m], now[m])) if x != y]
    then, now = threats(a, players, True), threats(b, players, False)
    why += [f"P{x[0]} piece {x[1]} could capture P{y[0]} piece {y[1]}; now the reverse"
            for x, y in sorted(then) if (y, x) in now and (x, y) not in now]
    why += [f"P{q} piece {i} starts on safe square {pos} (originally on {where(q, a[q][i])})"
            for q, t in sorted(b.items()) for i, pos in enumerate(t)
            if pos in rules.SAFE_SQUARES and a[q][i] not in rules.SAFE_SQUARES]
    why += [f"two players share non-safe square {s}" for s in shared_by_colours(b)]
    return why


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


def read_spots(src):
    """File name -> (its spots, its line ending) for each spot file of src."""
    files = {}
    for path in sorted(glob.glob(os.path.join(src, "spots_*.json"))):
        with open(path, encoding="utf-8", newline="") as f:
            raw = f.read()
        newline = "\r\n" if "\r\n" in raw else "\n"
        spots = json.loads(raw)
        assert json.dumps(spots, indent=2).replace("\n", newline) == raw, f"{path}: unexpected layout"
        files[os.path.basename(path)] = (spots, newline)
    return files


def write_spots(out, files, log):
    """Write the spot files (laid out as read) and the change log to out."""
    os.makedirs(out, exist_ok=True)
    for name, (kept, newline) in files.items():
        with open(os.path.join(out, name), "w", encoding="utf-8", newline="") as f:
            f.write(json.dumps([s for s in kept if s is not None], indent=2).replace("\n", newline))
    with open(os.path.join(out, "changes.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(write_log(log))


def build(src, out):
    """Write the repaired copy of src to out; return the change-log entries."""
    entries, files = [], {}
    for name, (spots, newline) in read_spots(src).items():
        kept = []
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
    counts = {}
    for e in entries:
        counts.setdefault(e["category"], dict.fromkeys(("unchanged", "repaired", "excluded"), 0))
        counts[e["category"]][e["status"]] += 1
    log = {"about": f"Every spot of {os.path.basename(os.path.normpath(src))}/ that was repaired, "
                    f"left out or flagged; see repair_spots.py.",
           "counts": dict(sorted(counts.items())),
           "spots": [e for e in entries if e["status"] != "unchanged" or e["flags"]]}
    write_spots(out, files, log)
    return entries, log


def reviewed(spot, fix):
    """The original spot with a review's fix: the pieces moved (from where they are) and the dice set."""
    tokens = board(spot)
    for m in fix["moved"]:
        if tokens[m["player"]][m["piece"]] != m["from"]:
            raise ValueError(f"{spot['id']}: P{m['player']} piece {m['piece']} is not on {m['from']}")
        tokens[m["player"]][m["piece"]] = m["to"]
    return dict(spot, dice=fix.get("dice", spot["dice"]), tokens={str(q): t for q, t in tokens.items()})


def moves_between(spot, new):
    """The pieces that stand elsewhere in new than in spot."""
    a, b = board(spot), board(new)
    return [{"player": q, "piece": i, "from": x, "to": y}
            for q in a for i, (x, y) in enumerate(zip(a[q], b[q])) if x != y]


def build_final(src, review_path, out):
    """
    Write the reviewed copy of src to out: each spot as review() repairs it,
    or, where the review rejected that repair, with the review's fix. Every
    spot is kept. Return the change log: one entry per spot that differs from
    src, with the full check and what run_spots.py derives from it.
    """
    with open(review_path, encoding="utf-8") as f:
        decisions = json.load(f)["spots"]
    entries, files, counts, boards = [], {}, {}, {}
    for name, (spots, newline) in read_spots(src).items():
        kept = []
        for spot in spots:
            decision = decisions.get(spot["id"], {"decision": "keep"})
            _, repaired = review(spot)
            if decision["decision"] == "reject":
                new = reviewed(spot, decision["fix"])
            elif repaired is None:
                raise ValueError(f"{spot['id']}: kept, but repair_spots.py leaves it out")
            else:
                new = repaired
            kept.append(new)
            boards[spot["id"]] = (new["dice"], new["tokens"])
            source = "unchanged" if new == spot else "review" if decision["decision"] == "reject" else "repair"
            counts.setdefault(spot["scenario"], dict.fromkeys(("unchanged", "repair", "review"), 0))
            counts[spot["scenario"]][source] += 1
            if source == "unchanged":
                continue
            p, players = new["current_player"], new["players"]
            entries.append({
                "id": spot["id"], "file": name, "category": spot["scenario"], "source": source,
                "note": decision.get("note"), "applied": decision.get("applied"),
                "dice": {"from": spot["dice"], "to": new["dice"]} if new["dice"] != spot["dice"] else None,
                "moved": moves_between(spot, new), "same_as_repaired": new == repaired,
                "check": audit(spot, new),
                "outcomes": [describe(p, x) for x in pieces(board(new), players, p, new["dice"], False)],
                "derived": run_spots.derived(new),
                "derived_repaired": None if new == repaired else run_spots.derived(repaired),
                "original_tokens": spot["tokens"], "final_tokens": new["tokens"]})
        files[name] = (kept, newline)
    for sid, b in boards.items():                     # the halves of a grudge pair share a board
        if sid.startswith("grudge"):
            assert boards[sid[:-1] + ("b" if sid.endswith("a") else "a")] == b, sid
    log = {"about": f"Every spot of {os.path.basename(os.path.normpath(src))}/ that differs here: "
                    f"source 'repair' (as repaired by repair_spots.py) or 'review' (the fix in "
                    f"{os.path.basename(review_path)}); see repair_spots.py --review.",
           "counts": dict(sorted(counts.items())), "spots": entries}
    write_spots(out, files, log)
    return log


def check_folder(src, folder):
    """Spot id -> audit() failures, for every spot of folder that fails against src."""
    original = {s["id"]: s for spots, _ in read_spots(src).values() for s in spots}
    return {s["id"]: why for spots, _ in read_spots(folder).values() for s in spots
            if (why := audit(original[s["id"]], s))}


def write_log(log):
    """The change log as JSON with one category or one spot per line."""
    rows = lambda items: ",\n".join(f"  {x}" for x in items)
    return (f'{{\n "about": {json.dumps(log["about"])},\n "counts": {{\n'
            + rows(f"{json.dumps(k)}: {json.dumps(v)}" for k, v in log["counts"].items())
            + '\n },\n "spots": [\n' + rows(json.dumps(e) for e in log["spots"]) + "\n ]\n}\n")


def print_counts(counts):
    columns = list(next(iter(counts.values())))
    total = Counter()
    print(f"{'category':<26}" + "".join(f"{c:>10}" for c in columns))
    for cat, n in counts.items():
        total.update(n)
        print(f"{cat:<26}" + "".join(f"{n[c]:>10}" for c in columns))
    print(f"{'total':<26}" + "".join(f"{total[c]:>10}" for c in columns))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("--src", default="spots_40")
    ap.add_argument("--out", help="output folder (default spots_40_repaired; spots_40_final with --review)")
    ap.add_argument("--review", metavar="FILE",
                    help="build the reviewed copy with the decisions and fixes in FILE")
    ap.add_argument("--check", metavar="FOLDER",
                    help="run the full check on every spot of FOLDER against --src; list the failures")
    args = ap.parse_args()
    if args.check:
        failing = check_folder(args.src, args.check)
        for sid, why in failing.items():
            print(f"{sid}: " + "; ".join(why))
        print(f"{len(failing)} spot(s) of {args.check} fail the full check")
    elif args.review:
        print_counts(build_final(args.src, args.review, args.out or "spots_40_final")["counts"])
    else:
        _, log = build(args.src, args.out or "spots_40_repaired")
        print_counts(log["counts"])


if __name__ == "__main__":
    main()
