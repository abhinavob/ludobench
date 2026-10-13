"""
Tables for the full-game runs in llm_game_results/.

Reads every completed run (run.json, games.jsonl, decisions.jsonl), groups runs
that share a player, opponent and prompt settings, and prints for each group:
wins by seat with a 95% Wilson interval, the invalid reply rate, and the
decision diagnostics with their chance levels (first-listed piece, agreement
with the heuristic player, captures, safe squares and bring-outs taken when
another kind of move was also legal).

Runs with fewer than --min-games completed games are skipped, as are folders
named _excluded.

Run from the repo root:  python analysis/full_game_summary.py
"""
import argparse
import collections
import glob
import json
import math
import os


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def fisher(a, b, c, d):
    """Two-sided Fisher's exact test for the table [[a, b], [c, d]]."""
    def logc(n, k):
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d

    def p(x):
        return math.exp(logc(r1, x) + logc(r2, c1 - x) - logc(n, c1))
    p0 = p(a)
    return min(1.0, sum(p(x) for x in range(max(0, c1 - r2), min(r1, c1) + 1)
                        if p(x) <= p0 * (1 + 1e-9)))


def short(spec):
    return spec.replace("ollama_native:", "")


def condition(run):
    s = run["settings"]
    cmd = " ".join(run["command_line"])
    if not s.get("structured_output"):
        return "-"
    parts = ["limit" if s.get("reason_max_chars") else "nolimit"]
    hist = s.get("last_moves") or (5 if "--last-moves" in cmd else 0)
    if hist:
        parts.append(f"hist{hist}")
    if s.get("move_outcomes"):
        parts.append("outcomes")
    if s.get("legal_order", "ascending") != "ascending":
        parts.append(s["legal_order"])
    return "+".join(parts)


def load(root, min_games):
    groups = collections.defaultdict(lambda: {"seat": {}, "c": collections.Counter(), "runs": []})
    runs = []
    for d in glob.glob(os.path.join(root, "**", "run.json"), recursive=True):
        folder = os.path.dirname(d)
        if "_excluded" in folder:
            continue
        run = json.load(open(d, encoding="utf-8"))
        g = run.get("games", {})
        if g.get("completed", 0) < min_games or g.get("completed") != g.get("attempted"):
            continue
        if len(run["settings"]["specs"]) == 2:
            runs.append((-g["completed"], folder, run))
    # A seed that two runs of the same matchup and settings both played is
    # counted once, from the larger run (baseline players repeat exactly).
    seen = set()
    for _, folder, run in sorted(runs, key=lambda x: x[:2]):
        specs = run["settings"]["specs"]
        cond = condition(run)
        games = [json.loads(l) for l in open(os.path.join(folder, "games.jsonl"), encoding="utf-8")]
        fresh = {x["seed"] for x in games if (tuple(specs), cond, x["seed"]) not in seen}
        seen |= {(tuple(specs), cond, sd) for sd in fresh}
        games = [x for x in games if x["seed"] in fresh]
        for pid in (0, 1):
            me, opp = specs[pid], specs[1 - pid]
            key = (short(me), short(opp), cond if me.startswith("ollama") else
                   ("vs " + cond if opp.startswith("ollama") else "-"))
            grp = groups[key]
            grp["runs"].append(os.path.basename(folder))
            w, n = grp["seat"].get(pid, (0, 0))
            grp["seat"][pid] = (w + sum(x["winner"] == pid for x in games), n + len(games))
            c = grp["c"]
            for x in games:
                pp = x["per_player"][str(pid)]
                c["games"] += 1
                c["cap_made"] += pp.get("captures_made", 0)
                c["cap_suffered"] += pp.get("times_captured", 0)
        for line in open(os.path.join(folder, "decisions.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            if r["skipped"] or r["seed"] not in fresh:
                continue
            pid = r["player_id"]
            me, opp = specs[pid], specs[1 - pid]
            key = (short(me), short(opp), cond if me.startswith("ollama") else
                   ("vs " + cond if opp.startswith("ollama") else "-"))
            c = groups[key]["c"]
            lm = r["legal_moves"]
            c["dec"] += 1
            if r.get("was_fallback"):
                c["invalid"] += 1
            if r.get("latency_ms") is not None:
                c["lat_n"] += 1
                c["lat"] += r["latency_ms"]
            if r.get("parsed_reason") and len(r["parsed_reason"]) >= 500:
                c["at_limit"] += 1
            if len(lm) < 2 or r.get("was_fallback"):
                continue
            ch = r["chosen_index"]
            c["n2"] += 1
            c["chance"] += 1 / len(lm)
            c["first"] += ch == lm[0]
            c["last"] += ch == lm[-1]
            hs = r["heuristic_score_per_legal_move"]
            best = [int(k) for k, v in hs.items() if v == max(hs.values())]
            c["heur"] += ch in best
            c["heur_ch"] += len(best) / len(lm)
            for kind in ("capture", "safe", "bring_out"):
                t = r["availability"][kind]["tokens"]
                if t and len(t) < len(lm):
                    c[kind + "_n"] += 1
                    c[kind + "_took"] += ch in t
                    c[kind + "_ch"] += len(t) / len(lm)
    return groups


def pct(a, b):
    return f"{100 * a / b:5.1f}" if b else "    -"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="llm_game_results")
    ap.add_argument("--min-games", type=int, default=25)
    args = ap.parse_args()
    groups = load(args.root, args.min_games)

    print("WINS (P0 = moves first)")
    print(f"{'player':<13}{'opponent':<13}{'condition':<18}{'P0':>7}{'P1':>7}{'all':>8}"
          f"{'win%':>6}{'95% CI':>14}{'inv%':>6}{'s/move':>7}{'cap+':>6}{'cap-':>6}")
    for key in sorted(groups):
        grp, c = groups[key], groups[key]["c"]
        s0, s1 = grp["seat"].get(0, (0, 0)), grp["seat"].get(1, (0, 0))
        w, n = s0[0] + s1[0], s0[1] + s1[1]
        lo, hi = wilson(w, n)
        f = lambda s: f"{s[0]}/{s[1]}" if s[1] else "-"
        inv = pct(c["invalid"], c["dec"]) if c["lat_n"] else "    -"
        lat = f"{c['lat'] / c['lat_n'] / 1000:6.2f}" if c["lat_n"] else "     -"
        print(f"{key[0]:<13}{key[1]:<13}{key[2]:<18}{f(s0):>7}{f(s1):>7}{f((w, n)):>8}"
              f"{pct(w, n):>6}{f'[{lo:.0f}, {hi:.0f}]':>14}{inv:>6}{lat:>7}"
              f"{c['cap_made'] / max(1, c['games']):>6.2f}{c['cap_suffered'] / max(1, c['games']):>6.2f}")

    print("\nDECISIONS with two or more legal moves, valid replies only (value / chance)")
    print(f"{'player':<13}{'opponent':<13}{'condition':<18}{'n':>6}{'first':>12}{'last':>6}"
          f"{'heuristic':>13}{'capture (n)':>18}{'safe (n)':>18}{'bring-out (n)':>18}{'limit%':>7}")
    for key in sorted(groups):
        c = groups[key]["c"]
        if not c["n2"]:
            continue
        k = lambda x: f"{pct(c[x + '_took'], c[x + '_n']).strip()}/{pct(c[x + '_ch'], c[x + '_n']).strip()} ({c[x + '_n']})"
        print(f"{key[0]:<13}{key[1]:<13}{key[2]:<18}{c['n2']:>6}"
              f"{pct(c['first'], c['n2']).strip() + '/' + pct(c['chance'], c['n2']).strip():>12}"
              f"{pct(c['last'], c['n2']):>6}"
              f"{pct(c['heur'], c['n2']).strip() + '/' + pct(c['heur_ch'], c['n2']).strip():>13}"
              f"{k('capture'):>18}{k('safe'):>18}{k('bring_out'):>18}"
              f"{pct(c['at_limit'], c['dec']):>7}")

    print("\nFISHER'S EXACT TEST on wins against the heuristic player (two-sided)")
    def wl(key):
        s = groups[key]["seat"]
        w = sum(v[0] for v in s.values())
        n = sum(v[1] for v in s.values())
        return w, n - w
    refs = [k for k in groups if k[1] == "heuristic" and k[0] in ("random", "first")]
    for key in sorted(groups):
        if key[1] != "heuristic" or key in refs or key[2] == "-" and key[0] not in ("gt:2", "gt:4"):
            continue
        a, b = wl(key)
        out = [f"{key[0]:<13}{key[2]:<18}{a}/{a + b}"]
        for ref in sorted(refs):
            c_, d_ = wl(ref)
            out.append(f"vs {ref[0]} ({c_}/{c_ + d_}): p={fisher(a, b, c_, d_):.3f}")
        base = (key[0], "heuristic", "limit")
        if key[2] not in ("limit", "-") and base in groups:
            c_, d_ = wl(base)
            out.append(f"vs own main condition ({c_}/{c_ + d_}): p={fisher(a, b, c_, d_):.3f}")
        print("  ".join(out))


if __name__ == "__main__":
    main()
