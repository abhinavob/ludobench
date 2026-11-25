"""
Replay logged runs through the engine of the commit that made them, with the
logged moves, and check every roll: dice, player, legal moves and the board
before and after; for a game also the roll-off, first player, winner, final
board and roll count; for a spot run the spot's board, dice and player.
Read-only on the run folders; each commit's code is taken with git archive.

    python analysis/verify_runs.py results
    python analysis/verify_runs.py --unknown-commit 0261653 path/to/copied/results

A run whose run.json records no commit (e.g. one copied from another machine)
is skipped unless --unknown-commit names the commit to replay it with. Prints
one line per run and exits 1 if any run has a problem.
"""
import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Null:
    def write(self, *_): pass
    def flush(self): pass
    def close(self): pass


class _Script:
    """An agent that plays the logged move."""
    move = None

    def select_move(self, env, player, dice, legal_moves):
        return self.move


def board(tokens):
    return {str(p): list(t) for p, t in tokens.items()}


def lines(path):
    """The complete JSON lines of a file (a live run may be mid-line)."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.endswith("\n"):
                yield json.loads(line)


def check_game(L, n_players, g, recs):
    """The first problem in one game's records, or None. g is its game record (None if unfinished)."""
    env = L.LudoEnv(seed=g["seed"] if g else recs[0]["seed"], num_players=n_players)
    for f in [env.log_file, env.llm_log_file, *env.player_logs.values()]:
        f.close()
    env.log_file = env.llm_log_file = _Null()
    env.player_logs = {p: _Null() for p in env.player_ids}
    if g and json.loads(json.dumps(env.roll_off)) != g["roll_off"]:
        return "roll-off differs"
    if g and env.first_player != g["first_player"]:
        return f"first player {env.first_player}, logged {g['first_player']}"
    p0 = recs[0]["ply"]
    if p0:                                  # records lost before this ply: roll the dice on
        for _ in range(p0):
            env.roll_dice()
        env.tokens = {int(p): list(t) for p, t in recs[0]["tokens_before"].items()}
        env.current_player_idx = env.player_ids.index(recs[0]["player_id"])
    seen, roll, legal = {}, env.roll_dice, env.get_legal_moves
    env.roll_dice = lambda: seen.setdefault("dice", roll())
    env.get_legal_moves = lambda p, d: seen.setdefault("legal", legal(p, d))
    agent = _Script()
    agents = {p: agent for p in env.player_ids}
    for k, r in enumerate(recs):
        ply = p0 + k
        if r["ply"] != ply:
            return f"ply {ply}: record has ply {r['ply']}"
        if env.game_over:
            return f"ply {ply}: game already over"
        player = env.player_ids[env.current_player_idx]
        if board(env.tokens) != r["tokens_before"]:
            return f"ply {ply}: board before differs"
        seen.clear()
        agent.move = r["chosen_index"]
        try:
            env.step(agents)
        except Exception as e:
            return f"ply {ply}: engine raised {e!r}"[:300]
        for what, mine, logged in (("dice", seen["dice"], r["dice"]),
                                   ("player", player, r["player_id"]),
                                   ("legal", seen["legal"], r["legal_moves"])):
            if mine != logged:
                return f"ply {ply}: {what} {mine}, logged {logged}"
        if bool(r["skipped"]) != (not seen["legal"]):
            return f"ply {ply}: skipped flag differs"
        if board(env.tokens) != r["tokens_after"]:
            return f"ply {ply}: board after differs"
    if g is None or g["termination_reason"] == "error":
        return None
    if g["dice_rolls"] != p0 + len(recs):
        return f"{g['dice_rolls']} dice rolls logged, {p0 + len(recs)} records"
    if g["termination_reason"] == "win" and not (env.game_over and env.winner == g["winner"]):
        return f"winner {env.winner} (over={env.game_over}), logged {g['winner']}"
    if g["termination_reason"] == "max_steps" and env.game_over:
        return "game over before max_steps"
    if board(env.tokens) != g["final_tokens"]:
        return "final board differs"
    return None


def check_games_run(L, d):
    meta = json.load(open(os.path.join(d, "run.json"), encoding="utf-8"))
    s = meta["settings"]
    games = list(lines(os.path.join(d, "games.jsonl")))
    by_id = {g["game_id"]: g for g in games}
    out = {"run": d, "kind": "games", "finished": 0, "errored": 0, "unfinished": 0,
           "partial_start": [], "records": 0, "problems": []}
    if s["original_rules"]:
        out["problems"].append("original-rules runs share one generator; not replayed here")
        return out
    if any(g["seed"] != s["seed0"] + g["game_index"] for g in games):
        out["problems"].append("seed != seed0 + game_index")
    done, cur, recs = set(), None, []

    def flush():
        if not recs:
            return
        g = by_id.get(cur)
        if cur in done:
            out["problems"].append(f"{cur}: records not contiguous")
        done.add(cur)
        kind = "unfinished" if g is None else ("errored" if g["termination_reason"] == "error"
                                               else "finished")
        out[kind] += 1
        if recs[0]["ply"]:
            out["partial_start"].append(f"{cur} from ply {recs[0]['ply']}")
        problem = check_game(L, len(s["specs"]), g, recs)
        if problem:
            out["problems"].append(f"{cur} ({kind}): {problem}")
        out["records"] += len(recs)

    for r in lines(os.path.join(d, "decisions.jsonl")):
        if r["game_id"] != cur:
            flush()
            cur, recs = r["game_id"], []
        recs.append(r)
    flush()
    out["problems"] += [f"{g['game_id']}: game logged, no decision records"
                        for g in games if g["game_id"] not in done and g["dice_rolls"]]
    return out


def check_spots_run(R, d, engine):
    s = json.load(open(os.path.join(d, "run.json"), encoding="utf-8"))["settings"]
    folder = os.path.join(engine, s["spots_folder"])
    spots = {x["id"]: x for f in sorted(os.listdir(folder)) if f.startswith("spots_")
             for x in json.load(open(os.path.join(folder, f), encoding="utf-8"))}
    out = {"run": d, "kind": "spots", "records": 0,
           "expected": s["spots"] * len(s["legal_moves"]), "problems": []}
    for r in lines(os.path.join(d, "decisions.jsonl")):
        out["records"] += 1
        sp = spots[r["spot_id"]]
        tokens = {int(p): list(t) for p, t in sp["tokens"].items()}
        p, dice = sp["current_player"], sp["dice"]
        legal = R.legal_moves(tokens, p, dice, False)
        after = R.apply_move(tokens, p, r["chosen_index"], dice, False)[0] if legal else tokens
        for what, ok in (("board before", r["tokens_before"] == sp["tokens"]),
                         ("dice", r["dice"] == dice), ("player", r["player_id"] == p),
                         ("legal moves", r["legal_moves"] == legal),
                         ("board after", r["tokens_after"] == board(after))):
            if not ok:
                out["problems"].append(f"{r['spot_id']} shown={r['config']['legal_moves_shown']}: "
                                       f"{what}")
    return out


def worker(engine, runs):
    """Check runs with the engine in `engine` (imported from there), one JSON line each."""
    sys.path.insert(0, engine)
    import ludo_env
    import ludo_rules
    home = os.getcwd()
    with tempfile.TemporaryDirectory() as cwd:      # the engine writes log files to its cwd
        os.chdir(cwd)
        for d in runs:
            spots = "spots_folder" in json.load(open(os.path.join(d, "run.json"),
                                                     encoding="utf-8"))["settings"]
            try:
                res = check_spots_run(ludo_rules, d, engine) if spots else check_games_run(ludo_env, d)
            except Exception as e:
                res = {"run": d, "kind": "error", "problems": [f"verifier failed: {e!r}"]}
            print(json.dumps(res), flush=True)
        os.chdir(home)                              # Windows cannot remove the cwd


def find_runs(roots):
    """Run folders (with a run.json) under the roots, by commit (None if not recorded)."""
    by_commit = defaultdict(list)
    for root in roots:
        for d, _, names in sorted(os.walk(root)):
            if "run.json" in names:
                git = json.load(open(os.path.join(d, "run.json"), encoding="utf-8")).get("git") or {}
                by_commit[(git.get("commit") or None) and git["commit"][:12]].append(os.path.abspath(d))
    return by_commit


def extract(commit, into):
    """The code of a commit, into a folder (git archive: read-only on the repository)."""
    data = subprocess.run(["git", "-C", REPO, "archive", commit], capture_output=True,
                          check=True).stdout
    tarfile.open(fileobj=io.BytesIO(data)).extractall(into, filter="data")


def summary(res):
    if res["kind"] == "error":
        n = "not checked"
    elif res["kind"] == "spots":
        n = f"{res['records']}/{res['expected']} records"
    else:
        n = (f"{res['finished']} finished, {res['errored']} errored, {res['unfinished']} "
             f"unfinished games, {res['records']} records")
    extra = f"; replayed from a later ply: {', '.join(res['partial_start'])}" \
        if res.get("partial_start") else ""
    return f"{'OK  ' if not res['problems'] else 'FAIL'} {res['run']}: {n}{extra}" + \
        "".join(f"\n       {p}" for p in res["problems"][:10])


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        return worker(sys.argv[2], sys.argv[3:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("roots", nargs="+", help="folders to search for runs")
    ap.add_argument("--unknown-commit", metavar="COMMIT",
                    help="replay runs that record no commit with this commit's engine")
    args = ap.parse_args()
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for commit, runs in find_runs(args.roots).items():
            commit = commit or args.unknown_commit
            if commit is None:
                print("\n".join(f"SKIP {d}: no commit recorded (see --unknown-commit)" for d in runs))
                continue
            engine = os.path.join(tmp, commit)
            if not os.path.isdir(engine):
                extract(commit, engine)
            proc = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", engine, *runs],
                                  capture_output=True, text=True, encoding="utf-8",
                                  env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
            if proc.returncode:
                raise SystemExit(proc.stderr)
            for line in proc.stdout.splitlines():
                res = json.loads(line)
                bad += bool(res["problems"])
                print(summary(res))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
