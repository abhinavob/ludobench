"""
Find and read runs for the analysis, read-only, and write its tables and figures.

A run is a folder with a run.json under one of the results roots. Its
experiment is the folder path between the root and the run, its model the
first "_"-separated part of the run folder (run_games.run_folder and
run_spots name runs that way), and its player the first LLM (else the player
in seat 0, for bot runs).

Exclusions (part 5): a game listed under games_affected in a run.json
"repairs" entry is left out, and a replay of its seed, a run of the same
name in the experiment "<experiment>-replay-s<seed>" of the same root, takes
its place.
"""
import csv
import json
import os
import re
from dataclasses import dataclass, field

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(REPO)
DEFAULT_ROOTS = {"main": os.path.join(PARENT, "LudoBench-5CBF", "results"),
                 "work": os.path.join(PARENT, "LudoBench-work", "results"),
                 "bedrock": os.path.join(PARENT, "LudoBench-bedrock", "results"),
                 "sharanga": os.path.join(PARENT, "sharanga_results")}
FINISHED = ("win", "max_steps")
# The fields of a decision record the analysis keeps.
DECISION_FIELDS = ("game_id", "seed", "ply", "player_id", "dice", "tokens_before", "legal_moves",
                   "shown_legal_moves", "chosen_index", "was_fallback", "completion_tokens",
                   "thinking_tokens", "spot_id", "spot_changed", "pair_id", "search_move",
                   "move_groups", "availability")


def lines(path):
    """The complete JSON lines of a file (a live run may be mid-line)."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.endswith("\n"):
                yield json.loads(line)


@dataclass
class Run:
    root: str                       # label of the results root
    experiment: str
    name: str                       # the run folder
    path: str
    meta: dict = field(repr=False)

    @property
    def settings(self):
        return self.meta["settings"]

    @property
    def kind(self):
        return "spots" if "spots_folder" in self.settings else "games"

    @property
    def model(self):
        return self.name.split("_")[0]

    @property
    def key(self):
        return f"{self.root}:{self.experiment}/{self.name}"

    @property
    def pid(self):
        """The analysed player: the first LLM, else the player in seat 0."""
        players = self.meta["players"]
        llm = [p for p, v in players.items() if v.get("provider")]
        return int(min(llm or players, key=lambda p: players[p]["seat_position"]))


def find_runs(roots=None):
    """Every run under the roots ({label: folder}), sorted by root, experiment and name."""
    runs = []
    for label, top in (roots or DEFAULT_ROOTS).items():
        for d, _, names in os.walk(top):
            if "run.json" in names:
                rel = os.path.relpath(d, top).replace("\\", "/")
                experiment, _, name = rel.rpartition("/")
                with open(os.path.join(d, "run.json"), encoding="utf-8") as f:
                    runs.append(Run(label, experiment, name, d, json.load(f)))
    return sorted(runs, key=lambda r: (r.root, r.experiment, r.name))


def excluded_games(run):
    """{game_id: seed} of the games a repairs entry of run.json takes out of the analysis."""
    return {g["game_id"]: g["seed"] for rep in run.meta.get("repairs", [])
            for g in rep.get("games_affected", [])}


def replay_of(run, seed, runs):
    return next((r for r in runs if (r.root, r.experiment, r.name) ==
                 (run.root, f"{run.experiment}-replay-s{seed}", run.name)), None)


def games(run, runs):
    """
    (finished game records of the run, notes): repaired games are left out and
    replays of their seeds put in. Each record gains "_run" and "_dir", the
    run it is from and its folder.
    """
    out = excluded_games(run)
    kept = [dict(g, _run=run.key, _dir=run.path) for g in lines(os.path.join(run.path, "games.jsonl"))
            if g["termination_reason"] in FINISHED and g["game_id"] not in out]
    notes = {"excluded": sorted(out), "replaced_by": []}
    for seed in out.values():
        replay = replay_of(run, seed, runs)
        if replay is not None:
            kept += [dict(g, _run=replay.key, _dir=replay.path)
                     for g in lines(os.path.join(replay.path, "games.jsonl"))
                     if g["termination_reason"] in FINISHED and g["seed"] == seed]
            notes["replaced_by"].append(replay.key)
    return sorted(kept, key=lambda g: g["seed"]), notes


def game_rows(run, recs):
    """Per game: seed, whether the run's player won, and its model calls and invalid replies."""
    pid = run.pid
    rows = []
    for g in recs:
        llm = g["per_player"][str(pid)].get("llm") or {}
        rows.append({"seed": g["seed"], "game": f"{g['_run']}#{g['game_id']}",
                     "won": int(g["winner"] == pid), "no_winner": int(g["winner"] is None),
                     "calls": llm.get("calls", 0), "invalid": llm.get("invalid", 0)})
    return rows


def decisions(run, recs=None):
    """
    The run's player's decisions with at least one legal move, trimmed to
    DECISION_FIELDS, from the games in recs; for a spot run (recs None) every
    decision, whichever player the spot puts to move. Each gains "game", the
    game it is from (the spot run for spots).
    """
    pid = run.pid
    sources = {run.key: (run.path, None)} if recs is None else {}
    for g in recs or ():
        sources.setdefault(g["_run"], (g["_dir"], set()))[1].add(g["game_id"])
    out = []
    for key, (folder, keep) in sources.items():
        for d in lines(os.path.join(folder, "decisions.jsonl")):
            if d["legal_moves"] and (keep is None or (d["player_id"] == pid and d["game_id"] in keep)):
                row = {k: d.get(k) for k in DECISION_FIELDS}
                row["shown"] = d["config"]["legal_moves_shown"]
                row["game"] = f"{key}#{d.get('game_id')}"
                out.append(row)
    return out


# ---------- output ----------

def write_csv(out, name, rows):
    """Write rows (a list of dicts) to out/name.csv; returns the path."""
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{name}.csv")
    keys = list(dict.fromkeys(k for r in rows for k in r)) if rows else ["empty"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return path


def save_figure(fig, out, name):
    """Save a matplotlib figure as out/name.png and .pdf, then close it."""
    import matplotlib.pyplot as plt
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure(**kw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt.subplots(**kw)


def is_replay(run):
    return re.search(r"-replay-s\d+$", run.experiment) is not None
