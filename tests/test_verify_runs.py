"""analysis/verify_runs.py: clean runs pass; changed dice, legal moves, boards, winners or records fail."""
import json
import shutil
import subprocess
import sys

import pytest

import ludo_env
import ludo_rules
import run_games
import run_spots
from analysis import verify_runs as V
from test_run_spots import Model
from test_decision_records import offline


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """A two-game bot run and a 13-spot run, made by the current code."""
    root = tmp_path_factory.mktemp("roots") / "results"
    mp = pytest.MonkeyPatch()
    mp.chdir(root.parent)
    run_games.run_games(["random", "heuristic"], games=2, seed0=3, out_dir=str(root / "x" / "game"))
    offline(mp, Model())
    run_spots.run_spots("llm:m:1b", str(root / "x" / "spot"), legal=("shown",), per_category=1)
    mp.undo()
    return root


def changed(src, dst, edit, name="decisions.jsonl"):
    """A copy of the run in src with edit(rows) applied to one of its files."""
    shutil.copytree(src, dst)
    rows = [json.loads(line) for line in open(dst / name, encoding="utf-8")]
    edit(rows)
    (dst / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(dst)


def problems(run, spots=False):
    res = V.check_spots_run(ludo_rules, run, V.REPO) if spots else V.check_games_run(ludo_env, run)
    return res["problems"]


def first_move(rows, start=0):
    return next(r for r in rows[start:] if len(r["legal_moves"]) > 1)


def test_clean_runs_pass(runs, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)                 # the engine writes its log files to the cwd
    assert problems(str(runs / "x" / "game")) == []
    assert problems(str(runs / "x" / "spot"), spots=True) == []


@pytest.mark.parametrize("name, edit, found", [
    ("dice", lambda rs: rs[5].update(dice=rs[5]["dice"] % 6 + 1), "dice"),
    ("legal", lambda rs: first_move(rs).update(legal_moves=first_move(rs)["legal_moves"][1:]),
     "legal"),
    ("move", lambda rs: first_move(rs, 30).update(
        chosen_index=[m for m in first_move(rs, 30)["legal_moves"]
                      if m != first_move(rs, 30)["chosen_index"]][0]), "board after differs"),
    ("dropped", lambda rs: rs.pop(12), "record has ply"),
])
def test_changed_records_fail(runs, monkeypatch, tmp_path, name, edit, found):
    monkeypatch.chdir(tmp_path)
    run = changed(runs / "x" / "game", tmp_path / name, edit)
    assert any(found in p for p in problems(run)), problems(run)


def test_changed_winner_and_spot_board_fail(runs, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    run = changed(runs / "x" / "game", tmp_path / "w",
                  lambda gs: gs[0].update(winner=[p for p in (0, 2) if p != gs[0]["winner"]][0]),
                  "games.jsonl")
    assert any("winner" in p for p in problems(run))
    run = changed(runs / "x" / "spot", tmp_path / "s",
                  lambda rs: rs[0]["tokens_before"]["0"].__setitem__(0, 7))
    assert any("board before" in p for p in problems(run, spots=True))


def test_command_line_replays_with_each_run_s_commit_and_skips_unknown_commits(runs, tmp_path):
    out = subprocess.run([sys.executable, V.__file__, str(runs)], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.count("OK  ") == 2 and "2 finished" in out.stdout and "13/13 records" in out.stdout
    unknown = tmp_path / "copied" / "game"
    shutil.copytree(runs / "x" / "game", unknown)
    meta = json.loads((unknown / "run.json").read_text(encoding="utf-8"))
    (unknown / "run.json").write_text(json.dumps(dict(meta, git={"commit": None, "dirty": None})),
                                      encoding="utf-8")
    out = subprocess.run([sys.executable, V.__file__, str(tmp_path / "copied")],
                         capture_output=True, text=True)
    assert out.stdout.startswith("SKIP") and str(unknown) in out.stdout
