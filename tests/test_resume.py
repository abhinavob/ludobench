"""Resuming a stopped run, and the run queue."""
import json
import subprocess
import sys

import pytest

import run_games
import run_queue
from ludo_env import LudoEnv


def play(out, games=4, specs=("heuristic", "random")):
    return run_games.run_games(list(specs), games=games, seed0=0, out_dir=str(out))


def read(out):
    """A run's plies and games, without the run_id (the folder name)."""
    plies = [{k: v for k, v in json.loads(line).items() if k != "run_id"}
             for line in (out / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    games = [json.loads(line) for line in (out / "games.jsonl").read_text(encoding="utf-8").splitlines()]
    return plies, [(g["game_id"], g["winner"], g["dice_rolls"]) for g in games]


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def stop_in_game(mp, seed):
    """Make the game with this seed stop after 50 rolls, as Ctrl+C would."""
    step = LudoEnv.step

    def stopping(env, agents):
        env.rolls = getattr(env, "rolls", 0) + 1
        if env.seed == seed and env.rolls > 50:
            raise KeyboardInterrupt
        return step(env, agents)

    init = LudoEnv.__init__

    def remember(env, seed=42, *a, **k):
        init(env, seed, *a, **k)
        env.seed = seed
    mp.setattr(LudoEnv, "__init__", remember)
    mp.setattr(LudoEnv, "step", stopping)


def test_stopped_run_resumes_to_the_same_result(tmp_path, monkeypatch):
    play(tmp_path / "whole")
    with monkeypatch.context() as mp:
        stop_in_game(mp, 2)
        with pytest.raises(KeyboardInterrupt):
            play(tmp_path / "stopped")
    assert len(read(tmp_path / "stopped")[1]) == 3               # the stopped game is logged as an error
    play(tmp_path / "stopped")
    assert read(tmp_path / "stopped") == read(tmp_path / "whole")
    meta = json.loads((tmp_path / "stopped" / "run.json").read_text(encoding="utf-8"))
    assert meta["games"] == {"attempted": 4, "completed": 4, "errored": 0}
    assert len(meta["resumed_at"]) == 1


def test_finished_run_is_left_as_it_is(tmp_path):
    play(tmp_path / "run")
    before = read(tmp_path / "run")
    assert len(play(tmp_path / "run")) == 4
    assert read(tmp_path / "run") == before


def test_more_games_continue_a_finished_run(tmp_path):
    play(tmp_path / "short", games=2)
    play(tmp_path / "short", games=4)
    play(tmp_path / "long", games=4)
    assert read(tmp_path / "short") == read(tmp_path / "long")


def test_reply_cut_off_mid_line_is_dropped(tmp_path):
    play(tmp_path / "whole")
    play(tmp_path / "cut", games=3)
    for name in ("decisions.jsonl", "games.jsonl"):
        path = tmp_path / "cut" / name
        path.write_text(path.read_text(encoding="utf-8") + '{"game_id": "g003-s3", "pl',
                        encoding="utf-8")
    play(tmp_path / "cut")
    assert read(tmp_path / "cut") == read(tmp_path / "whole")


def test_folder_with_other_settings_is_refused(tmp_path):
    play(tmp_path / "run")
    with pytest.raises(SystemExit, match="other settings"):
        play(tmp_path / "run", specs=("random", "heuristic"))


def test_queue_runs_each_line_in_order_and_reports_failures(tmp_path, monkeypatch, capsys):
    queue = tmp_path / "queue.txt"
    queue.write_text("# comment\n--players heuristic random\n\n"
                     "  --players llm:m heuristic --outcomes --experiment 'with space'\n",
                     encoding="utf-8")
    calls = []

    def fake_run(cmd):
        calls.append(cmd[2:])
        return subprocess.CompletedProcess(cmd, 1 if "--outcomes" in cmd else 0)
    monkeypatch.setattr(run_queue.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_queue.py", str(queue)])
    with pytest.raises(SystemExit) as stop:
        run_queue.main()
    assert stop.value.code == 1
    assert calls == [["--players", "heuristic", "random"],
                     ["--players", "llm:m", "heuristic", "--outcomes", "--experiment",
                      "with space"]]
    assert "1 of 2 runs done" in capsys.readouterr().out
