"""run_games.py command line: players and seats, switches, thinking, results layout, run.json."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import run_games

ROOT = Path(run_games.__file__).parent

@pytest.fixture
def cli(tmp_path, monkeypatch):
    """Run run_games.main() with a fake Ollama model; returns (run.json, make_llm calls, prompts)."""
    calls, prompts = [], []
    info = {"capabilities": ["completion", "thinking"], "thinking_values": [False, True],
            "digest": "sha256:abc", "quantization": "Q4_K_M", "parameter_size": "8B"}

    def make_llm(spec, **kw):
        calls.append(kw)

        def call(prompt):
            prompts.append(prompt)
            call.last_content = json.dumps({"reason": "first legal", "piece": 0})
            return call.last_content
        call.spec, call.base_url, call.last_content = spec, None, None
        return call

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_games, "RESULTS", str(tmp_path / "results"))
    monkeypatch.setattr(run_games, "make_llm", make_llm)
    monkeypatch.setattr(run_games, "model_info", lambda *a: info)
    monkeypatch.setattr(run_games, "check_ollama", lambda *a: "0.0-test")

    def run(*args, experiment="sandbox", folder="m-1b_heuristic"):
        monkeypatch.setattr(sys, "argv", ["run_games.py", *args])
        run_games.main()
        path = tmp_path / "results" / experiment / folder / "run.json"
        return json.loads(path.read_text(encoding="utf-8")), calls, prompts
    run.root = tmp_path / "results"
    return run


def test_two_players_given_without_seats_sit_in_seats_0_and_2(cli):
    run, _, _ = cli("--players", "llm:m:1b", "heuristic")
    assert run["settings"]["seats"] == [0, 2]
    assert {p: v["model_spec"] for p, v in run["players"].items()} == {"0": "llm:m:1b",
                                                                       "2": "heuristic"}


def test_results_go_to_experiment_and_players_with_a_row_in_cells(cli):
    cli("--players", "llm:m:1b", "random", "--outcomes", "--experiment", "extra",
        experiment="extra", folder="m-1b_random")
    folder = cli.root / "extra" / "m-1b_random"
    assert {f.name for f in folder.iterdir()} == {"run.json", "games.jsonl", "decisions.jsonl"}
    rows = list(csv.DictReader(open(cli.root / "cells.csv", encoding="utf-8")))
    game = json.loads((folder / "games.jsonl").read_text(encoding="utf-8"))
    llm = game["per_player"]["0"]["llm"]
    assert rows == [{"experiment": "extra", "run": "m-1b_random", "model": "m-1b",
                     "opponent": "random",
                     "settings": "--outcomes", "games_finished": "1",
                     "wins": str(int(game["winner"] == 0)),
                     "win_rate": f"{int(game['winner'] == 0):.3f}",
                     "invalid_rate": f"{llm['invalid'] / llm['calls']:.4f}",
                     "ollama_version": "0.0-test", "cost_usd": "0.0000",
                     "commit": rows[0]["commit"]}]
    assert len(rows[0]["commit"].split("-")[0]) == 12


@pytest.mark.parametrize("spec", ["llm:my_model:1b", "llm://host/my_model"])
def test_player_name_with_an_underscore_is_an_error(cli, spec):
    with pytest.raises(SystemExit):
        cli("--players", spec, "heuristic")
    with pytest.raises(ValueError, match="may not contain '_'"):
        run_games.run_folder([spec, "heuristic"])


def test_run_folder_names_every_player_in_seat_order_and_the_effort():
    assert run_games.run_folder(["llm:qwen3.5:4b", "llm:gemma4:e4b"]) == "qwen3.5-4b_gemma4-e4b"
    assert run_games.run_folder(["llm:gemma4:e4b", "llm:qwen3.5:4b"]) == "gemma4-e4b_qwen3.5-4b"
    assert (run_games.run_folder(["llm:nvidia/openai/gpt-oss-20b", "heuristic"], "low")
            == "nvidia-openai-gpt-oss-20b_heuristic_effort-low")


def test_two_queues_updating_cells_at_once_lose_no_row(tmp_path):
    path = str(tmp_path / "cells.csv")
    script = ("import sys, run_games\n"
              "for i in range(40):\n"
              "    run_games.update_cells(sys.argv[1], {'experiment': 'e', 'run': sys.argv[2] + str(i)})")
    queues = [subprocess.Popen([sys.executable, "-c", script, path, q], cwd=ROOT) for q in "ab"]
    assert [q.wait() for q in queues] == [0, 0]
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert sorted(r["run"] for r in rows) == sorted(f"{q}{i}" for q in "ab" for i in range(40))
    assert not [f for f in tmp_path.iterdir() if f.name != "cells.csv"]   # no lock or temp file left


def test_lock_left_by_a_dead_process_is_taken_over(tmp_path):
    lock = tmp_path / "cells.csv.lock"
    lock.write_text("")
    os.utime(lock, (0, 0))
    run_games.update_cells(str(tmp_path / "cells.csv"), {"experiment": "e", "run": "r"})
    assert not lock.exists()


def test_cells_count_wins_and_invalid_replies_of_the_run_model():
    def game(winner, llm):
        return {"winner": winner, "per_player": {"0": {"llm": llm}, "2": {"llm": None}}}
    records = [game(0, {"calls": 30, "invalid": 3}), game(2, {"calls": 10, "invalid": 1}),
               game(None, {"calls": 0, "invalid": 0})]
    assert run_games.cell_results(records, 0, {"http://a": "0.40.16"}) == {
        "games_finished": 3, "wins": 1, "win_rate": "0.333", "invalid_rate": "0.1000",
        "ollama_version": "0.40.16"}
    assert run_games.cell_results(records, 2, {}) == {
        "games_finished": 3, "wins": 1, "win_rate": "0.333", "invalid_rate": "",
        "ollama_version": ""}


def test_standard_setup_needs_no_switches(cli):
    run, calls, prompts = cli("--players", "llm:m:1b", "heuristic")
    s = run["settings"]
    assert not s["original_rules"] and s["structured_output"] and s["reason_max_chars"] == 500
    assert s["schema_order"] == "reason_first" and s["legal_order"] == "ascending"
    assert s["include_legal_moves"] and not s["move_outcomes"] and s["last_moves"] == 0
    assert calls[0]["max_tokens"] == 1024 and calls[0]["think"] is False
    assert calls[0]["num_ctx"] == 8192
    assert "Legal moves:" in prompts[0] and "What each legal move does" not in prompts[0]


def test_run_json_records_commit_and_model_version(cli):
    run, _, _ = cli("--players", "llm:m:1b", "heuristic")
    assert len(run["git"]["commit"]) == 40 and run["git"]["dirty"] in (True, False)
    p = run["players"]["0"]
    assert (p["digest"], p["quantization"]) == ("sha256:abc", "Q4_K_M")
    assert (p["think"], p["thinks"], p["ceiling"], p["num_ctx"]) == (False, False, 1024, 8192)


def test_think_turns_thinking_on_with_the_thinking_ceiling_and_context(cli):
    run, calls, _ = cli("--players", "llm:m:1b", "heuristic", "--think")
    assert calls[0]["think"] is True and calls[0]["max_tokens"] == 16384
    assert calls[0]["num_ctx"] == 20480
    assert (run["players"]["0"]["ceiling"], run["players"]["0"]["num_ctx"]) == (16384, 20480)
    assert run["settings"]["contexts"] == {"answer": 8192, "thinking": 20480}
    assert run["settings"]["ceilings"] == {"answer": 1024, "thinking": 16384}


def test_effort_level_on_a_model_without_levels_is_an_error(cli):
    with pytest.raises(SystemExit, match="effort levels"):
        cli("--players", "llm:m:1b", "heuristic", "--effort", "high")


@pytest.mark.parametrize("switch, check", [
    (["--hide-legal-moves"], lambda p: "Legal moves:" not in p),
    (["--outcomes"], lambda p: "What each legal move does" in p),
])
def test_switches_change_the_prompt(cli, switch, check):
    _, _, prompts = cli("--players", "llm:m:1b", "heuristic", *switch)
    assert all(check(p) for p in prompts)


def test_history_shows_the_last_moves(cli):
    _, _, prompts = cli("--players", "llm:m:1b", "heuristic", "--history", "2")
    assert any("Last 2 moves, oldest first:" in p for p in prompts)


def test_shuffle_uses_the_shuffled_order(cli):
    run, _, _ = cli("--players", "llm:m:1b", "heuristic", "--shuffle")
    assert run["settings"]["legal_order"] == "shuffled"


def test_original_rules_use_the_released_reply_setup(cli):
    run, calls, prompts = cli("--players", "llm:m:1b", "heuristic", "--original-rules")
    assert calls[0] == {} and run["settings"]["max_tokens"] == 60
    assert run["settings"]["stop"] == ["\n"] and "<int> | <one line reason>" in prompts[0]


@pytest.mark.parametrize("args", [["--p0", "heuristic", "--p1", "random"],
                                  ["--players", "heuristic", "random", "--no-think"],
                                  ["--players", "heuristic", "random", "--max-tokens", "9"],
                                  ["--players", "heuristic", "random", "--legal-order", "reversed"],
                                  ["--players", "ollama:m", "heuristic"],
                                  ["--players", "heuristic"]])
def test_removed_options_and_bad_players_are_rejected(cli, args):
    with pytest.raises(SystemExit):
        cli(*args)


def test_cells_of_a_self_play_run_count_the_first_listed_player(cli, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_games.py", "--players", "heuristic", "heuristic",
                                      "--games", "4"])
    run_games.main()
    log = cli.root / "sandbox" / "heuristic_heuristic" / "games.jsonl"
    games = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    row, = csv.DictReader(open(cli.root / "cells.csv", encoding="utf-8"))
    assert row["wins"] == str(sum(g["winner"] == 0 for g in games))
    assert row["invalid_rate"] == row["ollama_version"] == ""


def test_watch_shows_the_roll_off_and_every_roll_and_changes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    for out, watch in (("quiet", False), ("watched", True)):
        run_games.run_games(["heuristic", "random"], games=1, seed0=2, out_dir=str(tmp_path / out),
                            watch=watch)
    shown = capsys.readouterr().out
    logs = {out: (tmp_path / out / "decisions.jsonl").read_text(encoding="utf-8").replace(out, "")
            for out in ("quiet", "watched")}
    assert logs["quiet"] == logs["watched"]
    plies = [json.loads(line) for line in logs["watched"].splitlines()]
    game = json.loads((tmp_path / "watched" / "games.jsonl").read_text(encoding="utf-8"))
    roll_off = "; then ".join(", ".join(f"P{p} {d}" for p, d in r) for r in game["roll_off"])
    assert f"Roll-off: {roll_off}" in shown and f"P{game['first_player']} [" in shown
    rolls = [line.split(": ", 1)[1] for line in shown.splitlines() if line.startswith("  roll ")]
    assert [r.split(" rolled ")[0][1:].split()[0] for r in rolls] == [str(p["player_id"]) for p in plies]
    assert [int(r.split(" rolled ")[1].split()[0]) for r in rolls] == [p["dice"] for p in plies]
    assert shown.count("-> no legal move") == sum(p["skipped"] for p in plies)
