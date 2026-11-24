"""Round robins: the pair and group schedules, and runs that log seating, seeds and first player."""
import json
import sys
from collections import Counter

import pytest

import round_robin
import run_games

POOL = ["llm:bedrock/a.x", "llm:bedrock/b.x", "random", "heuristic", "first"]


def cyclic(order):
    """An order read round the table from its first player's seat."""
    i = order.index(min(order))
    return tuple(order[i:] + order[:i])


def test_four_player_groups_leave_one_out_and_cover_every_cyclic_seating():
    runs = round_robin.schedule(POOL, 4, 48)
    assert len(runs) == 30 and all(step == 6 and n == 8 for _, _, _, step, n in runs)
    groups = {}
    for folder, specs, seed0, step, n in runs:
        group, seating = folder.replace("\\", "/").split("/")
        assert seating == f"seating-{seed0}"
        groups.setdefault(group, []).append((specs, [seed0 + g * step for g in range(n)]))
    assert sorted(groups) == sorted(f"without-{run_games.folder_name(p)}" for p in POOL)
    for group, seatings in groups.items():
        seeds = sorted(s for _, ss in seatings for s in ss)
        assert seeds == list(range(48))                             # every seed once
        assert all(s % 6 == k for k, (_, ss) in enumerate(seatings) for s in ss)
        assert len({cyclic(specs) for specs, _ in seatings}) == 6   # all different round the table
        assert len({specs[0] for specs, _ in seatings}) == 1
    # the same schedule in every group: seating k puts the members in the same pool order
    ranks = {tuple(tuple(sorted(sp, key=POOL.index).index(p) for p in sp) for sp, _ in ss)
             for ss in groups.values()}
    assert len(ranks) == 1


def test_pairs_play_one_run_on_seeds_0_to_n():
    runs = round_robin.schedule(POOL[:3], 2, 10)
    assert [r[0] for r in runs] == ["bedrock-a.x_bedrock-b.x", "bedrock-a.x_random",
                                    "bedrock-b.x_random"]
    assert all((seed0, step, n) == (0, 1, 10) for _, _, seed0, step, n in runs)


def test_games_must_split_evenly_over_the_seatings():
    with pytest.raises(ValueError, match="multiple of 6"):
        round_robin.schedule(POOL, 4, 50)


def play(tmp_path, monkeypatch, *args):
    monkeypatch.chdir(tmp_path)                       # the engine's text logs
    monkeypatch.setattr(run_games, "RESULTS", str(tmp_path / "results"))
    monkeypatch.setattr(sys, "argv", ["round_robin.py", *args, "--experiment", "rr",
                                      "--max-steps", "150"])
    round_robin.main()
    return tmp_path / "results" / "rr"


def test_a_four_player_round_robin_logs_seating_seeds_and_who_started(tmp_path, monkeypatch):
    out = play(tmp_path, monkeypatch, "--players", "4", "--games", "12",
               "--pool", "random", "heuristic", "first", "gt:1")
    games = []
    for k in range(6):
        run = json.loads((out / "all" / f"seating-{k}" / "run.json").read_text(encoding="utf-8"))
        assert (run["settings"]["seed0"], run["settings"]["seed_step"]) == (k, 6)
        specs = run["settings"]["specs"]
        for line in (out / "all" / f"seating-{k}" / "games.jsonl").read_text("utf-8").splitlines():
            g = json.loads(line)
            assert list(g["players"].values()) == specs and g["seed"] % 6 == k
            assert g["first_player"] == max(g["roll_off"][-1], key=lambda r: r[1])[0]
            games.append(g)
    assert sorted(g["seed"] for g in games) == list(range(12))
    assert Counter(len(g["roll_off"][0]) for g in games) == {4: 12}   # all four rolled


def test_a_pair_round_robin_and_a_dry_run(tmp_path, monkeypatch, capsys):
    out = play(tmp_path, monkeypatch, "--games", "2", "--pool", "random", "first", "heuristic",
               "--dry-run")
    assert "random_first: random  vs  first; seeds 0, 1, ..., 1" in capsys.readouterr().out
    assert not out.exists()
    out = play(tmp_path, monkeypatch, "--games", "2", "--pool", "random", "first", "heuristic")
    assert sorted(p.name for p in out.iterdir()) == ["first_heuristic", "random_first",
                                                     "random_heuristic"]
    run = json.loads((out / "random_first" / "run.json").read_text(encoding="utf-8"))
    assert run["settings"]["seed0"] == 0 and "seed_step" not in run["settings"]
