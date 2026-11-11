"""Every component takes its rules from ludo_rules: engine, heuristic, search players, logs, renderer, spots."""
import glob
import json
import random

import pytest

import board_render
import decision_log
import ludo_rules as R
import reference as ref
from agents import HeuristicAgent
from game_theory_agent import GameTheoryAgent, best_move
from game_theory_multiplayer_agent import GameTheoryMultiplayerAgent
from spot_evaluation import SpotEnv, compute_legal_moves, evaluate_spot


def env(tokens, original=False):
    return SpotEnv(list(tokens), {p: list(t) for p, t in tokens.items()}, original)


@pytest.mark.parametrize("original", [False, True])
def test_components_agree_with_the_rules_module(original):
    rng = random.Random(3)
    for _ in range(3000):
        players = ref.seat_sets(rng)
        e = env(ref.random_state(rng, players, original), original)
        p, dice = rng.choice(players), rng.randint(1, 6)
        legal = compute_legal_moves(e, p, dice)
        assert legal == R.legal_moves(e.tokens, p, dice, original)
        for t in legal:
            to = R.destination(p, e.tokens[p][t], dice, original)
            assert board_render.move_outcome(e, p, dice, t)[0] == to
            effects = decision_log.move_effects(e, p, dice, t)
            assert effects["to"] == to
            assert [(c["player"], c["token"]) for c in effects["captures"]] == \
                R.captures(e.tokens, p, to)


def test_heuristic_scores_progress_capture_and_safe_square():
    e = env({0: [-1, 10, 4, R.home_range(0)[0]], 2: [14, -1, -1, -1]})
    assert HeuristicAgent().scores(e, 0, 6, [0, 1, 2]) == {0: 50, 1: 16, 2: 10}
    assert HeuristicAgent().scores(e, 0, 4, [1, 2, 3]) == {1: 14 + 100, 2: 8 + 20, 3: 55}
    assert HeuristicAgent().scores(e, 0, 5, [3]) == {3: 56}


def test_heuristic_takes_first_best_move_in_list_order():
    e = env({0: [-1, -1, -1, -1], 2: [-1, -1, -1, -1]})
    assert HeuristicAgent().select_move(e, 0, 6, [3, 2, 1, 0]) == 3


def test_progress_counts_pips_from_start():
    e = env({0: [-1, 0, 50, R.home_range(0)[1]], 2: [-1] * 4})
    assert [decision_log.token_progress(e, 0, x) for x in e.tokens[0]] == [0, 1, 51, 57]
    old = env({0: [R.home_range(0)[1]] * 4, 2: [-1] * 4}, original=True)
    assert decision_log.token_progress(old, 0, R.home_range(0)[1]) == 59


def test_exposure_check_stops_at_the_opponents_home_entry():
    # P2's last track square is 24; from 23 it can reach 24 but not 25 (it enters its home path).
    assert decision_log.is_exposed(env({0: [24, -1, -1, -1], 2: [23, -1, -1, -1]}), 0, 24)
    assert not decision_log.is_exposed(env({0: [25, -1, -1, -1], 2: [23, -1, -1, -1]}), 0, 25)
    # Original rules: P2 goes on to 25, but from 23 a 4 lands in its home path (skipping a pip).
    assert decision_log.is_exposed(env({0: [25, -1, -1, -1], 2: [23, -1, -1, -1]}, True), 0, 25)
    assert not decision_log.is_exposed(env({0: [27, -1, -1, -1], 2: [23, -1, -1, -1]}, True), 0, 27)


def step(a, b):
    return b[0] - a[0], b[1] - a[1]


def adjacent(a, b):
    """Neighbouring squares; the track turns the board's inner corners diagonally."""
    return max(map(abs, step(a, b))) == 1


@pytest.mark.parametrize("player", range(4))
def test_each_route_walks_one_square_at_a_time_on_the_drawn_board(player):
    pos, cells = R.START_POSITIONS[player], []
    while pos is not None:
        cells.append(board_render.coords(player, pos))
        pos = R.destination(player, pos, 1)
    assert len(cells) == 57 and len(set(cells)) == 57          # 56 steps, no square twice
    assert all(adjacent(a, b) for a, b in zip(cells, cells[1:]))
    assert all(adjacent(a, b) for a, b in zip(board_render.TRACK, board_render.TRACK[1:] +
                                               board_render.TRACK[:1]))
    assert step(cells[50], cells[51]) == step(cells[51], cells[52])   # straight into the column
    behind = board_render.TRACK[(R.START_POSITIONS[player] - 1) % 52]
    assert behind not in cells and adjacent(behind, cells[0])


def test_board_labels_home_moves_and_finishes():
    e = env({0: [50, R.home_range(0)[0], -1, -1], 2: [-1] * 4})
    assert board_render.move_outcome(e, 0, 1, 0)[1] == "sq50 -> home1"
    assert board_render.move_outcome(e, 0, 2, 1)[1] == "home1 -> home3"
    assert "FINISHES" in board_render.describe_move(e, 0, 5, 1)


# ---------- search players ----------

def test_search_ties_go_to_lowest_piece_index():
    values = {3: 0.21458333333333335, 2: 0.21458333333333332, 0: 0.205}
    assert best_move(values, [3, 2, 0], original=False) == 2
    assert best_move(values, [3, 2, 0], original=True) == 3


@pytest.mark.parametrize("players", [[0, 2], [0, 1, 2], [0, 1, 2, 3]])
def test_search_ties_ignore_legal_list_order(players):
    tokens = {p: [-1] * 4 for p in players}              # every bring-out is the same move
    agent = GameTheoryMultiplayerAgent(2)
    assert agent.select_move(env(tokens), 0, 6, [3, 2, 1, 0]) == 0
    assert agent.select_move(env(tokens, original=True), 0, 6, [3, 2, 1, 0]) == 3


def test_search_players_choose_identically_in_two_player_positions():
    rng = random.Random(11)
    a, b = GameTheoryAgent(2), GameTheoryMultiplayerAgent(2)
    checked = 0
    while checked < 300:
        players = rng.choice([[0, 1], [0, 2]])
        e = env(ref.random_state(rng, players))
        p, dice = rng.choice(players), rng.randint(1, 6)
        legal = R.legal_moves(e.tokens, p, dice)
        if len(legal) > 1:
            assert a.select_move(e, p, dice, legal) == b.select_move(e, p, dice, legal)
            checked += 1


def test_search_player_finishes_when_it_can_win():
    end = R.home_range(2)[1]
    e = env({0: [30, 31, -1, -1], 2: [end, end, end, R.home_range(2)[0] + 2]})
    assert GameTheoryMultiplayerAgent(2).select_move(e, 2, 3, [3]) == 3


# ---------- spots ----------

SPOTS = [s for f in sorted(glob.glob("spots_40/*.json")) for s in json.load(open(f))]


def test_every_spot_keeps_its_legal_moves_and_seats():
    assert len(SPOTS) == 520
    for s in SPOTS:
        tokens = {int(k): v for k, v in s["tokens"].items()}
        ours = compute_legal_moves(SpotEnv(s["players"], tokens), s["current_player"], s["dice"])
        orig = compute_legal_moves(SpotEnv(s["players"], tokens, True),
                                   s["current_player"], s["dice"])
        assert ours and ours == orig, s["id"]


def test_spot_log_reads_a_reply_without_a_bar(tmp_path):
    out = evaluate_spot([0, 1], {0: [5, -1, -1, -1], 1: [-1] * 4}, 6, 0,
                        llm_call=lambda prompt: "I move token 1",
                        llm_output_path=str(tmp_path / "spot.json"))
    assert out["response_int"] == 1
