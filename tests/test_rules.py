"""Rules module: one test per rule and edge case, and agreement with the reference rules."""
import random

import pytest

import ludo_rules as R
import reference as ref


def board(players=(0, 2), **pieces):
    """Tokens with every piece in base except the ones given, e.g. p0=[5, -1, -1, -1]."""
    tokens = {p: [-1] * 4 for p in players}
    for key, toks in pieces.items():
        tokens[int(key[1:])] = list(toks)
    return tokens


def square(player, rel):
    """Track square at relative position rel of a player's route."""
    return (R.START_POSITIONS[player] + rel) % R.BOARD_SIZE


def cell(player, k):
    """Home cell k (1..6) of a player; cell 6 is the home end."""
    return R.home_range(player)[0] + k - 1


# ---------- board ----------

def test_board_zones():
    assert R.START_POSITIONS == (0, 13, 26, 39)
    assert R.SAFE_SQUARES == {0, 8, 13, 21, 26, 34, 39, 47}
    assert [R.home_range(p) for p in range(4)] == [(52, 57), (58, 63), (64, 69), (70, 75)]


def test_two_players_sit_opposite():
    assert R.seats(2) == [0, 2]
    assert R.seats(3) == [0, 1, 2]
    assert R.seats(4) == [0, 1, 2, 3]


def test_original_rules_seat_two_players_side_by_side():
    assert R.seats(2, original=True) == [0, 1]


# ---------- leaving the base ----------

@pytest.mark.parametrize("dice", range(1, 6))
def test_leaving_base_needs_a_six(dice):
    assert R.destination(0, -1, dice) is None
    assert R.legal_moves(board(), 0, dice) == []


@pytest.mark.parametrize("player", range(4))
def test_leaving_base_lands_on_own_start_square(player):
    assert R.destination(player, -1, 6) == R.START_POSITIONS[player]


def test_leaving_base_onto_own_piece_on_start_square():
    tokens = board(p0=[0, -1, -1, -1])
    assert R.legal_moves(tokens, 0, 6) == [0, 1, 2, 3]
    after, caught = R.apply_move(tokens, 0, 1, 6)
    assert after[0][:2] == [0, 0] and caught == []


def test_leaving_base_onto_opponent_on_start_square_does_not_capture():
    tokens = board(p0=[-1, -1, -1, -1], p2=[0, -1, -1, -1])
    after, caught = R.apply_move(tokens, 0, 0, 6)
    assert caught == [] and after[0][0] == 0 and after[2][0] == 0


# ---------- movement on the track ----------

@pytest.mark.parametrize("player", range(4))
@pytest.mark.parametrize("rel", range(46, 52))
@pytest.mark.parametrize("dice", range(1, 7))
def test_home_entry_uses_every_pip(player, rel, dice):
    step = rel + dice
    expected = square(player, step) if step <= 51 else cell(player, step - 51)
    assert R.destination(player, square(player, rel), dice) == expected


def test_one_from_last_track_square_enters_first_home_cell():
    for p in range(4):
        assert R.destination(p, square(p, 51), 1) == cell(p, 1)


def test_six_from_last_track_square_finishes():
    for p in range(4):
        assert R.destination(p, square(p, 51), 6) == R.home_range(p)[1]


def test_piece_never_returns_to_its_start_square():
    for p in range(4):
        for rel in range(1, 52):
            for dice in range(1, 7):
                assert R.destination(p, square(p, rel), dice) != R.START_POSITIONS[p]


def test_route_from_start_to_home_end_is_57_pips():
    for p in range(4):
        assert R.route_index(p, R.START_POSITIONS[p]) == 0
        assert R.route_index(p, R.home_range(p)[1]) == 57


@pytest.mark.parametrize("player", [1, 2, 3])
def test_wrap_around_from_51_to_0(player):
    rel = (51 - R.START_POSITIONS[player]) % 52        # the piece is on square 51
    assert R.destination(player, 51, 1) == 0
    assert R.destination(player, 51, 4) == 3
    assert R.route_index(player, 3) == rel + 4


def test_player_zero_does_not_wrap():
    assert R.destination(0, 51, 1) == cell(0, 1)


def test_track_piece_can_always_use_its_roll():
    for p in range(4):
        for rel in range(52):
            for dice in range(1, 7):
                assert R.destination(p, square(p, rel), dice) is not None


def test_route_index_increases_along_the_route():
    for p in range(4):
        cells = [square(p, r) for r in range(52)] + [cell(p, k) for k in range(1, 7)]
        assert [R.route_index(p, c) for c in cells] == list(range(58))


# ---------- home path ----------

@pytest.mark.parametrize("k", range(1, 6))
@pytest.mark.parametrize("dice", range(1, 7))
def test_home_path_moves_any_roll_that_does_not_pass_home_end(k, dice):
    expected = cell(2, k + dice) if k + dice <= 6 else None
    assert R.destination(2, cell(2, k), dice) == expected


def test_overshoot_inside_home_path_is_illegal():
    tokens = board(p0=[cell(0, 4), -1, -1, -1])
    assert R.legal_moves(tokens, 0, 3) == []


def test_finishing_needs_the_exact_roll():
    assert R.destination(0, cell(0, 3), 3) == R.home_range(0)[1]
    assert R.destination(0, cell(0, 3), 4) is None


def test_finished_piece_never_moves():
    for dice in range(1, 7):
        assert R.destination(0, R.home_range(0)[1], dice) is None


def test_own_pieces_share_home_path_cells():
    tokens = board(p0=[cell(0, 2), cell(0, 4), -1, -1])
    assert 0 in R.legal_moves(tokens, 0, 2)
    assert R.apply_move(tokens, 0, 0, 2)[0][0][:2] == [cell(0, 4)] * 2


def test_own_pieces_share_home_end():
    end = R.home_range(0)[1]
    tokens = board(p0=[end, end, end, cell(0, 5)])
    assert R.legal_moves(tokens, 0, 1) == [3]
    after, _ = R.apply_move(tokens, 0, 3, 1)
    assert R.has_won(after, 0)


def test_no_capture_in_home_path():
    assert R.captures(board(), 0, cell(0, 3)) == []


# ---------- occupancy and capture ----------

def test_own_piece_blocks_a_non_safe_track_square():
    tokens = board(p0=[3, 5, -1, -1])
    assert R.legal_moves(tokens, 0, 2) == [1]


def test_own_pieces_share_safe_squares():
    tokens = board(p0=[5, 8, -1, -1])
    assert R.legal_moves(tokens, 0, 3) == [0, 1]


def test_pieces_jump_over_others():
    tokens = board(p0=[10, 11, 12, -1], p2=[13, 14, -1, -1])
    assert R.destination(0, 10, 6) == 16 and 0 in R.legal_moves(tokens, 0, 6)


def test_capture_on_non_safe_square_sends_piece_to_base():
    tokens = board(p0=[10, -1, -1, -1], p2=[14, -1, -1, -1])
    after, caught = R.apply_move(tokens, 0, 0, 4)
    assert caught == [(2, 0)] and after[2][0] == -1 and after[0][0] == 14


def test_no_capture_on_safe_square():
    tokens = board(p0=[5, -1, -1, -1], p2=[8, -1, -1, -1])
    after, caught = R.apply_move(tokens, 0, 0, 3)
    assert caught == [] and after[2][0] == 8 and after[0][0] == 8


def test_several_opponents_share_a_safe_square():
    tokens = board((0, 1, 2, 3), p0=[18, -1, -1, -1], p1=[21, -1, -1, -1],
                   p2=[21, -1, -1, -1], p3=[21, -1, -1, -1])
    after, caught = R.apply_move(tokens, 0, 0, 3)
    assert caught == [] and [after[p][0] for p in range(4)] == [21] * 4


def test_capture_on_opponents_start_square_is_not_allowed():
    tokens = board(p0=[22, -1, -1, -1], p2=[26, -1, -1, -1])
    assert R.apply_move(tokens, 0, 0, 4)[1] == []


def test_no_legal_move():
    assert R.legal_moves(board(p0=[cell(0, 5), -1, -1, -1]), 0, 3) == []


def test_six_with_no_legal_move():
    end = R.home_range(0)[1]
    assert R.legal_moves(board(p0=[end, end, end, cell(0, 2)]), 0, 6) == []


def test_six_that_wins():
    end = R.home_range(2)[1]
    tokens = board(p2=[end, end, end, square(2, 51)])
    assert R.legal_moves(tokens, 2, 6) == [3]
    assert R.has_won(R.apply_move(tokens, 2, 3, 6)[0], 2)


def test_legal_moves_are_ascending():
    tokens = board(p0=[30, 10, 20, -1])
    assert R.legal_moves(tokens, 0, 6) == [0, 1, 2, 3]


def test_move_outcome_flags():
    tokens = board(p0=[cell(0, 2), 10, -1, 4], p2=[14, -1, -1, -1])
    finish = R.move_outcome(tokens, 0, 0, 4)
    capture = R.move_outcome(tokens, 0, 1, 4)
    safe = R.move_outcome(tokens, 0, 3, 4)
    assert finish["home_finish"] and finish["in_home"]
    assert capture["captures"] == [(2, 0)] and not capture["safe"]
    assert safe["safe"] and safe["to"] == 8
    assert R.move_outcome(tokens, 0, 2, 6)["bring_out"]


# ---------- original rules ----------

def test_original_rules_skip_a_pip_at_home_entry():
    assert R.destination(0, 51, 1, original=True) is None
    assert R.destination(0, 51, 2, original=True) == cell(0, 1)
    assert R.route_index(0, R.home_range(0)[1], original=True) == 58


def test_original_rules_need_exact_roll_inside_home_path():
    assert R.destination(0, cell(0, 2), 2, original=True) is None
    assert R.destination(0, cell(0, 2), 4, original=True) == R.home_range(0)[1]


# ---------- agreement with the reference rules ----------

@pytest.mark.parametrize("original", [False, True])
def test_every_single_piece_position_matches_reference(original):
    checked = 0
    for p in range(4):
        cells = [-1] + [c for c in ref.route(p, original) if c != ref.PASS]
        for pos in cells:
            tokens = board((0, 1, 2, 3), **{f"p{p}": [pos, -1, -1, -1]})
            for dice in range(1, 7):
                assert R.destination(p, pos, dice, original) == ref.walk(p, pos, dice, original)
                assert R.legal_moves(tokens, p, dice, original) == ref.legal(tokens, p, dice, original)
                checked += 1
    assert checked == 4 * 59 * 6


@pytest.mark.slow
@pytest.mark.parametrize("original", [False, True])
def test_random_states_match_reference(original):
    rng = random.Random(7)
    moves = 0
    for _ in range(100_000):
        players = ref.seat_sets(rng)
        tokens = ref.random_state(rng, players)
        p, dice = rng.choice(players), rng.randint(1, 6)
        legal = R.legal_moves(tokens, p, dice, original)
        assert legal == ref.legal(tokens, p, dice, original)
        for i in legal:
            after, caught = R.apply_move(tokens, p, i, dice, original)
            ref_after, ref_caught = ref.play(tokens, p, i, dice, original)
            assert after == ref_after and set(caught) == ref_caught
            out = R.move_outcome(tokens, p, i, dice, original)
            assert out["to"] == ref_after[p][i] and set(out["captures"]) == ref_caught
            moves += 1
    assert moves > 100_000
