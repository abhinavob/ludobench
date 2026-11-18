"""analysis/search_validation.py: weighted variants match the originals; agreement and games are counted right."""
import os

import pytest

import ludo_rules as R
from analysis import search_validation as V
from game_theory_agent import GameTheoryAgent
from game_theory_multiplayer_agent import GameTheoryMultiplayerAgent
from spot_evaluation import SpotEnv

SPOTS = V.load_spots()


def env_of(s):
    return SpotEnv(s["players"], {int(p): list(t) for p, t in s["tokens"].items()})


@pytest.mark.parametrize("depth", [1, 2])
def test_original_weights_choose_exactly_as_the_search_player(depth):
    ours, theirs = V.WeightedSearch(depth, V.WEIGHTS), GameTheoryMultiplayerAgent(depth)
    for s in SPOTS:
        env, p, d = env_of(s), s["current_player"], s["dice"]
        legal = R.legal_moves(env.tokens, p, d)
        if legal:
            assert ours.select_move(env, p, d, legal) == theirs.select_move(env, p, d, legal), s["id"]


def test_original_weights_score_positions_exactly_as_both_evaluators():
    for s in SPOTS[::7]:
        state = {int(p): tuple(t) for p, t in s["tokens"].items()}
        ids = tuple(s["players"])
        if len(ids) == 2:
            pair = [GameTheoryAgent(2), V.WeightedTwoPlayer(2, V.WEIGHTS)]
        else:
            pair = [GameTheoryMultiplayerAgent(2), V.WeightedSearch(2, V.WEIGHTS)]
        for a in pair:
            a._root_player, a._player_ids, a._original = s["current_player"], ids, False
            a._idx = {pid: i for i, pid in enumerate(ids)}
        assert pair[0]._evaluate_state(state) == pair[1]._evaluate_state(state), s["id"]


def test_variants_are_seeded_and_within_their_spread():
    specs = V.variants()
    assert specs == V.variants() and len(specs) == 3 + len(V.SPREADS) * V.DRAWS
    for name, (depth, weights) in specs.items():
        if weights:
            spread = 0.25 if "25%" in name else 0.5
            assert depth == 2 and all(abs(weights[k] / V.WEIGHTS[k] - 1) <= spread for k in V.WEIGHTS)


def test_spot_agreement_counts_spots_with_a_real_choice():
    specs = {"depth 1": (1, None), "depth 2": (2, None), "weights ±25% #1": V.variants()["weights ±25% #1"]}
    choices = V.spot_choices(specs, SPOTS[:60])
    assert all(len(set(c["group"].values())) >= 2 for c in choices) and len(choices) <= 60
    rows = V.spot_agreement(choices, V.spot_pairs(list(specs)))
    assert {(r["a"], r["b"]) for r in rows} == {("depth 1", "depth 2"), ("depth 2", "weights ±25% #1")}
    total = next(r for r in rows if r["category"] == "all" and r["b"] == "depth 2")
    assert total["spots"] == len(choices) and 0 <= total["agreement"] <= 1
    assert V.spot_agreement(choices, [("depth 2", "depth 2")])[0]["agreement"] == 1


def test_head_to_head_plays_each_seed_in_both_seats(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    specs = {"depth 1": (1, None), "same": (1, None)}
    rows = {r["seats"]: r for r in V.head_to_head(specs, [("depth 1", "same")], games=6)}
    assert [rows[s]["games"] for s in ("as run", "swapped", "both")] == [6, 6, 12]
    for r in rows.values():
        assert r["wins_a"] + r["wins_b"] + r["no_winner"] == r["games"]
    # identical players: the swapped game is the same game with the seats exchanged
    assert (rows["swapped"]["wins_a"], rows["swapped"]["wins_b"]) == (rows["as run"]["wins_b"],
                                                                       rows["as run"]["wins_a"])
    both = rows["both"]
    assert both["wins_a"] == both["wins_b"] == 6 and both["seeds_a_both"] == both["seeds_b_both"] == 0
    assert os.getcwd() == str(tmp_path) and not list(tmp_path.iterdir())   # logs went elsewhere
