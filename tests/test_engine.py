"""Engine: seating, first player, turn order, extra turns, random streams, and full games."""
import io
import json
import os
import random
import tempfile
from multiprocessing import Pool

import pytest

import ludo_rules as R
import reference as ref
import run_games
from agents import HeuristicAgent, RandomAgent
from ludo_env import LudoEnv
from run_games import FirstLegalAgent


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)            # LudoEnv writes its text logs to the working directory


def quiet(env):
    """Keep the engine's per-move text logs in memory."""
    for f in (env.log_file, env.llm_log_file, *env.player_logs.values()):
        f.close()
    env.log_file, env.llm_log_file = io.StringIO(), io.StringIO()
    env.player_logs = {p: io.StringIO() for p in env.player_ids}
    return env


class Fixed:
    """Agent that always moves the given token if legal, else the first legal one."""

    def __init__(self, token=0):
        self.token = token

    def select_move(self, env, player, dice, legal_moves):
        return self.token if self.token in legal_moves else legal_moves[0]


def staged(tokens, dice, seats=(0, 2), first=0, original=False):
    """An engine on a given board whose dice come from a list."""
    env = quiet(LudoEnv(seed=0, player_ids=list(seats), original_rules=original))
    env.current_player_idx = first
    for p, t in tokens.items():
        env.tokens[p][:] = t
    rolls = iter(dice)
    env.roll_dice = lambda: next(rolls)
    return env


def turns(env, agents, n):
    """Player to move before each of n steps."""
    out = []
    for _ in range(n):
        out.append(env.player_ids[env.current_player_idx])
        env.step(agents)
    return out


# ---------- seating and first player ----------

def test_seats_by_number_of_players():
    assert [LudoEnv(num_players=n).player_ids for n in (2, 3, 4)] == [[0, 2], [0, 1, 2], [0, 1, 2, 3]]


def test_original_rules_seat_two_players_side_by_side_and_seat_0_starts():
    env = LudoEnv(num_players=2, original_rules=True)
    assert env.player_ids == [0, 1] and env.first_player == 0 and env.roll_off == []


@pytest.mark.parametrize("n", [2, 3, 4])
def test_roll_off_highest_roll_starts_and_only_tied_players_reroll(n):
    ties = 0
    for seed in range(300):
        env = LudoEnv(seed=seed, num_players=n)
        rounds = env.roll_off
        assert [p for p, _ in rounds[0]] == env.player_ids
        for this, nxt in zip(rounds, rounds[1:]):
            top = max(r for _, r in this)
            assert [p for p, _ in nxt] == [p for p, r in this if r == top]
        top = max(r for _, r in rounds[-1])
        assert [p for p, r in rounds[-1] if r == top] == [env.first_player]
        assert env.player_ids[env.current_player_idx] == env.first_player
        ties += len(rounds) > 1
    assert ties > 0


def test_roll_off_uses_the_dice_stream():
    env = LudoEnv(seed=5, num_players=4)
    dice = random.Random("5:dice")
    assert [r for rnd in env.roll_off for _, r in rnd] == \
        [dice.randint(1, 6) for _ in range(sum(map(len, env.roll_off)))]
    assert env.roll_dice() == dice.randint(1, 6)


def test_first_player_depends_only_on_the_seed():
    assert [LudoEnv(seed=s, num_players=4).first_player for s in range(20)] == \
        [LudoEnv(seed=s, num_players=4).first_player for s in range(20)]


# ---------- turns ----------

def test_turn_passes_in_seat_order():
    env = staged({}, [1, 2, 3, 4, 5, 1], seats=(0, 1, 2))
    assert turns(env, {}, 6) == [0, 1, 2, 0, 1, 2]


def test_two_player_turns_alternate_between_seats_0_and_2():
    env = staged({}, [1, 2, 3], seats=(0, 2))
    assert turns(env, {}, 3) == [0, 2, 0]


def test_six_gives_another_roll():
    env = staged({0: [10, -1, -1, -1]}, [6, 1, 2])
    assert turns(env, {0: Fixed(0), 2: Fixed()}, 3) == [0, 0, 2]


def test_six_with_no_legal_move_rolls_again():
    end = R.home_range(0)[1]
    env = staged({0: [end, end, end, R.home_range(0)[0] + 1]}, [6, 6, 1])
    assert turns(env, {0: Fixed(), 2: Fixed()}, 3) == [0, 0, 0]


def test_capture_does_not_give_another_roll():
    env = staged({0: [10, -1, -1, -1], 2: [14, -1, -1, -1]}, [4, 1])
    assert turns(env, {0: Fixed(0), 2: Fixed()}, 2) == [0, 2]
    assert env.tokens[2][0] == -1


def test_finishing_a_piece_does_not_give_another_roll():
    env = staged({0: [R.home_range(0)[0] + 2, 10, -1, -1]}, [3, 1])
    assert turns(env, {0: Fixed(0), 2: Fixed()}, 2) == [0, 2]
    assert env.tokens[0][0] == R.home_range(0)[1]


def test_no_legal_move_passes_the_turn():
    env = staged({0: [R.home_range(0)[0] + 4, -1, -1, -1]}, [3, 1])
    assert turns(env, {0: Fixed(), 2: Fixed()}, 2) == [0, 2]


def test_six_that_wins_ends_the_game():
    end = R.home_range(0)[1]
    env = staged({0: [end, end, end, 50]}, [6])
    env.play_game({0: Fixed(3), 2: Fixed()})
    assert env.game_over and env.winner == 0 and env.tokens[0] == [end] * 4


# ---------- random streams ----------

def test_each_source_of_randomness_has_its_own_stream():
    env = LudoEnv(seed=9)
    assert len({id(g) for g in env.rng.values()}) == len(env.rng) == 4
    legal = [0, 1, 2, 3]
    assert RandomAgent().select_move(env, 0, 6, legal) == random.Random("9:random_player").choice(legal)


def test_original_rules_share_the_global_generator():
    env = LudoEnv(seed=9, original_rules=True)
    assert all(g is random for g in env.rng.values())
    random.seed(9)
    assert env.roll_dice() == random.Random(9).randint(1, 6)


def dice_of_game(seed, agents_for):
    env = quiet(LudoEnv(seed=seed, num_players=2))
    rolled = []
    roll = env.roll_dice
    env.roll_dice = lambda: rolled.append(roll()) or rolled[-1]
    env.play_game({p: a for p, a in zip(env.player_ids, agents_for)})
    return env.first_player, rolled


def test_same_seed_gives_same_dice_whoever_plays():
    for seed in range(20):
        first_a, a = dice_of_game(seed, (RandomAgent(), HeuristicAgent()))
        first_b, b = dice_of_game(seed, (FirstLegalAgent(), RandomAgent()))
        n = min(len(a), len(b))
        assert first_a == first_b and a[:n] == b[:n]


def test_shuffled_legal_order_is_reproducible(tmp_path, monkeypatch):
    prompts = []

    def make_llm(spec, **_):
        def call(prompt):
            prompts.append(prompt)
            return "0"
        call.spec, call.base_url, call.last_content = spec, None, None
        return call
    monkeypatch.setattr(run_games, "make_llm", make_llm)
    monkeypatch.setattr(run_games, "model_info", lambda *a: {"capabilities": []})
    for out in ("a", "b"):
        run_games.run_games(["llm:x", "heuristic"], games=1, seed0=3,
                            out_dir=str(tmp_path / out), legal_order="shuffled")
    a, b = prompts[:len(prompts) // 2], prompts[len(prompts) // 2:]
    assert a == b
    lists = [json.loads(p.split(run_games.LEGAL_LIST_LEAD)[1].split("\n")[0]) for p in a]
    assert any(lst != sorted(lst) for lst in lists)


# ---------- full games against the reference rules ----------

def checked_game(seed, n, kinds):
    """Play one bot game, checking every roll against the reference rules; return rolls."""
    env = quiet(LudoEnv(seed=seed, num_players=n))
    seen = {}

    class Watch:
        def __init__(self, inner):
            self.inner = inner

        def select_move(self, env, player, dice, legal_moves):
            seen["legal"] = list(legal_moves)
            return self.inner.select_move(env, player, dice, legal_moves)

    agents = {p: Watch(k()) for p, k in zip(env.player_ids, kinds)}
    roll = env.roll_dice
    env.roll_dice = lambda: seen.setdefault("dice", []).append(roll()) or seen["dice"][-1]
    rolls = 0
    while not env.game_over:
        assert rolls < 1000, f"seed {seed}: no winner after 1000 rolls"
        player = env.player_ids[env.current_player_idx]
        before = {p: list(t) for p, t in env.tokens.items()}
        moved_before = len(env.move_log)
        seen.pop("legal", None)
        env.step(agents)
        dice = seen["dice"][-1]
        legal = ref.legal(before, player, dice)
        if not legal:
            assert "legal" not in seen and env.tokens == before
        else:
            assert seen["legal"] == legal
            token = [i for i in range(4) if env.tokens[player][i] != before[player][i]]
            after, _ = ref.play(before, player, token[0], dice)
            assert len(env.move_log) == moved_before + 1 and env.tokens == after
        won = all(x == R.home_range(player)[1] for x in env.tokens[player])
        assert env.game_over == won
        if not won:
            i = env.player_ids.index(player)
            expected = player if dice == 6 else env.player_ids[(i + 1) % n]
            assert env.player_ids[env.current_player_idx] == expected
        rolls += 1
    assert env.winner is not None
    return rolls


BOTS = [RandomAgent, HeuristicAgent, FirstLegalAgent]
GAMES = 10_000


def checked_games(job):
    """Worker: play games n players, seeds start, start+step, ...; return their roll counts."""
    n, start, step = job
    here = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            return [checked_game(seed, n, random.Random(f"{n}:{seed}").choices(BOTS, k=n))
                    for seed in range(start, GAMES, step)]
        finally:
            os.chdir(here)


@pytest.mark.slow
@pytest.mark.parametrize("n", [2, 3, 4])
def test_ten_thousand_bot_games_follow_the_reference_rules(n):
    workers = min(8, os.cpu_count() or 1)
    with Pool(workers) as pool:
        rolls = [r for part in pool.map(checked_games, [(n, i, workers) for i in range(workers)])
                 for r in part]
    assert len(rolls) == GAMES and max(rolls) < 1000
