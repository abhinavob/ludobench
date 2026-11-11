"""Every number in the prompt's game-state part matches the engine, over 1,000 decisions."""
import json
import random

import pytest

import ludo_rules as R
from llm_agent import OUR_TEXT
from ludo_env import LudoEnv
from run_games import TrackedLLMAgent, attach_last_move

HISTORY = 5


def where(pos):
    """A position as the prompt's outcome and history lines name it."""
    if pos == -1:
        return "base"
    return f"square {pos}" if pos < R.BOARD_SIZE else f"home-path position {pos}"


def outcome_line(tokens, player, dice, token):
    out = R.move_outcome(tokens, player, token, dice)
    notes = (["brings a new token out of base"] * out["bring_out"]
             + (["finishes this token"] if out["home_finish"] else
                ["is in your home path after the move"] if out["in_home"] else [])
             + ["captures an opponent token"] * bool(out["captures"])
             + ["lands on a safe square"] * out["safe"])
    return (f"- Token {token}: moves from {where(out['from'])} to {where(out['to'])}"
            + "".join(", " + n for n in notes) + ".")


def history_lines(moves, viewer):
    """The history block for moves recorded outside the engine: (player, dice, token, before, after)."""
    lines = []
    for player, dice, token, before, after in moves[-HISTORY:]:
        line = (f"{'You' if player == viewer else f'Player {player}'} rolled {dice} and moved "
                f"token {token} from {where(before[player][token])} to {where(after[player][token])}")
        if after[player][token] == R.home_range(player)[1]:
            line += ", finishing that token"
        for pid in sorted(before):
            if pid != player and before[pid] != after[pid]:
                whose = "your" if pid == viewer else f"Player {pid}'s"
                line += f", capturing one of {whose} tokens"
        lines.append(line + ".")
    if len(lines) == 1:
        return ["HISTORY / CONTEXT", "Previous move: " + lines[0]]
    return ["HISTORY / CONTEXT", f"Last {len(lines)} moves, oldest first:"] + ["- " + ln for ln in lines]


def check_prompt(prompt, tokens, players, player, dice, legal, moves):
    """Compare every line of the prompt's game-state part with the engine's state."""
    state = prompt.split("CURRENT GAME STATE", 1)[1].split("REMINDER:", 1)[0]
    lines = [ln for ln in state.splitlines() if ln.strip() and set(ln) != {"─"}]
    home_start, home_end = R.home_range(player)
    info = {p: {"start": R.START_POSITIONS[p], "home_start": R.home_range(p)[0],
                "home_end": R.home_range(p)[1]} for p in players}
    assert lines == [
        f"Number of players: {len(players)}", f"Active player ids: {players}",
        f"Dice rolled: {dice}",
        f"Your token positions (Player {player}):", str(tokens[player]),
        "Other players' token positions:", str({p: tokens[p] for p in players if p != player}),
        "Your start square:", str(R.START_POSITIONS[player]),
        f"Your home path range (for Player {player}):", f"{home_start} to {home_end}",
        "Safe squares:", str(sorted(R.SAFE_SQUARES)),
        "PLAYER PATH RANGES (START / HOME START / HOME END):", str(info),
        *OUR_TEXT["path_note"].splitlines(),
        *(history_lines(moves, player) if moves else []),
        "Legal moves:", "The following token indices are legal to move this turn:", str(legal),
        "What each legal move does:", *[outcome_line(tokens, player, dice, t) for t in legal],
    ]
    assert legal == R.legal_moves(tokens, player, dice)
    assert f"playing Ludo as Player {player}." in prompt


@pytest.mark.parametrize("n", [2, 3, 4])
def test_prompt_numbers_match_the_engine(n, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                       # LudoEnv writes its text logs here
    rng, checked, seed = random.Random(n), [], 0
    while len(checked) < 334:
        env = LudoEnv(seed=seed, num_players=n)
        attach_last_move(env, keep=HISTORY)
        roll, rolls, moves, turn = env.roll_dice, [], [], {}
        env.roll_dice = lambda: rolls.append(roll()) or rolls[-1]

        def reply(prompt):
            if rng.random() < 0.1:
                check_prompt(prompt, env.tokens, env.player_ids, *turn["now"], moves)
                checked.append(turn["now"])
            return json.dumps({"reason": "", "piece": rng.choice(turn["now"][2])})

        class Agent(TrackedLLMAgent):
            def select_move(self, env, player, dice, legal_moves):
                turn["now"] = (player, dice, list(legal_moves))
                return super().select_move(env, player, dice, legal_moves)

        agents = {p: Agent(reply, structured=True, last_move=HISTORY, move_outcomes=True)
                  for p in env.player_ids}
        while not env.game_over:
            player = env.player_ids[env.current_player_idx]
            before = {p: list(t) for p, t in env.tokens.items()}
            env.step(agents)
            after = {p: list(t) for p, t in env.tokens.items()}
            if after != before:
                token = next(i for i in range(4) if after[player][i] != before[player][i])
                moves.append((player, rolls[-1], token, before, after))
        seed += 1
    assert len({c[1] for c in checked}) == 6                 # every roll value was seen
