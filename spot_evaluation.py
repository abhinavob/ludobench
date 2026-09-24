import io
import json
import os
import re

from agents import HeuristicAgent
from llm_agent import LLMAgent


class SpotEnv:
    def __init__(self, player_ids, tokens):
        self.player_ids = list(player_ids)
        self.NUM_PLAYERS = len(self.player_ids)
        self.TOKENS_PER_PLAYER = 4
        self.BOARD_SIZE = 52
        self.HOME_START = 52
        self.HOME_LEN = 6
        self.SAFE_SQUARES = {0, 8, 13, 21, 26, 34, 39, 47}

        base_starts = [0, 13, 26, 39]
        self.START_POSITIONS = {pid: base_starts[pid] for pid in self.player_ids}

        self.tokens = tokens

        # LLM logging fields expected by LLMAgent
        self.log_file = io.StringIO()
        self.llm_calls = 0
        self.llm_fallbacks = 0


def compute_legal_moves(env, player, dice):
    legal = []
    start = env.START_POSITIONS[player]
    home_start = env.HOME_START + player * env.HOME_LEN
    home_end = home_start + env.HOME_LEN - 1

    for i, pos in enumerate(env.tokens[player]):
        if pos == -1:
            if dice == 6:
                legal.append(i)
        elif 0 <= pos < env.BOARD_SIZE:
            rel_pos = (pos - start) % env.BOARD_SIZE
            new_rel = rel_pos + dice

            if new_rel < env.BOARD_SIZE:
                new_pos = (start + new_rel) % env.BOARD_SIZE
                # Same-player stacking is allowed on safe squares only.
                if new_pos not in env.tokens[player] or new_pos in env.SAFE_SQUARES:
                    legal.append(i)
            else:
                home_step = new_rel - env.BOARD_SIZE
                if 1 <= home_step <= env.HOME_LEN:
                    new_pos = home_start + home_step - 1
                    # Home-path stacking is allowed.
                    legal.append(i)
        elif home_start <= pos <= home_end:
            new_pos = pos + dice
            if new_pos == home_end:
                legal.append(i)

    return legal


def evaluate_spot(
    players,
    tokens,
    dice,
    current_player,
    legal_moves=None,
    llm_call=None,
    llm_output_path="spot_llm.txt",
    history_text=None,
    persona=None,
):
    """
    Spot evaluation:
    (current board state + dice roll + legal moves) -> best move

    Args:
        players: list of player ids (e.g., [0, 1, 2]) in the match
        tokens: dict {player_id: [pos0, pos1, pos2, pos3]}
        dice: int (1-6)
        current_player: int (player id)
        legal_moves: optional list of legal token indices (computed if None)
        llm_call: required function for LLM call
        llm_output_path: path to write LLM prompt/response JSON
        history_text: optional string with prior context to include in prompt
        persona: optional persona label/instruction to include in prompt
    """
    if llm_call is None:
        raise ValueError("llm_call is required; no dummy fallback is used.")

    env = SpotEnv(players, tokens)
    if history_text:
        env.history_text = history_text
    if persona:
        env.persona_text = persona
    if legal_moves is None:
        legal_moves = compute_legal_moves(env, current_player, dice)
    if not legal_moves:
        raise ValueError("No legal moves for this spot; adjust tokens or dice.")

    llm_agent = LLMAgent(llm_call, include_legal_moves_in_prompt=False)
    heuristic_agent = HeuristicAgent()

    llm_move = llm_agent.select_move(env, current_player, dice, legal_moves)
    heuristic_move = heuristic_agent.select_move(env, current_player, dice, legal_moves)

    # Extract LLM output (prompt + response) from the in-memory log and write JSON
    log_text = env.log_file.getvalue()
    prompt_match = re.search(r"\[LLM CALL\]\nPrompt:\n(.*?)\nResponse:\n", log_text, re.S)
    response_match = re.search(r"\[LLM CALL\]\nPrompt:\n.*?\nResponse:\n(.*?)\n\n", log_text, re.S)
    prompt_text = prompt_match.group(1) if prompt_match else ""
    response_text = response_match.group(1) if response_match else ""

    response_int = None
    response_reason = ""
    try:
        text = (response_text or "").strip()
        first_line = text.splitlines()[0] if text else ""
        if "|" in first_line:
            left, _, reason = first_line.partition("|")
            response_int = int(left.strip())
            response_reason = reason.strip()
        else:
            m = re.search(r"-?\\d+", text)
            if m:
                response_int = int(m.group())
    except Exception:
        response_int = None

    llm_output = {
        "players": env.player_ids,
        "current_player": current_player,
        "dice": dice,
        "legal_moves": legal_moves,
        "history_text": history_text or "",
        "persona": persona or "",
        "prompt": prompt_text,
        "response_raw": response_text,
        "response_int": response_int,
        "response_reason": response_reason,
        "llm_move": llm_move,
        "llm_calls": env.llm_calls,
        "llm_fallbacks": env.llm_fallbacks
    }

    out_dir = os.path.dirname(llm_output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(llm_output_path, "w") as f:
        json.dump(llm_output, f, ensure_ascii=True, indent=2)

    summary = (
        "Spot Evaluation\n"
        f"players: {env.player_ids}\n"
        f"current_player: {current_player}  dice: {dice}\n"
        f"legal_moves: {legal_moves}\n"
        f"llm_move: {llm_move}  (fallbacks {env.llm_fallbacks}/{max(1, env.llm_calls)})\n"
        f"heuristic_move: {heuristic_move}"
    )

    return {
        "players": env.player_ids,
        "current_player": current_player,
        "dice": dice,
        "legal_moves": legal_moves,
        "persona": persona or "",
        "response_int": response_int,
        "response_reason": response_reason,
        "response_raw": response_text,
        "llm_move": llm_move,
        "heuristic_move": heuristic_move,
        "llm_calls": env.llm_calls,
        "llm_fallbacks": env.llm_fallbacks,
        "summary": summary
    }
