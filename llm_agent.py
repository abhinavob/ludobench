import re

import ludo_rules as rules
from agents import BaseAgent
from llm_backends import (DEFAULT_SCHEMA_ORDER, REASON_MAX_CHARS, SCHEMA_ORDERS,
                          parse_move_json)

# The prompt's rule and reply-format blocks. ORIGINAL_TEXT is the released
# prompt word for word (it asks for "<int> | <reason>"). OUR_TEXT changes only
# what our rules and the JSON reply need, states the direction of movement and
# corrects the turn after a 6 with no legal move; it leaves out the index-to-
# "token1" mapping, which the paper's printed prompt does not have. Its JSON
# reply format is built per agent by json_reply_format.
ORIGINAL_TEXT = {
    "important": """\
IMPORTANT:
- You must output ONLY a token index (0, 1, 2, or 3) followed by " | " and a one‑line reason.
- Do NOT output board positions or text without the token index.
- The token index MUST refer to one of your own 4 tokens (0–3).
- If you output anything else, it is invalid.
- Token index mapping (for clarity): 0→token1, 1→token2, 2→token3, 3→token4.""",
    "board": """\
1. The main circular board has 52 squares, indexed 0–51.
2. Each player has a fixed START square on the main board.
3. Tokens move forward along the main board relative to their START.
4. After completing one full lap (52 steps) from their START, tokens enter
   that player's HOME PATH.
5. Each player has a UNIQUE HOME PATH (private indices >= 52).
6. The final home position is HOME_END for your player.
7. Tokens in the home path must land EXACTLY on HOME_END to finish.
8. Overshooting HOME_END is illegal.""",
    "rules": """\
1. All tokens start in base (-1).
2. A token can leave base ONLY if the dice roll is exactly 6.
3. When leaving base, the token is placed at that player's START square.
4. Tokens move forward by the dice value.
5. Same-player stacking is allowed on SAFE squares and in home path squares.
   On NON-safe main-board squares, your own tokens cannot stack.
6. CAPTURE rule (main board only): if you land on an opponent token on
   a NON-safe square, that opponent token is captured and sent to base (-1).
7. Captures NEVER happen on safe squares or in any home path.
8. Safe squares are positions where tokens cannot be captured.
9. Tokens on safe squares are fully protected.
10. Rolling a 6 grants an extra turn.
11. If no legal move exists, the turn is skipped.
12. The first player to move ALL tokens to HOME_END wins.
13. Home paths are private: ONLY your tokens can enter your home path.""",
    "path_note": """\
PATH NOTE:
- Movement is RELATIVE to each player's START square.
- After a full lap (52 steps) from your START, you enter YOUR home path.
- Other players have DIFFERENT starts and DIFFERENT home path ranges.
- Home path ranges are player-specific; only that player can be there.
- Safe squares are SHARED board squares (0–51) for all players.""",
    "output": """\
- Return exactly one line in the format:
  <int> | <one line reason>
- The line MUST start with the integer
- The integer MUST be a TOKEN INDEX for your own pieces.
- In this game, each player has 4 tokens, so valid indices are: 0, 1, 2, 3.
- If the game ever changes the number of tokens per player, use 0..(tokens_per_player-1).
- The reason MUST be a single line (no newlines)
- Do NOT return anything else""",
}

OUR_TEXT = {
    "important": """\
IMPORTANT:
- You must output ONLY one JSON object (see OUTPUT FORMAT).
- Do NOT output board positions or text without the token index.
- The token index MUST refer to one of your own 4 tokens (0–3).
- If you output anything else, it is invalid.""",
    "board": """\
1. The main circular board has 52 squares, indexed 0–51.
   Tokens move to higher indices, wrapping from 51 to 0.
2. Each player has a fixed START square on the main board.
3. Tokens move forward along the main board relative to their START.
4. After 50 steps from START, the next step enters that player's HOME PATH.
5. Each player has a UNIQUE HOME PATH (private indices >= 52).
6. The final home position is HOME_END for your player.
7. Tokens in the home path must land EXACTLY on HOME_END to finish.
8. Tokens in the home path move by any roll that does not pass HOME_END.
9. Overshooting HOME_END is illegal.""",
    "rules": ORIGINAL_TEXT["rules"].replace(
        "11. If no legal move exists, the turn is skipped.",
        "11. If no legal move exists, the turn is skipped (after a 6, roll again)."),
    "path_note": ORIGINAL_TEXT["path_note"].replace(
        "- After a full lap (52 steps) from your START, you enter YOUR home path.",
        "- After 50 steps from your START, the next step enters YOUR home path."),
}


def json_reply_format(schema_order=DEFAULT_SCHEMA_ORDER, reason_max_chars=REASON_MAX_CHARS):
    """OUTPUT FORMAT block for a JSON reply with its fields in schema order."""
    reason = "one line" + (f", at most {reason_max_chars} characters" if reason_max_chars else "")
    fields = {"reason": f'"reason": "<{reason}>"', "piece": '"piece": <token index>'}
    example = "{" + ", ".join(fields[f] for f in SCHEMA_ORDERS[schema_order]) + "}"
    return (f"- Return exactly one JSON object: {example}\n"
            '- "piece" MUST be a TOKEN INDEX for your own pieces.\n'
            "- In this game, each player has 4 tokens, so valid indices are: 0, 1, 2, 3.\n"
            "- If the game ever changes the number of tokens per player, use "
            "0..(tokens_per_player-1).\n"
            "- Do NOT return anything else")


class LLMAgent(BaseAgent):
    def __init__(self, llm_call_fn, include_legal_moves_in_prompt=True,
                 robust_parse=False, schema_order=DEFAULT_SCHEMA_ORDER,
                 reason_max_chars=REASON_MAX_CHARS, structured=False):
        self.llm_call = llm_call_fn
        self.include_legal_moves_in_prompt = include_legal_moves_in_prompt
        # robust_parse=False reproduces the released behaviour exactly (first line
        # only). Set True for reasoning models, whose answer comes AFTER a long
        # reasoning block, so the first line is never the answer.
        self.robust_parse = robust_parse
        # Under our rules the prompt asks for a JSON reply in this field order.
        self.schema_order = schema_order
        self.reason_max_chars = reason_max_chars
        # The reply is JSON: always under our rules; with original rules only
        # when the backend constrains it (structured output).
        self.structured = structured

    def select_move(self, env, player, dice, legal_moves):

        # import time#this thing can be removed later on
        # time.sleep(10)


        your_tokens = env.tokens[player]
        other_players = {
            pid: env.tokens[pid] for pid in env.player_ids if pid != player
        }
        safe_squares = sorted(list(env.SAFE_SQUARES))
        legal_moves = legal_moves
        
        home_start, home_end = rules.home_range(player)
        player_info = {
            pid: {
                "start": env.START_POSITIONS[pid],
                "home_start": rules.home_range(pid)[0],
                "home_end": rules.home_range(pid)[1]
            }
            for pid in env.player_ids
        }
        state = {
            "dice": dice,
            "your_tokens": env.tokens[player],
            "other_players_tokens": other_players,
            "safe_squares": sorted(list(env.SAFE_SQUARES)),
            "legal_moves": legal_moves,
            "home_path_start": home_start,
            "home_path_end": home_end,
            "player_info": player_info
        }

        history_block = ""
        if getattr(env, "history_text", None):
            history_block = f"""
────────────────────
HISTORY / CONTEXT
────────────────────

{env.history_text}
"""
        persona_block = ""
        if getattr(env, "persona_text", None):
            persona_block = f"""
────────────────────
PERSONA
────────────────────

You must play with this persona style:
{env.persona_text}
"""

        text = (ORIGINAL_TEXT if env.original_rules
                else dict(OUR_TEXT, output=json_reply_format(self.schema_order,
                                                             self.reason_max_chars)))
        prompt = f"""
You are an AI agent playing Ludo as Player {player}. Your job is to choose exactly ONE of your token indices.

{text['important']}

All positions are given using an INTERNAL INDEXING SYSTEM described below.

────────────────────
BOARD & POSITION SYSTEM
────────────────────

{text['board']}

────────────────────
TOKEN STATES
────────────────────

- Position = -1   → Token is in base (not yet entered)
- Position 0–51   → Token is on the main board
- Position >= 52  → Token is in a home path (player-specific)
- HOME_END        → Token has reached home successfully for your player

────────────────────
GAME RULES
────────────────────

{text['rules']}

────────────────────
CURRENT GAME STATE
────────────────────

Number of players: {env.NUM_PLAYERS}
Active player ids: {env.player_ids}

Dice rolled: {dice}

Your token positions (Player {player}):
{your_tokens}

Other players' token positions:
{other_players}

Your start square:
{env.START_POSITIONS[player]}

Your home path range (for Player {player}):
{home_start} to {home_end}

Safe squares:
{safe_squares}

PLAYER PATH RANGES (START / HOME START / HOME END):
{player_info}

{text['path_note']}

{persona_block}

{history_block}

{"" if not self.include_legal_moves_in_prompt else f"""Legal moves:
The following token indices are legal to move this turn:
{legal_moves}
"""}

REMINDER: You must choose ONE token index that is legal for this turn.

────────────────────
YOUR TASK
────────────────────

Choose the BEST legal move to help you win the game.

────────────────────
OUTPUT FORMAT (STRICT)
────────────────────

{text['output']}




"""
# You are playing Ludo.

# Rules:
# - Tokens start in base (-1) and enter only on dice = 6.
# - Captures are not allowed on safe squares.
# - Exact dice required to reach home.

# Game state:
# {state}

# Objective:
# Play optimally to win.

# Output:
# Return ONLY the token index.

        response = self.llm_call(prompt)

        # 🔹 LOG PROMPT + RESPONSE
        env.log_file.write(
            f"\n[LLM CALL]\nPrompt:\n{prompt}\nResponse:\n{response}\n\n"
        )
        env.log_file.flush()

        # Optional LLM-only log
        if hasattr(env, "llm_log_file") and env.llm_log_file is not None:
            env.llm_log_file.write(
                f"\n[LLM CALL]\nPrompt:\n{prompt}\nResponse:\n{response}\n\n"
            )
            env.llm_log_file.flush()

        
        env.llm_calls += 1
        # Keep the raw reply and reset per-call fields so a stale reason from an
        # earlier turn is never shown next to this turn's move.
        env.last_llm_raw = (response or "").strip()
        env.last_llm_reason = ""
        env.last_llm_fallback = False
        # match = re.search(r"-?\d+", response or "")
        # TEMP: tolerate malformed LLM outputs so runs don't crash.
        # Remove this once LLM reliably returns "<int> | <reason>".
        try:
            answer, env.last_llm_reason = self.parse_reply(
                (response or "").strip(), legal_moves, env.original_rules)
        except Exception:
            env.llm_fallbacks += 1
            env.last_llm_fallback = True
            return env.rng["fallback"].choice(legal_moves)

        if answer in legal_moves:
            return answer
        env.llm_fallbacks += 1
        env.last_llm_fallback = True
        return env.rng["fallback"].choice(legal_moves)

    def parse_reply(self, text, legal_moves, original_rules):
        """
        (token index, reason) read from a reply; raises if there is none. A JSON
        reply must be a {reason, piece} object; "<int> | <reason>" is read only
        with original rules and no structured output.
        """
        if self.structured or not original_rules:
            parsed = parse_move_json(text)
            if parsed is None:
                raise ValueError("not a {reason, piece} JSON object")
            return parsed
        if self.robust_parse:
            return self._parse_robust(text, legal_moves)
        first_line = text.splitlines()[0] if text else ""
        if "|" in first_line:
            left, _, reason = first_line.partition("|")
            return int(left.strip()), reason.strip()
        # Fallback: extract first integer from entire response
        m = re.search(r"-?\d+", text)
        if not m:
            raise ValueError("no integer in response")
        return int(m.group()), ""

    @staticmethod
    def _parse_robust(text, legal_moves):
        """
        Find the answer in a reply that may open with a long reasoning block.
        Scans from the END, because the answer comes last. Returns (index, reason).
        """
        if not text:
            raise ValueError("empty response")

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        # 1. last line shaped like "<int> | <reason>"
        for ln in reversed(lines):
            m = re.match(r"^\**\s*(-?\d+)\s*\**\s*\|(.*)$", ln)
            if m:
                return int(m.group(1)), m.group(2).strip()

        # 2. last line that is just an integer
        for ln in reversed(lines):
            m = re.match(r"^\**\s*(-?\d+)\s*\**\s*$", ln)
            if m:
                return int(m.group(1)), ""

        # 3. last integer anywhere that is actually a legal token index
        for m in reversed(list(re.finditer(r"-?\d+", text))):
            val = int(m.group())
            if val in legal_moves:
                return val, ""

        raise ValueError("no usable token index in response")

        # if match:
        #     choice = int(match.group())
        #     if choice in legal_moves:
        #         return choice
     
        # env.llm_fallbacks += 1
        # return random.choice(legal_moves)


'''
        ################################################################################
        try:#this is to handle llm output which is not a number , later on remove this
            choice = int(response.strip())
            if choice in legal_moves:
                return choice
        except Exception:
            pass

        # fallback (MANDATORY)
        return random.choice(legal_moves)
        #################################################################################
'''