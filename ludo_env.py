import random

import ludo_rules as rules

STREAMS = ("dice", "random_player", "fallback", "legal_order")


def random_streams(seed=None):
    """
    One generator per source of randomness, each seeded from (seed, its label),
    so the dice of a seed are the same whoever plays. Without a seed every
    source shares the global generator, as in the released code.
    """
    if seed is None:
        return dict.fromkeys(STREAMS, random)
    return {label: random.Random(f"{seed}:{label}") for label in STREAMS}


class LudoEnv:
    def __init__(self, seed=42, num_players=2, player_ids=None, original_rules=False):
        # True: everything as in the released LudoBench code: its movement rules,
        # seats 0..n-1, seat 0 starts, one shared random generator
        self.original_rules = original_rules
        if original_rules:
            random.seed(seed)
        self.rng = random_streams(None if original_rules else seed)
        # Move log (list of dicts)
        self.log_file = open("game_log.txt", "w", encoding="utf-8")
        self.llm_log_file = open("llm_log.txt", "w", encoding="utf-8")
        if player_ids is None:
            player_ids = rules.seats(num_players, original_rules)
        self.player_ids = list(player_ids)
        self.NUM_PLAYERS = len(self.player_ids)
        self.player_logs = {
            pid: open(f"player{pid + 1}.txt", "w", encoding="utf-8") for pid in self.player_ids
        }
        self.move_counts = {pid: 0 for pid in self.player_ids}  # Track move numbers for each player
        self.move_log = []
        # LLM fallback tracking
        self.llm_calls = 0
        self.llm_fallbacks = 0
        # Constants
        self.TOKENS_PER_PLAYER = rules.TOKENS_PER_PLAYER
        self.BOARD_SIZE = rules.BOARD_SIZE
        self.HOME_START = rules.HOME_START
        self.HOME_LEN = rules.HOME_LEN
        self.START_POSITIONS = {pid: rules.START_POSITIONS[pid] for pid in self.player_ids}
        self.SAFE_SQUARES = rules.SAFE_SQUARES

        # Game state
        self.roll_off = []             # rounds of [player, roll] that chose the first player
        self.current_player_idx = 0 if original_rules else self._roll_off()
        self.first_player = self.player_ids[self.current_player_idx]
        self.tokens = {
            pid: [-1] * self.TOKENS_PER_PLAYER for pid in self.player_ids
        }
        self.game_over = False
        self.winner = None

    # ---------- Dice ----------
    def roll_dice(self):
        return self.rng["dice"].randint(1, 6)

    def _roll_off(self):
        """Seat index of the first player: all roll, the highest starts, tied highest re-roll."""
        contenders = self.player_ids
        while len(contenders) > 1:
            rolls = [[pid, self.roll_dice()] for pid in contenders]
            self.roll_off.append(rolls)
            top = max(r for _, r in rolls)
            contenders = [pid for pid, r in rolls if r == top]
        return self.player_ids.index(contenders[0])

    # ---------- Legal Moves ----------
    def get_legal_moves(self, player, dice):
        return rules.legal_moves(self.tokens, player, dice, self.original_rules)

    # ---------- Apply Move ----------
    def apply_move(self, player, token_idx, dice):
      # Store current position before move
      current_position = self.tokens[player].copy()
      old_pos = self.tokens[player][token_idx]
      after, caught = rules.apply_move(self.tokens, player, token_idx, dice, self.original_rules)
      for pid in self.player_ids:
          self.tokens[pid][:] = after[pid]
      new_pos = self.tokens[player][token_idx]
      capture = bool(caught)

      # Store new position after move
      new_position = self.tokens[player].copy()
      
      # Log to game_log.txt (existing behavior)
      self.log_file.write(f"Move applied: Player {player} | Dice {dice} | Token {token_idx} | " f"{old_pos} -> {new_pos} | Capture: {capture}\n")
      self.log_file.flush()
      
      # Log to player-specific file
      self.move_counts[player] += 1
      move_num = self.move_counts[player]
      player_log = self.player_logs[player]
      player_log.write(f"{move_num}. current position ->{current_position} got->{dice} new position ->{new_position}\n")
      player_log.flush()
      
      # Populate move_log for evaluation
      self.move_log.append({
          "player": player,
          "capture": capture,
          "to": new_pos
      })

      # Check win
      if self.check_winner(player):
          self.game_over = True
          self.winner = player

      # Switch turn if needed
      if dice != 6:
          self.current_player_idx = (self.current_player_idx + 1) % self.NUM_PLAYERS
          


    # ---------- Win Condition ----------
    def check_winner(self, player):
        return rules.has_won(self.tokens, player)

    # ---------- One Step ----------
    def step(self, agents):
        if self.game_over:
            return

        dice = self.roll_dice()
        player = self.player_ids[self.current_player_idx]
        legal_moves = self.get_legal_moves(player, dice)

        before = {pid: list(t) for pid, t in self.tokens.items()}
        token = None
        if legal_moves:
            token = agents[player].select_move(
                env=self,
                player=player,
                dice=dice,
                legal_moves=legal_moves
            )
            if token not in legal_moves:
                raise RuntimeError(f"P{player} chose {token!r}, not one of {legal_moves}")
            self.apply_move(player, token, dice)
        else:
            if dice != 6:
                self.current_player_idx = (self.current_player_idx + 1) % self.NUM_PLAYERS
        self.check_roll(before, player, dice, token)

    def check_roll(self, before, player, dice, token):
        """Raise if the board or the turn after a roll breaks a rule (token None: no move)."""
        def broken(what):
            raise RuntimeError(f"invariant broken: {what} (P{player}, dice {dice}, token {token}, "
                               f"before {before}, after {self.tokens})")

        for pid, toks in self.tokens.items():
            if len(toks) != self.TOKENS_PER_PLAYER:
                broken(f"P{pid} has {len(toks)} pieces")
            if not all(rules.on_route(pid, pos, self.original_rules) for pos in toks):
                broken(f"a P{pid} piece is off its route")
            crowded = [p for p in toks if 0 <= p < self.BOARD_SIZE and p not in self.SAFE_SQUARES]
            if len(crowded) != len(set(crowded)):
                broken(f"P{pid} pieces share a non-safe square")
        moved = [(pid, i) for pid, toks in self.tokens.items()
                 for i, pos in enumerate(toks) if pos != before[pid][i]]
        if token is None:
            if moved:
                broken("a roll without a legal move changed the board")
        else:
            to = rules.destination(player, before[player][token], dice, self.original_rules)
            caught = [(pid, i) for pid, i in moved if pid != player]
            on_landing = [(pid, i) for pid, toks in before.items() if pid != player
                          for i, pos in enumerate(toks) if pos == to]
            if [m for m in moved if m[0] == player] != [(player, token)] or self.tokens[player][token] != to:
                broken("the moved piece did not go to its destination alone")
            if caught and to in self.SAFE_SQUARES:
                broken("capture on a safe square")
            if any(self.tokens[pid][i] != rules.BASE for pid, i in caught):
                broken("a captured piece is not in its base")
            if caught != (on_landing if rules.captures(before, player, to) else []):
                broken("captured pieces are not exactly the opponents on the landing square")
        winners = [pid for pid in self.player_ids if rules.has_won(self.tokens, pid)]
        if winners != ([self.winner] if self.game_over else []):
            broken(f"winner {self.winner}, players with all pieces finished {winners}")
        if not self.game_over and (self.player_ids[self.current_player_idx] == player) != (dice == 6):
            broken("only a 6 gives another roll")


    # ---------- Play Full Game ----------
    def play_game(self, agents, max_steps=1000):
        steps = 0
        while not self.game_over and steps < max_steps:
            self.step(agents)
            steps += 1

        # Close all log files
        self.log_file.close()
        if self.llm_log_file:
            self.llm_log_file.close()
        for log in self.player_logs.values():
            log.close()

        return self.winner, steps



# ---------- Quick Test ----------

if __name__ == "__main__":
    from agents import RandomAgent
    from llm_agent import LLMAgent
    from real_llm import real_llm

    env = LudoEnv(seed=200)

    agents = {
        # 0: HeuristicAgent(),
        0: LLMAgent(real_llm),
        1: RandomAgent()
        # 1: HeuristicAgent()
    }
    print(env.play_game(agents))
    # winner, steps = env.play_game(agents)
    # print("Winner:", winner, "Steps:", steps)



# if __name__ == "__main__":
#     env = LudoEnv()
#     agents = {
#         0: RandomAgent(),
#         1: RandomAgent()
#     }

#     winner, steps = env.play_game(agents)
#     print("Winner:", winner, "Steps:", steps)

# for m in env.move_log[:202]:
#     print(m)
