import random

class LudoEnv:
    def __init__(self, seed=42, num_players=2, player_ids=None):
        random.seed(seed)
        # Move log (list of dicts)
        self.log_file = open("game_log.txt", "w")
        self.llm_log_file = open("llm_log.txt", "w")
        if player_ids is None:
            player_ids = list(range(num_players))
        self.player_ids = list(player_ids)
        self.NUM_PLAYERS = len(self.player_ids)
        self.player_logs = {
            pid: open(f"player{pid + 1}.txt", "w") for pid in self.player_ids
        }
        self.move_counts = {pid: 0 for pid in self.player_ids}  # Track move numbers for each player
        self.move_log = []
        # LLM fallback tracking
        self.llm_calls = 0
        self.llm_fallbacks = 0
        # Constants
        self.TOKENS_PER_PLAYER = 4
        self.BOARD_SIZE = 52
        self.HOME_START = 52
        self.HOME_LEN = 6
        base_starts = [0, 13, 26, 39]
        self.START_POSITIONS = {pid: base_starts[pid] for pid in self.player_ids}

        # Real Ludo safe squares (standard)
        self.SAFE_SQUARES = {0, 8, 13, 21, 26, 34, 39, 47}

        # Game state
        self.current_player_idx = 0
        self.tokens = {
            pid: [-1] * self.TOKENS_PER_PLAYER for pid in self.player_ids
        }
        self.game_over = False
        self.winner = None

    # ---------- Dice ----------
    def roll_dice(self):
        return random.randint(1, 6)

    # ---------- Legal Moves ----------
    def get_legal_moves(self, player, dice):
      legal = []
      start = self.START_POSITIONS[player]
      home_start = self.HOME_START + player * self.HOME_LEN
      home_end = home_start + self.HOME_LEN - 1

      for i, pos in enumerate(self.tokens[player]):

          # Token in base: can move out only on dice = 6
          if pos == -1:
              if dice == 6:
                  legal.append(i)

          # Token on main board
          elif 0 <= pos < self.BOARD_SIZE:
              rel_pos = (pos - start) % self.BOARD_SIZE
              new_rel = rel_pos + dice

              if new_rel < self.BOARD_SIZE:
                  new_pos = (start + new_rel) % self.BOARD_SIZE
                  # Same-player stacking is allowed on safe squares only.
                  if new_pos not in self.tokens[player] or new_pos in self.SAFE_SQUARES:
                      legal.append(i)
              else:
                  home_step = new_rel - self.BOARD_SIZE
                  if 1 <= home_step <= self.HOME_LEN:
                      new_pos = home_start + home_step - 1
                      # Home-path stacking is allowed.
                      legal.append(i)

          # Token in home path
          elif home_start <= pos <= home_end:
              new_pos = pos + dice
              if new_pos == home_end:
                  legal.append(i)

      return legal


    # ---------- Apply Move ----------
    def apply_move(self, player, token_idx, dice):
      # Store current position before move
      current_position = self.tokens[player].copy()
      start = self.START_POSITIONS[player]
      home_start = self.HOME_START + player * self.HOME_LEN
      home_end = home_start + self.HOME_LEN - 1
      
      old_pos = self.tokens[player][token_idx]
      capture = False

      # Move from base
      if old_pos == -1:
          new_pos = start
      elif 0 <= old_pos < self.BOARD_SIZE:
          rel_pos = (old_pos - start) % self.BOARD_SIZE
          new_rel = rel_pos + dice
          if new_rel < self.BOARD_SIZE:
              new_pos = (start + new_rel) % self.BOARD_SIZE
          else:
              home_step = new_rel - self.BOARD_SIZE
              new_pos = home_start + home_step - 1
      else:
          new_pos = old_pos + dice

      # Capture logic
      if 0 <= new_pos < self.BOARD_SIZE and new_pos not in self.SAFE_SQUARES:
          for opponent in self.player_ids:
              if opponent == player:
                  continue
              for i, opp_pos in enumerate(self.tokens[opponent]):
                  if opp_pos == new_pos:
                      self.tokens[opponent][i] = -1
                      capture = True

      # Apply move
      self.tokens[player][token_idx] = new_pos
      
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
        home_start = self.HOME_START + player * self.HOME_LEN
        home_end = home_start + self.HOME_LEN - 1
        return all(pos == home_end for pos in self.tokens[player])

    # ---------- One Step ----------
    def step(self, agents):
        if self.game_over:
            return

        dice = self.roll_dice()
        player = self.player_ids[self.current_player_idx]
        legal_moves = self.get_legal_moves(player, dice)

        if legal_moves:
            token = agents[player].select_move(
                env=self,
                player=player,
                dice=dice,
                legal_moves=legal_moves
            )
            self.apply_move(player, token, dice)
        else:
            if dice != 6:
                self.current_player_idx = (self.current_player_idx + 1) % self.NUM_PLAYERS


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
