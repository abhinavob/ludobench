class BaseAgent:
    def select_move(self, env, player, dice, legal_moves):
        """
        env         : LudoEnv object
        player      : current player id
        dice        : dice value
        legal_moves : list of token indices
        """
        raise NotImplementedError
import random

class RandomAgent(BaseAgent):
    def select_move(self, env, player, dice, legal_moves):
        return random.choice(legal_moves)
class HeuristicAgent(BaseAgent):
    def select_move(self, env, player, dice, legal_moves):
        best_token = None
        best_score = -1e9

        for token in legal_moves:
            pos = env.tokens[player][token]

            # Base → board is good
            if pos == -1:
                score = 50

            else:
                if 0 <= pos < env.BOARD_SIZE:
                    start = env.START_POSITIONS[player]
                    rel_pos = (pos - start) % env.BOARD_SIZE
                    new_rel = rel_pos + dice
                    if new_rel < env.BOARD_SIZE:
                        new_pos = (start + new_rel) % env.BOARD_SIZE
                        score = new_rel
                    else:
                        home_step = new_rel - env.BOARD_SIZE
                        new_pos = (env.HOME_START + player * env.HOME_LEN) + home_step - 1
                        score = env.BOARD_SIZE + home_step
                else:
                    new_pos = pos + dice
                    home_start = env.HOME_START + player * env.HOME_LEN
                    home_step = (new_pos - home_start) + 1
                    score = env.BOARD_SIZE + home_step

                # Capture is very good
                if (0 <= new_pos < env.BOARD_SIZE and
                    new_pos not in env.SAFE_SQUARES):
                    for opponent in env.player_ids:
                        if opponent == player:
                            continue
                        if new_pos in env.tokens[opponent]:
                            score += 100

                # Landing on safe square is good
                if new_pos in env.SAFE_SQUARES:
                    score += 20

            if score > best_score:
                best_score = score
                best_token = token

        return best_token
