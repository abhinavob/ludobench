class BaseAgent:
    def select_move(self, env, player, dice, legal_moves):
        """
        env         : LudoEnv object
        player      : current player id
        dice        : dice value
        legal_moves : list of token indices
        """
        raise NotImplementedError
import ludo_rules as rules


class RandomAgent(BaseAgent):
    def select_move(self, env, player, dice, legal_moves):
        return env.rng["random_player"].choice(legal_moves)
class HeuristicAgent(BaseAgent):
    def scores(self, env, player, dice, legal_moves):
        """
        Score per legal token: 50 for leaving base; otherwise the pips from the
        start square to the landing cell, +100 per opponent captured there and
        +20 for a safe square.
        """
        scores = {}
        for token in legal_moves:
            if env.tokens[player][token] == -1:
                scores[token] = 50
                continue
            out = rules.move_outcome(env.tokens, player, token, dice, env.original_rules)
            score = rules.route_index(player, out["to"], env.original_rules)
            score += 100 * len({opp for opp, _ in out["captures"]})
            if out["safe"]:
                score += 20
            scores[token] = score
        return scores

    def select_move(self, env, player, dice, legal_moves):
        scores = self.scores(env, player, dice, legal_moves)
        return max(legal_moves, key=scores.get)        # first best in legal-move order
