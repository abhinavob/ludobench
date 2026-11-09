import ludo_rules as rules
from agents import BaseAgent
from game_theory_agent import GameTheoryAgent, best_move


class GameTheoryMultiplayerAgent(BaseAgent):
    """
    N-player chance-aware game-theory agent.

    - For 2 players: delegates to existing GameTheoryAgent (keeps current behavior unchanged).
    - For 3/4 players: uses Expecti-MaxN approximation:
      each player chooses moves maximizing their own utility component.
    - Moves follow ludo_rules; env.original_rules selects the released movement.
    """

    def __init__(self, search_depth: int = 2):
        self.search_depth = max(1, int(search_depth))
        self._two_player_agent = GameTheoryAgent(search_depth=self.search_depth)
        self._cache = {}
        self._root_player = None
        self._player_ids = ()
        self._idx = {}
        self._original = False

    def select_move(self, env, player, dice, legal_moves):
        if len(env.player_ids) == 2:
            return self._two_player_agent.select_move(env, player, dice, legal_moves)

        self._cache.clear()
        self._root_player = player
        self._player_ids = tuple(env.player_ids)
        self._idx = {pid: i for i, pid in enumerate(self._player_ids)}
        self._original = env.original_rules

        state = {pid: tuple(env.tokens[pid]) for pid in self._player_ids}
        root_i = self._idx[self._root_player]

        values = {}
        for move in legal_moves:
            next_state, winner = self._apply_move(state, player, move, dice)
            if winner is not None:
                vec = self._terminal_vector(winner)
            else:
                next_player = player if dice == 6 else self._next_player(player)
                vec = self._chance_value(next_state, next_player, self.search_depth - 1)
            values[move] = vec[root_i]
        return best_move(values, legal_moves, self._original)

    def _next_player(self, player):
        i = self._player_ids.index(player)
        return self._player_ids[(i + 1) % len(self._player_ids)]

    def _better(self, a, b):
        """a beats b; outside the original rules, values within 1e-9 tie (first move kept)."""
        return a > b if self._original else round(a, 9) > round(b, 9)

    def _terminal_winner(self, state):
        for pid in self._player_ids:
            if rules.has_won(state, pid):
                return pid
        return None

    def _terminal_vector(self, winner: int):
        n = len(self._player_ids)
        if n <= 1:
            return (1.0,)
        lose_val = -1.0 / (n - 1)
        vals = []
        for pid in self._player_ids:
            vals.append(1.0 if pid == winner else lose_val)
        return tuple(vals)

    def _apply_move(self, state, player, token_idx, dice):
        after, _ = rules.apply_move(state, player, token_idx, dice, self._original)
        next_state = {pid: tuple(after[pid]) for pid in self._player_ids}
        return next_state, self._terminal_winner(next_state)

    def _evaluate_state(self, state):
        raw = []

        for pid in self._player_ids:
            hs, he = rules.home_range(pid)
            st = rules.START_POSITIONS[pid]

            prog = 0.0
            base = 0.0
            fin = 0.0
            safe_count = 0.0

            for pos in state[pid]:
                if pos == -1:
                    base += 1
                    continue
                if pos == he:
                    fin += 1
                if 0 <= pos < rules.BOARD_SIZE:
                    rel = (pos - st) % rules.BOARD_SIZE
                    prog += rel + 1
                    if pos in rules.SAFE_SQUARES:
                        safe_count += 1
                elif pos >= hs:
                    prog += rules.BOARD_SIZE + (pos - hs + 1)

            # Same weighted terms as 2-player evaluator, per player.
            score = 0.0
            score += 0.0025 * prog
            score += 0.20 * fin
            score += -0.05 * base
            score += 0.01 * safe_count
            raw.append(score)

        mean_raw = sum(raw) / max(1, len(raw))
        vec = [max(-0.999, min(0.999, s - mean_raw)) for s in raw]
        return tuple(vec)

    def _decision_value(self, state, current_player, dice, depth):
        winner = self._terminal_winner(state)
        if winner is not None:
            return self._terminal_vector(winner)
        if depth <= 0:
            return self._evaluate_state(state)

        cache_key = ("D", tuple((pid, state[pid]) for pid in self._player_ids),
                     current_player, dice, depth)
        if cache_key in self._cache:
            return self._cache[cache_key]

        legal = rules.legal_moves(state, current_player, dice, self._original)
        if not legal:
            next_player = current_player if dice == 6 else self._next_player(current_player)
            val = self._chance_value(state, next_player, depth - 1)
            self._cache[cache_key] = val
            return val

        actor_i = self._idx[current_player]
        best = None

        for move in legal:
            next_state, winner = self._apply_move(state, current_player, move, dice)
            if winner is not None:
                val = self._terminal_vector(winner)
            else:
                next_player = current_player if dice == 6 else self._next_player(current_player)
                val = self._chance_value(next_state, next_player, depth - 1)

            if best is None or self._better(val[actor_i], best[actor_i]):
                best = val

        self._cache[cache_key] = best
        return best

    def _chance_value(self, state, current_player, depth):
        winner = self._terminal_winner(state)
        if winner is not None:
            return self._terminal_vector(winner)
        if depth <= 0:
            return self._evaluate_state(state)

        cache_key = ("C", tuple((pid, state[pid]) for pid in self._player_ids),
                     current_player, depth)
        if cache_key in self._cache:
            return self._cache[cache_key]

        n = len(self._player_ids)
        total = [0.0] * n
        for dice in (1, 2, 3, 4, 5, 6):
            val = self._decision_value(state, current_player, dice, depth)
            for i in range(n):
                total[i] += val[i] / 6.0

        out = tuple(total)
        self._cache[cache_key] = out
        return out
