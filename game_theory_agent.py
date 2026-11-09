import math

import ludo_rules as rules
from agents import BaseAgent, HeuristicAgent


def best_move(values, legal_moves, original):
    """
    The move with the highest value. With original rules: the first best move in
    legal-move order, comparing raw floats. Otherwise values are rounded to 1e-9
    and ties go to the lowest piece index.
    """
    if original:
        return max(legal_moves, key=values.get)
    return min(legal_moves, key=lambda m: (-round(values[m], 9), m))


class GameTheoryAgent(BaseAgent):
    """
    2-player expectiminimax agent (chance-aware minimax).

    Notes:
    - Moves follow ludo_rules; env.original_rules selects the released movement.
    - Falls back to HeuristicAgent for non-2-player games.
    """

    def __init__(self, search_depth: int = 2):
        self.search_depth = max(1, int(search_depth))
        self._fallback = HeuristicAgent()
        self._cache = {}
        self._root_player = None
        self._player_ids = ()
        self._original = False

    def select_move(self, env, player, dice, legal_moves):
        if len(env.player_ids) != 2:
            return self._fallback.select_move(env, player, dice, legal_moves)

        self._cache.clear()
        self._root_player = player
        self._player_ids = tuple(env.player_ids)
        self._original = env.original_rules

        state = {pid: tuple(env.tokens[pid]) for pid in self._player_ids}
        values = {}
        for move in legal_moves:
            next_state, winner = self._apply_move(state, player, move, dice)
            if winner is not None:
                values[move] = 1.0 if winner == self._root_player else -1.0
            else:
                next_player = player if dice == 6 else self._other_player(player)
                values[move] = self._chance_value(next_state, next_player, self.search_depth - 1)
        return best_move(values, legal_moves, self._original)

    def _other_player(self, player):
        ids = self._player_ids
        return ids[1] if ids[0] == player else ids[0]

    def _terminal_winner(self, state):
        for pid in self._player_ids:
            if rules.has_won(state, pid):
                return pid
        return None

    def _apply_move(self, state, player, token_idx, dice):
        after, _ = rules.apply_move(state, player, token_idx, dice, self._original)
        next_state = {pid: tuple(after[pid]) for pid in self._player_ids}
        return next_state, self._terminal_winner(next_state)

    def _evaluate_state(self, state):
        me = self._root_player
        opp = self._other_player(me)

        def progress(pid):
            hs, he = rules.home_range(pid)
            st = rules.START_POSITIONS[pid]
            total = 0.0
            base = 0
            finished = 0
            safe_count = 0
            for pos in state[pid]:
                if pos == -1:
                    base += 1
                    continue
                if pos == he:
                    finished += 1
                if 0 <= pos < rules.BOARD_SIZE:
                    rel = (pos - st) % rules.BOARD_SIZE
                    total += rel + 1
                    if pos in rules.SAFE_SQUARES:
                        safe_count += 1
                elif pos >= hs:
                    total += rules.BOARD_SIZE + (pos - hs + 1)
            return total, base, finished, safe_count

        me_prog, me_base, me_fin, me_safe = progress(me)
        op_prog, op_base, op_fin, op_safe = progress(opp)

        # Weighted linear evaluator for cutoff leaves.
        score = 0.0
        score += 0.0025 * (me_prog - op_prog)
        score += 0.20 * (me_fin - op_fin)
        score += -0.05 * (me_base - op_base)
        score += 0.01 * (me_safe - op_safe)
        return max(-0.999, min(0.999, score))

    def _decision_value(self, state, current_player, dice, depth):
        winner = self._terminal_winner(state)
        if winner is not None:
            return 1.0 if winner == self._root_player else -1.0
        if depth <= 0:
            return self._evaluate_state(state)

        cache_key = ("D", tuple((pid, state[pid]) for pid in self._player_ids),
                     current_player, dice, depth)
        if cache_key in self._cache:
            return self._cache[cache_key]

        legal = rules.legal_moves(state, current_player, dice, self._original)
        if not legal:
            next_player = current_player if dice == 6 else self._other_player(current_player)
            val = self._chance_value(state, next_player, depth - 1)
            self._cache[cache_key] = val
            return val

        maximizing = current_player == self._root_player
        best = -math.inf if maximizing else math.inf

        for move in legal:
            next_state, winner = self._apply_move(state, current_player, move, dice)
            if winner is not None:
                val = 1.0 if winner == self._root_player else -1.0
            else:
                next_player = current_player if dice == 6 else self._other_player(current_player)
                val = self._chance_value(next_state, next_player, depth - 1)

            if maximizing:
                if val > best:
                    best = val
            else:
                if val < best:
                    best = val

        self._cache[cache_key] = best
        return best

    def _chance_value(self, state, current_player, depth):
        winner = self._terminal_winner(state)
        if winner is not None:
            return 1.0 if winner == self._root_player else -1.0
        if depth <= 0:
            return self._evaluate_state(state)

        cache_key = ("C", tuple((pid, state[pid]) for pid in self._player_ids),
                     current_player, depth)
        if cache_key in self._cache:
            return self._cache[cache_key]

        total = 0.0
        for dice in (1, 2, 3, 4, 5, 6):
            total += (1.0 / 6.0) * self._decision_value(state, current_player, dice, depth)

        self._cache[cache_key] = total
        return total
