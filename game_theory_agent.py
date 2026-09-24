import math
from typing import Dict, List, Tuple

from agents import BaseAgent, HeuristicAgent


class GameTheoryAgent(BaseAgent):
    """
    2-player expectiminimax agent (chance-aware minimax).

    Notes:
    - Optimized for the current ludo_env.py rule set.
    - Falls back to HeuristicAgent for non-2-player games.
    """

    def __init__(self, search_depth: int = 2):
        self.search_depth = max(1, int(search_depth))
        self._fallback = HeuristicAgent()
        self._cache = {}
        self._root_player = None

    def select_move(self, env, player, dice, legal_moves):
        if len(env.player_ids) != 2:
            return self._fallback.select_move(env, player, dice, legal_moves)

        self._cache.clear()
        self._root_player = player

        starts = dict(env.START_POSITIONS)
        state = {pid: tuple(env.tokens[pid]) for pid in env.player_ids}

        best_move = legal_moves[0]
        best_val = -math.inf
        for move in legal_moves:
            next_state, winner = self._apply_move(
                state=state,
                player=player,
                token_idx=move,
                dice=dice,
                starts=starts,
                safe_squares=env.SAFE_SQUARES,
                board_size=env.BOARD_SIZE,
                home_start_global=env.HOME_START,
                home_len=env.HOME_LEN,
                player_ids=env.player_ids,
            )
            if winner is not None:
                val = 1.0 if winner == self._root_player else -1.0
            else:
                next_player = player if dice == 6 else self._other_player(env.player_ids, player)
                val = self._chance_value(
                    state=next_state,
                    current_player=next_player,
                    depth=self.search_depth - 1,
                    starts=starts,
                    safe_squares=env.SAFE_SQUARES,
                    board_size=env.BOARD_SIZE,
                    home_start_global=env.HOME_START,
                    home_len=env.HOME_LEN,
                    player_ids=env.player_ids,
                )

            if val > best_val:
                best_val = val
                best_move = move

        return best_move

    @staticmethod
    def _other_player(player_ids: List[int], player: int) -> int:
        return player_ids[1] if player_ids[0] == player else player_ids[0]

    @staticmethod
    def _home_start(home_start_global: int, home_len: int, player: int) -> int:
        return home_start_global + player * home_len

    @staticmethod
    def _home_end(home_start_global: int, home_len: int, player: int) -> int:
        return home_start_global + player * home_len + home_len - 1

    def _terminal_winner(self, state, player_ids, home_start_global, home_len):
        for pid in player_ids:
            he = self._home_end(home_start_global, home_len, pid)
            if all(pos == he for pos in state[pid]):
                return pid
        return None

    def _legal_moves(
        self,
        state,
        player,
        dice,
        starts,
        safe_squares,
        board_size,
        home_start_global,
        home_len,
    ):
        legal = []
        start = starts[player]
        home_start = self._home_start(home_start_global, home_len, player)
        home_end = self._home_end(home_start_global, home_len, player)
        own_tokens = state[player]

        for i, pos in enumerate(own_tokens):
            if pos == -1:
                if dice == 6:
                    legal.append(i)
            elif 0 <= pos < board_size:
                rel_pos = (pos - start) % board_size
                new_rel = rel_pos + dice
                if new_rel < board_size:
                    new_pos = (start + new_rel) % board_size
                    # Stacking allowed on safe squares; blocked on non-safe squares.
                    if new_pos not in own_tokens or new_pos in safe_squares:
                        legal.append(i)
                else:
                    home_step = new_rel - board_size
                    if 1 <= home_step <= home_len:
                        # Home-path stacking allowed.
                        legal.append(i)
            elif home_start <= pos <= home_end:
                new_pos = pos + dice
                if new_pos == home_end:
                    legal.append(i)

        return legal

    def _apply_move(
        self,
        state,
        player,
        token_idx,
        dice,
        starts,
        safe_squares,
        board_size,
        home_start_global,
        home_len,
        player_ids,
    ):
        tokens = {pid: list(state[pid]) for pid in player_ids}

        start = starts[player]
        home_start = self._home_start(home_start_global, home_len, player)

        old_pos = tokens[player][token_idx]
        if old_pos == -1:
            new_pos = start
        elif 0 <= old_pos < board_size:
            rel_pos = (old_pos - start) % board_size
            new_rel = rel_pos + dice
            if new_rel < board_size:
                new_pos = (start + new_rel) % board_size
            else:
                home_step = new_rel - board_size
                new_pos = home_start + home_step - 1
        else:
            new_pos = old_pos + dice

        # Capture: only on non-safe main-board squares.
        if 0 <= new_pos < board_size and new_pos not in safe_squares:
            for opp in player_ids:
                if opp == player:
                    continue
                for j, opp_pos in enumerate(tokens[opp]):
                    if opp_pos == new_pos:
                        tokens[opp][j] = -1

        tokens[player][token_idx] = new_pos
        next_state = {pid: tuple(tokens[pid]) for pid in player_ids}
        winner = self._terminal_winner(next_state, player_ids, home_start_global, home_len)
        return next_state, winner

    def _evaluate_state(
        self,
        state,
        player_ids,
        starts,
        safe_squares,
        board_size,
        home_start_global,
        home_len,
    ):
        me = self._root_player
        opp = self._other_player(player_ids, me)

        def progress(pid):
            hs = self._home_start(home_start_global, home_len, pid)
            he = self._home_end(home_start_global, home_len, pid)
            st = starts[pid]
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
                if 0 <= pos < board_size:
                    rel = (pos - st) % board_size
                    total += rel + 1
                    if pos in safe_squares:
                        safe_count += 1
                elif pos >= hs:
                    total += board_size + (pos - hs + 1)
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

    def _decision_value(
        self,
        state,
        current_player,
        dice,
        depth,
        starts,
        safe_squares,
        board_size,
        home_start_global,
        home_len,
        player_ids,
    ):
        winner = self._terminal_winner(state, player_ids, home_start_global, home_len)
        if winner is not None:
            return 1.0 if winner == self._root_player else -1.0
        if depth <= 0:
            return self._evaluate_state(
                state, player_ids, starts, safe_squares, board_size, home_start_global, home_len
            )

        cache_key = (
            "D",
            tuple((pid, state[pid]) for pid in player_ids),
            current_player,
            dice,
            depth,
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        legal = self._legal_moves(
            state, current_player, dice, starts, safe_squares, board_size, home_start_global, home_len
        )
        if not legal:
            next_player = current_player if dice == 6 else self._other_player(player_ids, current_player)
            val = self._chance_value(
                state,
                next_player,
                depth - 1,
                starts,
                safe_squares,
                board_size,
                home_start_global,
                home_len,
                player_ids,
            )
            self._cache[cache_key] = val
            return val

        maximizing = current_player == self._root_player
        best = -math.inf if maximizing else math.inf

        for move in legal:
            next_state, winner = self._apply_move(
                state,
                current_player,
                move,
                dice,
                starts,
                safe_squares,
                board_size,
                home_start_global,
                home_len,
                player_ids,
            )
            if winner is not None:
                val = 1.0 if winner == self._root_player else -1.0
            else:
                next_player = current_player if dice == 6 else self._other_player(player_ids, current_player)
                val = self._chance_value(
                    next_state,
                    next_player,
                    depth - 1,
                    starts,
                    safe_squares,
                    board_size,
                    home_start_global,
                    home_len,
                    player_ids,
                )

            if maximizing:
                if val > best:
                    best = val
            else:
                if val < best:
                    best = val

        self._cache[cache_key] = best
        return best

    def _chance_value(
        self,
        state,
        current_player,
        depth,
        starts,
        safe_squares,
        board_size,
        home_start_global,
        home_len,
        player_ids,
    ):
        winner = self._terminal_winner(state, player_ids, home_start_global, home_len)
        if winner is not None:
            return 1.0 if winner == self._root_player else -1.0
        if depth <= 0:
            return self._evaluate_state(
                state, player_ids, starts, safe_squares, board_size, home_start_global, home_len
            )

        cache_key = ("C", tuple((pid, state[pid]) for pid in player_ids), current_player, depth)
        if cache_key in self._cache:
            return self._cache[cache_key]

        total = 0.0
        for dice in (1, 2, 3, 4, 5, 6):
            total += (1.0 / 6.0) * self._decision_value(
                state,
                current_player,
                dice,
                depth,
                starts,
                safe_squares,
                board_size,
                home_start_global,
                home_len,
                player_ids,
            )

        self._cache[cache_key] = total
        return total
