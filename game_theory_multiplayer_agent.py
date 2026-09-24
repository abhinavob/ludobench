import math
from typing import Dict, List, Tuple

from agents import BaseAgent
from game_theory_agent import GameTheoryAgent


class GameTheoryMultiplayerAgent(BaseAgent):
    """
    N-player chance-aware game-theory agent.

    - For 2 players: delegates to existing GameTheoryAgent (keeps current behavior unchanged).
    - For 3/4 players: uses Expecti-MaxN approximation:
      each player chooses moves maximizing their own utility component.
    """

    def __init__(self, search_depth: int = 2):
        self.search_depth = max(1, int(search_depth))
        self._two_player_agent = GameTheoryAgent(search_depth=self.search_depth)
        self._cache = {}
        self._root_player = None
        self._player_ids = ()
        self._idx = {}

    def select_move(self, env, player, dice, legal_moves):
        if len(env.player_ids) == 2:
            return self._two_player_agent.select_move(env, player, dice, legal_moves)

        self._cache.clear()
        self._root_player = player
        self._player_ids = tuple(env.player_ids)
        self._idx = {pid: i for i, pid in enumerate(self._player_ids)}

        starts = dict(env.START_POSITIONS)
        state = {pid: tuple(env.tokens[pid]) for pid in self._player_ids}
        root_i = self._idx[self._root_player]

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
                player_ids=self._player_ids,
            )
            if winner is not None:
                vec = self._terminal_vector(winner)
            else:
                next_player = player if dice == 6 else self._next_player(self._player_ids, player)
                vec = self._chance_value(
                    state=next_state,
                    current_player=next_player,
                    depth=self.search_depth - 1,
                    starts=starts,
                    safe_squares=env.SAFE_SQUARES,
                    board_size=env.BOARD_SIZE,
                    home_start_global=env.HOME_START,
                    home_len=env.HOME_LEN,
                    player_ids=self._player_ids,
                )
            if vec[root_i] > best_val:
                best_val = vec[root_i]
                best_move = move

        return best_move

    @staticmethod
    def _next_player(player_ids: Tuple[int, ...], player: int) -> int:
        i = player_ids.index(player)
        return player_ids[(i + 1) % len(player_ids)]

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

    def _terminal_vector(self, winner: int):
        n = len(self._player_ids)
        if n <= 1:
            return (1.0,)
        lose_val = -1.0 / (n - 1)
        vals = []
        for pid in self._player_ids:
            vals.append(1.0 if pid == winner else lose_val)
        return tuple(vals)

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
                    # Stacking allowed only on safe squares.
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

        # Capture only on non-safe main-board squares.
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
        raw = []

        for pid in player_ids:
            hs = self._home_start(home_start_global, home_len, pid)
            he = self._home_end(home_start_global, home_len, pid)
            st = starts[pid]

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
                if 0 <= pos < board_size:
                    rel = (pos - st) % board_size
                    prog += rel + 1
                    if pos in safe_squares:
                        safe_count += 1
                elif pos >= hs:
                    prog += board_size + (pos - hs + 1)

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
            return self._terminal_vector(winner)
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
            next_player = current_player if dice == 6 else self._next_player(player_ids, current_player)
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

        actor_i = self._idx[current_player]
        best = None

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
                val = self._terminal_vector(winner)
            else:
                next_player = current_player if dice == 6 else self._next_player(player_ids, current_player)
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

            if best is None or val[actor_i] > best[actor_i]:
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
            return self._terminal_vector(winner)
        if depth <= 0:
            return self._evaluate_state(
                state, player_ids, starts, safe_squares, board_size, home_start_global, home_len
            )

        cache_key = ("C", tuple((pid, state[pid]) for pid in player_ids), current_player, depth)
        if cache_key in self._cache:
            return self._cache[cache_key]

        n = len(player_ids)
        total = [0.0] * n
        for dice in (1, 2, 3, 4, 5, 6):
            val = self._decision_value(
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
            for i in range(n):
                total[i] += val[i] / 6.0

        out = tuple(total)
        self._cache[cache_key] = out
        return out
