"""
Per-decision logging for run_games.py.

A run writes three files to its own folder:
    decisions.jsonl   one line per ply (dice roll), flushed as it is written
    games.jsonl       one line per game
    run.json          configuration, versions and progress; rewritten per game

A ply is one dice roll. A roll with no legal move is logged too (skipped=true),
so ply numbers have no gaps and a game can be replayed from the log alone:
tokens_before + dice + legal_moves + player_id fix the position exactly.
Dice are logged per ply because with original_rules the engine, RandomAgent
and the LLM fallback share one global generator, so the seed alone does not fix
the dice; otherwise each has its own stream seeded from the game seed.

Nothing here draws random numbers or changes the game: it only observes
env.step, env.roll_dice and env.apply_move.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import ludo_rules as rules
from agents import HeuristicAgent

SCHEMA_VERSION = 1
CONTEXT_MODE = "stateless"

AVAILABILITY_KINDS = ("capture", "safe", "home_finish", "bring_out")

# Written to run.json so a results folder explains its own derived fields.
DEFINITIONS = {
    "ply": "0-based index of the dice roll within the game; skipped turns included",
    "turn_number": "0-based; increases when the player to move changes, so extra "
                   "rolls after a 6 share a turn_number",
    "is_extra_turn": "this roll exists because the same player rolled a 6 on the "
                     "previous ply",
    "seat_position": "index of the player in turn order",
    "positions": "-1 base, 0-51 main track, >=52 home path (player-specific); "
                 "see engine.home_ranges",
    "progress": "per token: 0 in base, otherwise 1 + pips from its start square "
                "along its route (finished: 58; with original_rules 59, and 53 is "
                "unreachable); summed over a player's tokens",
    "piece_counts": "home_path excludes finished tokens",
    "action_class": "one label per move, first match of: bring_out, home_finish, "
                    "home_entry, capture, safe, advance",
    "landed_on_safe": "destination is a safe square, bring-outs included (the "
                      "start square is safe); same as the paper's classify_move",
    "availability": "per kind, the legal token indices whose move achieves it. "
                    "'safe' uses landed_on_safe, so it includes bring-outs; "
                    "subtract availability.bring_out.tokens for board moves only",
    "left_piece_exposed": "after the move, the moved token is on a non-safe track "
                          "square that at least one opponent token on the track "
                          "could land on with a single roll of 1-6",
    "pieces_lost_after_exposure": "exposed tokens captured before they moved again",
    "fallback_cause": "timeout: no reply in time; empty: reply had no content; "
                      "unparseable: no token index could be read; illegal: an "
                      "index was read but is not a legal move",
    "parsed_index": "the index the reply parser read, legal or not; null if none",
    "reason_chars": "length in characters of the parsed reason; null if no reply "
                    "was parsed",
    "reason_first": "in a valid JSON reply, the reason came before the piece in "
                    "the raw text (not every model keeps the schema's order); "
                    "null for any other reply",
    "completion_tokens": "output tokens the server reported, thinking included",
    "thinking_tokens": "tokens the model streamed as thinking (Ollama native, one "
                       "chunk per token), or the reasoning tokens an API reported; "
                       "null if not counted",
    "provider": "who served the model: ollama_native, openrouter or nvidia",
    "served_by": "the upstream provider OpenRouter routed the request to",
    "cost_usd": "US dollars the call cost: OpenRouter's figure, 0 on NVIDIA and "
                "Ollama; run.json's cost_usd is the run's total",
    "retries": "rate-limit and server errors retried before the reply; they do "
               "not make the reply invalid",
    "hit_ceiling": "the reply stopped at the output-token ceiling (finish_reason "
                   "length); a reply cut off there is invalid",
    "latency_ms": "wall-clock time of the model call, the response time",
    "raw_reply": "the model's reply text as received, before any parsing",
    "prompt_hash": "first 16 hex chars of sha256 of the user prompt",
    "system_prompt_hash": "first 16 hex chars of sha256 of the system prompt",
    "num_ctx": "context length the Ollama server reports for the loaded model "
               "(/api/ps); null if not an Ollama model or not reported",
    "heuristic_score_per_legal_move": "agents.HeuristicAgent's score for each "
                                      "legal token; heuristic_move is its choice",
    "lead_changes": "plies after which a different player has strictly the most "
                    "total progress than the leader before; the first leader "
                    "does not count",
    "finishing_order": "winner first if any; the rest by tokens finished, then "
                       "total progress, then seat. The game stops at the first "
                       "winner, so later places are a ranking, not finishes",
    "turns_to_first_finish": "turn_number of the ply on which any token first "
                             "reached HOME_END; null if none did",
    "taken_vs_available": "per player: *_available counts decisions where some "
                          "legal move achieved it; *_taken counts those where "
                          "the chosen move did",
}


def short_hash(text):
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


# ---------- board analysis (all read-only) ----------

def snapshot(env):
    return {str(pid): list(env.tokens[pid]) for pid in env.player_ids}


def token_progress(env, pid, pos):
    if pos == -1:
        return 0
    return 1 + rules.route_index(pid, pos, env.original_rules)


def progress(env):
    return {str(pid): sum(token_progress(env, pid, p) for p in env.tokens[pid])
            for pid in env.player_ids}


def piece_counts(env):
    counts = {}
    for pid in env.player_ids:
        home_end = rules.home_range(pid)[1]
        toks = env.tokens[pid]
        counts[str(pid)] = {
            "base": sum(p == -1 for p in toks),
            "board": sum(0 <= p < env.BOARD_SIZE for p in toks),
            "home_path": sum(env.BOARD_SIZE <= p < home_end for p in toks),
            "finished": sum(p == home_end for p in toks),
        }
    return counts


def move_effects(env, player, dice, token_idx):
    """What moving this token would do, read from the position before the move."""
    e = rules.move_outcome(env.tokens, player, token_idx, dice, env.original_rules)
    captures = [{"player": opp, "token": i} for opp, i in e["captures"]]
    if e["bring_out"]:
        action = "bring_out"
    elif e["home_finish"]:
        action = "home_finish"
    elif e["in_home"]:
        action = "home_entry"
    elif captures:
        action = "capture"
    elif e["safe"]:
        action = "safe"
    else:
        action = "advance"
    return {"from": e["from"], "to": e["to"], "action_class": action, "captures": captures,
            "safe": e["safe"], "in_home": e["in_home"], "home_finish": e["home_finish"],
            "bring_out": e["bring_out"], "capture": bool(captures)}


def is_exposed(env, player, pos):
    """Could an opponent token capture a token of `player` on `pos` with one roll?"""
    if not (0 <= pos < env.BOARD_SIZE) or pos in env.SAFE_SQUARES:
        return False
    return any(rules.destination(opp, q, d, env.original_rules) == pos
               for opp in env.player_ids if opp != player
               for q in env.tokens[opp] for d in range(1, 7))


# ---------- environment facts for run.json ----------

def git_state():
    def git(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10,
                                  cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
        except Exception:
            return None
    commit = git("rev-parse", "HEAD") or None
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def package_versions(names=("openai", "httpx")):
    from importlib import metadata
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def ollama_num_ctx(base_url, model, timeout=3):
    """Context length Ollama reports for a loaded model, or None."""
    import urllib.request
    root = base_url[:-3] if base_url.endswith("/v1") else base_url
    try:
        with urllib.request.urlopen(root + "/api/ps", timeout=timeout) as resp:
            running = json.loads(resp.read().decode("utf-8")).get("models", [])
    except Exception:
        return None
    for m in running:
        if model in (m.get("name"), m.get("model")):
            return m.get("context_length")
    return None


# ---------- writers ----------

class RunLog:
    """Owns a run's folder and its three files; `resumed` continues a stopped run."""

    def __init__(self, out_dir, run_id, players, settings, config, ollama_versions,
                 env, resumed=None):
        os.makedirs(out_dir, exist_ok=True)
        self.dir = out_dir
        self.run_id = run_id
        self.players = players                    # pid -> {model_spec, host, ...}
        self.config = config
        self._num_ctx = {}
        mode = "a" if resumed else "w"
        self._decisions = open(os.path.join(out_dir, "decisions.jsonl"), mode,
                               encoding="utf-8")
        self._games = open(os.path.join(out_dir, "games.jsonl"), mode, encoding="utf-8")
        if resumed:
            self.meta = resumed
            self.meta.setdefault("resumed_at", []).append(time.strftime("%Y-%m-%dT%H:%M:%S"))
            self.write_meta()
            return
        self.meta = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "finished_at": None,
            "command_line": sys.argv,
            "git": git_state(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": package_versions(),
            "ollama_versions": ollama_versions,
            "players": {str(pid): p for pid, p in players.items()},
            "settings": settings,
            "config_on_every_decision": config,
            "engine": {
                "original_rules": env.original_rules,
                "player_ids": list(env.player_ids),
                "tokens_per_player": env.TOKENS_PER_PLAYER,
                "board_size": env.BOARD_SIZE,
                "safe_squares": sorted(env.SAFE_SQUARES),
                "start_positions": {str(p): env.START_POSITIONS[p]
                                    for p in env.player_ids},
                "home_ranges": {str(p): list(rules.home_range(p))
                                for p in env.player_ids},
            },
            "games": {"attempted": 0, "completed": 0, "errored": 0},
            "cost_usd": 0.0,
            "definitions": DEFINITIONS,
        }
        self.write_meta()

    def write_meta(self):
        with open(os.path.join(self.dir, "run.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2)

    def num_ctx(self, pid, model_replied):
        """
        Asked of the server once per player, after that player's first reply,
        because Ollama only reports a context length for a loaded model.
        """
        if pid not in self._num_ctx and model_replied:
            p = self.players[pid]
            self._num_ctx[pid] = (ollama_num_ctx(p["host"], p["model"])
                                  if p.get("provider") in ("ollama", "ollama_native")
                                  else None)
        return self._num_ctx.get(pid)

    def write_decision(self, rec):
        self._decisions.write(json.dumps(rec) + "\n")
        self._decisions.flush()

    def game_started(self):
        self.meta["games"]["attempted"] += 1
        self.write_meta()

    def write_game(self, rec):
        self._games.write(json.dumps(rec) + "\n")
        self._games.flush()
        key = "errored" if rec["termination_reason"] == "error" else "completed"
        self.meta["games"][key] += 1
        self.meta["cost_usd"] += sum((p["llm"] or {}).get("cost_usd", 0)
                                     for p in rec["per_player"].values())
        self.write_meta()

    def close(self):
        self.meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.write_meta()
        self._decisions.close()
        self._games.close()


def _new_player_stats():
    return {
        "decisions": 0, "skipped_turns": 0,
        "captures_made": 0, "times_captured": 0,
        "captures_taken": 0, "captures_available": 0,
        "safe_taken": 0, "safe_available": 0,
        "bring_outs_taken": 0, "bring_outs_available": 0,
        "home_finishes_taken": 0, "home_finishes_available": 0,
        "exposures": 0, "pieces_lost_after_exposure": 0,
        "dice_counts": {str(d): 0 for d in range(1, 7)},
    }


_STAT_KEYS = {"capture": "captures", "safe": "safe", "home_finish": "home_finishes",
              "bring_out": "bring_outs"}


class GameLog:
    """Observes one game and writes a decision record per ply."""

    def __init__(self, run, env, agents, game_index, seed):
        self.run = run
        self.env = env
        self.agents = agents
        self.seed = seed
        self.game_index = game_index
        self.game_id = f"g{game_index:03d}-s{seed}"
        self.ply = 0
        self.turn_number = -1
        self.prev_player = None
        self.prev_dice = None
        self.stats = {pid: _new_player_stats() for pid in env.player_ids}
        self.exposed = {}                         # (pid, token) -> square
        self.leader = None
        self.lead_changes = 0
        self.first_finish = None                  # (ply, turn_number)
        self.moves_applied = 0
        self._pre = None
        self._applied = None
        self._heuristic = HeuristicAgent()
        self.started = time.time()

        roll, apply_move, step = env.roll_dice, env.apply_move, env.step

        def roll_dice():
            dice = roll()
            self._analyse(dice)                   # position is still pre-move here
            return dice

        def wrapped_apply(player, token_idx, dice):
            self._applied = token_idx
            return apply_move(player, token_idx, dice)

        def wrapped_step(agents_):
            if env.game_over:
                return step(agents_)
            self._pre = self._applied = None
            before = snapshot(env)
            result = step(agents_)
            self._record(before)
            return result

        env.roll_dice, env.apply_move, env.step = roll_dice, wrapped_apply, wrapped_step

    def _analyse(self, dice):
        env = self.env
        player = env.player_ids[env.current_player_idx]
        legal = env.get_legal_moves(player, dice)
        effects = {t: move_effects(env, player, dice, t) for t in legal}
        self._pre = {
            "player": player, "dice": dice, "legal": legal, "effects": effects,
            "piece_counts": piece_counts(env), "progress": progress(env),
            "availability": {
                kind: {"any": any(e[kind] for e in effects.values()),
                       "tokens": [t for t, e in effects.items() if e[kind]]}
                for kind in AVAILABILITY_KINDS},
            "heuristic_move": (self._heuristic.select_move(env, player, dice, legal)
                               if legal else None),
            "heuristic_scores": {str(t): v for t, v in
                                 self._heuristic.scores(env, player, dice, legal).items()},
        }

    def _model_fields(self, player, skipped):
        agent = self.agents[player]
        agent = getattr(agent, "inner", agent)    # unwrap WatchAgent
        model = None if skipped else getattr(agent, "last_decision", None)
        if model is not None:
            agent.last_decision = None            # never reuse for a later ply
        return model or {
            "raw_reply": None, "parsed_index": None, "parsed_reason": None,
            "reasoning_text": None, "finish_reason": None, "was_fallback": None,
            "fallback_cause": None, "prompt_tokens": None, "completion_tokens": None,
            "reason_chars": None, "reason_first": None, "thinking_tokens": None, "hit_ceiling": None,
            "provider": None, "served_by": None, "cost_usd": None, "retries": None,
            "latency_ms": None, "prompt_hash": None}

    def _record(self, before):
        env, pre = self.env, self._pre
        player, dice, legal = pre["player"], pre["dice"], pre["legal"]
        chosen = self._applied
        skipped = chosen is None
        st = self.stats[player]

        if player != self.prev_player:
            self.turn_number += 1
        is_extra = player == self.prev_player and self.prev_dice == 6

        move = {"chosen_index": None, "from_position": None, "to_position": None,
                "action_class": None, "was_capture": None, "captured_player": None,
                "captured_token": None, "captures": [], "landed_on_safe": None,
                "landed_in_home_path": None, "finished_a_token": None,
                "left_piece_exposed": None}
        st["dice_counts"][str(dice)] += 1
        if skipped:
            st["skipped_turns"] += 1
        else:
            eff = pre["effects"][chosen]
            exposed = is_exposed(env, player, eff["to"])      # position after the move
            first = eff["captures"][0] if eff["captures"] else None
            move = {
                "chosen_index": chosen, "from_position": eff["from"],
                "to_position": eff["to"], "action_class": eff["action_class"],
                "was_capture": eff["capture"],
                "captured_player": first["player"] if first else None,
                "captured_token": first["token"] if first else None,
                "captures": eff["captures"],
                "landed_on_safe": eff["safe"], "landed_in_home_path": eff["in_home"],
                "finished_a_token": eff["home_finish"],
                "left_piece_exposed": exposed,
            }
            self.moves_applied += 1
            st["decisions"] += 1
            for kind in AVAILABILITY_KINDS:
                key = _STAT_KEYS[kind]
                if pre["availability"][kind]["any"]:
                    st[f"{key}_available"] += 1
                    if chosen in pre["availability"][kind]["tokens"]:
                        st[f"{key}_taken"] += 1
            st["captures_made"] += len(eff["captures"])
            for c in eff["captures"]:
                victim = self.stats[c["player"]]
                victim["times_captured"] += 1
                if self.exposed.pop((c["player"], c["token"]), None) is not None:
                    victim["pieces_lost_after_exposure"] += 1
            self.exposed.pop((player, chosen), None)          # it has moved on
            if exposed:
                st["exposures"] += 1
                self.exposed[(player, chosen)] = eff["to"]
            if eff["home_finish"] and self.first_finish is None:
                self.first_finish = (self.ply, self.turn_number)

        progress_after = progress(env)
        top = max(progress_after, key=lambda p: (progress_after[p], -int(p)))
        if progress_after[top] > 0 and (
                self.leader is None or progress_after[top] > progress_after[self.leader]):
            if self.leader is not None:
                self.lead_changes += 1
            self.leader = top

        model = self._model_fields(player, skipped)
        info = self.run.players[player]
        config = dict(self.run.config)
        config["num_ctx"] = self.run.num_ctx(
            player, model_replied=model["raw_reply"] is not None)

        self.run.write_decision({
            "run_id": self.run.run_id, "game_id": self.game_id, "seed": self.seed,
            "ply": self.ply, "turn_number": self.turn_number,
            "player_id": player, "seat_position": info["seat_position"],
            "model_spec": info["model_spec"], "host": info["host"],
            "skipped": skipped,
            "tokens_before": before, "tokens_after": snapshot(env),
            "dice": dice, "legal_moves": legal, "is_extra_turn": is_extra,
            "piece_counts_before": pre["piece_counts"],
            "progress_before": pre["progress"], "progress_after": progress_after,
            **move,
            "availability": pre["availability"],
            "heuristic_move": pre["heuristic_move"],
            "heuristic_score_per_legal_move": pre["heuristic_scores"],
            **model,
            "config": config,
        })
        self.prev_player, self.prev_dice = player, dice
        self.ply += 1

    def game_record(self, termination_reason, error=None):
        env = self.env
        final_progress = progress(env)
        finished = {pid: piece_counts(env)[str(pid)]["finished"] for pid in env.player_ids}
        order = sorted(env.player_ids,
                       key=lambda p: (p != env.winner, -finished[p],
                                      -final_progress[str(p)],
                                      self.run.players[p]["seat_position"]))
        per_player = {}
        for pid in env.player_ids:
            agent = self.agents[pid]
            agent = getattr(agent, "inner", agent)
            llm = agent.stats() if hasattr(agent, "stats") else None
            per_player[str(pid)] = {**self.run.players[pid], **self.stats[pid],
                                    "pieces_finished": finished[pid],
                                    "final_progress": final_progress[str(pid)],
                                    "llm": llm}
        now = time.time()
        return {
            "run_id": self.run.run_id, "game_id": self.game_id,
            "game_index": self.game_index, "seed": self.seed,
            "players": {str(pid): p["model_spec"] for pid, p in self.run.players.items()},
            "seat_assignment": {str(pid): p["seat_position"]
                                for pid, p in self.run.players.items()},
            "winner": env.winner,
            "winner_model": (self.run.players[env.winner]["model_spec"]
                             if env.winner is not None else None),
            "finishing_order": order,
            "termination_reason": termination_reason,
            "error": error,
            "dice_rolls": self.ply,
            "moves_applied": self.moves_applied,
            "skipped_turns": self.ply - self.moves_applied,
            "turns": self.turn_number + 1,
            "turns_to_first_finish": self.first_finish[1] if self.first_finish else None,
            "ply_of_first_finish": self.first_finish[0] if self.first_finish else None,
            "lead_changes": self.lead_changes,
            "final_tokens": snapshot(env),
            "timings": {
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S",
                                            time.localtime(self.started)),
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
                "seconds": round(now - self.started, 1),
                "llm_latency_ms": round(sum((s["llm"] or {}).get("latency_ms", 0)
                                            for s in per_player.values()), 1),
            },
            "per_player": per_player,
            "config": self.run.config,
        }
