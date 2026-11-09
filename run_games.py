"""
Run full Ludo games between two (or more) LLMs, or an LLM against a baseline.

Examples
--------
Two local Ollama models, 5 games each side:
    python run_games.py --p0 ollama:llama3.2 --p1 ollama:qwen2.5:7b --games 5

Local model against the heuristic baseline:
    python run_games.py --p0 ollama:llama3.2 --p1 heuristic --games 10

Play against a thinking model with its reasoning switched off:
    python run_games.py --p0 human --p1 ollama_native:qwen3.5:9b --no-think

Local model against a model on another laptop's Ollama (that machine must run
Ollama with OLLAMA_HOST=0.0.0.0), with JSON replies and last-move context:
    python run_games.py --p0 ollama_native:qwen3.5:9b \
        --p1 ollama_native://<host>:11434/qwen2.5vl:3b --structured --last-move

Offline smoke test (no API, no Ollama):
    python run_games.py --p0 dummy --p1 dummy --games 2

Withhold legal moves from the prompt (spot-mode behaviour):
    python run_games.py --p0 ollama:llama3.2 --p1 heuristic --no-legal-moves

Let a reasoning model actually reason (removes the 60-token cap and newline stop):
    python run_games.py --p0 ollama:deepseek-r1 --p1 heuristic --uncapped

Same, but with a longer token limit instead of none:
    python run_games.py --p0 ollama:deepseek-r1 --p1 heuristic --max-tokens 1024

Each run writes to its own folder (--out, default llm_game_results/<run_id>/):
    decisions.jsonl   one line per dice roll, written as the game is played
    games.jsonl       one line per game
    run.json          configuration, versions, and definitions of the fields
An existing folder is never reused, so a run cannot overwrite earlier results.
See decision_log.py for the record layout.
"""

import argparse
import os
import sys
import time

import ludo_rules as rules
from ludo_env import LudoEnv, STREAMS
from agents import BaseAgent, RandomAgent, HeuristicAgent
from llm_agent import LLMAgent
from evaluation import evaluate
from llm_backends import (make_llm, parse_spec, check_ollama, reason_first,
                          DEFAULT_SYSTEM_PROMPT, DEFAULT_TIMEOUT,
                          SCHEMA_ORDERS, DEFAULT_SCHEMA_ORDER, REASON_MAX_CHARS)
from board_render import render_board, describe_move, group_moves
from decision_log import RunLog, GameLog, CONTEXT_MODE, short_hash, move_effects

class FirstLegalAgent(BaseAgent):
    """Always moves the first piece of the legal-move list (the lowest index)."""

    def select_move(self, env, player, dice, legal_moves):
        return legal_moves[0]


BASELINES = {"random": RandomAgent, "heuristic": HeuristicAgent,
             "first": FirstLegalAgent}
LEGAL_ORDERS = ("ascending", "reversed", "shuffled")


def is_llm_spec(spec):
    return not (spec in ("human", "dummy") or spec in BASELINES
                or spec == "gt" or spec.startswith("gt:"))


def _short(text, limit=220):
    """Trim a long model reply for display; reasoning blocks can be huge."""
    if text is None:
        return "None"
    t = " ".join(str(text).split())
    if len(t) <= limit:
        return repr(t)
    return repr(t[:limit // 2] + f" ... [{len(t)} chars total] ... " + t[-limit // 2:])


class HumanAgent(BaseAgent):
    """Asks you which token to move. Draws the board first."""

    def __init__(self, labels=None):
        self.labels = labels

    def select_move(self, env, player, dice, legal_moves):
        print("\n" + "=" * 58)
        print(f"  YOUR TURN  (P{player})     dice rolled: {dice}")
        render_board(env, highlight=player, labels=self.labels)

        print("  Your options:")
        for g in group_moves(env, player, dice, legal_moves):
            primary = g[0]
            same = (f"   (tokens {','.join(map(str, g))} do the same thing)"
                    if len(g) > 1 else "")
            print(f"    [{primary}]  {describe_move(env, player, dice, primary)}{same}")
        print()

        while True:
            raw = input(f"  Choose token {legal_moves} (or 'q' to quit): ").strip()
            if raw.lower() in ("q", "quit", "exit"):
                print("\nBye.")
                sys.exit(0)
            try:
                choice = int(raw)
            except ValueError:
                print("  Enter a number.")
                continue
            if choice in legal_moves:
                return choice
            print(f"  {choice} is not legal here. Legal: {legal_moves}")


def _bump(counts, key):
    counts[key] = counts.get(key, 0) + 1


class _TimedCall:
    """
    Wraps a backend call to keep the prompt and the wall-clock latency. Every
    other attribute (last_finish_reason, base_url, ...) passes through to the
    backend function.
    """

    def __init__(self, fn):
        self._fn = fn
        self.last_prompt = None
        self.last_latency_ms = None
        self.edit_prompt = None       # optional prompt -> prompt, set per call

    def __call__(self, prompt):
        if self.edit_prompt is not None:
            prompt = self.edit_prompt(prompt)
        self.last_prompt = prompt
        t0 = time.perf_counter()
        try:
            return self._fn(prompt)
        finally:
            self.last_latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    def __getattr__(self, name):
        return getattr(self._fn, name)


class TrackedLLMAgent(LLMAgent):
    """
    LLMAgent that also records, for its own player, what each reply was and how
    it ended.

    env.llm_fallbacks is one counter shared by every LLM in the game, so it
    cannot give a per-model invalid rate; this can. After each model call,
    last_decision holds that call's details for the decision log.
    """

    def __init__(self, llm_call_fn, *args, last_move=False,
                 move_outcomes=False, legal_order="ascending", **kwargs):
        super().__init__(_TimedCall(llm_call_fn), *args, **kwargs)
        self.last_move = last_move
        self.move_outcomes = move_outcomes
        self.legal_order = legal_order
        self.last_decision = None
        self.calls = 0
        self.invalid = 0
        self.finish_reasons = {}
        self.invalid_by_finish_reason = {}
        self.invalid_by_cause = {}
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.latency_ms = 0.0

    def _parsed_index(self, text, legal_moves, original_rules):
        """
        The index LLMAgent's parser read from the reply, legal or not; None if it
        read none. LLMAgent only hands back the final move, so the same parse is
        repeated here on the same text.
        """
        try:
            return self.parse_reply(text, legal_moves, original_rules)[0]
        except Exception:
            return None

    def select_move(self, env, player, dice, legal_moves):
        if self.last_move:
            # Worded for this player ("you", "your token"), so set per prompt.
            recent = getattr(env, "last_move_records", [])[-int(self.last_move):]
            env.history_text = last_moves_text(env, recent, player) if recent else None
        self.last_decision = None
        calls_before = env.llm_calls
        # The list as the prompt shows it. The decision log keeps the engine's
        # ascending list either way.
        shown = list(legal_moves)
        if self.legal_order == "reversed":
            shown.reverse()
        elif self.legal_order == "shuffled":
            env.rng["legal_order"].shuffle(shown)
        self.llm_call.edit_prompt = None
        if self.move_outcomes and self.include_legal_moves_in_prompt:
            self.llm_call.edit_prompt = add_move_outcomes(env, player, dice, shown)
        choice = super().select_move(env, player, dice, shown)
        if env.llm_calls == calls_before:          # no model call was made
            return choice

        call = self.llm_call
        why = getattr(call, "last_finish_reason", None)
        fallback = env.last_llm_fallback
        # What the model sent: the backend's text, or LLMAgent's copy for the
        # dummy backend.
        raw = getattr(call, "last_content", None)
        if raw is None:                            # dummy backend
            raw = env.last_llm_raw
        parsed = self._parsed_index(env.last_llm_raw, legal_moves, env.original_rules)
        cause = None
        if fallback:
            if why == "timeout":
                cause = "timeout"
            elif not raw:
                cause = "empty"
            elif parsed is None:
                cause = "unparseable"
            else:
                cause = "illegal"
        p_tok = getattr(call, "last_prompt_tokens", None)
        c_tok = getattr(call, "last_completion_tokens", None)
        self.last_decision = {
            "raw_reply": raw,
            "parsed_index": parsed,
            "parsed_reason": env.last_llm_reason or None,
            "reason_chars": len(env.last_llm_reason) if parsed is not None else None,
            "reason_first": reason_first(raw or ""),
            "thinking_tokens": getattr(call, "last_thinking_tokens", None),
            "hit_ceiling": why == "length",
            "reasoning_text": getattr(call, "last_reasoning", "") or None,
            "finish_reason": why,
            "was_fallback": fallback,
            "fallback_cause": cause,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "latency_ms": call.last_latency_ms,
            "prompt_hash": short_hash(call.last_prompt),
        }

        self.calls += 1
        self.prompt_tokens += p_tok or 0
        self.completion_tokens += c_tok or 0
        self.latency_ms += call.last_latency_ms or 0
        _bump(self.finish_reasons, why or "unknown")
        if fallback:
            self.invalid += 1
            _bump(self.invalid_by_finish_reason, why or "unknown")
            _bump(self.invalid_by_cause, cause)
        return choice

    def stats(self):
        return {"calls": self.calls, "invalid": self.invalid,
                "invalid_by_cause": self.invalid_by_cause,
                "finish_reasons": self.finish_reasons,
                "invalid_by_finish_reason": self.invalid_by_finish_reason,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
                "latency_ms": round(self.latency_ms, 1)}


class WatchAgent(BaseAgent):
    """Wraps an agent and narrates the move it chose (board is drawn after apply)."""

    def __init__(self, inner, label, show_reason=True):
        self.inner = inner
        self.label = label
        # Only an LLMAgent sets last_llm_* on env. For baselines those attributes
        # hold stale values left by whichever LLM moved last, so don't read them.
        self.is_llm = isinstance(inner, LLMAgent)
        self.show_reason = show_reason and self.is_llm

    def select_move(self, env, player, dice, legal_moves):
        choice = self.inner.select_move(env, player, dice, legal_moves)
        print("\n" + "-" * 58)
        print(f"  P{player} [{self.label}]  rolled {dice}  ->  "
              f"{describe_move(env, player, dice, choice)}")
        if self.show_reason:
            reason = getattr(env, "last_llm_reason", "")
            raw = getattr(env, "last_llm_raw", None)
            if getattr(env, "last_llm_fallback", False):
                print("  !! INVALID reply, random legal move substituted")
                raw = raw or getattr(self.inner.llm_call, "last_content", "")
                print(f"     model said: {_short(raw) if raw else '(empty content)'}")
                why = getattr(self.inner.llm_call, "last_finish_reason", None)
                if why:
                    note = {"length": "ran out of tokens",
                            "timeout": "request timed out"}.get(why, "")
                    print(f"     reply ended: {why}" + (f" ({note})" if note else ""))
                # Display only: the reasoning is never parsed for a move.
                thinking = getattr(self.inner.llm_call, "last_reasoning", "")
                if thinking:
                    print(f"     reasoning (not parsed): {_short(thinking)}")
            elif reason:
                print(f"  reason: {_short(reason, 300)}")
            elif raw is not None:
                print(f"  (no reason given) model said: {_short(raw)}")
        return choice


def attach_watcher(env, labels, delay=0.0):
    """Redraw the board after every applied move, so what you see is post-move."""
    original = env.apply_move

    def wrapped(player, token_idx, dice):
        result = original(player, token_idx, dice)
        render_board(env, highlight=player, labels=labels)
        if delay:
            time.sleep(delay)
        return result

    env.apply_move = wrapped


def attach_last_move(env, keep=1):
    """
    Record the last `keep` applied moves on env.last_move_records for the
    --last-move / --last-moves prompt lines. env.move_log only keeps {player, capture, to}, so the token, its
    origin, and whose tokens were captured are taken from the board before and
    after the move instead.
    """
    original = env.apply_move
    env.last_move_records = []

    def wrapped(player, token_idx, dice):
        before = {pid: list(env.tokens[pid]) for pid in env.player_ids}
        result = original(player, token_idx, dice)
        captured = [pid for pid in env.player_ids if pid != player
                    for b, a in zip(before[pid], env.tokens[pid])
                    if b != -1 and a == -1]
        env.last_move_records.append({
            "player": player, "dice": dice, "token": token_idx,
            "from": before[player][token_idx], "to": env.tokens[player][token_idx],
            "captured": captured,
        })
        del env.last_move_records[:-keep]
        return result

    env.apply_move = wrapped


def move_outcome_text(env, player, dice, token):
    """One line saying what moving this token would do, for --move-outcomes."""
    e = move_effects(env, player, dice, token)

    def where(pos):
        if pos == -1:
            return "base"
        if pos < env.BOARD_SIZE:
            return f"square {pos}"
        return f"home-path position {pos}"

    line = f"- Token {token}: moves from {where(e['from'])} to {where(e['to'])}"
    notes = []
    if e["bring_out"]:
        notes.append("brings a new token out of base")
    if e["home_finish"]:
        notes.append("finishes this token")
    elif e["in_home"]:
        notes.append("is in your home path after the move")
    if e["capture"]:
        n = len(e["captures"])
        notes.append("captures an opponent token" if n == 1
                     else f"captures {n} opponent tokens")
    if e["safe"]:
        notes.append("lands on a safe square")
    return line + (", " + ", ".join(notes) if notes else "") + "."


LEGAL_LIST_LEAD = "The following token indices are legal to move this turn:\n"


def add_move_outcomes(env, player, dice, shown):
    """
    A prompt editor that adds one outcome line per legal move right after the
    legal-move list. The outcomes are facts the engine computes (destination,
    capture, safe square, base, home path); no move is recommended.
    """
    lines = "\n".join(move_outcome_text(env, player, dice, t) for t in shown)
    anchor = f"{LEGAL_LIST_LEAD}{shown}\n"
    block = f"{anchor}\nWhat each legal move does:\n{lines}\n"

    def edit(prompt):
        if prompt.count(anchor) != 1:
            raise RuntimeError("--move-outcomes: legal-move list not found in prompt")
        return prompt.replace(anchor, block)
    return edit


def last_moves_text(env, moves, viewer):
    """
    The history block: one line for --last-move, or a short oldest-first list
    for --last-moves N. Turns with no legal move are not listed.
    """
    if len(moves) == 1:
        return last_move_text(env, moves[0], viewer)
    lines = [last_move_text(env, mv, viewer, prefix="- ") for mv in moves]
    return f"Last {len(moves)} moves, oldest first:\n" + "\n".join(lines)


def last_move_text(env, mv, viewer, prefix="Previous move: "):
    """One line describing a move, from `viewer`'s point of view."""
    mover = mv["player"]
    home_end = rules.home_range(mover)[1]

    def where(pos):
        if pos == -1:
            return "base"
        if pos < env.BOARD_SIZE:
            return f"square {pos}"
        return f"home-path position {pos}"

    who = "You" if mover == viewer else f"Player {mover}"
    line = (f"{prefix}{who} rolled {mv['dice']} and moved token {mv['token']} "
            f"from {where(mv['from'])} to {where(mv['to'])}")
    if mv["to"] == home_end:
        line += ", finishing that token"
    for pid in sorted(set(mv["captured"])):
        n = mv["captured"].count(pid)
        whose = "your" if pid == viewer else f"Player {pid}'s"
        line += f", capturing {n} of {whose} tokens" if n > 1 \
            else f", capturing one of {whose} tokens"
    return line + "."


def build_agent(spec, include_legal_moves, uncapped, labels=None, no_think=False,
                max_tokens=None, structured=False, last_move=False,
                schema_order=DEFAULT_SCHEMA_ORDER, reason_max_chars=None,
                move_outcomes=False, legal_order="ascending"):
    """spec is 'human', 'random', 'heuristic', 'first', 'gt[:depth]', 'dummy',
    or 'provider:model'."""
    if spec == "human":
        return HumanAgent(labels), "human"

    if spec in BASELINES:
        return BASELINES[spec](), spec

    if spec == "gt" or spec.startswith("gt:"):
        from game_theory_multiplayer_agent import GameTheoryMultiplayerAgent
        depth = int(spec.split(":", 1)[1]) if ":" in spec else 2
        return GameTheoryMultiplayerAgent(search_depth=depth), f"gt(d={depth})"

    kwargs = {}
    robust = False
    if uncapped:
        # No newline stop, and no token limit unless --max-tokens was given, so
        # the model can reason first. robust_parse then reads the answer off the
        # end. The request timeout still bounds a runaway reply.
        kwargs.update(max_tokens=max_tokens, stop=None)
        robust = True
    if no_think:
        kwargs.update(think=False)
    if structured:
        kwargs.update(structured=True, schema_order=schema_order,
                      reason_max_chars=reason_max_chars)
    fn = make_llm(spec, **kwargs)
    return TrackedLLMAgent(fn, include_legal_moves_in_prompt=include_legal_moves,
                           robust_parse=robust, structured=structured,
                           last_move=last_move, move_outcomes=move_outcomes,
                           legal_order=legal_order, schema_order=schema_order,
                           reason_max_chars=reason_max_chars), spec


def player_info(specs, seats):
    """pid -> who plays that seat and where its model is served."""
    players = {}
    for seat, (pid, spec) in enumerate(zip(seats, specs)):
        provider = model = host = None
        if is_llm_spec(spec):
            provider, model, host = parse_spec(spec)
        players[pid] = {"model_spec": spec, "host": host, "seat_position": seat,
                        "provider": provider, "model": model}
    return players


def run_games(specs, games, seed0, include_legal_moves, uncapped, out_dir, run_id,
              max_steps=1000, watch=False, delay=0.0, no_think=False,
              max_tokens=None, structured=False, last_move=False,
              ollama_versions=None, schema_order=DEFAULT_SCHEMA_ORDER,
              reason_max_chars=None, move_outcomes=False, legal_order="ascending",
              original_rules=False):
    seats = rules.seats(len(specs), original_rules)
    players = player_info(specs, seats)
    # The field order only exists when structured output is on.
    schema_order = schema_order if structured else None
    # Kept in run.json so a results folder is self-describing. The capped values
    # are make_llm's defaults, which reproduce the released code. max_tokens is
    # null when uncapped without --max-tokens.
    settings = {
        "original_rules": original_rules,
        "seats": seats,
        "first_player": ("seat 0" if original_rules else
                         "roll-off on the dice stream; only players tied for the "
                         "highest roll re-roll"),
        "random_streams": ("one shared generator: random.seed(game seed)"
                           if original_rules else
                           {label: f"random.Random('<game seed>:{label}')"
                            for label in STREAMS}),
        "specs": list(specs),
        "games": games,
        "seed0": seed0,
        "structured_output": structured,
        "schema_order": schema_order,
        "reason_max_chars": reason_max_chars,
        "last_move_context": bool(last_move),
        "last_moves": int(last_move),
        "move_outcomes": move_outcomes,
        "legal_order": legal_order,
        # Resolved server per LLM player, e.g. a partner's laptop on the LAN.
        "llm_endpoints": {str(pid): p["host"] for pid, p in players.items()
                          if p["host"]},
        "include_legal_moves": include_legal_moves,
        "uncapped": uncapped,
        "max_tokens": max_tokens if uncapped else 60,
        "stop": None if uncapped else ["\n"],
        "no_think": no_think,
        "max_steps": max_steps,
        "temperature": 0,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "request_timeout_s": DEFAULT_TIMEOUT,
    }
    # Copied onto every decision record. num_ctx is filled in per player.
    config = {
        "original_rules": original_rules,
        "no_think": no_think,
        "uncapped": uncapped,
        "max_tokens": settings["max_tokens"],
        "stop": settings["stop"],
        "structured": structured,
        "schema_order": schema_order,
        "reason_max_chars": reason_max_chars,
        "legal_moves_shown": include_legal_moves,
        "last_move_context": bool(last_move),
        "last_moves": int(last_move),
        "move_outcomes": move_outcomes,
        "legal_order": legal_order,
        "temperature": settings["temperature"],
        "system_prompt_hash": short_hash(DEFAULT_SYSTEM_PROMPT),
        "num_ctx": None,
        "context_mode": CONTEXT_MODE,
    }

    run = None
    records = []
    for g in range(games):
        seed = seed0 + g
        env = LudoEnv(seed=seed, num_players=len(specs), original_rules=original_rules)
        if run is None:                             # needs an env for its constants
            run = RunLog(out_dir, run_id, players, settings, config,
                         ollama_versions or {}, env)

        labels = {pid: spec for pid, spec in zip(env.player_ids, specs)}
        human_playing = "human" in specs
        agents = {}
        for pid, spec in zip(env.player_ids, specs):
            agent, label = build_agent(spec, include_legal_moves, uncapped, labels,
                                       no_think=no_think, max_tokens=max_tokens,
                                       structured=structured, last_move=last_move,
                                       schema_order=schema_order or DEFAULT_SCHEMA_ORDER,
                                       reason_max_chars=reason_max_chars,
                                       move_outcomes=move_outcomes,
                                       legal_order=legal_order)
            if watch and spec != "human":
                agent = WatchAgent(agent, label)
            agents[pid] = agent
            labels[pid] = label

        if last_move:
            attach_last_move(env, keep=int(last_move))

        if watch or human_playing:
            print(f"\n{'=' * 58}\n  GAME {g + 1}/{games}  (seed {seed})\n{'=' * 58}")
            if watch:
                render_board(env, labels=labels)
                attach_watcher(env, labels, delay=delay)

        glog = GameLog(run, env, agents, game_index=g, seed=seed)
        run.game_started()
        failure = None
        try:
            winner, _ = env.play_game(agents, max_steps=max_steps)
            reason = "win" if winner is not None else "max_steps"
        except BaseException as e:                  # includes Ctrl+C and a human quitting
            failure, winner, reason = e, None, "error"

        stats = evaluate(env)
        rec = glog.game_record(reason, error=repr(failure) if failure else None)
        rec["llm_calls"] = stats.get("llm_calls", 0)
        rec["llm_fallbacks"] = stats.get("llm_fallbacks", 0)
        rec["captures"] = stats.get("captures", 0)
        rec["safe_moves"] = stats.get("safe_moves", 0)
        rec["first_player"] = env.first_player
        rec["roll_off"] = env.roll_off
        run.write_game(rec)
        records.append(rec)
        if failure is not None:
            # The plies and the partial game are on disk. A failure such as an
            # unreachable server would hit every later game too, so stop here.
            run.close()
            raise failure

        fb, calls = rec["llm_fallbacks"], rec["llm_calls"]
        rate = f"{fb}/{calls}" if calls else "n/a"
        print(f"  game {g + 1}/{games} (seed {seed}): "
              f"winner P{winner} = {rec['winner_model']}, "
              f"{rec['dice_rolls']} dice rolls, {rec['timings']['seconds']:.0f}s, "
              f"invalid {rate}")

    if run is not None:
        run.close()
    return records


def summarise(records, specs):
    n = len(records)
    if not n:
        return
    print("\n" + "=" * 60)
    print(f"{n} games: " + "  vs  ".join(specs))
    print("=" * 60)

    wins = {}
    for r in records:
        key = f"P{r['winner']} ({r['winner_model']})" if r["winner"] is not None \
            else "no winner"
        wins[key] = wins.get(key, 0) + 1
    for k, v in sorted(wins.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<40} {v}/{n}  ({v / n:.0%})")

    calls = sum(r["llm_calls"] for r in records)
    fallbacks = sum(r["llm_fallbacks"] for r in records)
    print(f"\n  mean dice rolls   {sum(r['dice_rolls'] for r in records) / n:.0f}")
    print(f"  mean seconds/game "
          f"{sum(r['timings']['seconds'] for r in records) / n:.0f}")
    print(f"  mean captures     {sum(r['captures'] for r in records) / n:.1f}")
    if calls:
        print(f"  LLM calls         {calls}")
        print(f"  invalid/fallback  {fallbacks}/{calls}  ({fallbacks / calls:.1%})")

    per_player = {}
    for r in records:
        for pid, p in r["per_player"].items():
            llm = p["llm"]
            if not llm:
                continue
            agg = per_player.setdefault((pid, p["model_spec"]),
                                        {"calls": 0, "invalid": 0, "why": {}, "cause": {}})
            agg["calls"] += llm["calls"]
            agg["invalid"] += llm["invalid"]
            for why, k in llm["invalid_by_finish_reason"].items():
                agg["why"][why] = agg["why"].get(why, 0) + k
            for cause, k in llm["invalid_by_cause"].items():
                agg["cause"][cause] = agg["cause"].get(cause, 0) + k
    for (pid, model), agg in sorted(per_player.items()):
        if not agg["calls"]:
            continue
        print(f"  P{pid} {model}: invalid {agg['invalid']}/{agg['calls']} "
              f"({agg['invalid'] / agg['calls']:.1%})")
        if agg["invalid"]:
            cause = ", ".join(f"{k} {v}" for k, v in sorted(agg["cause"].items()))
            why = ", ".join(f"{k} {v}" for k, v in sorted(agg["why"].items()))
            print(f"      by cause: {cause}   by finish reason: {why}")


def main():
    ap = argparse.ArgumentParser(description="Full Ludo games: LLMs, baselines, or you.")
    ap.add_argument("--p0", default="dummy",
                    help="agent for the first seat (player 0): human | random | heuristic "
                         "| first | gt[:depth] "
                         "| dummy | provider:model")
    ap.add_argument("--p1", default="heuristic",
                    help="agent for the second seat: player 2 in a two-player game "
                         "(player 1 with --original-rules), otherwise player 1")
    ap.add_argument("--p2", default=None, help="agent for the third seat, player 2 (optional)")
    ap.add_argument("--p3", default=None, help="agent for the fourth seat, player 3 (optional)")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=0, help="first seed; games use seed0..seed0+n-1")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--no-legal-moves", action="store_true",
                    help="withhold the legal-move list from the prompt (spot-mode behaviour)")
    ap.add_argument("--uncapped", action="store_true",
                    help="remove max_tokens=60 and the newline stop, with no token "
                         "limit at all (still bounded by the request timeout); "
                         "use --max-tokens N instead to keep a limit")
    ap.add_argument("--no-think", action="store_true",
                    help="disable a thinking model's reasoning phase; only works with "
                         "ollama_native:<model> (Ollama's /v1 endpoint ignores it)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="replace the paper's 60-token limit with N and drop the "
                         "newline stop (default: 60 tokens + newline stop, as in "
                         "the paper)")
    ap.add_argument("--out", default=None,
                    help="folder for this run's logs (must not already exist); "
                         "default is llm_game_results/<timestamp>_<players>/")
    ap.add_argument("--watch", action="store_true",
                    help="draw the board after every move (spectator mode)")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="seconds to pause after each move when watching")
    ap.add_argument("--structured", action="store_true",
                    help="with --original-rules: constrain replies to JSON {reason, "
                         "piece} via Ollama's 'format' parameter (always on otherwise); "
                         "every LLM player must be ollama_native")
    ap.add_argument("--schema-order", default=None,
                    choices=[o.replace("_", "-") for o in SCHEMA_ORDERS],
                    help="with --structured: which JSON field the model writes first "
                         "(default reason-first; piece-first makes it answer before "
                         "it has reasoned)")
    ap.add_argument("--reason-max-chars", type=int, default=None, metavar="N",
                    help="limit the reason to N characters inside the JSON schema, so "
                         "a long-winded model must stop and name its piece instead of "
                         "running into the token limit (default: %d; no limit with "
                         "--original-rules)" % REASON_MAX_CHARS)
    ap.add_argument("--last-move", action="store_true",
                    help="add the previous move to the prompt's HISTORY / CONTEXT block")
    ap.add_argument("--last-moves", type=int, default=None, metavar="N",
                    help="add the last N moves (both players') to that block "
                         "instead; --last-moves 1 is the same as --last-move")
    ap.add_argument("--move-outcomes", action="store_true",
                    help="after the legal-move list, add one line per legal move "
                         "saying where the token lands and whether it captures, "
                         "reaches a safe square, leaves base or enters the home path")
    ap.add_argument("--legal-order", default="ascending", choices=LEGAL_ORDERS,
                    help="order of the legal-move list in the prompt (default "
                         "ascending, as in the released code; shuffled draws a new "
                         "order for each prompt from the game's legal_order stream)")
    ap.add_argument("--original-rules", action="store_true",
                    help="play by the released LudoBench rules, to reproduce the "
                         "original paper and its logged games")
    args = ap.parse_args()
    if args.move_outcomes and args.no_legal_moves:
        ap.error("--move-outcomes needs the legal-move list in the prompt")
    if args.last_moves is not None:
        if args.last_moves < 1:
            ap.error("--last-moves must be a positive number")
        if args.last_move and args.last_moves != 1:
            ap.error("give --last-move or --last-moves N, not both")
    last_moves = args.last_moves or int(args.last_move)
    if args.max_tokens is not None:
        # A custom token limit always drops the newline stop too: keeping the
        # stop would give a setup that is neither paper nor uncapped.
        args.uncapped = True

    # Windows writes redirected or piped stdout as cp1252, and printing a model
    # reply containing e.g. U+2011 or an arrow then crashes the run.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    specs = [s for s in (args.p0, args.p1, args.p2, args.p3) if s]
    if len(specs) not in (2, 3, 4):
        raise SystemExit("need 2-4 players")

    seat_specs = dict(zip(rules.seats(len(specs), args.original_rules), specs))
    llm_players = [(pid, s) for pid, s in seat_specs.items() if is_llm_spec(s)]
    try:
        resolved = {pid: parse_spec(s) for pid, s in llm_players}
    except ValueError as e:
        ap.error(str(e))

    # Our rules always ask for the JSON reply; the released one asked for "<int> | <reason>".
    structured = args.structured or not args.original_rules
    if args.schema_order and not structured:
        ap.error("--schema-order requires --structured")
    schema_order = (args.schema_order or DEFAULT_SCHEMA_ORDER).replace("-", "_")
    reason_max_chars = args.reason_max_chars
    if reason_max_chars is not None:
        if not structured:
            ap.error("--reason-max-chars requires --structured")
        if reason_max_chars < 1:
            ap.error("--reason-max-chars must be a positive number")
    elif not args.original_rules:
        reason_max_chars = REASON_MAX_CHARS

    if structured:
        other = [s for pid, s in seat_specs.items()
                 if pid in resolved and resolved[pid][0] != "ollama_native"]
        if other:
            ap.error(f"JSON replies need every LLM player to be ollama_native "
                     f"(or use --original-rules); not: {', '.join(other)}")

    # Check every Ollama server answers before any game starts, so a partner's
    # laptop that is off or firewalled fails now with its host and port named.
    ollama_versions = {}
    for pid, (provider, model, base_url) in resolved.items():
        where = base_url
        if provider in ("ollama", "ollama_native"):
            try:
                ollama_versions[base_url] = check_ollama(base_url)
                where += f"  (Ollama {ollama_versions[base_url]})"
            except RuntimeError as e:
                raise SystemExit(f"P{pid} {seat_specs[pid]}: {e}")
        print(f"P{pid} {model} -> {where}")

    if args.no_think:
        ignored = [s for s in specs if s.startswith(("ollama:", "openrouter:"))]
        if ignored:
            print(f"WARNING: --no-think has no effect on {', '.join(ignored)}; "
                  f"use ollama_native:<model> to switch thinking off.")

    out = args.out
    if out is None:
        tag = "_vs_".join(s.replace("/", "-").replace(":", "-") for s in specs)
        out = os.path.join("llm_game_results",
                           f"{time.strftime('%Y%m%d-%H%M%S')}_{tag}")
    elif os.path.exists(out):
        raise SystemExit(f"{out} already exists; pick another --out "
                         f"(refusing to overwrite earlier results)")
    run_id = os.path.basename(os.path.normpath(out))

    print(f"\nRunning {args.games} game(s): " + "  vs  ".join(specs))
    if not args.uncapped:
        cap = "60 tokens + newline stop (paper)"
    elif args.max_tokens is None:
        cap = "none"
    else:
        cap = f"{args.max_tokens} tokens, no newline stop"
    print(f"legal moves in prompt: {not args.no_legal_moves}   output cap: {cap}")
    shown = "False"
    if structured:
        limit = f", reason <= {reason_max_chars} chars" if reason_max_chars else ""
        shown = f"True ({schema_order}{limit})"
    print(f"structured output: {shown}   last-move context: {last_moves or False}")
    if args.move_outcomes or args.legal_order != "ascending":
        print(f"move outcomes in prompt: {args.move_outcomes}   "
              f"legal-move order: {args.legal_order}")
    print("rules: " + ("original (released LudoBench)" if args.original_rules else "ours"))
    print(f"writing to {out}\n")

    records = run_games(
        specs,
        games=args.games,
        seed0=args.seed0,
        include_legal_moves=not args.no_legal_moves,
        uncapped=args.uncapped,
        out_dir=out,
        run_id=run_id,
        ollama_versions=ollama_versions,
        max_steps=args.max_steps,
        watch=args.watch,
        delay=args.delay,
        no_think=args.no_think,
        max_tokens=args.max_tokens,
        structured=structured,
        schema_order=schema_order,
        reason_max_chars=reason_max_chars,
        last_move=last_moves,
        move_outcomes=args.move_outcomes,
        legal_order=args.legal_order,
        original_rules=args.original_rules,
    )
    summarise(records, specs)
    print(f"\nLogs: {out}  (decisions.jsonl, games.jsonl, run.json)")


if __name__ == "__main__":
    main()