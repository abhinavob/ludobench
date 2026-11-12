"""
Run full Ludo games between LLMs, bots or you, and log every decision.

With no switches a run uses the standard setup: our rules, the JSON reply
(reason first, at most 500 characters), a 1,024-token ceiling and an Ollama
context of 8,192 tokens (8,192 and 16,384 when the model thinks), legal moves shown in piece order, no move outcomes, no history.

    python run_games.py --players llm:qwen3:8b heuristic

Players are listed without seat numbers; seats follow the rules (two players
sit in seats 0 and 2). A player is llm:MODEL (Ollama on this machine),
llm://HOST[:PORT]/MODEL (Ollama on another machine, started there with
OLLAMA_HOST=0.0.0.0), llm:openrouter/MODEL or llm:nvidia/MODEL (an API, key in
OPENROUTER_API_KEY or NVIDIA_API_KEY), random, heuristic, first, gt[:DEPTH],
human or dummy. API models get the same prompt and reply settings, with
reasoning off where the API allows (the thinking ceiling where it cannot be);
--requests-per-minute N paces each API.

Switches, each off by default:
    --hide-legal-moves    leave the legal-move list out of the prompt
    --shuffle             show the legal moves in a random order (own stream)
    --outcomes            say after the list what each legal move does
    --history N           show the last N moves
    --think               let a thinking model think (off where it can be)
    --effort high         send a reasoning effort level (models that take levels)
    --original-rules      the released LudoBench rules, prompt and reply format

Results go to results/<experiment>/<players>/, the players in seat order and
any effort joined by "_", e.g. qwen3.5-4b_heuristic_effort-low (run.json,
games.jsonl, decisions.jsonl; see decision_log.py), and results/cells.csv lists
one row per run. Without --experiment and --games a run plays one game into
results/sandbox/, so trial games never land in a real experiment. Play against
a model, or watch a game:
    python run_games.py --players human llm:qwen3:8b
    python run_games.py --players llm:qwen3:8b heuristic --watch
"""

import argparse
import contextlib
import csv
import json
import os
import re
import sys
import time

import ludo_rules as rules
from ludo_env import LudoEnv, STREAMS
from agents import BaseAgent, RandomAgent, HeuristicAgent
from llm_agent import LLMAgent
from evaluation import evaluate
from llm_backends import (make_llm, parse_spec, check_ollama, model_info, reason_first,
                          thinking_setting, api_model_info, api_reasoning,
                          DEFAULT_SYSTEM_PROMPT, DEFAULT_TIMEOUT, TIMEOUT_RETRIES,
                          DEFAULT_SCHEMA_ORDER, REASON_MAX_CHARS, ANSWER_CEILING,
                          THINKING_CEILING, API_CEILING, ANSWER_CONTEXT, THINKING_CONTEXT,
                          API_PROVIDERS, API_RETRIES, PROVIDER_DEFAULTS)
from board_render import render_board, describe_move, group_moves
from decision_log import (RunLog, GameLog, CONTEXT_MODE, git_state, short_hash,
                          move_effects)

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
        self.cost_usd = 0.0
        self.retries = 0

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
        # The list as the prompt shows it; the decision log keeps both it and
        # the engine's ascending list.
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
        listed = self.include_legal_moves_in_prompt
        cause = None
        if fallback:
            if not raw:
                cause = "empty"
            elif parsed is None:
                cause = "unparseable"
            else:
                cause = "illegal"
        p_tok = getattr(call, "last_prompt_tokens", None)
        c_tok = getattr(call, "last_completion_tokens", None)
        cost = getattr(call, "last_cost", None)
        retries = getattr(call, "last_retries", 0)
        self.last_decision = {
            "shown_legal_moves": shown if listed else None,
            "chosen_list_position": shown.index(choice) if listed and not fallback else None,
            "provider": getattr(call, "provider", None),
            "served_by": getattr(call, "last_served_by", None),
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
            "cost_usd": cost,
            "retries": retries,
            "latency_ms": call.last_latency_ms,
            "prompt_hash": short_hash(call.last_prompt),
        }

        self.calls += 1
        self.prompt_tokens += p_tok or 0
        self.completion_tokens += c_tok or 0
        self.latency_ms += call.last_latency_ms or 0
        self.cost_usd += cost or 0
        self.retries += retries
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
                "cost_usd": self.cost_usd, "retries": self.retries,
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
        print(f"  -> {describe_move(env, player, dice, choice)}")
        if self.show_reason:
            reason = getattr(env, "last_llm_reason", "")
            raw = getattr(env, "last_llm_raw", None)
            if getattr(env, "last_llm_fallback", False):
                print("  !! INVALID reply, random legal move substituted")
                raw = raw or getattr(self.inner.llm_call, "last_content", "")
                print(f"     model said: {_short(raw) if raw else '(empty content)'}")
                why = getattr(self.inner.llm_call, "last_finish_reason", None)
                if why:
                    note = " (ran out of tokens)" if why == "length" else ""
                    print(f"     reply ended: {why}{note}")
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
    """
    Show how the game was started and every roll in order, with whose roll it
    is, and redraw the board after every applied move, so what you see is
    post-move. A roll with no legal move says whether the turn passes or, after
    a 6, the same player rolls again. Display only: nothing here changes the game.
    """
    name = lambda pid: f"P{pid} [{labels[pid]}]"
    if env.roll_off:
        print("  Roll-off: " + "; then ".join(", ".join(f"P{pid} {d}" for pid, d in rolls)
                                             for rolls in env.roll_off))
    print(f"  {name(env.first_player)} moves first")
    roll, apply_move = env.roll_dice, env.apply_move
    last = {"n": 0, "player": None, "dice": None}

    def rolled():
        dice = roll()
        player = env.player_ids[env.current_player_idx]
        again = " (another roll after a 6)" if (last["player"], last["dice"]) == (player, 6) else ""
        last.update(n=last["n"] + 1, player=player, dice=dice)
        print("\n" + "-" * 58)
        print(f"  roll {last['n']}: {name(player)} rolled {dice}{again}")
        if not env.get_legal_moves(player, dice):
            nxt = env.player_ids[(env.current_player_idx + 1) % env.NUM_PLAYERS]
            print("  -> no legal move; " + (f"a 6, so P{player} rolls again" if dice == 6
                                            else f"the turn passes to P{nxt}"))
        return dice

    def applied(player, token_idx, dice):
        result = apply_move(player, token_idx, dice)
        render_board(env, highlight=player, labels=labels)
        if delay:
            time.sleep(delay)
        return result

    env.roll_dice, env.apply_move = rolled, applied


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


# The reply setup of the released code (the paper): a 60-token ceiling, a
# newline stop and "<int> | <reason>". --original-rules uses it; the tests
# override it to replay earlier runs.
RELEASED = {"uncapped": False, "max_tokens": None, "no_think": False,
            "structured": False, "schema_order": None, "reason_max_chars": None}


def llm_setup(player, original_rules, think, effort, released, requests_per_minute=None):
    """make_llm arguments for an LLM player, and what run.json records about it."""
    provider, model, host = player["provider"], player["model"], player["host"]
    if provider in API_PROVIDERS:
        if original_rules or think:
            raise ValueError(f"{player['model_spec']}: --original-rules and --think are for "
                             f"Ollama models; an API model takes --effort")
        info = api_model_info(provider, model)
        if info["structured_outputs"] is False:
            raise ValueError(f"{provider} lists no structured-output support for {model}")
        extra, thinks = api_reasoning(provider, info, effort)
        temperature = None if info["temperature"] is False else 0
        rpm = requests_per_minute or PROVIDER_DEFAULTS[provider]["requests_per_minute"]
        ceiling = API_CEILING
        # "reasoning" holds the reasoning fields sent with every request.
        record = {"reasoning": extra, "effort": effort, "thinks": thinks, "ceiling": ceiling,
                  "temperature": temperature, "requests_per_minute": rpm}
        return dict(max_tokens=ceiling, stop=None, structured=True,
                    schema_order=DEFAULT_SCHEMA_ORDER, reason_max_chars=REASON_MAX_CHARS,
                    extra_body=extra, temperature=temperature, requests_per_minute=rpm,
                    retries=API_RETRIES), record
    info = model_info(host.removesuffix("/v1"), model) if "ollama" in provider else {}
    record = {k: info.get(k) for k in ("digest", "quantization", "parameter_size")}
    if original_rules:
        kw = {}
        if released["uncapped"]:
            kw.update(max_tokens=released["max_tokens"], stop=None)
        if released["no_think"]:
            kw.update(think=False)
        if released["structured"]:
            kw.update(structured=True, schema_order=released["schema_order"],
                      reason_max_chars=released["reason_max_chars"])
        return kw, record
    think_value, thinks = thinking_setting(model, info, think, effort)
    ceiling = THINKING_CEILING if thinks else ANSWER_CEILING
    num_ctx = THINKING_CONTEXT if thinks else ANSWER_CONTEXT
    record.update(think=think_value, thinks=thinks, ceiling=ceiling, num_ctx=num_ctx)
    return dict(max_tokens=ceiling, num_ctx=num_ctx, stop=None, think=think_value,
                structured=True, schema_order=DEFAULT_SCHEMA_ORDER,
                reason_max_chars=REASON_MAX_CHARS), record


def build_agent(spec, llm_kwargs, labels, include_legal_moves, robust, history,
                move_outcomes, legal_order):
    """spec is 'human', 'random', 'heuristic', 'first', 'gt[:depth]', 'dummy',
    or an LLM (llm_kwargs are its make_llm arguments)."""
    if spec == "human":
        return HumanAgent(labels), "human"

    if spec in BASELINES:
        return BASELINES[spec](), spec

    if spec == "gt" or spec.startswith("gt:"):
        from game_theory_multiplayer_agent import GameTheoryMultiplayerAgent
        depth = int(spec.split(":", 1)[1]) if ":" in spec else 2
        return GameTheoryMultiplayerAgent(search_depth=depth), f"gt(d={depth})"

    return TrackedLLMAgent(make_llm(spec, **llm_kwargs),
                           include_legal_moves_in_prompt=include_legal_moves,
                           robust_parse=robust,
                           structured=llm_kwargs.get("structured", False),
                           schema_order=llm_kwargs.get("schema_order") or DEFAULT_SCHEMA_ORDER,
                           reason_max_chars=llm_kwargs.get("reason_max_chars"),
                           last_move=history, move_outcomes=move_outcomes,
                           legal_order=legal_order), spec


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


def folder_name(spec):
    """A player's name in results/: the model of an LLM (after its API, if any), else the spec."""
    name = spec
    if is_llm_spec(spec):
        provider, name, _ = parse_spec(spec)
        if provider in API_PROVIDERS:
            name = f"{provider}-{name}"
    return name.replace(":", "-").replace("/", "-")


def run_folder(specs, effort=None):
    """
    A run's folder in its experiment: every player in seat order, and the effort
    if set, joined by "_". A player name with "_" in it is a ValueError, so a
    folder name always reads back one way.
    """
    names = [folder_name(s) for s in specs]
    bad = [n for n in names if "_" in n]
    if bad:
        raise ValueError(f"player names may not contain '_': {', '.join(bad)}")
    return "_".join(names + ([f"effort-{effort}"] if effort else []))


RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CELL_FIELDS = ["experiment", "run", "model", "opponent", "settings", "games_finished", "wins",
               "win_rate", "invalid_rate", "ollama_version", "cost_usd", "commit"]


def cell_results(records, pid, ollama_versions):
    """
    The results columns of a run's row in cells.csv for player pid (the run's
    model): wins, win rate and invalid-reply rate over its finished games, and
    the Ollama versions that served it. Empty where they do not apply. The
    run's total cost comes from run.json.
    """
    n = len(records)
    wins = sum(r["winner"] == pid for r in records)
    llm = [r["per_player"][str(pid)]["llm"] for r in records]
    calls = sum(x["calls"] for x in llm if x)
    return {"games_finished": n, "wins": wins,
            "win_rate": f"{wins / n:.3f}" if n else "",
            "invalid_rate": f"{sum(x['invalid'] for x in llm if x) / calls:.4f}" if calls else "",
            "ollama_version": " ".join(sorted(set(ollama_versions.values())))}


@contextlib.contextmanager
def locked(path, stale_s=60):
    """
    Hold the lock file `path` while the block runs, waiting while another
    process holds it. A lock older than stale_s seconds was left by a process
    that died holding it, and is taken over.
    """
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(path) > stale_s:
                    os.remove(path)
            except FileNotFoundError:
                pass
            time.sleep(0.05)
    try:
        yield
    finally:
        os.close(fd)
        os.remove(path)


def update_cells(path, row):
    """
    Write `row` into the run table at `path`, replacing that run's earlier row.
    Safe for several queues at once: the update holds a lock, and the new table
    replaces the old one whole, so it is never read half written.
    """
    with locked(path + ".lock"):
        rows = []
        if os.path.exists(path):
            with open(path, newline="", encoding="utf-8") as f:
                rows = [r for r in csv.DictReader(f)
                        if (r["experiment"], r.get("run")) != (row["experiment"], row["run"])]
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CELL_FIELDS)
            writer.writeheader()
            writer.writerows(rows + [row])
        os.replace(tmp, path)


def resume(out_dir, settings, players):
    """
    (run.json, finished game records) of an earlier run stopped in out_dir, or
    (None, []). Its finished games are kept; the plies of a game it did not
    finish are dropped, so that game restarts from its seed. A folder holding a
    run with other settings or players is refused. So is one made by other
    code, unless it has no finished game to keep (it starts afresh) or none left
    to play (it is left as it is): one run's games all come from one commit.
    """
    path = os.path.join(out_dir, "run.json")
    if not os.path.exists(path):
        return None, []
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    if ({**meta["settings"], "games": 0} != json.loads(json.dumps({**settings, "games": 0}))
            or meta["players"] != json.loads(json.dumps(players))):
        raise SystemExit(f"{out_dir} holds a run with other settings or players; "
                         f"choose another --experiment")
    games_path = os.path.join(out_dir, "games.jsonl")
    decisions_path = os.path.join(out_dir, "decisions.jsonl")
    with open(games_path, encoding="utf-8") as f:     # a line cut off by a stop has no newline
        games = [g for g in (json.loads(line) for line in f if line.endswith("\n"))
                 if g["termination_reason"] != "error"]
    if meta["git"]["commit"] != git_state()["commit"]:
        if not games:
            return None, []
        if len(games) < settings["games"]:
            raise SystemExit(f"{out_dir} holds an unfinished run made by other code "
                             f"({meta['git']['commit'][:12]}); choose another --experiment")
        return meta, games
    finished = {g["game_id"] for g in games}
    with open(decisions_path, encoding="utf-8") as f:
        plies = [line for line in f if line.endswith("\n")
                 and re.search(r'"game_id": "([^"]+)"', line).group(1) in finished]
    with open(games_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(g) + "\n" for g in games)
    with open(decisions_path, "w", encoding="utf-8") as f:
        f.writelines(plies)
    meta["settings"] = settings
    meta["games"] = {"attempted": len(games), "completed": len(games), "errored": 0}
    return meta, games


def run_games(specs, games, seed0, out_dir, include_legal_moves=True, legal_order="ascending",
              move_outcomes=False, history=0, think=False, effort=None, original_rules=False,
              released=None, max_steps=1000, watch=False, delay=0.0, ollama_versions=None,
              cell=None, requests_per_minute=None):
    """
    Play one run into out_dir. With original_rules the reply setup is RELEASED,
    updated by `released`. `cell` (experiment, model, opponent, settings) adds
    the run's row to results/cells.csv after every game.
    """
    released = {**RELEASED, **(released or {})}
    seats = rules.seats(len(specs), original_rules)
    players = player_info(specs, seats)
    llm_kwargs = {}
    for pid, player in players.items():
        if player["provider"]:
            llm_kwargs[pid], record = llm_setup(player, original_rules, think, effort, released,
                                                requests_per_minute)
            player.update(record)
        elif player["model_spec"] == "dummy":
            llm_kwargs[pid] = {"structured": released["structured"] or not original_rules}
    if original_rules:
        reply = dict(released, max_tokens=released["max_tokens"] if released["uncapped"] else 60,
                     stop=None if released["uncapped"] else ["\n"])
        if not reply["structured"]:
            reply["schema_order"] = None
    else:
        reply = {"uncapped": True, "max_tokens": None, "stop": None, "no_think": not think,
                 "structured": True, "schema_order": DEFAULT_SCHEMA_ORDER,
                 "reason_max_chars": REASON_MAX_CHARS}
    # Kept in run.json so a results folder is self-describing. Each LLM player's
    # model version, thinking setting, output ceiling and context are under "players".
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
        "structured_output": reply["structured"],
        "schema_order": reply["schema_order"],
        "reason_max_chars": reply["reason_max_chars"],
        "last_move_context": bool(history),
        "last_moves": history,
        "move_outcomes": move_outcomes,
        "legal_order": legal_order,
        # Resolved server per LLM player, e.g. a partner's laptop on the LAN.
        "llm_endpoints": {str(pid): p["host"] for pid, p in players.items()
                          if p["host"]},
        "include_legal_moves": include_legal_moves,
        "uncapped": reply["uncapped"],
        "max_tokens": reply["max_tokens"],
        "stop": reply["stop"],
        "no_think": reply["no_think"],
        "think": think,
        "effort": effort,
        "ceilings": None if original_rules else {"answer": ANSWER_CEILING,
                                                 "thinking": THINKING_CEILING},
        "contexts": None if original_rules else {"answer": ANSWER_CONTEXT,
                                                 "thinking": THINKING_CONTEXT},
        "max_steps": max_steps,
        "temperature": 0,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "request_timeout_s": DEFAULT_TIMEOUT,
        "timeout_retries": TIMEOUT_RETRIES,
    }
    # Copied onto every decision record. num_ctx is filled in per player.
    config = {
        "original_rules": original_rules,
        "no_think": reply["no_think"],
        "uncapped": reply["uncapped"],
        "max_tokens": reply["max_tokens"],
        "stop": reply["stop"],
        "structured": reply["structured"],
        "schema_order": reply["schema_order"],
        "reason_max_chars": reply["reason_max_chars"],
        "legal_moves_shown": include_legal_moves,
        "last_move_context": bool(history),
        "last_moves": history,
        "move_outcomes": move_outcomes,
        "legal_order": legal_order,
        "think": think,
        "effort": effort,
        "temperature": settings["temperature"],
        "system_prompt_hash": short_hash(DEFAULT_SYSTEM_PROMPT),
        "num_ctx": None,
        "context_mode": CONTEXT_MODE,
    }

    meta, records = resume(out_dir, settings, players)
    if records:
        print(f"  {len(records)} of {games} games already finished"
              + ("" if len(records) >= games else "; resuming"))
    run = None
    for g in range(len(records), games):
        seed = seed0 + g
        env = LudoEnv(seed=seed, num_players=len(specs), original_rules=original_rules)
        if run is None:                             # needs an env for its constants
            run = RunLog(out_dir, os.path.basename(os.path.normpath(out_dir)), players,
                         settings, config, ollama_versions or {}, env, resumed=meta)

        labels = {pid: spec for pid, spec in zip(env.player_ids, specs)}
        human_playing = "human" in specs
        agents = {}
        for pid, spec in zip(env.player_ids, specs):
            agent, label = build_agent(spec, llm_kwargs.get(pid, {}), labels,
                                       include_legal_moves, released["uncapped"] and original_rules,
                                       history, move_outcomes, legal_order)
            if watch and spec != "human":
                agent = WatchAgent(agent, label)
            agents[pid] = agent
            labels[pid] = label

        if history:
            attach_last_move(env, keep=history)

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
        if failure is not None:
            # The plies and the partial game are on disk. A failure such as an
            # unreachable server would hit every later game too, so stop here.
            run.close()
            raise failure
        records.append(rec)
        if cell:
            git = run.meta["git"]
            pid = min((p for p, v in players.items()
                       if folder_name(v["model_spec"]) == cell["model"]),
                      key=lambda p: players[p]["seat_position"])
            update_cells(os.path.join(os.path.dirname(os.path.dirname(out_dir)), "cells.csv"),
                         dict(cell, **cell_results(records, pid, run.meta["ollama_versions"]),
                              cost_usd=f"{run.meta['cost_usd']:.4f}",
                              commit=(git["commit"] or "")[:12] + ("-dirty" if git["dirty"] else "")))

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
    ap = argparse.ArgumentParser(
        description="Full Ludo games: LLMs, bots, or you. With no switches a run "
                    "uses the standard setup (see the top of this file).")
    ap.add_argument("--players", nargs="+", required=True, metavar="PLAYER",
                    help="2-4 players in seat order: llm:MODEL, llm://HOST[:PORT]/MODEL, "
                         "random, heuristic, first, gt[:DEPTH], human, dummy")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=0, help="first seed; games use seed0..seed0+n-1")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--experiment", default="sandbox",
                    help="results go to results/<experiment>/<players>/ (default: sandbox, "
                         "for trial games)")
    ap.add_argument("--hide-legal-moves", action="store_true",
                    help="leave the legal-move list out of the prompt")
    ap.add_argument("--shuffle", action="store_true",
                    help="show the legal moves in a random order (the game's legal_order stream)")
    ap.add_argument("--outcomes", action="store_true",
                    help="after the legal-move list, say what each legal move does")
    ap.add_argument("--history", type=int, default=0, metavar="N",
                    help="show the last N moves in the prompt")
    ap.add_argument("--think", action="store_true",
                    help="let a thinking model think (by default thinking is off for "
                         "models that can switch it off)")
    ap.add_argument("--effort", choices=["low", "medium", "high"],
                    help="reasoning effort: Ollama's think level, NVIDIA's reasoning_effort "
                         "or OpenRouter's reasoning.effort, for models that take it")
    ap.add_argument("--requests-per-minute", type=float, metavar="N",
                    help="pace requests to each API to N per minute (default: NVIDIA 40, "
                         "OpenRouter unpaced)")
    ap.add_argument("--original-rules", action="store_true",
                    help="the released LudoBench rules, prompt and reply format (60 tokens, "
                         "'<int> | <reason>'), to reproduce the original paper")
    ap.add_argument("--watch", action="store_true",
                    help="draw the board after every move (spectator mode)")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="seconds to pause after each move when watching")
    args = ap.parse_args()

    specs = args.players
    if len(specs) not in (2, 3, 4):
        ap.error("need 2-4 players")
    try:
        folder = run_folder(specs, args.effort)
    except ValueError as e:
        ap.error(str(e))
    if args.outcomes and args.hide_legal_moves:
        ap.error("--outcomes needs the legal-move list in the prompt")
    if args.history < 0:
        ap.error("--history must be 0 or more")
    if args.original_rules and (args.think or args.effort):
        ap.error("--think and --effort do not apply with --original-rules")

    # Windows writes redirected or piped stdout as cp1252, and printing a model
    # reply containing e.g. U+2011 or an arrow then crashes the run.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    seat_specs = dict(zip(rules.seats(len(specs), args.original_rules), specs))
    try:
        resolved = {pid: parse_spec(s) for pid, s in seat_specs.items() if is_llm_spec(s)}
    except ValueError as e:
        ap.error(str(e))
    other = [seat_specs[pid] for pid, r in resolved.items()
             if r[0] not in ("ollama_native", *API_PROVIDERS)]
    if other:
        ap.error(f"LLM players are llm:MODEL (Ollama), llm:openrouter/MODEL or "
                 f"llm:nvidia/MODEL; not: {', '.join(other)}")
    for pid, (provider, model, base_url) in resolved.items():
        if provider in API_PROVIDERS and not os.environ.get(PROVIDER_DEFAULTS[provider]["key_env"]):
            ap.error(f"{seat_specs[pid]} needs {PROVIDER_DEFAULTS[provider]['key_env']} set")

    # Check every Ollama server answers before any game starts, so a partner's
    # laptop that is off or firewalled fails now with its host and port named.
    ollama_versions = {}
    for pid, (provider, model, base_url) in resolved.items():
        if provider in API_PROVIDERS:
            print(f"P{pid} {model} -> {base_url}")
            continue
        try:
            ollama_versions[base_url] = check_ollama(base_url)
        except RuntimeError as e:
            raise SystemExit(f"P{pid} {seat_specs[pid]}: {e}")
        print(f"P{pid} {model} -> {base_url}  (Ollama {ollama_versions[base_url]})")

    llm = [s for s in specs if is_llm_spec(s)]
    model = folder_name((llm or specs)[0])
    opponent = "+".join(folder_name(s) for s in specs if s != (llm or specs)[0])
    switches = [name for name, on in (("hide-legal-moves", args.hide_legal_moves),
                                      ("shuffle", args.shuffle), ("outcomes", args.outcomes),
                                      (f"history {args.history}", args.history),
                                      ("think", args.think), (f"effort {args.effort}", args.effort),
                                      ("original-rules", args.original_rules)) if on]
    out = os.path.join(RESULTS, args.experiment, folder)
    if git_state()["dirty"]:
        print("WARNING: tracked files have uncommitted changes (recorded in run.json)")
    print(f"\nRunning {args.games} game(s): " + "  vs  ".join(specs))
    print(f"settings: {', '.join(switches) or 'standard'}")
    print(f"writing to {out}\n")

    try:
        records = run_games(
            specs, games=args.games, seed0=args.seed0, out_dir=out,
            include_legal_moves=not args.hide_legal_moves,
            legal_order="shuffled" if args.shuffle else "ascending",
            move_outcomes=args.outcomes, history=args.history, think=args.think,
            effort=args.effort, original_rules=args.original_rules,
            max_steps=args.max_steps, watch=args.watch, delay=args.delay,
            ollama_versions=ollama_versions, requests_per_minute=args.requests_per_minute,
            cell={"experiment": args.experiment, "run": os.path.basename(out),
                  "model": model, "opponent": opponent,
                  "settings": " ".join(f"--{x}" for x in switches) or "standard"})
    except ValueError as e:                          # e.g. a thinking switch the model cannot take
        raise SystemExit(str(e))
    summarise(records, specs)
    print(f"\nLogs: {out}  (decisions.jsonl, games.jsonl, run.json)")


if __name__ == "__main__":
    main()