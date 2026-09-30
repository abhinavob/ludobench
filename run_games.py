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

Offline smoke test (no API, no Ollama):
    python run_games.py --p0 dummy --p1 dummy --games 2

Withhold legal moves from the prompt (spot-mode behaviour):
    python run_games.py --p0 ollama:llama3.2 --p1 heuristic --no-legal-moves

Let a reasoning model actually reason (removes the 60-token cap and newline stop):
    python run_games.py --p0 ollama:deepseek-r1 --p1 heuristic --uncapped

Results go to a JSON file (--out), one record per game. The file is rewritten
after every game, so a run that dies partway keeps the games already played.
The default filename is timestamped, and an existing --out file is never
overwritten.
"""

import argparse
import json
import os
import time

import sys

from ludo_env import LudoEnv
from agents import BaseAgent, RandomAgent, HeuristicAgent
from llm_agent import LLMAgent
from evaluation import evaluate
from llm_backends import make_llm, DEFAULT_SYSTEM_PROMPT, DEFAULT_TIMEOUT
from board_render import render_board, describe_move, group_moves

BASELINES = {"random": RandomAgent, "heuristic": HeuristicAgent}


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


class TrackedLLMAgent(LLMAgent):
    """
    LLMAgent that also counts, for its own player, how each reply ended.

    env.llm_fallbacks is one counter shared by every LLM in the game, so it
    cannot give a per-model invalid rate; this can. The finish reason separates
    "ran out of tokens" ("length") and "timeout" from a complete reply that was
    still unusable ("stop").
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0
        self.invalid = 0
        self.finish_reasons = {}
        self.invalid_by_finish_reason = {}

    def select_move(self, env, player, dice, legal_moves):
        calls_before = env.llm_calls
        choice = super().select_move(env, player, dice, legal_moves)
        if env.llm_calls == calls_before:          # no model call was made
            return choice
        why = getattr(self.llm_call, "last_finish_reason", None) or "unknown"
        self.calls += 1
        _bump(self.finish_reasons, why)
        if env.last_llm_fallback:
            self.invalid += 1
            _bump(self.invalid_by_finish_reason, why)
        return choice

    def stats(self):
        return {"calls": self.calls, "invalid": self.invalid,
                "finish_reasons": self.finish_reasons,
                "invalid_by_finish_reason": self.invalid_by_finish_reason}


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


def build_agent(spec, include_legal_moves, uncapped, labels=None, no_think=False,
                max_tokens=None):
    """spec is 'human', 'random', 'heuristic', 'gt[:depth]', 'dummy',
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
    fn = make_llm(spec, **kwargs)
    return TrackedLLMAgent(fn, include_legal_moves_in_prompt=include_legal_moves,
                           robust_parse=robust), spec


def run_games(specs, games, seed0, include_legal_moves, uncapped, out_path,
              max_steps=1000, watch=False, delay=0.0, no_think=False,
              max_tokens=None):
    # Stored in every record so a results file is self-describing. The capped
    # values are make_llm's defaults, which reproduce the released code.
    # max_tokens is null when uncapped without --max-tokens.
    settings = {
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
    records = []
    for g in range(games):
        seed = seed0 + g
        env = LudoEnv(seed=seed, num_players=len(specs))

        labels = {pid: spec for pid, spec in zip(env.player_ids, specs)}
        human_playing = "human" in specs
        agents = {}
        tracked = {}                                # pid -> TrackedLLMAgent
        for pid, spec in zip(env.player_ids, specs):
            agent, label = build_agent(spec, include_legal_moves, uncapped, labels,
                                       no_think=no_think, max_tokens=max_tokens)
            if isinstance(agent, TrackedLLMAgent):
                tracked[pid] = agent
            if watch and spec != "human":
                agent = WatchAgent(agent, label)
            agents[pid] = agent
            labels[pid] = label

        if watch or human_playing:
            print(f"\n{'=' * 58}\n  GAME {g + 1}/{games}  (seed {seed})\n{'=' * 58}")
            if watch:
                render_board(env, labels=labels)
                attach_watcher(env, labels, delay=delay)

        t0 = time.time()
        winner, steps = env.play_game(agents, max_steps=max_steps)
        elapsed = time.time() - t0

        stats = evaluate(env)
        rec = {
            "seed": seed,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "players": {str(pid): labels[pid] for pid in env.player_ids},
            "winner": winner,
            "winner_model": labels.get(winner) if winner is not None else None,
            "hit_max_steps": winner is None,
            # A dice roll with no legal move still uses a step, so
            # dice_rolls >= moves_applied.
            "dice_rolls": steps,
            "moves_applied": stats.get("total_moves", 0),
            "seconds": round(elapsed, 1),
            "llm_calls": stats.get("llm_calls", 0),
            "llm_fallbacks": stats.get("llm_fallbacks", 0),
            # Per LLM player: calls, invalid replies, and why each reply ended.
            "llm_players": {str(pid): {"model": labels[pid], **a.stats()}
                            for pid, a in tracked.items()},
            "captures": stats.get("captures", 0),
            "safe_moves": stats.get("safe_moves", 0),
            "settings": settings,
        }
        records.append(rec)

        fb = rec["llm_fallbacks"]
        calls = rec["llm_calls"]
        rate = f"{fb}/{calls}" if calls else "n/a"
        print(f"  game {g + 1}/{games} (seed {seed}): "
              f"winner P{winner} = {rec['winner_model']}, "
              f"{steps} dice rolls, {elapsed:.0f}s, invalid {rate}")

        # write after every game so a crash does not lose the run
        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)

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
    print(f"  mean seconds/game {sum(r['seconds'] for r in records) / n:.0f}")
    print(f"  mean captures     {sum(r['captures'] for r in records) / n:.1f}")
    if calls:
        print(f"  LLM calls         {calls}")
        print(f"  invalid/fallback  {fallbacks}/{calls}  ({fallbacks / calls:.1%})")

    per_player = {}
    for r in records:
        for pid, s in r["llm_players"].items():
            agg = per_player.setdefault((pid, s["model"]),
                                        {"calls": 0, "invalid": 0, "why": {}})
            agg["calls"] += s["calls"]
            agg["invalid"] += s["invalid"]
            for why, k in s["invalid_by_finish_reason"].items():
                agg["why"][why] = agg["why"].get(why, 0) + k
    for (pid, model), agg in sorted(per_player.items()):
        if not agg["calls"]:
            continue
        why = ", ".join(f"{k} {v}" for k, v in sorted(agg["why"].items()))
        print(f"  P{pid} {model}: invalid {agg['invalid']}/{agg['calls']} "
              f"({agg['invalid'] / agg['calls']:.1%})"
              + (f"  by finish reason: {why}" if why else ""))


def main():
    ap = argparse.ArgumentParser(description="Full Ludo games: LLMs, baselines, or you.")
    ap.add_argument("--p0", default="dummy",
                    help="agent for player 0: human | random | heuristic | gt[:depth] "
                         "| dummy | provider:model")
    ap.add_argument("--p1", default="heuristic", help="agent for player 1")
    ap.add_argument("--p2", default=None, help="agent for player 2 (optional)")
    ap.add_argument("--p3", default=None, help="agent for player 3 (optional)")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=0, help="first seed; games use seed0..seed0+n-1")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--no-legal-moves", action="store_true",
                    help="withhold the legal-move list from the prompt (spot-mode behaviour)")
    ap.add_argument("--uncapped", action="store_true",
                    help="remove max_tokens=60 and the newline stop entirely, for "
                         "reasoning models (still bounded by the request timeout)")
    ap.add_argument("--no-think", action="store_true",
                    help="disable a thinking model's reasoning phase; only works with "
                         "ollama_native:<model> (Ollama's /v1 endpoint ignores it)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="optional token limit, only with --uncapped (default: none)")
    ap.add_argument("--out", default=None,
                    help="JSON file for per-game records (must not already exist); "
                         "default is a timestamped file in llm_game_results/")
    ap.add_argument("--watch", action="store_true",
                    help="draw the board after every move (spectator mode)")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="seconds to pause after each move when watching")
    args = ap.parse_args()
    if args.max_tokens is not None and not args.uncapped:
        # Paper-exact mode is fixed at 60 tokens + newline stop; changing only
        # the token limit would give a setup that is neither paper nor uncapped.
        ap.error("--max-tokens requires --uncapped")

    # Windows writes redirected or piped stdout as cp1252, and printing a model
    # reply containing e.g. U+2011 or an arrow then crashes the run.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    specs = [s for s in (args.p0, args.p1, args.p2, args.p3) if s]
    if len(specs) not in (2, 3, 4):
        raise SystemExit("need 2-4 players")

    if args.no_think:
        ignored = [s for s in specs if s.startswith(("ollama:", "openrouter:"))]
        if ignored:
            print(f"WARNING: --no-think has no effect on {', '.join(ignored)}; "
                  f"use ollama_native:<model> to switch thinking off.")

    out = args.out
    if out is None:
        tag = "_vs_".join(s.replace("/", "-").replace(":", "-") for s in specs)
        out = os.path.join("llm_game_results",
                           f"{tag}_{time.strftime('%Y%m%d-%H%M%S')}.json")
    elif os.path.exists(out):
        raise SystemExit(f"{out} already exists; pick another --out "
                         f"(refusing to overwrite earlier results)")
    # Create the folder now, not after game 1, so a bad path fails before any play.
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)

    print(f"\nRunning {args.games} game(s): " + "  vs  ".join(specs))
    if not args.uncapped:
        cap = "60 tokens + newline stop (paper)"
    elif args.max_tokens is None:
        cap = "none"
    else:
        cap = f"{args.max_tokens} tokens, no newline stop"
    print(f"legal moves in prompt: {not args.no_legal_moves}   output cap: {cap}")
    print(f"writing to {out}\n")

    records = run_games(
        specs,
        games=args.games,
        seed0=args.seed0,
        include_legal_moves=not args.no_legal_moves,
        uncapped=args.uncapped,
        out_path=out,
        max_steps=args.max_steps,
        watch=args.watch,
        delay=args.delay,
        no_think=args.no_think,
        max_tokens=args.max_tokens,
    )
    summarise(records, specs)
    print(f"\nPer-game records: {out}")


if __name__ == "__main__":
    main()