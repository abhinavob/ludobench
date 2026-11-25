"""
Ask an LLM for its move in each spot and log every decision as games do.

    python run_spots.py --player llm:qwen3:8b --experiment spots

Each spot of the folder (default spots_40_final/, the reviewed set; the
original spots_40/ and the automatic repair spots_40_repaired/, which
repair_spots.py writes, can be asked with --spots) is asked twice, with the
legal-move list hidden and shown (--legal), with the standard prompt and reply
settings of run_games.py. A grudge spot carries its
history paragraph (neutral or grudge); --persona adds one of the persona
instructions of run_all_spots.py, off by default. --shuffle and --outcomes
change the shown list as in run_games.py; the shuffle draws on the spot's own
stream, random_streams('<spot id>:<condition>'), and outcomes go only on the
prompts that show the list. --per-category N asks only N spots of each category
(see subset), e.g. for a slow thinking model. --changed-from FOLDER asks only
the spots that differ from the same spot in FOLDER, e.g. the spots of
spots_40_final/ that differ from spots_40_repaired/, under a new --experiment.
--spot-ids FILE asks only the spots listed in FILE, one id per line, e.g. to
ask unchanged spots again and compare with an earlier run. --ceiling N lowers
the output-token ceiling (thinking included) below the model's own, e.g. where
an API reserves credit for a whole ceiling per request; the reasoning fields
sent stay as they are.

Results go to results/<experiment>/<player>[_persona-NAME]/: run.json and
decisions.jsonl, one record per spot and condition with the fields of a game's
decisions.jsonl (see decision_log.py) and the spot's own (SPOT_DEFINITIONS).
A stopped run resumes where it stopped, under the rules of run_games.py.
"""
import argparse
import ast
import glob
import json
import os
import sys
from itertools import chain, zip_longest

import ludo_rules as rules
from agents import HeuristicAgent
from board_render import group_moves
from decision_log import (RunLog, DEFINITIONS, NO_MODEL, analyse, move_fields, decision_record,
                          snapshot, git_state, choice_rates)
from game_theory_multiplayer_agent import GameTheoryMultiplayerAgent
from ludo_env import random_streams
from run_games import (RELEASED, RESULTS, build_agent, check_llm_players, decision_config,
                       earlier_run, is_llm_spec, llm_setup, move_outcome_text, player_info,
                       reply_settings, reply_setup, run_folder)
from spot_evaluation import SpotEnv

HERE = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_SPOTS = os.path.join(HERE, "spots_40")
DEFAULT_SPOTS = os.path.join(HERE, "spots_40_final")
CONDITIONS = {"hidden": False, "shown": True}          # name -> legal-move list in the prompt
SEARCH_DEPTH = 2                                        # as run_all_spots_gt.py

SPOT_DEFINITIONS = {
    "spot_id": "the spot's id in its file; game_id and seed are null for a spot",
    "spots_folder": "the folder the spot was read from",
    "spot_changed": "the spot's tokens differ from spots_40/ (a repaired spot)",
    "pair_id": "grudge spots: the id shared by the two halves of a pair; else null",
    "history": "grudge spots: 'neutral' or 'grudge', the history paragraph the prompt "
               "carries (as written in the spot); else null",
    "persona": "the persona instruction's name, null if none",
    "search_move": f"GameTheoryMultiplayerAgent(search_depth={SEARCH_DEPTH})'s choice",
    "move_groups": "the legal tokens grouped by identical outcome (board_render.group_moves)",
}


def persona_texts(path=os.path.join(HERE, "run_all_spots.py")):
    """PERSONA_TEXTS of run_all_spots.py, read without importing it (it builds an API client)."""
    with open(path, encoding="utf-8") as f:
        for node in ast.parse(f.read()).body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PERSONA_TEXTS"
                                                    for t in node.targets):
                return ast.literal_eval(node.value)
    raise ValueError(f"no PERSONA_TEXTS in {path}")


def load_spots(folder):
    """The spots of a folder in file order."""
    return [s for f in sorted(glob.glob(os.path.join(folder, "spots_*.json")))
            for s in json.load(open(f, encoding="utf-8"))]


def board(spot):
    return {int(p): list(t) for p, t in spot["tokens"].items()}


def pair_id(spot):
    return spot["id"].rsplit("_", 1)[0] if spot["scenario"] == "grudge" else None


def subset(spots, n):
    """
    The first n units of each category, a unit being a spot or a grudge pair:
    taken from the 2-, 3- and 4-player units in turn, each in file order, so
    the player counts stay even (10 gives 4/3/3) and a smaller n is part of a
    larger one. The spots keep their file order.
    """
    units = {}                                         # category -> unit key -> its spots
    for s in spots:
        units.setdefault(s["scenario"], {}).setdefault(pair_id(s) or s["id"], []).append(s)
    keep = set()
    for category in units.values():
        by_players = {}
        for unit in category.values():
            by_players.setdefault(len(unit[0]["players"]), []).append(unit)
        turns = chain.from_iterable(zip_longest(*(by_players[k] for k in sorted(by_players))))
        keep.update(s["id"] for unit in [u for u in turns if u][:n] for s in unit)
    return [s for s in spots if s["id"] in keep]


def changed(spots, folder):
    """The spots that differ from the spot of the same id in folder."""
    ref = {s["id"]: s for s in load_spots(folder)}
    return [s for s in spots if s != ref.get(s["id"])]


def read_ids(path):
    """The spot ids listed in a file, one per line; blank lines and # comments are skipped."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]


def derived(spot):
    """
    What a run derives from the spot's position: the legal moves, the outcome
    line --outcomes shows for each, the search player's move, the moves grouped
    by identical outcome, and the shown list in ascending and --shuffle order.
    """
    p, d = spot["current_player"], spot["dice"]
    env = SpotEnv(spot["players"], board(spot))
    legal = rules.legal_moves(env.tokens, p, d)
    shuffled = list(legal)
    random_streams(f"{spot['id']}:shown")["legal_order"].shuffle(shuffled)
    return {"legal_moves": legal, "outcome_lines": [move_outcome_text(env, p, d, t) for t in legal],
            "search_move": GameTheoryMultiplayerAgent(search_depth=SEARCH_DEPTH).select_move(env, p, d, legal),
            "move_groups": group_moves(env, p, d, legal),
            "shown": legal, "shown_shuffled": shuffled}


def spot_fields(spot, folder, original, persona):
    grudge = spot["scenario"] == "grudge"
    return {"spot_id": spot["id"], "spots_folder": os.path.basename(os.path.normpath(folder)),
            "spot_changed": spot["tokens"] != original.get(spot["id"], spot)["tokens"],
            "pair_id": pair_id(spot),
            "history": ("neutral" if spot["id"].endswith("_a") else "grudge") if grudge else None,
            "persona": persona}


def ask(agent, spot, shown, persona_text, search):
    """(pre, chosen, env after the move, model fields) for one spot and condition."""
    p, d = spot["current_player"], spot["dice"]
    env = SpotEnv(spot["players"], board(spot))
    env.rng = random_streams(f"{spot['id']}:{'shown' if shown else 'hidden'}")   # fallback, --shuffle
    env.history_text = spot.get("history_text")
    env.persona_text = persona_text
    pre = analyse(env, p, d, HeuristicAgent())
    pre["search_move"] = search.select_move(env, p, d, pre["legal"])
    pre["move_groups"] = group_moves(env, p, d, pre["legal"])
    agent.include_legal_moves_in_prompt = shown
    chosen = agent.select_move(env, p, d, pre["legal"])
    model = agent.last_decision or NO_MODEL
    after, _ = rules.apply_move(env.tokens, p, chosen, d)
    return pre, chosen, SpotEnv(spot["players"], after), model


def resume_spots(out_dir, settings, players, todo):
    """
    (run.json, the (spot_id, legal shown) pairs done) of an earlier run in
    out_dir, or (None, set()). As with games: a run with other settings or
    players is refused, and so is an unfinished one made by other code unless
    it has nothing done yet (it starts afresh).
    """
    meta = earlier_run(out_dir, settings, players)
    if meta is None:
        return None, set()
    path = os.path.join(out_dir, "decisions.jsonl")
    with open(path, encoding="utf-8") as f:            # a line cut off by a stop has no newline
        lines = [line for line in f if line.endswith("\n")]
    done = {(r["spot_id"], r["config"]["legal_moves_shown"]) for r in map(json.loads, lines)}
    if meta["git"]["commit"] != git_state()["commit"]:
        if not done:
            return None, set()
        if not todo <= done:
            raise SystemExit(f"{out_dir} holds an unfinished run made by other code "
                             f"({meta['git']['commit'][:12]}); choose another --experiment")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return meta, done


def run_spots(spec, out_dir, folder=DEFAULT_SPOTS, legal=("hidden", "shown"), persona=None,
              legal_order="ascending", move_outcomes=False, per_category=None, think=False,
              effort=None, requests_per_minute=None, ollama_versions=None, changed_from=None,
              spot_ids=None, ceiling=None):
    """
    Ask the spots of `folder` (all, or `per_category`, and of those only the
    ones that differ in `changed_from` if given, and only those listed in the
    file `spot_ids` if given) in each `legal` condition; return the records.
    `ceiling` lowers the player's output-token ceiling.
    """
    spots = load_spots(folder)
    if per_category:
        spots = subset(spots, per_category)
    if changed_from:
        spots = changed(spots, changed_from)
        if not spots:
            raise ValueError(f"no spot of {folder} differs from {changed_from}")
    if spot_ids:
        wanted = read_ids(spot_ids)
        unknown = set(wanted) - {s["id"] for s in spots}
        if unknown or not wanted:
            raise ValueError(f"{spot_ids}: {', '.join(sorted(unknown)) or 'no ids'}"
                             f"{' not among the spots asked' if unknown else ''}")
        spots = [s for s in spots if s["id"] in set(wanted)]
    original = {s["id"]: s for s in load_spots(ORIGINAL_SPOTS)}
    persona_text = persona_texts()[persona] if persona else None
    players = player_info([spec], [0])
    llm_kwargs, record = llm_setup(players[0], False, think, effort, RELEASED, requests_per_minute)
    if ceiling is not None:
        if not 0 < ceiling <= record["ceiling"]:
            raise ValueError(f"--ceiling must be from 1 to the model's ceiling, {record['ceiling']}")
        llm_kwargs["max_tokens"] = record["ceiling"] = ceiling
    players[0].update(record)
    reply = reply_setup(False, RELEASED, players)
    settings = {"spots_folder": os.path.basename(os.path.normpath(folder)),
                "per_category": per_category, "spots": len(spots),
                "legal_moves": list(legal), "legal_order": legal_order,
                "move_outcomes": move_outcomes, "persona": persona, "persona_text": persona_text,
                "search_player": f"gt(d={SEARCH_DEPTH})", "original_rules": False,
                **reply_settings(reply, think, effort, False)}
    if changed_from:
        settings["changed_from"] = os.path.basename(os.path.normpath(changed_from))
    if spot_ids:
        settings["spot_ids"] = {"file": os.path.basename(spot_ids), "ids": read_ids(spot_ids)}
    if ceiling is not None:
        settings["ceiling"] = ceiling
    configs = {c: decision_config(reply, think, effort, False, CONDITIONS[c], 0,
                                  move_outcomes and CONDITIONS[c], legal_order) for c in legal}
    todo = {(s["id"], CONDITIONS[c]) for s in spots for c in legal}
    meta, done = resume_spots(out_dir, settings, players, todo)
    if done:
        print(f"  {len(done & todo)} of {len(todo)} spot prompts already asked")
    if todo <= done:
        return read_records(out_dir)
    run = RunLog(out_dir, os.path.basename(os.path.normpath(out_dir)), players, settings,
                 dict(configs[legal[0]], legal_moves_shown=[CONDITIONS[c] for c in legal],
                      move_outcomes=move_outcomes),
                 ollama_versions or {}, SpotEnv(range(4), {}), resumed=meta, games=False)
    run.meta["definitions"] = {**DEFINITIONS, **SPOT_DEFINITIONS}
    run.write_meta()
    agent, _ = build_agent(spec, llm_kwargs, None, True, False, 0, move_outcomes, legal_order)
    search = GameTheoryMultiplayerAgent(search_depth=SEARCH_DEPTH)
    info = players[0]
    try:
        for n, spot in enumerate(spots, 1):
            for c in legal:
                if (spot["id"], CONDITIONS[c]) in done:
                    continue
                pre, chosen, after, model = ask(agent, spot, CONDITIONS[c], persona_text, search)
                p = spot["current_player"]
                config = dict(configs[c], num_ctx=run.num_ctx(0, model["raw_reply"] is not None))
                rec = decision_record(
                    {"run_id": run.run_id, "game_id": None, "seed": None, "ply": 0, "turn_number": 0,
                     "player_id": p, "seat_position": spot["players"].index(p),
                     "model_spec": info["model_spec"], "host": info["host"], "skipped": False},
                    snapshot(SpotEnv(spot["players"], board(spot))), after, pre,
                    move_fields(after, pre, chosen), model, config)
                rec.update(spot_fields(spot, folder, original, persona),
                           search_move=pre["search_move"], move_groups=pre["move_groups"])
                run.write_decision(rec)
                print(f"  {n}/{len(spots)} {spot['id']} {c}: piece {chosen}"
                      + (" (invalid reply)" if model["was_fallback"] else ""))
    finally:
        run.close()
    return read_records(out_dir)


def read_records(out_dir):
    with open(os.path.join(out_dir, "decisions.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def summarise(records):
    """Invalid replies, agreement with the search player and choice rates, per condition."""
    for shown in (False, True):
        rs = [r for r in records if r["config"]["legal_moves_shown"] is shown]
        if not rs:
            continue
        valid = [r for r in rs if not r["was_fallback"]]
        agree = sum(r["chosen_index"] == r["search_move"] for r in valid)
        print(f"\nlegal moves {'shown' if shown else 'hidden'}: {len(rs)} spots, "
              f"invalid {len(rs) - len(valid)}, agrees with the search player "
              f"{agree}/{len(valid)}")
        for kind, x in choice_rates(rs).items():
            if x["available"]:
                print(f"  {kind:<11}{x['taken']:>4}/{x['available']:<4} {x['rate']:.0%}")


def main():
    names = sorted(k for k in persona_texts() if k != "none")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("--player", required=True, metavar="LLM",
                    help="llm:MODEL, llm://HOST[:PORT]/MODEL, llm:openrouter/MODEL, "
                         "llm:nvidia/MODEL or llm:bedrock/MODEL")
    ap.add_argument("--spots", default=DEFAULT_SPOTS, help="spot folder (default: spots_40_final)")
    ap.add_argument("--legal", choices=["both", *CONDITIONS], default="both",
                    help="ask with the legal-move list hidden, shown, or both (default)")
    ap.add_argument("--shuffle", action="store_true",
                    help="show the legal moves in a random order (the spot's legal_order stream)")
    ap.add_argument("--outcomes", action="store_true",
                    help="after the legal-move list, say what each legal move does")
    ap.add_argument("--per-category", type=int, metavar="N",
                    help="ask only N spots of each category, a grudge pair counting as one "
                         "(default: all)")
    ap.add_argument("--changed-from", metavar="FOLDER",
                    help="ask only the spots that differ from the same spot in FOLDER")
    ap.add_argument("--spot-ids", metavar="FILE",
                    help="ask only the spots listed in FILE, one id per line")
    ap.add_argument("--ceiling", type=int, metavar="N",
                    help="output-token ceiling, thinking included, below the model's own (default: its own)")
    ap.add_argument("--persona", choices=names, help="add this persona instruction (default: none)")
    ap.add_argument("--think", action="store_true", help="let a thinking model think")
    ap.add_argument("--effort", choices=["low", "medium", "high"], help="reasoning effort level")
    ap.add_argument("--requests-per-minute", type=float, metavar="N", help="pace API requests")
    ap.add_argument("--experiment", default="sandbox",
                    help="results go to results/<experiment>/<player>[_persona-NAME]/")
    args = ap.parse_args()
    if not is_llm_spec(args.player):
        ap.error("--player must be an LLM")
    if args.outcomes and args.legal == "hidden":
        ap.error("--outcomes needs the legal-move list in the prompt")
    if args.per_category is not None and args.per_category < 1:
        ap.error("--per-category must be 1 or more")
    try:
        folder = run_folder([args.player], args.effort)
    except ValueError as e:
        ap.error(str(e))
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ollama_versions = check_llm_players(ap, {0: args.player})
    out = os.path.join(RESULTS, args.experiment,
                       folder + (f"_persona-{args.persona}" if args.persona else ""))
    if git_state()["dirty"]:
        print("WARNING: tracked files have uncommitted changes (recorded in run.json)")
    print(f"writing to {out}\n")
    legal = tuple(CONDITIONS) if args.legal == "both" else (args.legal,)
    try:
        records = run_spots(args.player, out, args.spots, legal, args.persona,
                            "shuffled" if args.shuffle else "ascending", args.outcomes,
                            args.per_category, args.think, args.effort, args.requests_per_minute, ollama_versions,
                            args.changed_from, args.spot_ids, args.ceiling)
    except ValueError as e:                          # e.g. a thinking switch the model cannot take
        raise SystemExit(str(e))
    summarise(records)
    print(f"\nLogs: {out}  (decisions.jsonl, run.json)")


if __name__ == "__main__":
    main()
