"""
Stage 1b: tidy tables, one row per decision, game or spot, from the datasets
of catalog.py. Everything later reads these tables, not the raw runs.

  spot_reference   per spot of spots_40_final/: its category and board facts,
                   and the moves of the reference players (search player at
                   depths 2 and 1, heuristic, first-listed, i.e. the lowest
                   legal piece) with the search player's value of each move
  spot_decisions   per spot prompt asked: the model's choice, its measures
                   (positions.measures), what the chosen move does (our rules),
                   the labels of the original release's scoring, and the
                   reason's themes and claim checks (reasons.py)
  games            per game of a single-model experiment (and the bots): the
                   analysed player's result and its per-game statistics
  game_decisions   per decision of that player in those games, with measures
  h2h_players      per player of every round-robin game
"""
import json
import os
import re

import game_theory_agent
import game_theory_multiplayer_agent
import ludo_rules as rules
from analysis import reasons as R
from analysis.common import games as kept_games
from analysis.common import lines
from analysis.final.catalog import REASONING_SHARE, changed_spots, complete_spots
from analysis.positions import complete, measures
from game_theory_multiplayer_agent import GameTheoryMultiplayerAgent
from repair_spots import cannot_move

KINDS = ("capture", "safe", "open", "home_entry", "finish")
TARGET = re.compile(r"Player\s+(\d+)\s+captured")
CLAIM_NAMES = tuple(R.CLAIMS)
THEME_NAMES = tuple(R.THEMES)


def search_values(env, player, dice, legal, depth=2):
    """(move the search player chooses, {move: its value}) - the values its best_move sees."""
    seen = {}
    original = game_theory_agent.best_move

    def capture(values, legal_moves, orig):
        seen.update(values)
        return original(values, legal_moves, orig)
    game_theory_agent.best_move = game_theory_multiplayer_agent.best_move = capture
    try:
        move = GameTheoryMultiplayerAgent(search_depth=depth).select_move(env, player, dice, legal)
    finally:
        game_theory_agent.best_move = game_theory_multiplayer_agent.best_move = original
    return move, seen


def spot_reference(spots):
    """One row per spot (spots: the spot dicts of spots_40_final/)."""
    from agents import HeuristicAgent
    from board_render import group_moves
    from spot_evaluation import SpotEnv
    changed = changed_spots()
    rows = []
    for s in spots:
        tokens = {int(p): list(t) for p, t in s["tokens"].items()}
        env, p, dice = SpotEnv(s["players"], tokens), s["current_player"], s["dice"]
        legal = rules.legal_moves(tokens, p, dice)
        groups = group_moves(env, p, dice, legal)
        group = {t: i for i, g in enumerate(groups) for t in g}
        d2, values = search_values(env, p, dice, legal, 2)
        d1, _ = search_values(env, p, dice, legal, 1)
        heur = HeuristicAgent().select_move(env, p, dice, legal)
        hist = s.get("history_text") or ""
        target = TARGET.search(hist)
        rows.append({"spot_id": s["id"], "category": s["scenario"], "n_players": len(s["players"]),
                     "player": p, "dice": dice, "legal": json.dumps(legal), "groups": json.dumps(groups),
                     "n_groups": len(groups), "search_move": d2, "search_d1": d1, "heuristic": heur,
                     "lowest": min(legal), "values": json.dumps({str(m): v for m, v in values.items()}),
                     "d1_agree": group[d1] == group[d2], "heuristic_agree": group[heur] == group[d2],
                     "lowest_agree": group[min(legal)] == group[d2], "chance": 1 / len(groups),
                     "pair_id": s.get("pair_id") or (s["id"][:-2] if s["scenario"] == "grudge" else None),
                     "history": ("grudge" if "captured" in hist.lower() else "neutral")
                     if s["scenario"] == "grudge" else None,
                     "target": int(target.group(1)) if target else None, "reviewed": s["id"] in changed})
    # a grudge pair's retaliation target is the player its grudge half names
    by_pair = {r["pair_id"]: r["target"] for r in rows if r["target"] is not None}
    for r in rows:
        if r["category"] == "grudge":
            r["target"] = by_pair.get(r["pair_id"])
    return rows


def effects(d):
    """What the chosen move does under our rules, and the labels of the original scoring."""
    tokens = {int(q): list(t) for q, t in d["tokens_before"].items()}
    e = rules.move_outcome(tokens, d["player_id"], d["chosen_index"], d["dice"])
    return {"to": e["to"], "captured_players": ";".join(sorted({str(o) for o, _ in e["captures"]})),
            "eff_capture": bool(e["captures"]), "eff_bring_out": e["bring_out"],
            "eff_safe_square": e["safe"],                       # bring-outs included, as the original
            "eff_safe": e["safe"] and not e["bring_out"], "eff_in_home": e["in_home"],
            "eff_home_entry": e["home_entry"], "eff_finish": e["home_finish"]}


def attempt(d):
    """Why the piece an invalid reply named cannot move ("" if none was named or it is legal)."""
    i = d.get("parsed_index")
    tokens = {int(q): list(t) for q, t in d["tokens_before"].items()}
    if i is None or i in d["legal_moves"] or not 0 <= i < len(tokens[d["player_id"]]):
        return "" if i is None or i in d["legal_moves"] else "not a piece"
    return cannot_move(tokens, d["player_id"], i, d["dice"], False)


def reason_columns(d, m):
    """Themes and claim checks of a valid reply's reason (blank without one)."""
    reason = d.get("parsed_reason")
    if not reason or not m["valid"] or not m["choice"]:
        return {"has_reason": False}
    facts = R.move_facts(d)
    th, cl = R.themes(reason), R.claims(reason, d["chosen_index"], facts)
    named = {int(t) for t in R.OWN_TOKEN.findall(reason)}
    return {"has_reason": True, **{f"theme_{k}": v for k, v in th.items()},
            **{f"claim_{k}": cl.get(k) for k in CLAIM_NAMES},
            "any_claim": bool(cl), "any_false": any(v is False for v in cl.values()),
            "names_chosen": d["chosen_index"] in named if named else None,
            "names_only_other": bool(named) and d["chosen_index"] not in named}


def measured(d):
    """Measures of a completed decision, flattened."""
    m = measures(d)
    row = {"choice": m["choice"], "valid": m["valid"], "agree": m["agree"] and m["valid"],
           "chance": m["chance"], "first": m.get("first"), "first_agree": m.get("first_agree")}
    for k in KINDS:
        for f in ("offered", "taken", "chance", "search"):
            row[f"{k}_{f}"] = m[k][f]
    return row, m


def model_fields(d):
    reasoned = bool((d.get("thinking_tokens") or 0) > 0 or d.get("reasoning_text"))
    return {"cause": d.get("fallback_cause") or "", "completion_tokens": d.get("completion_tokens"),
            "thinking_tokens": d.get("thinking_tokens"), "latency_ms": d.get("latency_ms"),
            "cost_usd": d.get("cost_usd"), "reasoned": reasoned, "replied": d.get("raw_reply") is not None}


def spot_rows(ds, records, ref):
    """Tidy rows of one spot dataset."""
    out = []
    for d in records:
        complete(d)
        row, m = measured(d)
        r = ref[d["spot_id"]]
        valid = m["valid"]
        out.append({"experiment": ds.experiment, "model": ds.model, "setting": ds.setting, "subset": ds.subset,
                    "condition": "shown" if d["config"]["legal_moves_shown"] else "hidden",
                    "spot_id": d["spot_id"], "category": r["category"], "n_players": r["n_players"],
                    "pair_id": r["pair_id"], "history": r["history"], "target": r["target"],
                    "reviewed": r["reviewed"], "dice": d["dice"], "n_legal": len(d["legal_moves"]),
                    "n_groups": r["n_groups"], "chosen": d["chosen_index"], "parsed_index": d.get("parsed_index"),
                    "heuristic_move": d.get("heuristic_move"), "search_move": d["search_move"],
                    "shown_legal_moves": json.dumps(d.get("shown_legal_moves")),
                    "move_groups": json.dumps(d["move_groups"]), "list_position": d.get("chosen_list_position"),
                    "legal_order": d["config"].get("legal_order"), "outcomes": d["config"].get("move_outcomes"),
                    "prompt_hash": d.get("prompt_hash"), **row, **model_fields(d),
                    "attempt": attempt(d) if not valid else "",
                    **(effects(d) if valid else {}), **reason_columns(d, m),
                    "reason": d.get("parsed_reason") if valid else None})
    return out


def dataset_spots(ds, ref):
    records, check = complete_spots(ds)
    return records, spot_rows(ds, records, ref), check


def game_row(ds, g, pid):
    p = g["per_player"][str(pid)]
    llm = p.get("llm") or {}
    return {"experiment": ds.experiment, "model": ds.model, "setting": ds.setting, "seed": g["seed"],
            "game": f"{g['_run']}#{g['game_id']}", "won": int(g["winner"] == pid),
            "no_winner": int(g["winner"] is None), "first_player": int(g.get("first_player") == pid),
            "seat_position": p["seat_position"], "dice_rolls": g["dice_rolls"], "turns": g["turns"],
            "decisions": p["decisions"], "captures_made": p["captures_made"],
            "times_captured": p["times_captured"], "pieces_finished": p.get("pieces_finished"),
            "calls": llm.get("calls", 0), "invalid": llm.get("invalid", 0),
            **{f"invalid_{c}": llm.get("invalid_by_cause", {}).get(c, 0) for c in ("empty", "unparseable", "illegal")},
            "prompt_tokens": llm.get("prompt_tokens"), "completion_tokens": llm.get("completion_tokens"),
            "latency_ms": llm.get("latency_ms"), "cost_usd": llm.get("cost_usd"),
            "retries": llm.get("retries"), "timeouts": llm.get("timeouts")}


def game_decisions(recs, pid):
    """The analysed player's full decision records with a legal move, in the kept games (replays included)."""
    sources = {}
    for g in recs:
        sources.setdefault(g["_dir"], set()).add(g["game_id"])
    for folder, keep in sources.items():
        for d in lines(os.path.join(folder, "decisions.jsonl")):
            if d["player_id"] == pid and d["game_id"] in keep:
                yield d


def dataset_games(ds, runs):
    """(game rows, decision rows, notes, reasoning) of a single-model game dataset (its run's player)."""
    run = ds.runs[0]
    recs, notes = kept_games(run, runs)
    pid = run.pid
    grows = [game_row(ds, g, pid) for g in recs]
    drows, replies = [], []
    for d in game_decisions(recs, pid):
        if d.get("raw_reply") is not None:
            replies.append(d)
        if not d["legal_moves"]:
            continue
        complete(d)
        row, m = measured(d)
        reason = reason_columns(d, m)
        drows.append({"experiment": ds.experiment, "model": ds.model, "seed": d["seed"], "game_id": d["game_id"],
                      "ply": d["ply"], "dice": d["dice"], "n_groups": len(d["move_groups"]),
                      "shown": d["config"]["legal_moves_shown"], **row, **model_fields(d),
                      **{k: v for k, v in reason.items() if k != "reason"}})
    return grows, drows, notes, reasoning(replies) if replies else (0, 0, 0.0, "")


def h2h_rows(ds, runs):
    """One row per player of each round-robin game."""
    from analysis.final.catalog import model_info
    out = []
    for run in ds.runs:
        for g in kept_games(run, runs)[0]:
            order = g["finishing_order"]
            for pid, spec in g["players"].items():
                p = g["per_player"][pid]
                llm = p.get("llm") or {}
                out.append({"experiment": ds.experiment, "run": run.key, "seating": run.name, "seed": g["seed"],
                            "game": f"{run.key}#{g['game_id']}", "n_players": len(g["players"]),
                            "model": model_info(spec)["model"], "display": model_info(spec)["display"],
                            "pid": int(pid), "seat_position": p["seat_position"],
                            "first_player": int(g.get("first_player") == int(pid)),
                            "won": int(g["winner"] == int(pid)), "place": order.index(int(pid)) + 1,
                            "captures_made": p["captures_made"], "calls": llm.get("calls", 0),
                            "invalid": llm.get("invalid", 0), "prompt_tokens": llm.get("prompt_tokens"),
                            "completion_tokens": llm.get("completion_tokens"), "dice_rolls": g["dice_rolls"]})
    return out


def reasoning(records):
    """(replies, replies with reasoning, share, group) from a dataset's records."""
    replies = [r for r in records if r.get("raw_reply") is not None]
    k = sum(bool((r.get("thinking_tokens") or 0) > 0 or r.get("reasoning_text")) for r in replies)
    share = k / len(replies) if replies else 0.0
    return len(replies), k, share, "reasoning" if share >= REASONING_SHARE else "non-reasoning"
