import argparse
import glob
import json
import os
import re
import shutil
from collections import defaultdict

from spot_evaluation import evaluate_spot, SpotEnv, compute_legal_moves
from real_llm import real_llm

PERSONA_TEXTS = {
    "none": "",
    "aggressive": "Prioritize captures and direct pressure on opponents whenever reasonable.",
    "greedy": "Prioritize fastest progress to home and finishing tokens as quickly as possible.",
    "safe": "Prioritize low-risk moves, safe squares, and minimizing chances of being captured.",
    "unforgiving": "Prioritize retaliating against opponents who captured you in history/context when possible.",
}


def load_spots(files):
    spots = []
    for fname in files:
        with open(fname, "r") as f:
            data = json.load(f)
        for spot in data:
            spot["_source_file"] = fname
            spots.append(spot)
    return spots


def ensure_clean_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def classify_move(env, player, dice, token_idx):
    start = env.START_POSITIONS[player]
    home_start = env.HOME_START + player * env.HOME_LEN
    home_end = home_start + env.HOME_LEN - 1
    old_pos = env.tokens[player][token_idx]

    # Compute new_pos exactly as in ludo_env.py
    if old_pos == -1:
        new_pos = start
    elif 0 <= old_pos < env.BOARD_SIZE:
        rel_pos = (old_pos - start) % env.BOARD_SIZE
        new_rel = rel_pos + dice
        if new_rel < env.BOARD_SIZE:
            new_pos = (start + new_rel) % env.BOARD_SIZE
        else:
            home_step = new_rel - env.BOARD_SIZE
            new_pos = home_start + home_step - 1
    else:
        new_pos = old_pos + dice

    # Capture?
    capture = False
    captured_player = None
    if 0 <= new_pos < env.BOARD_SIZE and new_pos not in env.SAFE_SQUARES:
        for opponent in env.player_ids:
            if opponent == player:
                continue
            if new_pos in env.tokens[opponent]:
                capture = True
                captured_player = opponent
                break

    # Home entry / home path move
    home_entry = new_pos >= home_start

    return {
        "new_pos": new_pos,
        "capture": capture,
        "captured_player": captured_player,
        "home_entry": home_entry,
        "home_end": home_end,
        "old_pos": old_pos,
        "bring_out": old_pos == -1 and new_pos == start,
    }


def process_spots(spots, out_dir, no_plot=False, persona_name="none", persona_text=""):
    ensure_clean_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "llm_outputs"))

    results = []
    summary = defaultdict(lambda: {
        "total_all": 0,
        "total_valid": 0,
        "llm_equals_heuristic": 0,
        "llm_differs": 0,
        "llm_invalid": 0,
    })
    capture_vs_home = {
        "llm_capture": 0, "llm_home": 0, "llm_other": 0, "llm_total": 0,
        "heur_capture": 0, "heur_home": 0, "heur_other": 0, "heur_total": 0,
    }
    capture_vs_finish = {
        "llm_capture": 0, "llm_finish": 0, "llm_other": 0, "llm_total": 0,
        "heur_capture": 0, "heur_finish": 0, "heur_other": 0, "heur_total": 0,
    }
    capture_vs_openexisting = {
        "llm_capture": 0, "llm_open": 0, "llm_other": 0, "llm_total": 0,
        "heur_capture": 0, "heur_open": 0, "heur_other": 0, "heur_total": 0,
    }
    home_entry_stats = {"home_entry": 0, "other": 0, "total": 0}
    mixed_stats = {"capture": 0, "safe": 0, "home_entry": 0, "other": 0, "total": 0}
    overshoot_stats = {"overshoot_move": 0, "other": 0, "total": 0}
    safe_stats = {"safe": 0, "other": 0, "total": 0}
    capture_stats = {
        "llm_capture": 0, "llm_other": 0, "llm_total": 0,
        "heur_capture": 0, "heur_other": 0, "heur_total": 0,
    }
    roll6_stats = {"bring_out": 0, "move_existing": 0, "total": 0}
    capture_on_6_stats = {"capture": 0, "no_capture": 0, "total": 0}
    extra_turn_stats = {"bring_out": 0, "move_existing": 0, "total": 0}
    grudge_pairs = {}
    style_pairs = {}
    style_stats = {
        "aggressive": {"total": 0, "capture": 0, "safe": 0},
        "avoid": {"total": 0, "capture": 0, "safe": 0},
        "forgiving": {"total": 0, "capture": 0, "safe": 0},
    }

    for spot in spots:
        players = spot["players"]
        tokens_raw = spot["tokens"]
        tokens = {int(k): v for k, v in tokens_raw.items()}
        current_player = spot["current_player"]
        dice = spot["dice"]
        scenario = spot.get("scenario", "unknown")
        spot_id = spot.get("id", "spot")
        history_text = spot.get("history_text", "")

        env = SpotEnv(players, tokens)
        legal_moves = compute_legal_moves(env, current_player, dice)
        if not legal_moves:
            raise SystemExit(f"No legal moves for {spot_id} in {spot['_source_file']}")

        llm_outputs_dir = os.path.join(out_dir, "llm_outputs")
        ensure_dir(llm_outputs_dir)
        llm_output_path = os.path.join(llm_outputs_dir, f"spot_llm_{spot_id}.json")

        result = evaluate_spot(
            players=players,
            tokens=tokens,
            dice=dice,
            current_player=current_player,
            legal_moves=legal_moves,
            llm_call=real_llm,
            llm_output_path=llm_output_path,
            history_text=history_text,
            persona=persona_text,
        )

        llm_move = result["llm_move"]
        llm_raw = result.get("response_int", None)
        heuristic_move = result["heuristic_move"]

        llm_raw_valid = llm_raw in legal_moves if llm_raw is not None else False
        is_valid = llm_raw_valid
        equals = llm_move == heuristic_move

        summary[scenario]["total_all"] += 1
        if not is_valid:
            summary[scenario]["llm_invalid"] += 1
        else:
            summary[scenario]["total_valid"] += 1
            if equals:
                summary[scenario]["llm_equals_heuristic"] += 1
            else:
                summary[scenario]["llm_differs"] += 1

        results.append({
            "id": spot_id,
            "scenario": scenario,
            "players": players,
            "llm_player_id": spot.get("llm_player_id"),
            "current_player": current_player,
            "dice": dice,
            "tokens": tokens,
            "legal_moves": legal_moves,
            "llm_move": llm_move,
            "llm_raw_move": llm_raw,
            "llm_raw_valid": llm_raw_valid,
            "heuristic_move": heuristic_move,
            "llm_valid": is_valid,
            "llm_equals_heuristic": equals,
            "note": spot.get("note", ""),
            "history_text": history_text,
            "source_file": spot.get("_source_file", ""),
            "persona": persona_name,
        })

        if scenario == "grudge":
            pair_id = spot_id[:-2] if spot_id.endswith("_a") or spot_id.endswith("_b") else spot_id
            entry = grudge_pairs.setdefault(pair_id, {})
            entry[spot_id[-1]] = {
                "spot_id": spot_id,
                "history_text": history_text,
                "llm_move": llm_move,
                "llm_valid": is_valid,
                "players": players,
                "current_player": current_player,
                "dice": dice,
                "tokens": tokens,
            }

        if scenario == "opponent_style":
            pair_id = spot_id[:-2] if spot_id.endswith("_a") or spot_id.endswith("_b") else spot_id
            entry = style_pairs.setdefault(pair_id, {})
            entry[spot_id[-1]] = {
                "spot_id": spot_id,
                "history_text": history_text,
                "llm_move": llm_move,
                "llm_valid": is_valid,
                "players": players,
                "current_player": current_player,
                "dice": dice,
                "tokens": tokens,
            }

        if scenario == "capture_vs_home" and is_valid:
            env2 = SpotEnv(players, tokens)
            info = classify_move(env2, current_player, dice, llm_move)
            capture_vs_home["llm_total"] += 1
            if info["capture"]:
                capture_vs_home["llm_capture"] += 1
            elif info["home_entry"]:
                capture_vs_home["llm_home"] += 1
            else:
                capture_vs_home["llm_other"] += 1
        if scenario == "capture_vs_home" and is_valid:
            env2 = SpotEnv(players, tokens)
            info_h = classify_move(env2, current_player, dice, heuristic_move)
            capture_vs_home["heur_total"] += 1
            if info_h["capture"]:
                capture_vs_home["heur_capture"] += 1
            elif info_h["home_entry"]:
                capture_vs_home["heur_home"] += 1
            else:
                capture_vs_home["heur_other"] += 1

        if scenario == "capture_vs_home_finish" and is_valid:
            env2 = SpotEnv(players, tokens)
            info = classify_move(env2, current_player, dice, llm_move)
            capture_vs_finish["llm_total"] += 1
            if info["capture"]:
                capture_vs_finish["llm_capture"] += 1
            elif info["new_pos"] == info["home_end"]:
                capture_vs_finish["llm_finish"] += 1
            else:
                capture_vs_finish["llm_other"] += 1
        if scenario == "capture_vs_home_finish" and is_valid:
            env2 = SpotEnv(players, tokens)
            info_h = classify_move(env2, current_player, dice, heuristic_move)
            capture_vs_finish["heur_total"] += 1
            if info_h["capture"]:
                capture_vs_finish["heur_capture"] += 1
            elif info_h["new_pos"] == info_h["home_end"]:
                capture_vs_finish["heur_finish"] += 1
            else:
                capture_vs_finish["heur_other"] += 1

        if scenario == "capture" and is_valid:
            env2 = SpotEnv(players, tokens)
            info = classify_move(env2, current_player, dice, llm_move)
            capture_stats["llm_total"] += 1
            if info["capture"]:
                capture_stats["llm_capture"] += 1
            else:
                capture_stats["llm_other"] += 1
        if scenario == "capture" and is_valid:
            env2 = SpotEnv(players, tokens)
            info_h = classify_move(env2, current_player, dice, heuristic_move)
            capture_stats["heur_total"] += 1
            if info_h["capture"]:
                capture_stats["heur_capture"] += 1
            else:
                capture_stats["heur_other"] += 1

        if scenario == "capture_vs_openexisting" and is_valid:
            env2 = SpotEnv(players, tokens)
            info = classify_move(env2, current_player, dice, llm_move)
            capture_vs_openexisting["llm_total"] += 1
            if info["capture"]:
                capture_vs_openexisting["llm_capture"] += 1
            elif info["bring_out"]:
                capture_vs_openexisting["llm_open"] += 1
            else:
                capture_vs_openexisting["llm_other"] += 1
        if scenario == "capture_vs_openexisting" and is_valid:
            env2 = SpotEnv(players, tokens)
            info_h = classify_move(env2, current_player, dice, heuristic_move)
            capture_vs_openexisting["heur_total"] += 1
            if info_h["capture"]:
                capture_vs_openexisting["heur_capture"] += 1
            elif info_h["bring_out"]:
                capture_vs_openexisting["heur_open"] += 1
            else:
                capture_vs_openexisting["heur_other"] += 1

        if scenario == "home_entry" and is_valid:
            env2 = SpotEnv(players, tokens)
            info = classify_move(env2, current_player, dice, llm_move)
            home_entry_stats["total"] += 1
            if info["home_entry"]:
                home_entry_stats["home_entry"] += 1
            else:
                home_entry_stats["other"] += 1

        if scenario == "mixed" and is_valid:
            env2 = SpotEnv(players, tokens)
            info = classify_move(env2, current_player, dice, llm_move)
            mixed_stats["total"] += 1
            if info["capture"]:
                mixed_stats["capture"] += 1
            elif info["home_entry"]:
                mixed_stats["home_entry"] += 1
            elif info["new_pos"] in env2.SAFE_SQUARES:
                mixed_stats["safe"] += 1
            else:
                mixed_stats["other"] += 1

        if scenario == "overshoot" and is_valid:
            env2 = SpotEnv(players, tokens)
            info = classify_move(env2, current_player, dice, llm_move)
            overshoot_stats["total"] += 1
            home_end = info["home_end"]
            if info["new_pos"] > home_end:
                overshoot_stats["overshoot_move"] += 1
            else:
                overshoot_stats["other"] += 1

        if scenario == "safe" and is_valid:
            env2 = SpotEnv(players, tokens)
            info = classify_move(env2, current_player, dice, llm_move)
            safe_stats["total"] += 1
            if info["new_pos"] in env2.SAFE_SQUARES:
                safe_stats["safe"] += 1
            else:
                safe_stats["other"] += 1

        if scenario == "positional_roll6" and is_valid:
            env2 = SpotEnv(players, tokens)
            info = classify_move(env2, current_player, dice, llm_move)
            roll6_stats["total"] += 1
            if info["bring_out"]:
                roll6_stats["bring_out"] += 1
            else:
                roll6_stats["move_existing"] += 1

        if scenario == "extra_turn" and is_valid:
            env2 = SpotEnv(players, tokens)
            info = classify_move(env2, current_player, dice, llm_move)
            extra_turn_stats["total"] += 1
            if info["bring_out"]:
                extra_turn_stats["bring_out"] += 1
            else:
                extra_turn_stats["move_existing"] += 1

        if scenario == "positional_capture_on_6" and is_valid:
            env2 = SpotEnv(players, tokens)
            info = classify_move(env2, current_player, dice, llm_move)
            capture_on_6_stats["total"] += 1
            if info["capture"]:
                capture_on_6_stats["capture"] += 1
            else:
                capture_on_6_stats["no_capture"] += 1

    # finalize per-scenario rates
    for sc in summary:
        total = summary[sc]["total_all"]
        summary[sc]["llm_invalid_rate"] = (
            summary[sc]["llm_invalid"] / total if total else 0.0
        )

    summary_path = os.path.join(out_dir, "spot_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    results_path = os.path.join(out_dir, "spot_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, ensure_ascii=True, indent=2)

    grudge_report = {
        "pairs_total": 0,
        "pairs_with_change": 0,
        "change_rate": 0.0,
        "retaliation_grudge_hits": 0,
        "retaliation_grudge_total": 0,
        "retaliation_noconflict_hits": 0,
        "retaliation_noconflict_total": 0,
        "retaliation_grudge_rate": 0.0,
        "retaliation_noconflict_rate": 0.0,
        "transition_counts": {},
    }

    for pair_id, pair in grudge_pairs.items():
        if "a" not in pair or "b" not in pair:
            continue
        if not (pair["a"]["llm_valid"] and pair["b"]["llm_valid"]):
            continue
        grudge_report["pairs_total"] += 1

        move_a = pair["a"]["llm_move"]
        move_b = pair["b"]["llm_move"]
        if move_a != move_b:
            grudge_report["pairs_with_change"] += 1
            env_a = SpotEnv(pair["a"]["players"], {int(k): v for k, v in pair["a"]["tokens"].items()})
            env_b = SpotEnv(pair["b"]["players"], {int(k): v for k, v in pair["b"]["tokens"].items()})
            info_a = classify_move(env_a, pair["a"]["current_player"], pair["a"]["dice"], move_a)
            info_b = classify_move(env_b, pair["b"]["current_player"], pair["b"]["dice"], move_b)

            def label(info, env):
                if info["capture"]:
                    return "capture"
                if info["home_entry"]:
                    return "home_entry"
                if info["new_pos"] in env.SAFE_SQUARES:
                    return "safe"
                return "other"

            lab_a = label(info_a, env_a)
            lab_b = label(info_b, env_b)
            key = f"{lab_a}->{lab_b}"
            grudge_report["transition_counts"][key] = grudge_report["transition_counts"].get(key, 0) + 1

        def is_grudge(history_text):
            return "captured" in (history_text or "").lower()

        # Determine a single retaliation target for the pair from whichever side
        # has explicit "Player X captured..." history.
        target = None
        for key in ["a", "b"]:
            hist = pair[key].get("history_text", "")
            if not hist:
                continue
            m = re.search(r"Player\s+(\d+)\s+captured", hist)
            if m:
                target = int(m.group(1))
                break

        for key in ["a", "b"]:
            item = pair[key]
            if not item["llm_valid"]:
                continue
            env2 = SpotEnv(item["players"], {int(k): v for k, v in item["tokens"].items()})
            info = classify_move(env2, item["current_player"], item["dice"], item["llm_move"])

            if is_grudge(item["history_text"]):
                grudge_report["retaliation_grudge_total"] += 1
                if target is not None and info["captured_player"] == target:
                    grudge_report["retaliation_grudge_hits"] += 1
            else:
                grudge_report["retaliation_noconflict_total"] += 1
                if target is not None and info["captured_player"] == target:
                    grudge_report["retaliation_noconflict_hits"] += 1

    if grudge_report["pairs_total"]:
        grudge_report["change_rate"] = grudge_report["pairs_with_change"] / grudge_report["pairs_total"]
    if grudge_report["retaliation_grudge_total"]:
        grudge_report["retaliation_grudge_rate"] = (
            grudge_report["retaliation_grudge_hits"] / grudge_report["retaliation_grudge_total"]
        )
    if grudge_report["retaliation_noconflict_total"]:
        grudge_report["retaliation_noconflict_rate"] = (
            grudge_report["retaliation_noconflict_hits"] / grudge_report["retaliation_noconflict_total"]
        )

    grudge_path = os.path.join(out_dir, "grudge_report.json")
    with open(grudge_path, "w") as f:
        json.dump(grudge_report, f, ensure_ascii=True, indent=2)

    style_report = {
        "pairs_total": 0,
        "pairs_with_change": 0,
        "change_rate": 0.0,
        "capture_rate_by_style": {
            "aggressive": 0.0,
            "avoid": 0.0,
            "forgiving": 0.0,
        },
        "safe_rate_by_style": {
            "aggressive": 0.0,
            "avoid": 0.0,
            "forgiving": 0.0,
        },
    }

    def style_bucket(text):
        t = (text or "").lower()
        if "aggressive" in t or "captures whenever possible" in t:
            return "aggressive"
        if "avoid captures" in t or "safe squares" in t:
            return "avoid"
        if "forgives" in t or "forgiving" in t:
            return "forgiving"
        return None

    for pair_id, pair in style_pairs.items():
        if "a" not in pair or "b" not in pair:
            continue
        if not (pair["a"]["llm_valid"] and pair["b"]["llm_valid"]):
            continue
        style_report["pairs_total"] += 1
        if pair["a"]["llm_move"] != pair["b"]["llm_move"]:
            style_report["pairs_with_change"] += 1

        for key in ["a", "b"]:
            item = pair[key]
            if not item["llm_valid"]:
                continue
            bucket = style_bucket(item["history_text"])
            if not bucket:
                continue
            env2 = SpotEnv(item["players"], {int(k): v for k, v in item["tokens"].items()})
            info = classify_move(env2, item["current_player"], item["dice"], item["llm_move"])
            style_stats[bucket]["total"] += 1
            if info["capture"]:
                style_stats[bucket]["capture"] += 1
            if info["new_pos"] in env2.SAFE_SQUARES:
                style_stats[bucket]["safe"] += 1

    if style_report["pairs_total"]:
        style_report["change_rate"] = style_report["pairs_with_change"] / style_report["pairs_total"]
    for k in style_stats:
        if style_stats[k]["total"]:
            style_report["capture_rate_by_style"][k] = (
                style_stats[k]["capture"] / style_stats[k]["total"]
            )
            style_report["safe_rate_by_style"][k] = (
                style_stats[k]["safe"] / style_stats[k]["total"]
            )

    style_path = os.path.join(out_dir, "opponent_style_report.json")
    with open(style_path, "w") as f:
        json.dump(style_report, f, ensure_ascii=True, indent=2)

    # Behavior profile (per category)
    profile = {
        "persona": persona_name,
        "summary": summary,
        "capture_vs_home": None,
        "capture_vs_openexisting": None,
        "home_entry": None,
        "mixed": None,
        "overshoot": None,
        "safe": None,
        "roll6": None,
        "extra_turn": None,
        "capture_on_6": None,
        "grudge": grudge_report,
        "opponent_style": style_report,
    }

    if capture_vs_home["llm_total"] > 0:
        profile["capture_vs_home"] = {
            "capture_rate": capture_vs_home["llm_capture"] / capture_vs_home["llm_total"],
            "home_entry_rate": capture_vs_home["llm_home"] / capture_vs_home["llm_total"],
            "other_rate": capture_vs_home["llm_other"] / capture_vs_home["llm_total"],
        }
    if capture_vs_openexisting["llm_total"] > 0:
        profile["capture_vs_openexisting"] = {
            "capture_rate": capture_vs_openexisting["llm_capture"] / capture_vs_openexisting["llm_total"],
            "open_rate": capture_vs_openexisting["llm_open"] / capture_vs_openexisting["llm_total"],
            "other_rate": capture_vs_openexisting["llm_other"] / capture_vs_openexisting["llm_total"],
        }
    if home_entry_stats["total"] > 0:
        profile["home_entry"] = {
            "home_entry_rate": home_entry_stats["home_entry"] / home_entry_stats["total"],
            "other_rate": home_entry_stats["other"] / home_entry_stats["total"],
        }
    if mixed_stats["total"] > 0:
        profile["mixed"] = {
            "capture_rate": mixed_stats["capture"] / mixed_stats["total"],
            "safe_rate": mixed_stats["safe"] / mixed_stats["total"],
            "home_entry_rate": mixed_stats["home_entry"] / mixed_stats["total"],
            "other_rate": mixed_stats["other"] / mixed_stats["total"],
        }
    if overshoot_stats["total"] > 0:
        profile["overshoot"] = {
            "overshoot_rate": overshoot_stats["overshoot_move"] / overshoot_stats["total"],
            "other_rate": overshoot_stats["other"] / overshoot_stats["total"],
        }
    if safe_stats["total"] > 0:
        profile["safe"] = {
            "safe_rate": safe_stats["safe"] / safe_stats["total"],
            "other_rate": safe_stats["other"] / safe_stats["total"],
        }
    if roll6_stats["total"] > 0:
        profile["roll6"] = {
            "bring_out_rate": roll6_stats["bring_out"] / roll6_stats["total"],
            "move_existing_rate": roll6_stats["move_existing"] / roll6_stats["total"],
        }
    if extra_turn_stats["total"] > 0:
        profile["extra_turn"] = {
            "bring_out_rate": extra_turn_stats["bring_out"] / extra_turn_stats["total"],
            "move_existing_rate": extra_turn_stats["move_existing"] / extra_turn_stats["total"],
        }
    if capture_on_6_stats["total"] > 0:
        profile["capture_on_6"] = {
            "capture_rate": capture_on_6_stats["capture"] / capture_on_6_stats["total"],
            "no_capture_rate": capture_on_6_stats["no_capture"] / capture_on_6_stats["total"],
        }

    # Write per-category behavior profile (TXT)
    profile_path = os.path.join(out_dir, "behavior_profile.txt")
    with open(profile_path, "w") as f:
        f.write("Behavior Profile (Category)\n")
        f.write("============================\n\n")
        f.write(f"Total spots: {sum(v['total_all'] for v in summary.values())}\n")
        f.write(f"Valid spots: {sum(v['total_valid'] for v in summary.values())}\n\n")
        # overall invalid rate for this category
        total_spots = sum(v["total_all"] for v in summary.values())
        total_invalid = sum(v["llm_invalid"] for v in summary.values())
        if total_spots:
            f.write(f"llm_invalid_rate: {total_invalid/total_spots:.3f}\n\n")
        for k in ["capture_vs_home", "capture_vs_openexisting", "home_entry", "mixed", "overshoot", "safe", "roll6", "extra_turn", "capture_on_6"]:
            if profile.get(k):
                f.write(f"{k}:\n")
                for mk, mv in profile[k].items():
                    f.write(f"  {mk}: {mv:.3f}\n")
                f.write("\n")
        if grudge_report["pairs_total"] > 0:
            f.write("grudge:\n")
            f.write(f"  change_rate: {grudge_report['change_rate']:.3f}\n")
            f.write(f"  retaliation_grudge_rate: {grudge_report['retaliation_grudge_rate']:.3f}\n")
            f.write(f"  retaliation_noconflict_rate: {grudge_report['retaliation_noconflict_rate']:.3f}\n\n")
        if style_report["pairs_total"] > 0:
            f.write("opponent_style:\n")
            f.write(f"  change_rate: {style_report['change_rate']:.3f}\n")
            for k, v in style_report["capture_rate_by_style"].items():
                f.write(f"  capture_rate_{k}: {v:.3f}\n")
            f.write("\n")
    if no_plot:
        print(f"Wrote: {summary_path}")
        print(f"Wrote: {results_path}")
        print(f"Wrote: {profile_path}")
        return profile

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(
            f"matplotlib is required for plots (use --no-plot). {exc}"
        ) from exc

    scenarios = sorted(summary.keys())
    match_rates = []
    invalid_rates = []
    for s in scenarios:
        total_valid = summary[s]["total_valid"]
        total_all = summary[s]["total_all"]
        matches = summary[s]["llm_equals_heuristic"]
        match_rates.append(matches / total_valid if total_valid else 0.0)
        invalid_rates.append(summary[s]["llm_invalid_rate"] if total_all else 0.0)

    plt.figure(figsize=(10, 5))
    plt.bar(scenarios, match_rates)
    plt.ylim(0, 1)
    plt.ylabel("LLM == Heuristic (rate)")
    plt.title("LLM vs Heuristic Agreement by Scenario")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "spot_summary.png")
    plt.savefig(plot_path)

    # invalid-move rate graph
    plt.figure(figsize=(10, 5))
    plt.bar(scenarios, invalid_rates)
    plt.ylim(0, 1)
    plt.ylabel("Invalid Move Rate")
    plt.title("LLM Invalid Move Rate by Scenario")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plot_path_invalid = os.path.join(out_dir, "llm_invalid_rate.png")
    plt.savefig(plot_path_invalid)

    # grudge transition breakdown graph
    if grudge_report["transition_counts"]:
        labels = list(grudge_report["transition_counts"].keys())
        counts = [grudge_report["transition_counts"][k] for k in labels]
        plt.figure(figsize=(10, 5))
        plt.bar(labels, counts)
        plt.ylabel("Count")
        plt.title("Grudge Transition Counts")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plot_path_grudge = os.path.join(out_dir, "grudge_transitions.png")
        plt.savefig(plot_path_grudge)

    if capture_vs_home["llm_total"] > 0 and capture_vs_home["heur_total"] > 0:
        labels = ["capture", "home_entry", "other"]
        llm_rates = [
            capture_vs_home["llm_capture"] / capture_vs_home["llm_total"],
            capture_vs_home["llm_home"] / capture_vs_home["llm_total"],
            capture_vs_home["llm_other"] / capture_vs_home["llm_total"],
        ]
        heur_rates = [
            capture_vs_home["heur_capture"] / capture_vs_home["heur_total"],
            capture_vs_home["heur_home"] / capture_vs_home["heur_total"],
            capture_vs_home["heur_other"] / capture_vs_home["heur_total"],
        ]
        x = list(range(len(labels)))
        width = 0.35
        plt.figure(figsize=(7, 4))
        plt.bar([i - width/2 for i in x], llm_rates, width=width, label="LLM")
        plt.bar([i + width/2 for i in x], heur_rates, width=width, label="Heuristic", color="orange")
        plt.xticks(x, labels)
        plt.ylim(0, 1)
        plt.ylabel("Rate")
        plt.title("Capture vs Home (LLM vs Heuristic)")
        plt.legend()
        plt.tight_layout()
        plot_path2 = os.path.join(out_dir, "capture_vs_home.png")
        plt.savefig(plot_path2)

    if capture_vs_openexisting["llm_total"] > 0 and capture_vs_openexisting["heur_total"] > 0:
        labels = ["capture", "open_existing", "other"]
        llm_rates = [
            capture_vs_openexisting["llm_capture"] / capture_vs_openexisting["llm_total"],
            capture_vs_openexisting["llm_open"] / capture_vs_openexisting["llm_total"],
            capture_vs_openexisting["llm_other"] / capture_vs_openexisting["llm_total"],
        ]
        heur_rates = [
            capture_vs_openexisting["heur_capture"] / capture_vs_openexisting["heur_total"],
            capture_vs_openexisting["heur_open"] / capture_vs_openexisting["heur_total"],
            capture_vs_openexisting["heur_other"] / capture_vs_openexisting["heur_total"],
        ]
        x = list(range(len(labels)))
        width = 0.35
        plt.figure(figsize=(7, 4))
        plt.bar([i - width / 2 for i in x], llm_rates, width=width, label="LLM")
        plt.bar([i + width / 2 for i in x], heur_rates, width=width, label="Heuristic", color="orange")
        plt.xticks(x, labels)
        plt.ylim(0, 1)
        plt.ylabel("Rate")
        plt.title("Capture vs Open Existing (LLM vs Heuristic)")
        plt.legend()
        plt.tight_layout()
        plot_path2x = os.path.join(out_dir, "capture_vs_openexisting.png")
        plt.savefig(plot_path2x)

    if capture_vs_finish["llm_total"] > 0 and capture_vs_finish["heur_total"] > 0:
        labels = ["capture", "finish", "other"]
        llm_rates = [
            capture_vs_finish["llm_capture"] / capture_vs_finish["llm_total"],
            capture_vs_finish["llm_finish"] / capture_vs_finish["llm_total"],
            capture_vs_finish["llm_other"] / capture_vs_finish["llm_total"],
        ]
        heur_rates = [
            capture_vs_finish["heur_capture"] / capture_vs_finish["heur_total"],
            capture_vs_finish["heur_finish"] / capture_vs_finish["heur_total"],
            capture_vs_finish["heur_other"] / capture_vs_finish["heur_total"],
        ]
        x = list(range(len(labels)))
        width = 0.35
        plt.figure(figsize=(7, 4))
        plt.bar([i - width/2 for i in x], llm_rates, width=width, label="LLM")
        plt.bar([i + width/2 for i in x], heur_rates, width=width, label="Heuristic", color="orange")
        plt.xticks(x, labels)
        plt.ylim(0, 1)
        plt.ylabel("Rate")
        plt.title("Capture vs Home Finish (LLM vs Heuristic)")
        plt.legend()
        plt.tight_layout()
        plot_path2f = os.path.join(out_dir, "capture_vs_home_finish.png")
        plt.savefig(plot_path2f)

    if capture_stats["llm_total"] > 0 and capture_stats["heur_total"] > 0:
        labels = ["capture", "other"]
        llm_rates = [
            capture_stats["llm_capture"] / capture_stats["llm_total"],
            capture_stats["llm_other"] / capture_stats["llm_total"],
        ]
        heur_rates = [
            capture_stats["heur_capture"] / capture_stats["heur_total"],
            capture_stats["heur_other"] / capture_stats["heur_total"],
        ]
        x = list(range(len(labels)))
        width = 0.35
        plt.figure(figsize=(7, 4))
        plt.bar([i - width/2 for i in x], llm_rates, width=width, label="LLM")
        plt.bar([i + width/2 for i in x], heur_rates, width=width, label="Heuristic", color="orange")
        plt.xticks(x, labels)
        plt.ylim(0, 1)
        plt.ylabel("Rate")
        plt.title("Capture Rate (LLM vs Heuristic)")
        plt.legend()
        plt.tight_layout()
        plot_path2a = os.path.join(out_dir, "capture_rate.png")
        plt.savefig(plot_path2a)

    if home_entry_stats["total"] > 0:
        labels = ["home_entry", "other"]
        counts = [home_entry_stats[k] for k in labels]
        rates = [c / home_entry_stats["total"] for c in counts]
        plt.figure(figsize=(7, 4))
        plt.bar(labels, rates)
        plt.ylim(0, 1)
        plt.ylabel("Rate")
        plt.title("Home Entry Preference Rate")
        plt.tight_layout()
        plot_path2b = os.path.join(out_dir, "home_entry_preference.png")
        plt.savefig(plot_path2b)

    if mixed_stats["total"] > 0:
        labels = ["capture", "safe", "home_entry", "other"]
        counts = [mixed_stats[k] for k in labels]
        rates = [c / mixed_stats["total"] for c in counts]
        plt.figure(figsize=(8, 4))
        plt.bar(labels, rates)
        plt.ylim(0, 1)
        plt.ylabel("Rate")
        plt.title("Mixed Spots: Choice Breakdown")
        plt.tight_layout()
        plot_path2c = os.path.join(out_dir, "mixed_choice_breakdown.png")
        plt.savefig(plot_path2c)

    if overshoot_stats["total"] > 0:
        labels = ["overshoot_move", "other"]
        counts = [overshoot_stats[k] for k in labels]
        rates = [c / overshoot_stats["total"] for c in counts]
        plt.figure(figsize=(7, 4))
        plt.bar(labels, rates)
        plt.ylim(0, 1)
        plt.ylabel("Rate")
        plt.title("Overshoot Move Rate")
        plt.tight_layout()
        plot_path2d = os.path.join(out_dir, "overshoot_rate.png")
        plt.savefig(plot_path2d)

    if safe_stats["total"] > 0:
        labels = ["safe", "other"]
        counts = [safe_stats[k] for k in labels]
        rates = [c / safe_stats["total"] for c in counts]
        plt.figure(figsize=(7, 4))
        plt.bar(labels, rates)
        plt.ylim(0, 1)
        plt.ylabel("Rate")
        plt.title("Safe Square Preference Rate")
        plt.tight_layout()
        plot_path2e = os.path.join(out_dir, "safe_square_preference.png")
        plt.savefig(plot_path2e)

    if grudge_report["pairs_total"] > 0:
        labels = ["change_rate", "retaliation_grudge", "retaliation_no_conflict"]
        rates = [
            grudge_report["change_rate"],
            grudge_report["retaliation_grudge_rate"],
            grudge_report["retaliation_noconflict_rate"],
        ]
        plt.figure(figsize=(8, 4))
        plt.bar(labels, rates)
        plt.ylim(0, 1)
        plt.ylabel("Rate")
        plt.title("Grudge Metrics")
        plt.tight_layout()
        plot_path3 = os.path.join(out_dir, "grudge_metrics.png")
        plt.savefig(plot_path3)

    if roll6_stats["total"] > 0:
        labels = ["bring_out", "move_existing"]
        counts = [roll6_stats[k] for k in labels]
        rates = [c / roll6_stats["total"] for c in counts]
        plt.figure(figsize=(7, 4))
        plt.bar(labels, rates)
        plt.ylim(0, 1)
        plt.ylabel("Rate")
        plt.title("Roll-6 Choice: Bring Out vs Move Existing")
        plt.tight_layout()
        plot_path4 = os.path.join(out_dir, "roll6_bringout_vs_existing.png")
        plt.savefig(plot_path4)

    if capture_on_6_stats["total"] > 0:
        labels = ["capture", "no_capture"]
        counts = [capture_on_6_stats[k] for k in labels]
        rates = [c / capture_on_6_stats["total"] for c in counts]
        plt.figure(figsize=(7, 4))
        plt.bar(labels, rates)
        plt.ylim(0, 1)
        plt.ylabel("Rate")
        plt.title("Capture on 6 Rate")
        plt.tight_layout()
        plot_path5 = os.path.join(out_dir, "capture_on_6_rate.png")
        plt.savefig(plot_path5)

    if extra_turn_stats["total"] > 0:
        labels = ["bring_out", "move_existing"]
        counts = [extra_turn_stats[k] for k in labels]
        rates = [c / extra_turn_stats["total"] for c in counts]
        plt.figure(figsize=(7, 4))
        plt.bar(labels, rates)
        plt.ylim(0, 1)
        plt.ylabel("Rate")
        plt.title("Extra Turn (6): Bring Out vs Move Existing")
        plt.tight_layout()
        plot_path8 = os.path.join(out_dir, "extra_turn_bringout_vs_existing.png")
        plt.savefig(plot_path8)

    if style_report["pairs_total"] > 0:
        labels = ["change_rate"]
        rates = [style_report["change_rate"]]
        plt.figure(figsize=(6, 4))
        plt.bar(labels, rates)
        plt.ylim(0, 1)
        plt.ylabel("Rate")
        plt.title("Opponent Style: Change Rate")
        plt.tight_layout()
        plot_path6 = os.path.join(out_dir, "opponent_style_change_rate.png")
        plt.savefig(plot_path6)

        labels = ["aggressive", "avoid", "forgiving"]
        rates = [style_report["capture_rate_by_style"][k] for k in labels]
        plt.figure(figsize=(7, 4))
        plt.bar(labels, rates)
        plt.ylim(0, 1)
        plt.ylabel("Capture Rate")
        plt.title("Capture Rate by Opponent Style")
        plt.tight_layout()
        plot_path7 = os.path.join(out_dir, "opponent_style_capture_rate.png")
        plt.savefig(plot_path7)

        rates = [style_report["safe_rate_by_style"][k] for k in labels]
        plt.figure(figsize=(7, 4))
        plt.bar(labels, rates)
        plt.ylim(0, 1)
        plt.ylabel("Safe-Square Rate")
        plt.title("Safe-Square Rate by Opponent Style")
        plt.tight_layout()
        plot_path9 = os.path.join(out_dir, "opponent_style_safe_rate.png")
        plt.savefig(plot_path9)

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {results_path}")
    print(f"Wrote: {profile_path}")
    return profile


def main():
    parser = argparse.ArgumentParser(
        description="Run all spot scenarios and compare LLM vs Heuristic."
    )
    parser.add_argument(
        "--spots-glob",
        default="spots/spots_*.json",
        help="Glob for spot files (default: spots/spots_*.json)",
    )
    parser.add_argument(
        "--out-dir",
        default="spot_results",
        help="Output directory root for category results",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation (matplotlib required otherwise)",
    )
    parser.add_argument(
        "--persona",
        default="none",
        choices=sorted(PERSONA_TEXTS.keys()),
        help="Persona instruction for LLM prompt (default: none)",
    )
    parser.add_argument(
        "--all-personas",
        action="store_true",
        help="Run once per persona and write into out-dir/<persona>/...",
    )

    args = parser.parse_args()

    files = sorted(glob.glob(args.spots_glob))
    if not files:
        raise SystemExit(f"No spot files match {args.spots_glob}")

    spots = load_spots(files)

    by_category = defaultdict(list)
    for spot in spots:
        cat = spot.get("scenario", "unknown")
        by_category[cat].append(spot)

    personas_to_run = sorted(PERSONA_TEXTS.keys()) if args.all_personas else [args.persona]
    for persona_name in personas_to_run:
        persona_text = PERSONA_TEXTS[persona_name]
        # Always store outputs under persona folder: <out-dir>/<persona>/<category>/...
        persona_out_root = os.path.join(args.out_dir, persona_name)
        overall_profiles = {}
        for cat in sorted(by_category.keys()):
            out_dir = os.path.join(persona_out_root, cat)
            overall_profiles[cat] = process_spots(
                by_category[cat],
                out_dir,
                no_plot=args.no_plot,
                persona_name=persona_name,
                persona_text=persona_text,
            )

        # Write overall behavior profile summary
        overall_path = os.path.join(persona_out_root, "overall_behavior_profile.txt")
        with open(overall_path, "w") as f:
            f.write("Overall Behavior Profile\n")
            f.write("=========================\n\n")
            f.write(f"persona={persona_name}\n\n")
            for cat in sorted(overall_profiles.keys()):
                f.write(f"[{cat}]\n")
                prof = overall_profiles[cat]
                if not prof:
                    f.write("  (no data)\n\n")
                    continue
                if prof.get("capture_vs_home"):
                    f.write(f"  capture_vs_home.capture_rate={prof['capture_vs_home']['capture_rate']:.3f}\n")
                    f.write(f"  capture_vs_home.home_entry_rate={prof['capture_vs_home']['home_entry_rate']:.3f}\n")
                if prof.get("capture_vs_openexisting"):
                    f.write(f"  capture_vs_openexisting.capture_rate={prof['capture_vs_openexisting']['capture_rate']:.3f}\n")
                    f.write(f"  capture_vs_openexisting.open_rate={prof['capture_vs_openexisting']['open_rate']:.3f}\n")
                if prof.get("home_entry"):
                    f.write(f"  home_entry.home_entry_rate={prof['home_entry']['home_entry_rate']:.3f}\n")
                if prof.get("safe"):
                    f.write(f"  safe.safe_rate={prof['safe']['safe_rate']:.3f}\n")
                if prof.get("mixed"):
                    f.write(f"  mixed.capture_rate={prof['mixed']['capture_rate']:.3f}\n")
                    f.write(f"  mixed.safe_rate={prof['mixed']['safe_rate']:.3f}\n")
                    f.write(f"  mixed.home_entry_rate={prof['mixed']['home_entry_rate']:.3f}\n")
                if prof.get("overshoot"):
                    f.write(f"  overshoot.overshoot_rate={prof['overshoot']['overshoot_rate']:.3f}\n")
                if prof.get("roll6"):
                    f.write(f"  roll6.bring_out_rate={prof['roll6']['bring_out_rate']:.3f}\n")
                if prof.get("extra_turn"):
                    f.write(f"  extra_turn.bring_out_rate={prof['extra_turn']['bring_out_rate']:.3f}\n")
                if prof.get("capture_on_6"):
                    f.write(f"  capture_on_6.capture_rate={prof['capture_on_6']['capture_rate']:.3f}\n")
                if prof.get("grudge") and prof["grudge"]["pairs_total"]:
                    f.write(f"  grudge.change_rate={prof['grudge']['change_rate']:.3f}\n")
                if prof.get("opponent_style") and prof["opponent_style"]["pairs_total"]:
                    f.write(f"  opponent_style.change_rate={prof['opponent_style']['change_rate']:.3f}\n")
                # overall invalid rate (category)
                if prof.get("summary"):
                    total_spots = sum(v["total_all"] for v in prof["summary"].values())
                    total_invalid = sum(v["llm_invalid"] for v in prof["summary"].values())
                    if total_spots:
                        f.write(f"  llm_invalid_rate={total_invalid/total_spots:.3f}\n")
                f.write("\n")
        print(f"Wrote: {overall_path}")


if __name__ == "__main__":
    main()
