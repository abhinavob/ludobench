import argparse
import glob
import json
import os
import re
import shutil
from collections import defaultdict

from agents import HeuristicAgent
from game_theory_multiplayer_agent import GameTheoryMultiplayerAgent
from spot_evaluation import SpotEnv, compute_legal_moves


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


def classify_move(env, player, dice, token_idx):
    start = env.START_POSITIONS[player]
    home_start = env.HOME_START + player * env.HOME_LEN
    home_end = home_start + env.HOME_LEN - 1
    old_pos = env.tokens[player][token_idx]

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

    return {
        "new_pos": new_pos,
        "capture": capture,
        "captured_player": captured_player,
        "home_entry": new_pos >= home_start,
        "home_end": home_end,
        "old_pos": old_pos,
        "bring_out": old_pos == -1 and new_pos == start,
    }


def process_spots(spots, out_dir, depth):
    ensure_clean_dir(out_dir)

    results = []
    summary = defaultdict(
        lambda: {
            "total_all": 0,
            "total_valid": 0,
            "gt_equals_heuristic": 0,
            "gt_differs": 0,
            "gt_invalid": 0,
        }
    )

    capture_vs_home = {
        "gt_capture": 0,
        "gt_home": 0,
        "gt_other": 0,
        "gt_total": 0,
        "heur_capture": 0,
        "heur_home": 0,
        "heur_other": 0,
        "heur_total": 0,
    }
    capture_vs_finish = {
        "gt_capture": 0,
        "gt_finish": 0,
        "gt_other": 0,
        "gt_total": 0,
        "heur_capture": 0,
        "heur_finish": 0,
        "heur_other": 0,
        "heur_total": 0,
    }
    capture_vs_openexisting = {
        "gt_capture": 0,
        "gt_open": 0,
        "gt_other": 0,
        "gt_total": 0,
        "heur_capture": 0,
        "heur_open": 0,
        "heur_other": 0,
        "heur_total": 0,
    }
    home_entry_stats = {"home_entry": 0, "other": 0, "total": 0}
    mixed_stats = {"capture": 0, "safe": 0, "home_entry": 0, "other": 0, "total": 0}
    overshoot_stats = {"overshoot_move": 0, "other": 0, "total": 0}
    safe_stats = {"safe": 0, "other": 0, "total": 0}
    capture_stats = {
        "gt_capture": 0,
        "gt_other": 0,
        "gt_total": 0,
        "heur_capture": 0,
        "heur_other": 0,
        "heur_total": 0,
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

    gt_agent = GameTheoryMultiplayerAgent(search_depth=depth)
    heuristic_agent = HeuristicAgent()

    for spot in spots:
        players = spot["players"]
        tokens = {int(k): v for k, v in spot["tokens"].items()}
        current_player = spot["current_player"]
        dice = spot["dice"]
        scenario = spot.get("scenario", "unknown")
        spot_id = spot.get("id", "spot")
        history_text = spot.get("history_text", "")

        env = SpotEnv(players, tokens)
        legal_moves = compute_legal_moves(env, current_player, dice)
        if not legal_moves:
            raise SystemExit(f"No legal moves for {spot_id} in {spot['_source_file']}")

        gt_move = gt_agent.select_move(env, current_player, dice, legal_moves)
        heuristic_move = heuristic_agent.select_move(env, current_player, dice, legal_moves)

        is_valid = gt_move in legal_moves
        equals = gt_move == heuristic_move
        gt_mode = "expectiminimax_2p" if len(players) == 2 else "expectimaxn_multiplayer"

        summary[scenario]["total_all"] += 1
        if not is_valid:
            summary[scenario]["gt_invalid"] += 1
        else:
            summary[scenario]["total_valid"] += 1
            if equals:
                summary[scenario]["gt_equals_heuristic"] += 1
            else:
                summary[scenario]["gt_differs"] += 1

        results.append(
            {
                "id": spot_id,
                "scenario": scenario,
                "players": players,
                "llm_player_id": spot.get("llm_player_id"),
                "current_player": current_player,
                "dice": dice,
                "tokens": tokens,
                "legal_moves": legal_moves,
                "gt_move": gt_move,
                "heuristic_move": heuristic_move,
                "gt_valid": is_valid,
                "gt_equals_heuristic": equals,
                "gt_mode": gt_mode,
                "note": spot.get("note", ""),
                "history_text": history_text,
                "source_file": spot.get("_source_file", ""),
            }
        )

        if scenario == "grudge":
            pair_id = spot_id[:-2] if spot_id.endswith("_a") or spot_id.endswith("_b") else spot_id
            entry = grudge_pairs.setdefault(pair_id, {})
            entry[spot_id[-1]] = {
                "spot_id": spot_id,
                "history_text": history_text,
                "gt_move": gt_move,
                "gt_valid": is_valid,
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
                "gt_move": gt_move,
                "gt_valid": is_valid,
                "players": players,
                "current_player": current_player,
                "dice": dice,
                "tokens": tokens,
            }

        if scenario == "capture_vs_home" and is_valid:
            info = classify_move(SpotEnv(players, tokens), current_player, dice, gt_move)
            capture_vs_home["gt_total"] += 1
            if info["capture"]:
                capture_vs_home["gt_capture"] += 1
            elif info["home_entry"]:
                capture_vs_home["gt_home"] += 1
            else:
                capture_vs_home["gt_other"] += 1

            info_h = classify_move(SpotEnv(players, tokens), current_player, dice, heuristic_move)
            capture_vs_home["heur_total"] += 1
            if info_h["capture"]:
                capture_vs_home["heur_capture"] += 1
            elif info_h["home_entry"]:
                capture_vs_home["heur_home"] += 1
            else:
                capture_vs_home["heur_other"] += 1

        if scenario == "capture_vs_home_finish" and is_valid:
            info = classify_move(SpotEnv(players, tokens), current_player, dice, gt_move)
            capture_vs_finish["gt_total"] += 1
            if info["capture"]:
                capture_vs_finish["gt_capture"] += 1
            elif info["new_pos"] == info["home_end"]:
                capture_vs_finish["gt_finish"] += 1
            else:
                capture_vs_finish["gt_other"] += 1

            info_h = classify_move(SpotEnv(players, tokens), current_player, dice, heuristic_move)
            capture_vs_finish["heur_total"] += 1
            if info_h["capture"]:
                capture_vs_finish["heur_capture"] += 1
            elif info_h["new_pos"] == info_h["home_end"]:
                capture_vs_finish["heur_finish"] += 1
            else:
                capture_vs_finish["heur_other"] += 1

        if scenario == "capture" and is_valid:
            info = classify_move(SpotEnv(players, tokens), current_player, dice, gt_move)
            capture_stats["gt_total"] += 1
            if info["capture"]:
                capture_stats["gt_capture"] += 1
            else:
                capture_stats["gt_other"] += 1

            info_h = classify_move(SpotEnv(players, tokens), current_player, dice, heuristic_move)
            capture_stats["heur_total"] += 1
            if info_h["capture"]:
                capture_stats["heur_capture"] += 1
            else:
                capture_stats["heur_other"] += 1

        if scenario == "capture_vs_openexisting" and is_valid:
            info = classify_move(SpotEnv(players, tokens), current_player, dice, gt_move)
            capture_vs_openexisting["gt_total"] += 1
            if info["capture"]:
                capture_vs_openexisting["gt_capture"] += 1
            elif info["bring_out"]:
                capture_vs_openexisting["gt_open"] += 1
            else:
                capture_vs_openexisting["gt_other"] += 1

            info_h = classify_move(SpotEnv(players, tokens), current_player, dice, heuristic_move)
            capture_vs_openexisting["heur_total"] += 1
            if info_h["capture"]:
                capture_vs_openexisting["heur_capture"] += 1
            elif info_h["bring_out"]:
                capture_vs_openexisting["heur_open"] += 1
            else:
                capture_vs_openexisting["heur_other"] += 1

        if scenario == "home_entry" and is_valid:
            info = classify_move(SpotEnv(players, tokens), current_player, dice, gt_move)
            home_entry_stats["total"] += 1
            if info["home_entry"]:
                home_entry_stats["home_entry"] += 1
            else:
                home_entry_stats["other"] += 1

        if scenario == "mixed" and is_valid:
            env2 = SpotEnv(players, tokens)
            info = classify_move(env2, current_player, dice, gt_move)
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
            info = classify_move(SpotEnv(players, tokens), current_player, dice, gt_move)
            overshoot_stats["total"] += 1
            if info["new_pos"] > info["home_end"]:
                overshoot_stats["overshoot_move"] += 1
            else:
                overshoot_stats["other"] += 1

        if scenario == "safe" and is_valid:
            env2 = SpotEnv(players, tokens)
            info = classify_move(env2, current_player, dice, gt_move)
            safe_stats["total"] += 1
            if info["new_pos"] in env2.SAFE_SQUARES:
                safe_stats["safe"] += 1
            else:
                safe_stats["other"] += 1

        if scenario == "positional_roll6" and is_valid:
            info = classify_move(SpotEnv(players, tokens), current_player, dice, gt_move)
            roll6_stats["total"] += 1
            if info["bring_out"]:
                roll6_stats["bring_out"] += 1
            else:
                roll6_stats["move_existing"] += 1

        if scenario == "extra_turn" and is_valid:
            info = classify_move(SpotEnv(players, tokens), current_player, dice, gt_move)
            extra_turn_stats["total"] += 1
            if info["bring_out"]:
                extra_turn_stats["bring_out"] += 1
            else:
                extra_turn_stats["move_existing"] += 1

        if scenario == "positional_capture_on_6" and is_valid:
            info = classify_move(SpotEnv(players, tokens), current_player, dice, gt_move)
            capture_on_6_stats["total"] += 1
            if info["capture"]:
                capture_on_6_stats["capture"] += 1
            else:
                capture_on_6_stats["no_capture"] += 1

    for sc in summary:
        total = summary[sc]["total_all"]
        summary[sc]["gt_invalid_rate"] = summary[sc]["gt_invalid"] / total if total else 0.0

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
    }

    for pair_id, pair in grudge_pairs.items():
        if "a" not in pair or "b" not in pair:
            continue
        if not (pair["a"]["gt_valid"] and pair["b"]["gt_valid"]):
            continue
        grudge_report["pairs_total"] += 1

        if pair["a"]["gt_move"] != pair["b"]["gt_move"]:
            grudge_report["pairs_with_change"] += 1

        target = None
        for key in ["a", "b"]:
            hist = pair[key].get("history_text", "")
            if not hist:
                continue
            m = re.search(r"Player\\s+(\\d+)\\s+captured", hist)
            if m:
                target = int(m.group(1))
                break

        for key in ["a", "b"]:
            item = pair[key]
            env2 = SpotEnv(item["players"], {int(k): v for k, v in item["tokens"].items()})
            info = classify_move(env2, item["current_player"], item["dice"], item["gt_move"])
            is_grudge = "captured" in (item.get("history_text", "").lower())
            if is_grudge:
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
        "capture_rate_by_style": {"aggressive": 0.0, "avoid": 0.0, "forgiving": 0.0},
        "safe_rate_by_style": {"aggressive": 0.0, "avoid": 0.0, "forgiving": 0.0},
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
        if not (pair["a"]["gt_valid"] and pair["b"]["gt_valid"]):
            continue
        style_report["pairs_total"] += 1
        if pair["a"]["gt_move"] != pair["b"]["gt_move"]:
            style_report["pairs_with_change"] += 1

        for key in ["a", "b"]:
            item = pair[key]
            bucket = style_bucket(item["history_text"])
            if not bucket:
                continue
            env2 = SpotEnv(item["players"], {int(k): v for k, v in item["tokens"].items()})
            info = classify_move(env2, item["current_player"], item["dice"], item["gt_move"])
            style_stats[bucket]["total"] += 1
            if info["capture"]:
                style_stats[bucket]["capture"] += 1
            if info["new_pos"] in env2.SAFE_SQUARES:
                style_stats[bucket]["safe"] += 1

    if style_report["pairs_total"]:
        style_report["change_rate"] = style_report["pairs_with_change"] / style_report["pairs_total"]
    for k in style_stats:
        if style_stats[k]["total"]:
            style_report["capture_rate_by_style"][k] = style_stats[k]["capture"] / style_stats[k]["total"]
            style_report["safe_rate_by_style"][k] = style_stats[k]["safe"] / style_stats[k]["total"]

    style_path = os.path.join(out_dir, "opponent_style_report.json")
    with open(style_path, "w") as f:
        json.dump(style_report, f, ensure_ascii=True, indent=2)

    profile = {
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

    if capture_vs_home["gt_total"] > 0:
        profile["capture_vs_home"] = {
            "capture_rate": capture_vs_home["gt_capture"] / capture_vs_home["gt_total"],
            "home_entry_rate": capture_vs_home["gt_home"] / capture_vs_home["gt_total"],
            "other_rate": capture_vs_home["gt_other"] / capture_vs_home["gt_total"],
        }
    if capture_vs_openexisting["gt_total"] > 0:
        profile["capture_vs_openexisting"] = {
            "capture_rate": capture_vs_openexisting["gt_capture"] / capture_vs_openexisting["gt_total"],
            "open_rate": capture_vs_openexisting["gt_open"] / capture_vs_openexisting["gt_total"],
            "other_rate": capture_vs_openexisting["gt_other"] / capture_vs_openexisting["gt_total"],
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

    profile_path = os.path.join(out_dir, "behavior_profile.txt")
    with open(profile_path, "w") as f:
        f.write("Behavior Profile (Category) - GT Agent\n")
        f.write("=====================================\n\n")
        f.write(f"Total spots: {sum(v['total_all'] for v in summary.values())}\n")
        f.write(f"Valid spots: {sum(v['total_valid'] for v in summary.values())}\n\n")
        total_spots = sum(v["total_all"] for v in summary.values())
        total_invalid = sum(v["gt_invalid"] for v in summary.values())
        if total_spots:
            f.write(f"gt_invalid_rate: {total_invalid / total_spots:.3f}\n\n")
        for k in [
            "capture_vs_home",
            "capture_vs_openexisting",
            "home_entry",
            "mixed",
            "overshoot",
            "safe",
            "roll6",
            "extra_turn",
            "capture_on_6",
        ]:
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

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {results_path}")
    print(f"Wrote: {profile_path}")
    return profile


def main():
    parser = argparse.ArgumentParser(description="Run all spot scenarios with GT agent (and compare with heuristic).")
    parser.add_argument(
        "--spots-glob",
        default="spots_40/spots_*.json",
        help="Glob for spot files (default: spots_40/spots_*.json)",
    )
    parser.add_argument(
        "--out-dir",
        default="spot_results_gt",
        help="Output directory root for category results",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Expectiminimax depth for GT agent (2-player spots only).",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.spots_glob))
    if not files:
        raise SystemExit(f"No spot files match {args.spots_glob}")

    spots = load_spots(files)
    by_category = defaultdict(list)
    for spot in spots:
        by_category[spot.get("scenario", "unknown")].append(spot)

    overall_profiles = {}
    for cat in sorted(by_category.keys()):
        cat_out = os.path.join(args.out_dir, cat)
        overall_profiles[cat] = process_spots(by_category[cat], cat_out, args.depth)

    overall_path = os.path.join(args.out_dir, "overall_behavior_profile.txt")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(overall_path, "w") as f:
        f.write("Overall Behavior Profile - GT Agent\n")
        f.write("===================================\n\n")
        f.write(f"depth={args.depth}\n\n")
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
            if prof.get("summary"):
                total_spots = sum(v["total_all"] for v in prof["summary"].values())
                total_invalid = sum(v["gt_invalid"] for v in prof["summary"].values())
                if total_spots:
                    f.write(f"  gt_invalid_rate={total_invalid / total_spots:.3f}\n")
            f.write("\n")
    print(f"Wrote: {overall_path}")


if __name__ == "__main__":
    main()
