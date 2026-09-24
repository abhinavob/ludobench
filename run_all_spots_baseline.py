import argparse
import glob
import json
import os
import random
import re
import shutil
from collections import defaultdict

from agents import HeuristicAgent, RandomAgent
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


def make_agent(name):
    name = name.lower()
    if name == "random":
        return RandomAgent()
    if name == "heuristic":
        return HeuristicAgent()
    raise ValueError(f"Unsupported baseline agent: {name}")


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


def process_spots(spots, out_dir, agent_name):
    ensure_clean_dir(out_dir)

    results = []
    summary = defaultdict(
        lambda: {
            "total_all": 0,
            "total_valid": 0,
            "agent_equals_heuristic": 0,
            "agent_differs": 0,
            "agent_invalid": 0,
        }
    )

    agent = make_agent(agent_name)
    heuristic_agent = HeuristicAgent()
    grudge_pairs = {}

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

        agent_move = agent.select_move(env, current_player, dice, legal_moves)
        heuristic_move = heuristic_agent.select_move(env, current_player, dice, legal_moves)

        is_valid = agent_move in legal_moves
        equals = agent_move == heuristic_move

        summary[scenario]["total_all"] += 1
        if not is_valid:
            summary[scenario]["agent_invalid"] += 1
        else:
            summary[scenario]["total_valid"] += 1
            if equals:
                summary[scenario]["agent_equals_heuristic"] += 1
            else:
                summary[scenario]["agent_differs"] += 1

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
                "agent_move": agent_move,
                "heuristic_move": heuristic_move,
                "agent_valid": is_valid,
                "agent_equals_heuristic": equals,
                "agent_name": agent_name,
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
                "agent_move": agent_move,
                "agent_valid": is_valid,
                "players": players,
                "current_player": current_player,
                "dice": dice,
                "tokens": tokens,
            }

    for sc in summary:
        total = summary[sc]["total_all"]
        summary[sc]["agent_invalid_rate"] = summary[sc]["agent_invalid"] / total if total else 0.0

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

    for pair in grudge_pairs.values():
        if "a" not in pair or "b" not in pair:
            continue
        if not (pair["a"]["agent_valid"] and pair["b"]["agent_valid"]):
            continue
        grudge_report["pairs_total"] += 1

        if pair["a"]["agent_move"] != pair["b"]["agent_move"]:
            grudge_report["pairs_with_change"] += 1

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
            env2 = SpotEnv(item["players"], {int(k): v for k, v in item["tokens"].items()})
            info = classify_move(env2, item["current_player"], item["dice"], item["agent_move"])
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

    profile_path = os.path.join(out_dir, "behavior_profile.txt")
    with open(profile_path, "w") as f:
        f.write(f"Behavior Profile (Category) - {agent_name}\n")
        f.write("=====================================\n\n")
        f.write(f"Total spots: {sum(v['total_all'] for v in summary.values())}\n")
        f.write(f"Valid spots: {sum(v['total_valid'] for v in summary.values())}\n\n")
        total_spots = sum(v["total_all"] for v in summary.values())
        total_invalid = sum(v["agent_invalid"] for v in summary.values())
        if total_spots:
            f.write(f"agent_invalid_rate: {total_invalid / total_spots:.3f}\n\n")
        if grudge_report["pairs_total"] > 0:
            f.write("grudge:\n")
            f.write(f"  change_rate: {grudge_report['change_rate']:.3f}\n")
            f.write(f"  retaliation_grudge_rate: {grudge_report['retaliation_grudge_rate']:.3f}\n")
            f.write(f"  retaliation_noconflict_rate: {grudge_report['retaliation_noconflict_rate']:.3f}\n")

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {results_path}")
    print(f"Wrote: {profile_path}")


def main():
    parser = argparse.ArgumentParser(description="Run all spot scenarios with a baseline agent.")
    parser.add_argument(
        "--agent",
        choices=["random", "heuristic"],
        required=True,
        help="Baseline agent to evaluate.",
    )
    parser.add_argument(
        "--spots-glob",
        default="spots_40/spots_*.json",
        help="Glob for spot files (default: spots_40/spots_*.json)",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory root for category results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used by RandomAgent runs.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    files = sorted(glob.glob(args.spots_glob))
    if not files:
        raise SystemExit(f"No spot files match {args.spots_glob}")

    spots = load_spots(files)
    by_category = defaultdict(list)
    for spot in spots:
        by_category[spot.get("scenario", "unknown")].append(spot)

    ensure_clean_dir(args.out_dir)
    for cat in sorted(by_category.keys()):
        cat_out = os.path.join(args.out_dir, cat)
        process_spots(by_category[cat], cat_out, args.agent)


if __name__ == "__main__":
    main()
