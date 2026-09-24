import argparse
import csv
import json
import os
import tempfile

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def classify_move(record, move_key):
    player = int(record["current_player"])
    tokens = {int(k): v for k, v in record["tokens"].items()}
    dice = int(record["dice"])
    token_idx = int(record[move_key])

    board_size = 52
    home_start_global = 52
    home_len = 6
    safe_squares = {0, 8, 13, 21, 26, 34, 39, 47}
    starts = {0: 0, 1: 13, 2: 26, 3: 39}

    start = starts[player]
    home_start = home_start_global + player * home_len
    home_end = home_start + home_len - 1
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

    capture = False
    if 0 <= new_pos < board_size and new_pos not in safe_squares:
        for pid, p_tokens in tokens.items():
            if pid == player:
                continue
            if new_pos in p_tokens:
                capture = True
                break

    return {
        "new_pos": new_pos,
        "home_start": home_start,
        "home_end": home_end,
        "capture": capture,
        "home_entry": new_pos >= home_start,
        "safe": new_pos in safe_squares,
        "bring_out": old_pos == -1 and new_pos == start,
    }


def scenario_action_label(scenario, info):
    if scenario == "capture_vs_home_finish":
        if info["capture"]:
            return "capture"
        if info["new_pos"] == info["home_end"]:
            return "home_finish"
        return "other"
    if scenario == "capture_vs_home":
        if info["capture"]:
            return "capture"
        if info["home_entry"]:
            return "home_entry"
        return "other"
    if scenario == "capture_vs_openexisting":
        if info["capture"]:
            return "capture"
        if info["bring_out"]:
            return "open"
        return "other"
    if scenario == "capture_vs_safe":
        if info["capture"]:
            return "capture"
        if info["safe"]:
            return "safe"
        return "other"
    if scenario == "capture":
        return "capture" if info["capture"] else "other"
    if scenario in ("safe", "safe_vs_openexisting"):
        if info["safe"]:
            return "safe"
        if info["bring_out"]:
            return "open"
        return "other"
    if scenario == "extra_turn":
        return "bring_out" if info["bring_out"] else "move_existing"
    if scenario == "home_entry":
        return "home_entry" if info["home_entry"] else "other"
    if scenario == "overshoot":
        return "overshoot" if info["new_pos"] > info["home_end"] else "other"
    return "other"


def compare_category(llm_rows, gt_rows):
    by_id_llm = {r["id"]: r for r in llm_rows}
    by_id_gt = {r["id"]: r for r in gt_rows}
    common_ids = sorted(set(by_id_llm.keys()) & set(by_id_gt.keys()))

    out = {
        "total_common": len(common_ids),
        "valid_both": 0,
        "agreement_on_valid": 0,
        "agreement_rate": 0.0,
        "llm_invalid_rate": 0.0,
        "gt_invalid_rate": 0.0,
        "llm_2p_share": 0.0,
        "gt_expectiminimax_share": 0.0,
    }
    action_counts = {}

    if not common_ids:
        return out, action_counts

    llm_invalid = 0
    gt_invalid = 0
    llm_2p = 0
    gt_2p_mode = 0

    for sid in common_ids:
        l = by_id_llm[sid]
        g = by_id_gt[sid]
        scenario = l.get("scenario", "")

        if len(l.get("players", [])) == 2:
            llm_2p += 1
        if g.get("gt_mode") == "expectiminimax_2p":
            gt_2p_mode += 1

        l_valid = bool(l.get("llm_valid"))
        g_valid = bool(g.get("gt_valid"))
        if not l_valid:
            llm_invalid += 1
        if not g_valid:
            gt_invalid += 1
        if not (l_valid and g_valid):
            continue

        out["valid_both"] += 1
        if int(l["llm_move"]) == int(g["gt_move"]):
            out["agreement_on_valid"] += 1

        li = classify_move(l, "llm_move")
        gi = classify_move(g, "gt_move")
        la = scenario_action_label(scenario, li)
        ga = scenario_action_label(scenario, gi)
        key = f"{la}->{ga}"
        action_counts[key] = action_counts.get(key, 0) + 1

    out["agreement_rate"] = out["agreement_on_valid"] / out["valid_both"] if out["valid_both"] else 0.0
    out["llm_invalid_rate"] = llm_invalid / len(common_ids)
    out["gt_invalid_rate"] = gt_invalid / len(common_ids)
    out["llm_2p_share"] = llm_2p / len(common_ids)
    out["gt_expectiminimax_share"] = gt_2p_mode / len(common_ids)
    return out, action_counts


def plot_agreement(rows, out_png):
    if not rows:
        return
    cats = [r["category"] for r in rows]
    vals = [r["agreement_rate"] for r in rows]
    try:
        plt.figure(figsize=(max(9, len(cats) * 0.8), 5))
        plt.bar(cats, vals, color="#2f6bff")
        plt.ylim(0, 1)
        plt.ylabel("LLM vs GT Agreement (valid spots)")
        plt.title("LLM vs GT Agreement by Category")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        out_dir = os.path.dirname(out_png)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_png, dpi=180)
        plt.close()
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Compare LLM spot results vs GT spot results (category-wise).")
    parser.add_argument(
        "--llm-root",
        default="spot_results_40/none",
        help="LLM category root (default: spot_results_40/none)",
    )
    parser.add_argument(
        "--gt-root",
        default="spot_results_gt",
        help="GT category root (default: spot_results_gt)",
    )
    parser.add_argument(
        "--out-csv",
        default="spot_results_gt/llm_vs_gt_comparison.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--out-plot",
        default="spot_results_gt/llm_vs_gt_agreement.png",
        help="Output agreement plot path",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate agreement plot (disabled by default).",
    )
    parser.add_argument(
        "--out-actions-json",
        default="spot_results_gt/llm_vs_gt_action_transitions.json",
        help="Output action transition counts path",
    )
    args = parser.parse_args()

    categories = sorted(
        set(
            d
            for d in os.listdir(args.llm_root)
            if os.path.isdir(os.path.join(args.llm_root, d)) and os.path.isdir(os.path.join(args.gt_root, d))
        )
    )

    rows = []
    all_actions = {}
    for cat in categories:
        llm_path = os.path.join(args.llm_root, cat, "spot_results.json")
        gt_path = os.path.join(args.gt_root, cat, "spot_results.json")
        llm_rows = load_json(llm_path) or []
        gt_rows = load_json(gt_path) or []
        if not llm_rows or not gt_rows:
            continue
        comp, actions = compare_category(llm_rows, gt_rows)
        comp["category"] = cat
        rows.append(comp)
        all_actions[cat] = actions

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "total_common",
                "valid_both",
                "agreement_on_valid",
                "agreement_rate",
                "llm_invalid_rate",
                "gt_invalid_rate",
                "llm_2p_share",
                "gt_expectiminimax_share",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    actions_dir = os.path.dirname(args.out_actions_json)
    if actions_dir:
        os.makedirs(actions_dir, exist_ok=True)
    with open(args.out_actions_json, "w") as f:
        json.dump(all_actions, f, indent=2)

    plot_ok = False
    if args.plot:
        plot_ok = plot_agreement(rows, args.out_plot)
    print(f"Wrote: {args.out_csv}")
    if args.plot and plot_ok:
        print(f"Wrote: {args.out_plot}")
    elif args.plot:
        print(f"Skipped plot: {args.out_plot}")
    print(f"Wrote: {args.out_actions_json}")


if __name__ == "__main__":
    main()
