import argparse
import csv
import json
import os


PERSONAS = ["aggressive", "greedy", "safe", "unforgiving", "none"]
SAFE_SQUARES = {0, 8, 13, 21, 26, 34, 39, 47}
STARTS = {0: 0, 1: 13, 2: 26, 3: 39}
BOARD_SIZE = 52
HOME_START_GLOBAL = 52
HOME_LEN = 6


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def get_summary_entry(root, category, scenario):
    path = os.path.join(root, category, "spot_summary.json")
    data = load_json(path)
    if not data:
        return None
    return data.get(scenario)


def classify_move(record):
    player = int(record["current_player"])
    players = [int(p) for p in record["players"]]
    tokens = {int(k): v for k, v in record["tokens"].items()}
    dice = int(record["dice"])
    token_idx = int(record["agent_move"])

    start = STARTS[player]
    home_start = HOME_START_GLOBAL + player * HOME_LEN
    home_end = home_start + HOME_LEN - 1
    old_pos = tokens[player][token_idx]

    if old_pos == -1:
        new_pos = start
    elif 0 <= old_pos < BOARD_SIZE:
        rel_pos = (old_pos - start) % BOARD_SIZE
        new_rel = rel_pos + dice
        if new_rel < BOARD_SIZE:
            new_pos = (start + new_rel) % BOARD_SIZE
        else:
            home_step = new_rel - BOARD_SIZE
            new_pos = home_start + home_step - 1
    else:
        new_pos = old_pos + dice

    capture = False
    captured_player = None
    if 0 <= new_pos < BOARD_SIZE and new_pos not in SAFE_SQUARES:
        for opp in players:
            if opp == player:
                continue
            if new_pos in tokens[opp]:
                capture = True
                captured_player = opp
                break

    return {
        "new_pos": new_pos,
        "home_start": home_start,
        "home_end": home_end,
        "capture": capture,
        "captured_player": captured_player,
        "home_entry": new_pos >= home_start,
        "bring_out": old_pos == -1 and new_pos == start,
        "safe": new_pos in SAFE_SQUARES,
    }


def metric_rows():
    return [
        ("blocked", "llm_invalid_rate"),
        ("blocked", "block_rate"),
        ("capture_vs_home_finish", "llm_invalid_rate"),
        ("capture_vs_home_finish", "capture_rate"),
        ("capture_vs_home_finish", "home_finish_rate"),
        ("capture_vs_home", "llm_invalid_rate"),
        ("capture_vs_home", "capture_rate"),
        ("capture_vs_home", "home_entry_rate"),
        ("capture_vs_openexisting", "llm_invalid_rate"),
        ("capture_vs_openexisting", "capture_rate"),
        ("capture_vs_openexisting", "open_rate"),
        ("capture_vs_safe", "llm_invalid_rate"),
        ("capture_vs_safe", "capture_rate"),
        ("capture_vs_safe", "safe_rate"),
        ("capture", "llm_invalid_rate"),
        ("capture", "capture_rate"),
        ("overshoot", "llm_invalid_rate"),
        ("overshoot", "overshoot_rate"),
        ("safe", "llm_invalid_rate"),
        ("safe", "safe_rate"),
        ("safe_vs_openexisting", "llm_invalid_rate"),
        ("safe_vs_openexisting", "safe_rate"),
        ("safe_vs_openexisting", "open_rate"),
        ("extra_turn", "llm_invalid_rate"),
        ("extra_turn", "bring_out_rate"),
        ("extra_turn", "move_existing_rate"),
        ("home_entry", "llm_invalid_rate"),
        ("home_entry", "home_entry_rate"),
        ("grudge", "change_rate"),
        ("grudge", "retaliation_grudge_rate"),
        ("grudge", "retaliation_noconflict_rate"),
    ]


def compute_metrics(root):
    out = {}

    s = get_summary_entry(root, "blocked", "blocked")
    if s:
        out[("blocked", "llm_invalid_rate")] = s.get("agent_invalid_rate", 0.0)
        total_valid = s.get("total_valid", 0)
        out[("blocked", "block_rate")] = (
            s.get("agent_equals_heuristic", 0) / total_valid if total_valid else 0.0
        )

    scenario_metric_map = [
        ("capture_vs_home_finish", "capture_rate", "home_finish_rate"),
        ("capture_vs_home", "capture_rate", "home_entry_rate"),
        ("capture_vs_openexisting", "capture_rate", "open_rate"),
        ("capture_vs_safe", "capture_rate", "safe_rate"),
        ("safe_vs_openexisting", "safe_rate", "open_rate"),
    ]
    for scenario, metric_a, metric_b in scenario_metric_map:
        s = get_summary_entry(root, scenario, scenario)
        if s:
            out[(scenario, "llm_invalid_rate")] = s.get("agent_invalid_rate", 0.0)
        results = load_json(os.path.join(root, scenario, "spot_results.json")) or []
        valid = [r for r in results if r.get("scenario") == scenario and r.get("agent_valid")]
        if not valid:
            continue
        count_a = 0
        count_b = 0
        for r in valid:
            info = classify_move(r)
            if metric_a == "capture_rate" and info["capture"]:
                count_a += 1
            if metric_a == "safe_rate" and info["safe"]:
                count_a += 1
            if metric_b == "home_finish_rate" and info["new_pos"] == info["home_end"]:
                count_b += 1
            if metric_b == "home_entry_rate" and info["home_entry"]:
                count_b += 1
            if metric_b == "open_rate" and info["bring_out"]:
                count_b += 1
            if metric_b == "safe_rate" and info["safe"]:
                count_b += 1
        out[(scenario, metric_a)] = count_a / len(valid)
        out[(scenario, metric_b)] = count_b / len(valid)

    for scenario, metric in [("capture", "capture_rate"), ("overshoot", "overshoot_rate"), ("safe", "safe_rate"), ("home_entry", "home_entry_rate")]:
        s = get_summary_entry(root, scenario, scenario)
        if s:
            out[(scenario, "llm_invalid_rate")] = s.get("agent_invalid_rate", 0.0)
        results = load_json(os.path.join(root, scenario, "spot_results.json")) or []
        valid = [r for r in results if r.get("scenario") == scenario and r.get("agent_valid")]
        if not valid:
            continue
        hits = 0
        for r in valid:
            info = classify_move(r)
            if metric == "capture_rate" and info["capture"]:
                hits += 1
            if metric == "overshoot_rate" and info["new_pos"] > info["home_end"]:
                hits += 1
            if metric == "safe_rate" and info["safe"]:
                hits += 1
            if metric == "home_entry_rate" and info["home_entry"]:
                hits += 1
        out[(scenario, metric)] = hits / len(valid)

    s = get_summary_entry(root, "extra_turn", "extra_turn")
    if s:
        out[("extra_turn", "llm_invalid_rate")] = s.get("agent_invalid_rate", 0.0)
    results = load_json(os.path.join(root, "extra_turn", "spot_results.json")) or []
    valid = [r for r in results if r.get("scenario") == "extra_turn" and r.get("agent_valid")]
    if valid:
        bring_out = sum(1 for r in valid if classify_move(r)["bring_out"])
        out[("extra_turn", "bring_out_rate")] = bring_out / len(valid)
        out[("extra_turn", "move_existing_rate")] = 1.0 - out[("extra_turn", "bring_out_rate")]

    gr = load_json(os.path.join(root, "grudge", "grudge_report.json")) or {}
    if gr:
        out[("grudge", "change_rate")] = gr.get("change_rate", 0.0)
        out[("grudge", "retaliation_grudge_rate")] = gr.get("retaliation_grudge_rate", 0.0)
        out[("grudge", "retaliation_noconflict_rate")] = gr.get("retaliation_noconflict_rate", 0.0)

    return out


def write_csv(path, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "metric"] + PERSONAS)
        for category, metric in metric_rows():
            value = metrics.get((category, metric), "")
            writer.writerow([category, metric] + [value] * len(PERSONAS))


def main():
    parser = argparse.ArgumentParser(description="Build persona-style comparison CSV for a fixed non-persona agent.")
    parser.add_argument("--root", required=True, help="Category result root, e.g. spot_results_gt")
    parser.add_argument("--out-csv", required=True, help="Output CSV path")
    args = parser.parse_args()

    metrics = compute_metrics(args.root)
    write_csv(args.out_csv, metrics)
    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
