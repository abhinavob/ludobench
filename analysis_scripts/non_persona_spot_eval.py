import argparse
import csv
import json
import math
import os
import statistics
import tempfile

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SAFE_SQUARES = {0, 8, 13, 21, 26, 34, 39, 47}
STARTS = {0: 0, 1: 13, 2: 26, 3: 39}
BOARD_SIZE = 52
HOME_START_GLOBAL = 52
HOME_LEN = 6


def load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def discover_models(results_root):
    models = []
    if not os.path.isdir(results_root):
        return models
    for name in sorted(os.listdir(results_root)):
        p = os.path.join(results_root, name)
        if not os.path.isdir(p):
            continue
        if name.startswith("."):
            continue
        if name == "model_wise_figures":
            continue
        models.append(name)
    return models


def resolve_persona_root(results_root, model, persona):
    a = os.path.join(results_root, model, "personas", persona)
    b = os.path.join(results_root, model, persona)
    if os.path.isdir(a):
        return a
    if os.path.isdir(b):
        return b
    return None


def classify_move(record, move_key):
    player = int(record["current_player"])
    tokens = {int(k): v for k, v in record["tokens"].items()}
    dice = int(record["dice"])
    token_idx = int(record[move_key])

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
    if 0 <= new_pos < BOARD_SIZE and new_pos not in SAFE_SQUARES:
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
        "safe": new_pos in SAFE_SQUARES,
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
    if scenario == "blocked":
        return "legal_choice"
    return "other"


def compare_category(llm_rows, gt_rows):
    by_id_llm = {r["id"]: r for r in llm_rows}
    by_id_gt = {r["id"]: r for r in gt_rows}
    common_ids = sorted(set(by_id_llm.keys()) & set(by_id_gt.keys()))

    out = {
        "total_common": len(common_ids),
        "valid_both": 0,
        "move_agree_count": 0,
        "action_agree_count": 0,
        "llm_invalid_count": 0,
        "gt_invalid_count": 0,
        "move_agreement_rate": 0.0,
        "action_agreement_rate": 0.0,
        "llm_invalid_rate": 0.0,
        "gt_invalid_rate": 0.0,
        "llm_valid_rate": 0.0,
        "valid_both_rate": 0.0,
    }
    transitions = {}

    if not common_ids:
        return out, transitions

    for sid in common_ids:
        l = by_id_llm[sid]
        g = by_id_gt[sid]
        scenario = l.get("scenario", "")

        l_valid = bool(l.get("llm_valid"))
        g_valid = bool(g.get("gt_valid"))
        if not l_valid:
            out["llm_invalid_count"] += 1
        if not g_valid:
            out["gt_invalid_count"] += 1
        if not (l_valid and g_valid):
            continue

        out["valid_both"] += 1
        if int(l["llm_move"]) == int(g["gt_move"]):
            out["move_agree_count"] += 1

        li = classify_move(l, "llm_move")
        gi = classify_move(g, "gt_move")
        la = scenario_action_label(scenario, li)
        ga = scenario_action_label(scenario, gi)
        if la == ga:
            out["action_agree_count"] += 1
        key = f"{la}->{ga}"
        transitions[key] = transitions.get(key, 0) + 1

    total = out["total_common"]
    out["move_agreement_rate"] = out["move_agree_count"] / out["valid_both"] if out["valid_both"] else 0.0
    out["action_agreement_rate"] = out["action_agree_count"] / out["valid_both"] if out["valid_both"] else 0.0
    out["llm_invalid_rate"] = out["llm_invalid_count"] / total if total else 0.0
    out["gt_invalid_rate"] = out["gt_invalid_count"] / total if total else 0.0
    out["llm_valid_rate"] = 1.0 - out["llm_invalid_rate"]
    out["valid_both_rate"] = out["valid_both"] / total if total else 0.0
    return out, transitions


def write_csv(path, rows, fieldnames):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def heatmap(models, categories, value_map, title, out_png, cmap="viridis", vmin=0.0, vmax=1.0):
    if not models or not categories:
        return
    matrix = []
    for m in models:
        row = []
        for c in categories:
            row.append(value_map.get((m, c), float("nan")))
        matrix.append(row)

    plt.figure(figsize=(max(10, len(categories) * 0.7), max(4, len(models) * 0.45)))
    im = plt.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=12)
    plt.xticks(range(len(categories)), categories, rotation=35, ha="right", fontsize=13)
    plt.yticks(range(len(models)), models, fontsize=13)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_scatter(model_rows, out_png):
    if not model_rows:
        return
    xs = [r["llm_valid_rate"] for r in model_rows]
    ys = [r["move_agreement_rate"] for r in model_rows]
    labels = [r["model"] for r in model_rows]

    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, s=70)
    for x, y, label in zip(xs, ys, labels):
        plt.text(x + 0.003, y + 0.003, label, fontsize=8)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("LLM Validity Rate (1 - invalid)")
    plt.ylabel("Move Agreement vs GT (valid spots)")
    plt.title("Model Reliability vs GT Agreement (none persona, 40-spot set)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_category_bars(category_rows, out_png):
    if not category_rows:
        return
    cats = [r["category"] for r in category_rows]
    move = [r["move_agreement_mean"] for r in category_rows]
    action = [r["action_agreement_mean"] for r in category_rows]
    invalid = [r["llm_invalid_mean"] for r in category_rows]
    x = list(range(len(cats)))
    w = 0.26

    plt.figure(figsize=(max(11, len(cats) * 0.75), 5))
    plt.bar([i - w for i in x], move, width=w, label="Move agreement")
    plt.bar(x, action, width=w, label="Action agreement")
    plt.bar([i + w for i in x], invalid, width=w, label="LLM invalid rate")
    plt.ylim(0, 1)
    plt.xticks(x, cats, rotation=35, ha="right")
    plt.ylabel("Rate")
    plt.title("Scenario Difficulty/Quality Profile (across models)")
    plt.legend()
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Non-persona (none only) spot-level LLM analysis vs GT on results_for_40."
    )
    parser.add_argument("--results-root", default="results_for_40")
    parser.add_argument("--gt-root", default="spot_results_gt")
    parser.add_argument("--persona", default="none")
    parser.add_argument("--out-dir", default="results_for_40/non_persona_eval")
    args = parser.parse_args()

    models = discover_models(args.results_root)
    per_model_category_rows = []
    transitions_by_model = {}

    for model in models:
        llm_root = resolve_persona_root(args.results_root, model, args.persona)
        if not llm_root:
            continue

        categories = sorted(
            d
            for d in os.listdir(llm_root)
            if os.path.isdir(os.path.join(llm_root, d)) and os.path.isdir(os.path.join(args.gt_root, d))
        )
        model_transitions = {}
        for cat in categories:
            llm_path = os.path.join(llm_root, cat, "spot_results.json")
            gt_path = os.path.join(args.gt_root, cat, "spot_results.json")
            llm_rows = load_json(llm_path) or []
            gt_rows = load_json(gt_path) or []
            if not llm_rows or not gt_rows:
                continue
            comp, trans = compare_category(llm_rows, gt_rows)
            comp["model"] = model
            comp["category"] = cat
            per_model_category_rows.append(comp)
            model_transitions[cat] = trans
        transitions_by_model[model] = model_transitions

    if not per_model_category_rows:
        raise SystemExit("No overlapping LLM/GT category data found.")

    # Model-level aggregate (weighted by total_common)
    by_model = {}
    for r in per_model_category_rows:
        m = r["model"]
        a = by_model.setdefault(
            m,
            {
                "model": m,
                "total_common": 0,
                "valid_both": 0,
                "move_agree_count": 0,
                "action_agree_count": 0,
                "llm_invalid_count": 0,
                "gt_invalid_count": 0,
            },
        )
        a["total_common"] += r["total_common"]
        a["valid_both"] += r["valid_both"]
        a["move_agree_count"] += r["move_agree_count"]
        a["action_agree_count"] += r["action_agree_count"]
        a["llm_invalid_count"] += r["llm_invalid_count"]
        a["gt_invalid_count"] += r["gt_invalid_count"]

    model_rows = []
    for m in sorted(by_model.keys()):
        a = by_model[m]
        total = a["total_common"]
        valid = a["valid_both"]
        row = {
            "model": m,
            "total_common": total,
            "valid_both": valid,
            "llm_invalid_rate": (a["llm_invalid_count"] / total) if total else 0.0,
            "gt_invalid_rate": (a["gt_invalid_count"] / total) if total else 0.0,
            "llm_valid_rate": 1.0 - ((a["llm_invalid_count"] / total) if total else 0.0),
            "valid_both_rate": (valid / total) if total else 0.0,
            "move_agreement_rate": (a["move_agree_count"] / valid) if valid else 0.0,
            "action_agreement_rate": (a["action_agree_count"] / valid) if valid else 0.0,
        }
        model_rows.append(row)

    # Category-level aggregate (mean/std across models)
    by_category = {}
    for r in per_model_category_rows:
        c = r["category"]
        arr = by_category.setdefault(c, {"move": [], "action": [], "invalid": [], "valid_both": []})
        arr["move"].append(r["move_agreement_rate"])
        arr["action"].append(r["action_agreement_rate"])
        arr["invalid"].append(r["llm_invalid_rate"])
        arr["valid_both"].append(r["valid_both_rate"])

    category_rows = []
    for c in sorted(by_category.keys()):
        arr = by_category[c]
        category_rows.append(
            {
                "category": c,
                "models_count": len(arr["move"]),
                "move_agreement_mean": statistics.mean(arr["move"]) if arr["move"] else 0.0,
                "move_agreement_std": statistics.pstdev(arr["move"]) if len(arr["move"]) > 1 else 0.0,
                "action_agreement_mean": statistics.mean(arr["action"]) if arr["action"] else 0.0,
                "action_agreement_std": statistics.pstdev(arr["action"]) if len(arr["action"]) > 1 else 0.0,
                "llm_invalid_mean": statistics.mean(arr["invalid"]) if arr["invalid"] else 0.0,
                "llm_invalid_std": statistics.pstdev(arr["invalid"]) if len(arr["invalid"]) > 1 else 0.0,
                "valid_both_mean": statistics.mean(arr["valid_both"]) if arr["valid_both"] else 0.0,
            }
        )

    model_cat_csv = os.path.join(args.out_dir, "per_model_category_metrics.csv")
    model_csv = os.path.join(args.out_dir, "model_overall_summary.csv")
    cat_csv = os.path.join(args.out_dir, "category_overall_summary.csv")
    trans_json = os.path.join(args.out_dir, "action_transitions_by_model.json")
    gt_align_long_csv = os.path.join(args.out_dir, "gt_alignment_by_model_category.csv")
    gt_align_matrix_csv = os.path.join(args.out_dir, "gt_alignment_matrix.csv")
    gt_align_overall_csv = os.path.join(args.out_dir, "gt_alignment_overall.csv")

    write_csv(
        model_cat_csv,
        sorted(per_model_category_rows, key=lambda x: (x["model"], x["category"])),
        [
            "model",
            "category",
            "total_common",
            "valid_both",
            "move_agree_count",
            "action_agree_count",
            "llm_invalid_count",
            "gt_invalid_count",
            "move_agreement_rate",
            "action_agreement_rate",
            "llm_invalid_rate",
            "gt_invalid_rate",
            "llm_valid_rate",
            "valid_both_rate",
        ],
    )
    write_csv(
        gt_align_long_csv,
        sorted(
            [
                {
                    "model": r["model"],
                    "category": r["category"],
                    "gt_alignment_score": r["move_agreement_rate"],
                    "valid_both": r["valid_both"],
                    "total_common": r["total_common"],
                }
                for r in per_model_category_rows
            ],
            key=lambda x: (x["model"], x["category"]),
        ),
        ["model", "category", "gt_alignment_score", "valid_both", "total_common"],
    )
    write_csv(
        model_csv,
        sorted(model_rows, key=lambda x: x["move_agreement_rate"], reverse=True),
        [
            "model",
            "total_common",
            "valid_both",
            "llm_invalid_rate",
            "gt_invalid_rate",
            "llm_valid_rate",
            "valid_both_rate",
            "move_agreement_rate",
            "action_agreement_rate",
        ],
    )
    write_csv(
        gt_align_overall_csv,
        sorted(
            [
                {
                    "model": r["model"],
                    "gt_alignment_score": r["move_agreement_rate"],
                    "valid_both": r["valid_both"],
                    "total_common": r["total_common"],
                }
                for r in model_rows
            ],
            key=lambda x: x["gt_alignment_score"],
            reverse=True,
        ),
        ["model", "gt_alignment_score", "valid_both", "total_common"],
    )
    write_csv(
        cat_csv,
        sorted(category_rows, key=lambda x: x["move_agreement_mean"]),
        [
            "category",
            "models_count",
            "move_agreement_mean",
            "move_agreement_std",
            "action_agreement_mean",
            "action_agreement_std",
            "llm_invalid_mean",
            "llm_invalid_std",
            "valid_both_mean",
        ],
    )
    # GT alignment matrix in paper-friendly wide format (rows=model, cols=category).
    ensure_dir(os.path.dirname(gt_align_matrix_csv))
    with open(gt_align_matrix_csv, "w", newline="") as f:
        category_order = sorted({r["category"] for r in per_model_category_rows})
        w = csv.writer(f)
        w.writerow(["model"] + category_order)
        row_map = {}
        for r in per_model_category_rows:
            row_map.setdefault(r["model"], {})[r["category"]] = r["move_agreement_rate"]
        for model in sorted(row_map.keys()):
            w.writerow([model] + [row_map[model].get(cat, "") for cat in category_order])
    ensure_dir(os.path.dirname(trans_json))
    with open(trans_json, "w") as f:
        json.dump(transitions_by_model, f, indent=2)

    # Figures
    model_order = [r["model"] for r in sorted(model_rows, key=lambda x: x["move_agreement_rate"], reverse=True)]
    category_order = sorted({r["category"] for r in per_model_category_rows})
    move_map = {(r["model"], r["category"]): r["move_agreement_rate"] for r in per_model_category_rows}
    invalid_map = {(r["model"], r["category"]): r["llm_invalid_rate"] for r in per_model_category_rows}
    action_map = {(r["model"], r["category"]): r["action_agreement_rate"] for r in per_model_category_rows}

    heatmap(
        model_order,
        category_order,
        move_map,
        "GT Alignment Score (Move Agreement) by Model x Category",
        os.path.join(args.out_dir, "heatmap_gt_alignment_score.png"),
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
    )
    # Keep backward-compatible filename too.
    heatmap(
        model_order,
        category_order,
        move_map,
        "Move Agreement vs GT (none persona, results_for_40)",
        os.path.join(args.out_dir, "heatmap_move_agreement.png"),
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
    )
    heatmap(
        model_order,
        category_order,
        invalid_map,
        "LLM Invalid Rate by Scenario (none persona, results_for_40)",
        os.path.join(args.out_dir, "heatmap_llm_invalid_rate.png"),
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
    )
    heatmap(
        model_order,
        category_order,
        action_map,
        "Action Agreement vs GT (none persona, results_for_40)",
        os.path.join(args.out_dir, "heatmap_action_agreement.png"),
        cmap="PuBuGn",
        vmin=0.0,
        vmax=1.0,
    )
    plot_scatter(model_rows, os.path.join(args.out_dir, "scatter_validity_vs_move_agreement.png"))
    plot_category_bars(category_rows, os.path.join(args.out_dir, "category_profile_bars.png"))

    print(f"Wrote: {model_cat_csv}")
    print(f"Wrote: {model_csv}")
    print(f"Wrote: {cat_csv}")
    print(f"Wrote: {trans_json}")
    print(f"Wrote: {gt_align_long_csv}")
    print(f"Wrote: {gt_align_matrix_csv}")
    print(f"Wrote: {gt_align_overall_csv}")
    print(f"Wrote: {os.path.join(args.out_dir, 'heatmap_gt_alignment_score.png')}")
    print(f"Wrote: {os.path.join(args.out_dir, 'heatmap_move_agreement.png')}")
    print(f"Wrote: {os.path.join(args.out_dir, 'heatmap_llm_invalid_rate.png')}")
    print(f"Wrote: {os.path.join(args.out_dir, 'heatmap_action_agreement.png')}")
    print(f"Wrote: {os.path.join(args.out_dir, 'scatter_validity_vs_move_agreement.png')}")
    print(f"Wrote: {os.path.join(args.out_dir, 'category_profile_bars.png')}")


if __name__ == "__main__":
    main()
