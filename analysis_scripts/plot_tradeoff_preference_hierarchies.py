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


SAFE_SQUARES = {0, 8, 13, 21, 26, 34, 39, 47}
STARTS = {0: 0, 1: 13, 2: 26, 3: 39}
BOARD_SIZE = 52
HOME_START_GLOBAL = 52
HOME_LEN = 6

TRADEOFFS = [
    {
        "code": "cvs",
        "scenario": "capture_vs_safe",
        "left_action": "capture",
        "right_action": "safe",
        "left_label": "Capture",
        "right_label": "Safe",
    },
    {
        "code": "cvh",
        "scenario": "capture_vs_home",
        "left_action": "capture",
        "right_action": "home_entry",
        "left_label": "Capture",
        "right_label": "Home Entry",
    },
    {
        "code": "cvf",
        "scenario": "capture_vs_home_finish",
        "left_action": "capture",
        "right_action": "home_finish",
        "left_label": "Capture",
        "right_label": "Home Finish",
    },
    {
        "code": "cvo",
        "scenario": "capture_vs_openexisting",
        "left_action": "capture",
        "right_action": "open",
        "left_label": "Capture",
        "right_label": "Open",
    },
]


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def discover_models(results_root):
    models = []
    if not os.path.isdir(results_root):
        return models
    for name in sorted(os.listdir(results_root)):
        p = os.path.join(results_root, name)
        if not os.path.isdir(p):
            continue
        if name.startswith(".") or name in {"model_wise_figures", "non_persona_eval", "game-theory-agent"}:
            continue
        models.append(name)
    return models


def resolve_none_root(results_root, model):
    p1 = os.path.join(results_root, model, "personas", "none")
    p2 = os.path.join(results_root, model, "none")
    if os.path.isdir(p1):
        return p1
    if os.path.isdir(p2):
        return p2
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
    return "other"


def compute_tradeoff_stats(rows, scenario, move_key, valid_key, left_action, right_action):
    total = len(rows)
    valid_rows = [r for r in rows if r.get(valid_key)]
    valid_n = len(valid_rows)

    left_count = 0
    right_count = 0
    other_count = 0
    for r in valid_rows:
        info = classify_move(r, move_key)
        action = scenario_action_label(scenario, info)
        if action == left_action:
            left_count += 1
        elif action == right_action:
            right_count += 1
        else:
            other_count += 1

    two_way_n = left_count + right_count
    left_rate = (left_count / two_way_n) if two_way_n else 0.0
    right_rate = (right_count / two_way_n) if two_way_n else 0.0
    preference_score = right_rate - left_rate  # [-1, 1]: left-heavy to right-heavy

    return {
        "total": total,
        "valid_n": valid_n,
        "valid_rate": (valid_n / total) if total else 0.0,
        "left_count": left_count,
        "right_count": right_count,
        "other_count": other_count,
        "two_way_n": two_way_n,
        "left_rate": left_rate,
        "right_rate": right_rate,
        "preference_score": preference_score,
    }


def write_csv(path, rows, fieldnames):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_tradeoffs(rows, out_png):
    ensure_dir(os.path.dirname(out_png))
    by_code = {}
    gt_by_code = {}
    models = sorted({r["model"] for r in rows if r["model"] != "GT"})
    for r in rows:
        code = r["tradeoff_code"]
        by_code.setdefault(code, [])
        by_code[code].append(r)
        if r["model"] == "GT":
            gt_by_code[code] = r

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True)
    axes = axes.flatten()

    for idx, t in enumerate(TRADEOFFS):
        ax = axes[idx]
        code = t["code"]
        subset = {r["model"]: r for r in by_code.get(code, []) if r["model"] != "GT"}
        gt_row = gt_by_code.get(code)
        if not subset:
            ax.set_visible(False)
            continue

        vals = [subset[m]["preference_score"] for m in models]
        y = list(range(len(models)))
        colors = []
        for v in vals:
            if v > 0.02:
                colors.append("#1f77b4")
            elif v < -0.02:
                colors.append("#d62728")
            else:
                colors.append("#7f7f7f")

        ax.barh(y, vals, color=colors, alpha=0.9, edgecolor="white")
        ax.axvline(0, color="black", linewidth=1.0)
        if gt_row is not None:
            ax.axvline(
                gt_row["preference_score"],
                color="#2ca02c",
                linestyle="--",
                linewidth=2.0,
                label=f'GT ref ({gt_row["preference_score"]:+.2f})',
            )
        ax.set_yticks(y)
        ax.set_yticklabels(models, fontsize=13)
        ax.set_xlim(-1.0, 1.0)
        ax.tick_params(axis="x", labelsize=12)
        ax.grid(axis="x", linestyle=":", alpha=0.35)
        ax.set_title(
            f'{code.upper()} ({t["scenario"]})\nLeft: {t["left_label"]} | Right: {t["right_label"]}',
            fontsize=14,
        )
        ax.text(-0.98, len(models) - 0.35, t["left_label"], ha="left", va="bottom", fontsize=11, color="#666666")
        ax.text(+0.98, len(models) - 0.35, t["right_label"], ha="right", va="bottom", fontsize=11, color="#666666")
        if gt_row is not None:
            ax.legend(loc="lower right", fontsize=11, frameon=True)

    fig.suptitle(
        "Tradeoff Preference Hierarchies (none persona)\nNegative = left option preference, Positive = right option preference",
        fontsize=17,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot diverging tradeoff preference bars vs GT reference (none persona).")
    parser.add_argument("--results-root", default="results_for_40")
    parser.add_argument("--gt-root", default="spot_results_gt")
    parser.add_argument("--out-dir", default="results_for_40/non_persona_eval")
    args = parser.parse_args()

    rows = []
    models = discover_models(args.results_root)

    for model in models:
        none_root = resolve_none_root(args.results_root, model)
        if not none_root:
            continue
        for t in TRADEOFFS:
            scenario = t["scenario"]
            path = os.path.join(none_root, scenario, "spot_results.json")
            scenario_rows = load_json(path) or []
            stats = compute_tradeoff_stats(
                rows=scenario_rows,
                scenario=scenario,
                move_key="llm_move",
                valid_key="llm_valid",
                left_action=t["left_action"],
                right_action=t["right_action"],
            )
            rows.append(
                {
                    "model": model,
                    "tradeoff_code": t["code"],
                    "scenario": scenario,
                    "left_label": t["left_label"],
                    "right_label": t["right_label"],
                    **stats,
                }
            )

    for t in TRADEOFFS:
        scenario = t["scenario"]
        path = os.path.join(args.gt_root, scenario, "spot_results.json")
        scenario_rows = load_json(path) or []
        stats = compute_tradeoff_stats(
            rows=scenario_rows,
            scenario=scenario,
            move_key="gt_move",
            valid_key="gt_valid",
            left_action=t["left_action"],
            right_action=t["right_action"],
        )
        rows.append(
            {
                "model": "GT",
                "tradeoff_code": t["code"],
                "scenario": scenario,
                "left_label": t["left_label"],
                "right_label": t["right_label"],
                **stats,
            }
        )

    out_csv = os.path.join(args.out_dir, "tradeoff_preference_scores.csv")
    out_png = os.path.join(args.out_dir, "tradeoff_preference_hierarchies.png")
    write_csv(
        out_csv,
        rows,
        [
            "model",
            "tradeoff_code",
            "scenario",
            "left_label",
            "right_label",
            "total",
            "valid_n",
            "valid_rate",
            "left_count",
            "right_count",
            "other_count",
            "two_way_n",
            "left_rate",
            "right_rate",
            "preference_score",
        ],
    )
    plot_tradeoffs(rows, out_png)
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()
