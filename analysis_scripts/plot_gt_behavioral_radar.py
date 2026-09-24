import argparse
import csv
import json
import math
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


def load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def discover_models(results_root):
    out = []
    if not os.path.isdir(results_root):
        return out
    for name in sorted(os.listdir(results_root)):
        p = os.path.join(results_root, name)
        if not os.path.isdir(p):
            continue
        if name.startswith(".") or name == "model_wise_figures":
            continue
        out.append(name)
    return out


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
        "safe": new_pos in SAFE_SQUARES,
        "home_entry": new_pos >= home_start,
        "bring_out": old_pos == -1 and new_pos == start,
        "overshoot": new_pos > home_end,
    }


def scenario_rows(root, scenario):
    path = os.path.join(root, scenario, "spot_results.json")
    return load_json(path) or []


def rate(num, den):
    return (num / den) if den else 0.0


def compute_metrics_llm(none_root):
    # block_rate
    blocked = scenario_rows(none_root, "blocked")
    blocked_valid = [r for r in blocked if r.get("llm_valid")]
    block_rate = rate(sum(1 for r in blocked_valid if r.get("llm_equals_heuristic")), len(blocked_valid))

    # capture_rate (capture scenario)
    capture = scenario_rows(none_root, "capture")
    capture_valid = [r for r in capture if r.get("llm_valid")]
    capture_rate = rate(
        sum(1 for r in capture_valid if classify_move(r, "llm_move")["capture"]),
        len(capture_valid),
    )

    # safe_rate (safe scenario)
    safe = scenario_rows(none_root, "safe")
    safe_valid = [r for r in safe if r.get("llm_valid")]
    safe_rate = rate(
        sum(1 for r in safe_valid if classify_move(r, "llm_move")["safe"]),
        len(safe_valid),
    )

    # home_entry_rate (home_entry scenario)
    home_entry = scenario_rows(none_root, "home_entry")
    home_entry_valid = [r for r in home_entry if r.get("llm_valid")]
    home_entry_rate = rate(
        sum(1 for r in home_entry_valid if classify_move(r, "llm_move")["home_entry"]),
        len(home_entry_valid),
    )

    # bring_out_rate (extra_turn scenario)
    extra_turn = scenario_rows(none_root, "extra_turn")
    extra_turn_valid = [r for r in extra_turn if r.get("llm_valid")]
    bring_out_rate = rate(
        sum(1 for r in extra_turn_valid if classify_move(r, "llm_move")["bring_out"]),
        len(extra_turn_valid),
    )

    # home_finish_rate (capture_vs_home_finish scenario)
    cvhf = scenario_rows(none_root, "capture_vs_home_finish")
    cvhf_valid = [r for r in cvhf if r.get("llm_valid")]
    home_finish_rate = rate(
        sum(1 for r in cvhf_valid if classify_move(r, "llm_move")["new_pos"] == classify_move(r, "llm_move")["home_end"]),
        len(cvhf_valid),
    )

    # overshoot_avoid_rate (1 - overshoot_rate in overshoot scenario)
    overshoot = scenario_rows(none_root, "overshoot")
    overshoot_valid = [r for r in overshoot if r.get("llm_valid")]
    overshoot_rate = rate(
        sum(1 for r in overshoot_valid if classify_move(r, "llm_move")["overshoot"]),
        len(overshoot_valid),
    )
    overshoot_avoid_rate = 1.0 - overshoot_rate

    return {
        "capture_rate": capture_rate,
        "safe_rate": safe_rate,
        "home_entry_rate": home_entry_rate,
        "bring_out_rate": bring_out_rate,
        "home_finish_rate": home_finish_rate,
        "block_rate": block_rate,
        "overshoot_avoid_rate": overshoot_avoid_rate,
    }


def compute_metrics_gt(gt_root):
    blocked = scenario_rows(gt_root, "blocked")
    blocked_valid = [r for r in blocked if r.get("gt_valid")]
    block_rate = rate(sum(1 for r in blocked_valid if r.get("gt_equals_heuristic")), len(blocked_valid))

    capture = scenario_rows(gt_root, "capture")
    capture_valid = [r for r in capture if r.get("gt_valid")]
    capture_rate = rate(
        sum(1 for r in capture_valid if classify_move(r, "gt_move")["capture"]),
        len(capture_valid),
    )

    safe = scenario_rows(gt_root, "safe")
    safe_valid = [r for r in safe if r.get("gt_valid")]
    safe_rate = rate(
        sum(1 for r in safe_valid if classify_move(r, "gt_move")["safe"]),
        len(safe_valid),
    )

    home_entry = scenario_rows(gt_root, "home_entry")
    home_entry_valid = [r for r in home_entry if r.get("gt_valid")]
    home_entry_rate = rate(
        sum(1 for r in home_entry_valid if classify_move(r, "gt_move")["home_entry"]),
        len(home_entry_valid),
    )

    extra_turn = scenario_rows(gt_root, "extra_turn")
    extra_turn_valid = [r for r in extra_turn if r.get("gt_valid")]
    bring_out_rate = rate(
        sum(1 for r in extra_turn_valid if classify_move(r, "gt_move")["bring_out"]),
        len(extra_turn_valid),
    )

    cvhf = scenario_rows(gt_root, "capture_vs_home_finish")
    cvhf_valid = [r for r in cvhf if r.get("gt_valid")]
    home_finish_rate = rate(
        sum(1 for r in cvhf_valid if classify_move(r, "gt_move")["new_pos"] == classify_move(r, "gt_move")["home_end"]),
        len(cvhf_valid),
    )

    overshoot = scenario_rows(gt_root, "overshoot")
    overshoot_valid = [r for r in overshoot if r.get("gt_valid")]
    overshoot_rate = rate(
        sum(1 for r in overshoot_valid if classify_move(r, "gt_move")["overshoot"]),
        len(overshoot_valid),
    )
    overshoot_avoid_rate = 1.0 - overshoot_rate

    return {
        "capture_rate": capture_rate,
        "safe_rate": safe_rate,
        "home_entry_rate": home_entry_rate,
        "bring_out_rate": bring_out_rate,
        "home_finish_rate": home_finish_rate,
        "block_rate": block_rate,
        "overshoot_avoid_rate": overshoot_avoid_rate,
    }


def radar_angles(n_axes):
    base = [n / float(n_axes) * 2 * math.pi for n in range(n_axes)]
    return base + base[:1]


def closed(vals):
    return vals + vals[:1]


def plot_combined_radar(metrics_by_model, gt_metrics, axes, out_png):
    angles = radar_angles(len(axes))
    fig = plt.figure(figsize=(12, 9))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids([a * 180 / math.pi for a in angles[:-1]], labels=[""] * len(axes))
    ax.set_ylim(0, 1)
    ax.tick_params(axis="y", labelsize=11)

    gt_vals = closed([gt_metrics[a] for a in axes])
    ax.plot(angles, gt_vals, linewidth=3, linestyle="-", label="GT (reference)", color="black")
    ax.fill(angles, gt_vals, alpha=0.08, color="black")

    for model, m in metrics_by_model.items():
        vals = closed([m[a] for a in axes])
        ax.plot(angles, vals, linewidth=1.5, alpha=0.8, label=model)

    label_r = 1.10
    for angle, label in zip(angles[:-1], axes):
        deg = (angle * 180 / math.pi) % 360
        if deg == 90:
            ha, va = "center", "bottom"
        elif 0 < deg < 180:
            ha, va = "left", "center"
        elif deg == 270:
            ha, va = "center", "top"
        else:
            ha, va = "right", "center"
        ax.text(
            angle,
            label_r,
            label,
            fontsize=13,
            ha=ha,
            va=va,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.92),
            clip_on=False,
            zorder=10,
        )

    ax.set_title("Behavioral Radar: LLM Models vs GT (none persona)", fontsize=17, y=1.14)
    plt.legend(loc="upper right", bbox_to_anchor=(1.34, 1.12), fontsize=11)
    fig.subplots_adjust(left=0.08, right=0.78, top=0.82, bottom=0.07)
    ensure_dir(os.path.dirname(out_png))
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_panel_radar(metrics_by_model, gt_metrics, axes, out_png):
    models = list(metrics_by_model.keys())
    n = len(models)
    cols = 3
    rows = math.ceil(n / cols)
    angles = radar_angles(len(axes))

    fig = plt.figure(figsize=(cols * 4.6, rows * 4.2))
    for i, model in enumerate(models, start=1):
        ax = fig.add_subplot(rows, cols, i, polar=True)
        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids([a * 180 / math.pi for a in angles[:-1]], axes, fontsize=7)
        ax.set_ylim(0, 1)

        gt_vals = closed([gt_metrics[a] for a in axes])
        m_vals = closed([metrics_by_model[model][a] for a in axes])

        ax.plot(angles, gt_vals, linewidth=2.2, color="black", label="GT")
        ax.plot(angles, m_vals, linewidth=2.0, color="#1f77b4", label=model)
        ax.fill(angles, m_vals, alpha=0.10, color="#1f77b4")
        ax.set_title(model, fontsize=10, pad=10)

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Per-Model Behavioral Radar vs GT (none persona)", y=1.04, fontsize=12)
    fig.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_metrics_csv(path, metrics_by_model, gt_metrics, axes):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + axes)
        w.writerow(["GT"] + [gt_metrics[a] for a in axes])
        for model in sorted(metrics_by_model.keys()):
            w.writerow([model] + [metrics_by_model[model][a] for a in axes])


def main():
    parser = argparse.ArgumentParser(
        description="Plot non-persona behavioral radar charts for LLM models vs GT."
    )
    parser.add_argument("--results-root", default="results_for_40")
    parser.add_argument("--gt-root", default="spot_results_gt")
    parser.add_argument("--out-dir", default="results_for_40/non_persona_eval")
    args = parser.parse_args()

    axes = [
        "capture_rate",
        "safe_rate",
        "home_entry_rate",
        "bring_out_rate",
        "home_finish_rate",
        "block_rate",
        "overshoot_avoid_rate",
    ]

    metrics_by_model = {}
    for model in discover_models(args.results_root):
        none_root = resolve_none_root(args.results_root, model)
        if not none_root:
            continue
        metrics_by_model[model] = compute_metrics_llm(none_root)

    if not metrics_by_model:
        raise SystemExit("No model data found under results root.")

    gt_metrics = compute_metrics_gt(args.gt_root)

    csv_path = os.path.join(args.out_dir, "behavioral_radar_metrics.csv")
    combined_png = os.path.join(args.out_dir, "behavioral_radar_all_models_vs_gt.png")
    panel_png = os.path.join(args.out_dir, "behavioral_radar_model_panels_vs_gt.png")

    write_metrics_csv(csv_path, metrics_by_model, gt_metrics, axes)
    plot_combined_radar(metrics_by_model, gt_metrics, axes, combined_png)
    plot_panel_radar(metrics_by_model, gt_metrics, axes, panel_png)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {combined_png}")
    print(f"Wrote: {panel_png}")


if __name__ == "__main__":
    main()
