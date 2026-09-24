import argparse
import csv
import json
import os
import math
import tempfile

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")


def find_persona_dirs(root, personas):
    found = []
    for p in personas:
        d = os.path.join(root, p)
        if os.path.isdir(d):
            found.append((p, d))
    return found


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def get_summary_entry(persona_dir, category, scenario):
    path = os.path.join(persona_dir, category, "spot_summary.json")
    data = load_json(path)
    if not data:
        return None
    return data.get(scenario)


def classify_move(record):
    player = int(record["current_player"])
    players = [int(p) for p in record["players"]]
    tokens = {int(k): v for k, v in record["tokens"].items()}
    dice = int(record["dice"])
    token_idx = int(record["llm_move"])

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
    captured_player = None
    if 0 <= new_pos < board_size and new_pos not in safe_squares:
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
        "safe": new_pos in safe_squares,
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


def compute_persona_metrics(persona_dir):
    out = {}

    # blocked
    s = get_summary_entry(persona_dir, "blocked", "blocked")
    if s:
        out[("blocked", "llm_invalid_rate")] = s.get("llm_invalid_rate", 0.0)
        total_valid = s.get("total_valid", 0)
        out[("blocked", "block_rate")] = (
            s.get("llm_equals_heuristic", 0) / total_valid if total_valid else 0.0
        )

    # capture_vs_home_finish
    s = get_summary_entry(persona_dir, "capture_vs_home_finish", "capture_vs_home_finish")
    if s:
        out[("capture_vs_home_finish", "llm_invalid_rate")] = s.get("llm_invalid_rate", 0.0)
    results = load_json(os.path.join(persona_dir, "capture_vs_home_finish", "spot_results.json")) or []
    valid = [r for r in results if r.get("scenario") == "capture_vs_home_finish" and r.get("llm_valid")]
    if valid:
        capture = 0
        finish = 0
        for r in valid:
            info = classify_move(r)
            if info["capture"]:
                capture += 1
            if info["new_pos"] == info["home_end"]:
                finish += 1
        out[("capture_vs_home_finish", "capture_rate")] = capture / len(valid)
        out[("capture_vs_home_finish", "home_finish_rate")] = finish / len(valid)

    # capture_vs_home
    s = get_summary_entry(persona_dir, "capture_vs_home", "capture_vs_home")
    if s:
        out[("capture_vs_home", "llm_invalid_rate")] = s.get("llm_invalid_rate", 0.0)
    results = load_json(os.path.join(persona_dir, "capture_vs_home", "spot_results.json")) or []
    valid = [r for r in results if r.get("scenario") == "capture_vs_home" and r.get("llm_valid")]
    if valid:
        capture = 0
        home_entry = 0
        for r in valid:
            info = classify_move(r)
            if info["capture"]:
                capture += 1
            if info["home_entry"]:
                home_entry += 1
        out[("capture_vs_home", "capture_rate")] = capture / len(valid)
        out[("capture_vs_home", "home_entry_rate")] = home_entry / len(valid)

    # capture
    s = get_summary_entry(persona_dir, "capture", "capture")
    if s:
        out[("capture", "llm_invalid_rate")] = s.get("llm_invalid_rate", 0.0)
    results = load_json(os.path.join(persona_dir, "capture", "spot_results.json")) or []
    valid = [r for r in results if r.get("scenario") == "capture" and r.get("llm_valid")]
    if valid:
        capture = sum(1 for r in valid if classify_move(r)["capture"])
        out[("capture", "capture_rate")] = capture / len(valid)

    # capture_vs_openexisting
    s = get_summary_entry(persona_dir, "capture_vs_openexisting", "capture_vs_openexisting")
    if s:
        out[("capture_vs_openexisting", "llm_invalid_rate")] = s.get("llm_invalid_rate", 0.0)
    results = load_json(os.path.join(persona_dir, "capture_vs_openexisting", "spot_results.json")) or []
    valid = [r for r in results if r.get("scenario") == "capture_vs_openexisting" and r.get("llm_valid")]
    if valid:
        capture = 0
        open_rate = 0
        for r in valid:
            info = classify_move(r)
            if info["capture"]:
                capture += 1
            if info["bring_out"]:
                open_rate += 1
        out[("capture_vs_openexisting", "capture_rate")] = capture / len(valid)
        out[("capture_vs_openexisting", "open_rate")] = open_rate / len(valid)

    # capture_vs_safe
    s = get_summary_entry(persona_dir, "capture_vs_safe", "capture_vs_safe")
    if s:
        out[("capture_vs_safe", "llm_invalid_rate")] = s.get("llm_invalid_rate", 0.0)
    results = load_json(os.path.join(persona_dir, "capture_vs_safe", "spot_results.json")) or []
    valid = [r for r in results if r.get("scenario") == "capture_vs_safe" and r.get("llm_valid")]
    if valid:
        capture = 0
        safe_rate = 0
        for r in valid:
            info = classify_move(r)
            if info["capture"]:
                capture += 1
            if info["safe"]:
                safe_rate += 1
        out[("capture_vs_safe", "capture_rate")] = capture / len(valid)
        out[("capture_vs_safe", "safe_rate")] = safe_rate / len(valid)

    # overshoot
    s = get_summary_entry(persona_dir, "overshoot", "overshoot")
    if s:
        out[("overshoot", "llm_invalid_rate")] = s.get("llm_invalid_rate", 0.0)
    results = load_json(os.path.join(persona_dir, "overshoot", "spot_results.json")) or []
    valid = [r for r in results if r.get("scenario") == "overshoot" and r.get("llm_valid")]
    if valid:
        overshoot = 0
        for r in valid:
            info = classify_move(r)
            if info["new_pos"] > info["home_end"]:
                overshoot += 1
        out[("overshoot", "overshoot_rate")] = overshoot / len(valid)

    # safe
    s = get_summary_entry(persona_dir, "safe", "safe")
    if s:
        out[("safe", "llm_invalid_rate")] = s.get("llm_invalid_rate", 0.0)
    results = load_json(os.path.join(persona_dir, "safe", "spot_results.json")) or []
    valid = [r for r in results if r.get("scenario") == "safe" and r.get("llm_valid")]
    if valid:
        safe = sum(1 for r in valid if classify_move(r)["safe"])
        out[("safe", "safe_rate")] = safe / len(valid)

    # safe_vs_openexisting
    s = get_summary_entry(persona_dir, "safe_vs_openexisting", "safe_vs_openexisting")
    if s:
        out[("safe_vs_openexisting", "llm_invalid_rate")] = s.get("llm_invalid_rate", 0.0)
    results = load_json(os.path.join(persona_dir, "safe_vs_openexisting", "spot_results.json")) or []
    valid = [r for r in results if r.get("scenario") == "safe_vs_openexisting" and r.get("llm_valid")]
    if valid:
        safe_rate = 0
        open_rate = 0
        for r in valid:
            info = classify_move(r)
            if info["safe"]:
                safe_rate += 1
            if info["bring_out"]:
                open_rate += 1
        out[("safe_vs_openexisting", "safe_rate")] = safe_rate / len(valid)
        out[("safe_vs_openexisting", "open_rate")] = open_rate / len(valid)

    # extra_turn
    s = get_summary_entry(persona_dir, "extra_turn", "extra_turn")
    if s:
        out[("extra_turn", "llm_invalid_rate")] = s.get("llm_invalid_rate", 0.0)
    results = load_json(os.path.join(persona_dir, "extra_turn", "spot_results.json")) or []
    valid = [r for r in results if r.get("scenario") == "extra_turn" and r.get("llm_valid")]
    if valid:
        bring_out = sum(1 for r in valid if classify_move(r)["bring_out"])
        out[("extra_turn", "bring_out_rate")] = bring_out / len(valid)
        out[("extra_turn", "move_existing_rate")] = 1.0 - out[("extra_turn", "bring_out_rate")]

    # home_entry
    s = get_summary_entry(persona_dir, "home_entry", "home_entry")
    if s:
        out[("home_entry", "llm_invalid_rate")] = s.get("llm_invalid_rate", 0.0)
    results = load_json(os.path.join(persona_dir, "home_entry", "spot_results.json")) or []
    valid = [r for r in results if r.get("scenario") == "home_entry" and r.get("llm_valid")]
    if valid:
        home_entry = sum(1 for r in valid if classify_move(r)["home_entry"])
        out[("home_entry", "home_entry_rate")] = home_entry / len(valid)

    # grudge
    gr = load_json(os.path.join(persona_dir, "grudge", "grudge_report.json")) or {}
    if gr:
        out[("grudge", "change_rate")] = gr.get("change_rate", 0.0)
        out[("grudge", "retaliation_grudge_rate")] = gr.get("retaliation_grudge_rate", 0.0)
        out[("grudge", "retaliation_noconflict_rate")] = gr.get("retaliation_noconflict_rate", 0.0)

    return out


def collect_metrics(root, personas):
    persona_dirs = find_persona_dirs(root, personas)
    if not persona_dirs:
        raise SystemExit(
            f"No persona folders found under {root}. "
            f"Expected one or more of: {', '.join(personas)}"
        )
    by_persona = {}
    for persona, pdir in persona_dirs:
        by_persona[persona] = compute_persona_metrics(pdir)
    return by_persona, metric_rows(), [p for p, _ in persona_dirs]


def write_csv(out_csv, by_persona, rows, persona_order):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "metric"] + persona_order)
        for category, metric in rows:
            row = [category, metric]
            for persona in persona_order:
                v = by_persona.get(persona, {}).get((category, metric), "")
                row.append(v)
            writer.writerow(row)


def write_plot(out_png, by_persona, rows, persona_order):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"matplotlib is required for plotting: {exc}") from exc

    labels = [f"{cat}.{met}" for cat, met in rows]
    if not labels:
        return

    x = list(range(len(labels)))
    width = 0.8 / max(1, len(persona_order))

    plt.figure(figsize=(max(12, len(labels) * 0.6), 6))
    for i, persona in enumerate(persona_order):
        vals = []
        for cat, met in rows:
            v = by_persona.get(persona, {}).get((cat, met), 0.0)
            vals.append(v if isinstance(v, (int, float)) else 0.0)
        offsets = [xi - 0.4 + (i + 0.5) * width for xi in x]
        plt.bar(offsets, vals, width=width, label=persona)

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Metric Value")
    plt.title("Persona Comparison")
    plt.legend()
    plt.tight_layout()
    out_dir = os.path.dirname(out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def write_none_correlation_plot(out_png, by_persona, rows, persona_order):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f"matplotlib is required for plotting: {exc}") from exc

    if "none" not in persona_order:
        print("Skipped none-correlation plot: 'none' persona not present.")
        return

    # Build aligned vectors by metric rows.
    none_vals = []
    others = []
    for p in persona_order:
        if p != "none":
            others.append(p)

    persona_vals = {p: [] for p in others}
    for cat, met in rows:
        v_none = by_persona.get("none", {}).get((cat, met), 0.0)
        if not isinstance(v_none, (int, float)):
            v_none = 0.0
        none_vals.append(float(v_none))
        for p in others:
            v = by_persona.get(p, {}).get((cat, met), 0.0)
            if not isinstance(v, (int, float)):
                v = 0.0
            persona_vals[p].append(float(v))

    def pearson(x, y):
        n = len(x)
        if n == 0:
            return 0.0
        mx = sum(x) / n
        my = sum(y) / n
        num = sum((a - mx) * (b - my) for a, b in zip(x, y))
        denx = math.sqrt(sum((a - mx) ** 2 for a in x))
        deny = math.sqrt(sum((b - my) ** 2 for b in y))
        den = denx * deny
        return (num / den) if den > 0 else 0.0

    corr = {p: pearson(none_vals, persona_vals[p]) for p in others}

    plt.figure(figsize=(7, 5))
    xs = list(corr.keys())
    ys = [corr[p] for p in xs]
    bars = plt.bar(xs, ys)
    plt.axhline(0, color="black", linewidth=1)
    plt.ylim(-1, 1)
    plt.ylabel("Pearson correlation (r)")
    plt.title("Correlation With 'none' Persona")
    for b, y in zip(bars, ys):
        plt.text(b.get_x() + b.get_width() / 2, y + (0.03 if y >= 0 else -0.07), f"{y:.2f}", ha="center")
    plt.tight_layout()
    out_dir = os.path.dirname(out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare persona outputs for fixed scenario metrics and generate CSV + plot."
    )
    parser.add_argument(
        "--root",
        default="spot_results",
        help="Root directory containing persona subfolders (default: spot_results).",
    )
    parser.add_argument(
        "--personas",
        default="aggressive,greedy,safe,unforgiving,none",
        help="Comma-separated persona folder names to include.",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-plot",
        default=None,
        help="Output plot path.",
    )
    parser.add_argument(
        "--out-none-corr",
        default=None,
        help="Output plot path for correlation with 'none' persona.",
    )
    args = parser.parse_args()

    out_csv = args.out_csv or os.path.join(args.root, "persona_comparison.csv")
    out_plot = args.out_plot or os.path.join(args.root, "persona_comparison.png")
    out_none_corr = args.out_none_corr or os.path.join(args.root, "persona_none_correlation.png")

    personas = [p.strip() for p in args.personas.split(",") if p.strip()]
    by_persona, rows, persona_order = collect_metrics(args.root, personas)
    write_csv(out_csv, by_persona, rows, persona_order)
    write_plot(out_plot, by_persona, rows, persona_order)
    write_none_correlation_plot(out_none_corr, by_persona, rows, persona_order)

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_plot}")
    print(f"Wrote: {out_none_corr}")


if __name__ == "__main__":
    main()
