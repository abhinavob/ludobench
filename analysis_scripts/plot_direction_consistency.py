import argparse
import os
import tempfile
from typing import Dict, List, Tuple

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


META_COLS = {"category", "metric"}

# +1 => persona should increase this metric vs none, -1 => decrease.
EXPECTED_DIRECTIONS: Dict[str, List[Tuple[str, int]]] = {
    "aggressive": [
        ("capture.capture_rate", +1),
        ("capture_vs_home.capture_rate", +1),
        ("capture_vs_openexisting.capture_rate", +1),
        ("capture_vs_safe.capture_rate", +1),
        ("grudge.retaliation_grudge_rate", +1),
    ],
    "greedy": [
        ("capture_vs_home_finish.home_finish_rate", +1),
        ("capture_vs_home.home_entry_rate", +1),
        ("home_entry.home_entry_rate", +1),
        ("capture_vs_home_finish.capture_rate", -1),
    ],
    "safe": [
        ("safe.safe_rate", +1),
        ("capture_vs_safe.safe_rate", +1),
        ("capture.capture_rate", -1),
        ("capture_vs_openexisting.capture_rate", -1),
    ],
    "unforgiving": [
        ("grudge.change_rate", +1),
        ("grudge.retaliation_grudge_rate", +1),
    ],
}


def discover_model_csvs(root: str, model_filter: List[str]) -> List[Tuple[str, str]]:
    wanted = set(model_filter)
    found: List[Tuple[str, str]] = []
    for name in sorted(os.listdir(root)):
        model_dir = os.path.join(root, name)
        if not os.path.isdir(model_dir):
            continue
        if wanted and name not in wanted:
            continue
        csv_path = os.path.join(model_dir, "persona_comparison.csv")
        if os.path.isfile(csv_path):
            found.append((name, csv_path))
    return found


def load_metric_table(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    persona_cols = [c for c in df.columns if c not in META_COLS]
    if "none" not in persona_cols:
        raise ValueError(f"Missing 'none' column in {csv_path}")
    df["metric_key"] = df["category"].astype(str) + "." + df["metric"].astype(str)
    tab = df[["metric_key"] + persona_cols].drop_duplicates(subset=["metric_key"]).set_index("metric_key")
    for c in persona_cols:
        tab[c] = pd.to_numeric(tab[c], errors="coerce")
    return tab


def compute_direction_consistency(tab: pd.DataFrame, persona: str) -> Tuple[float, int, int]:
    rules = EXPECTED_DIRECTIONS.get(persona, [])
    if persona not in tab.columns:
        return float("nan"), 0, len(rules)

    hits = 0
    used = 0
    for metric_key, direction in rules:
        if metric_key not in tab.index:
            continue
        p_val = tab.at[metric_key, persona]
        n_val = tab.at[metric_key, "none"]
        if pd.isna(p_val) or pd.isna(n_val):
            continue
        delta = float(p_val) - float(n_val)
        if delta == 0:
            continue
        used += 1
        if direction * delta > 0:
            hits += 1

    pct = (100.0 * hits / used) if used > 0 else float("nan")
    return pct, hits, used


def main() -> None:
    parser = argparse.ArgumentParser(
        description="% of expected-direction metrics satisfied per model/persona."
    )
    parser.add_argument("--root", default="results_for_40", help="Root with model folders")
    parser.add_argument("--models", default="", help="Optional comma-separated models")
    parser.add_argument("--personas", default="aggressive,greedy,safe,unforgiving", help="Personas to include")
    parser.add_argument("--out-dir", default="results_for_40/model_wise_figures", help="Output directory")
    parser.add_argument("--out-png", default="direction_consistency_barplot.png", help="Output PNG")
    parser.add_argument("--out-csv", default="direction_consistency_scores.csv", help="Output CSV")
    parser.add_argument("--dpi", type=int, default=360, help="PNG DPI")
    args = parser.parse_args()

    model_filter = [m.strip() for m in args.models.split(",") if m.strip()]
    personas = [p.strip() for p in args.personas.split(",") if p.strip()]
    model_csvs = discover_model_csvs(args.root, model_filter)
    if not model_csvs:
        raise SystemExit(f"No model persona_comparison.csv found under {args.root}")

    rows = []
    for model_name, csv_path in model_csvs:
        tab = load_metric_table(csv_path)
        for persona in personas:
            pct, hits, used = compute_direction_consistency(tab, persona)
            rows.append(
                {
                    "model": model_name,
                    "persona": persona,
                    "direction_consistency_pct": pct,
                    "hits": hits,
                    "used_rules": used,
                }
            )

    out_df = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, args.out_csv)
    out_png = os.path.join(args.out_dir, args.out_png)
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["font.family"] = "DejaVu Sans"

    model_order = list(dict.fromkeys(out_df["model"].tolist()))
    persona_order = [p for p in personas if p in out_df["persona"].unique()]
    piv = out_df.pivot(index="model", columns="persona", values="direction_consistency_pct").reindex(index=model_order)

    x = np.arange(len(model_order), dtype=float)
    width = 0.18
    tiny_stub = 0.9  # visual stub for exact 0% bars so they are still visible
    colors = sns.color_palette("Set2", n_colors=max(1, len(persona_order)))

    fig, ax = plt.subplots(figsize=(max(10, len(model_order) * 1.55), 5.6))
    for i, persona in enumerate(persona_order):
        vals = piv[persona].fillna(0.0).values.astype(float)
        draw_vals = np.where(vals > 0, vals, tiny_stub)
        pos = x - (len(persona_order) - 1) * width / 2 + i * width
        bars = ax.bar(
            pos,
            draw_vals,
            width=width,
            label=persona,
            color=colors[i],
            edgecolor="#555555",
            linewidth=0.5,
            zorder=3,
        )
        for b, v in zip(bars, vals):
            if v == 0:
                b.set_hatch("//")
            y_text = (tiny_stub + 1.2) if v == 0 else (v + 1.2)
            ax.text(
                b.get_x() + b.get_width() / 2,
                y_text,
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=7.8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(model_order)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Expected-Direction Consistency (%)")
    ax.set_xlabel("Model")
    ax.set_title("Direction-Consistency Bar Plot")
    ax.grid(axis="y", alpha=0.22)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=18)
    ax.legend(title="Persona", frameon=False, loc="upper right")
    ax.text(
        0.01,
        -0.14,
        "Note: hatched tiny bars denote exact 0%.",
        transform=ax.transAxes,
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()
