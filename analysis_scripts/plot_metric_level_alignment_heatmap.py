import argparse
import os
import tempfile
from typing import Dict, List, Tuple

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


META_COLS = {"category", "metric"}

# Curated key metrics for readability in paper figures.
KEY_METRICS = [
    "blocked.llm_invalid_rate",
    "blocked.block_rate",
    "capture.llm_invalid_rate",
    "capture.capture_rate",
    "capture_vs_home.capture_rate",
    "capture_vs_home.home_entry_rate",
    "capture_vs_home_finish.capture_rate",
    "capture_vs_home_finish.home_finish_rate",
    "capture_vs_openexisting.capture_rate",
    "capture_vs_openexisting.open_rate",
    "capture_vs_safe.capture_rate",
    "capture_vs_safe.safe_rate",
    "safe.safe_rate",
    "extra_turn.bring_out_rate",
    "extra_turn.move_existing_rate",
    "home_entry.home_entry_rate",
    "grudge.change_rate",
    "grudge.retaliation_grudge_rate",
    "grudge.retaliation_noconflict_rate",
]


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


def model_persona_deltas(csv_path: str, model_name: str, personas: List[str], key_metrics: List[str]) -> List[Dict[str, float]]:
    df = pd.read_csv(csv_path)
    persona_cols = [c for c in df.columns if c not in META_COLS]
    if "none" not in persona_cols:
        raise ValueError(f"Missing 'none' column in {csv_path}")

    df["metric_key"] = df["category"].astype(str) + "." + df["metric"].astype(str)
    slim = df[["metric_key"] + persona_cols].copy()
    slim = slim.drop_duplicates(subset=["metric_key"], keep="first").set_index("metric_key")
    for c in persona_cols:
        slim[c] = pd.to_numeric(slim[c], errors="coerce")

    rows = []
    usable_personas = [p for p in personas if p in slim.columns and p != "none"]
    for persona in usable_personas:
        out = {"row_key": f"{model_name} | {persona}", "model": model_name, "persona": persona}
        for mk in key_metrics:
            if mk not in slim.index:
                out[mk] = float("nan")
                continue
            p_val = slim.at[mk, persona]
            n_val = slim.at[mk, "none"]
            if pd.isna(p_val) or pd.isna(n_val):
                out[mk] = float("nan")
            else:
                out[mk] = float(p_val - n_val)  # signed delta
        rows.append(out)
    return rows


def save_outputs(df: pd.DataFrame, key_metrics: List[str], out_csv: str, out_png: str, dpi: int) -> None:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    df.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")

    heat = df.set_index("row_key")[key_metrics]

    vmax = float(heat.abs().max(numeric_only=True).max())
    if vmax == 0 or pd.isna(vmax):
        vmax = 0.05

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["font.family"] = "DejaVu Sans"

    fig, ax = plt.subplots(figsize=(max(12, len(key_metrics) * 0.65), max(6, len(heat) * 0.45)))
    sns.heatmap(
        heat,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        linecolor="#EEEEEE",
        cbar_kws={"label": "Signed Delta vs none"},
        ax=ax,
    )
    ax.set_title("Metric-Level Alignment Heatmap (Signed Delta vs none)")
    ax.set_xlabel("Key Metrics")
    ax.set_ylabel("Model | Persona")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rows=model-persona, columns=key metrics, values=signed delta (persona - none)."
    )
    parser.add_argument("--root", default="results_for_40", help="Root containing model folders with persona_comparison.csv")
    parser.add_argument("--models", default="", help="Optional comma-separated model folder names")
    parser.add_argument("--personas", default="aggressive,greedy,safe,unforgiving", help="Personas to include (excluding none)")
    parser.add_argument("--out-dir", default="results_for_40/model_wise_figures", help="Output directory")
    parser.add_argument("--out-png", default="metric_level_alignment_heatmap.png", help="Output PNG name")
    parser.add_argument("--out-csv", default="metric_level_alignment_deltas.csv", help="Output CSV name")
    parser.add_argument("--dpi", type=int, default=360, help="PNG DPI")
    args = parser.parse_args()

    model_filter = [m.strip() for m in args.models.split(",") if m.strip()]
    personas = [p.strip() for p in args.personas.split(",") if p.strip()]
    model_csvs = discover_model_csvs(args.root, model_filter)
    if not model_csvs:
        raise SystemExit(f"No model persona_comparison.csv found under {args.root}")

    rows = []
    for model_name, csv_path in model_csvs:
        rows.extend(model_persona_deltas(csv_path, model_name, personas, KEY_METRICS))

    if not rows:
        raise SystemExit("No rows produced for requested models/personas.")

    out_df = pd.DataFrame(rows)
    out_png = os.path.join(args.out_dir, args.out_png)
    out_csv = os.path.join(args.out_dir, args.out_csv)
    save_outputs(out_df, KEY_METRICS, out_csv, out_png, args.dpi)


if __name__ == "__main__":
    main()
