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

# Direction map: +1 means persona metric should increase vs "none", -1 means decrease.
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
        ("safe_vs_openexisting.safe_rate", +1),
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
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        if wanted and name not in wanted:
            continue
        p = os.path.join(d, "persona_comparison.csv")
        if os.path.isfile(p):
            found.append((name, p))
    return found


def load_metric_frame(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    persona_cols = [c for c in df.columns if c not in META_COLS]
    if "none" not in persona_cols:
        raise ValueError(f"Missing 'none' column in {csv_path}")
    df["metric_key"] = df["category"].astype(str) + "." + df["metric"].astype(str)
    slim = df[["metric_key"] + persona_cols].copy()
    slim = slim.drop_duplicates(subset=["metric_key"], keep="first").set_index("metric_key")
    for c in persona_cols:
        slim[c] = pd.to_numeric(slim[c], errors="coerce")
    return slim


def compute_alignment_score(delta: float, scale: float = 0.10) -> float:
    # maps signed delta -> [0,1], centered at 0.5
    return float(0.5 + 0.5 * np.tanh(delta / scale))


def compute_model_alignment(frame: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, Tuple[int, int]]]:
    scores: Dict[str, float] = {}
    coverage: Dict[str, Tuple[int, int]] = {}

    for persona, rules in EXPECTED_DIRECTIONS.items():
        if persona not in frame.columns:
            scores[persona] = np.nan
            coverage[persona] = (0, len(rules))
            continue

        vals = []
        used = 0
        for metric_key, direction in rules:
            if metric_key not in frame.index:
                continue
            p_val = frame.at[metric_key, persona]
            n_val = frame.at[metric_key, "none"]
            if pd.isna(p_val) or pd.isna(n_val):
                continue
            delta = direction * (float(p_val) - float(n_val))
            vals.append(compute_alignment_score(delta))
            used += 1

        scores[persona] = float(np.mean(vals)) if vals else np.nan
        coverage[persona] = (used, len(rules))

    return scores, coverage


def save_outputs(
    score_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    out_png: str,
    out_csv: str,
    dpi: int,
) -> None:
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    # CSV (long format with coverage columns)
    long = score_df.reset_index().melt(id_vars=["model"], var_name="persona", value_name="alignment_score")
    cov_long = coverage_df.reset_index().melt(id_vars=["model"], var_name="persona", value_name="coverage")
    cov_long[["used_metrics", "total_expected"]] = pd.DataFrame(cov_long["coverage"].tolist(), index=cov_long.index)
    out = long.merge(cov_long.drop(columns=["coverage"]), on=["model", "persona"], how="left")
    out.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")

    # Heatmap figure
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["font.family"] = "DejaVu Sans"

    fig, ax = plt.subplots(figsize=(8.5, max(4, 0.8 * len(score_df.index))))
    sns.heatmap(
        score_df,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="#EEEEEE",
        cbar_kws={"label": "Persona Alignment Score"},
        ax=ax,
    )
    ax.set_title("Persona Alignment Heatmap (vs none baseline)")
    ax.set_xlabel("Persona")
    ax.set_ylabel("Model")
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model-wise persona alignment heatmap from results_for_40.")
    parser.add_argument("--root", default="results_for_40", help="Root with model folders containing persona_comparison.csv")
    parser.add_argument("--models", default="", help="Optional comma-separated model folders")
    parser.add_argument(
        "--out-dir",
        default="results_for_40/model_wise_figures",
        help="Output directory",
    )
    parser.add_argument("--out-png", default="persona_alignment_heatmap.png", help="Heatmap PNG name")
    parser.add_argument("--out-csv", default="persona_alignment_scores.csv", help="Scores CSV name")
    parser.add_argument("--dpi", type=int, default=360, help="PNG DPI")
    args = parser.parse_args()

    model_filter = [m.strip() for m in args.models.split(",") if m.strip()]
    model_csvs = discover_model_csvs(args.root, model_filter)
    if not model_csvs:
        raise SystemExit(f"No model persona_comparison.csv found under {args.root}")

    score_rows = []
    coverage_rows = []
    for model_name, csv_path in model_csvs:
        frame = load_metric_frame(csv_path)
        scores, coverage = compute_model_alignment(frame)
        row = {"model": model_name}
        row.update(scores)
        score_rows.append(row)

        crow = {"model": model_name}
        crow.update(coverage)
        coverage_rows.append(crow)

    score_df = pd.DataFrame(score_rows).set_index("model")
    # fixed persona order
    persona_order = [p for p in ["aggressive", "greedy", "safe", "unforgiving"] if p in score_df.columns]
    score_df = score_df[persona_order]

    coverage_df = pd.DataFrame(coverage_rows).set_index("model")
    coverage_df = coverage_df[persona_order]

    out_png = os.path.join(args.out_dir, args.out_png)
    out_csv = os.path.join(args.out_dir, args.out_csv)
    save_outputs(score_df, coverage_df, out_png, out_csv, args.dpi)


if __name__ == "__main__":
    main()
