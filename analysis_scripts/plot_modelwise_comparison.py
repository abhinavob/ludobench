import argparse
import os
import tempfile
from typing import List, Tuple
from textwrap import fill

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


META_COLS = {"category", "metric"}
DEFAULT_EXCLUDED_DIRS = {
    "model_wise_figures",
    "non_persona_eval",
    "game-theory-agent",
    "heuristic-agent",
    "random-agent",
}


def choose_persona_csv(model_dir: str, model_name: str) -> str:
    generic = os.path.join(model_dir, "persona_comparison.csv")
    if os.path.isfile(generic):
        return generic

    candidates = []
    for fname in os.listdir(model_dir):
        if not fname.startswith("persona_comparison") or not fname.endswith(".csv"):
            continue
        candidates.append(fname)

    if not candidates:
        return ""

    exact_name = f"persona_comparison_{model_name}.csv"
    if exact_name in candidates:
        return os.path.join(model_dir, exact_name)

    containing = [f for f in candidates if model_name in f]
    if containing:
        containing.sort(key=len, reverse=True)
        return os.path.join(model_dir, containing[0])

    candidates.sort(key=len, reverse=True)
    return os.path.join(model_dir, candidates[0])


def discover_model_csvs(root: str, model_filter: List[str]) -> List[Tuple[str, str]]:
    rows = []
    wanted = set(model_filter)
    for name in sorted(os.listdir(root)):
        model_dir = os.path.join(root, name)
        if not os.path.isdir(model_dir):
            continue
        if not wanted and name in DEFAULT_EXCLUDED_DIRS:
            continue
        if wanted and name not in wanted:
            continue
        csv_path = choose_persona_csv(model_dir, name)
        if csv_path:
            rows.append((name, csv_path))
    return rows


def model_reliability_from_csv(csv_path: str) -> float:
    df = pd.read_csv(csv_path)
    persona_cols = [c for c in df.columns if c not in META_COLS]
    if not persona_cols:
        raise ValueError(f"No persona columns found in {csv_path}")

    invalid_rows = df[df["metric"] == "llm_invalid_rate"]
    if invalid_rows.empty:
        raise ValueError(f"No llm_invalid_rate rows found in {csv_path}")

    vals = []
    for p in persona_cols:
        series = pd.to_numeric(invalid_rows[p], errors="coerce").dropna()
        vals.extend(series.tolist())

    if not vals:
        raise ValueError(f"No numeric invalid-rate values found in {csv_path}")
    return float(np.mean(vals))


def plot_reliability(models: List[str], avg_invalid: List[float], out_png: str, dpi: int) -> None:
    avg_reliability = [1.0 - v for v in avg_invalid]
    x = np.arange(len(models), dtype=float)
    width = 0.34

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["axes.labelsize"] = 15
    plt.rcParams["xtick.labelsize"] = 13
    plt.rcParams["ytick.labelsize"] = 13

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.45), 5.5))
    bars_invalid = ax.bar(
        x - width / 2,
        avg_invalid,
        width=width,
        color="#F28E8B",
        edgecolor="#C95E5A",
        linewidth=0.8,
        label="Average Invalid Rate",
        zorder=3,
    )
    bars_rel = ax.bar(
        x + width / 2,
        avg_reliability,
        width=width,
        color="#7CC47B",
        edgecolor="#4F9E51",
        linewidth=0.8,
        label="Average Reliability (1 - Invalid Rate)",
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([fill(m, width=14) for m in models], rotation=0, ha="center")
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_ylabel("Rate")
    ax.set_xlabel("LLM Model")
    ax.grid(axis="y", alpha=0.2, linewidth=0.7, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.165),
        ncol=2,
        frameon=False,
        fontsize=12,
    )

    for bars in (bars_invalid, bars_rel):
        for b in bars:
            h = b.get_height()
            ax.annotate(
                f"{h:.3f}",
                (b.get_x() + b.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=11,
                color="#333333",
            )

    fig.tight_layout(rect=[0, 0, 1, 0.90], pad=1.1)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create one publication-quality PNG with per-model average invalid rate and average reliability "
            "(1-invalid), averaged over all scenarios and all personas."
        )
    )
    parser.add_argument("--root", default="results_for_40", help="Root containing model/agent folders with persona comparison CSVs")
    parser.add_argument("--models", default="", help="Optional comma-separated model folder names to include")
    parser.add_argument(
        "--out-png",
        default="results_for_40/model_wise_figures/model_reliability_summary.png",
        help="Output PNG path",
    )
    parser.add_argument("--dpi", type=int, default=360, help="PNG DPI")
    args = parser.parse_args()

    model_filter = [m.strip() for m in args.models.split(",") if m.strip()]
    model_csvs = discover_model_csvs(args.root, model_filter)
    if not model_csvs:
        raise SystemExit(f"No persona comparison CSVs found under {args.root}")

    models = []
    avg_invalid = []
    for model_name, csv_path in model_csvs:
        v = model_reliability_from_csv(csv_path)
        models.append(model_name)
        avg_invalid.append(v)

    plot_reliability(models, avg_invalid, args.out_png, args.dpi)
    print(f"Wrote: {args.out_png}")


if __name__ == "__main__":
    main()
