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

# Pair rule format:
# (metric_a, expected_sign_a, metric_b, expected_sign_b)
# expected_sign: +1 means should increase vs none, -1 should decrease vs none.
PAIR_RULES: Dict[str, List[Tuple[str, int, str, int]]] = {
    "aggressive": [
        ("capture.capture_rate", +1, "safe.safe_rate", -1),
        ("capture_vs_home.capture_rate", +1, "capture_vs_home.home_entry_rate", -1),
        ("capture_vs_openexisting.capture_rate", +1, "capture_vs_openexisting.open_rate", -1),
    ],
    "greedy": [
        ("capture_vs_home_finish.home_finish_rate", +1, "capture_vs_home_finish.capture_rate", -1),
        ("home_entry.home_entry_rate", +1, "capture.capture_rate", -1),
        ("capture_vs_home.home_entry_rate", +1, "capture_vs_home.capture_rate", -1),
    ],
    "safe": [
        ("safe.safe_rate", +1, "capture.capture_rate", -1),
        ("capture_vs_safe.safe_rate", +1, "capture_vs_safe.capture_rate", -1),
        ("safe_vs_openexisting.safe_rate", +1, "capture_vs_openexisting.capture_rate", -1),
    ],
    "unforgiving": [
        ("grudge.change_rate", +1, "grudge.retaliation_grudge_rate", +1),
        ("grudge.retaliation_grudge_rate", +1, "capture.capture_rate", +1),
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


def load_delta_table(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    persona_cols = [c for c in df.columns if c not in META_COLS]
    if "none" not in persona_cols:
        raise ValueError(f"Missing 'none' column in {csv_path}")

    df["metric_key"] = df["category"].astype(str) + "." + df["metric"].astype(str)
    slim = df[["metric_key"] + persona_cols].drop_duplicates(subset=["metric_key"], keep="first").set_index("metric_key")
    for c in persona_cols:
        slim[c] = pd.to_numeric(slim[c], errors="coerce")

    deltas = {}
    for p in persona_cols:
        if p == "none":
            continue
        deltas[p] = slim[p] - slim["none"]
    return pd.DataFrame(deltas)


def eval_pair_contradiction(
    delta_a: float, sign_a: int, delta_b: float, sign_b: int, eps: float
) -> Tuple[bool, bool]:
    """
    Returns:
      usable: if both deltas are confidently directional (outside epsilon dead-zone)
      contradiction: True if usable and pair does not satisfy both expected directions
    """
    if pd.isna(delta_a) or pd.isna(delta_b):
        return False, False

    if abs(delta_a) < eps or abs(delta_b) < eps:
        return False, False

    ok_a = (sign_a * float(delta_a)) > 0
    ok_b = (sign_b * float(delta_b)) > 0
    contradiction = not (ok_a and ok_b)
    return True, contradiction


def compute_model_persona_scores(
    model: str, deltas: pd.DataFrame, personas: List[str], eps: float
) -> List[dict]:
    rows = []
    for persona in personas:
        rules = PAIR_RULES.get(persona, [])
        if persona not in deltas.columns:
            rows.append(
                {
                    "model": model,
                    "persona": persona,
                    "contradiction_rate": float("nan"),
                    "coherence_score": float("nan"),
                    "contradictions": 0,
                    "usable_pairs": 0,
                    "total_rules": len(rules),
                }
            )
            continue

        contradictions = 0
        usable = 0
        for m_a, s_a, m_b, s_b in rules:
            if m_a not in deltas.index or m_b not in deltas.index:
                continue
            delta_a = deltas.at[m_a, persona]
            delta_b = deltas.at[m_b, persona]
            is_usable, is_contra = eval_pair_contradiction(delta_a, s_a, delta_b, s_b, eps)
            if not is_usable:
                continue
            usable += 1
            if is_contra:
                contradictions += 1

        rate = (contradictions / usable) if usable > 0 else float("nan")
        rows.append(
            {
                "model": model,
                "persona": persona,
                "contradiction_rate": rate,
                "coherence_score": (1.0 - rate) if usable > 0 else float("nan"),
                "contradictions": contradictions,
                "usable_pairs": usable,
                "total_rules": len(rules),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Persona contradiction matrix: lower contradiction rate means more coherent persona behavior."
    )
    parser.add_argument("--root", default="results_for_40", help="Root containing model folders with persona_comparison.csv")
    parser.add_argument("--models", default="", help="Optional comma-separated model folder names")
    parser.add_argument("--personas", default="aggressive,greedy,safe,unforgiving", help="Comma-separated personas")
    parser.add_argument("--eps", type=float, default=0.01, help="Dead-zone threshold for tiny deltas")
    parser.add_argument("--out-dir", default="results_for_40/model_wise_figures", help="Output directory")
    parser.add_argument("--out-csv", default="persona_contradiction_scores.csv", help="Output CSV name")
    parser.add_argument("--out-png", default="persona_contradiction_matrix.png", help="Output PNG name")
    parser.add_argument("--dpi", type=int, default=360, help="PNG DPI")
    args = parser.parse_args()

    model_filter = [m.strip() for m in args.models.split(",") if m.strip()]
    personas = [p.strip() for p in args.personas.split(",") if p.strip()]
    model_csvs = discover_model_csvs(args.root, model_filter)
    if not model_csvs:
        raise SystemExit(f"No model persona_comparison.csv found under {args.root}")

    rows = []
    for model, csv_path in model_csvs:
        delta_table = load_delta_table(csv_path)
        rows.extend(compute_model_persona_scores(model, delta_table, personas, args.eps))

    out_df = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, args.out_csv)
    out_png = os.path.join(args.out_dir, args.out_png)
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")

    piv = out_df.pivot(index="model", columns="persona", values="contradiction_rate")
    piv = piv.reindex(columns=[p for p in personas if p in piv.columns])

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(8.8, max(4.6, 0.78 * len(piv.index))))
    sns.heatmap(
        piv,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.45,
        linecolor="#EFEFEF",
        cbar_kws={"label": "Contradiction Rate (lower is better)"},
        ax=ax,
    )
    ax.set_title("Persona Contradiction Matrix")
    ax.set_xlabel("Persona")
    ax.set_ylabel("Model")
    fig.tight_layout()
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()
