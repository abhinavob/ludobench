import argparse
import csv
import os
import statistics
import tempfile

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── Display names (title-cased for paper) ────────────────────────────
DISPLAY_NAME = {
    "claude-3.5-haiku": "Claude-3.5-Haiku",
    "deepseek-chat": "DeepSeek-Chat",
    "gemma-3-12b-it": "Gemma-3-12B",
    "llama-4-scout": "Llama-4-Scout",
    "qwen-2.5-7b-instruct": "Qwen-2.5-7B",
    "qwen-plus": "Qwen-Plus",
}

# ── Markers: distinct shapes (B&W-safe for print) ───────────────────
MARKERS = {
    "claude-3.5-haiku": "o",
    "deepseek-chat": "s",
    "gemma-3-12b-it": "D",
    "llama-4-scout": "P",
    "qwen-2.5-7b-instruct": "^",
    "qwen-plus": "X",
}

# ── Colours: muted, print-safe, colour-blind-friendly ───────────────
MODEL_COLOURS = {
    "claude-3.5-haiku": "#1f77b4",
    "deepseek-chat": "#ff7f0e",
    "gemma-3-12b-it": "#2ca02c",
    "llama-4-scout": "#d62728",
    "qwen-2.5-7b-instruct": "#9467bd",
    "qwen-plus": "#8c564b",
}


# =====================================================================
#  Logic helpers — UNCHANGED
# =====================================================================

def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def read_metrics(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                {
                    "model": row["model"],
                    "bring_out_rate": float(row["bring_out_rate"]),
                    "home_finish_rate": float(row["home_finish_rate"]),
                    "capture_rate": float(row["capture_rate"]),
                    "safe_rate": float(row["safe_rate"]),
                    "block_rate": float(row["block_rate"]),
                }
            )
    return rows


def assign_archetype(x, y, x_mid, y_mid):
    if x >= x_mid and y < y_mid:
        return "Builder-leaning"
    if x < x_mid and y >= y_mid:
        return "Finisher-leaning"
    if x >= x_mid and y >= y_mid:
        return "Dual-high"
    return "Dual-low"


def write_points_csv(path, points):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "bring_out_rate",
                "home_finish_rate",
                "capture_rate",
                "safe_rate",
                "block_rate",
                "archetype",
            ]
        )
        for p in points:
            w.writerow(
                [
                    p["model"],
                    p["bring_out_rate"],
                    p["home_finish_rate"],
                    p["capture_rate"],
                    p["safe_rate"],
                    p["block_rate"],
                    p["archetype"],
                ]
            )


# =====================================================================
#  Plot — research-paper style (style-only changes)
# =====================================================================

def plot_scatter(points, gt, x_mid, y_mid, out_png):
    ensure_dir(os.path.dirname(out_png))

    # ── RC params: serif font, inward ticks (LaTeX / IEEE / ACL style) ─
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
    })

    fig, ax = plt.subplots(figsize=(7, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # All four spines visible, thin black
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.7)

    # Subtle dotted grid behind data
    ax.grid(True, linestyle=":", linewidth=0.35, color="#C0C0C0", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # ── Median split lines ───────────────────────────────────────────
    ax.axvline(x_mid, color="grey", ls="--", lw=0.8, alpha=0.55, zorder=1)
    ax.axhline(y_mid, color="grey", ls="--", lw=0.8, alpha=0.55, zorder=1)

    # ── Quadrant labels (small, italic, grey — unobtrusive) ──────────
    ql = dict(fontsize=8, fontstyle="italic", color="#777777")
    pad = 0.03
    ax.text(pad, 1.07 - pad, "Finisher-leaning", va="top", ha="left", **ql)
    ax.text(1.07 - pad, 1.07 - pad, "Dual-high", va="top", ha="right", **ql)
    ax.text(pad, -0.03 + pad, "Dual-low", va="bottom", ha="left", **ql)
    ax.text(1.07 - pad, -0.03 + pad, "Builder-leaning", va="bottom", ha="right", **ql)

    # ── Plot models ──────────────────────────────────────────────────
    point_rows = sorted(points, key=lambda p: p["model"])

    # Hand-tuned label offsets: (text_x, text_y, ha, va)
    label_pos = {
        "claude-3.5-haiku":      (0.14,  0.94, "left", "bottom"),
        "qwen-plus":             (0.40,  0.74, "left", "top"),
        "qwen-2.5-7b-instruct": (0.40,  1.07, "left", "bottom"),
        "llama-4-scout":        (0.52,  0.17, "left", "bottom"),
        "gemma-3-12b-it":       (0.60,  0.09, "left", "bottom"),
        "deepseek-chat":        (0.68, -0.01, "left", "top"),
    }

    for p in point_rows:
        x = p["bring_out_rate"]
        y = p["home_finish_rate"]
        c = MODEL_COLOURS.get(p["model"], "#333")
        m = MARKERS.get(p["model"], "o")
        name = DISPLAY_NAME.get(p["model"], p["model"])

        ax.scatter(
            x, y,
            s=80,
            marker=m,
            facecolors=c,
            edgecolors="black",
            linewidths=0.45,
            zorder=3,
            label=name,
        )

        lp = label_pos.get(p["model"])
        if lp:
            tx, ty, ha, va = lp
            ax.annotate(
                name,
                xy=(x, y),
                xytext=(tx, ty),
                fontsize=7.5,
                color="black",
                ha=ha, va=va,
                arrowprops=dict(
                    arrowstyle="-",
                    color="#999999",
                    lw=0.5,
                    shrinkA=2, shrinkB=2,
                    connectionstyle="arc3,rad=0.10",
                ),
                zorder=4,
            )

    # ── GT reference ─────────────────────────────────────────────────
    gt_x, gt_y = gt["bring_out_rate"], gt["home_finish_rate"]
    ax.scatter(
        gt_x, gt_y,
        marker="*", s=220,
        facecolors="black", edgecolors="black",
        linewidths=0.3, zorder=5,
        label="GT (Optimal)",
    )
    ax.annotate(
        "GT",
        xy=(gt_x, gt_y),
        xytext=(gt_x - 0.04, gt_y - 0.07),
        fontsize=8.5, fontweight="bold", color="black",
        ha="right", va="top",
        arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
        zorder=5,
    )

    # ── Axes ─────────────────────────────────────────────────────────
    ax.set_xlim(-0.03, 1.09)
    ax.set_ylim(-0.05, 1.10)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax.tick_params(axis="both", which="major", labelsize=9, length=4)
    ax.tick_params(axis="both", which="minor", length=2)

    ax.set_xlabel(r"$\mathrm{bring\_out\_rate}$" + "  (development tendency)",
                  fontsize=10.5, labelpad=5)
    ax.set_ylabel(r"$\mathrm{home\_finish\_rate}$" + "  (completion tendency)",
                  fontsize=10.5, labelpad=5)
    ax.set_title("Archetype Clustering: Builder vs. Finisher (no persona)",
                 fontsize=11.5, pad=8)

    # ── Legend (compact, boxed) ──────────────────────────────────────
    leg = ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.01, 0.08),
        fontsize=7.5,
        frameon=True,
        framealpha=0.92,
        edgecolor="#BBBBBB",
        fancybox=False,
        borderpad=0.5,
        labelspacing=0.45,
        handletextpad=0.35,
        handlelength=1.0,
        ncol=1,
        markerscale=0.9,
    )
    leg.get_frame().set_linewidth(0.4)

    # ── Save ─────────────────────────────────────────────────────────
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    # Vector PDF for direct LaTeX \includegraphics
    pdf_path = out_png.replace(".png", ".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] {out_png}")
    print(f"[OK] {pdf_path}")


# =====================================================================
#  Main — UNCHANGED
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot builder-vs-finisher archetype clustering from behavioral radar metrics."
    )
    parser.add_argument(
        "--metrics-csv",
        default="results_for_40/non_persona_eval/behavioral_radar_metrics.csv",
        help="Input metrics CSV from plot_gt_behavioral_radar.py",
    )
    parser.add_argument(
        "--out-dir",
        default="results_for_40/non_persona_eval",
        help="Output directory",
    )
    args = parser.parse_args()

    rows = read_metrics(args.metrics_csv)
    gt = next((r for r in rows if r["model"] == "GT"), None)
    points = [r for r in rows if r["model"] != "GT"]
    if not gt or not points:
        raise SystemExit("Need GT row and at least one model row in metrics CSV.")

    x_mid = statistics.median([p["bring_out_rate"] for p in points])
    y_mid = statistics.median([p["home_finish_rate"] for p in points])

    for p in points:
        p["archetype"] = assign_archetype(p["bring_out_rate"], p["home_finish_rate"], x_mid, y_mid)

    gt_label = assign_archetype(gt["bring_out_rate"], gt["home_finish_rate"], x_mid, y_mid)

    points_csv = os.path.join(args.out_dir, "archetype_clustering_points.csv")
    summary_txt = os.path.join(args.out_dir, "archetype_clustering_summary.txt")
    plot_png = os.path.join(args.out_dir, "archetype_builder_finisher_scatter.png")

    write_points_csv(points_csv, points)
    plot_scatter(points, gt, x_mid, y_mid, plot_png)

    ensure_dir(os.path.dirname(summary_txt))
    with open(summary_txt, "w") as f:
        f.write("Archetype Clustering Summary\n")
        f.write("===========================\n\n")
        f.write(f"Median bring_out_rate (models): {x_mid:.4f}\n")
        f.write(f"Median home_finish_rate (models): {y_mid:.4f}\n")
        f.write(f"GT archetype (relative to medians): {gt_label}\n\n")
        counts = {}
        for p in points:
            counts[p["archetype"]] = counts.get(p["archetype"], 0) + 1
        f.write("Model counts by archetype:\n")
        for k in sorted(counts.keys()):
            f.write(f"- {k}: {counts[k]}\n")

    print(f"Wrote: {points_csv}")
    print(f"Wrote: {plot_png}")
    print(f"Wrote: {summary_txt}")


if __name__ == "__main__":
    main()