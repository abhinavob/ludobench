"""
One figure style for the final analysis: a white surface, recessive grid and
axes, text in ink colours (never the series colour), a fixed categorical
order, a one-hue sequential ramp and a two-hue diverging pair with a grey
midpoint. Every figure is saved as PNG and PDF with the data it plots
(figures/<name>.csv), which stage 3 compares with the tables.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
REASONING = {"reasoning": SERIES[1], "non-reasoning": SERIES[0], "bot": MUTED}
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5",
                                                        "#1c5cab", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("bluered", ["#1c5cab", "#86b6ef", "#f0efec", "#f19a99",
                                                          "#b8302f"])

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "lines.markersize": 5, "legend.frameon": False, "pdf.fonttype": 42,
    "font.family": "DejaVu Sans", "figure.constrained_layout.use": True,
})


COLUMN = {"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7}
BLOG = {"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 14,
        "legend.fontsize": 14, "hatch.color": "white", "hatch.linewidth": 1.5}
SHORT_NAMES = {"Nemotron 3.5 Super 120B A12B": "Nemotron 3.5 Super",
               "Nemotron 3.5 Lightning 30B A3B": "Nemotron 3.5 Lightning"}


def figure(**kw):
    return plt.subplots(**kw)


def save(fig, out, name, data=None, pad=0.1, exts=("png", "pdf")):
    """Save out/figures/name.png and .pdf, or `exts` (and the plotted data as name.csv); return the base path."""
    folder = os.path.join(out, "figures")
    os.makedirs(folder, exist_ok=True)
    base = os.path.join(folder, name)
    for ext in exts:
        fig.savefig(f"{base}.{ext}", dpi=200, bbox_inches="tight", pad_inches=pad)
    plt.close(fig)
    if data is not None:
        data.to_csv(f"{base}.csv", index=False)
    return base


def reference_line(ax, x=None, y=None, label=None, style="--", where="top"):
    """A grey reference line with a direct label at its top (or bottom) end."""
    if x is not None:
        ax.axvline(x, color=MUTED, linewidth=1, linestyle=style, zorder=1)
        if label:
            top = where == "top"
            ax.annotate(label, (x, 1 if top else 0), xycoords=("data", "axes fraction"),
                        xytext=(3, -2 if top else 2), textcoords="offset points", fontsize=7, color=INK2,
                        va="top" if top else "bottom")
    if y is not None:
        ax.axhline(y, color=MUTED, linewidth=1, linestyle=style, zorder=1)
        if label:
            ax.annotate(label, (1, y), xycoords=("axes fraction", "data"), xytext=(-2, 3),
                        textcoords="offset points", fontsize=7, color=INK2, ha="right")


def label_above(ax, x, label, ha="center", fontsize=6.5):
    dx = {"center": 0, "left": 2, "right": -2}[ha]
    ax.annotate(label, (x, 1), xycoords=("data", "axes fraction"), xytext=(dx, 2), textcoords="offset points",
                fontsize=fontsize, color=INK2, ha=ha, va="bottom")


def bar_chart(out, name, data, value, xlabel, title, refs=(), hatched=None, hatch_label=None):
    """
    A web (blog) version of a ranked chart: horizontal bars, best at the top,
    coloured by reasoning group, without intervals, in large type; `hatched`
    marks rows (a boolean per row of data) and `refs` are labelled dashed
    lines (x, label, alignment). Saves the PNG and the plotted data.
    """
    import numpy as np
    from matplotlib.patches import Patch
    order = np.argsort(-data[value].to_numpy(), kind="stable")
    data = data.iloc[order]
    marks = np.zeros(len(data), bool) if hatched is None else np.asarray(hatched)[order]
    with plt.rc_context(BLOG):
        fig, ax = figure(figsize=(11, 0.34 * len(data) + 1.8))
        y = np.arange(len(data))[::-1]
        bars = ax.barh(y, data[value], height=0.72, color=[REASONING.get(g, MUTED) for g in data["reasoning_group"]],
                       edgecolor="white", linewidth=0)
        for bar, m in zip(bars, marks):
            bar.set_hatch("//" if m else None)
        ax.set_yticks(y, data["display"])
        ax.set_ylim(-0.6, len(data) - 0.4)
        ax.grid(axis="y", visible=False)
        for x, label, ha in refs:
            ax.axvline(x, color=INK2, linewidth=1.5, linestyle="--", zorder=3)
            label_above(ax, x, label, ha, fontsize=14)
        ax.set_xlim(0, np.ceil(max([data[value].max(), *(r[0] for r in refs)]) * 10) / 10)
        pct(ax)
        ax.set_xlabel(xlabel)
        ax.set_title(title, pad=30 if refs else 8, loc="left")
        handles = [Patch(color=REASONING[g], label=g.replace("-", " ")) for g in ("reasoning", "non-reasoning")]
        if marks.any():
            handles.append(Patch(facecolor=MUTED, hatch="//", edgecolor="white", linewidth=0, label=hatch_label))
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
        return save(fig, out, name, data.assign(hatched=marks), exts=("png",))


def pct(ax, axis="x"):
    from matplotlib.ticker import PercentFormatter
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(PercentFormatter(1.0, decimals=0))


def dot_plot(ax, labels, est, lo, hi, colors, xlabel, markers=None):
    """Horizontal dots with 95% intervals, one row per label (top = first); markers: one per row, default "o"."""
    import numpy as np
    y = np.arange(len(labels))[::-1]
    for yi, e, a, b, c, m in zip(y, est, lo, hi, colors, markers or ["o"] * len(labels)):
        ax.plot([a, b], [yi, yi], color=c, linewidth=1.5, alpha=0.6, solid_capstyle="round")
        ax.plot(e, yi, m, color=c, markersize=5, markeredgecolor="white", markeredgewidth=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.7, len(labels) - 0.3)


def short_dot_plot(out, name, data, top, value, lo, hi, xlabel, refs=(), band=None, named_groups=()):
    """
    A one-column (3 in) version of a ranked dot plot: the first `top` rows of
    data (best first), and every row of a reasoning group in named_groups,
    named with their 95% intervals; the rest as one row of unlabelled dots.
    Reference lines (x, label, style, alignment) and a shaded band (lo, hi,
    label) are labelled above the axes. Saves the plotted data with the label
    shown and whether the row is named.
    """
    import numpy as np
    from matplotlib.ticker import MaxNLocator
    data = data.assign(label=[SHORT_NAMES.get(d, d) for d in data["display"]],
                       named=(np.arange(len(data)) < top) | data["reasoning_group"].isin(named_groups).to_numpy())
    color = lambda x: [REASONING.get(g, MUTED) for g in x["reasoning_group"]]
    named, rest = data[data["named"]], data[~data["named"]]
    n = len(named)
    with plt.rc_context(COLUMN):
        fig, ax = figure(figsize=(3.0, 0.16 * (n + 1) + 0.9))
        dot_plot(ax, list(named["label"]), named[value], named[lo], named[hi], color(named), xlabel)
        ax.scatter(rest[value], [-1.3] * len(rest), s=14, c=color(rest), edgecolors="white", linewidths=0.5,
                   zorder=3)
        ax.set_yticks(list(range(n))[::-1] + [-1.3], list(named["label"]) + [f"{len(rest)} other models"])
        ax.set_ylim(-2, n - 0.3)
        for x, label, style, ha in refs:
            reference_line(ax, x=x, style=style)
            label_above(ax, x, label, ha)
        if band:
            ax.axvspan(band[0], band[1], color=GRID, alpha=0.8, zorder=0)
            ax.annotate(band[2], ((band[0] + band[1]) / 2, 1), xycoords=("data", "axes fraction"), xytext=(0, -3),
                        textcoords="offset points", ha="center", va="top", rotation=90, fontsize=6.5, color=INK2)
        ends = [data[value].min(), named[lo].min(), named[hi].max(), *(r[0] for r in refs), *(band or ())[:2]]
        ax.set_xlim(np.floor(min(ends) * 10) / 10, np.ceil(max(ends) * 10) / 10)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
        pct(ax)
        for g in ("non-reasoning", "reasoning"):
            ax.plot([], [], "o", color=REASONING[g], label=g.replace("-", " "))
        fig.legend(loc="outside lower center", ncol=2)
        return save(fig, out, name, data, pad=0.02)


def label_points(ax, xs, ys, labels, fontsize=6, first=None):
    """
    Direct labels beside points, nudged up or down where they would overlap
    (axes-fraction boxes kept inside the axes); `first` maps a label to the
    vertical nudge to try before the others, for clusters where the default
    order lets a label sit on a neighbour's point. A point with no free place
    is left unlabelled. Returns the labels left out.
    """
    fig = ax.figure
    fig.canvas.draw()
    to_axes = ax.transData + ax.transAxes.inverted()
    placed, left_out = [], []
    for x, y, text in sorted(zip(xs, ys, labels), key=lambda t: (-t[1], t[0])):
        ax_x, ax_y = to_axes.transform((x, y))
        width = 0.0105 * len(text) * fontsize / 6
        free = None
        for dy in ((first or {}).get(text, 0), 0, 0.03, -0.03, 0.06, -0.06, 0.09, -0.09):
            box = (min(ax_x + 0.008, 0.99 - width), ax_y + dy)
            if 0.01 < box[1] < 0.99 and all(abs(box[1] - b[1]) > 0.028 or box[0] > b[0] + b[2] or
                                            box[0] + width < b[0] for b in placed):
                free = box
                break
        if free is None:
            left_out.append(text)
            continue
        placed.append((free[0], free[1], width))
        ax.annotate(text, (x, y), xytext=free, textcoords="axes fraction", fontsize=fontsize, color=INK2,
                    va="center", arrowprops=dict(arrowstyle="-", color=GRID, linewidth=0.6) if dy else None)
    return left_out


def note_left_out(fig, left_out):
    """A small note under the figure naming the points left unlabelled."""
    if left_out:
        fig.text(0.01, -0.01, "Not labelled (overlapping): " + ", ".join(left_out), fontsize=6, color=INK2,
                 va="top", ha="left", wrap=True)
