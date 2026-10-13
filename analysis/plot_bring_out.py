"""
Figure: how often each player brings a new piece out of the base when it could
also move a piece on the board, with the chance level of each player.

The values are read from the logs in llm_game_results/ through
full_game_summary.load, so the figure and the tables share one source. Models
and the search player use their games against the heuristic player (main
condition); the heuristic row uses its games against the random player.

Run from the repo root:  python analysis/plot_bring_out.py OUT.pdf
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from full_game_summary import load

REFERENCE = [  # label, group key
    ("Search, depth 2", ("gt:2", "heuristic", "-")),
    ("Heuristic", ("heuristic", "random", "-")),
    ("Random", ("random", "heuristic", "-")),
    ("First-listed", ("first", "heuristic", "-")),
]
MODELS = ["qwen3:1.7b", "qwen3:4b", "qwen3:8b", "qwen3.5:9b", "llama3.1:8b",
          "gemma3:4b", "gemma3:12b"]
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"


def rates(groups, key):
    c = groups[key]["c"]
    n = c["bring_out_n"]
    return 100 * c["bring_out_took"] / n, 100 * c["bring_out_ch"] / n


def main():
    groups = load("llm_game_results", 25)
    rows = [(label, *rates(groups, key), False) for label, key in REFERENCE]
    models = [(m, *rates(groups, (m, "heuristic", "limit")), True) for m in MODELS]
    rows += sorted(models, key=lambda r: -r[1])

    fig, ax = plt.subplots(figsize=(3.4, 3.3))
    ys = list(range(len(rows)))[::-1]
    for y, (name, rate, chance, model) in zip(ys, rows):
        ax.barh(y, rate, height=0.62, color=BLUE if model else ORANGE, zorder=2)
        ax.plot([chance, chance], [y - 0.42, y + 0.42], color=INK, lw=1.6, zorder=3)
        near = chance - 12 <= rate <= chance + 2     # keep the label off the mark
        ax.text(chance + 2.5 if near else rate + 1.5, y, f"{rate:.0f}",
                va="center", ha="left", fontsize=7, color=MUTED)
        print(f"{name:<16}{rate:6.1f}{chance:6.1f}")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_xlim(0, 105)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", labelsize=7.5, colors=MUTED, length=0)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Bring-out rate (%)", fontsize=8, color=INK)
    ax.xaxis.grid(True, color="#e4e3df", lw=0.6, zorder=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.axhline(len(MODELS) - 0.5, color="#c9c8c3", lw=0.6)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ORANGE),
               plt.Rectangle((0, 0), 1, 1, color=BLUE),
               plt.Line2D([0], [0], color=INK, lw=1.6)]
    ax.legend(handles, ["Reference player", "LLM", "Chance level"], fontsize=7,
              frameon=False, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=3,
              handlelength=1.2, columnspacing=1.0)
    fig.tight_layout()
    out = sys.argv[1] if len(sys.argv) > 1 else "bring_out.pdf"
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
