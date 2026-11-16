"""
Part 4: side experiments.

side_hidden_vs_main.csv
    Models with games in both "hidden" (no legal-move list) and "main": win
    rates on their shared seeds (paired, exact McNemar) and invalid-reply rates
    (difference with a seed-bootstrap interval), Holm within each measure.
side_thinking.csv
    Models with a "spots-think" run (thinking on, the 130-spot subset of
    --per-category 10): on the same spots asked with the list shown in "spots"
    (thinking off), agreement with the search player (paired over spots with
    a valid reply and a real choice in both, exact McNemar), invalid replies
    and output tokens in each. A run still going counts the spots it has.
Figures: side_hidden_vs_main, side_thinking.
"""
from collections import defaultdict

import numpy as np

from analysis.common import figure, game_rows, games, save_figure, write_csv
from analysis.spot_tables import spot_decisions
from analysis.stats import bootstrap, holm, paired


def hidden_vs_main(runs):
    by = defaultdict(dict)
    for run in runs:
        if run.kind == "games" and run.experiment in ("main", "hidden"):
            by[run.model][run.experiment] = {r["seed"]: r for r in game_rows(run, games(run, runs)[0])}
    rows = []
    for model, exps in sorted(by.items()):
        if len(exps) < 2:
            continue
        seeds = sorted(set(exps["main"]) & set(exps["hidden"]))
        m, h = [exps["main"][s] for s in seeds], [exps["hidden"][s] for s in seeds]
        win = paired([r["won"] for r in h], [r["won"] for r in m])
        inv = bootstrap([[a["invalid"], a["calls"], b["invalid"], b["calls"]] for a, b in zip(h, m)],
                        lambda s: s[..., 0] / s[..., 1] - s[..., 2] / s[..., 3])
        rows.append({"model": model, "seeds": len(seeds), "win_hidden": win["rate_a"],
                     "win_main": win["rate_b"], "win_diff": win["diff"], "win_lo": win["diff_lo"],
                     "win_hi": win["diff_hi"], "win_p": win["p_mcnemar"],
                     "invalid_hidden": sum(r["invalid"] for r in h) / max(1, sum(r["calls"] for r in h)),
                     "invalid_main": sum(r["invalid"] for r in m) / max(1, sum(r["calls"] for r in m)),
                     "invalid_diff": inv[0], "invalid_lo": inv[1], "invalid_hi": inv[2], "invalid_p": inv[3]})
    for key in ("win", "invalid"):
        for r, p in zip(rows, holm([r[f"{key}_p"] for r in rows])):
            r[f"{key}_p_holm"] = p
    return rows


def thinking(runs):
    base = {r.model: r for r in runs if r.kind == "spots" and r.experiment == "spots"}
    rows = []
    for run in runs:
        if not (run.kind == "spots" and run.experiment == "spots-think" and run.model in base):
            continue
        think = {d["spot_id"]: d for d in spot_decisions(run)}
        plain = {d["spot_id"]: d for d in spot_decisions(base[run.model]) if d["shown"] and d["spot_id"] in think}
        both = [s for s in think if s in plain and think[s]["m"]["choice"]
                and think[s]["m"]["valid"] and plain[s]["m"]["valid"]]
        r = paired([think[s]["m"]["agree"] for s in both], [plain[s]["m"]["agree"] for s in both])
        mean = lambda ds, k: float(np.mean([d[k] for d in ds if d[k] is not None] or [np.nan]))
        rows.append({"model": run.model, "spots_asked": len(think), "spots_paired": r["n"],
                     "agree_think": r["rate_a"], "agree_no_think": r["rate_b"], "diff": r["diff"],
                     "lo": r["diff_lo"], "hi": r["diff_hi"], "p": r["p_mcnemar"],
                     "invalid_think": sum(d["was_fallback"] for d in think.values()),
                     "invalid_no_think": sum(d["was_fallback"] for d in plain.values()),
                     "output_tokens_think": mean(think.values(), "completion_tokens"),
                     "thinking_tokens_think": mean(think.values(), "thinking_tokens"),
                     "output_tokens_no_think": mean(plain.values(), "completion_tokens")})
    for r, p in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = p
    return rows


def plot(hvm, think, out):
    if hvm:
        fig, axes = figure(ncols=2, figsize=(9, 0.4 * len(hvm) + 1.5), sharey=True)
        y = np.arange(len(hvm))
        for ax, key, label in zip(axes, ("win", "invalid"), ("win rate", "invalid-reply rate")):
            for k, (cond, color) in enumerate((("main", "#4C72B0"), ("hidden", "#DD8452"))):
                ax.barh(y + 0.4 * k - 0.2, [r[f"{key}_{cond}"] for r in hvm], 0.4, color=color,
                        label=f"list {'shown' if cond == 'main' else 'hidden'}")
            ax.set_xlabel(label)
        axes[0].set_yticks(y, [r["model"] for r in hvm], fontsize=7)
        axes[0].legend(fontsize=7)
        save_figure(fig, out, "side_hidden_vs_main")
    if think:
        fig, ax = figure(figsize=(6, 0.4 * len(think) + 1.5))
        y = np.arange(len(think))
        ax.barh(y - 0.2, [r["agree_no_think"] for r in think], 0.4, color="#4C72B0", label="thinking off")
        ax.barh(y + 0.2, [r["agree_think"] for r in think], 0.4, color="#55A868", label="thinking on")
        ax.set_yticks(y, [f"{r['model']} (n={r['spots_paired']})" for r in think], fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_xlabel("agreement with the search player, same spots")
        ax.legend(fontsize=7)
        save_figure(fig, out, "side_thinking")


def run(runs, out):
    hvm, think = hidden_vs_main(runs), thinking(runs)
    write_csv(out, "side_hidden_vs_main", hvm)
    write_csv(out, "side_thinking", think)
    plot(hvm, think, out)
    return [f"side experiments: hidden vs main for {len(hvm)} models; thinking vs not for {len(think)} models"]
