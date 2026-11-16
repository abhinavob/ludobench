"""
Part 3: the four hypotheses, per model. Decisions count only with at least 2
distinct moves (equivalent moves grouped) and a valid reply.

H1 first-listed move. Primary, main games (list shown in ascending order):
   (a) the share of choices in the first-listed move's group against its
       chance (the group's share of the legal moves), game-clustered
       bootstrap; (b) the model's win rate against the first-listed bot's on
       shared seeds, exact McNemar.
H2 bring-out under-use. Primary, main games: where a bring-out is offered,
   the model's bring-out rate against (a) the random policy's and (b) the
   search player's on the same decisions, game-clustered bootstrap.
H3 shuffled order. The same spots with the list ascending (experiment
   "spots") and shuffled ("spots-shuffle"): does the shuffled choice keep the
   list position of the ascending choice, or the move (its group)? Exact
   McNemar on the spots where exactly one holds; the difference of the two
   rates with a spot-bootstrap interval.
H4 move outcomes. The same spots without and with the outcome lines
   ("spots-outcomes"): the paired change in agreement with the search player,
   exact McNemar, spot-bootstrap interval.
Secondary: H1 (a) and H2 on the spots with the list shown, clustered by spot
(a grudge pair is one cluster).

Holm correction within the primary and within the secondary family. Tables:
hypotheses.csv; figure: hypotheses (differences with 95% intervals).
"""
from collections import defaultdict

import numpy as np

from analysis.common import decisions, figure, games, save_figure, write_csv
from analysis.main_games import run_table
from analysis.positions import complete, measures
from analysis.spot_tables import spot_decisions
from analysis.stats import bootstrap, holm, paired


def against(ds, rate, ref, cluster):
    """Model rate minus a reference rate over decisions, with a cluster bootstrap."""
    sums = defaultdict(lambda: np.zeros(3))
    for d in ds:
        sums[cluster(d)] += (rate(d), ref(d), 1)
    s = np.array(list(sums.values()))
    est, lo, hi, p = bootstrap(s, lambda t: (t[..., 0] - t[..., 1]) / t[..., 2])
    n = s[:, 2].sum() if len(s) else 0
    return {"n": int(n), "clusters": len(s), "rate": s[:, 0].sum() / n if n else None,
            "reference": s[:, 1].sum() / n if n else None, "diff": est, "lo": lo, "hi": hi, "p": p}


def usable(ds):
    return [d for d in ds if d["m"]["choice"] and d["m"]["valid"]]


def game_decisions(run, runs):
    out = []
    for d in decisions(run, games(run, runs)[0]):
        if d["shown"]:
            complete(d)
            d["m"] = measures(d)
            out.append(d)
    return usable(out)


def spot_cluster(d):
    return d["pair_id"] or d["spot_id"]


def first_and_bring_out(model, ds, cluster, data):
    rows = [{"hypothesis": "H1 first-listed", "test": "first-listed move vs chance",
             **against(ds, lambda d: d["m"]["first"], lambda d: d["m"]["chance_first"], cluster)}]
    out = [d for d in ds if d["m"]["open"]["offered"]]
    for ref, name in (("chance", "random policy"), ("search", "search player")):
        rows.append({"hypothesis": "H2 bring-out", "test": f"bring-out rate vs {name}",
                     **against(out, lambda d: d["m"]["open"]["taken"], lambda d: d["m"]["open"][ref], cluster)})
    return [{"model": model, "data": data, **r} for r in rows]


def by_spot(ds):
    return {(d["spot_id"], d["shown"]): d for d in usable(ds)}


def shuffle_rows(model, base, shuffled):
    """H3: does a shuffled choice keep the list position or the move of the ascending choice?"""
    pos, piece = [], []
    for key, s in shuffled.items():
        b = base.get(key)
        if b is None:
            continue
        group = {t: i for i, g in enumerate(s["move_groups"]) for t in g}
        at = s["shown_legal_moves"][b["shown_legal_moves"].index(b["chosen_index"])]
        pos.append(int(group[s["chosen_index"]] == group[at]))
        piece.append(int(group[s["chosen_index"]] == group[b["chosen_index"]]))
    r = paired(pos, piece)
    return {"model": model, "data": "spots vs spots-shuffle", "hypothesis": "H3 shuffled order",
            "test": "same list position vs same move", "n": r["n"], "clusters": r["n"],
            "rate": r["rate_a"], "reference": r["rate_b"], "diff": r["diff"], "lo": r["diff_lo"],
            "hi": r["diff_hi"], "p": r["p_mcnemar"], "only_position": r["only_a"], "only_move": r["only_b"]}


def outcomes_rows(model, base, with_outcomes):
    keys = [k for k in with_outcomes if k in base]
    r = paired([with_outcomes[k]["m"]["agree"] for k in keys], [base[k]["m"]["agree"] for k in keys])
    return {"model": model, "data": "spots vs spots-outcomes", "hypothesis": "H4 move outcomes",
            "test": "agreement with outcomes vs without", "n": r["n"], "clusters": r["n"],
            "rate": r["rate_a"], "reference": r["rate_b"], "diff": r["diff"], "lo": r["diff_lo"],
            "hi": r["diff_hi"], "p": r["p_mcnemar"]}


def plot(rows, out):
    rows = [r for r in rows if r["family"] == "primary" and r["n"]]
    if not rows:
        return
    tests = list(dict.fromkeys(r["test"] for r in rows))
    fig, axes = figure(ncols=len(tests), figsize=(3.2 * len(tests), 0.3 * len({r["model"] for r in rows}) + 2),
                       sharey=True, squeeze=False)
    models = sorted({r["model"] for r in rows})
    for ax, test in zip(axes[0], tests):
        sel = {r["model"]: r for r in rows if r["test"] == test}
        for i, m in enumerate(models):
            if m in sel:
                r = sel[m]
                ax.errorbar(r["diff"], i, xerr=[[r["diff"] - r["lo"]], [r["hi"] - r["diff"]]], fmt="o",
                            color="#C44E52" if r["p_holm"] < 0.05 else "#4C72B0", ms=3, capsize=2)
        ax.axvline(0, color="0.5", linewidth=0.8)
        ax.set_title(test, fontsize=8)
        ax.set_yticks(range(len(models)), models, fontsize=7)
        ax.tick_params(axis="x", labelsize=7)
    fig.supxlabel("difference with 95% interval (red: Holm p < 0.05)", fontsize=8)
    save_figure(fig, out, "hypotheses")


def run(runs, out):
    rows = []
    main = [r for r in runs if r.kind == "games" and r.experiment == "main"]
    _, won = run_table(main, runs)
    _, bots = run_table([r for r in runs if r.kind == "games" and r.experiment == "bots"
                         and r.model == "first"], runs)
    for run in main:
        rows += [dict(r, family="primary") for r in
                 first_and_bring_out(run.model, game_decisions(run, runs), lambda d: d["game"], "main games")]
        shared = sorted(set(won.get(run.model, {})) & set(bots.get("first", {})))
        if shared:
            r = paired([won[run.model][s] for s in shared], [bots["first"][s] for s in shared])
            rows.append({"model": run.model, "data": "main games vs first-listed bot", "family": "primary",
                         "hypothesis": "H1 first-listed", "test": "win rate vs first-listed bot",
                         "n": r["n"], "clusters": r["n"], "rate": r["rate_a"], "reference": r["rate_b"],
                         "diff": r["diff"], "lo": r["diff_lo"], "hi": r["diff_hi"], "p": r["p_mcnemar"]})
    spots = defaultdict(dict)
    for run in runs:
        if run.kind == "spots" and run.experiment in ("spots", "spots-shuffle", "spots-outcomes"):
            spots[run.model][run.experiment] = spot_decisions(run)
    for model, exps in sorted(spots.items()):
        base = by_spot(exps.get("spots", []))
        shown = [d for d in usable(exps.get("spots", [])) if d["shown"]]
        rows += [dict(r, family="secondary") for r in
                 first_and_bring_out(model, shown, spot_cluster, "spots (list shown)")]
        if "spots-shuffle" in exps:
            rows.append(dict(shuffle_rows(model, base, by_spot(exps["spots-shuffle"])), family="primary"))
        if "spots-outcomes" in exps:
            rows.append(dict(outcomes_rows(model, base, by_spot(exps["spots-outcomes"])), family="primary"))
    for family in ("primary", "secondary"):
        sel = [r for r in rows if r["family"] == family]
        for r, p in zip(sel, holm([r["p"] if r["n"] else np.nan for r in sel])):
            r["p_holm"] = p
    write_csv(out, "hypotheses", rows)
    plot(rows, out)
    primary = [r for r in rows if r["family"] == "primary"]
    return [f"hypotheses: {sum(r['p_holm'] < 0.05 for r in primary)} of {len(primary)} primary tests "
            f"significant after Holm"]
