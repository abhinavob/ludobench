"""
Part 3: the four preregistered hypotheses (recorded 11 Nov 2026, before the
decision-level analysis), per model.

Common: only decisions with at least 2 distinct moves (equivalent moves
grouped) and a valid reply; chance = 1 / number of distinct moves, averaged
over decisions; alpha 0.05 with Holm correction across every test of the four
hypotheses; game-clustered bootstrap for decision-level metrics on games;
paired comparisons on the same decisions, seeds or spots. p-values are
two-sided; "result" reads each against the hypothesis's direction.

H1 Non-thinking models (run.json says the player does not think, or says
   nothing and no reply carries reasoning tokens), main games, list shown:
   (a) the first-listed move is chosen more often than chance;
   (b) agreement with the search player is no higher than that of a bot that
       always plays the first-listed move, on the same decisions (paired,
       game-clustered bootstrap); refuted if significantly above.
H2 Main games, a 6 offering a bring-out and another move: models bring a piece
   out less often than (a) chance (the share of distinct moves that bring a
   piece out) and (b) the search player on the same decisions.
H3 The same spots with the list in ascending order ("spots", shown) and
   shuffled ("spots-shuffle"): does the shuffled choice keep the list position
   of the original choice, or the piece (its move)? Same-position against
   same-piece choices, exact McNemar; refuted if same-piece choices are
   significantly more frequent.
H4 Outcome lines change choices and raise agreement with the search player: on
   the same spots without and with them ("spots-outcomes"), the share of
   changed choices and the change in agreement (exact McNemar); on games
   ("outcomes" against "main", shared seeds) the change in agreement, paired
   bootstrap over seeds.
Exploratory, not preregistered and outside the Holm family: under shuffle,
among equivalent moves, does the model pick the one listed first or the
lowest-numbered one (where they differ)? Exact McNemar.

Table: hypotheses.csv; figure: hypotheses (differences with 95% intervals).
"""
from collections import defaultdict

import numpy as np

from analysis.common import decisions, figure, games, save_figure, write_csv
from analysis.positions import complete, measures
from analysis.spot_tables import spot_decisions
from analysis.stats import bootstrap, holm, paired

ALPHA = 0.05


def against(ds, rate, ref, cluster=lambda d: d["game"]):
    """Mean of rate minus mean of ref over the same decisions, with a cluster bootstrap."""
    sums = defaultdict(lambda: np.zeros(3))
    for d in ds:
        sums[cluster(d)] += (rate(d), ref(d), 1)
    s = np.array(list(sums.values())).reshape(-1, 3)
    est, lo, hi, p = bootstrap(s, lambda t: (t[..., 0] - t[..., 1]) / t[..., 2])
    n = s[:, 2].sum()
    return {"n": int(n), "clusters": len(s), "rate": s[:, 0].sum() / n if n else None,
            "reference": s[:, 1].sum() / n if n else None, "diff": est, "lo": lo, "hi": hi, "p": p}


def usable(ds):
    return [d for d in ds if d["m"]["choice"] and d["m"]["valid"]]


def game_decisions(run, runs):
    """The run's usable decisions with the list shown, completed and measured."""
    out = []
    for d in decisions(run, games(run, runs)[0]):
        if d["shown"]:
            complete(d)
            d["m"] = measures(d)
            out.append(d)
    return usable(out)


def non_thinking(run, ds):
    thinks = run.meta["players"][str(run.pid)].get("thinks")
    return not thinks if thinks is not None else not any((d["thinking_tokens"] or 0) > 0 for d in ds)


def h1(ds):
    return [{"hypothesis": "H1", "test": "first-listed move vs chance", "expected": "+",
             **against(ds, lambda d: d["m"]["first"], lambda d: d["m"]["chance"])},
            {"hypothesis": "H1", "test": "agreement vs first-listed bot", "expected": "not +",
             **against(ds, lambda d: d["m"]["agree"], lambda d: d["m"]["first_agree"])}]


def h2(ds):
    six = [d for d in ds if d["dice"] == 6 and d["m"]["open"]["offered"]]
    return [{"hypothesis": "H2", "test": f"bring-out rate vs {name}", "expected": "-",
             **against(six, lambda d: d["m"]["open"]["taken"], lambda d, k=key: d["m"]["open"][k])}
            for key, name in (("chance", "chance"), ("search", "search player"))]


def by_spot(ds):
    return {d["spot_id"]: d for d in usable(ds) if d["shown"]}


def group_of(d):
    return {t: i for i, g in enumerate(d["move_groups"]) for t in g}


def h3(base, shuffled):
    """Same list position vs same piece (move) of the original choice, on the same spots."""
    pos, piece = [], []
    for sid, s in shuffled.items():
        b = base.get(sid)
        if b is not None:
            g = group_of(s)
            at = s["shown_legal_moves"][b["shown_legal_moves"].index(b["chosen_index"])]
            pos.append(int(g[s["chosen_index"]] == g[at]))
            piece.append(int(g[s["chosen_index"]] == g[b["chosen_index"]]))
    r = paired(pos, piece)
    return {"hypothesis": "H3", "test": "same position vs same piece", "expected": "+",
            "n": r["n"], "clusters": r["n"], "rate": r["rate_a"], "reference": r["rate_b"],
            "diff": r["diff"], "lo": r["diff_lo"], "hi": r["diff_hi"], "p": r["p_mcnemar"],
            "only_a": r["only_a"], "only_b": r["only_b"]}


def h4_spots(base, outcomes):
    sids = [s for s in outcomes if s in base]
    r = paired([outcomes[s]["m"]["agree"] for s in sids], [base[s]["m"]["agree"] for s in sids])
    changed = [group_of(outcomes[s])[outcomes[s]["chosen_index"]] != group_of(outcomes[s])[base[s]["chosen_index"]]
               for s in sids]
    return {"hypothesis": "H4", "test": "agreement with outcomes vs without", "expected": "+",
            "n": r["n"], "clusters": r["n"], "rate": r["rate_a"], "reference": r["rate_b"],
            "diff": r["diff"], "lo": r["diff_lo"], "hi": r["diff_hi"], "p": r["p_mcnemar"],
            "only_a": r["only_a"], "only_b": r["only_b"],
            "changed_choices": float(np.mean(changed)) if changed else None}


def h4_games(with_outcomes, without):
    """Agreement per seed with and without outcome lines, paired bootstrap over shared seeds."""
    per_seed = lambda ds: {s: (sum(d["m"]["agree"] for d in g), len(g)) for s, g in _by_seed(ds).items()}
    a, b = per_seed(with_outcomes), per_seed(without)
    seeds = sorted(set(a) & set(b))
    s = np.array([[*a[k], *b[k]] for k in seeds], dtype=float).reshape(-1, 4)
    est, lo, hi, p = bootstrap(s, lambda t: t[..., 0] / t[..., 1] - t[..., 2] / t[..., 3])
    return {"hypothesis": "H4", "test": "agreement with outcomes vs without", "expected": "+",
            "n": int(s[:, 1].sum() + s[:, 3].sum()), "clusters": len(seeds),
            "rate": s[:, 0].sum() / s[:, 1].sum() if len(s) else None,
            "reference": s[:, 2].sum() / s[:, 3].sum() if len(s) else None,
            "diff": est, "lo": lo, "hi": hi, "p": p}


def _by_seed(ds):
    out = defaultdict(list)
    for d in ds:
        out[d["seed"]].append(d)
    return out


def equivalent_picks(shuffled):
    """Exploratory: among equivalent moves, the one listed first or the lowest-numbered one?"""
    pos, low = [], []
    for s in shuffled.values():
        group = next(g for g in s["move_groups"] if s["chosen_index"] in g)
        listed = min(group, key=s["shown_legal_moves"].index)
        if listed != min(group):
            pos.append(int(s["chosen_index"] == listed))
            low.append(int(s["chosen_index"] == min(group)))
    r = paired(pos, low)
    return {"hypothesis": "exploratory", "test": "listed first vs lowest number among equivalent moves",
            "expected": "", "n": r["n"], "clusters": r["n"], "rate": r["rate_a"], "reference": r["rate_b"],
            "diff": r["diff"], "lo": r["diff_lo"], "hi": r["diff_hi"], "p": r["p_mcnemar"],
            "only_a": r["only_a"], "only_b": r["only_b"]}


def result(r):
    """The hypothesis's verdict for a row, from its direction and Holm-adjusted p."""
    if not r["n"] or r["p_holm"] != r["p_holm"]:
        return "no data"
    sig, up = r["p_holm"] < ALPHA, r["diff"] > 0
    if r["expected"] == "not +":
        return "refuted" if sig and up else "not refuted"
    if r["expected"] in ("+", "-"):
        return ("supported" if up == (r["expected"] == "+") else "refuted") if sig else "not significant"
    return "significant" if sig else "not significant"


def plot(rows, out):
    rows = [r for r in rows if r["family"] == "preregistered" and r["n"]]
    if not rows:
        return
    tests = list(dict.fromkeys((r["test"], r["data"]) for r in rows))
    models = sorted({r["model"] for r in rows})
    fig, axes = figure(ncols=len(tests), figsize=(3.2 * len(tests), 0.3 * len(models) + 2),
                       sharey=True, squeeze=False)
    for ax, (test, data) in zip(axes[0], tests):
        for r in rows:
            if (r["test"], r["data"]) == (test, data):
                i = models.index(r["model"])
                ax.errorbar(r["diff"], i, xerr=[[r["diff"] - r["lo"]], [r["hi"] - r["diff"]]], fmt="o", ms=3,
                            capsize=2, color="#C44E52" if r["p_holm"] < ALPHA else "#4C72B0")
        ax.axvline(0, color="0.5", linewidth=0.8)
        ax.set_title(f"{test}\n({data})", fontsize=7)
        ax.set_yticks(range(len(models)), models, fontsize=7)
        ax.tick_params(axis="x", labelsize=7)
    fig.supxlabel("difference with 95% interval (red: Holm p < 0.05)", fontsize=8)
    save_figure(fig, out, "hypotheses")


def run(runs, out):
    rows = []
    main = {r.model: r for r in runs if r.kind == "games" and r.experiment == "main"}
    game_ds = {m: game_decisions(r, runs) for m, r in sorted(main.items())}
    for model, ds in game_ds.items():
        tests = (h1(ds) if non_thinking(main[model], ds) else []) + h2(ds)
        rows += [dict(r, model=model, data="main games") for r in tests]
    for run in runs:
        if run.kind == "games" and run.experiment == "outcomes" and run.model in game_ds:
            rows.append(dict(h4_games(game_decisions(run, runs), game_ds[run.model]),
                             model=run.model, data="outcomes vs main games"))
    spots = defaultdict(dict)
    for run in runs:
        if run.kind == "spots" and run.experiment in ("spots", "spots-shuffle", "spots-outcomes"):
            spots[run.model][run.experiment] = by_spot(spot_decisions(run))
    for model, exps in sorted(spots.items()):
        base = exps.get("spots", {})
        if "spots-shuffle" in exps:
            rows.append(dict(h3(base, exps["spots-shuffle"]), model=model, data="spots vs spots-shuffle"))
            rows.append(dict(equivalent_picks(exps["spots-shuffle"]), model=model, data="spots-shuffle"))
        if "spots-outcomes" in exps:
            rows.append(dict(h4_spots(base, exps["spots-outcomes"]), model=model, data="spots vs spots-outcomes"))
    for r in rows:
        r["family"] = "exploratory" if r["hypothesis"] == "exploratory" else "preregistered"
    for family in ("preregistered", "exploratory"):
        sel = [r for r in rows if r["family"] == family]
        for r, p in zip(sel, holm([r["p"] if r["n"] else np.nan for r in sel])):
            r["p_holm"] = p
    for r in rows:
        r["result"] = result(r)
    order = ["model", "hypothesis", "test", "data", "expected", "result", "n", "clusters", "rate",
             "reference", "diff", "lo", "hi", "p", "p_holm", "family"]
    rows = [{**{k: r.get(k) for k in order}, **r} for r in rows]
    write_csv(out, "hypotheses", rows)
    plot(rows, out)
    pre = [r for r in rows if r["family"] == "preregistered"]
    counts = defaultdict(lambda: defaultdict(int))
    for r in pre:
        counts[f"{r['hypothesis']} {r['test']}"][r["result"]] += 1
    return [f"hypotheses ({len(pre)} preregistered tests, Holm across all): "
            + "; ".join(f"{k}: " + ", ".join(f"{v} {res}" for res, v in sorted(c.items()))
                        for k, c in counts.items())]
