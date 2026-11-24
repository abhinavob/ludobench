"""
Part 6: what the models' reasons talk about, and whether what they claim about
the chosen move is true.

Reasons: the "reason" of every valid reply on a decision with at least 2
distinct moves, in the games of "main" (replays in place of repaired games),
"hidden" and "outcomes" (the run's player) and in every spot experiment
("spots", "spots-shuffle", "spots-outcomes", "spots-think").

Themes: keyword rules (THEMES) on the whole reason, case-insensitive; a
capture the model fears ("avoid capture", "be captured") counts as danger or
safety, not as capture.

Claims (CLAIMS): keyword rules on the clauses about the chosen move, each
checked with the engine against what that move does. A reason is split into
clauses at sentence ends, semicolons and contrast words (while, whereas, but,
however, instead, unlike, compared, rather); a clause that names another of
the player's tokens ("token 2" when token 0 moved, "the other token") or
speaks hypothetically (would, could, might, if, later, future, potential,
opportunity, set up...) is not read for claims, nor is a match shortly after
a negation (not, no, without, avoid...). A safe-square, home-entry or finish
claim on the way there ("toward the safe square", "closest to entering the
home path") is not read either. A safe-square claim is about the square named
right after it ("safe square 47") or else just before it ("47, a safe
square"), else the destination: the destination's and the starting square's
are checked, another square's is not read.

Tables:
  reasons_themes.csv  per model and experiment: reasons, and the share that
                      mention each theme
  reasons_claims.csv  per model, experiment and claim: claims checked, false,
                      false rate; and "any": reasons with a checked claim, and
                      the share with at least one false claim
  reasons_false_vs_disagreement.csv
                      per model and experiment: among reasons with a checked
                      claim, the share with a false claim when the move agrees
                      with the search player and when it does not; the
                      difference (disagree - agree) with a cluster bootstrap
                      over games (spots for spot runs), Holm across rows
  reasons_sample.csv  50 random reasons with a theme (seeded), their themes and
                      claim verdicts and what the move did, to hand-check the
                      rules
Figures: reasons_themes, reasons_false_vs_disagreement.
"""
import random
import re
from collections import defaultdict

import numpy as np

import ludo_rules as rules
from analysis.common import decisions, figure, games, save_figure, write_csv
from analysis.positions import complete, env_of, measures
from analysis.stats import bootstrap, holm
from decision_log import is_exposed
from spot_evaluation import SpotEnv

GAME_EXPERIMENTS = ("main", "hidden", "outcomes")
SPOT_EXPERIMENTS = ("spots", "spots-shuffle", "spots-outcomes", "spots-think")
SAMPLE = 50

# A capture the model fears, not one it makes; read as danger or safety.
FEARED = re.compile(r"\b(?:avoid\w*|escap\w*|prevent\w*|risk\w*(?: of)?|from|without|no|safe from|"
                    r"immune to|vulnerable to|exposed to|be|being|been|get|getting)\s+"
                    r"(?:[a-z']+\s+)?captur\w*"
                    r"|\bcaptur\w*\s+(?:risk|threat|range)|\bcaptured by\b|\bout of capture\b", re.I)
SENT_HOME = re.compile(r"\b(?:send\w*|sent)\s+(?:\w+\s+){0,3}(?:back\s+)?to\s+(?:its\s+|their\s+|the\s+)?"
                       r"(?:base|start)\b|\bback to (?:its |their |the )?(?:base|start)\b", re.I)
THEMES = {
    "capture": r"\bcaptur\w*|\bsend\w*\s+(?:\w+\s+){0,3}back\b|\bknock\w*|\bkill\w*",
    "danger": r"\bsafe\w*|\bunsafe|\brisk\w*|\bthreat\w*|\bdanger\w*|\bvulnerab\w*|\bprotect\w*"
              r"|\bexpos\w*|\battack\w*|\bescap\w*",
    "bring_out": r"\bbase\b|\bbring\w*\s+(?:\w+\s+){0,3}out\b|\bleav\w*\s+(?:the\s+)?base|"
                 r"\b(?:enter\w*|onto|into)\s+(?:the\s+)?(?:main\s+)?board\b|\bnew (?:token|piece)\b",
    "home": r"\bhome[ _-]?(?:path|stretch|column|run|straight|entry|end|square)\b|"
            r"\benter\w*\s+(?:the\s+|its\s+|my\s+|our\s+)?home\b|\bfinish\w*|\bcomplet\w*",
    "progress": r"\badvanc\w*|\bprogress\w*|\bcloser\b|\bforward\b|\bfurthest\b|\bfarthest\b|"
                r"\bahead\b|\bfurther along\b",
    "blocking": r"\bblock\w*",
}
THEMES = {k: re.compile(v, re.I) for k, v in THEMES.items()}

# claim -> (pattern, check(move) -> whether the claim is true of the chosen move)
CLAIMS = {
    "capture": (r"\bcaptur(?:es|ing)\b|\bcapture\s+(?:the|an?|opponent|player|their|enemy|rival|it|"
                r"that|this|token|piece|p\d)\b|\bsend\w*\s+(?:\w+\s+){0,3}back\b|\bknock\w*",
                lambda m: m["capture"]),
    "safe_square": (r"\bsafe\s+(?:start(?:ing)?\s+)?(?:square|spot|zone|tile|cell|space)\b|\(safe\)|"
                    r"\bstar square\b",
                    None),                       # checked against the square it names (safe_claim)
    "out_of_reach": (r"\bavoid\w*\s+(?:any\s+|immediate\s+|being\s+)?(?:capture|risk|danger)|"
                     r"\bsafe from (?:capture|attack)|\bout of (?:capture\s+)?(?:reach|range)\b|"
                     r"\bprotect\w*\s+(?:\w+\s+){0,2}from (?:capture|attack|being captured)|"
                     r"\b(?:no|without any|without)\s+(?:immediate\s+)?(?:risk\w*|threat|danger)\b|"
                     r"\b(?:cannot|can't|can not|won't)\s+be\s+captured",
                     lambda m: not m["exposed_after"]),
    "threatened": (r"\b(?:escap\w*|flee\w*|rescu\w*)\b|\bout of (?:danger|harm)\b|"
                   r"\b(?:is|was|currently|now)\s+(?:under\s+(?:threat|attack)|threatened|in danger|"
                   r"at risk|vulnerable|exposed)\b|\baway from (?:danger|the threat|harm)\b",
                   lambda m: m["exposed_before"]),
    "bring_out": (r"\bbring\w*\s+(?:\w+\s+){0,3}out\b|\bout of (?:the\s+)?base\b|\bleav\w*\s+(?:the\s+)?base\b|"
                  r"\b(?:mov\w*|bring\w*|tak\w*|get\w*|roll\w*|enter\w*)\s+(?:\w+\s+){0,3}"
                  r"from (?:the\s+)?(?:base|-1)\b|"
                  r"\b(?:enter\w*|onto)\s+(?:the\s+)?(?:main\s+)?board\b",
                  lambda m: m["bring_out"]),
    "home_entry": (r"\b(?:enter\w*|into|onto|reach\w*)\s+(?:the\s+|its\s+|my\s+|our\s+)?"
                   r"home(?:[ _-]?(?:path|stretch|column|run|straight))?\b",
                   lambda m: m["in_home"]),
    "finish": (r"\bhome[ _-]?end\b|\bfinish\w*|\bcomplet\w*", lambda m: m["finish"]),
}
CLAIMS = {k: (re.compile(p, re.I), f) for k, (p, f) in CLAIMS.items()}
SPLIT = re.compile(r"(?<=[.!?;])\s+|\s*;\s*|,?\s+\b(?:while|whereas|but|however|instead|unlike|"
                   r"compared|rather)\b", re.I)
HYPOTHETICAL = re.compile(r"\b(?:would|could|might|if|otherwise|later|future|potential\w*|"
                          r"eventually|next turn|less|rather than|instead of|opportunit\w*|"
                          r"set(?:s|ting)? up|chance\w*|threaten\w*|possib\w*|more steps?|"
                          r"steps? away|other (?:token|piece)s?|another (?:token|piece)|the other)\b",
                          re.I)
# Before a safe-square, home-entry or finish claim: on the way there, not there yet.
TOWARD = re.compile(r"\b(?:toward|towards|closer to|closest to|close to|nearer to|near|approach\w*|"
                    r"path to|way to|progress\w* to|will|need\w* to|steps? to|in order to|start to|"
                    r"able to|until|before|aim\w* to|goal|(?:steps?|squares?) from|away from|then|"
                    r"again|overshoot\w*|exceed\w*|beyond|past|pass\w*)\W+(?:\w+\W+){0,2}$", re.I)
NEGATION = re.compile(r"\b(?:not|no|never|cannot|\w+n't|less|than|non|un|"
                      r"without|avoid\w*|wast\w*)\W*(?:\w+\W+){0,4}$", re.I)
TOWARD_CLAIMS = ("safe_square", "home_entry", "finish")
OPPONENT_TOKEN = re.compile(r"\b(?:opponent'?s?'?|enemy|rival|their|player\s*\d'?s?|p\d'?s?)\s+"
                            r"(?:token|piece|pawn)s?\s*(?:index\s*)?#?\d?", re.I)
OWN_TOKEN = re.compile(r"\b(?:token|piece|pawn)s?\s*(?:index\s*)?#?(\d)\b", re.I)
NUMBER = re.compile(r"-?\d+")


def themes(reason):
    text = SENT_HOME.sub(" sent back ", FEARED.sub(" ", reason))
    out = {k: bool(p.search(text)) for k, p in THEMES.items()}
    out["danger"] = out["danger"] or bool(FEARED.search(reason))
    return out


def clauses(reason, chosen):
    """The clauses of a reason about the chosen move (see the module docstring)."""
    out = []
    for c in SPLIT.split(reason):
        c = OPPONENT_TOKEN.sub(" opponent ", c or "")
        named = {int(t) for t in OWN_TOKEN.findall(c)}
        if c.strip() and not (named - {chosen}) and not HYPOTHETICAL.search(c):
            out.append(c)
    return out


def move_facts(d):
    """What the chosen move does, from the engine."""
    env, p, dice, t = env_of(d), d["player_id"], d["dice"], d["chosen_index"]
    e = rules.move_outcome(env.tokens, p, t, dice)
    after, _ = rules.apply_move(env.tokens, p, t, dice)
    return {"from": e["from"], "to": e["to"], "capture": bool(e["captures"]),
            "safe_to": e["safe"] or e["in_home"], "safe_from": e["from"] in rules.SAFE_SQUARES,
            "bring_out": e["bring_out"], "in_home": e["in_home"], "finish": e["home_finish"],
            "exposed_before": is_exposed(env, p, e["from"]),
            "exposed_after": is_exposed(SpotEnv(env.player_ids, after), p, e["to"])}


def safe_claim(clause, match, m):
    """
    Whether a safe-square claim is true: of the square named right after it
    ("safe square 47"), else just before it ("47, a safe square"), else of the
    destination.
    """
    after = re.match(r"\W{0,3}(?:at|on)?\W{0,3}(-?\d+)", clause[match.end():])
    near = NUMBER.findall(OWN_TOKEN.sub(" ", clause[max(0, match.start() - 30):match.start()]))
    square = int(after.group(1)) if after else int(near[-1]) if near else m["to"]
    if square == m["to"]:
        return m["safe_to"]
    if square == m["from"]:
        return m["safe_from"]
    return None                                   # about another square: not read


def claims(reason, chosen, m):
    """{claim: True if every such claim is true, False if one is false} for the claims made."""
    out = {}
    for c in clauses(reason, chosen):
        for name, (pattern, check) in CLAIMS.items():
            text = FEARED.sub(" feared ", c) if name == "capture" else c
            for match in pattern.finditer(text):
                before = text[max(0, match.start() - 40):match.start()]
                if NEGATION.search(before) or (name in TOWARD_CLAIMS and TOWARD.search(before)):
                    continue
                ok = safe_claim(text, match, m) if check is None else check(m)
                if ok is not None:
                    out[name] = out.get(name, True) and ok
    return out


def reason_rows(runs):
    """Every reason in scope: model, experiment, cluster, themes, claims and agreement."""
    rows = []
    for run in runs:
        if run.kind == "games" and run.experiment in GAME_EXPERIMENTS:
            ds = decisions(run, games(run, runs)[0])
        elif run.kind == "spots" and run.experiment in SPOT_EXPERIMENTS:
            ds = decisions(run)
        else:
            continue
        for d in ds:
            reason = d.get("parsed_reason")
            if not reason or d["was_fallback"] or d["chosen_index"] is None:
                continue
            complete(d)
            mm = measures(d)
            if not mm["choice"]:
                continue
            m = move_facts(d)
            rows.append({"model": run.model, "experiment": run.experiment, "run": run.key,
                         "cluster": d["spot_id"] or d["game"], "reason": reason,
                         "chosen": d["chosen_index"], "agree": mm["agree"], "move": m,
                         "themes": themes(reason), "claims": claims(reason, d["chosen_index"], m)})
    return rows


def theme_rows(rows):
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["experiment"])].append(r)
    return [{"model": mo, "experiment": ex, "reasons": len(rs),
             **{t: sum(r["themes"][t] for r in rs) / len(rs) for t in THEMES}}
            for (mo, ex), rs in sorted(by.items())]


def claim_rows(rows):
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["experiment"])].append(r)
    out = []
    for (mo, ex), rs in sorted(by.items()):
        for name in [*CLAIMS, "any"]:
            if name == "any":
                made = [r for r in rs if r["claims"]]
                false = sum(not all(r["claims"].values()) for r in made)
            else:
                made = [r for r in rs if name in r["claims"]]
                false = sum(not r["claims"][name] for r in made)
            out.append({"model": mo, "experiment": ex, "claim": name, "reasons": len(rs),
                        "claims": len(made), "false": false,
                        "false_rate": false / len(made) if made else None})
    return out


def disagreement_rows(rows):
    by = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))
    for r in rows:
        if r["claims"]:
            k = by[(r["model"], r["experiment"])][r["cluster"]]
            false = not all(r["claims"].values())
            k[0 if not r["agree"] else 2] += false
            k[1 if not r["agree"] else 3] += 1
    out = []
    for (mo, ex), clusters in sorted(by.items()):
        s = np.array(list(clusters.values()), dtype=float)
        n_dis, n_ag = s[:, 1].sum(), s[:, 3].sum()
        if not (n_dis and n_ag):
            continue
        est, lo, hi, p = bootstrap(s, lambda t: t[..., 0] / t[..., 1] - t[..., 2] / t[..., 3])
        out.append({"model": mo, "experiment": ex, "clusters": len(s),
                    "reasons_disagree": int(n_dis), "false_disagree": s[:, 0].sum() / n_dis,
                    "reasons_agree": int(n_ag), "false_agree": s[:, 2].sum() / n_ag,
                    "diff": est, "lo": lo, "hi": hi, "p": p})
    for r, p in zip(out, holm([r["p"] for r in out])):
        r["p_holm"] = p
    return out


def sample_rows(rows, n=SAMPLE, seed=0):
    tagged = [r for r in rows if any(r["themes"].values())]
    pick = random.Random(seed).sample(tagged, min(n, len(tagged)))
    return [{"run": r["run"], "chosen": r["chosen"], "reason": r["reason"],
             "themes": " ".join(t for t, v in r["themes"].items() if v),
             "claims": " ".join(f"{c}={'true' if v else 'FALSE'}" for c, v in r["claims"].items()),
             **{f"move_{k}": v for k, v in r["move"].items()}} for r in pick]


def plot(themes_tab, dis, out):
    if themes_tab:
        mat = np.array([[r[t] for t in THEMES] for r in themes_tab])
        fig, ax = figure(figsize=(1.2 + 0.7 * len(THEMES), 0.28 * len(themes_tab) + 1.2))
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        for i, row in enumerate(mat):
            for j, v in enumerate(row):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5,
                        color="white" if v < 0.5 else "black")
        ax.set_xticks(range(len(THEMES)), list(THEMES), rotation=30, ha="right", fontsize=7)
        ax.set_yticks(range(len(themes_tab)), [f"{r['model']} ({r['experiment']})" for r in themes_tab],
                      fontsize=5)
        fig.colorbar(im, label="share of reasons mentioning the theme")
        save_figure(fig, out, "reasons_themes")
    if dis:
        fig, ax = figure(figsize=(6, 0.28 * len(dis) + 1.2))
        y = np.arange(len(dis))
        ax.scatter([r["false_agree"] for r in dis], y, color="#4C72B0", s=12, label="agrees with search")
        ax.scatter([r["false_disagree"] for r in dis], y, color="#DD8452", s=12, label="disagrees")
        ax.set_yticks(y, [f"{r['model']} ({r['experiment']})" for r in dis], fontsize=5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of reasons with a false claim about the chosen move")
        ax.legend(fontsize=7)
        save_figure(fig, out, "reasons_false_vs_disagreement")


def run(runs, out):
    rows = reason_rows(runs)
    themes_tab, claims_tab, dis = theme_rows(rows), claim_rows(rows), disagreement_rows(rows)
    write_csv(out, "reasons_themes", themes_tab)
    write_csv(out, "reasons_claims", claims_tab)
    write_csv(out, "reasons_false_vs_disagreement", dis)
    write_csv(out, "reasons_sample", sample_rows(rows))
    plot(themes_tab, dis, out)
    anyc = [r for r in claims_tab if r["claim"] == "any" and r["claims"]]
    false = sum(r["false"] for r in anyc) / max(1, sum(r["claims"] for r in anyc))
    more = sum(r["p_holm"] < 0.05 and r["diff"] > 0 for r in dis)
    return [f"reasons: {len(rows)} reasons, {len({r['model'] for r in rows})} models; "
            f"{false:.2f} of reasons with a checked claim have a false one; false claims "
            f"significantly more common when disagreeing with the search player in {more} of "
            f"{len(dis)} model-experiments"]
