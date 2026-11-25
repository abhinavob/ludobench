"""
Findings about the original spot set (spots_40/) and what spots_40_final/ changed.

  duplicates  spots with an identical position (players, player to move, dice,
              pieces and history text) under different ids
  decoys      spots with a piece on a square where our route never stops
              (the square just behind its owner's start: there it has already
              turned into its home path), so it cannot be where the spot puts it
  changes     every spot of spots_40_final/ that differs from spots_40/, from
              its changes.json: by category and source (the automatic repair,
              or a fix chosen on review), the repair's cause grouped by kind,
              and the review's decisions (review.json)

Tables: dataset_duplicates.csv, dataset_decoys.csv, dataset_changes.csv,
dataset_change_causes.csv, dataset_summary.csv.
"""
import glob
import json
import os
import re
from collections import Counter

import pandas as pd

from analysis.common import REPO
from repair_spots import board, off_route

CAUSES = [("off route", r"never stop|off our route|just behind its start"),
          ("move outcome type changed", r"home_entry|home path|finish|capture|bring|outcome|plain move"),
          ("exposure changed", r"captur|expos|threat"),
          ("shared or safe square", r"safe square|share")]


def spots(folder):
    return [s for p in sorted(glob.glob(os.path.join(REPO, folder, "spots_*.json")))
            for s in json.load(open(p, encoding="utf-8"))]


def position(s):
    return json.dumps([sorted(s["players"]), s["current_player"], s["dice"],
                       {str(k): v for k, v in sorted(s["tokens"].items())}, s.get("history_text") or ""])


def duplicates(spot_list):
    groups = {}
    for s in spot_list:
        groups.setdefault(position(s), []).append(s["id"])
    return [{"ids": "; ".join(v), "count": len(v)} for v in groups.values() if len(v) > 1]


def decoys(spot_list):
    rows = []
    for s in spot_list:
        for q, i in off_route(board(s)):
            rows.append({"spot_id": s["id"], "category": s["scenario"], "player": q, "piece": i,
                         "square": s["tokens"][str(q)][i], "mover": q == s["current_player"]})
    return rows


def cause_kind(text):
    for kind, pattern in CAUSES:
        if re.search(pattern, text, re.I):
            return kind
    return "other"


def run(ctx):
    orig, final = spots("spots_40"), spots("spots_40_final")
    dup_o, dup_f = duplicates(orig), duplicates(final)
    ctx.write("dataset_duplicates", [dict(r, set="spots_40") for r in dup_o] + [dict(r, set="spots_40_final")
                                                                                 for r in dup_f])
    dec_o, dec_f = decoys(orig), decoys(final)
    ctx.write("dataset_decoys", [dict(r, set="spots_40") for r in dec_o] + [dict(r, set="spots_40_final") for r in dec_f])
    with open(os.path.join(REPO, "spots_40_final", "changes.json"), encoding="utf-8") as f:
        ch = json.load(f)["spots"]
    with open(os.path.join(REPO, "spots_40_final", "review.json"), encoding="utf-8") as f:
        review = json.load(f)["spots"]
    repaired = {}
    rep_path = os.path.join(REPO, "spots_40_repaired", "changes.json")
    if os.path.exists(rep_path):
        with open(rep_path, encoding="utf-8") as f:
            repaired = {e["id"]: e for e in json.load(f)["spots"]}
    rows = []
    for e in ch:
        cause = " ".join(repaired.get(e["id"], {}).get("cause") or [])
        rows.append({"spot_id": e["id"], "category": e["category"], "source": e["source"],
                     "decision": e.get("decision"), "proposal": e.get("proposal"),
                     "dice_changed": e.get("dice") is not None, "pieces_moved": len(e.get("moved") or []),
                     "repair_cause": cause, "cause_kind": cause_kind(cause) if cause else "",
                     "review_note": e.get("note") or e.get("reason") or ""})
    t = pd.DataFrame(rows)
    ctx.write("dataset_changes", t)
    causes = Counter(t[t["cause_kind"] != ""]["cause_kind"])
    ctx.write("dataset_change_causes", [{"cause_kind": k, "spots": v} for k, v in causes.most_common()])
    by_cat = t.groupby(["category", "source"]).size().unstack(fill_value=0).reset_index()
    ctx.write("dataset_changes_by_category", by_cat)
    decisions = Counter(v.get("decision") for v in review.values())
    summary = [{"finding": "spots in spots_40", "value": len(orig)},
               {"finding": "exact duplicate groups in spots_40", "value": len(dup_o)},
               {"finding": "spots in duplicate groups in spots_40", "value": sum(r["count"] for r in dup_o)},
               {"finding": "exact duplicate groups in spots_40_final", "value": len(dup_f)},
               {"finding": "spots_40 spots with a piece where our route never stops",
                "value": len({r["spot_id"] for r in dec_o})},
               {"finding": "... with an opponent's piece there (decoys: a target that would have turned into "
                           "its home path)", "value": len({r["spot_id"] for r in dec_o if not r["mover"]})},
               {"finding": "... with the mover's own piece there", "value": len({r["spot_id"] for r in dec_o if r["mover"]})},
               {"finding": "spots_40_final spots with such a piece", "value": len({r["spot_id"] for r in dec_f})},
               {"finding": "spots of spots_40_final differing from spots_40", "value": len(t)},
               {"finding": "... as repaired automatically", "value": int((t["source"] == "repair").sum())},
               {"finding": "... with a fix chosen on review", "value": int((t["source"] == "review").sum())},
               {"finding": "grudge pairs changed", "value": t[t["category"] == "grudge"]["spot_id"].str[:-2].nunique()}]
    summary += [{"finding": f"review decisions: {k}", "value": v} for k, v in decisions.items()]
    ctx.write("dataset_summary", summary)
    for r in summary:
        ctx.note("dataset", r["finding"], r["value"], unit="count")
