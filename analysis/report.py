"""
Build the analysis tables (CSV) and figures (PNG, PDF) from the results,
read-only.

    python analysis/report.py                       # all parts, to results/analysis/
    python analysis/report.py --out DIR --parts main spots

Runs are read from this repository's results/ (label "main"); --root
LABEL=FOLDER, repeatable, reads these folders instead, e.g. results copied from
another machine. A run's key is LABEL:experiment/run. runs.csv lists every run
found; summary.txt the headline of each part.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis import hypotheses, main_games, reasons, side_experiments, spot_tables  # noqa: E402
from analysis.common import DEFAULT_ROOTS, EXCLUDED_RUNS, REPO, find_runs, write_csv  # noqa: E402

PARTS = {"main": main_games, "spots": spot_tables, "hypotheses": hypotheses, "side": side_experiments,
         "reasons": reasons}


def inventory(runs):
    return [{"run": r.key, "kind": r.kind, "model": r.model, "experiment": r.experiment,
             "commit": (r.meta.get("git") or {}).get("commit"), "repairs": len(r.meta.get("repairs", [])),
             "excluded": EXCLUDED_RUNS.get(r.key, "")} for r in runs]


def build(roots, out, parts=tuple(PARTS)):
    runs = find_runs(roots)
    write_csv(out, "runs", inventory(runs))
    used = [r for r in runs if r.key not in EXCLUDED_RUNS]
    summary = [line for p in parts for line in PARTS[p].run(used, out)]
    with open(os.path.join(out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("--out", default=os.path.join(REPO, "results", "analysis"))
    ap.add_argument("--root", action="append", metavar="LABEL=FOLDER",
                    help="a results folder to read, repeatable (default: main=results)")
    ap.add_argument("--parts", nargs="+", choices=list(PARTS), default=list(PARTS))
    args = ap.parse_args()
    roots = dict(r.split("=", 1) for r in args.root) if args.root else DEFAULT_ROOTS
    print("\n".join(build(roots, args.out, args.parts)))
    print(f"\nTables and figures: {args.out}")


if __name__ == "__main__":
    main()
