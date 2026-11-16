"""
Build the analysis tables (CSV) and figures (PNG, PDF) from the results,
read-only.

    python analysis/report.py                       # all parts, to results/analysis/
    python analysis/report.py --out DIR --parts main spots

Roots (--root LABEL=FOLDER, repeatable) default to common.DEFAULT_ROOTS: the
main folder's results, ../LudoBench-work/results, ../LudoBench-bedrock/results
and ../sharanga_results. runs.csv lists every run found; summary.txt the
headline of each part.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis import main_games  # noqa: E402
from analysis.common import DEFAULT_ROOTS, REPO, find_runs, write_csv  # noqa: E402

PARTS = {"main": main_games}


def inventory(runs):
    return [{"run": r.key, "kind": r.kind, "model": r.model, "experiment": r.experiment,
             "commit": (r.meta.get("git") or {}).get("commit"), "repairs": len(r.meta.get("repairs", []))}
            for r in runs]


def build(roots, out, parts=tuple(PARTS)):
    runs = find_runs(roots)
    write_csv(out, "runs", inventory(runs))
    summary = [line for p in parts for line in PARTS[p].run(runs, out)]
    with open(os.path.join(out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("--out", default=os.path.join(REPO, "results", "analysis"))
    ap.add_argument("--root", action="append", metavar="LABEL=FOLDER",
                    help="a results root (default: the four in common.DEFAULT_ROOTS)")
    ap.add_argument("--parts", nargs="+", choices=list(PARTS), default=list(PARTS))
    args = ap.parse_args()
    roots = dict(r.split("=", 1) for r in args.root) if args.root else DEFAULT_ROOTS
    print("\n".join(build(roots, args.out, args.parts)))
    print(f"\nTables and figures: {args.out}")


if __name__ == "__main__":
    main()
