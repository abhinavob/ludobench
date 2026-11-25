"""
The final analysis, read-only on the results roots, into one output folder.

    python -m analysis.final.report --out ../final_results \
        --root main=results --root work=path/to/results ... [--stages data metrics checks]

Stages: data (stage 1, data.py), metrics (stage 2, metrics modules) and
checks (stage 3). Each writes into OUT and notes its progress in
OUT/PROGRESS.md, so an interrupted analysis resumes where it stopped.
"""
import argparse
import os
import sys
import time

STAGES = ("data", "metrics", "checks")


def log_to(out):
    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(os.path.join(out, "progress_log.txt"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return log


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", action="append", required=True, metavar="LABEL=FOLDER")
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    log = log_to(out)
    roots = dict(r.split("=", 1) for r in args.root)
    from analysis.final import data
    data.build(roots, out, log)
    log("stage 1 (data) done")
    if "metrics" in args.stages:
        from analysis.final import metrics
        metrics.run(out, log, roots)
        log("stage 2 (metrics) done")
    if "checks" in args.stages:
        from analysis.final import checks
        checks.run(out, roots, log)
        log("stage 3 (checks) done")


if __name__ == "__main__":
    main()
