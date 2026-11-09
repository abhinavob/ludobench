"""
Run a list of runs in order: one run per line of a plain text file, written as
the arguments of run_games.py. Blank lines and lines starting with # are
skipped. run_games.py skips a run that already finished and resumes one that
stopped, so a stopped queue continues where it stopped when started again.
Works on Windows and Linux.

    python run_queue.py queue.txt

queue.txt:
    # main condition, then move outcomes
    --players llm:qwen3:8b heuristic --games 25 --experiment main
    --players llm:qwen3:8b heuristic --games 25 --outcomes --experiment outcomes
"""
import shlex
import subprocess
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python run_queue.py QUEUE_FILE")
    lines = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    runs = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    script = Path(__file__).with_name("run_games.py")
    failed = []
    for i, args in enumerate(runs, 1):
        print(f"\n=== run {i}/{len(runs)}: {args}", flush=True)
        if subprocess.run([sys.executable, str(script), *shlex.split(args)]).returncode:
            failed.append(args)
    print(f"\n{len(runs) - len(failed)} of {len(runs)} runs done")
    for args in failed:
        print(f"  failed: {args}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
