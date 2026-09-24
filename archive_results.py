import os
import shutil
import argparse
from datetime import datetime


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def copy_path(src, dst):
    if not os.path.exists(src):
        return
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        ensure_dir(os.path.dirname(dst))
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Archive current run outputs into timestamped results folder.")
    parser.add_argument(
        "--archive-root",
        default="results",
        help="Destination root folder where timestamped archive will be created.",
    )
    parser.add_argument(
        "--source-dir",
        default="spot_results",
        help="Source results root to archive (e.g., spot_results or spot_results_40).",
    )
    parser.add_argument(
        "--spots-dir",
        default=None,
        help="Spot definitions folder to archive (default: auto-detect spots_40 when source is *_40, else spots).",
    )
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archive_root = args.archive_root
    archive_dir = os.path.join(archive_root, ts)
    ensure_dir(archive_dir)

    source_dir = args.source_dir
    spots_dir = args.spots_dir
    if spots_dir is None:
        if source_dir.endswith("_40") and os.path.isdir("spots_40"):
            spots_dir = "spots_40"
        else:
            spots_dir = "spots"

    # Persona snapshots (primary archived results)
    persona_names = ["aggressive", "greedy", "safe", "unforgiving", "none"]
    for persona in persona_names:
        src = os.path.join(source_dir, persona)
        dst = os.path.join(archive_dir, "personas", persona)
        copy_path(src, dst)

    # Persona comparison artifacts (if generated)
    copy_path(os.path.join(source_dir, "persona_comparison.csv"), os.path.join(archive_dir, "persona_comparison.csv"))
    copy_path(os.path.join(source_dir, "persona_comparison.png"), os.path.join(archive_dir, "persona_comparison.png"))
    copy_path(os.path.join(source_dir, "persona_none_correlation.png"), os.path.join(archive_dir, "persona_none_correlation.png"))
    copy_path(os.path.join(source_dir, "overall_behavior_profile.txt"), os.path.join(archive_dir, "overall_behavior_profile.txt"))

    # Active spot definitions used for current experiments
    copy_path(spots_dir, os.path.join(archive_dir, os.path.basename(spots_dir)))

    # Logs and single-spot output
    copy_path("llm_log.txt", os.path.join(archive_dir, "llm_log.txt"))
    copy_path("game_log.txt", os.path.join(archive_dir, "game_log.txt"))
    copy_path("spot_llm.txt", os.path.join(archive_dir, "spot_llm.txt"))

    # Optional notes/logs (if present)
    copy_path("spot_run.txt", os.path.join(archive_dir, "spot_run.txt"))

    print(f"Archived results from '{source_dir}' to: {archive_dir}")


if __name__ == "__main__":
    main()
