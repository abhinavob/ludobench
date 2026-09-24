import argparse
import json
import os
import random
import re
from collections import defaultdict


ACTIVE_FILES = [
    "spots_blocked.json",
    "spots_capture.json",
    "spots_capture_vs_home.json",
    "spots_capture_vs_home_finish.json",
    "spots_capture_vs_openexisting.json",
    "spots_capture_vs_safe.json",
    "spots_extra_turn.json",
    "spots_grudge_paired.json",
    "spots_home_entry.json",
    "spots_overshoot.json",
    "spots_safe.json",
    "spots_safe_vs_openexisting.json",
]


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def dump_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def allocate_targets(values, total):
    values = sorted(values)
    if not values:
        return {}
    base = total // len(values)
    rem = total % len(values)
    out = {v: base for v in values}
    for i in range(rem):
        out[values[i]] += 1
    return out


def balanced_sample(entries, k, rng):
    """
    Balanced on:
    - player_count = len(entry["players"])
    - llm_player_id = entry["llm_player_id"]
    """
    if k >= len(entries):
        return list(entries)

    pool = list(entries)
    rng.shuffle(pool)

    pc_vals = sorted({len(e["players"]) for e in pool})
    llm_vals = sorted({int(e["llm_player_id"]) for e in pool})
    target_pc = allocate_targets(pc_vals, k)
    target_llm = allocate_targets(llm_vals, k)

    cnt_pc = defaultdict(int)
    cnt_llm = defaultdict(int)
    selected = []

    while len(selected) < k and pool:
        best_i = None
        best_score = None
        for i, e in enumerate(pool):
            pc = len(e["players"])
            lid = int(e["llm_player_id"])
            dpc = max(0, target_pc[pc] - cnt_pc[pc])
            dllm = max(0, target_llm[lid] - cnt_llm[lid])
            score = (dpc + dllm, dpc, dllm)
            if best_score is None or score > best_score:
                best_score = score
                best_i = i

        pick = pool.pop(best_i)
        selected.append(pick)
        cnt_pc[len(pick["players"])] += 1
        cnt_llm[int(pick["llm_player_id"])] += 1

    return selected


def pair_key(spot_id):
    m = re.match(r"(.+)_([ab])$", spot_id)
    if m:
        return m.group(1)
    return spot_id


def sample_pairs(entries, pair_count, rng):
    by_key = defaultdict(list)
    for e in entries:
        by_key[pair_key(e["id"])].append(e)

    pair_reps = []
    for k, vals in by_key.items():
        a = next((x for x in vals if str(x["id"]).endswith("_a")), vals[0])
        pair_reps.append((k, a))

    reps_only = [rep for _, rep in pair_reps]
    chosen_reps = balanced_sample(reps_only, pair_count, rng)
    chosen_keys = {pair_key(rep["id"]) for rep in chosen_reps}

    # Preserve original order in output.
    out = [e for e in entries if pair_key(e["id"]) in chosen_keys]
    return out


def summarize(entries):
    pc = defaultdict(int)
    lid = defaultdict(int)
    for e in entries:
        pc[len(e["players"])] += 1
        lid[int(e["llm_player_id"])] += 1
    return dict(sorted(pc.items())), dict(sorted(lid.items()))


def main():
    parser = argparse.ArgumentParser(description="Create spots_40 subset (balanced by player count and llm_player_id).")
    parser.add_argument("--in-dir", default="spots", help="Input spots directory")
    parser.add_argument("--out-dir", default="spots_40", help="Output subset directory")
    parser.add_argument("--seed", type=int, default=42, help="Fixed seed for reproducibility")
    parser.add_argument("--n", type=int, default=40, help="Entries per non-paired category")
    parser.add_argument("--pairs", type=int, default=40, help="Pairs for paired category (grudge)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    manifest = {
        "seed": args.seed,
        "non_paired_n": args.n,
        "paired_pairs": args.pairs,
        "files": {},
    }

    for fname in ACTIVE_FILES:
        in_path = os.path.join(args.in_dir, fname)
        if not os.path.exists(in_path):
            continue

        data = load_json(in_path)

        if fname == "spots_grudge_paired.json":
            sampled = sample_pairs(data, args.pairs, rng)
            expected = args.pairs * 2
            sampled = sampled[:expected]
        else:
            sampled_unordered = balanced_sample(data, args.n, rng)
            sampled_ids = {x["id"] for x in sampled_unordered}
            # Preserve original order for readability.
            sampled = [e for e in data if e["id"] in sampled_ids][:args.n]

        out_path = os.path.join(args.out_dir, fname)
        dump_json(out_path, sampled)

        pc, lid = summarize(sampled)
        manifest["files"][fname] = {
            "count": len(sampled),
            "player_count_distribution": pc,
            "llm_player_id_distribution": lid,
        }
        print(f"Wrote {out_path} ({len(sampled)} entries)")
        print(f"  player_count: {pc}")
        print(f"  llm_player_id: {lid}")

    dump_json(os.path.join(args.out_dir, "sampling_manifest.json"), manifest)
    print(f"Wrote {os.path.join(args.out_dir, 'sampling_manifest.json')}")


if __name__ == "__main__":
    main()
