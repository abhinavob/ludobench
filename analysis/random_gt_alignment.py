"""
Chance level for agreement with the search ("GT") player on the original spots.

For every spot in spot_results_gt/*/spot_results.json where the search player's
move is legal, a uniformly random legal player picks the same token index with
probability 1 / len(legal_moves). The mean of that value is the agreement a
random policy is expected to reach, overall and per category.

legal_moves in those files is the list of legal token indices that
spot_evaluation.compute_legal_moves returned for the spot's player and dice.

A second column counts tokens that stand on the same square as the search
player's token (for example two tokens in the base) as the same move, since
moving either one gives the same board.

Run from the repo root:  python analysis/random_gt_alignment.py
"""
import glob
import json
import os


def spot_chance(rec):
    """(index-level chance, same-square chance) for one spot, or None."""
    legal = rec["legal_moves"]
    gt = rec["gt_move"]
    if not legal or gt not in legal:
        return None
    own = rec["tokens"][str(rec["current_player"])]
    same = [t for t in legal if own[t] == own[gt]]
    return 1.0 / len(legal), len(same) / len(legal)


def main(root="spot_results_gt"):
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "*", "spot_results.json"))):
        category = os.path.basename(os.path.dirname(path))
        with open(path, encoding="utf-8") as f:
            recs = json.load(f)
        chances = [c for c in map(spot_chance, recs) if c is not None]
        rows.append((category, len(recs), chances))

    print(f"{'category':<26}{'spots':>6}{'used':>6}{'mean legal':>12}"
          f"{'random = GT':>13}{'same square':>13}")
    everything = []
    for category, total, chances in rows:
        everything += chances
        n = len(chances)
        legal = sum(1 / c[0] for c in chances) / n
        print(f"{category:<26}{total:>6}{n:>6}{legal:>12.2f}"
              f"{100 * sum(c[0] for c in chances) / n:>12.1f}%"
              f"{100 * sum(c[1] for c in chances) / n:>12.1f}%")
    n = len(everything)
    print(f"{'ALL':<26}{sum(r[1] for r in rows):>6}{n:>6}"
          f"{sum(1 / c[0] for c in everything) / n:>12.2f}"
          f"{100 * sum(c[0] for c in everything) / n:>12.1f}%"
          f"{100 * sum(c[1] for c in everything) / n:>12.1f}%")
    macro = sum(sum(c[0] for c in ch) / len(ch) for _, _, ch in rows) / len(rows)
    print(f"mean of the {len(rows)} category means: {100 * macro:.1f}%")


if __name__ == "__main__":
    main()
