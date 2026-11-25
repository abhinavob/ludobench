# Changes from the original LudoBench release

This version builds on the original LudoBench code (<https://anonymous.4open.science/r/LudoBench-5CBF/>). The
original release's notes are in [`docs/original/`](original/), unchanged; its `rules.txt` gives the original rules
(the `--original-rules` option), which our corrected rules replace.

## Rules

All rule logic is in one module, `ludo_rules.py`, which the engine, the bots, both search players, the prompt, the
logs, the board renderer and the spot evaluation share. Under our rules:

- A piece reaches its home end 56 pips from its start square. In the released code it passed its start square once
  more first (58 pips), so a 1 from the square behind its start could not be used.
- Inside the home path a piece moves by any roll that does not overshoot the home end. In the released code it could
  only move by the exact roll to the home end.
- Two players sit in seats 0 and 2 (the released code used 0 and 1), and a roll-off picks the first player (the
  released code always started with seat 0; only players tied for the highest roll re-roll).
- The dice, the random bot, the shuffle and the fallback move each draw on their own random stream seeded from the
  game seed (the released code shared one generator).
- The search players break ties between equal values (rounded to 1e-9) by the lowest piece index.

`--original-rules` restores all of the above, and the released prompt and reply format. With it, the games logged with
the released code replay exactly (`tests/test_original_rules_replay.py`).

## Prompt and reply

The released code capped every reply at 60 tokens and stopped it at the first newline, which leaves no room for a
model to reason. Our setup asks for a JSON reply, `{"reason": "...", "piece": N}` (reason first, at most 500
characters). The output ceiling is 1,024 tokens for an Ollama model that does not think, and 16,384 for one that
thinks and for API models. The legal moves are shown in piece order unless `--hide-legal-moves` is given. The system
prompt is "You are an expert Ludo player."

## Spots

`spots_40/` was built for the released rules, and under our rules some of its boards no longer test what they were
meant to (a move that was a capture is not, or a piece sits where our route never stops). `repair_spots.py` repairs
such spots automatically; each fix was then reviewed by hand, and the result is `spots_40_final/` (see
[Spot sets](../README.md#spot-sets)).

## Records

Every decision is logged with the board, the legal moves, the model's raw reply, tokens, latency, retries and cost, and
what the heuristic bot and the search player would have done. Each run's `run.json` holds its command line, commit,
model versions and every setting. A field's meaning is in `run.json` under `definitions`.
