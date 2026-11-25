# LudoBench: corrected rules, reviewed spots and head-to-head games

A benchmark of how large language models play Ludo. It asks a model for its move in two settings: full games against
scripted bots or other models, and *spots*, single board positions built so that one kind of move (a capture, a safe
square, bringing a piece out, a finish, ...) can be told apart from the others. Every decision is logged, and the same
runners work with local models (Ollama) and with models on OpenRouter, NVIDIA and Amazon Bedrock.

It builds on LudoBench (Jain & Kumar, 2026, <https://arxiv.org/abs/2604.05681>; original repository
<https://anonymous.4open.science/r/LudoBench-5CBF/>).

**What it measures**

- In games: the win rate against the heuristic bot or against other models, the rate of invalid replies, and how
  often the model takes a capture, a safe square, a bring-out, a home entry or a finish when one is available.
- In spots: how often the model's move agrees with the search player's (moves with the same outcome count as one),
  per category, with the legal moves hidden or shown in the prompt, and the same choice rates as in games.

**Quick start**

```bash
pip install -r requirements.txt
ollama pull qwen3.5:4b
python run_games.py --players llm:qwen3.5:4b heuristic --games 10 --experiment trial
python run_spots.py --player llm:qwen3.5:4b --experiment trial
```

This version corrects the original release's rules (`--original-rules` restores them), gives the model room to
reason, uses a reviewed spot set and logs every decision; see [docs/changes_from_original.md](docs/changes_from_original.md).
The original release's notes are in `docs/original/`, unchanged; its `rules.txt` gives the original rules (the
`--original-rules` option), which our corrected rules replace.

## Installation

Python 3.12 or later.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For local models install [Ollama](https://ollama.com) and pull a model, e.g. `ollama pull qwen3.5:4b`.

## Models and API keys

A player is named by a spec:

| Backend | Spec | Key (environment variable) |
|---|---|---|
| Ollama on this machine | `llm:qwen3.5:4b` | none |
| Ollama on another machine | `llm://HOST[:PORT]/qwen3.5:4b` (start it there with `OLLAMA_HOST=0.0.0.0`) | none |
| OpenRouter | `llm:openrouter/z-ai/glm-5.3-flash` | `OPENROUTER_API_KEY` |
| NVIDIA API catalog | `llm:nvidia/nvidia/nemotron-3.5-super-vl-120b-a12b` | `NVIDIA_API_KEY` |
| Amazon Bedrock | `llm:bedrock/openai.gpt-oss-20b-1:0`, or an `au.`, `apac.` or `global.` profile | `AWS_BEARER_TOKEN_BEDROCK` |
| Bots | `heuristic`, `random`, `first`, `gt[:DEPTH]` (the search player), `human` | none |

Keys are read from the environment only. They are never written to a file, printed or logged. Set them in the shell
that runs the command:

```bash
export OPENROUTER_API_KEY="<your key>"
```

```powershell
$env:OPENROUTER_API_KEY = [Environment]::GetEnvironmentVariable('OPENROUTER_API_KEY', 'User')
```

Bedrock requests go to the Sydney region (ap-southeast-2); to use another, change it in `llm_backends.py`.

Every model gets the same prompt and reply settings. Reasoning is switched off where the model allows it, else kept at
its lowest effort; `--effort low|medium|high` sends an effort level to the models that take one, and `--think` turns
thinking on for an Ollama model. `run.json` records the reasoning fields sent to each player (`reasoning`), whether
they leave it reasoning (`thinks`), and how many of its replies carried reasoning (`reasoned`). NVIDIA requests are
paced to 40 per minute; `--requests-per-minute N` paces any API. To see what a model would be sent, without a key:
`python analysis/check_api_models.py llm:openrouter/z-ai/glm-5.3-flash`. `analysis/bedrock_models.py` lists the
Bedrock models the region offers.

## Running games, spots and round robins

**Games** (`run_games.py`): players are listed without seat numbers.

```bash
python run_games.py --players llm:qwen3.5:4b heuristic --games 100 --experiment main
python run_games.py --players llm:openrouter/z-ai/glm-5.3-flash heuristic --games 100 --experiment main
python run_games.py --players human llm:qwen3.5:4b               # play against a model
python run_games.py --players llm:qwen3.5:4b heuristic --watch   # watch a game
```

Switches, each off by default: `--hide-legal-moves`, `--shuffle` (show the legal moves in a random order), `--outcomes`
(say what each legal move does), `--history N` (show the last N moves), `--think`, `--effort LEVEL`,
`--original-rules`. Without `--experiment` and `--games` a run plays one game into `results/sandbox/`.
`run_queue.py queue.txt` runs a list of `run_games.py` argument lines in order.

**Spots** (`run_spots.py`): each spot is asked with the legal moves hidden and shown, unless `--legal hidden|shown`.

```bash
python run_spots.py --player llm:qwen3.5:4b --experiment spots
python run_spots.py --player llm:qwen3.5:4b --legal shown --outcomes --experiment spots-outcomes
```

Options: `--spots FOLDER` (default `spots_40_final`), `--per-category N` (N spots of each category, e.g. for a slow
model), `--spot-ids FILE` (only the spots listed, one id per line), `--changed-from FOLDER` (only the spots that
differ from FOLDER), `--shuffle`, `--outcomes`, `--persona NAME` (one of the original release's persona
instructions), `--ceiling N` (a lower output-token ceiling, e.g. where an API reserves credit for the whole ceiling).

**Round robins** (`round_robin.py`): every group of `--players` from `--pool` plays `--games` games over the same seeds;
in a group of k > 2 players, game g uses the (g mod k)-th cyclic seating.

```bash
python round_robin.py --players 4 --games 96 --effort low --experiment h2h-4p --pool \
    llm:bedrock/openai.gpt-oss-120b-1:0 llm:bedrock/qwen.qwen3-235b-a22b-2507-v1:0 \
    llm:bedrock/mistral.mistral-large-3-675b-instruct llm:bedrock/google.gemma-3-12b-it
```

**Results** go to `results/<experiment>/<players>/` (`run.json`, `decisions.jsonl` and, for games, `games.jsonl`);
`results/cells.csv` has one row per game run. A stopped run resumes when its command is run again: finished games are
kept and an unfinished one is replayed from its seed. A finished run is left as it is, and a folder holding a run with
other settings, or an unfinished run made by other code, is refused, so a run with other settings needs its own
`--experiment` name.

## The search player ("gt")

The reference player for the spots is the original release's search agent, `GameTheoryMultiplayerAgent`
(`game_theory_multiplayer_agent.py`). It searches ahead over the dice: with two players an expectiminimax search, with
three or four an expecti-max-n search in which each player maximises its own value, scoring the positions at the
search depth with the release's hand-weighted evaluator. The code calls it "gt" (game theory), the release's name for
it: the player spec is `gt` or `gt:DEPTH`, and runs record it as `gt(d=2)`, the depth every spot run uses
(`search_move` in each record).
`analysis/search_validation.py` checks how much its choices depend on the depth and on the evaluator's weights.

## Spot sets

| Set | What it is | Use |
|---|---|---|
| `spots_40_final/` | Ours and the default: the original 520 spots (12 categories) for our rules. 96 differ from `spots_40/`, each repaired so the spot tests what it was built to test, and checked by hand. | All runs. |
| `spots_40/` | The original release's 520 spots, built for the released rules. Never changed. | With `--original-rules`, or to compare with the original release. |

`spots_40_final/changes.json` lists every spot that differs from `spots_40/`, with the fix, the full check and what a
run derives from the position; `spots_40_final/review.json` holds the review's decisions.
`python repair_spots.py --review spots_40_final/review.json --out spots_40_final` rebuilds the set from `spots_40/`,
`python repair_spots.py --check spots_40_final` runs the full check on it, and `review_spots.py` writes the review page
(`review/spots.html`).

## Kinds of experiment

`--experiment NAME` names the results folder; each kind below is one command or switch.

| Kind | How to run it |
|---|---|
| Games against a bot | `run_games.py --players MODEL heuristic --games 100` (or `random`, `first`, `gt`) |
| Bot baselines | `run_games.py --players gt heuristic --games 1000` (likewise `random`, `first`, `heuristic`) |
| Games with the legal moves hidden | add `--hide-legal-moves` |
| Spots, legal moves hidden and shown | `run_spots.py --player MODEL` (one condition: `--legal hidden` or `--legal shown`) |
| Move outcomes in the prompt | add `--outcomes` (games; spots with `--legal shown`) |
| Legal moves in a shuffled order | add `--shuffle` (games; spots with `--legal shown`) |
| Thinking or a reasoning effort | add `--think` (Ollama) or `--effort low\|medium\|high` |
| Recent moves in the prompt | add `--history N` (games) |
| Persona instructions | add `--persona NAME` (spots) |
| Repeatability | ask the same spots again with the same settings under another `--experiment` (`--spot-ids FILE` for a fixed list) and compare the answers spot by spot |
| Two-player head-to-head | `round_robin.py --players 2 --games 50 --pool MODEL MODEL ...` (every pair) |
| Four-player head-to-head | `round_robin.py --players 4 --games 96 --pool MODEL MODEL MODEL MODEL ...` (every group of four, over its six seatings) |

## Reproducing results

Every `run.json` records the exact command line and the commit it ran from (`git.commit`, and `git.dirty` if tracked
files had uncommitted changes). To redo a run, check out that commit and run the command again, with the key set if it
uses an API.

- **Games:** game i of a run uses seed `seed0 + i` (`--seed0`, default 0). The dice, the random bot, the shuffle and
  the fallback move each draw on their own stream seeded from the game seed, so a rerun rolls the same dice.
- **Round robins:** every group plays seeds 0 to `games - 1`, and the seed fixes the seating.
- **Spots:** a spot has no seed; a shuffled list draws on the spot's own stream, so it is the same in every run.

Models are asked at temperature 0, but a model's answers can still differ between runs (a repeatability run measures
by how much), and a model served by an API may change behind the same name. `run.json` records each Ollama model's
digest.

To check logged runs, and to build the tables and figures:

```bash
python analysis/verify_runs.py results        # replay every logged roll with the code of its commit
python analysis/report.py                     # tables and figures, to results/analysis/
python analysis/report.py --root main=results --root copy=path/to/other/results
python analysis/search_validation.py          # the search player's depth and weights, no models
```

## Tests

```bash
python -m pytest                  # everything
python -m pytest -m "not slow"    # skip the randomised and replay tests
```

The replay tests of the released code need the logs in `llm_game_results/`, which are not in the repository; without
them they skip.

## Repository layout

- Our code: `ludo_rules.py`, `run_games.py`, `run_spots.py`, `round_robin.py`, `run_queue.py`, `llm_backends.py`,
  `decision_log.py`, `board_render.py`, `repair_spots.py`, `review_spots.py`, `analysis/` and `tests/`.
- The original release, changed only where the rules needed it: `ludo_env.py`, `agents.py`, `game_theory_agent.py`,
  `game_theory_multiplayer_agent.py`, `llm_agent.py` and `spot_evaluation.py`.
- The original release, unchanged: `real_llm.py`, `run_experiments.py`, `run_spot_evaluation.py`, `run_all_spots.py`,
  `run_all_spots_baseline.py`, `run_all_spots_gt.py`, `run_gt_experiments.py`, `evaluation.py`, `sample_spots_40.py`,
  `archive_results.py`, `dummy_llm.py`, `analysis_scripts/`, `spots_40/`, `ludo.svg` and `docs/original/`.

## Known issues

- The original release's standalone scripts are kept as released and are not used by our runs. Some keep their own
  copy of rule logic: `classify_move` in `run_all_spots.py`, `run_all_spots_baseline.py`, `run_all_spots_gt.py` and
  `analysis_scripts/`. The analysis scripts read the original spot results (`spot_results_40/`, `spot_results_gt/`),
  which are in the original repository, not here.

## How to cite

Please cite the original LudoBench paper:

```bibtex
@misc{jain2026ludobenchevaluatingllmbehavioural,
      title={LUDOBENCH: Evaluating LLM Behavioural Decision-Making Through Spot-Based Board Game Scenarios in Ludo},
      author={Ojas Jain and Dhruv Kumar},
      year={2026},
      eprint={2604.05681},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2604.05681},
}
```

The citation for this version will be added here.

## Links

- Blog post: to be added
- Paper: to be added
