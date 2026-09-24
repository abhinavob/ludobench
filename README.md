# Ludo + LLM Behavioral Evaluation

This repository contains the code and evaluation assets for Ludo behavioral analysis with:
- LLM agents (`llm_agent.py` + `real_llm.py`)
- deterministic baselines (`agents.py`)
- game-theory search agents (`game_theory_agent.py`, `game_theory_multiplayer_agent.py`)

## Core Pipelines
- Full-game experiments: `run_experiments.py`
- Spot single-case evaluation: `run_spot_evaluation.py`
- Spot batch (LLM vs heuristic): `run_all_spots.py`
- Spot batch (GT vs heuristic): `run_all_spots_gt.py`
- Persona comparison: `analysis_scripts/compare_personas.py`
- LLM-vs-GT comparison on spots: `analysis_scripts/compare_llm_vs_gt.py`
- Archive snapshots: `archive_results.py`

## Public Benchmark Scope
- Final public benchmark subset: `spots_40/`
- Broader source set used during construction: `spots/`
- Temporary/deprecated sets: `spots_temporary/`

## Typical Commands
LLM run on final benchmark subset:
- `python3 run_all_spots.py --spots-glob "spots_40/spots_*.json" --out-dir spot_results_40`

LLM all personas on final benchmark subset:
- `python3 run_all_spots.py --spots-glob "spots_40/spots_*.json" --all-personas --out-dir spot_results_40`

GT run on final benchmark subset:
- `python3 run_all_spots_gt.py --spots-glob "spots_40/spots_*.json" --out-dir spot_results_gt --depth 2`

Persona aggregation:
- `python3 analysis_scripts/compare_personas.py --root spot_results_40 --personas aggressive,greedy,safe,unforgiving,none`

LLM vs GT category comparison:
- `python3 analysis_scripts/compare_llm_vs_gt.py --llm-root spot_results_40/none --gt-root spot_results_gt --out-csv spot_results_gt/llm_vs_gt_comparison.csv --out-actions-json spot_results_gt/llm_vs_gt_action_transitions.json`

## Primary Documentation
- `read.txt` - architecture overview
- `overall.txt` - file responsibility map
- `rules.txt` - implemented game rules
- `evaluation.txt` - metric definitions
- `spot_creation.txt` - benchmark subset and source-set construction notes
- `spot_docs/README.txt` - active spot-doc conventions for `spots_40/`
