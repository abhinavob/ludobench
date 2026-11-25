Spot Evaluation Docs (Current Active Scope)
===========================================

This folder documents the active reduced evaluation scope aligned with `spots_40/`.

Active categories documented here
---------------------------------
- blocked
- capture
- capture_vs_home
- capture_vs_home_finish
- capture_vs_openexisting
- capture_vs_safe
- extra_turn
- grudge_paired
- home_entry
- overshoot
- safe
- safe_vs_openexisting

Evaluation conventions
----------------------
1) Legal moves are computed internally (`spot_evaluation.py`).
2) Spot prompts do NOT include legal move lists (LLM is instructed to output token index 0..3).
3) Invalid-rate uses raw LLM move validity before fallback (`response_int in legal_moves`).
4) Category outputs are written as:
   - `spot_results_40/<persona>/<category>/...` for LLM runs
   - `spot_results_gt/<category>/...` for GT runs

Suggested run commands
----------------------
LLM reduced set:
- `python3 run_all_spots.py --spots-glob "spots_40/spots_*.json" --out-dir spot_results_40`

LLM all personas:
- `python3 run_all_spots.py --spots-glob "spots_40/spots_*.json" --all-personas --out-dir spot_results_40`

GT reduced set:
- `python3 run_all_spots_gt.py --spots-glob "spots_40/spots_*.json" --out-dir spot_results_gt --depth 2`

Persona aggregation:
- `python3 analysis_scripts/compare_personas.py --root spot_results_40 --personas aggressive,greedy,safe,unforgiving,none`

LLM vs GT compare:
- `python3 analysis_scripts/compare_llm_vs_gt.py --llm-root spot_results_40/none --gt-root spot_results_gt --out-csv spot_results_gt/llm_vs_gt_comparison.csv --out-actions-json spot_results_gt/llm_vs_gt_action_transitions.json`

Note on broader `spots/`
------------------------
`spots/` may contain additional legacy/experimental categories not represented in `spots_40/`.
This docs folder intentionally tracks the active reduced set used for primary reporting.
