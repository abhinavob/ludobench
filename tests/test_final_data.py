"""The final analysis, stage 1 (analysis/final): run roles, model names, complete spot runs, search values."""
import json

import pytest

from analysis.common import Run
from analysis.final import catalog as C
from analysis.final import tidy
from spot_evaluation import SpotEnv


def run(tmp_path, experiment, name, settings=None, records=()):
    d = tmp_path / experiment / name
    d.mkdir(parents=True)
    (d / "decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    meta = {"settings": settings or {"spots_folder": "spots_40_repaired"},
            "players": {"0": {"model_spec": "llm:qwen3.5:4b", "provider": "ollama_native", "seat_position": 0}}}
    return Run("work", experiment, name, str(d), meta)


def rec(spot, shown, chosen, extra=None):
    return {"spot_id": spot, "config": {"legal_moves_shown": shown}, "chosen_index": chosen, **(extra or {})}


def test_roles_name_the_tidy_experiment_or_the_reason(tmp_path):
    role = lambda exp, name: C.role(Run("x", exp, name, "", {"settings": {}, "players": {}}))
    assert role("history", "qwen3.5-4b_heuristic") == ("use", "main-history")
    assert role("h2h-4p/all", "seating-3") == ("use", "h2h-4p")
    assert role("spots-outcomes-final", "qwen3.5-4b") == ("part", "spots-outcomes")
    assert role("spots-argon390b", "openrouter-google-gemini-4-argon") == ("part", "spots")
    assert role("main-replay-s1", "m_heuristic") == ("replay", "main")
    assert role("pilot-nvidia-final", "x")[0] == role("archive-rules-v1/main", "x")[0] == "excluded"
    assert role("main", "gpt-oss-20b_heuristic") == ("excluded", C.EXCLUDED["main/gpt-oss-20b_heuristic"])
    assert role("seat-check-884", "h") == ("seat-check", "seat-check")


def test_model_ids_and_names_across_providers():
    assert C.model_id("llm://127.0.0.1:11541/gemma4:12b") == ("ollama", "gemma4:12b")
    assert C.model_id("llm:bedrock/openai.gpt-oss-20b-1:0") == ("bedrock", "openai.gpt-oss-20b-1:0")
    info = C.model_info("llm:qwen.qwen3-235b-a22b-2507-v1:0".replace("llm:", "llm:bedrock/"))
    assert (info["display"], info["size_b"], info["active_b"]) == ("Qwen3 235B A22B", 235, 22)
    assert C.model_info("gt")["display"] == "Search player (depth 2)"


def test_a_complete_spot_run_takes_the_changed_spots_from_its_rerun(tmp_path):
    base = run(tmp_path, "spots", "m", records=[rec(s, c, 0, {"extra": 1}) for s in ("a", "b", "c") for c in (False, True)])
    rerun = run(tmp_path, "spots-final", "m", {"spots_folder": "spots_40_final"},
                [rec("b", c, 2, {"new_field": 9}) for c in (False, True)])
    d = C.Dataset("spots", "ollama:qwen3.5:4b", "", "spots", [base, rerun])
    recs, check = C.complete_spots(d, changed={"b", "z"})
    assert [(r["spot_id"], r["config"]["legal_moves_shown"], r["chosen_index"]) for r in recs] == [
        ("a", False, 0), ("a", True, 0), ("b", False, 2), ("b", True, 2), ("c", False, 0), ("c", True, 0)]
    assert all(list(r) == list(recs[0]) for r in recs) and "new_field" not in recs[2]
    assert check["replaced"] == 2
    wrong = run(tmp_path, "spots-final2", "m", {"spots_folder": "spots_40_final"}, [rec("a", True, 1)])
    with pytest.raises(ValueError, match="expected the 2 of the changed spots"):
        C.complete_spots(C.Dataset("spots", "m", "", "spots", [base, wrong]), changed={"b"})
    alone = C.Dataset("spots-repeat", "m", "", "spots", [base])
    assert C.complete_spots(alone, changed={"z"})[1]["replaced"] == 0


def test_search_values_are_the_ones_the_search_player_chooses_by():
    tokens = {0: [10, 20, -1, -1], 2: [12, -1, -1, -1]}
    move, values = tidy.search_values(SpotEnv([0, 2], tokens), 0, 2, [0, 1])
    assert set(values) == {0, 1} and values[move] == max(values.values())
