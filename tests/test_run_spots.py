"""run_spots.py: spot prompts as in games (shuffled, with outcomes), history and persona,
records, rates, resume and the per-category subset."""
import json
import sys
from collections import Counter

import pytest

import run_games
import run_spots
from decision_log import choice_rates
from game_theory_multiplayer_agent import GameTheoryMultiplayerAgent
from ludo_env import LudoEnv, random_streams
from spot_evaluation import SpotEnv
from test_decision_records import offline

IDS = ["svo_4p_004", "grudge_pair002_a", "grudge_pair002_b", "home_entry_2p_067", "capture_2p_002"]
SPOTS = {s["id"]: s for s in run_spots.load_spots(run_spots.DEFAULT_SPOTS)}


class Model:
    """make_llm stand-in that answers with piece `pick(prompt)`; it keeps every prompt."""

    def __init__(self, pick=lambda prompt: 0, stop_after=None):
        self.pick, self.stop_after, self.prompts = pick, stop_after, []

    def __call__(self, spec, **_):
        def call(prompt):
            if self.stop_after is not None and len(self.prompts) >= self.stop_after:
                raise KeyboardInterrupt
            self.prompts.append(prompt)
            call.last_content = json.dumps({"reason": "test", "piece": self.pick(prompt)})
            return call.last_content
        call.base_url, call.last_content = None, None
        return call


@pytest.fixture
def folder(tmp_path):
    """A spot folder with the spots in IDS."""
    path = tmp_path / "spots_test"
    path.mkdir()
    (path / "spots_test.json").write_text(json.dumps([SPOTS[i] for i in IDS], indent=2), encoding="utf-8")
    return str(path)


def run(tmp_path, monkeypatch, folder, model=None, name="run", **kw):
    model = model or Model()
    offline(monkeypatch, model)
    records = run_spots.run_spots("llm:m:1b", str(tmp_path / name), folder, **kw)
    return records, model


def by(records, sid, shown=True):
    return next(r for r in records if r["spot_id"] == sid and r["config"]["legal_moves_shown"] is shown)


@pytest.mark.parametrize("order,outcomes", [("ascending", False), ("shuffled", True)])
@pytest.mark.parametrize("shown", [True, False])
def test_spot_prompt_is_the_game_prompt(tmp_path, monkeypatch, folder, shown, order, outcomes):
    condition = "shown" if shown else "hidden"
    _, model = run(tmp_path, monkeypatch, folder, legal=(condition,), legal_order=order,
                   move_outcomes=outcomes)
    spot = SPOTS["svo_4p_004"]
    env = LudoEnv(seed=0, num_players=4)
    env.tokens = run_spots.board(spot)
    env.rng = random_streams(f"svo_4p_004:{condition}")
    game = Model()
    offline(monkeypatch, game)
    player = run_games.player_info(["llm:m:1b"], [0])[0]
    kwargs, _ = run_games.llm_setup(player, False, False, None, run_games.RELEASED)
    agent, _ = run_games.build_agent("llm:m:1b", kwargs, None, shown, False, 0, outcomes, order)
    agent.select_move(env, spot["current_player"], spot["dice"], [0, 1, 2, 3])
    assert model.prompts[0] == game.prompts[0]
    assert ("Legal moves:" in game.prompts[0]) is shown
    assert ("What each legal move does" in game.prompts[0]) is (shown and outcomes)


def test_shuffle_and_outcomes_are_recorded_and_shown_lists_vary(tmp_path, monkeypatch, folder):
    records, model = run(tmp_path, monkeypatch, folder, legal_order="shuffled", move_outcomes=True)
    settings = json.loads(open(tmp_path / "run" / "run.json").read())["settings"]
    assert settings["legal_order"] == "shuffled" and settings["move_outcomes"] is True
    for rec, prompt in zip(records, model.prompts):
        shown = rec["config"]["legal_moves_shown"]
        assert rec["config"]["legal_order"] == "shuffled"
        assert rec["config"]["move_outcomes"] is shown
        assert ("What each legal move does" in prompt) is shown
        if shown:
            assert sorted(rec["shown_legal_moves"]) == rec["legal_moves"]
            assert f"{run_games.LEGAL_LIST_LEAD}{rec['shown_legal_moves']}\n" in prompt
    assert any(r["shown_legal_moves"] not in (None, r["legal_moves"]) for r in records)


def test_grudge_spots_carry_their_history_and_others_none(tmp_path, monkeypatch, folder):
    records, model = run(tmp_path, monkeypatch, folder, legal=("shown",))
    prompts = dict(zip([r["spot_id"] for r in records], model.prompts))
    for sid in IDS:
        has = "HISTORY / CONTEXT" in prompts[sid]
        assert has == sid.startswith("grudge")
        if has:
            assert SPOTS[sid]["history_text"] in prompts[sid]
    a, b = by(records, "grudge_pair002_a"), by(records, "grudge_pair002_b")
    assert (a["history"], b["history"], a["pair_id"]) == ("neutral", "grudge", "grudge_pair002")
    assert a["tokens_before"] == b["tokens_before"]
    assert by(records, "svo_4p_004")["history"] is None


def test_persona_is_off_by_default_and_from_run_all_spots(tmp_path, monkeypatch, folder):
    _, plain = run(tmp_path, monkeypatch, folder, legal=("shown",))
    records, styled = run(tmp_path, monkeypatch, folder, name="aggressive", legal=("shown",),
                          persona="aggressive")
    with open(run_spots.os.path.join(run_spots.HERE, "run_all_spots.py"), encoding="utf-8") as f:
        source = f.read()
    texts = run_spots.persona_texts()
    assert set(texts) == {"none", "aggressive", "greedy", "safe", "unforgiving"}
    assert all(t in source for t in texts.values())
    assert not any("PERSONA" in p for p in plain.prompts)
    assert all(texts["aggressive"] in p for p in styled.prompts)
    assert {r["persona"] for r in records} == {"aggressive"}


def test_records_have_the_game_fields_and_the_spot_fields(tmp_path, monkeypatch, folder):
    records, model = run(tmp_path, monkeypatch, folder)
    definitions = json.loads(open(tmp_path / "run" / "run.json").read())["definitions"]
    assert len(records) == 2 * len(IDS) and len(model.prompts) == 2 * len(IDS)
    keys = set(records[0])
    assert set(run_spots.SPOT_DEFINITIONS) <= keys and {"availability", "heuristic_move",
                                                         "raw_reply", "config", "tokens_after"} <= keys
    assert set(definitions) >= set(run_spots.SPOT_DEFINITIONS)
    spot = SPOTS["home_entry_2p_067"]
    rec = by(records, "home_entry_2p_067")
    env = SpotEnv(spot["players"], run_spots.board(spot))
    assert rec["search_move"] == GameTheoryMultiplayerAgent(2).select_move(env, 0, spot["dice"], [0, 1])
    assert rec["spot_changed"] and not by(records, "capture_2p_002")["spot_changed"]
    assert rec["spots_folder"] == "spots_test" and rec["move_groups"] == [[0], [1]]
    assert by(records, "svo_4p_004")["move_groups"] == [[0, 2, 3], [1]]
    assert not (tmp_path / "run" / "games.jsonl").exists()


def test_svo_4p_004_counts_base_pieces_as_bring_outs_and_opening(tmp_path, monkeypatch, folder):
    records, _ = run(tmp_path, monkeypatch, folder, legal=("shown",))
    rec = by(records, "svo_4p_004")
    a = rec["availability"]
    assert a["bring_out"]["tokens"] == [0, 2, 3]
    assert 1 in a["safe"]["tokens"] and 1 not in a["bring_out"]["tokens"]
    assert rec["chosen_index"] == 0                     # the model chose a base piece
    rates = choice_rates([rec])
    assert rates["open"] == {"available": 1, "taken": 1, "rate": 1.0}
    assert rates["safe"] == {"available": 1, "taken": 0, "rate": 0.0}


@pytest.mark.parametrize("kw", [{}, {"legal_order": "shuffled", "move_outcomes": True}])
def test_stopped_run_resumes_to_the_same_records(tmp_path, monkeypatch, folder, kw):
    whole, _ = run(tmp_path / "a", monkeypatch, folder, **kw)
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path / "b", monkeypatch, folder, model=Model(stop_after=3), **kw)
    rest, model = run(tmp_path / "b", monkeypatch, folder, **kw)
    assert len(model.prompts) == 2 * len(IDS) - 3
    assert rest == whole
    meta = json.loads(open(tmp_path / "b" / "run" / "run.json").read())
    assert len(meta["resumed_at"]) == 1 and "games" not in meta
    with pytest.raises(SystemExit):                     # other settings in the same folder
        run(tmp_path / "b", monkeypatch, folder, persona="safe")


def test_subset_takes_n_units_per_category_with_even_player_counts():
    every = run_spots.load_spots(run_spots.DEFAULT_SPOTS)
    ten, twenty = run_spots.subset(every, 10), run_spots.subset(every, 20)
    order = [s["id"] for s in every]
    assert [s["id"] for s in ten] == [i for i in order if i in {s["id"] for s in ten}]
    assert {s["id"] for s in ten} < {s["id"] for s in twenty}
    for category in {s["scenario"] for s in every}:
        picked = [s for s in ten if s["scenario"] == category]
        units = {run_spots.pair_id(s) or s["id"]: len(s["players"]) for s in picked}
        assert len(units) == 10 and sorted(Counter(units.values()).values()) == [3, 3, 4]
        if category == "grudge":
            assert len(picked) == 20 and all(n == 2 for n in Counter(map(run_spots.pair_id, picked)).values())
        for k in (2, 3, 4):                            # the first units of each player count
            pool = [s["id"] for s in every if s["scenario"] == category and len(s["players"]) == k]
            mine = [s["id"] for s in picked if len(s["players"]) == k]
            assert mine == pool[:len(mine)]
    assert run_spots.subset(every, 100) == every


def cli(tmp_path, monkeypatch, *args):
    offline(monkeypatch, Model())
    monkeypatch.setattr(run_spots, "RESULTS", str(tmp_path / "results"))
    monkeypatch.setattr(run_games, "check_ollama", lambda *a: "0.0-test")
    monkeypatch.setattr(sys, "argv", ["run_spots.py", "--player", "llm:m:1b", *args,
                                      "--experiment", "trial"])
    run_spots.main()


def test_command_line_writes_to_the_experiment(tmp_path, monkeypatch, folder):
    cli(tmp_path, monkeypatch, "--spots", folder, "--legal", "hidden", "--persona", "greedy")
    out = tmp_path / "results" / "trial" / "m-1b_persona-greedy"
    meta = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert meta["settings"]["legal_moves"] == ["hidden"] and meta["settings"]["persona"] == "greedy"
    assert meta["settings"]["legal_order"] == "ascending" and not meta["settings"]["move_outcomes"]
    assert meta["settings"]["per_category"] is None and meta["settings"]["spots"] == len(IDS)
    assert meta["settings"]["reason_max_chars"] == 500 and meta["players"]["0"]["ceiling"] == 1024
    assert len((out / "decisions.jsonl").read_text(encoding="utf-8").splitlines()) == len(IDS)


def test_command_line_shuffle_and_outcomes(tmp_path, monkeypatch, folder, capsys):
    cli(tmp_path, monkeypatch, "--spots", folder, "--shuffle", "--outcomes")
    meta = json.loads((tmp_path / "results" / "trial" / "m-1b" / "run.json").read_text(encoding="utf-8"))
    assert meta["settings"]["legal_order"] == "shuffled" and meta["settings"]["move_outcomes"]
    with pytest.raises(SystemExit):
        cli(tmp_path, monkeypatch, "--spots", folder, "--legal", "hidden", "--outcomes")
    assert "--outcomes needs the legal-move list" in capsys.readouterr().err


def test_command_line_per_category(tmp_path, monkeypatch):
    cli(tmp_path, monkeypatch, "--legal", "shown", "--per-category", "1")
    out = tmp_path / "results" / "trial" / "m-1b"
    meta = json.loads((out / "run.json").read_text(encoding="utf-8"))
    ids = [json.loads(line)["spot_id"] for line in (out / "decisions.jsonl").open(encoding="utf-8")]
    assert meta["settings"]["per_category"] == 1 and meta["settings"]["spots"] == len(ids) == 13
    assert sum(i.startswith("grudge") for i in ids) == 2
    with pytest.raises(SystemExit):
        cli(tmp_path, monkeypatch, "--per-category", "0")


def test_changed_from_asks_only_the_spots_that_differ(tmp_path, monkeypatch, folder):
    before = tmp_path / "before"
    before.mkdir()
    moved = [dict(SPOTS[i], dice=1) if i == "capture_2p_002" else SPOTS[i] for i in IDS]
    (before / "spots_test.json").write_text(json.dumps(moved, indent=2), encoding="utf-8")
    cli(tmp_path, monkeypatch, "--spots", folder, "--changed-from", str(before))
    out = tmp_path / "results" / "trial" / "m-1b"
    meta = json.loads((out / "run.json").read_text(encoding="utf-8"))
    ids = {json.loads(line)["spot_id"] for line in (out / "decisions.jsonl").open(encoding="utf-8")}
    assert ids == {"capture_2p_002"} and meta["settings"]["spots"] == 1
    assert meta["settings"]["changed_from"] == "before"
    with pytest.raises(SystemExit, match="no spot"):
        cli(tmp_path, monkeypatch, "--spots", folder, "--changed-from", folder)
    records, _ = run(tmp_path, monkeypatch, folder, name="all")
    assert "changed_from" not in json.loads((tmp_path / "all" / "run.json").read_text())["settings"]


def test_derived_is_what_a_run_records(tmp_path, monkeypatch, folder):
    records, model = run(tmp_path, monkeypatch, folder, legal=("shown",), legal_order="shuffled",
                         move_outcomes=True)
    for rec, prompt in zip(records, model.prompts):
        d = run_spots.derived(SPOTS[rec["spot_id"]])
        assert (d["legal_moves"], d["search_move"], d["move_groups"], d["shown_shuffled"]) == (
            rec["legal_moves"], rec["search_move"], rec["move_groups"], rec["shown_legal_moves"])
        assert d["shown"] == rec["legal_moves"] and all(line in prompt for line in d["outcome_lines"])


def test_spot_ids_asks_only_the_listed_spots(tmp_path, monkeypatch, folder):
    ids = tmp_path / "ids.txt"
    ids.write_text("# a comment\ncapture_2p_002\n\nsvo_4p_004\n", encoding="utf-8")
    cli(tmp_path, monkeypatch, "--spots", folder, "--spot-ids", str(ids))
    out = tmp_path / "results" / "trial" / "m-1b"
    meta = json.loads((out / "run.json").read_text(encoding="utf-8"))
    asked = [(json.loads(line)["spot_id"], json.loads(line)["config"]["legal_moves_shown"])
             for line in (out / "decisions.jsonl").open(encoding="utf-8")]
    assert sorted(asked) == [(i, s) for i in ("capture_2p_002", "svo_4p_004") for s in (False, True)]
    assert meta["settings"]["spot_ids"] == {"file": "ids.txt", "ids": ["capture_2p_002", "svo_4p_004"]}
    assert meta["settings"]["spots"] == 2
    ids.write_text("capture_2p_002\nnot_a_spot\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="not_a_spot not among the spots asked"):
        cli(tmp_path / "x", monkeypatch, "--spots", folder, "--spot-ids", str(ids))


class KeptArgs(Model):
    """Model that also keeps the make_llm arguments."""

    def __call__(self, spec, **kwargs):
        self.kwargs = kwargs
        return super().__call__(spec)


ARGON = {"structured_outputs": True, "temperature": True,          # reasoning on, efforts only
         "reasoning": {"mandatory": True, "supported_efforts": ["high", "medium", "low"]}}


def test_ceiling_lowers_only_the_output_cap_and_is_recorded(tmp_path, monkeypatch, folder):
    runs = {}
    for name, ceiling in (("default", None), ("capped", 1024)):
        model = KeptArgs()
        offline(monkeypatch, model)
        monkeypatch.setattr(run_games, "api_model_info", lambda *a: ARGON)
        run_spots.run_spots("llm:openrouter/g/m", str(tmp_path / name), folder, ("shown",), ceiling=ceiling)
        runs[name] = model.kwargs, json.loads((tmp_path / name / "run.json").read_text(encoding="utf-8"))
    (kw, meta), (capped, capped_meta) = runs["default"], runs["capped"]
    assert kw["max_tokens"] == meta["players"]["0"]["ceiling"] == run_games.API_CEILING
    assert "ceiling" not in meta["settings"]
    assert capped["max_tokens"] == capped_meta["players"]["0"]["ceiling"] == capped_meta["settings"]["ceiling"] == 1024
    assert dict(capped, max_tokens=None) == dict(kw, max_tokens=None)
    assert capped["extra_body"] == {"reasoning": {"effort": "low"}}
    with pytest.raises(ValueError, match="--ceiling must be from 1 to the model's ceiling"):
        run_spots.run_spots("llm:openrouter/g/m", str(tmp_path / "x"), folder, ("shown",),
                            ceiling=run_games.API_CEILING + 1)


class Paid(Model):
    """Model whose every reply costs $0.25 and carries 7 thinking tokens."""

    def __call__(self, spec, **_):
        call = super().__call__(spec)
        call.last_cost, call.last_thinking_tokens = 0.25, 7
        return call


def test_spot_run_json_has_its_cost_and_whether_the_model_reasoned(tmp_path, monkeypatch, folder):
    for name, info, off in (("always", ARGON, False),
                            ("switch", dict(ARGON, reasoning={"mandatory": False}), True)):
        offline(monkeypatch, Paid())
        monkeypatch.setattr(run_games, "api_model_info", lambda *a: info)
        records = run_spots.run_spots("llm:openrouter/g/m", str(tmp_path / name), folder, ("shown",))
        meta = json.loads((tmp_path / name / "run.json").read_text(encoding="utf-8"))
        assert meta["cost_usd"] == 0.25 * len(records) == sum(r["cost_usd"] for r in records)
        assert meta["reasoned"] == {"0": {"replies": len(records), "with_reasoning": len(records)}}
        assert meta["players"]["0"]["thinks"] is not off
        assert meta["settings"]["no_think"] is off and all(r["config"]["no_think"] is off for r in records)
