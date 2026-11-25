"""repair_spots.py: the repaired copy of spots_40/, its change log and the review page."""
import glob
import hashlib
import json
import os

import pytest

import ludo_rules as R
import repair_spots as rs
import review_spots

SRC = "spots_40"
# spots_40_repaired/ as the runs on it read it; it is not tracked, repair_spots.py rebuilds it.
REPAIRED_SHA256 = "3584d39b877d295e88f004851c94e100b5ef6c5d8a49c126110a22a9bdb348aa"


def spots(folder):
    return {s["id"]: s for f in sorted(glob.glob(os.path.join(folder, "spots_*.json")))
            for s in json.load(open(f, encoding="utf-8"))}


def text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def OUT(tmp_path_factory):
    """spots_40_repaired/, as repair_spots.py writes it."""
    out = tmp_path_factory.mktemp("spots_40_repaired")
    rs.build(SRC, str(out))
    return str(out)


def test_repair_spots_rebuilds_the_set_the_runs_used(OUT):
    names = sorted(os.listdir(OUT))
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode() + b"\0" + text(os.path.join(OUT, name)).encode())
    assert "changes.json" in names and digest.hexdigest() == REPAIRED_SHA256


def test_every_kept_spot_means_the_same_under_our_rules(OUT):
    original, kept = spots(SRC), spots(OUT)
    log = {e["id"]: e for e in json.load(open(os.path.join(OUT, "changes.json")))["spots"]}
    for sid, spot in original.items():
        if sid not in kept:
            assert log[sid]["status"] == "excluded" and log[sid]["cause"]
            continue
        new = kept[sid]
        assert {k: v for k, v in new.items() if k != "tokens"} == \
               {k: v for k, v in spot.items() if k != "tokens"}
        assert rs.check(spot, rs.board(new))[0], sid
        assert not rs.off_route(rs.board(new)), sid
        assert (new["tokens"] != spot["tokens"]) == (log.get(sid, {}).get("status") == "repaired")


def test_counts_cover_every_spot_and_flags_are_logged(OUT):
    log = json.load(open(os.path.join(OUT, "changes.json")))
    assert sum(sum(n.values()) for n in log["counts"].values()) == len(spots(SRC)) == 520
    flagged = {e["id"]: e["flags"] for e in log["spots"] if e["flags"]}
    assert flagged["blocked_4p_011"] == ["single legal move"]
    assert sum("pieces of two players on one non-safe square" in f for f in flagged.values()) == 3


def test_repairs_fix_the_cause(OUT):
    log = {e["id"]: e for e in json.load(open(os.path.join(OUT, "changes.json")))["spots"]}
    # A piece of the mover on the square our route never stops on: one square back.
    assert log["home_entry_2p_067"]["fix"]["moved"] == [
        {"player": 0, "piece": 0, "from": 51, "to": 50}]
    # The off-route piece is a capture target: the capturer moves with it.
    assert log["cvs_4p_005"]["fix"]["moved"] == [
        {"player": 0, "piece": 0, "from": 36, "to": 35}, {"player": 3, "piece": 0, "from": 38, "to": 37}]
    # On our route, but the capture lands where our route turns into the home path.
    assert {m["player"] for m in log["cvo_2p_009"]["fix"]["moved"]} == {0, 1}
    assert "home_entry under ours" in log["cvo_2p_009"]["cause"][0]


def test_spot_without_a_fix_is_left_out(tmp_path):
    # Under the original rules a 1 cannot move a piece off square 51; under ours
    # any square before the home path can, so no repair keeps "cannot move".
    spot = {"id": "t_1", "scenario": "home_entry", "players": [0, 2], "llm_player_id": 0,
            "current_player": 0, "dice": 1, "tokens": {"0": [51, 30, -1, -1], "2": [5, -1, -1, -1]},
            "note": "test"}
    src = tmp_path / "src"
    src.mkdir()
    (src / "spots_test.json").write_text(json.dumps([spot], indent=2), encoding="utf-8")
    entries, log = rs.build(str(src), str(tmp_path / "out"))
    assert entries[0]["status"] == "excluded" and "off our route" in entries[0]["check"]["failed_on"][0]
    assert json.loads((tmp_path / "out" / "spots_test.json").read_text()) == []
    assert log["counts"] == {"home_entry": {"unchanged": 0, "repaired": 0, "excluded": 1}}


def test_cannot_move_names_the_reason():
    tokens = {0: [R.home_range(0)[1], -1, R.home_range(0)[0] + 3, 10], 2: [-1] * 4}
    assert [rs.cannot_move(tokens, 0, i, 5, False) for i in range(3)] == [
        "finished", "in base", "overshoot"]


def test_review_page_has_every_spot(tmp_path):
    data = review_spots.build(SRC, str(tmp_path / "spots.html"))
    page = text(tmp_path / "spots.html")
    assert len(data) == 520 and "__DATA__" not in page
    view = next(v for v in data if v["id"] == "cvo_2p_009")
    assert view["moved"] and view["fix"] and view["intent"]
    assert [cell[1] for cell in view["moves"]["rows"][1]] == [False, True, False]


FINAL, REVIEW = "spots_40_final", os.path.join("spots_40_final", "review.json")


def test_committed_final_copy_is_what_review_builds(tmp_path):
    rs.build_final(SRC, REVIEW, str(tmp_path))
    names = sorted(os.listdir(tmp_path))
    assert names == sorted(n for n in os.listdir(FINAL) if n != "review.json")
    for name in names:
        assert text(os.path.join(FINAL, name)) == text(tmp_path / name), name


def test_final_copy_keeps_every_spot_and_applies_the_review(OUT):
    original, repaired, final = spots(SRC), spots(OUT), spots(FINAL)
    decisions = json.load(open(REVIEW, encoding="utf-8"))["spots"]
    assert list(final) == list(original)
    for sid, spot in final.items():
        d = decisions.get(sid, {"decision": "keep"})
        assert spot == (rs.reviewed(original[sid], d["fix"]) if "fix" in d else repaired[sid])
        if spot["scenario"] == "grudge":
            twin = final[sid[:-1] + ("b" if sid.endswith("a") else "a")]
            assert (twin["dice"], twin["tokens"]) == (spot["dice"], spot["tokens"])
    log = json.load(open(os.path.join(FINAL, "changes.json"), encoding="utf-8"))
    assert {e["id"] for e in log["spots"]} == {i for i in final if final[i] != original[i]}
    assert sum(e["source"] == "review" for e in log["spots"]) == 44
    assert all(not e["check"] for e in log["spots"])
    e = next(e for e in log["spots"] if e["id"] == "safe_2p_055")
    assert e["dice"] == {"from": 3, "to": 2} and not e["same_as_repaired"]
    assert e["derived"]["legal_moves"] == [0, 1] and e["derived_repaired"]


def test_reviewed_fix_must_start_from_the_original():
    spot = spots(SRC)["safe_2p_055"]
    fix = {"moved": [{"player": 0, "piece": 0, "from": 17, "to": 19}], "dice": 2}
    with pytest.raises(ValueError, match="not on 17"):
        rs.reviewed(spot, fix)


def test_full_check_of_the_final_copy():
    assert rs.check_folder(SRC, FINAL) == {}


def test_full_check_finds_each_kind_of_change(OUT):
    original, repaired = spots(SRC), spots(OUT)
    audit = lambda sid, spot: " | ".join(rs.audit(original[sid], spot))
    # A piece that stays still becomes capturable after a move.
    assert "after piece 2 moves, piece 1 can be captured" in audit("blocked_4p_021", repaired["blocked_4p_021"])
    # A threat between two pieces flips direction.
    assert "P1 piece 0 could capture P0 piece 1; now the reverse" in audit("safe_3p_086",
                                                                           repaired["safe_3p_086"])
    # A piece placed on a safe square, and two players on one non-safe square.
    note = rs.reviewed(original["cvs_4p_031"], {"moved": [{"player": 2, "piece": 0, "from": 19, "to": 34}]})
    assert "P2 piece 0 starts on safe square 34" in audit("cvs_4p_031", note)
    assert "two players share non-safe square 27" in audit("svo_4p_011", original["svo_4p_011"])
    # A piece off our route, and a move whose outcome type changes.
    assert "off our route" in audit("home_entry_2p_067", original["home_entry_2p_067"])
    assert "home_entry now" in audit("cvo_2p_009", original["cvo_2p_009"])
    assert audit("capture_2p_002", original["capture_2p_002"]) == ""


def test_review_page_has_a_proposals_section(tmp_path):
    proposals = tmp_path / "proposals.json"
    fix = {"dice": 3, "moved": [{"player": 1, "piece": 0, "from": 4, "to": 5}]}
    proposals.write_text(json.dumps({"spots": {"capture_2p_002": {
        "kind": "decoy", "reason": "test", "fix": fix}}}), encoding="utf-8")
    review_spots.build(SRC, str(tmp_path / "spots.html"), str(proposals), FINAL)
    page = text(tmp_path / "spots.html")
    [v] = review_spots.proposal_views(SRC, FINAL, str(proposals), review_spots.intents())
    assert "proposals_review.json" in page and '"proposals": [{' in page
    assert len(v["boards"]) == 3 and len(v["moves"]["columns"]) == 3 and v["reason"] == "test"
    assert v["dice"] == [2, 2, 3] and v["changes"] == "dice 2 -> 3; P1 piece 0: square 4 -> square 5"
    assert v["audit"] == {"current": [], "proposed": []}       # 2 + 3 still captures on 5
    assert v["moved"] == [[1, 0]]
