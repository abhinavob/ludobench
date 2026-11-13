"""repair_spots.py: the repaired copy of spots_40/, its change log and the review page."""
import glob
import json
import os

import ludo_rules as R
import repair_spots as rs
import review_spots

SRC, OUT = "spots_40", "spots_40_repaired"


def spots(folder):
    return {s["id"]: s for f in sorted(glob.glob(os.path.join(folder, "spots_*.json")))
            for s in json.load(open(f, encoding="utf-8"))}


def text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_committed_copy_is_what_repair_spots_builds(tmp_path):
    rs.build(SRC, str(tmp_path))
    names = sorted(os.listdir(OUT))
    assert names == sorted(os.listdir(tmp_path)) and "changes.json" in names
    for name in names:
        assert text(os.path.join(OUT, name)) == text(tmp_path / name), name


def test_every_kept_spot_means_the_same_under_our_rules():
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


def test_counts_cover_every_spot_and_flags_are_logged():
    log = json.load(open(os.path.join(OUT, "changes.json")))
    assert sum(sum(n.values()) for n in log["counts"].values()) == len(spots(SRC)) == 520
    flagged = {e["id"]: e["flags"] for e in log["spots"] if e["flags"]}
    assert flagged["blocked_4p_011"] == ["single legal move"]
    assert sum("pieces of two players on one non-safe square" in f for f in flagged.values()) == 3


def test_repairs_fix_the_cause():
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
