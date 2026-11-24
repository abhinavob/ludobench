"""
Write review/spots.html, a local page for checking repair_spots.py's work by hand.

One spot per screen: the original and repaired boards side by side, each piece
labelled player.piece and the moved pieces ringed; the category's intent and
the spot's note; what each of the mover's pieces does under the original rules
and ours, marking where ours differs; the cause, the fix and the check. Arrows step
through the spots, K keeps and R rejects; choices and notes are kept in the
browser and saved to a JSON file with Save.

With --proposals, the page also has a Proposals section (the last choice of
the status filter): each proposed fix to spots_40_final/ with three boards
(original, current, proposed), the moves under each, the reason, the exact
changes and the full check. Its choices are kept apart and saved to
proposals_review.json.

    python review_spots.py                 # after repair_spots.py
    python review_spots.py --proposals spots_40_final/proposals.json
"""
import argparse
import glob
import json
import os
import re

import board_render
import ludo_rules as rules
import repair_spots as rs

BASE_SLOTS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def intents(docs="spot_docs"):
    """Category file name -> the Intent paragraph of its spot_docs page."""
    out = {}
    for path in glob.glob(os.path.join(docs, "spots_*.json.txt")):
        m = re.search(r"Intent\n-+\n(.*?)\n\s*\n", open(path, encoding="utf-8").read(), re.S)
        out[os.path.basename(path)[:-len(".txt")]] = " ".join(m.group(1).split()) if m else ""
    return out


def layout(tokens):
    """Each piece's grid cell: [player, piece, row, col, position]."""
    out = []
    for q, t in sorted(tokens.items()):
        for i, pos in enumerate(t):
            if pos == rules.BASE:
                r0, c0 = board_render.BASE_ANCHOR[q]
                r, c = r0 + BASE_SLOTS[i][0], c0 + BASE_SLOTS[i][1]
            else:
                r, c = board_render.coords(q, pos)
            out.append([q, i, r, c, pos])
    return out


def spot_view(spot, entry, intent):
    tokens, p, d, players = rs.board(spot), spot["current_player"], spot["dice"], spot["players"]
    fixed = {int(q): t for q, t in (entry["repaired_tokens"] or spot["tokens"]).items()}
    ref = rs.pieces(tokens, players, p, d, True)
    columns = [("original rules", ref), ("ours, as given", rs.pieces(tokens, players, p, d, False))]
    if entry["status"] == "repaired":
        columns.append(("ours, repaired", rs.pieces(fixed, players, p, d, False)))
    moved = [[m["player"], m["piece"]] for m in (entry["fix"] or {}).get("moved", [])]
    return {
        "id": spot["id"], "file": entry["file"], "category": spot["scenario"], "intent": intent,
        "note": spot["note"], "history": spot.get("history_text"), "status": entry["status"],
        "flags": entry["flags"], "cause": entry["cause"] or [],
        "fix": (entry["fix"] or {}).get("text"), "check": entry["check"],
        "mover": p, "dice": d, "players": players, "moved": moved,
        "boards": [layout(tokens), layout(fixed)],
        "moves": {"columns": [c for c, _ in columns],
                  "rows": [[[rs.describe(p, col[i]), not rs.same(ref[i], col[i])] for _, col in columns]
                           for i in range(len(tokens[p]))]},
    }


def changes_text(current, proposed):
    """The exact piece and dice changes from current to proposed."""
    out = [f"dice {current['dice']} -> {proposed['dice']}"] if proposed["dice"] != current["dice"] else []
    out += [f"P{m['player']} piece {m['piece']}: {rs.where(m['player'], m['from'])} -> "
            f"{rs.where(m['player'], m['to'])}" for m in rs.moves_between(current, proposed)]
    return "; ".join(out)


def proposal_view(original, current, proposal, file, intent):
    """A proposed fix: the original, current and proposed boards and what each piece does in each."""
    proposed = rs.reviewed(current, proposal["fix"])
    p, players = current["current_player"], current["players"]
    versions = [(original, True), (current, False), (proposed, False)]
    cols = [rs.pieces(rs.board(s), players, p, s["dice"], o) for s, o in versions]
    return {
        "id": current["id"], "file": file, "category": current["scenario"], "intent": intent,
        "note": current["note"], "history": current.get("history_text"), "status": "proposal",
        "kind": proposal["kind"], "reason": proposal["reason"],
        "changes": changes_text(current, proposed), "flags": [],
        "audit": {"current": rs.audit(original, current), "proposed": rs.audit(original, proposed)},
        "mover": p, "dice": [s["dice"] for s, _ in versions], "players": players,
        "moved": [[m["player"], m["piece"]] for m in rs.moves_between(current, proposed)],
        "titles": ["Original (spots_40)", "Current (spots_40_final)", "Proposed"],
        "boards": [layout(rs.board(s)) for s, _ in versions],
        "moves": {"columns": ["original rules", "ours, current", "ours, proposed"],
                  "rows": [[[rs.describe(p, col[i]), not rs.same(cols[0][i], col[i])] for col in cols]
                           for i in range(len(current["tokens"][str(p)]))]},
    }


def proposal_views(src, final, path, docs):
    with open(path, encoding="utf-8") as f:
        proposals = json.load(f)["spots"]
    original = {s["id"]: s for spots, _ in rs.read_spots(src).values() for s in spots}
    return [proposal_view(original[s["id"]], s, proposals[s["id"]], name, docs.get(name, ""))
            for name, (spots, _) in rs.read_spots(final).items() for s in spots if s["id"] in proposals]


def build(src, out_path, proposals=None, final="spots_40_final"):
    data, docs = [], intents()
    for path in sorted(glob.glob(os.path.join(src, "spots_*.json"))):
        for spot in json.load(open(path, encoding="utf-8")):
            entry, _ = rs.review(spot)
            entry["file"] = os.path.basename(path)
            data.append(spot_view(spot, entry, docs.get(entry["file"], "")))
    geometry = {"track": board_render.TRACK, "safe": sorted(rules.SAFE_SQUARES),
                "home": {q: board_render.HOME_COORDS[q] for q in range(4)},
                "starts": rules.START_POSITIONS, "bases": board_render.BASE_ANCHOR}
    extra = proposal_views(src, final, proposals, docs) if proposals else []
    payload = json.dumps({"spots": data, "proposals": extra, "geometry": geometry}).replace("</", "<\\/")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(PAGE.replace("__DATA__", payload))
    return data


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spot Repair Review</title>
<style>
:root { --bg:#f7f7f5; --panel:#fff; --ink:#1d1d1f; --muted:#6b6b70; --line:#d9d9dc; --track:#fbfbfa;
  --safe:#e9e3c8; --diff:#ffe1d6; --diff-ink:#9a2b0b; --ring:#ff7a00; --keep:#1f7a3a; --reject:#b3261e;
  --p0:#d64545; --p1:#2e9b4f; --p2:#d1a400; --p3:#3a6fd8; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#17181a; --panel:#202124;
  --ink:#ececee; --muted:#a0a0a8; --line:#3a3b40; --track:#26272b; --safe:#4a4530; --diff:#5a2a1c;
  --diff-ink:#ffb59c; } }
:root[data-theme="dark"] { --bg:#17181a; --panel:#202124; --ink:#ececee; --muted:#a0a0a8; --line:#3a3b40;
  --track:#26272b; --safe:#4a4530; --diff:#5a2a1c; --diff-ink:#ffb59c; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.45 system-ui, sans-serif; }
header { position:sticky; top:0; z-index:2; display:flex; flex-wrap:wrap; gap:10px; align-items:center;
  padding:10px 16px; background:var(--panel); border-bottom:1px solid var(--line); }
header .grow { flex:1; }
select, button { font:inherit; color:var(--ink); background:var(--bg); border:1px solid var(--line);
  border-radius:6px; padding:4px 10px; }
button { cursor:pointer; }
main { max-width:1180px; margin:0 auto; padding:16px; }
.title { display:flex; flex-wrap:wrap; gap:10px; align-items:baseline; }
.title h1 { font-size:22px; margin:0; }
.badge { font-size:12px; padding:2px 8px; border-radius:99px; border:1px solid var(--line); }
.badge.repaired { border-color:var(--ring); } .badge.excluded { border-color:var(--reject); }
.badge.keep { background:var(--keep); color:#fff; border-color:var(--keep); }
.badge.reject { background:var(--reject); color:#fff; border-color:var(--reject); }
.about { margin:8px 0 14px; color:var(--muted); }
.about b { color:var(--ink); font-weight:600; }
.boards { display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr)); gap:16px; }
.board { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:10px; }
.board h2 { font-size:14px; margin:0 0 6px; color:var(--muted); font-weight:600; }
svg { width:100%; height:auto; display:block; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 14px; margin-top:16px; }
table { border-collapse:collapse; width:100%; font-size:14px; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
td.diff { background:var(--diff); color:var(--diff-ink); font-weight:600; }
ul { margin:4px 0; padding-left:20px; }
textarea { width:100%; min-height:60px; font:inherit; color:var(--ink); background:var(--bg);
  border:1px solid var(--line); border-radius:6px; padding:6px; }
.keys { color:var(--muted); font-size:13px; }
.empty { padding:40px; text-align:center; color:var(--muted); }
@media (max-width:600px) { main { padding:12px 16px; } }
</style></head>
<body>
<header>
  <button id="prev" title="Previous (left arrow)">&larr;</button>
  <button id="next" title="Next (right arrow)">&rarr;</button>
  <select id="cat"></select>
  <select id="status">
    <option value="attention">repaired, excluded or flagged</option>
    <option value="all">all spots</option>
    <option value="repaired">repaired</option><option value="excluded">excluded</option>
    <option value="flagged">flagged</option><option value="unchanged">unchanged</option>
    <option value="unreviewed">unreviewed</option>
  </select>
  <span id="progress" class="grow"></span>
  <button id="keep" title="Keep (K)">Keep (K)</button>
  <button id="reject" title="Reject (R)">Reject (R)</button>
  <button id="save">Save choices</button>
  <button id="load">Load</button>
  <input id="file" type="file" accept="application/json" hidden>
</header>
<main id="main"></main>
<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById("data").textContent);
const G = DATA.geometry, SPOTS = DATA.spots, PROPOSALS = DATA.proposals || [];
const COLORS = ["var(--p0)", "var(--p1)", "var(--p2)", "var(--p3)"];
// The 85 reviewed spots and the proposals keep separate choices and save files.
const SECTIONS = {
  spots: {key: "ludobench-spot-review", file: "spot_review.json",
          about: "Choices made on review/spots.html for spots_40_repaired/", choices: {}},
  proposals: {key: "ludobench-proposal-review", file: "proposals_review.json",
              about: "Choices made on review/spots.html (Proposals) for spots_40_final/proposals.json", choices: {}},
};
for (const sec of Object.values(SECTIONS)) {
  try { sec.choices = JSON.parse(localStorage.getItem(sec.key) || "{}"); } catch (e) {}
}
let shown = [], at = 0;
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const section = () => SECTIONS[$("status").value === "proposals" ? "proposals" : "spots"];

function persist() { const sec = section(); try { localStorage.setItem(sec.key, JSON.stringify(sec.choices)); } catch (e) {} }

function matches(s) {
  const cat = $("cat").value, st = $("status").value;
  if (cat !== "all" && s.category !== cat) return false;
  if (st === "proposals") return true;
  const review = SECTIONS.spots.choices;
  if (st === "attention") return s.status !== "unchanged" || s.flags.length > 0;
  if (st === "flagged") return s.flags.length > 0;
  if (st === "unreviewed") return !(review[s.id] && review[s.id].decision);
  return st === "all" || s.status === st;
}

function refilter() {
  const current = shown[at];
  shown = ($("status").value === "proposals" ? PROPOSALS : SPOTS).filter(matches);
  at = Math.max(0, shown.indexOf(current));
  draw();
}

function svgBoard(pieces, s, after) {
  const C = 30, P = 4, size = 15 * C + 2 * P;
  const cell = (r, c) => [P + c * C, P + r * C];
  let out = `<svg viewBox="0 0 ${size} ${size}" role="img" aria-label="board">`;
  G.track.forEach(([r, c], sq) => {
    const [x, y] = cell(r, c), owner = G.starts.indexOf(sq);
    const fill = owner >= 0 ? COLORS[owner] : G.safe.includes(sq) ? "var(--safe)" : "var(--track)";
    out += `<rect x="${x}" y="${y}" width="${C}" height="${C}" fill="${fill}" fill-opacity="${owner >= 0 ? 0.35 : 1}" stroke="var(--line)"/>`;
    out += `<text x="${x + 2}" y="${y + 9}" font-size="7.5" fill="var(--muted)">${sq}${G.safe.includes(sq) ? "*" : ""}</text>`;
  });
  for (const q of [0, 1, 2, 3]) G.home[q].forEach(([r, c], k) => {
    const [x, y] = cell(r, c);
    out += `<rect x="${x}" y="${y}" width="${C}" height="${C}" fill="${COLORS[q]}" fill-opacity="${k === 5 ? 0.45 : 0.18}" stroke="var(--line)"/>`;
  });
  for (const q of s.players) {
    const [r, c] = G.bases[q], [x, y] = cell(r - 1, c - 1), mover = q === s.mover;
    out += `<rect x="${x}" y="${y}" width="${3 * C}" height="${3 * C}" rx="8" fill="${COLORS[q]}" fill-opacity="0.12" stroke="${mover ? "var(--ink)" : COLORS[q]}" stroke-width="${mover ? 2.5 : 1}"/>`;
    out += `<text x="${x + 1.5 * C}" y="${y + 1.5 * C + 4}" text-anchor="middle" font-size="12" font-weight="700" fill="${COLORS[q]}">P${q}</text>`;
    if (mover) out += `<text x="${x + 1.5 * C}" y="${y + 3 * C + 12}" text-anchor="middle" font-size="10" fill="var(--ink)">to move</text>`;
  }
  const byCell = {};
  pieces.forEach(p => (byCell[p[2] + "," + p[3]] = byCell[p[2] + "," + p[3]] || []).push(p));
  for (const group of Object.values(byCell)) group.forEach((p, k) => {
    const [q, i, r, c] = p, n = group.length;
    let [x, y] = cell(r, c);
    const off = n === 1 ? [[0.5, 0.55]] : [[0.3, 0.35], [0.7, 0.35], [0.3, 0.78], [0.7, 0.78]];
    x += off[k % off.length][0] * C; y += off[k % off.length][1] * C;
    const rad = n === 1 ? 11 : 8, moved = s.moved.some(m => m[0] === q && m[1] === i);
    if (moved) out += `<circle cx="${x}" cy="${y}" r="${rad + 4}" fill="none" stroke="var(--ring)" stroke-width="3" ${after ? "" : 'stroke-dasharray="4 3"'}/>`;
    out += `<circle cx="${x}" cy="${y}" r="${rad}" fill="${COLORS[q]}" stroke="${q === s.mover ? "var(--ink)" : "none"}" stroke-width="1.5"/>`;
    out += `<text x="${x}" y="${y + 3.5}" text-anchor="middle" font-size="${n === 1 ? 10 : 8}" font-weight="700" fill="#fff">${q}.${i}</text>`;
  });
  return out + "</svg>";
}

function checkText(why) { return why.length ? "fails: " + why.join("; ") : "passes"; }

function draw() {
  const choices = section().choices, pool = $("status").value === "proposals" ? PROPOSALS : SPOTS;
  const done = pool.filter(s => choices[s.id] && choices[s.id].decision).length;
  const doneHere = shown.filter(s => choices[s.id] && choices[s.id].decision).length;
  $("progress").textContent = shown.length
    ? `${at + 1} / ${shown.length} shown · reviewed ${doneHere} of these, ${done} of ${pool.length} overall` : "";
  if (!shown.length) { $("main").innerHTML = `<div class="empty">No spot matches these filters.</div>`; return; }
  const s = shown[at], r = choices[s.id] || {}, proposal = s.status === "proposal";
  const head = s.moves.columns.map(c => `<th>${esc(c)}</th>`).join("");
  const rows = s.moves.rows.map((row, i) => `<tr><td>piece ${i}</td>${
    row.map(([text, differs]) => `<td class="${differs ? "diff" : ""}">${esc(text)}</td>`).join("")}</tr>`).join("");
  const titles = s.titles || ["Original", s.status === "repaired" ? "Repaired"
    : s.status === "excluded" ? "Excluded (no fix)" : "Unchanged"];
  const boards = s.boards.map((b, k) => `<div class="board"><h2>${esc(titles[k])}${
    proposal ? ` · dice ${s.dice[k]}` : ""}</h2>${svgBoard(b, s, k === s.boards.length - 1)}</div>`).join("");
  const dice = proposal ? (s.dice[1] === s.dice[2] ? `dice ${s.dice[1]}` : `dice ${s.dice[1]} now, ${s.dice[2]} proposed`)
    : `dice ${s.dice}`;
  const check = proposal ? "" : s.check.passed ? (s.status === "repaired" ? "passes after the fix" : "passes as given")
    : "fails: " + s.check.failed_on.join("; ");
  const details = proposal ? `
      <b>Reason</b> (${esc(s.kind)})<ul><li>${esc(s.reason)}</li></ul>
      <b>Proposed changes</b><ul><li>${esc(s.changes)}</li></ul>
      <b>Full check</b>: current ${esc(checkText(s.audit.current))}; proposed ${esc(checkText(s.audit.proposed))}` : `
      ${s.cause.length ? `<b>Cause</b><ul>${s.cause.map(c => `<li>${esc(c)}</li>`).join("")}</ul>` : ""}
      ${s.fix ? `<b>Fix</b><ul><li>${esc(s.fix)}</li></ul>` : ""}
      <b>Check</b>: ${esc(check)}`;
  $("main").innerHTML = `
    <div class="title"><h1>${esc(s.id)}</h1>
      <span class="badge ${s.status}">${proposal ? "proposal: " + esc(s.kind) : s.status}</span>
      ${s.flags.map(f => `<span class="badge">${esc(f)}</span>`).join("")}
      ${r.decision ? `<span class="badge ${r.decision}">${r.decision === "keep" ? "kept" : "rejected"}</span>` : ""}</div>
    <div class="about"><b>${esc(s.category)}</b>: ${esc(s.intent)}<br>
      <b>Note:</b> ${esc(s.note)}${s.history ? `<br><b>History:</b> ${esc(s.history)}` : ""}<br>
      <b>P${s.mover} to move, ${dice}</b> · players ${s.players.join(", ")} · ${esc(s.file)}</div>
    <div class="boards">${boards}</div>
    <div class="panel"><table><thead><tr><th></th>${head}</tr></thead><tbody>${rows}</tbody></table></div>
    <div class="panel">${details}
    </div>
    <div class="panel"><label for="note"><b>Notes</b></label>
      <textarea id="note" placeholder="Anything to remember about this spot">${esc(r.note || "")}</textarea>
      <div class="keys">&larr; &rarr; previous / next · K keep · R reject · choices are kept in this browser; Save writes them to a file</div></div>`;
  $("note").addEventListener("input", e => {
    choices[s.id] = Object.assign(choices[s.id] || {}, {note: e.target.value}); persist();
  });
}

function decide(d) {
  if (!shown.length) return;
  const id = shown[at].id, choices = section().choices;
  choices[id] = Object.assign(choices[id] || {}, {decision: d}); persist();
  if ($("status").value === "unreviewed") { refilter(); return; }
  if (at < shown.length - 1) at++;
  draw();
}

async function save() {
  const sec = section();
  const body = JSON.stringify({about: sec.about, spots: sec.choices}, null, 1);
  try {
    if (window.showSaveFilePicker) {
      const h = await showSaveFilePicker({suggestedName: sec.file,
        types: [{description: "JSON", accept: {"application/json": [".json"]}}]});
      const w = await h.createWritable(); await w.write(body); await w.close(); return;
    }
  } catch (e) { if (e.name === "AbortError") return; }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([body], {type: "application/json"}));
  a.download = sec.file; a.click(); URL.revokeObjectURL(a.href);
}

if (PROPOSALS.length) $("status").insertAdjacentHTML("beforeend",
  `<option value="proposals">Proposals (${PROPOSALS.length})</option>`);
$("cat").innerHTML = `<option value="all">all categories</option>` +
  [...new Set(SPOTS.map(s => s.category))].map(c => `<option>${c}</option>`).join("");
$("cat").onchange = $("status").onchange = () => { at = 0; shown = []; refilter(); };
$("prev").onclick = () => { if (at > 0) { at--; draw(); } };
$("next").onclick = () => { if (at < shown.length - 1) { at++; draw(); } };
$("keep").onclick = () => decide("keep");
$("reject").onclick = () => decide("reject");
$("save").onclick = save;
$("load").onclick = () => $("file").click();
$("file").onchange = async e => {
  const f = e.target.files[0]; if (!f) return;
  try { Object.assign(section().choices, JSON.parse(await f.text()).spots || {}); persist(); draw(); }
  catch (err) { alert("Not a saved review file: " + err.message); }
  e.target.value = "";
};
document.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT" || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === "ArrowLeft") $("prev").click();
  else if (e.key === "ArrowRight") $("next").click();
  else if (e.key === "k" || e.key === "K") decide("keep");
  else if (e.key === "r" || e.key === "R") decide("reject");
});
if (location.hash === "#proposals" && PROPOSALS.length) { $("status").value = "proposals"; refilter(); }
else {
const linked = SPOTS.findIndex(s => s.id === decodeURIComponent(location.hash.slice(1)));
if (linked >= 0) { $("status").value = "all"; shown = SPOTS.filter(matches); at = shown.indexOf(SPOTS[linked]); draw(); }
else refilter();
}
</script>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("--src", default="spots_40")
    ap.add_argument("--out", default=os.path.join("review", "spots.html"))
    ap.add_argument("--proposals", metavar="FILE",
                    help="add a Proposals section with the proposed fixes in FILE")
    ap.add_argument("--final", default="spots_40_final",
                    help="the folder the proposals change (default: spots_40_final)")
    args = ap.parse_args()
    data = build(args.src, args.out, args.proposals, args.final)
    extra = f" and {len(json.load(open(args.proposals, encoding='utf-8'))['spots'])} proposals"         if args.proposals else ""
    print(f"{len(data)} spots{extra} -> {args.out}")


if __name__ == "__main__":
    main()
