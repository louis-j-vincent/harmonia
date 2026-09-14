"""One-off migration: fix motif chord grouping on already-rendered chart
HTML. Bug: run-id was keyed on name+pos, which is unique per chord within
one motif instance almost by construction (pos increments 0,1,2...) — so
every chord got its own run-id and a "grouped" motif (e.g. a 2-chord ii-V)
never actually grouped into one visual bracket; it drew one box per chord,
with a visible gap/overlap whenever the pair wrapped onto two lines.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLOTS_DIR = REPO / "docs" / "plots"

OLD_RUNID = """  // Assign a unique run id per contiguous occurrence (same name+pos, consecutive indices)
  const chordRunId = {};
  let runCounter = 0, prevKey = null, prevIdx = -2;
  P.chords.forEach((_, i) => {
    const a = ann[i];
    if (!a || a.len < 2) { prevKey = null; return; }
    const key = a.name + ':' + a.pos;
    if (key !== prevKey || i !== prevIdx + 1) runCounter++;
    chordRunId[i] = runCounter;
    prevKey = key; prevIdx = i;
  });"""
NEW_RUNID = """  // Assign one run id per motif *instance* — a new id starts at pos===0
  // (the first chord of an occurrence) and every following chord of that
  // same occurrence (pos 1, 2, ...) keeps it, so a 2+ chord motif like a
  // ii-V draws as one grouped bracket instead of one box per chord. (Bug
  // history: this used to key on name+pos, which is unique per chord within
  // an instance almost by construction — so every chord got its own id and
  // "grouped" motifs never visually grouped.)
  const chordRunId = {};
  let runCounter = 0, prevIdx = -2;
  P.chords.forEach((_, i) => {
    const a = ann[i];
    if (!a || a.len < 2) { return; }
    if (a.pos === 0 || i !== prevIdx + 1) runCounter++;
    chordRunId[i] = runCounter;
    prevIdx = i;
  });"""

OLD_SEGGROUPS = """  // Group chord indices by (runId, parent .chords container)
  // Track insertion order so we know if a segment is start/mid/end of its run
  let parentCounter = 0;
  const segGroups = new Map(); // key -> {els, a, rid, insertOrder}
  const ridSegCount = {}; // rid -> how many segments
  P.chords.forEach((_, i) => {
    const a = ann[i];
    if (!a || a.len < 2) return;
    const el = document.getElementById('chord-' + i);
    if (!el) return;
    const rid = chordRunId[i];
    const parent = el.parentNode;
    if (!parent._msid) parent._msid = ++parentCounter;
    const key = rid + '|' + parent._msid;
    if (!segGroups.has(key)) {
      segGroups.set(key, {els: [], a, rid, order: segGroups.size});
      ridSegCount[rid] = (ridSegCount[rid] || 0) + 1;
    }
    segGroups.get(key).els.push(el);
  });
  document.querySelectorAll('.chords').forEach(p => delete p._msid);

  // Count total segments per run so we can assign start/mid/end
  const ridSeenCount = {};
  // Build ordered list grouped by rid
  const byRid = {};
  segGroups.forEach((g, key) => {
    if (!byRid[g.rid]) byRid[g.rid] = [];
    byRid[g.rid].push(g);
  });
  Object.values(byRid).forEach(segs => segs.sort((a,b) => a.order - b.order));

  // Inject one .motif-segment per group, tagged with split position class
  segGroups.forEach(({els, a, rid}) => {
    const segsInRun = byRid[rid];
    const myIdx = segsInRun.indexOf(segGroups.get(
      rid + '|' + (els[0].parentNode._msid || 0)
    ));
    // recompute: just tag in order after injection
    const parent = els[0].parentNode;"""
NEW_SEGGROUPS = """  // Group chord indices by (runId, parent .chords container) — one segment
  // per bar per motif instance; drawMotifOutlines() unifies same-run
  // segments that land in different bars into one visual bracket.
  let parentCounter = 0;
  const segGroups = new Map(); // key -> {els, a, rid}
  P.chords.forEach((_, i) => {
    const a = ann[i];
    if (!a || a.len < 2) return;
    const el = document.getElementById('chord-' + i);
    if (!el) return;
    const rid = chordRunId[i];
    const parent = el.parentNode;
    if (!parent._msid) parent._msid = ++parentCounter;
    const key = rid + '|' + parent._msid;
    if (!segGroups.has(key)) segGroups.set(key, {els: [], a, rid});
    segGroups.get(key).els.push(el);
  });
  document.querySelectorAll('.chords').forEach(p => delete p._msid);

  // Inject one .motif-segment per group (one per bar; drawMotifOutlines()
  // unifies same-run segments across bars/rows into one visual bracket)
  segGroups.forEach(({els, a, rid}) => {
    const parent = els[0].parentNode;"""

REPLACEMENTS = [(OLD_RUNID, NEW_RUNID), (OLD_SEGGROUPS, NEW_SEGGROUPS)]


def main() -> None:
    files = sorted(PLOTS_DIR.glob("inferred_*.html"))
    patched, skipped = 0, 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        hits = sum(1 for old, _ in REPLACEMENTS if old in text)
        if hits == 0:
            skipped += 1
            continue
        for old, new in REPLACEMENTS:
            if old in text:
                text = text.replace(old, new, 1)
        f.write_text(text, encoding="utf-8")
        patched += 1
        print(f"patched {f.name} ({hits}/{len(REPLACEMENTS)} blocks)")
    print(f"\n{patched} patched, {skipped} skipped")


if __name__ == "__main__":
    sys.exit(main())
