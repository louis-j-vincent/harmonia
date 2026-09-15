"""Cheap symbolic experiments on the accompaniment DB (db.jsonl only, no MIDI).

H2  — Jazz harmonic rhythm: chord-duration PMF, P(change | beat phase),
      and cadence acceleration (does harmonic rhythm speed up at section ends?).
      Also the untested "trigram implies timing" hypothesis from
      docs/structure_trigram_design_2026-07-04.md: is the I of a ii-V-I held
      longer than other I chords?
H5  — Bass reliability by beat phase: P(bass interval | beat-in-bar), i.e.
      which beats actually carry root evidence — learned weights for the
      bass-anchored scorer.
H3  — Jazz chord n-grams relative to the song key, comparable to the POP909
      table in docs/architecture_extensions.md item #10.

Usage: .venv/bin/python scripts/analyze_accomp_priors.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DB = REPO / "data" / "accomp_db" / "db.jsonl"
PLOT_DIR = REPO / "docs" / "plots" / "accomp_db"

NOTE_TO_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
DEGREE_NAMES = ["I", "bII", "II", "bIII", "III", "IV", "bV", "V", "bVI", "VI", "bVII", "VII"]


def parse_root(chord: str) -> int | None:
    if not chord or chord == "z":
        return None
    pc = NOTE_TO_PC.get(chord[0])
    if pc is None:
        return None
    if len(chord) > 1:
        if chord[1] == "#":
            pc += 1
        elif chord[1] == "b":
            pc -= 1
    return pc % 12


def parse_key(key: str) -> tuple[int, str] | None:
    """iReal key like 'Ab' (major) or 'E-' (minor) → (tonic pc, mode)."""
    if not key:
        return None
    pc = NOTE_TO_PC.get(key[0])
    if pc is None:
        return None
    i = 1
    if len(key) > i and key[i] in "#b":
        pc += 1 if key[i] == "#" else -1
        i += 1
    mode = "minor" if key[i:].startswith("-") else "major"
    return pc % 12, mode


def quality_bucket(chord: str) -> str:
    """Collapse an MMA chord name to maj/min/dom/other for n-gram readability."""
    body = chord[1:]
    if body[:1] in ("#", "b"):
        body = body[1:]
    body = body.split("/")[0]
    if body.startswith(("m7b5", "m9b5", "m11b5")):
        return "min"  # half-dim grouped with min for the coarse table
    if body.startswith("m") and not body.startswith("maj") and not body.startswith("M"):
        return "min"
    if body.startswith(("7", "9", "13", "11")) or "7alt" in body:
        return "dom"
    if body.startswith(("dim", "aug", "sus", "5")):
        return "other"
    return "maj"


def per_beat_chords(rec: dict) -> list[str]:
    """Chord active at each beat of the song (from the slot timeline)."""
    bpb = rec["beats_per_bar"]
    n_beats = rec["n_bars"] * bpb
    beats = ["z"] * n_beats
    for ev in rec["chord_timeline"]:
        idx = (ev["bar"] - 1) * bpb + ev["beat"]
        if idx < n_beats:
            beats[idx] = ev["mma"]
    # forward-fill within the song
    for i in range(1, n_beats):
        if beats[i] == "z" and i % bpb != 0:
            pass  # explicit z stays z only if it was an actual slot; slots fill below
    cur = "z"
    filled = []
    slot_starts = {(ev["bar"] - 1) * bpb + ev["beat"] for ev in rec["chord_timeline"]}
    for i in range(n_beats):
        if i in slot_starts:
            cur = beats[i]
        filled.append(cur)
    return filled


def merged_events(rec: dict) -> list[tuple[str, int, int]]:
    """(chord, start_beat, duration_beats) with adjacent identical chords merged."""
    beats = per_beat_chords(rec)
    events = []
    i = 0
    while i < len(beats):
        j = i
        while j < len(beats) and beats[j] == beats[i]:
            j += 1
        if beats[i] != "z":
            events.append((beats[i], i, j - i))
        i = j
    return events


def main() -> None:
    records = [json.loads(line) for line in open(DB)]
    jazz = [r for r in records if r["corpus"] == "jazz1460" and r["beats_per_bar"] == 4]
    print(f"{len(records)} records total; {len(jazz)} jazz 4/4 songs used for phase analyses\n")

    # ── H2a: chord-duration PMF (jazz) ─────────────────────────────────────────
    durations = Counter()
    for r in jazz:
        for _, _, d in merged_events(r):
            durations[min(d, 12)] += 1
    total = sum(durations.values())
    print("H2a — Jazz chord-duration PMF (beats), vs POP909 (15%/49%/9%/26% for 1/2/3/4):")
    for d in sorted(durations):
        if durations[d] / total > 0.005:
            print(f"    d={d:>2}: {durations[d]/total:6.1%}")
    print(f"    (n={total} merged chord events)\n")

    # ── H2b: P(change | beat phase) ────────────────────────────────────────────
    change = np.zeros(4)
    count = np.zeros(4)
    per_song_adv = []
    for r in jazz:
        beats = per_beat_chords(r)
        ch = np.zeros(4)
        ct = np.zeros(4)
        for i in range(1, len(beats)):
            ph = i % 4
            ct[ph] += 1
            if beats[i] != beats[i - 1]:
                ch[ph] += 1
        change += ch
        count += ct
        with np.errstate(invalid="ignore"):
            p = ch / ct
        if ct.min() > 4 and not np.isnan(p).any():
            per_song_adv.append(p[0] - p[3])
    p_change = change / count
    adv = np.array(per_song_adv)
    print("H2b — P(chord change | beat phase), jazz corpus (POP909: 50.8/42.0/29.3/22.8%):")
    for ph in range(4):
        print(f"    beat {ph}: {p_change[ph]:6.1%}   (n={int(count[ph])})")
    print(f"    per-song downbeat advantage P(b0)-P(b3): mean {adv.mean():+.2f}, "
          f"std {adv.std():.2f}, range [{adv.min():+.2f}, {adv.max():+.2f}]  "
          f"(POP909: +0.28 ± 0.53)\n")

    # ── H2c: cadence acceleration — harmonic rhythm by position in section ────
    pos_changes = defaultdict(lambda: [0, 0])  # quartile → [changes, beats]
    last2 = [0, 0]
    rest = [0, 0]
    for r in jazz:
        beats = per_beat_chords(r)
        bpb = r["beats_per_bar"]
        sections = r["section_per_bar"]
        # find section runs in bars
        runs = []
        i = 0
        while i < len(sections):
            j = i
            while j < len(sections) and sections[j] == sections[i]:
                j += 1
            runs.append((i, j))
            i = j
        for s, e in runs:
            n_bars = e - s
            if n_bars < 4:
                continue
            for bar in range(s, e):
                is_last2 = bar >= e - 2
                for b in range(bpb):
                    idx = bar * bpb + b
                    if idx == 0 or idx >= len(beats):
                        continue
                    tgt = last2 if is_last2 else rest
                    tgt[1] += 1
                    if beats[idx] != beats[idx - 1]:
                        tgt[0] += 1
    print("H2c — Cadence acceleration (chord-change rate, sections ≥4 bars):")
    print(f"    last 2 bars of section : {last2[0]/last2[1]:6.1%}  (n={last2[1]})")
    print(f"    rest of section        : {rest[0]/rest[1]:6.1%}  (n={rest[1]})\n")

    # ── H2d: is the I of a ii-V-I held longer than other tonic chords? ────────
    dur_I_after_iiV = Counter()
    dur_I_other = Counter()
    for r in jazz:
        k = parse_key(r["key"])
        if k is None:
            continue
        tonic, _ = k
        evs = merged_events(r)
        seq = [(parse_root(c), quality_bucket(c), d) for c, _, d in evs]
        for i, (root, qual, d) in enumerate(seq):
            if root is None:
                continue
            deg = (root - tonic) % 12
            if deg != 0 or qual not in ("maj", "min"):
                continue
            is_iiV = (
                i >= 2
                and seq[i - 1][0] is not None and seq[i - 2][0] is not None
                and (seq[i - 1][0] - tonic) % 12 == 7 and seq[i - 1][1] == "dom"
                and (seq[i - 2][0] - tonic) % 12 == 2 and seq[i - 2][1] == "min"
            )
            (dur_I_after_iiV if is_iiV else dur_I_other)[min(d, 12)] += 1
    nA, nB = sum(dur_I_after_iiV.values()), sum(dur_I_other.values())
    print("H2d — Duration of tonic chord: resolving a ii-V vs otherwise:")
    print(f"    {'d':>4} {'after ii-V':>12} {'other I':>10}")
    for d in sorted(set(dur_I_after_iiV) | set(dur_I_other)):
        a = dur_I_after_iiV[d] / nA if nA else 0
        b = dur_I_other[d] / nB if nB else 0
        if a > 0.01 or b > 0.01:
            print(f"    {d:>4} {a:>12.1%} {b:>10.1%}")
    meanA = sum(d * c for d, c in dur_I_after_iiV.items()) / max(nA, 1)
    meanB = sum(d * c for d, c in dur_I_other.items()) / max(nB, 1)
    print(f"    mean duration: after ii-V {meanA:.2f} beats (n={nA}), other {meanB:.2f} (n={nB})\n")

    # ── H5: bass interval by beat phase ────────────────────────────────────────
    phase_int = defaultdict(Counter)  # phase → Counter(interval)
    for r in jazz:
        if r["groove"] not in ("Swing", "Ballad"):  # walking-bass grooves only
            continue
        spb = 60.0 / r["tempo"]
        bpb = r["beats_per_bar"]
        beats = per_beat_chords(r)
        for pitch, start, _end, _vel in r["bass_notes"]:
            beat_f = start / spb
            beat_i = int(round(beat_f))
            if abs(beat_f - beat_i) > 0.25 or beat_i >= len(beats):
                continue  # skip off-beat passing notes for the phase table
            chord = beats[beat_i]
            root = parse_root(chord)
            if root is None:
                continue
            phase_int[beat_i % bpb][(pitch - root) % 12] += 1
    print("H5 — P(bass interval | beat phase), Swing+Ballad grooves, on-beat notes:")
    print(f"    {'phase':>5} {'root':>7} {'3rd':>6} {'5th':>6} {'7th':>6} {'other':>7} {'n':>8}")
    for ph in sorted(phase_int):
        c = phase_int[ph]
        n = sum(c.values())
        root_p = c[0] / n
        third = (c[3] + c[4]) / n
        fifth = c[7] / n
        seventh = (c[10] + c[11]) / n
        other = 1 - root_p - third - fifth - seventh
        print(f"    {ph:>5} {root_p:>7.1%} {third:>6.1%} {fifth:>6.1%} {seventh:>6.1%} "
              f"{other:>7.1%} {n:>8}")
    print()

    # ── H3: jazz n-grams relative to key ───────────────────────────────────────
    bigrams = Counter()
    trigram_ctx = defaultdict(Counter)
    n_trans = 0
    for r in jazz:
        k = parse_key(r["key"])
        if k is None:
            continue
        tonic, _ = k
        evs = merged_events(r)
        seq = []
        for c, _, _ in evs:
            root = parse_root(c)
            if root is None:
                continue
            seq.append((DEGREE_NAMES[(root - tonic) % 12], quality_bucket(c)))
        for a, b in zip(seq, seq[1:]):
            if a == b:
                continue
            bigrams[(a, b)] += 1
            n_trans += 1
        for a, b, c in zip(seq, seq[1:], seq[2:]):
            trigram_ctx[(a, b)][c] += 1
    print(f"H3 — Top jazz scale-degree bigrams ({n_trans} transitions, "
          "vs POP909: V→I 9.6%, IV→V 5.2%, I→IV 4.6%):")

    def fmt(dq):
        d, q = dq
        return {"maj": d, "min": d.lower() + "m", "dom": d + "7", "other": d + "*"}[q]

    for (a, b), n in bigrams.most_common(15):
        print(f"    {fmt(a):>6} → {fmt(b):<6} {n/n_trans:6.2%}")
    # predictiveness of trigram context for the classic case
    ii = ("II", "min")
    V = ("V", "dom")
    ctx = trigram_ctx[(ii, V)]
    n_ctx = sum(ctx.values())
    print(f"\n    (ii,V) context: n={n_ctx}; next-chord distribution:")
    for c, n in ctx.most_common(5):
        print(f"        → {fmt(c):<6} {n/n_ctx:6.1%}")
    # compare against bigram P(next | V) alone
    after_V = Counter()
    for (a, b), n in bigrams.items():
        if a == V:
            after_V[b] += n
    n_afterV = sum(after_V.values())
    print(f"    P(next | V alone): n={n_afterV}")
    for c, n in after_V.most_common(5):
        print(f"        → {fmt(c):<6} {n/n_afterV:6.1%}")
    # tritone sub / classic jazz moves POP909 lacks
    print("\n    Jazz-specific moves:")
    for name, pair in [
        ("tritone sub  bII7→I", (("bII", "dom"), ("I", "maj"))),
        ("backdoor    bVII7→I", (("bVII", "dom"), ("I", "maj"))),
        ("iim→V7", (("II", "min"), ("V", "dom"))),
        ("V7→I", (("V", "dom"), ("I", "maj"))),
        ("V7→im", (("V", "dom"), ("I", "min"))),
    ]:
        print(f"        {name:<22} {bigrams[pair]/n_trans:6.2%}")

    # trigram sparsity
    n_ctx_total = len(trigram_ctx)
    obs_per_ctx = [sum(c.values()) for c in trigram_ctx.values()]
    obs = np.array(obs_per_ctx)
    print(f"\n    Trigram sparsity: {n_ctx_total} observed contexts, "
          f"median obs/context {np.median(obs):.0f}, "
          f"{(obs >= 20).mean():.0%} of contexts have ≥20 obs, "
          f"top-50 contexts cover {np.sort(obs)[::-1][:50].sum()/obs.sum():.0%} of mass")


if __name__ == "__main__":
    main()
