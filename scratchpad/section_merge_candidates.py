"""section_merge_candidates.py — 2026-07-18, section-level suggestion tool,
task 4: generate SECTION-level (8-bar standard, 4-bar secondary) repeat-
detection candidate JSON for the 3 real songs, in a format analogous to
(but distinct from) `bar_merge_candidates_<slug>.json`.

**Operating point, derived from section_roc_jazz_only.py's nested corpus
calibration (jazz1460-only — the honestly relevant subset, see that
script's docstring for why the full 7-playlist corpus dilutes the signal):
grain=8, target_recall=0.80 -> tau_symbolic_proxy~=0.227 on the SYMBOLIC
proxy feature scale.** That number is NOT directly portable to real audio
as a literal threshold (same lesson as bar-level tau_auto=0.96 -- see
section_realaudio_check.py's finding that real-audio-only block similarity
at grain=8 is an even NOISIER same-section proxy than bar-level, with only
1194 pooled real pairs across all 3 songs and single-digit counts above
tau=0.93). Given that and the tiny absolute candidate pool sizes on real
audio (aretha: 36 total grain=8 pairs; autumn_leaves: 780; abba: 378), this
generator does NOT hard-threshold on a ported symbolic tau. Instead it:

  1. Ranks ALL non-adjacent block pairs (min_gap=1 block) by real-audio
     bt_concat block_sim (position-aligned, hierarchy_shortcut-derived).
  2. Computes symbolic_sim (baseline-decode V1_binary chord-tone cosine,
     position-aligned) for every candidate as a TRANSPARENCY field, exactly
     like bar_merge_candidates.py's `symbolic_sim`/`tier_reason` pattern --
     NOT used to hard-filter here (the joint real-audio sweep showed doing
     so on this granularity leaves as few as 0-11 candidates per song,
     an empty tool), but exposed so the UI (or a future tighter gate) can
     use it as a secondary confidence signal / sort key.
  3. Emits ALL candidates above a low audio floor (0.5) up to a cap, tagged
     tier="suggest" ONLY -- no "auto" tier. Given the corpus-scale finding
     that even the best (jazz-only, grain=8) operating point tops out at
     ~50% precision at usable recall, auto-apply-without-review would be
     irresponsible here; unlike the bar-merge tool, EVERY section
     suggestion should require human confirmation.

Honest confidence caveat baked into `meta`: this tool's candidates are
LOWER CONFIDENCE than the bar-merge tool's (symbolic corpus precision
~0.47-0.54 at 80-70% recall vs bar-level's ~0.84-0.92); the UI should not
reuse the bar tool's "high-confidence, quick-confirm" framing verbatim --
see the handoff note in docs/known_issues.md / research session log.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from rawchroma import per_bar_rawchroma
from hierarchy_shortcut import diagonal_prefix_sums, diag_sum
from chord_distance_eval import nuclear_spans
from chord_distance import chord_vector_binary
from realaudio_threshold_check import get_baseline_chords, label_bucket
from auto_apply_merges import SONGS, AUDIO_DIR

OUT_DIR = Path(__file__).resolve().parent
# see section_pairs.py's identical fix note: adjacent blocks are the
# primary use case at block grain (e.g. Autumn Leaves bars 1-8 vs 9-16),
# not a trivial case to exclude.
MIN_GAP_BLOCKS = 0
AUDIO_FLOOR = 0.50       # low floor -- rank-based tool, not threshold-gated
# matches bar_merge_candidates.py's max_candidates=60 convention. Checked
# empirically (this call): at 40, the user's own Autumn Leaves worked
# example (block 0 vs 1, audio_sim=0.789) ranked 41st (just outside the
# cap) after the per-block dedup -- raised to 60 so the tool's own
# motivating example isn't silently dropped by an arbitrary cap.
MAX_CANDIDATES = 60
MAX_PAIRS_PER_BLOCK = 3
# jazz1460-only nested calibration (section_roc_jazz_only_results.json),
# grain=8, target_recall=0.80: tau_mean=0.2274 on the SYMBOLIC proxy scale
# -- reported in meta for context, not applied as a real-audio hard gate
# (see module docstring).
SYMBOLIC_TAU_REFERENCE = {8: 0.2274, 4: 0.1582}


def block_gram_sim(prefix, sq, n, i0, j0, L):
    d = j0 - i0
    num = diag_sum(prefix, n, d, i0, L)
    na = np.sqrt(float(np.sum(sq[i0:i0 + L])))
    nb = np.sqrt(float(np.sum(sq[j0:j0 + L])))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(num / (na * nb))


def bar_label_bucket(base_chords, t0, t1):
    overlaps = {}
    for c in base_chords:
        s, e = max(c["start_s"], t0), min(c["end_s"], t1)
        if e <= s:
            continue
        b = label_bucket(c.get("label"))
        if b is None:
            continue
        overlaps[b] = overlaps.get(b, 0.0) + (e - s)
    if not overlaps:
        return None
    return max(overlaps.items(), key=lambda kv: kv[1])[0]


def generate(slug, audio_name, grain, max_candidates=MAX_CANDIDATES):
    audio_path = AUDIO_DIR / audio_name
    variants, bar_times, tempo, tonic = per_bar_rawchroma(audio_path)
    base_ch = get_baseline_chords(slug)
    n = len(variants["bt_concat"])
    spans = nuclear_spans(n, grain)
    m = len(spans)
    bar_bucket = [bar_label_bucket(base_ch, bar_times[i], bar_times[i + 1]) for i in range(n)]

    v = variants["bt_concat"]
    row_norm = np.linalg.norm(v, axis=1, keepdims=True)
    v_unit = v / np.clip(row_norm, 1e-9, None)
    G = v_unit @ v_unit.T
    sq = np.diag(G).copy()
    prefix = diagonal_prefix_sums(G)

    edges = []
    for i in range(m):
        si, ei = spans[i]
        for j in range(i + 1 + MIN_GAP_BLOCKS, m):
            sj, ej = spans[j]
            L = min(ei - si, ej - sj)
            audio_sim = block_gram_sim(prefix, sq, n, si, sj, L)
            if audio_sim < AUDIO_FLOOR:
                continue
            sym_vecs_a, sym_vecs_b, valid = [], [], 0
            for t in range(L):
                ba, bb = bar_bucket[si + t], bar_bucket[sj + t]
                sym_vecs_a.append(chord_vector_binary(*ba) if ba else np.zeros(12))
                sym_vecs_b.append(chord_vector_binary(*bb) if bb else np.zeros(12))
                if ba is not None and bb is not None:
                    valid += 1
            num = sum(float(np.dot(a, b)) for a, b in zip(sym_vecs_a, sym_vecs_b))
            na = np.sqrt(sum(float(np.dot(a, a)) for a in sym_vecs_a))
            nb = np.sqrt(sum(float(np.dot(b, b)) for b in sym_vecs_b))
            symbolic_sim = num / (na * nb) if na > 1e-9 and nb > 1e-9 else None
            edges.append((audio_sim, i, j, si, ei, sj, ej, symbolic_sim, valid, L))

    edges.sort(key=lambda e: -e[0])
    used = {}
    out = []
    for audio_sim, i, j, si, ei, sj, ej, symbolic_sim, valid, L in edges:
        if used.get(i, 0) >= MAX_PAIRS_PER_BLOCK or used.get(j, 0) >= MAX_PAIRS_PER_BLOCK:
            continue
        used[i] = used.get(i, 0) + 1
        used[j] = used.get(j, 0) + 1
        out.append({
            "blocks": [i, j],
            "bars": [[si, ei], [sj, ej]],
            "spans": [[float(bar_times[si]), float(bar_times[ei])],
                      [float(bar_times[sj]), float(bar_times[ej])]],
            "confidence": audio_sim,
            "symbolic_sim": symbolic_sim,
            "n_bars_aligned": L,
            "n_bars_symbolic_valid": valid,
            "grain": grain,
            "tier": "suggest",
        })
        if len(out) >= max_candidates:
            break

    meta = {
        "n_bars_total": n, "n_blocks": m, "tempo_bpm": tempo, "tonic_pc": tonic,
        "grain": grain, "min_gap_blocks": MIN_GAP_BLOCKS, "audio_floor": AUDIO_FLOOR,
        "n_total_pairs_computed": len(edges), "n_candidates_emitted": len(out),
        "symbolic_tau_reference_jazz1460_80pct_recall": SYMBOLIC_TAU_REFERENCE.get(grain),
        "confidence_caveat": (
            "LOWER CONFIDENCE than bar_merge_candidates: corpus precision at "
            "usable recall is ~0.47-0.54 (grain=8, jazz1460-only calibration), "
            "vs the bar tool's ~0.84-0.92. Real-audio-only similarity does not "
            "reliably indicate same-section identity even at high sim (see "
            "section_realaudio_check_results.json) -- confidence should be "
            "surfaced to the user as 'rank order, not a probability', and "
            "candidates require human confirmation (no auto-tier)."
        ),
    }
    return out, meta


def main():
    for slug, sm in SONGS.items():
        for grain in (8, 4):
            candidates, meta = generate(slug, sm["audio_name"], grain)
            fname = f"section_merge_candidates_{slug}_grain{grain}.json"
            (OUT_DIR / fname).write_text(json.dumps({"candidates": candidates, "meta": meta}, indent=2))
            print(f"{slug} grain={grain}: {len(candidates)} candidates "
                  f"(of {meta['n_total_pairs_computed']} pairs >= floor {AUDIO_FLOOR}) -> {fname}",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
