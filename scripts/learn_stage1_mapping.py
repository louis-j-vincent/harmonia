"""Learn/measure the audio→notes mapping on rendered accompaniment audio.

Uses the (audio, perfect MIDI ground truth) pairs from build_accomp_audio.py.
The beat grid is KNOWN (fixed tempo, no beat-tracker noise), so every
degradation measured here is attributable to Basic Pitch's audio→notes step —
the bottleneck identified in docs/accomp_db_signal_analysis_2026-07-03.md.

Measurements / learned components:
  1. Per-beat evidence quality (cosine BP chroma vs GT chroma), by condition
     (clean / noisy / soundfont) and by observation channel
     (onset-only = pipeline default, vs hybrid onset + α·note).
  2. Per-key sensitivity curve (BP activation when key sounding vs not) —
     fits the per-key calibration `docs/suggestions.md` proposed, on real GT.
  3. THE headline: chord-quality classification with real BP evidence
     (true root given), hand vs MIDI-learned vs noisy-learned templates,
     with and without per-key calibration → how much of the
     perfect-notes-95% ceiling survives the audio round-trip, and how much
     a learned linear mapping recovers.

Usage: .venv/bin/python scripts/learn_stage1_mapping.py [--alpha 0.15]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pretty_midi

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from analyze_accomp_emission import (  # noqa: E402
    HAND_TEMPLATES,
    parse_chord,
    song_chord_spans,
)
from harmonia.models.stage1_pitch import PitchExtractor  # noqa: E402

DB = REPO / "data" / "accomp_db" / "db.jsonl"
MANIFEST = REPO / "data" / "accomp_db" / "audio" / "manifest.jsonl"
MIDI_START = 21


def gt_beat_roll(pm: pretty_midi.PrettyMIDI, n_beats: int, spb: float,
                 transpose: int) -> np.ndarray:
    """(n_beats, 88) duration-weighted GT note activity (transposed)."""
    roll = np.zeros((n_beats, 88), dtype=np.float32)
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            key = n.pitch + transpose - MIDI_START
            if not (0 <= key < 88):
                continue
            b0, b1 = int(n.start / spb), int(np.ceil(n.end / spb))
            for b in range(max(b0, 0), min(b1, n_beats)):
                ov = min(n.end, (b + 1) * spb) - max(n.start, b * spb)
                if ov > 0:
                    roll[b, key] += ov
    return roll


def pool_beats(frame_times: np.ndarray, probs: np.ndarray, n_beats: int,
               spb: float) -> np.ndarray:
    """Sum frame activations into beat windows (mirrors rhythm.quantise_frames)."""
    out = np.zeros((n_beats, probs.shape[1]), dtype=np.float32)
    mask = frame_times < n_beats * spb
    idx = (frame_times[mask] / spb).astype(int)
    np.add.at(out, idx, probs[mask])
    return out


def to_chroma(v88: np.ndarray) -> np.ndarray:
    c = np.zeros(v88.shape[:-1] + (12,))
    for k in range(88):
        c[..., (k + MIDI_START) % 12] += v88[..., k]
    return c


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return np.nan
    return float(a @ b / (na * nb))


def tvec(d: dict) -> np.ndarray:
    v = np.zeros(12)
    for i, w in d.items():
        v[i] = w
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.15,
                    help="note-channel weight in the hybrid observation")
    ap.add_argument("--limit", type=int, default=None, help="max renders to process")
    args = ap.parse_args()

    records = {r["song_id"]: r for r in map(json.loads, open(DB))}
    manifest = [json.loads(line) for line in open(MANIFEST)]
    if args.limit:
        manifest = manifest[: args.limit]
    extractor = PitchExtractor(cache_dir=REPO / "data" / "cache" / "accomp")

    # accumulators
    quality_cos = defaultdict(list)          # condition → per-beat cosines (onset-only)
    quality_cos_hybrid = defaultdict(list)
    key_on = np.zeros(88)
    key_on_n = np.zeros(88)
    key_off = np.zeros(88)
    key_off_n = np.zeros(88)
    # chord instances: (quality, BP chroma onset, BP chroma hybrid, BP 88 hybrid, song_id)
    instances = []
    n_files = 0

    for m in manifest:
        wav = REPO / m["wav"]
        if not wav.exists():
            continue
        rec = records[m["song_id"]]
        spb = 60.0 / m["tempo"]
        n_beats = m["n_bars"] * m["beats_per_bar"]
        try:
            acts = extractor.extract(wav)
        except Exception as e:
            print(f"  BP failed on {wav.name}: {e}")
            continue
        onset_b = pool_beats(acts.frame_times, acts.onset_probs, n_beats, spb)
        note_b = pool_beats(acts.frame_times, acts.note_probs, n_beats, spb)
        # scale the (much heavier) sustain channel to the onset channel's total
        # mass before mixing, so alpha means what it says
        note_scale = onset_b.sum() / max(note_b.sum(), 1e-9)
        note_b = note_b * note_scale
        hybrid_b = onset_b + args.alpha * note_b

        pm = pretty_midi.PrettyMIDI(str(REPO / m["midi_path"]))
        gt = gt_beat_roll(pm, n_beats, spb, m["transpose"])
        gt_c = to_chroma(gt)
        on_c = to_chroma(onset_b)
        hy_c = to_chroma(hybrid_b)

        cond = ("clean" if m["snr_db"] is None else f"snr{int(m['snr_db'])}")
        if m["transpose"] == 0 and m["soundfont"].startswith("MuseScore") and m["snr_db"] is None:
            cond = "canonical"
        for b in range(n_beats):
            c1, c2 = cos(on_c[b], gt_c[b]), cos(hy_c[b], gt_c[b])
            if not np.isnan(c1):
                quality_cos[cond].append(c1)
            if not np.isnan(c2):
                quality_cos_hybrid[cond].append(c2)

        # per-key sensitivity on the SUSTAIN channel (the onset channel is
        # sparse by design — sustained chord beats have no onset, which would
        # conflate onset sparsity with per-key sensitivity)
        active = gt > 0.2 * spb
        key_on += (note_b * active).sum(axis=0)
        key_on_n += active.sum(axis=0)
        key_off += (note_b * ~active).sum(axis=0)
        key_off_n += (~active).sum(axis=0)

        # chord instances (root transposed with the MIDI)
        for t0, t1, root, qual in song_chord_spans(rec):
            b0, b1 = int(t0 / spb), min(int(t1 / spb), n_beats)
            if b1 <= b0:
                continue
            root_t = (root + m["transpose"]) % 12
            span_on = to_chroma(onset_b[b0:b1].sum(axis=0))
            span_hy88 = hybrid_b[b0:b1].sum(axis=0)
            span_hy = to_chroma(span_hy88)
            instances.append((qual, np.roll(span_on, -root_t),
                              np.roll(span_hy, -root_t), span_hy88, root_t,
                              m["song_id"], cond))
        n_files += 1

    print(f"{n_files} renders analysed, {len(instances)} chord instances\n")

    # ── 1. evidence quality by condition ───────────────────────────────────────
    print("1 — Per-beat chroma agreement with GT (cosine), onset-only vs hybrid "
          f"(α={args.alpha}):")
    for cond in sorted(quality_cos):
        a = np.array(quality_cos[cond])
        h = np.array(quality_cos_hybrid[cond])
        print(f"    {cond:<10} onset {np.nanmean(a):.3f}   hybrid {np.nanmean(h):.3f}"
              f"   (n={len(a)} beats)")
    print()

    # ── 2. per-key sensitivity + calibration ──────────────────────────────────
    mu_on = key_on / np.maximum(key_on_n, 1)
    mu_off = key_off / np.maximum(key_off_n, 1)
    sens = mu_on / np.maximum(mu_off, 1e-9)
    print("2 — Per-key sensitivity (mean activation ratio on/off), by register:")
    for lo, hi, name in [(0, 19, "A0–E2 (low)"), (19, 43, "F2–E4 (mid-low)"),
                         (43, 67, "F4–E6 (mid-high)"), (67, 88, "F6–C8 (high)")]:
        print(f"    {name:<18} on/off ratio {np.nanmean(sens[lo:hi]):6.1f}   "
              f"mean μ_on {np.nanmean(mu_on[lo:hi]):.3f}")
    calib = 1.0 / np.maximum(mu_on, np.nanmax(mu_on) * 0.05)  # sensitivity equalization
    print()

    # ── 3. chord-quality classification on real BP evidence ───────────────────
    rng = np.random.default_rng(0)
    song_ids = sorted({s for *_, s, _ in instances})
    train_songs = {s for s in song_ids if rng.random() < 0.5}

    # MIDI-learned templates (perfect-notes, from the same songs' GT MIDI)
    midi_tmpl = defaultdict(lambda: np.zeros(12))
    midi_n = Counter()
    for sid in song_ids:
        rec = records[sid]
        pm = pretty_midi.PrettyMIDI(str(REPO / rec["midi_path"]))
        notes = [n for i in pm.instruments if not i.is_drum for n in i.notes]
        for t0, t1, root, qual in song_chord_spans(rec):
            v = np.zeros(12)
            for n in notes:
                ov = min(n.end, t1) - max(n.start, t0)
                if ov > 0:
                    v[(n.pitch - root) % 12] += ov
            if v.sum() > 1e-9 and sid in train_songs:
                midi_tmpl[qual] += v / v.sum()
                midi_n[qual] += 1

    # noisy-learned templates (BP evidence, train split, calibrated + raw)
    noisy_tmpl = defaultdict(lambda: np.zeros(12))
    noisy_cal_tmpl = defaultdict(lambda: np.zeros(12))
    noisy_n = Counter()
    for qual, _on, hy, hy88, root_t, sid, _cond in instances:
        if sid in train_songs and hy.sum() > 1e-9:
            noisy_tmpl[qual] += hy / hy.sum()
            calv = np.roll(to_chroma(hy88 * calib), -root_t)
            noisy_cal_tmpl[qual] += calv / max(calv.sum(), 1e-9)
            noisy_n[qual] += 1

    quals = sorted(q for q in noisy_n if noisy_n[q] >= 20 and midi_n[q] >= 20
                   and q in HAND_TEMPLATES)
    if not quals:
        sys.exit("Not enough chord instances per quality — run on more renders.")

    def norm_stack(td, ns=None):
        M = np.stack([td[q] / (ns[q] if ns else 1) for q in quals])
        return M / np.linalg.norm(M, axis=1, keepdims=True)

    sets = {
        "hand": np.stack([tvec(HAND_TEMPLATES[q]) for q in quals]),
        "midi-learned": norm_stack(midi_tmpl, midi_n),
        "noisy-learned": norm_stack(noisy_tmpl, noisy_n),
        "noisy-learned+calib": norm_stack(noisy_cal_tmpl, noisy_n),
    }
    sets = {k: M / np.linalg.norm(M, axis=1, keepdims=True) for k, M in sets.items()}

    def collapse(q):
        return {"maj": "maj", "maj7": "maj", "6": "maj", "aug": "maj", "augmaj7": "maj",
                "min": "min", "min7": "min", "m6": "min", "minmaj7": "min",
                "dom7": "dom", "dom7alt": "dom", "aug7": "dom", "7sus4": "dom",
                "sus4": "other", "sus2": "other", "m7b5": "dim", "dim": "dim",
                "dim7": "dim"}.get(q, "other")

    results = {k: [0, 0] for k in sets}   # fine, collapsed
    n_test = 0
    conf = defaultdict(Counter)
    for qual, _on, hy, hy88, root_t, sid, _cond in instances:
        if sid in train_songs or qual not in quals:
            continue
        feats = {
            "hand": hy, "midi-learned": hy, "noisy-learned": hy,
            "noisy-learned+calib": np.roll(to_chroma(hy88 * calib), -root_t),
        }
        n_test += 1
        for k, M in sets.items():
            v = feats[k]
            vn = v / (np.linalg.norm(v) + 1e-12)
            pred = quals[int(np.argmax(M @ vn))]
            results[k][0] += pred == qual
            results[k][1] += collapse(pred) == collapse(qual)
            if k == "noisy-learned+calib" and pred != qual:
                conf[k][(qual, pred)] += 1

    print(f"3 — Chord-quality classification on REAL Basic Pitch evidence "
          f"({n_test} test instances, {len(quals)} qualities, true root given):")
    print("    (reference ceilings with perfect MIDI notes: hand 90.3%, learned 95.4%)")
    for k, (fine, coll) in results.items():
        print(f"    {k:<22} fine {fine/n_test:6.1%}   collapsed maj/min/dom/dim "
              f"{coll/n_test:6.1%}")
    print("    top confusions (noisy-learned+calib):",
          [(f"{a}→{b}", n) for (a, b), n in conf["noisy-learned+calib"].most_common(6)])


if __name__ == "__main__":
    main()
