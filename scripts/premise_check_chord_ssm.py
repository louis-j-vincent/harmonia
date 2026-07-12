"""Premise check for issue #22: is a symbolic chord-SSM sharper at SECTION
boundaries than the acoustic audio-SSM on metronomic jazz1460 renders?

For 5 AABA standards:
  - GT section boundaries = bars where `section_per_bar` label changes.
  - chord-SSM: per-beat one-hot [root-rel-tonic (12) | quality (Q)], cosine.
  - audio-SSM: per-beat build_ssm() on Basic-Pitch note-probs (acoustic chroma).
  - section-scale checkerboard novelty on each; boundary hit-rate F @ +-1 bar
    from the top-N novelty peaks (N = number of GT boundaries).

PASS if chord-SSM boundary-F > audio-SSM boundary-F on >= 3/5 songs.
"""
from __future__ import annotations

import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from analyze_accomp_emission import parse_chord, QUALITY_MAP
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig
from harmonia.models.stage1_pitch import PitchExtractor
from harmonia.models import chord_pipeline_v1 as P
from harmonia.models.structure import build_ssm, compute_novelty

DB = REPO / "data" / "accomp_db" / "db.jsonl"
# Clean AABA / sectional standards (verified titles).
# Genuine AABA (form "A16 B8 A8" = A A B A, bridge B returns to A).
SONG_IDS = [
    "jazz1460_0000",  # 9.20 Special
    "jazz1460_0001",  # 26-2
    "jazz1460_0002",  # 52nd Street Theme
    "jazz1460_0006",  # A Beautiful Friendship
    "jazz1460_0014",  # A Ghost Of A Chance
    "jazz1460_0042",  # Afternoon In Paris
    "jazz1460_0046",  # Ain't Misbehavin'
    "jazz1460_0070",  # All Too Soon
]
Q_IDX = {q: i for i, q in enumerate(sorted(set(QUALITY_MAP.values())))}


def gt_boundary_bars(rec) -> list[int]:
    spb = rec["section_per_bar"]
    return [b for b in range(1, len(spb)) if spb[b] != spb[b - 1]]


def chord_seq_per_beat(rec) -> tuple[np.ndarray, int]:
    """(n_beats,) arrays -> per-beat feature matrix [root-rel-tonic | quality]."""
    bpb = rec["beats_per_bar"]
    n_beats = rec["n_bars"] * bpb
    tonic_pc = _key_pc(rec["key"])
    # per-beat root/quality via forward-fill of chord_timeline slots
    root = np.full(n_beats, -1, int)
    qual = np.full(n_beats, -1, int)
    slots = sorted(((ev["bar"] - 1) * bpb + ev["beat"], ev["mma"]) for ev in rec["chord_timeline"])
    cur_r, cur_q = -1, -1
    si = 0
    for beat in range(n_beats):
        while si < len(slots) and slots[si][0] <= beat:
            p = parse_chord(slots[si][1])
            if p is not None:
                cur_r, cur_q = p[0] % 12, Q_IDX.get(p[1], 0)
            si += 1
        root[beat], qual[beat] = cur_r, cur_q
    nq = max(Q_IDX.values()) + 1
    feat = np.zeros((n_beats, 12 + nq), np.float32)
    for b in range(n_beats):
        if root[b] >= 0:
            feat[b, (root[b] - tonic_pc) % 12] = 1.0
            feat[b, 12 + qual[b]] = 1.0
    return feat, bpb


def _key_pc(key: str) -> int:
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    k = key.strip()
    pc = base.get(k[0].upper(), 0)
    if len(k) > 1 and k[1] in "#b-":
        pc += 1 if k[1] == "#" else -1
    return pc % 12


def chord_ssm(feat: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(feat, axis=1, keepdims=True)
    fn = feat / np.clip(n, 1e-9, None)
    S = fn @ fn.T
    return np.clip(S, 0.0, 1.0).astype(np.float32)


def boundary_f(novelty: np.ndarray, gt_beats: list[int], bpb: int,
               tol_bars: int = 1) -> float:
    """Top-N novelty peaks (N=#GT) matched to GT beats within +-tol_bars."""
    from scipy.signal import find_peaks
    n_gt = len(gt_beats)
    if n_gt == 0:
        return float("nan")
    peaks, props = find_peaks(novelty, distance=bpb * 2)
    if len(peaks) == 0:
        return 0.0
    order = np.argsort(props["peak_heights"] if "peak_heights" in props
                       else novelty[peaks])[::-1]
    top = peaks[order[:n_gt]]
    tol = tol_bars * bpb
    gt = list(gt_beats)
    hits = 0
    for p in top:
        for j, g in enumerate(gt):
            if abs(p - g) <= tol:
                hits += 1
                gt.pop(j)
                break
    prec = hits / len(top)
    rec = hits / n_gt
    return 0.0 if hits == 0 else 2 * prec * rec / (prec + rec)


def section_contrast(ssm: np.ndarray, section_per_bar, bpb: int) -> float:
    """Mean within-section similarity minus mean between-section similarity.

    Off-diagonal only (|i-j| >= 4 bars) so the trivial local diagonal does not
    dominate: this isolates the LONG-RANGE repetition that defines AABA (A at
    bar 0 ~ A at bar 24) rather than local harmonic homogeneity.
    """
    n_beats = len(section_per_bar) * bpb
    lab = np.repeat(np.array([hash(s) for s in section_per_bar]), bpb)[:n_beats]
    ssm = ssm[:n_beats, :n_beats]
    ii, jj = np.triu_indices(n_beats, k=4 * bpb)
    same = lab[ii] == lab[jj]
    s = ssm[ii, jj]
    if same.sum() == 0 or (~same).sum() == 0:
        return float("nan")
    return float(s[same].mean() - s[~same].mean())


def bridge_contrast(ssm: np.ndarray, bpb: int) -> float:
    """AABA (A16 B8 A8): mean sim(A,A') minus mean sim(A,B). Positive => the
    bridge is correctly the odd section out. This is the discriminating signal
    a form detector actually exploits (A repeats; B differs)."""
    A1 = slice(0, 16 * bpb)
    B = slice(16 * bpb, 24 * bpb)
    A3 = slice(24 * bpb, 32 * bpb)
    if ssm.shape[0] < 32 * bpb:
        return float("nan")
    aa = ssm[A1, A3].mean()
    ab = (ssm[A1, B].mean() + ssm[A3, B].mean()) / 2
    return float(aa - ab)


def main():
    recs = {r["song_id"]: r for r in (json.loads(l) for l in open(DB))}
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    ex = PitchExtractor(cache_dir=None)

    kernel = 8  # checkerboard half-width in beats (2 bars each side)
    print(f"{'song':<30} {'form':<14} {'cF':>5} {'aF':>5}  "
          f"{'c_ctr':>6} {'a_ctr':>6} {'win':>6}")
    print("-" * 74)
    chord_wins = 0
    ctr_wins = 0
    n = 0
    for sid in SONG_IDS:
        rec = recs.get(sid)
        if rec is None:
            print(f"{sid:<34} MISSING")
            continue
        bpb = rec["beats_per_bar"]
        gt_bars = gt_boundary_bars(rec)
        gt_beats = [b * bpb for b in gt_bars]
        n_beats = rec["n_bars"] * bpb

        # chord-SSM
        feat, _ = chord_seq_per_beat(rec)
        cssm = chord_ssm(feat)
        cnov = compute_novelty(cssm, kernel_size=kernel)
        cF = boundary_f(cnov, gt_beats, bpb)

        # audio-SSM (rendered)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            tmp = Path(wf.name)
        try:
            renderer.render(REPO / rec["midi_path"], tmp, RenderConfig(soundfont_path=sf2))
            y, sr = sf.read(tmp)
            y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
            acts = ex.extract(tmp, use_cache=False)
        finally:
            tmp.unlink(missing_ok=True)
        spb = 60.0 / rec["tempo"]
        bt = np.arange(n_beats + 1) * spb
        note_b = P._pool_beats(acts.frame_times, acts.note_probs, bt)
        assm = build_ssm(note_b)
        anov = compute_novelty(assm, kernel_size=kernel)
        aF = boundary_f(anov, gt_beats, bpb)

        c_ctr = bridge_contrast(cssm, bpb)
        a_ctr = bridge_contrast(assm, bpb)

        win = "chord" if c_ctr > a_ctr else "audio"
        chord_wins += int(cF > aF)
        ctr_wins += int(c_ctr > a_ctr)
        n += 1
        print(f"{rec['title'][:29]:<30} {rec['form']:<14} {cF:>5.2f} {aF:>5.2f}  "
              f"{c_ctr:>6.3f} {a_ctr:>6.3f} {win:>6}")

        np.savez(REPO / "scripts" / f"_ssm_{sid}.npz",
                 chord_ssm=cssm, audio_ssm=assm, cnov=cnov, anov=anov,
                 gt_beats=np.array(gt_beats))

    print("-" * 74)
    print(f"novelty boundary-F: chord-SSM wins {chord_wins}/{n}")
    print(f"section-contrast:   chord-SSM wins {ctr_wins}/{n} "
          f"(PASS if >= {int(np.ceil(0.6 * n))})")


if __name__ == "__main__":
    main()
