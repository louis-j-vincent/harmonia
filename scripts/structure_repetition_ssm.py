"""Structure detection the right way (step 1): AABA lives in REPETITION (A returns
as a run of similar bars = a diagonal stripe offset by the section length), not in
local contrast. Plain Foote novelty scored only F=0.25 on symbolic chords because
jazz ii-V churn creates stronger local novelty than the sections do.

Fix: diagonally ENHANCE the SSM (average along diagonals) so repeated *sequences*
light up, then detect boundaries. Validate on symbolic chords first (cheap, no
audio), then on audio chroma. GT = section_per_bar boundaries.

Usage: .venv/bin/python scripts/structure_repetition_ssm.py [--audio] [--n-songs 12] [--plot]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from analyze_accomp_emission import parse_chord  # noqa: E402
from build_audio_chord_features import BUCKET_FAMILY  # noqa: E402

DB = REPO / "data" / "accomp_db" / "db.jsonl"
NOTE_TO_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
FAM_TONES = {  # semitone offsets from root defining the chord's pitch-class set
    "major": [0, 4, 7], "minor": [0, 3, 7], "diminished": [0, 3, 6],
    "augmented": [0, 4, 8], "suspended": [0, 5, 7]}


def symbolic_bar_feats(rec):
    nb, bpb, spb = rec["n_bars"], rec["beats_per_bar"], 60.0 / rec["tempo"]
    tl = sorted(rec["chord_timeline"], key=lambda e: e["time"])
    F = np.zeros((nb, 12))
    for b in range(nb):
        t = b * bpb * spb
        cur = None
        for ev in tl:
            if ev["time"] <= t + 1e-6:
                s = ev["ireal"] or ev["mma"]
                if s and s[0] in NOTE_TO_PC:
                    pc = NOTE_TO_PC[s[0]]
                    if len(s) > 1 and s[1] == "#": pc = (pc + 1) % 12
                    elif len(s) > 1 and s[1] == "b": pc = (pc - 1) % 12
                    p = parse_chord(s)
                    fam = BUCKET_FAMILY.get(p[1], "major") if p else "major"
                    cur = (pc, fam)
            else:
                break
        if cur:
            for off in FAM_TONES[cur[1]]:
                F[b, (cur[0] + off) % 12] = 1
    n = np.linalg.norm(F, axis=1, keepdims=True)
    return F / (n + 1e-9)


def enhance(S, K=3):
    """Average the SSM along diagonals so repeated *sequences* (not single bars) win."""
    n = len(S); out = np.zeros_like(S); cnt = np.zeros_like(S)
    for k in range(-K, K + 1):
        sh = np.zeros_like(S)
        if k >= 0:
            sh[:n - k, :n - k] = S[k:, k:] if False else 0  # placeholder
        # shift both indices by k (move along the main diagonal)
        i0 = max(0, -k); j0 = max(0, -k); L = n - abs(k)
        idx = np.arange(L)
        out[i0 + idx[:, None] * 0, :]  # no-op to keep shape clear
        for d in range(L):
            out[i0 + d] += np.roll(S[i0 + d], -k) * 0  # avoid; handled below
    # simpler explicit diagonal averaging
    out = np.zeros_like(S); cnt = np.zeros_like(S)
    for k in range(-K, K + 1):
        Sk = np.full_like(S, np.nan)
        i = np.arange(n)
        # element (a,b) gets S[a+k, b+k] when in range
        a = i[:, None]; b = i[None, :]
        va, vb = a + k, b + k
        m = (va >= 0) & (va < n) & (vb >= 0) & (vb < n)
        out[m] += S[np.clip(va, 0, n - 1)[m], np.clip(vb, 0, n - 1)[m]]
        cnt[m] += 1
    return out / (cnt + 1e-9)


def _checker(L):
    x = np.linspace(-2, 2, 2 * L)
    g = np.outer(np.exp(-0.5 * x ** 2), np.exp(-0.5 * x ** 2))
    k = np.ones((2 * L, 2 * L)); k[:L, L:] = k[L:, :L] = -1
    return k * g


def novelty(S, L=4):
    n = len(S); k = _checker(L); P = np.pad(S, L)
    return np.array([(k * P[i:i + 2 * L, i:i + 2 * L]).sum() for i in range(n)])


def boundary_f(nov, gt, tol=1):
    k = len(gt)
    order = [i for i in np.argsort(-nov) if 0 < i < len(nov)]
    est = []
    for i in order:
        if all(abs(i - c) > 1 for c in est):
            est.append(i)
        if len(est) >= k:
            break
    hits = sum(any(abs(e - g) <= tol for e in est) for g in gt)
    p = hits / (len(est) + 1e-9); r = hits / (k + 1e-9)
    return 2 * p * r / (p + r + 1e-9)


def audio_bar_feats(rec):
    import librosa
    import soundfile as sf
    from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
        tmp = Path(wf.name)
    try:
        renderer.render(REPO / rec["midi_path"], tmp, RenderConfig(soundfont_path=sf2))
        y, sr = sf.read(tmp); y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
    finally:
        pass
    tmp.unlink(missing_ok=True)
    nb, bpb, spb = rec["n_bars"], rec["beats_per_bar"], 60.0 / rec["tempo"]
    ch = librosa.feature.chroma_cqt(y=y, sr=sr)
    ct = librosa.frames_to_time(np.arange(ch.shape[1]), sr=sr)
    F = np.zeros((nb, 12))
    for b in range(nb):
        m = (ct >= b * bpb * spb) & (ct < (b + 1) * bpb * spb)
        if m.any():
            F[b] = ch[:, m].mean(1)
    n = np.linalg.norm(F, axis=1, keepdims=True)
    return F / (n + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", action="store_true")
    ap.add_argument("--n-songs", type=int, default=12)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(DB)]
    songs = [r for r in recs if r["corpus"] == "jazz1460" and r["beats_per_bar"] == 4
             and (REPO / r["midi_path"]).exists() and len(set(r["section_per_bar"])) > 1]
    songs = songs[:: max(len(songs) // args.n_songs, 1)][: args.n_songs]

    raw_fs, enh_fs = [], []
    for rec in songs:
        sv = rec["section_per_bar"]
        gt = [b for b in range(1, len(sv)) if sv[b] != sv[b - 1]]
        F = audio_bar_feats(rec) if args.audio else symbolic_bar_feats(rec)
        S = F @ F.T
        raw_fs.append(boundary_f(novelty(S), gt))
        enh_fs.append(boundary_f(novelty(enhance(S, K=3)), gt))

    src = "AUDIO chroma" if args.audio else "SYMBOLIC chords"
    print(f"\n=== structure via SSM, {len(songs)} songs, {src} ===")
    print(f"  raw-SSM novelty      boundary F: {np.mean(raw_fs):.3f}")
    print(f"  diag-enhanced novelty boundary F: {np.mean(enh_fs):.3f}")
    print("\n  (enhanced >> raw => repetition, not contrast, is the right structure cue.)")

    if args.plot:
        import matplotlib.pyplot as plt
        rec = songs[0]; sv = rec["section_per_bar"]
        gt = [b for b in range(1, len(sv)) if sv[b] != sv[b - 1]]
        F = symbolic_bar_feats(rec); S = F @ F.T; E = enhance(S, K=3)
        fig, ax = plt.subplots(1, 2, figsize=(11, 5))
        for a, M, t in [(ax[0], S, "raw SSM"), (ax[1], E, "diag-enhanced SSM")]:
            a.imshow(M, origin="lower", cmap="magma"); a.set_title(f"{t}  ({rec['form']})")
            for g in gt:
                a.axhline(g - .5, color="c", lw=.8); a.axvline(g - .5, color="c", lw=.8)
        out = REPO / "docs" / "plots" / "structure_repetition_ssm.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(out, dpi=110); print(f"  plot → {out}")


if __name__ == "__main__":
    main()
