"""Premise check: render a small diverse synthetic batch, extract BP48 features
with the SAME pipeline as real RWC, compare chroma entropy vs real audio."""
from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from scratchpad.synth_gen import gen_progression, build_midi, render_wav, CHORD_PROGRAMS, BASS_PROGRAMS
from harmonia.models.stage1_pitch import PitchExtractor
from harmonia.data.yt_chord_corpus import seg_feature_clipped, seg_feature_abs_clipped, QUALITY_IDX
from scripts.build_jaah_corpus import parse_jaah as parse_harte

SF = {
    "musescore": "/Users/vincente/harmonia/data/soundfonts/MuseScore_General.sf2",
    "generaluser": "/Users/vincente/harmonia/data/soundfonts/GeneralUser.sf2",
}
OUT = REPO / "scratchpad" / "synth_premise"
OUT.mkdir(exist_ok=True)


def block_ent(f, sl):
    p = np.abs(f[:, sl]); p = p / (p.sum(1, keepdims=True) + 1e-12)
    return (-(p * np.log(p + 1e-12)).sum(1) / np.log(12))


def full_ent(f):
    p = np.abs(f); p = p / (p.sum(1, keepdims=True) + 1e-12)
    return (-(p * np.log(p + 1e-12)).sum(1) / np.log(48))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", type=int, default=6)
    ap.add_argument("--chords", type=int, default=16)
    args = ap.parse_args()

    ex = PitchExtractor(cache_dir=OUT / "bp_cache")
    all_abs, all_rel, all_qidx, labels = [], [], [], []
    rng = np.random.RandomState(20260717)
    for s in range(args.songs):
        tempo = int(rng.choice([76, 90, 104, 120, 132, 148]))
        cp = int(rng.choice(CHORD_PROGRAMS)); bp = int(rng.choice(BASS_PROGRAMS))
        sfname = list(SF)[s % len(SF)]
        prog = gen_progression(rng, args.chords)
        pm, ann = build_midi(prog, tempo, rng, cp, bp)
        wav = str(OUT / f"song_{s:02d}.wav")
        render_wav(pm, SF[sfname], wav)
        acts = ex.extract(Path(wav))
        ft, on, nt = acts.frame_times, acts.onset_probs, acts.note_probs
        nrec = 0
        for t0, t1, lab in ann:
            root, fam, _ = parse_harte(lab)
            if root is None:
                continue
            fr = seg_feature_clipped(ft, on, nt, t0, t1, root)
            fa = seg_feature_abs_clipped(ft, on, nt, t0, t1)
            if fr is None or fa is None:
                continue
            all_rel.append(fr); all_abs.append(fa)
            all_qidx.append(QUALITY_IDX[fam]); labels.append(lab)
            nrec += 1
        print(f"song {s:02d}: tempo={tempo} sf={sfname} chordprog={cp} bassprog={bp} -> {nrec} chords, {len(ft)} frames", flush=True)

    A = np.array(all_abs); R = np.array(all_rel)
    print(f"\nTotal synthetic chords: {len(A)}")
    print("\n=== SYNTHETIC chroma norm-entropy (same computation as real) ===")
    for name, sl in [("onset", slice(0, 12)), ("note", slice(12, 24)),
                     ("bass", slice(24, 36)), ("treble", slice(36, 48))]:
        e = block_ent(A, sl)
        print(f"  {name:7s} mean={e.mean():.4f} std={e.std():.4f}")
    fe = full_ent(A)
    print(f"  full-48 mean={fe.mean():.4f} std={fe.std():.4f}")
    print("\n=== REAL RWC baseline (for reference) ===")
    print("  onset 0.885  note 0.999  bass 0.830  treble 0.814  full-48 0.931")

    np.savez(OUT / "premise_feats.npz", feat48_abs=A, feat48=R,
             quality_idx=np.array(all_qidx), labels=np.array(labels))
    print(f"\nsaved {OUT/'premise_feats.npz'}")


if __name__ == "__main__":
    main()
