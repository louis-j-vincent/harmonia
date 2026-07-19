"""Build a synthetic BP48 training corpus matching harmonia corpus_schema.
Renders each song, extracts features via the real pipeline, DELETES the wav,
then writes one .npz with all REQUIRED_KEYS."""
from __future__ import annotations
import sys, argparse, os
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from scratchpad.synth_gen import (gen_progression, build_midi, render_wav, add_melody,
                                  add_noise_wav, CHORD_PROGRAMS, BASS_PROGRAMS)
from harmonia.models.stage1_pitch import PitchExtractor
from harmonia.data.yt_chord_corpus import (seg_feature_clipped, seg_feature_abs_clipped,
                                           QUALITY_IDX, QUALITIES)
from scripts.build_jaah_corpus import parse_jaah as parse_harte
from harmonia.data.corpus_schema import save_corpus

SF = {
    "musescore": "/Users/vincente/harmonia/data/soundfonts/MuseScore_General.sf2",
    "generaluser": "/Users/vincente/harmonia/data/soundfonts/GeneralUser.sf2",
}
OUT = REPO / "data" / "cache" / "synth"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", type=int, default=50)
    ap.add_argument("--chords", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(OUT / "synth_bp48.npz"))
    ap.add_argument("--rich", action="store_true",
                    help="add melody layer + broadband noise for realism")
    args = ap.parse_args()

    ex = PitchExtractor(cache_dir=OUT / "bp_cache")
    cols = {k: [] for k in ("feat48", "feat48_abs", "root", "quality_idx",
                            "quality", "labels", "match", "t0", "t1", "song_id")}
    rng = np.random.RandomState(args.seed)
    tmpwav = str(OUT / "_tmp.wav")
    for s in range(args.songs):
        tempo = int(rng.choice([72, 82, 92, 104, 116, 128, 140, 152]))
        cp = int(rng.choice(CHORD_PROGRAMS)); bp = int(rng.choice(BASS_PROGRAMS))
        sfname = list(SF)[rng.randint(len(SF))]
        prog = gen_progression(rng, args.chords)
        pm, ann = build_midi(prog, tempo, rng, cp, bp)
        if args.rich:
            add_melody(pm, prog, tempo, rng)
        render_wav(pm, SF[sfname], tmpwav)
        if args.rich:
            add_noise_wav(tmpwav, snr_db=float(rng.uniform(12, 20)), rng=rng)
        acts = ex.extract(Path(tmpwav))
        ft, on, nt = acts.frame_times, acts.onset_probs, acts.note_probs
        sid = f"synth_{s:03d}"
        n = 0
        for t0, t1, lab in ann:
            root, fam, _ = parse_harte(lab)
            if root is None:
                continue
            fr = seg_feature_clipped(ft, on, nt, t0, t1, root)
            fa = seg_feature_abs_clipped(ft, on, nt, t0, t1)
            if fr is None or fa is None:
                continue
            cols["feat48"].append(fr); cols["feat48_abs"].append(fa)
            cols["root"].append(int(root % 12)); cols["quality"].append(fam)
            cols["quality_idx"].append(QUALITY_IDX[fam]); cols["labels"].append(lab)
            cols["match"].append("exact"); cols["t0"].append(float(t0))
            cols["t1"].append(float(t1)); cols["song_id"].append(sid)
            n += 1
        # clear per-song bp cache + wav to bound disk
        os.path.exists(tmpwav) and os.unlink(tmpwav)
        for f in (OUT / "bp_cache").glob("*"):
            f.unlink(missing_ok=True)
        if s % 10 == 0 or s == args.songs - 1:
            print(f"song {s:03d}: {n} chords (total {len(cols['root'])})", flush=True)

    arrays = {
        "feat48": np.array(cols["feat48"], np.float32),
        "feat48_abs": np.array(cols["feat48_abs"], np.float32),
        "root": np.array(cols["root"], np.int32),
        "quality_idx": np.array(cols["quality_idx"], np.int32),
        "quality": np.array(cols["quality"]),
        "labels": np.array(cols["labels"]),
        "match": np.array(cols["match"]),
        "t0": np.array(cols["t0"], np.float64),
        "t1": np.array(cols["t1"], np.float64),
        "song_id": np.array(cols["song_id"]),
        "qualities": np.array(QUALITIES),
    }
    save_corpus(args.out, **arrays)
    print(f"\nsaved {args.out}  N={len(arrays['root'])}")
    inv = np.array(["/" in l for l in cols["labels"]])
    print(f"inversion frac = {inv.mean():.3f}")
    import collections
    print("quality dist:", dict(collections.Counter(cols["quality"])))


if __name__ == "__main__":
    main()
