"""Diagnose why the issue-#20 diatonic prior hurts POP909 majmin.

Renders each POP909 song once, then (a) sweeps (boost, thresh) configs and
(b) tallies per-segment fire outcomes in the majmin sense: does an override flip
a correct call to wrong, or a wrong call to correct?  This tells us whether the
mechanism is sound but the inferred local key is unreliable, or vice versa.
"""
from __future__ import annotations

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

import librosa
import mir_eval

from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig
from harmonia.data.pop909_parser import POP909Parser
from harmonia.models.stage1_pitch import PitchExtractor
from harmonia.models import chord_pipeline_v1 as P
from harmonia.theory.key_profiles import infer_key
from eval_diatonic_prior import _Q_HARTE, tempo_grid, gmerge_segs, NOTE

POP909_DIR = REPO / "data" / "pop909" / "POP909"


def majmin_root(lab):
    """Reduce a Harte label to (root_pc, 'maj'/'min'/'X') for a coarse compare."""
    try:
        r, semis, bass = mir_eval.chord.encode(lab)
    except Exception:
        return None
    return r


def prep(song_ids, renderer, sf2, ex, v4, fam):
    songs = []
    parser = POP909Parser(POP909_DIR)
    for sid in song_ids:
        song = parser.parse_song(sid)
        if song is None:
            continue
        spans = []
        for ev in song.chord_events:
            if ev.root == -1 or ev.end_beat <= ev.start_beat:
                continue
            h = _Q_HARTE.get(ev.quality)
            if h is None:
                continue
            spans.append((ev.start_beat, ev.end_beat, ev.root, h))
        if not spans:
            continue
        ref_int = np.array([[t0, t1] for t0, t1, _, _ in spans])
        ref_lab = [f"{NOTE[r]}:{h}" for _, _, r, h in spans]
        midi = POP909_DIR / sid / f"{sid}.mid"
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            tmp = Path(wf.name)
        try:
            renderer.render(midi, tmp, RenderConfig(soundfont_path=sf2))
            y, sr = sf.read(tmp); y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
            acts = ex.extract(tmp, use_cache=False)
        finally:
            tmp.unlink(missing_ok=True)
        bt = tempo_grid(y, sr)
        onset_b = P._pool_beats(acts.frame_times, acts.onset_probs, bt)
        note_b = P._pool_beats(acts.frame_times, acts.note_probs, bt)
        beat_proba = v4.predict_proba(onset_b, note_b)
        segs = gmerge_segs(beat_proba)
        songs.append((sid, bt, onset_b, note_b, beat_proba, segs, ref_int, ref_lab))
        print(f"  prepped {sid}: {len(segs)} segs", flush=True)
    return songs


def score(songs, fam, boost, thresh, tally=False):
    ms, rs = [], []
    fires = corr_to_wrong = wrong_to_corr = wrong_to_wrong = keybad = 0
    for sid, bt, onset_b, note_b, beat_proba, segs, ref_int, ref_lab in songs:
        n_beats = len(onset_b)
        labeled = []
        for s, e in segs:
            root = int(beat_proba[s:e].sum(0).argmax())
            seg_on = onset_b[s:e].sum(0); seg_nt = note_b[s:e].sum(0)
            seg_bs = P._reg_raw(seg_on, 0, 52); seg_tr = P._reg_raw(seg_on, 60, 200)
            _, sev0, conf = fam.predict(root, seg_on, seg_nt, seg_bs, seg_tr, 0.0)
            if (e - s) < 8:
                c = (s + e) // 2; lo, hi = max(0, c - 16), min(n_beats, c + 16)
            else:
                lo, hi = s, e
            kp = infer_key(P._reg_raw(onset_b[lo:hi].sum(0)))
            sev1 = P.apply_diatonic_prior(root, sev0, conf, kp.tonic, kp.mode,
                                          kp.confidence, diatonic_boost=boost,
                                          threshold_chromatic=thresh)
            if tally and sev1 != sev0:
                fires += 1
                # GT majmin at segment midpoint
                mid = (bt[s] + bt[min(e, len(bt) - 1)]) / 2
                gi = np.searchsorted(ref_int[:, 0], mid, "right") - 1
                gt = ref_lab[gi] if 0 <= gi < len(ref_lab) else None
                if gt is not None:
                    g_third = _third(gt); a0 = _third(f"{NOTE[root]}:{sev0}")
                    a1 = _third(f"{NOTE[root]}:{sev1}")
                    ok0, ok1 = (a0 == g_third), (a1 == g_third)
                    if ok0 and not ok1: corr_to_wrong += 1
                    elif not ok0 and ok1: wrong_to_corr += 1
                    elif not ok0 and not ok1: wrong_to_wrong += 1
            lab = f"{NOTE[root]}:{sev1}"
            if labeled and labeled[-1][2] == lab:
                labeled[-1][1] = e
            else:
                labeled.append([s, e, lab])
        est_int = np.array([[bt[s], bt[min(e, len(bt) - 1)]] for s, e, _ in labeled])
        est_lab = [lab for _, _, lab in labeled]
        sc = mir_eval.chord.evaluate(ref_int, ref_lab, est_int, est_lab)
        ms.append(sc["majmin"]); rs.append(sc["root"])
    out = (np.mean(rs), np.mean(ms))
    if tally:
        return out, (fires, wrong_to_corr, corr_to_wrong, wrong_to_wrong)
    return out


def _third(lab):
    """Return 'maj'/'min'/'X' from the third of a Harte chord."""
    try:
        _, semis, _ = mir_eval.chord.encode(lab)
    except Exception:
        return "X"
    if semis[4] and not semis[3]:
        return "maj"
    if semis[3] and not semis[4]:
        return "min"
    return "X"


def main():
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    ex = PitchExtractor(cache_dir=None)
    v4 = P._get_beat_seq(); fam = P._get_family_clf()
    ids = [f"{i:03d}" for i in range(1, 6)]
    print("prepping POP909 renders...", flush=True)
    songs = prep(ids, renderer, sf2, ex, v4, fam)

    r0, m0 = score(songs, fam, boost=0.0, thresh=0.0)  # boost 0 ~ never flips
    # baseline with prior fully off: use thresh=-1 so conf>=thresh always → skip
    (rb, mb) = score(songs, fam, 4.0, -1.0)
    print(f"\nbaseline (prior off): root={rb:.1%} majmin={mb:.1%}")

    print("\n=== (boost, thresh) sweep — POP909 majmin ===")
    print(f"{'boost':>6} {'thresh':>7} {'root':>7} {'majmin':>7}  {'Δmajmin':>8}")
    for boost in (2.0, 4.0, 8.0):
        for thresh in (0.5, 0.65, 0.8, 0.9, 0.95):
            r, m = score(songs, fam, boost, thresh)
            print(f"{boost:>6.1f} {thresh:>7.2f} {r:>7.1%} {m:>7.1%}  {m-mb:>+8.2%}")

    print("\n=== fire tally at boost=4 thresh=0.65 (default) ===")
    _, (f, w2c, c2w, w2w) = score(songs, fam, 4.0, 0.65, tally=True)
    print(f"fires={f}  wrong→correct={w2c}  correct→wrong={c2w}  wrong→wrong={w2w}")
    print("\n=== fire tally at boost=4 thresh=0.90 ===")
    _, (f, w2c, c2w, w2w) = score(songs, fam, 4.0, 0.90, tally=True)
    print(f"fires={f}  wrong→correct={w2c}  correct→wrong={c2w}  wrong→wrong={w2w}")


if __name__ == "__main__":
    main()
