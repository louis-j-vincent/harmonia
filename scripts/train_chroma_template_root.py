"""Train a generative chroma-template root model.

For every beat in the iReal corpus:
  1. Extract 12-dim onset chroma.
  2. Roll by -GT_chord_root so root lands at position 0.
  3. Accumulate → fit mean μ and diagonal σ².

Inference: given observed chroma c, score each root candidate r as
  log N(roll(c, -r) | μ, σ²)   (diagonal Gaussian)
and pick argmax.

This is a purely generative model — no discriminative features, no context.
It captures the consistent pattern seen in the key-normalised heatmap
(root > 4th ≈ 5th >> chromatic tones).

Saves: harmonia/models/chroma_root_template.npz  {mu, sigma, n_beats}

Usage:
    .venv/bin/python scripts/train_chroma_template_root.py [--n-songs 200]
    .venv/bin/python scripts/train_chroma_template_root.py --eval
"""
from __future__ import annotations
import argparse, json, sys, tempfile
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from analyze_accomp_emission import song_chord_spans
from build_audio_chord_features import BUCKET_FAMILY
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig
from harmonia.models.chord_pipeline_v1 import _pool_beats, _reg_raw
from harmonia.models.stage1_pitch import PitchExtractor
from harmonic_rhythm_probe import pool_beats
import soundfile as sf

DB  = REPO / "data" / "accomp_db" / "db.jsonl"
OUT = REPO / "harmonia" / "models" / "chroma_root_template.npz"
NOTE = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]


def beat_chroma(onset_b: np.ndarray) -> np.ndarray:
    """12-dim L2-normed chroma from 88-dim onset piano roll."""
    ch = np.zeros(12)
    for k in range(88):
        ch[(21 + k) % 12] += onset_b[k]
    n = np.linalg.norm(ch)
    return ch / n if n > 1e-9 else ch


def collect_song(rec, renderer, sf2, ex) -> tuple[np.ndarray, np.ndarray]:
    """Return (chromas, roots) arrays for all valid beats in one song."""
    spb   = 60.0 / rec["tempo"]
    bpb   = rec["beats_per_bar"]
    n_beats = rec["n_bars"] * bpb

    spans = [(t0, t1, r % 12)
             for t0, t1, r, q in song_chord_spans(rec)
             if t1 > t0 and q in BUCKET_FAMILY]

    def gt_root(t):
        for t0, t1, r in spans:
            if t0 <= t < t1:
                return r
        return None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
        tmp = Path(wf.name)
    try:
        renderer.render(REPO / rec["midi_path"], tmp, RenderConfig(soundfont_path=sf2))
        acts = ex.extract(tmp, use_cache=False)
    finally:
        tmp.unlink(missing_ok=True)

    bt      = np.arange(n_beats + 1) * spb
    onset_b = _pool_beats(acts.frame_times, acts.onset_probs, bt)

    chromas, roots = [], []
    for b in range(n_beats):
        r = gt_root((b + 0.5) * spb)
        if r is None:
            continue
        ch = beat_chroma(onset_b[b])
        if ch.sum() < 1e-9:
            continue
        chromas.append(ch)
        roots.append(r)

    if not chromas:
        return np.empty((0, 12)), np.empty(0, int)
    return np.array(chromas), np.array(roots)


def fit_template(chromas: np.ndarray, roots: np.ndarray):
    """Roll each chroma by -root and fit mean + std."""
    normalised = np.array([np.roll(ch, -r) for ch, r in zip(chromas, roots)])
    mu    = normalised.mean(0)
    sigma = normalised.std(0) + 1e-6   # diagonal, add floor
    return mu, sigma, normalised


def score_roots(ch: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Log-likelihood of each root under diagonal Gaussian. Returns (12,)."""
    scores = np.zeros(12)
    for r in range(12):
        ch_r = np.roll(ch, -r)
        scores[r] = -0.5 * np.sum(((ch_r - mu) / sigma) ** 2)
    return scores


def eval_on_pop909(mu, sigma):
    """Run template root model on POP909 and report per-root accuracy."""
    from harmonia.data.pop909_parser import POP909Parser
    import librosa

    parser = POP909Parser(REPO / "data" / "pop909" / "POP909")
    ex     = PitchExtractor(cache_dir=REPO / "data" / "cache")

    all_gt, all_pred = [], []
    for sid in ["001", "002", "003", "004", "005"]:
        wav = REPO / "data" / "renders" / "pop909" / sid / f"{sid}_v005_musescoregeneral.wav"
        if not wav.exists():
            continue
        y, sr = sf.read(wav)
        y = (y.mean(1) if y.ndim > 1 else y).astype("float32")

        tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr, units="time")
        tempo = float(np.atleast_1d(tempo_arr)[0]); period = 60 / max(tempo, 1)
        _, rf = librosa.beat.beat_track(y=y, sr=sr)
        rt = librosa.frames_to_time(rf, sr=sr)
        if len(rt):
            ph = (rt % period) / period
            p0 = float(np.arctan2(np.sin(2*np.pi*ph).mean(), np.cos(2*np.pi*ph).mean())
                       / (2*np.pi) * period) % period
        else:
            p0 = 0.0
        dur = librosa.get_duration(y=y, sr=sr)
        bt  = np.arange(0, dur + period, period) + p0
        bt  = bt[bt < dur + 0.5 * period]

        acts    = ex.extract(wav)
        onset_b = _pool_beats(acts.frame_times, acts.onset_probs, bt)
        song    = parser.parse_song(sid)

        for b in range(min(len(bt) - 1, len(onset_b))):
            t  = 0.5 * (bt[b] + bt[b + 1])
            ev = song.chord_at_time(t)
            if ev is None:
                continue
            rs = ev.label.split(":")[0].split("/")[0]
            if rs == "N" or rs not in NOTE:
                continue
            gt = NOTE.index(rs)
            ch = beat_chroma(onset_b[b])
            pred = int(score_roots(ch, mu, sigma).argmax())
            all_gt.append(gt); all_pred.append(pred)

    gt   = np.array(all_gt);  pred = np.array(all_pred)
    acc  = (gt == pred).mean()
    print(f"\nPOP909 template model: {acc:.1%}  (N={len(gt)} beats)")
    print(f"{'Root':>4}  {'acc':>6}  n")
    for r in range(12):
        m = gt == r
        if m.sum() < 5: continue
        print(f"{NOTE[r]:>4}  {(pred[m]==r).mean():6.1%}  n={m.sum()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-songs", type=int, default=200)
    ap.add_argument("--eval",    action="store_true")
    args = ap.parse_args()

    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2      = renderer._find_soundfont("MuseScore_General.sf2")
    ex       = PitchExtractor(cache_dir=None)

    recs  = [json.loads(l) for l in open(DB)]
    songs = [r for r in recs if r.get("corpus") == "jazz1460"
             and r["beats_per_bar"] == 4
             and (REPO / r["midi_path"]).exists()]
    step  = max(1, len(songs) // args.n_songs)
    songs = songs[::step][: args.n_songs]
    print(f"Collecting {len(songs)} songs …")

    all_ch, all_rt = [], []
    for i, rec in enumerate(songs):
        print(f"  [{i+1}/{len(songs)}] {rec['song_id']}", end="\r", flush=True)
        try:
            ch, rt = collect_song(rec, renderer, sf2, ex)
        except Exception as e:
            continue
        if len(ch):
            all_ch.append(ch); all_rt.append(rt)

    chromas = np.vstack(all_ch)
    roots   = np.concatenate(all_rt)
    print(f"\nCollected {len(chromas)} beat samples from {len(all_ch)} songs")

    mu, sigma, normalised = fit_template(chromas, roots)

    print("\nLearned chroma template (root at position 0):")
    positions = ["root","m2","M2","m3","M3","P4","TT","P5","m6","M6","m7","M7"]
    for i, (m, s, pos) in enumerate(zip(mu, sigma, positions)):
        bar = "█" * int(m / mu.max() * 30)
        print(f"  {pos:>6}  μ={m:.4f}  σ={s:.4f}  {bar}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, mu=mu.astype("float32"), sigma=sigma.astype("float32"),
             n_beats=np.array(len(chromas)))
    print(f"\nSaved → {OUT}")

    if args.eval:
        eval_on_pop909(mu, sigma)


if __name__ == "__main__":
    main()
