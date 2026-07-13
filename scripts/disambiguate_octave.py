"""Tempo-octave disambiguator + falsification harness (known_issues.md #1).

PROBLEM
-------
librosa/madmom return a tempo that is often an integer multiple (½×, 2×, 3/2×)
of the true beat. Given a base detection we build octave-related candidates and
try to pick the true one.

WHAT THIS SCRIPT ESTABLISHES
----------------------------
It scores every candidate with four *independent* signals and evaluates, on the
10-song `docs/audio` corpus (3 hard-GT anchors + 5 soft style refs + 2 unknown),
which signal — if any — recovers the true octave:

    acf     onset-envelope autocorrelation at the beat lag  (tempo salience)
    hr      harmonic-rhythm regularity: beats-per-chord near a musical integer
    alt     metrical strong/weak alternation (fires when a grid is 2x too fast)
    prior   absolute log-normal tempo prior centred ~120 BPM  (EXTERNAL info)

Finding (see docs/octave_disambiguator_results.md): the three *audio-internal*
signals are octave-symmetric or actively wrong; only the external prior pulls
toward the truth, and it provably fails on tunes whose true tempo is far from
its centre (ballads 65-79 BPM, bebop 225 BPM). Blind ceiling ≈ 4/10 << 8/10.

Run:  .venv/bin/python scripts/disambiguate_octave.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import librosa  # noqa: E402
import soundfile as sf  # noqa: E402

AUDIO_DIR = _REPO_ROOT / "docs" / "audio"
_WAV_CACHE = Path(tempfile.gettempdir()) / "harmonia_reinfer_wav"
HOP = 512

# --- Ground truth -----------------------------------------------------------
# trust: hard = documented / filename (factor-2 error unambiguous);
#        soft = well-known style tempo, good to ~±30% (still octave-diagnostic);
#        none = genuinely unknown -> reported but EXCLUDED from accuracy.
GT: dict[str, tuple[float, str]] = {
    "blue_bossa_150bpm_backing_track": (150.0, "hard"),   # filename
    "adele_hello_official_music_video": (79.0, "hard"),   # documented 79 BPM
    "the_beatles_the_beatles_let_it_be_official_music_video_remas": (72.0, "hard"),  # ~71-73
    "autumn_leaves": (120.0, "soft"),
    "ghost_of_a_chance": (65.0, "soft"),
    "airegin": (225.0, "soft"),
    "a_foggy_day": (150.0, "soft"),
    "blue_bossa": (155.0, "soft"),
    "nina_simone_feeling_good_lyric_video": (130.0, "none"),
    "muppets_kermit_its_not_easy_being_green_original": (75.0, "none"),
}

RATIOS = [1 / 3, 1 / 2, 2 / 3, 3 / 4, 1, 4 / 3, 3 / 2, 2, 3]
# "correct octave" = not a ½×/2×/3⁄2× error. ±factor-1.32 window (log2 < 0.4);
# lenient enough for the soft GTs, tight enough to exclude the nearest wrong
# candidate (3/4 and 4/3 sit at |log2| = 0.415, just outside).
OCTAVE_TOL = 0.4


def _to_wav(path: Path) -> Path:
    _WAV_CACHE.mkdir(parents=True, exist_ok=True)
    out = _WAV_CACHE / (path.stem + ".wav")
    if not out.exists():
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
                        "-ac", "1", "-ar", "44100", str(out)], check=True)
    return out


def load(name: str):
    y, sr = sf.read(_to_wav(AUDIO_DIR / f"{name}.m4a"))
    y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
    return y, sr


# --- Signals ----------------------------------------------------------------

def candidates(base: float) -> list[float]:
    return sorted({round(base * r, 1) for r in RATIOS if 40 <= base * r <= 300})


def tempo_prior(b: float, mu: float = 120.0, sigma: float = 0.35) -> float:
    return float(np.exp(-((np.log(b) - np.log(mu)) ** 2) / (2 * sigma ** 2)))


def onset_acf(oenv: np.ndarray, sr: int, b: float) -> float:
    lag = int(round(60.0 / b * sr / HOP))
    if lag < 1 or lag >= len(oenv):
        return 0.0
    o = oenv - oenv.mean()
    d = float(np.dot(o, o))
    return float(np.dot(o[:-lag], o[lag:]) / d) if d > 0 else 0.0


def alternation(oenv: np.ndarray, sr: int, b: float) -> float:
    period = 60.0 / b * sr / HOP
    n = len(oenv)
    beats = np.arange(0, n - 1, period)
    if len(beats) < 8:
        return 0.0
    ob = np.interp(beats, np.arange(n), oenv)
    best = 0.0
    for ph in (0, 1):
        ev, od = ob[ph::2].mean(), ob[1 - ph::2].mean()
        s = ev + od
        if s > 0:
            best = max(best, (ev - od) / s)
    return float(best)


def chroma_novelty_ici(y: np.ndarray, sr: int):
    hop = 2048
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    chroma = librosa.decompose.nn_filter(chroma, aggregate=np.median, metric="cosine")
    c = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)
    nov = 1 - np.sum(c[:, 1:] * c[:, :-1], axis=0)
    times = librosa.frames_to_time(np.arange(len(nov)), sr=sr, hop_length=hop)
    peaks = librosa.util.peak_pick(nov, pre_max=3, post_max=3, pre_avg=5,
                                   post_avg=5, delta=0.1, wait=5)
    thr = nov.mean() + 0.5 * nov.std()
    peaks = [p for p in peaks if nov[p] > thr]
    if len(peaks) < 3:
        return None
    return float(np.median(np.diff(times[peaks])))


def hr_score(ici_s, b: float):
    if ici_s is None:
        return 0.0, None
    bpc = ici_s * b / 60.0
    targets = np.array([1, 2, 3, 4, 6, 8, 16])
    d = float(np.min(np.abs(targets - bpc) / targets))
    return float(np.exp(-(d ** 2) / (2 * 0.15 ** 2))), bpc


def analyse(name: str) -> dict:
    y, sr = load(name)
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    base = float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP)[0])[0])
    ici = chroma_novelty_ici(y, sr)
    rows = []
    for b in candidates(base):
        acf = onset_acf(oenv, sr, b)
        alt = alternation(oenv, sr, b)
        prior = tempo_prior(b)
        hr, bpc = hr_score(ici, b)
        rows.append(dict(bpm=b, acf=acf, alt=alt, prior=prior, hr=hr, bpc=bpc,
                         # combined disambiguator score (best blind version):
                         # salience gated by the octave prior, off-beat penalty.
                         combo=max(acf, 0.0) * prior * (1 - alt)))
    return dict(base=base, ici=ici, rows=rows)


# --- Strategies -------------------------------------------------------------

def pick(rows, key):
    return max(rows, key=lambda r: (r[key] if r[key] is not None else -9))["bpm"]


STRATEGIES = {
    "acf":       lambda rows: pick(rows, "acf"),
    "hr":        lambda rows: pick(rows, "hr"),
    "prior":     lambda rows: pick(rows, "prior"),
    "acf*prior": lambda rows: max(rows, key=lambda r: max(r["acf"], 0) * r["prior"])["bpm"],
    "combo":     lambda rows: pick(rows, "combo"),
}


def octave_ok(pick_bpm: float, gt: float) -> bool:
    return abs(np.log2(pick_bpm / gt)) < OCTAVE_TOL


def oracle(rows, gt):
    return min((r["bpm"] for r in rows), key=lambda b: abs(np.log2(b / gt)))


def main() -> None:
    results = {}
    for name in GT:
        results[name] = analyse(name)

    scored = {s: [] for s in STRATEGIES}
    oracle_hits, prior_hits = [], []
    scored["oracle"] = []

    print(f"\n{'song':45s} {'GT':>6} {'base':>6} | " +
          " ".join(f"{s:>9}" for s in STRATEGIES) + f" {'oracle':>7}")
    for name, (gt, trust) in GT.items():
        rows = results[name]["rows"]
        base = results[name]["base"]
        picks = {s: fn(rows) for s, fn in STRATEGIES.items()}
        orc = oracle(rows, gt)
        line = f"{name[:44]:45s} {gt:6.0f} {base:6.1f} | "
        for s in STRATEGIES:
            ok = octave_ok(picks[s], gt)
            line += f"{picks[s]:6.1f}{'✓' if ok else '✗':>2} " if False else \
                    f"{picks[s]:6.1f}{'Y' if ok else '.':>2} "
            if trust != "none":
                scored[s].append(ok)
        ok_orc = octave_ok(orc, gt)
        line += f"{orc:6.1f}{'Y' if ok_orc else '.':>1}"
        if trust != "none":
            scored["oracle"].append(ok_orc)
        print(line + f"   [{trust}]")

    n = len(scored["acf"])
    print(f"\nAccuracy over {n} GT-known songs (soft+hard; 2 'none' excluded):")
    for s in list(STRATEGIES) + ["oracle"]:
        hits = sum(scored[s])
        print(f"  {s:10s} {hits}/{n}  ({hits/n:.0%})")

    # Ambiguity test: for each hard-GT song, are GT-octave and 2×GT-octave
    # separable by ANY audio-internal signal?
    print("\nAmbiguity (hard-GT songs): GT-octave vs 2xGT-octave on each signal")
    print(f"{'song':40s} {'sig':>6} {'@GT':>7} {'@2xGT':>7}  separable?")
    for name, (gt, trust) in GT.items():
        if trust != "hard":
            continue
        rows = results[name]["rows"]
        def nearest(target):
            return min(rows, key=lambda r: abs(np.log2(r["bpm"] / target)))
        rg, r2 = nearest(gt), nearest(2 * gt)
        for sig in ("acf", "hr", "alt"):
            vg, v2 = rg[sig], r2[sig]
            fav = "->2xGT(WRONG)" if v2 > vg + 0.02 else \
                  ("->GT(ok)" if vg > v2 + 0.02 else "tie(ambiguous)")
            print(f"{name[:39]:40s} {sig:>6} {vg:7.3f} {v2:7.3f}  {fav}")

    out = _REPO_ROOT / "docs" / "octave_disambiguator_data.json"
    dump = {name: {"gt": GT[name][0], "trust": GT[name][1],
                   "base": results[name]["base"], "ici": results[name]["ici"],
                   "rows": results[name]["rows"],
                   "picks": {s: STRATEGIES[s](results[name]["rows"]) for s in STRATEGIES},
                   "oracle": oracle(results[name]["rows"], GT[name][0])}
            for name in GT}
    dump["_accuracy"] = {s: [bool(x) for x in scored[s]] for s in scored}
    dump["_songs_scored"] = [n for n in GT if GT[n][1] != "none"]
    out.write_text(json.dumps(dump, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
