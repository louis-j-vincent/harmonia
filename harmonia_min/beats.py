"""harmonia_min/beats.py — Beat This! beats + downbeats. The ONLY beat tracker.

Re-extracted 2026-07-30 from chord_pipeline_v1._get_beatthis (~15 lines) +
harmonia/serving/audio.py::_beats_and_downbeats (the m4a lesson), for the
minimal rebuild. Two hard-won rules are load-bearing here:

1. **librosa is banned.** It locks 2x tempo octaves (validated 2026-07-21:
   tempo-octave 65% vs Beat This! 78% on POP909; song 002 doubled to ~129 BPM
   against three agreeing GT annotations at ~64). The old code silently fell
   back to librosa whenever Beat This! threw — which was EVERY m4a on this box
   (see 2), so the fallback was the code path. Here a failure raises.

2. **Beat This! must get a wav.** It reads audio via torchaudio / soundfile /
   madmom and on this box all three refuse .m4a (no torchcodec; libsndfile has
   no AAC; madmom is py3.12-broken). Discovered 2026-07-30 after the librosa
   fallback had silently served every song. So: try the file as-is, and on any
   decode error transcode to a temp wav with ffmpeg and retry. Only if THAT
   fails do we raise.

Results are disk-cached to harmonia_min/state/beats/<stem>.json (stem-keyed,
same rationale as the musx cache: fresh downloads of the same video get new
mtimes, the stem is the stable key).
"""
from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent / "state" / "beats"

_f2b = None  # loaded once per process (~2 s model load)


def _get_beatthis():
    global _f2b
    if _f2b is None:
        from beat_this.inference import File2Beats
        _f2b = File2Beats(device="cpu", dbn=False)
        logger.info("beats: loaded Beat This! (MIT) beat+downbeat tracker")
    return _f2b


class BeatTrackingError(RuntimeError):
    """Beat This! could not produce a usable beat grid. No fallback exists —
    librosa is banned (2x tempo-octave lock), so the caller must surface this."""


# ── grid guard (Louis, 2026-08-05) ──────────────────────────────────────────
# « utilise ça comme garde-fou et tu refuses, signales LOUDLY une chanson dont
# la grille échoue pour mauvais calage rythmique. »
#
# The whole pipeline downstream assumes FOUR beats per bar: `pipeline.analyze`
# builds bars as `off + b*bpb` over beat INDICES with bpb=4. When the tracker
# does not actually deliver that, the "bars" are not musical bars, and every
# similarity computed on them compares misaligned material — Georgia On My Mind
# has 5.5% of its SSM above 0.80 and Billie Jean 56.7%, and neither number says
# anything about repetition (docs/known_issues.md, 2026-08-05).
#
# The measure is the beat count BETWEEN consecutive downbeats: a healthy 4/4
# track gives 4,4,4,4… Its mode is the detected metre and the share of bars
# hitting that mode is the grid's self-consistency. Measured over the 82 cached
# tracks: median consistency 1.000, and the songs that fail are exactly the ones
# whose charts were wrong — Georgia 0.45 with mode 2, blue_bossa 0.53 mode 2,
# Close to You 0.64 — while This Love, Sunny and Don't Know Why sit at 1.00 and
# Billie Jean at 0.94.
#
# A cruder ratio (len(beats)/len(downbeats)) was tried first and rejected: its
# corpus median is 3.82, so it would have flagged half of a healthy corpus.
GRID_MIN_CONSISTENCY = 0.80   # 0.70/0.75/0.80 all refuse the same 5 of 22 real
                              # songs; 0.85 starts taking healthy ones (8/22).
GRID_MIN_BARS = 30            # below this the statistic is noise (GuitarSet
                              # excerpts run 10-16 bars) — don't judge.


def grid_quality(beats, downbeats) -> dict:
    """{metre, consistency, n_bars} — how well the downbeats tile the beats."""
    import numpy as np
    from collections import Counter
    b = np.asarray(beats, float)
    db = np.asarray(downbeats, float)
    if len(b) < 8 or len(db) < 4:
        return {"metre": None, "consistency": 0.0, "n_bars": 0}
    idx = [int(np.argmin(np.abs(b - t))) for t in db]
    gaps = [j - i for i, j in zip(idx, idx[1:]) if j > i]
    if len(gaps) < 3:
        return {"metre": None, "consistency": 0.0, "n_bars": len(gaps)}
    metre, cnt = Counter(gaps).most_common(1)[0]
    return {"metre": int(metre), "consistency": cnt / len(gaps),
            "n_bars": len(gaps)}


def check_grid(beats, downbeats, name: str, bpb: int = 4,
               allowed: tuple = (3, 4)) -> dict:
    """Raise BeatTrackingError unless the grid carries a legitimate metre.

    Refuses loudly rather than producing a chart built on bars that are not
    bars — a wrong chart is worse than no chart, and this failure was
    previously invisible.

    2026-08-07 (Louis: « les tiers de barre doivent pouvoir s'afficher ») —
    a detected metre of 3 is a WALTZ, not an error: it is accepted alongside
    4, with the same consistency doctrine. 2 stays refused (that is the
    half-tempo/octave-error signature — Georgia On My Mind), and so do
    5/6/7 for now: no corpus song has exercised them, and letting an
    unvalidated metre through would silently rebuild the georgia trap one
    number higher (rule #4: that remainder is NOT solved here).
    """
    q = grid_quality(beats, downbeats)
    if q["n_bars"] < GRID_MIN_BARS:
        return q                       # too short to judge; let it through
    if q["metre"] not in allowed:
        raise BeatTrackingError(
            f"{name}: the beat tracker reports {q['metre']} beats per bar — "
            f"not a metre this chart can carry (allowed: "
            f"{'/'.join(map(str, allowed))}) — the bar grid would be wrong "
            f"for the whole song. "
            f"(grid consistency {q['consistency']:.0%} over {q['n_bars']} bars)")
    if q["consistency"] < GRID_MIN_CONSISTENCY:
        raise BeatTrackingError(
            f"{name}: only {q['consistency']:.0%} of bars actually hold "
            f"{q['metre']} beats (needs {GRID_MIN_CONSISTENCY:.0%}) over "
            f"{q['n_bars']} bars — the rhythm is too loose or rubato for a "
            f"fixed bar grid, so the chart would be built on bars that are "
            f"not bars.")
    return q


def track(audio_path: str | Path, *, use_cache: bool = True) -> dict:
    """Beats + downbeats for one audio file.

    Returns {"beats": [s...], "downbeats": [s...], "bpm": float}.
    Raises BeatTrackingError instead of ever falling back to another tracker.
    """
    audio_path = Path(audio_path)
    cache = CACHE_DIR / f"{audio_path.stem}.json"
    if use_cache and cache.exists():
        try:
            d = json.loads(cache.read_text(encoding="utf-8"))
            if len(d.get("beats", [])) >= 4:
                return d
        except ValueError:
            pass

    f2b = _get_beatthis()
    try:
        bts, dbs = f2b(str(audio_path))
    except Exception:
        # Almost always "cannot decode m4a" — transcode and retry (lesson 2).
        with tempfile.TemporaryDirectory() as td:
            wav = f"{td}/a.wav"
            try:
                subprocess.run(
                    ["ffmpeg", "-v", "error", "-y", "-i", str(audio_path),
                     "-ac", "1", "-ar", "22050", wav],
                    check=True, timeout=300)
                bts, dbs = f2b(wav)
            except Exception as exc:
                raise BeatTrackingError(
                    f"Beat This! failed on {audio_path.name} even after ffmpeg "
                    f"transcode ({exc}). No librosa fallback — fix the input."
                ) from exc

    beats = [round(float(t), 4) for t in bts]
    downbeats = [round(float(t), 4) for t in dbs]
    if len(beats) < 4:
        raise BeatTrackingError(
            f"Beat This! returned only {len(beats)} beats for {audio_path.name}")
    import numpy as np
    bpm = round(60.0 / float(np.median(np.diff(beats))), 2)
    out = {"beats": beats, "downbeats": downbeats, "bpm": bpm}
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out), encoding="utf-8")
    return out
