"""Brick 0 — reproducible real-audio chord-accuracy scorer.

This is the *scoring harness* for Harmonia's frozen real-audio benchmark
("Brick 0"; see docs/known_issues.md STEP 11 / "BRICK 0 GREENLIT" and
golden/brick0/SCHEMA.md).  It is buildable and testable *now*, independent of
the real ground truth existing: the scoring math is unit-tested on synthetic
GT+prediction pairs, and the pipeline wiring is smoke-tested on any one audio
file with a placeholder GT.

WHAT THIS KILLS (the circularity, CLAUDE.md #3).
--------------------------------------------------
`scripts/validate_against_ireal.py` scores by *re-aligning* the iReal GT to the
model's OWN predictions via DTW (`align_irealb_to_inferred` picks the transpose
+ tiling with the lowest cost against the model output, then copies timestamps
off matched inferred segments onto the GT).  When the model is wrong, the GT
slides to fit it → the error is measured as zero.  We reuse that script's
scoring *ideas* (partial-credit family map, sounding-bass target) but NOT its
aligner.  Here the GT timeline is FROZEN (real audio-clock seconds, produced +
hand-verified offline) and we score by direct **frozen-GT lookup**: overlap the
prediction timeline against the frozen GT timeline on the shared audio clock,
duration-weighted, no re-alignment.

METRICS (per song + pooled micro-average, all duration-weighted).
-----------------------------------------------------------------
1. MIREX weighted-overlap — duration-weighted chord-symbol recall at the
   root / majmin / sevenths levels.  Computed by our own deterministic
   overlap engine (`score_timeline`).  `mir_eval.chord` is used, when present,
   as an independent CROSS-CHECK on the root/majmin/sevenths numbers (logged;
   asserted in tests) — it is deliberately NOT the reported source, so the
   headline number does not silently change with whether an optional package
   is installed (reproducibility is a hard requirement; mir_eval on the
   pipeline's non-standard quality tokens — dom11/dom13/7sus4 — is also
   fragile).  See `mir_eval_crosscheck`.
2. Partial-credit — root + PARENT-FAMILY match (a maj7 prediction for a maj GT
   scores as correct; the family map is the Harte-token analog of
   `harmonia.tab_aligner._family`, reused per the Brick-0 spec).
3. Strict exact-match — root + full (canonicalised) quality token.
4. Sounding-bass root target — the bass pitch class scored against the
   SOUNDING bass (`harmonia.data.corpus_schema.sounding_bass_pc`), the
   project-wide root target since 2026-07-16, NOT the functional root.

HONESTY GATE.
-------------
`score_song` REFUSES to score a GT whose ``verified`` flag is false unless the
caller passes ``allow_unverified=True`` (which additionally stamps the result
``verified=False`` and emits a loud warning) — so no fabricated accuracy number
can ship off an un-hand-verified chart.

Production config.  The prediction is produced by the *shipped* pipeline —
``infer_chords_v1`` with the live nnls24 config
(feature_frontend=nnls24, segment_source=nnls, quality/bass=musx,
beat_backend=beatthis, beat_period_mode=bestfit; ==
``harmonia.pipeline.PipelineConfig.live_defaults()``).
"""
from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]

# Shipped production pipeline config (== PipelineConfig.live_defaults() /
# scripts/harmonia_server.py analyze path).  Kept explicit here so a run is
# reproducible from this file alone.
SHIPPED_CONFIG: dict = {
    "feature_frontend": "nnls24",
    "bass_frontend": "musx",
    "quality_frontend": "musx",
    "segment_source": "nnls",
    "beat_backend": "beatthis",
    "beat_period_mode": "bestfit",
}

# ---------------------------------------------------------------------------
# Chord label parsing (Harte + Harmonia spellings) and family taxonomy
# ---------------------------------------------------------------------------

NOTE_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_NAME_TO_PC = {
    "C": 0, "B#": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "E#": 5, "F": 5, "F#": 6, "Gb": 6, "G": 7,
    "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11,
}

NO_CHORD = "N"
_NC_TOKENS = {"", "N", "X", "NC", "N.C."}

# Quality-token -> coarse family.  The Harte/pipeline-token analog of
# tab_aligner._family (which speaks iReal tokens).  Reuses the SAME 7-bucket
# scheme {maj,min,dom,dim,hdim,aug,sus} so partial-credit is consistent with
# the rest of the project.
_FAMILY: dict[str, str] = {}
for _q in ("", "maj", "major", "maj7", "maj6", "6", "maj9", "maj11", "maj13",
           "add9", "majadd9", "M", "M7"):
    _FAMILY[_q] = "maj"
for _q in ("min", "m", "-", "min7", "m7", "-7", "min6", "m6", "-6", "min9",
           "m9", "-9", "min11", "min13", "minmaj7", "min/maj7", "-^7", "mmaj7"):
    _FAMILY[_q] = "min"
for _q in ("7", "9", "11", "13", "dom", "dom7", "dom9", "dom11", "dom13",
           "7b9", "7#9", "7alt", "7sus4", "sus4(b7)", "dom7sus4"):
    _FAMILY[_q] = "dom"
for _q in ("dim", "o", "°", "dim7", "o7", "°7"):
    _FAMILY[_q] = "dim"
for _q in ("hdim7", "hdim", "m7b5", "ø", "ø7", "h7", "min7b5"):
    _FAMILY[_q] = "hdim"
for _q in ("aug", "+", "aug7", "+7", "augmaj7", "+^7", "augMaj7"):
    _FAMILY[_q] = "aug"
for _q in ("sus", "sus2", "sus4"):
    _FAMILY[_q] = "sus"

# Quality-token -> seventh-level class (mir_eval "sevenths" semantics for the
# common qualities): distinguishes maj vs maj7, min vs min7, etc.
_SEV_CLASS: dict[str, str] = {}
for _q in ("", "maj", "major", "M", "maj6", "6", "add9", "majadd9"):
    _SEV_CLASS[_q] = "maj"
for _q in ("maj7", "M7", "maj9", "maj11", "maj13"):
    _SEV_CLASS[_q] = "maj7"
for _q in ("min", "m", "-", "min6", "m6", "-6"):
    _SEV_CLASS[_q] = "min"
for _q in ("min7", "m7", "-7", "min9", "m9", "-9", "min11", "min13"):
    _SEV_CLASS[_q] = "min7"
for _q in ("minmaj7", "min/maj7", "-^7", "mmaj7"):
    _SEV_CLASS[_q] = "minmaj7"
for _q in ("7", "9", "11", "13", "dom7", "dom9", "dom11", "dom13",
           "7b9", "7#9", "7sus4", "sus4(b7)", "dom7sus4"):
    _SEV_CLASS[_q] = "7"
for _q in ("dim",):
    _SEV_CLASS[_q] = "dim"
for _q in ("dim7", "o7", "°7"):
    _SEV_CLASS[_q] = "dim7"
for _q in ("hdim7", "hdim", "m7b5", "ø7", "ø", "min7b5"):
    _SEV_CLASS[_q] = "hdim7"
for _q in ("aug",):
    _SEV_CLASS[_q] = "aug"
for _q in ("aug7", "+7"):
    _SEV_CLASS[_q] = "aug7"
for _q in ("sus2",):
    _SEV_CLASS[_q] = "sus2"
for _q in ("sus4", "sus"):
    _SEV_CLASS[_q] = "sus4"


def chord_family(quality: str) -> str:
    """Coarse family of a quality token (maj/min/dom/dim/hdim/aug/sus/N)."""
    q = (quality or "").strip()
    if q in _NC_TOKENS and q not in _FAMILY:
        return NO_CHORD
    if q in _FAMILY:
        return _FAMILY[q]
    # Unknown-token fallback (mirrors tab_aligner._family's prefix heuristics).
    if q.startswith(("-", "m")):
        return "min"
    if q.startswith(("o", "°")):
        return "dim"
    if q.startswith("+"):
        return "aug"
    if q.startswith("h") or q.startswith("ø"):
        return "hdim"
    if q.startswith("sus"):
        return "sus"
    logger.debug("chord_family: unknown quality %r -> maj", quality)
    return "maj"


def _sev_class(quality: str) -> str:
    q = (quality or "").strip()
    return _SEV_CLASS.get(q, chord_family(q))


def _canon_quality(quality: str) -> str:
    """Canonicalise a quality token for STRICT equality (spelling synonyms
    collapse; distinctions like maj7 vs maj9 are preserved)."""
    q = (quality or "").strip()
    synonyms = {
        "": "maj", "major": "maj", "M": "maj", "M7": "maj7",
        "m": "min", "-": "min", "m7": "min7", "-7": "min7",
        "m6": "min6", "-6": "min6", "m9": "min9", "-9": "min9",
        "o": "dim", "°": "dim", "o7": "dim7", "°7": "dim7",
        "ø7": "hdim7", "ø": "hdim7", "h7": "hdim7", "m7b5": "hdim7",
        "+": "aug", "sus": "sus4",
    }
    return synonyms.get(q, q)


@dataclass(frozen=True)
class Chord:
    """One chord span on the audio clock.  ``root_pc``/``bass_pc`` are None for
    no-chord (N)."""
    t0: float
    t1: float
    root_pc: int | None
    quality: str
    bass_pc: int | None
    label: str

    @property
    def is_nc(self) -> bool:
        return self.root_pc is None


def _split_root(base: str) -> tuple[str, str]:
    """Split a colon-less label like 'Cmaj7' or 'Bbm7' into (root, quality)."""
    if len(base) >= 2 and base[1] in "#b" and base[:2] in _NAME_TO_PC:
        return base[:2], base[2:]
    if base[:1] in _NAME_TO_PC:
        return base[:1], base[1:]
    return base, ""


def parse_label(label: str) -> tuple[int | None, str, int | None]:
    """Parse a chord label -> (root_pc, quality, sounding_bass_pc).

    Accepts Harte colon labels ('C:maj7', 'A:min7/C', 'G:7/B'), Harmonia
    colon-less labels ('Cmaj7'), and no-chord tokens.  The sounding bass is
    resolved with the canonical `corpus_schema.sounding_bass_pc` so the
    scorer's bass target is byte-identical to the GT producer's.
    """
    from harmonia.data.corpus_schema import sounding_bass_pc

    if label is None:
        return None, NO_CHORD, None
    s = label.strip()
    if s in _NC_TOKENS:
        return None, NO_CHORD, None

    base = s.split("/", 1)[0].strip()
    if ":" in base:
        root_str, quality = base.split(":", 1)
    else:
        root_str, quality = _split_root(base)
    root_str, quality = root_str.strip(), quality.strip()
    root_pc = _NAME_TO_PC.get(root_str)
    if root_pc is None:
        logger.debug("parse_label: unparseable root in %r", label)
        return None, NO_CHORD, None
    bass_pc = sounding_bass_pc(s, root_pc)
    return root_pc, quality, bass_pc


def chord_from_label(t0: float, t1: float, label: str) -> Chord:
    root_pc, quality, bass_pc = parse_label(label)
    if root_pc is None:
        return Chord(t0, t1, None, NO_CHORD, None, NO_CHORD)
    return Chord(t0, t1, root_pc, quality, bass_pc, label)


# ---------------------------------------------------------------------------
# The scoring engine — duration-weighted frozen-GT overlap
# ---------------------------------------------------------------------------

_NC_CHORD_CACHE = Chord(0.0, 0.0, None, NO_CHORD, None, NO_CHORD)


def _lookup(chords: list[Chord], t: float) -> Chord:
    """The chord active at time ``t`` (t0 <= t < t1), or a no-chord filler."""
    for c in chords:
        if c.t0 <= t < c.t1:
            return c
    return _NC_CHORD_CACHE


def _majmin_bucket(c: Chord) -> str:
    if c.is_nc:
        return NO_CHORD
    fam = chord_family(c.quality)
    return "min" if fam in ("min", "hdim", "dim") else "maj"


def _root_match(gt: Chord, pr: Chord) -> bool:
    if gt.is_nc and pr.is_nc:
        return True
    if gt.is_nc or pr.is_nc:
        return False
    return gt.root_pc == pr.root_pc


def score_timeline(gt_chords: list[Chord], pred_chords: list[Chord]) -> dict:
    """Score a prediction timeline against a FROZEN GT timeline.

    Both are lists of `Chord` on the same audio clock.  We build the union of
    all interval boundaries within the GT span, and for each sub-interval look
    up the GT chord and the predicted chord active there (frozen-GT lookup — no
    re-alignment), weighting every metric by the sub-interval's duration.

    Returns a dict with the six duration-weighted metrics in [0,1], the scored
    duration, and the raw weighted numerators (seconds) so multiple songs can
    be pooled exactly into a micro-average.
    """
    if not gt_chords:
        raise ValueError("empty GT timeline")

    span_lo = min(c.t0 for c in gt_chords)
    span_hi = max(c.t1 for c in gt_chords)

    pts = {span_lo, span_hi}
    for c in list(gt_chords) + list(pred_chords):
        if span_lo < c.t0 < span_hi:
            pts.add(c.t0)
        if span_lo < c.t1 < span_hi:
            pts.add(c.t1)
    bps = sorted(pts)

    num = {k: 0.0 for k in ("root", "majmin", "sevenths",
                            "partial", "strict", "bass")}
    total = 0.0

    for a, b in zip(bps[:-1], bps[1:]):
        dur = b - a
        if dur <= 0:
            continue
        mid = 0.5 * (a + b)
        gt = _lookup(gt_chords, mid)
        pr = _lookup(pred_chords, mid)
        total += dur

        rmatch = _root_match(gt, pr)
        if rmatch:
            num["root"] += dur
            if _majmin_bucket(gt) == _majmin_bucket(pr):
                num["majmin"] += dur
            if _sev_class(gt.quality) == _sev_class(pr.quality):
                num["sevenths"] += dur
            if chord_family(gt.quality) == chord_family(pr.quality):
                num["partial"] += dur
            if _canon_quality(gt.quality) == _canon_quality(pr.quality):
                num["strict"] += dur
        # Sounding-bass target is independent of the root match.
        if gt.bass_pc is None and pr.bass_pc is None:
            num["bass"] += dur
        elif gt.bass_pc is not None and gt.bass_pc == pr.bass_pc:
            num["bass"] += dur

    def _frac(k: str) -> float:
        return num[k] / total if total else 0.0

    return {
        "duration_s": round(total, 4),
        "mirex_root": round(_frac("root"), 4),
        "mirex_majmin": round(_frac("majmin"), 4),
        "mirex_sevenths": round(_frac("sevenths"), 4),
        "partial_credit": round(_frac("partial"), 4),
        "strict": round(_frac("strict"), 4),
        "bass_root": round(_frac("bass"), 4),
        "_num": num,
        "_total": total,
    }


# ---------------------------------------------------------------------------
# mir_eval cross-check (optional, best-effort)
# ---------------------------------------------------------------------------

def _to_harte(c: Chord) -> str:
    """A mir_eval-safe Harte label for a `Chord` (best-effort; pipeline-only
    quality tokens are reduced to the nearest mir_eval shorthand)."""
    if c.is_nc:
        return "N"
    reduce = {"dom11": "11", "dom13": "13", "7sus4": "sus4(b7)",
              "sus4(b7)": "sus4(b7)"}
    q = reduce.get(c.quality, c.quality) or "maj"
    root = NOTE_SHARP[c.root_pc % 12]
    lab = f"{root}:{q}"
    if c.bass_pc is not None and c.bass_pc != c.root_pc:
        off = (c.bass_pc - c.root_pc) % 12
        deg = {1: "b2", 2: "2", 3: "b3", 4: "3", 5: "4", 6: "b5", 7: "5",
               8: "b6", 9: "6", 10: "b7", 11: "7"}.get(off)
        if deg:
            lab += f"/{deg}"
    return lab


def mireval_crosscheck(gt_chords: list[Chord],
                       pred_chords: list[Chord]) -> dict | None:
    """Independent MIREX root/majmin/sevenths via mir_eval, or None if mir_eval
    is unavailable or the labels trip it.  Used only to VALIDATE the own-engine
    numbers (not reported as the headline)."""
    try:
        import mir_eval.chord as mec
        import numpy as np
    except ImportError:
        return None
    try:
        ref_int = np.array([[c.t0, c.t1] for c in gt_chords])
        ref_lab = [_to_harte(c) for c in gt_chords]
        est_int = np.array([[c.t0, c.t1] for c in pred_chords])
        est_lab = [_to_harte(c) for c in pred_chords]
        scores = mec.evaluate(ref_int, ref_lab, est_int, est_lab)
        return {
            "root": float(scores["root"]),
            "majmin": float(scores["majmin"]),
            "sevenths": float(scores["sevenths"]),
        }
    except Exception as e:  # noqa: BLE001 — cross-check must never break scoring
        logger.debug("mir_eval cross-check skipped: %s", e)
        return None


# ---------------------------------------------------------------------------
# Frozen GT loading
# ---------------------------------------------------------------------------

class UnverifiedGTError(RuntimeError):
    """Raised when scoring an un-hand-verified frozen GT without opt-in."""


@dataclass
class FrozenGT:
    song_id: str
    title: str
    audio_path: str
    verified: bool
    gt_chords: list[Chord]
    raw: dict = field(default_factory=dict)

    @property
    def resolved_audio_path(self) -> Path:
        p = Path(self.audio_path)
        return p if p.is_absolute() else (REPO / p)


_REQUIRED_TOP = ("song_id", "audio_path", "gt_chords", "verified")
_REQUIRED_CHORD = ("t0", "t1", "root_pc", "quality", "bass_pc", "label")


def load_frozen_gt(path: str | Path) -> FrozenGT:
    """Load + validate a per-song frozen GT json (golden/brick0/SCHEMA.md)."""
    path = Path(path)
    data = json.loads(path.read_text())
    for k in _REQUIRED_TOP:
        if k not in data:
            raise ValueError(f"{path.name}: missing required field {k!r}")

    chords: list[Chord] = []
    prev_t1 = None
    for i, gc in enumerate(data["gt_chords"]):
        for k in _REQUIRED_CHORD:
            if k not in gc:
                raise ValueError(
                    f"{path.name}: gt_chords[{i}] missing field {k!r}")
        t0, t1 = float(gc["t0"]), float(gc["t1"])
        if t1 <= t0:
            raise ValueError(
                f"{path.name}: gt_chords[{i}] non-positive span [{t0},{t1}]")
        if prev_t1 is not None and t0 < prev_t1 - 1e-6:
            logger.warning("%s: gt_chords[%d] overlaps previous (%.3f < %.3f)",
                           path.name, i, t0, prev_t1)
        prev_t1 = t1
        root_pc = gc["root_pc"]
        chords.append(Chord(
            t0=t0, t1=t1,
            root_pc=None if root_pc is None else int(root_pc) % 12,
            quality=str(gc["quality"]),
            bass_pc=None if gc["bass_pc"] is None else int(gc["bass_pc"]) % 12,
            label=str(gc["label"]),
        ))

    return FrozenGT(
        song_id=str(data["song_id"]),
        title=str(data.get("title", data["song_id"])),
        audio_path=str(data["audio_path"]),
        verified=bool(data["verified"]),
        gt_chords=chords,
        raw=data,
    )


# ---------------------------------------------------------------------------
# Production pipeline prediction
# ---------------------------------------------------------------------------

def _decode_to_wav(audio_path: Path, out_dir: Path) -> Path:
    """soundfile (sf.read, used by infer_chords_v1) cannot read m4a/mp3 — decode
    to a temp WAV with ffmpeg (deterministic).  WAV/FLAC pass through."""
    if audio_path.suffix.lower() in (".wav", ".flac"):
        return audio_path
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / (audio_path.stem + ".wav")
    if not wav.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio_path),
             "-ac", "1", "-ar", "44100", str(wav)],
            check=True,
        )
    return wav


def run_prediction(audio_path: str | Path,
                   *,
                   cache_dir: Path | None = None,
                   work_dir: Path | None = None) -> list[Chord]:
    """Run the SHIPPED pipeline (`infer_chords_v1`, nnls24 live config) on an
    audio file and return its chord timeline as `Chord`s.

    Determinism: given a fixed audio file and fixed model weights the pipeline
    is deterministic (torch eval-mode nets, no sampling), so re-running yields
    the same prediction and therefore the same score.
    """
    from harmonia.models.chord_pipeline_v1 import infer_chords_v1

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)
    tmp = work_dir or Path(tempfile.mkdtemp(prefix="brick0_"))
    wav = _decode_to_wav(audio_path, tmp)

    chart = infer_chords_v1(wav, cache_dir=cache_dir, **SHIPPED_CONFIG)
    return [chord_from_label(c["start_s"], c["end_s"], c["label"])
            for c in chart.chords]


# ---------------------------------------------------------------------------
# Per-song + dataset scoring
# ---------------------------------------------------------------------------

@dataclass
class SongScore:
    song_id: str
    title: str
    verified: bool
    n_gt_chords: int
    duration_s: float
    mirex_root: float
    mirex_majmin: float
    mirex_sevenths: float
    partial_credit: float
    strict: float
    bass_root: float
    mir_eval_crosscheck: dict | None = None
    _num: dict = field(default_factory=dict)
    _total: float = 0.0

    def summary_line(self) -> str:
        v = "" if self.verified else "  [UNVERIFIED]"
        return (f"root={self.mirex_root:.3f} majmin={self.mirex_majmin:.3f} "
                f"7ths={self.mirex_sevenths:.3f} | partial={self.partial_credit:.3f} "
                f"strict={self.strict:.3f} bass={self.bass_root:.3f} "
                f"[{self.duration_s:.0f}s n={self.n_gt_chords}]{v}")


def score_song(gt_path: str | Path,
               *,
               allow_unverified: bool = False,
               cache_dir: Path | None = None,
               work_dir: Path | None = None,
               pred_chords: list[Chord] | None = None) -> SongScore:
    """Score one frozen-GT song end to end.

    Refuses (UnverifiedGTError) if the GT is not hand-verified, unless
    ``allow_unverified=True`` (which stamps the result verified=False + warns).
    Pass ``pred_chords`` to score a supplied prediction timeline instead of
    re-running the pipeline (used by the math unit tests).
    """
    gt = load_frozen_gt(gt_path)
    if not gt.verified and not allow_unverified:
        raise UnverifiedGTError(
            f"{gt.song_id}: frozen GT is UNVERIFIED (verified=false). Refusing "
            f"to score — hand-verify the chart (transpose/form/bar-1/timeline) "
            f"and set verified=true, or pass allow_unverified=True for a wiring "
            f"test whose number MUST NOT be reported as an accuracy result.")
    if not gt.verified:
        logger.warning("SCORING UNVERIFIED GT %s — result is a WIRING artifact, "
                       "NOT a reportable accuracy number.", gt.song_id)

    if pred_chords is None:
        pred_chords = run_prediction(gt.resolved_audio_path,
                                     cache_dir=cache_dir, work_dir=work_dir)

    res = score_timeline(gt.gt_chords, pred_chords)
    xcheck = mireval_crosscheck(gt.gt_chords, pred_chords)
    if xcheck is not None:
        d = abs(xcheck["root"] - res["mirex_root"])
        if d > 0.02:
            logger.warning("%s: own-engine root %.3f vs mir_eval root %.3f "
                           "(|Δ|=%.3f > 0.02)", gt.song_id,
                           res["mirex_root"], xcheck["root"], d)

    return SongScore(
        song_id=gt.song_id, title=gt.title, verified=gt.verified,
        n_gt_chords=len(gt.gt_chords), duration_s=res["duration_s"],
        mirex_root=res["mirex_root"], mirex_majmin=res["mirex_majmin"],
        mirex_sevenths=res["mirex_sevenths"],
        partial_credit=res["partial_credit"], strict=res["strict"],
        bass_root=res["bass_root"], mir_eval_crosscheck=xcheck,
        _num=res["_num"], _total=res["_total"],
    )


@dataclass
class DatasetScore:
    per_song: list[SongScore] = field(default_factory=list)

    @property
    def n_songs(self) -> int:
        return len(self.per_song)

    @property
    def pooled(self) -> dict:
        """Duration-weighted micro-average over all scored sub-intervals."""
        keys = ("root", "majmin", "sevenths", "partial", "strict", "bass")
        num = {k: 0.0 for k in keys}
        total = 0.0
        for s in self.per_song:
            total += s._total
            for k in keys:
                num[k] += s._num.get(k, 0.0)
        return {
            "n_songs": self.n_songs,
            "duration_s": round(total, 3),
            "mirex_root": round(num["root"] / total, 4) if total else 0.0,
            "mirex_majmin": round(num["majmin"] / total, 4) if total else 0.0,
            "mirex_sevenths": round(num["sevenths"] / total, 4) if total else 0.0,
            "partial_credit": round(num["partial"] / total, 4) if total else 0.0,
            "strict": round(num["strict"] / total, 4) if total else 0.0,
            "bass_root": round(num["bass"] / total, 4) if total else 0.0,
        }

    def print(self) -> None:
        print(f"\n{'-'*72}")
        print(f"  BRICK 0 accuracy — {self.n_songs} song(s)")
        print(f"{'-'*72}")
        for s in self.per_song:
            print(f"  {s.song_id[:34]:34} {s.summary_line()}")
        p = self.pooled
        print(f"{'-'*72}")
        print(f"  POOLED (micro, duration-weighted, {p['duration_s']:.0f}s): "
              f"root={p['mirex_root']:.3f} majmin={p['mirex_majmin']:.3f} "
              f"7ths={p['mirex_sevenths']:.3f} | partial={p['partial_credit']:.3f} "
              f"strict={p['strict']:.3f} bass={p['bass_root']:.3f}")
        print(f"{'-'*72}\n")


def score_dataset(gt_paths: list[str | Path],
                  *,
                  allow_unverified: bool = False,
                  cache_dir: Path | None = None) -> DatasetScore:
    ds = DatasetScore()
    for gp in gt_paths:
        try:
            ds.per_song.append(score_song(
                gp, allow_unverified=allow_unverified, cache_dir=cache_dir))
        except UnverifiedGTError as e:
            logger.error("SKIP %s", e)
    return ds


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Brick 0 chord-accuracy scorer")
    ap.add_argument("gt", nargs="+", help="frozen-GT json file(s)")
    ap.add_argument("--allow-unverified", action="store_true",
                    help="score UNVERIFIED GT (wiring only; NOT a real number)")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cache = Path(args.cache_dir) if args.cache_dir else None
    ds = score_dataset(args.gt, allow_unverified=args.allow_unverified,
                       cache_dir=cache)
    ds.print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
