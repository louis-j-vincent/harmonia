"""(chart, audio) -> high-precision (audio-segment -> chord) dataset rows.

The harvest pipeline:

  1. **Align** — run the existing Brick-0 aligner (``scripts/brick0_propose.py``,
     imported READ-ONLY; never edited) on the (chart, audio) pair. The aligner is
     monkey-pointed at a PRIVATE cache dir (``.dataset_cache/``) so it never
     touches ``golden/`` or ``docs/brick0_review/`` (another lane owns those).
     It emits a proposal JSON with per-chord harmonic agreement, per-section
     coverage, transpose margins, cross-repetition divergences and mid-span
     split fires — all **non-circular** (raw CQT chroma vs chart chord-tones,
     never the shipped model's decode).
  2. **Beat-lock** — run the Stage-1 drum tracker
     (``harmonia.align.drum_pattern``) for the per-span beat reliability +
     octave-lock flag (the downbeat model will fold in here later).
  3. **Gate** — turn the per-segment signals into a 3-way decision
     (``harmonia.dataset.gate``): CLEAN (emit), REVIEW (substitution queue),
     DROP.
  4. **Emit** — CLEAN segments -> the clean GT manifest rows; REVIEW segments ->
     the review/substitution manifest rows. Audio stays on disk; a row only
     REFERENCES it (path + offsets).

Audio never gets copied into the repo; rows carry ``audio_path`` + ``t0/t1``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .gate import Bucket, GateConfig, GateDecision, SegmentSignals, gate_segment

log = logging.getLogger("harmonia.dataset.harvest")

# ── repo-relative paths ──────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / ".dataset_cache"                 # gitignored; aligner proposals + drum tracks
PROPOSAL_CACHE = CACHE / "proposals"
DRUM_CACHE = CACHE / "drum"
DATASET_DIR = REPO / "data" / "chord_dataset"   # gitignored; the emitted manifests
CLEAN_MANIFEST = DATASET_DIR / "manifest.jsonl"
REVIEW_MANIFEST = DATASET_DIR / "review_manifest.jsonl"

# cross-rep positional-variance ceiling: above this a repeated section is judged
# internally inconsistent across passes => its spans are not CLEAN-eligible.
_XREP_VAR_MAX = 0.11
# divergence "uniformly low" agreement ceiling (mirrors cross_rep_analysis DIV_MEAN).
_DIV_MEAN = 0.16
# per-song beat-lock reference percentile: this song's steadiest-drum level == 1.0.
_BEATLOCK_REF_PCTL = 85.0
_BEATLOCK_REF_FLOOR = 0.05
# a chord onset/offset within this many seconds of a gap boundary is "near_gap".
_GAP_TOL_S = 0.6

# ── downbeat fold (harmonia/align/downbeat.py -> per-span beat_lock) ──────────
# The global-phase downbeat resolver's per-SONG confidence/flag folds into the
# per-SPAN beat_lock. gate.py needs no change (it already consumes a single [0,1]
# beat-lock). The fold is DIRECTIONAL and precision-first:
#
#  * An UNFLAGGED, CONFIDENT downbeat corroborates the timing -> a bounded beat_lock
#    BOOST (up to +``_DB_BOOST_MAX``, saturating at confidence ``_DB_CONF_REF``).
#    This lifts recall on well-placed spans of the confident-downbeat pop songs.
#  * A FLAGGED (ambiguous-phase / weak-margin / mid-song phase-flip) downbeat
#    ABSTAINS: gain == 1.0. It grants NO boost -> a flagged song can never be
#    promoted to CLEAN *by the downbeat* ("never emit on a shaky downbeat"). It is
#    deliberately NOT capped DOWN: the downbeat PHASE (which beat is beat 1) is
#    ORTHOGONAL to chord-label correctness, so a literal "cap beat_lock low" would
#    DELETE verified-correct data and REGRESS the frozen precision bar — e.g. the
#    chroma-flat 9-min Blue Bossa jam is downbeat-FLAGGED (conf ~0.03) yet harvests
#    at 100% label precision (human-anchored timing + high agreement); capping it
#    removes ~235s of perfect rows and drops duration-weighted pooled precision
#    ~0.944 -> ~0.92. Precision is paramount (CLAUDE.md) -> the flag withholds
#    LIFT, it does not tear down independently-justified beat_lock.
_DB_BOOST_MAX = 0.30   # max fractional beat_lock boost from a confident downbeat
_DB_CONF_REF = 0.50    # downbeat confidence at which the boost saturates


def _downbeat_gain(confidence: float, flagged: bool) -> float:
    """Per-song multiplier folded into every span's beat_lock.

    ``flagged`` -> 1.0 (abstain: no boost, no teardown). Otherwise a bounded boost
    ``1 + _DB_BOOST_MAX * clip(confidence/_DB_CONF_REF, 0, 1)`` in
    ``[1, 1+_DB_BOOST_MAX]``. Never < 1 -> the downbeat fold can only ADD recall on
    a confident song, never SUBTRACT confidence the drum tracker + agreement
    already justify (see the module note; precision-first)."""
    if flagged:
        return 1.0
    return 1.0 + _DB_BOOST_MAX * float(np.clip(confidence / _DB_CONF_REF, 0.0, 1.0))


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1: run the aligner (read-only import) into a private cache
# ═════════════════════════════════════════════════════════════════════════════

def _load_brick0():
    """Import ``scripts/brick0_propose.py`` READ-ONLY (never edited)."""
    scripts = REPO / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import brick0_propose as bp  # type: ignore
    return bp


def run_aligner(song: dict, *, force: bool = False, write_html: bool = False) -> dict:
    """Align one song and return the aligner's proposal dict.

    ``song`` needs ``song_id``, ``title``, ``audio`` (repo-relative or absolute),
    ``ireal_file`` and ``tune_title`` (the aligner reads ``data/ireal/<file>.txt``).
    Optional timing hints (``human_anchor``, ``onset_nudge``, ``scored_end``) pass
    through. **Do NOT** pass label overrides (``gt_overrides``) here — the harvest
    consumes the aligner's *raw* chart-derived labels so the gate/divergence
    detectors can do their job.

    The aligner is pointed at a private cache dir, so ``golden/`` and
    ``docs/brick0_review/`` are never touched. Result is cached per (song_id,
    config-hash).
    """
    bp = _load_brick0()
    key = _song_key(song)
    out = PROPOSAL_CACHE / f"{song['song_id']}.{key}.gt.json"
    if out.exists() and not force:
        return json.loads(out.read_text())

    PROPOSAL_CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "review").mkdir(parents=True, exist_ok=True)
    (CACHE / "workdir").mkdir(parents=True, exist_ok=True)

    # Point the aligner's write targets at the private cache (both are under REPO,
    # so its relative_to(REPO) logging stays happy). Restore afterwards.
    saved = (bp.GOLDEN, bp.REVIEW)
    bp.GOLDEN = PROPOSAL_CACHE
    bp.REVIEW = CACHE / "review"
    try:
        # process() writes <GOLDEN>/<song_id>.gt.json (+ an HTML in <REVIEW>).
        bp.process(dict(song), CACHE / "workdir", write=True)
        raw = bp.GOLDEN / f"{song['song_id']}.gt.json"
        data = json.loads(raw.read_text())
        out.write_text(json.dumps(data))
        raw.unlink(missing_ok=True)
        if not write_html:
            (bp.REVIEW / f"{song['song_id']}.html").unlink(missing_ok=True)
        return data
    finally:
        bp.GOLDEN, bp.REVIEW = saved


def _song_key(song: dict) -> str:
    keyable = {k: song.get(k) for k in
               ("audio", "ireal_file", "tune_title", "human_anchor",
                "onset_nudge", "scored_end")}
    return hashlib.sha1(json.dumps(keyable, sort_keys=True).encode()).hexdigest()[:10]


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2: drum-tracker beat reliability
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class BeatLock:
    """Per-song beat-lock summary the gate consumes (normalised).

    ``db_apply`` folds the downbeat resolver's per-song ``db_confidence`` /
    ``db_flagged`` into every span's beat_lock via ``_downbeat_gain`` (see the
    module note). Defaults keep the fold OFF so the pure gate tests (which build a
    synthetic BeatLock) and any pre-downbeat caller are byte-identical to before;
    ``compute_beat_lock(use_downbeat=True)`` turns it on.
    """
    octave_locked: bool
    ref_level: float                      # this-song's steady-drum reliability (== 1.0)
    _rel_times: np.ndarray
    _rel: np.ndarray
    db_confidence: float = 1.0            # downbeat resolver confidence [0,1]
    db_flagged: bool = False             # downbeat resolver precision-first flag
    db_apply: bool = False               # fold the downbeat gain into span_lock?

    def span_lock(self, t0: float, t1: float) -> float:
        """Per-song-normalised beat-lock over [t0, t1], clipped to [0, 1], with
        the downbeat gain folded in when ``db_apply``."""
        if len(self._rel) == 0 or self.ref_level <= 0:
            return 0.0
        m = (self._rel_times >= t0) & (self._rel_times <= t1)
        if not m.any():
            v = float(np.interp((t0 + t1) / 2, self._rel_times, self._rel))
        else:
            v = float(self._rel[m].mean())
        v = float(np.clip(v / self.ref_level, 0.0, 1.0))
        if self.db_apply:
            v = float(np.clip(v * _downbeat_gain(self.db_confidence, self.db_flagged),
                              0.0, 1.0))
        return v


def compute_beat_lock(audio_path: Path, prior_bpm: Optional[float] = None,
                      prior_period: Optional[float] = None, *,
                      use_downbeat: bool = True,
                      force: bool = False) -> BeatLock:
    """Run the Stage-1 drum tracker and summarise its reliability curve.

    ``prior_period`` (or ``prior_bpm``) is the tempo-octave prior — pass the
    aligner's own ``beat_period`` so the drum tracker locks the same octave and
    we don't invoke Beat This! twice.

    When ``use_downbeat`` (default), the SAME drum track is fed to the global-phase
    downbeat resolver (``harmonia.align.downbeat.resolve_downbeat``) together with
    raw CQT chroma (harmonic-rhythm term) + the bass-salience stream (bass root-on-1
    term); its per-song ``confidence`` / ``flagged`` are stored on the returned
    ``BeatLock`` and folded into every span's beat_lock (see ``_downbeat_gain`` +
    the module note). ``use_downbeat=False`` reproduces the pre-downbeat behaviour
    exactly (identity fold).
    """
    from harmonia.align.drum_pattern import track_from_audio

    DRUM_CACHE.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha1(
        f"{audio_path}:{prior_bpm}:{prior_period}:db{int(use_downbeat)}".encode()
    ).hexdigest()[:10]
    cache = DRUM_CACHE / f"{Path(audio_path).stem}.{tag}.npz"
    if cache.exists() and not force:
        z = np.load(cache)
        return BeatLock(bool(z["octave_locked"]), float(z["ref_level"]),
                        z["rel_times"], z["rel"],
                        db_confidence=float(z["db_confidence"]) if "db_confidence" in z else 1.0,
                        db_flagged=bool(z["db_flagged"]) if "db_flagged" in z else False,
                        db_apply=bool(z["db_apply"]) if "db_apply" in z else False)

    trk = track_from_audio(str(audio_path), prior_bpm=prior_bpm,
                           prior_period=prior_period, run_beat_this=False)
    rel = np.asarray(trk.reliability, float)
    ref = max(float(np.percentile(rel, _BEATLOCK_REF_PCTL)) if len(rel) else 0.0,
              _BEATLOCK_REF_FLOOR)

    db_conf, db_flag = 1.0, False
    if use_downbeat:
        db_conf, db_flag = _resolve_downbeat_confidence(audio_path, trk)

    bl = BeatLock(bool(trk.octave_locked), ref,
                  np.asarray(trk.reliability_times, float), rel,
                  db_confidence=db_conf, db_flagged=db_flag, db_apply=use_downbeat)
    np.savez(cache, octave_locked=trk.octave_locked, ref_level=ref,
             rel_times=bl._rel_times, rel=bl._rel,
             db_confidence=db_conf, db_flagged=db_flag, db_apply=use_downbeat)
    return bl


def _cqt_chroma(audio_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Raw librosa CQT chroma (idx0==C) + frame times — identical convention to
    the downbeat model's validation feed (``brick0_propose.load_chroma_frames``);
    the PRIMARY harmonic-rhythm downbeat term."""
    import librosa
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    ch = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
    times = librosa.frames_to_time(np.arange(ch.shape[1]), sr=sr, hop_length=512)
    return ch.T, times


def _resolve_downbeat_confidence(audio_path: Path, trk) -> tuple[float, bool]:
    """Fold-input: run the global-phase downbeat resolver on this song and return
    its ``(confidence, flagged)``. READ-ONLY use of ``harmonia.align`` (never
    edited). Any failure (missing dep / degenerate audio) falls back to
    ``(1.0, False)`` == abstain, so the harvest never crashes on the fold."""
    try:
        from harmonia.align.bass_salience import bass_chroma
        from harmonia.align.downbeat import resolve_downbeat
        chroma = _cqt_chroma(audio_path)
        bass = bass_chroma(str(audio_path))
        res = resolve_downbeat(trk, chart_alignment=None, chroma=chroma, bass=bass)
        return float(res.confidence), bool(res.flagged)
    except Exception as e:                                    # pragma: no cover
        log.warning("downbeat fold skipped for %s (%s); abstaining", audio_path, e)
        return 1.0, False


# ═════════════════════════════════════════════════════════════════════════════
# Stage 3: per-segment signals from the proposal + beat-lock
# ═════════════════════════════════════════════════════════════════════════════

def _agr_by_t0(proposal: dict) -> dict[float, float | None]:
    ad = proposal.get("agreement_detail", {})
    return {round(float(c["t0"]), 2): c.get("agr") for c in ad.get("per_chord", [])}


def _transpose_margin(proposal: dict) -> float:
    tr = proposal.get("agreement_detail", {}).get("per_transpose", [])
    if len(tr) >= 2:
        return float(tr[0][1]) - float(tr[1][1])
    return float(tr[0][1]) if tr else 0.0


def _section_of(t_mid: float, sections: list[dict]) -> dict | None:
    for s in sections:
        if s["t0"] - 1e-6 <= t_mid <= s["t1"] + 1e-6:
            return s
    return None


def _xrep_index(proposal: dict) -> dict[str, dict]:
    """Map section label -> its cross-rep report (first match)."""
    idx: dict[str, dict] = {}
    for r in proposal.get("refinement", {}).get("cross_repetition", []):
        idx.setdefault(r["label"], r)
    return idx


def _split_index(proposal: dict) -> list[dict]:
    return proposal.get("georgia_bundle", {}).get("split_detector_fires", [])


def build_segment_signals(proposal: dict, beat_lock: BeatLock
                          ) -> list[tuple[dict, SegmentSignals]]:
    """Join the proposal + beat-lock into per-chord (chord_dict, SegmentSignals).

    Returns one entry per emitted GT chord (each is a candidate segment).
    """
    gt_chords = proposal.get("gt_chords", [])
    prop = proposal["proposal"] if "proposal" in proposal else proposal
    agr_map = _agr_by_t0(prop)
    region = prop.get("harmonic_agreement", {})
    ceiling = max(float(region.get("ceiling") or 0.0), 1e-3)
    frac_low = float(region.get("frac_low") or 0.0)
    tmargin = _transpose_margin(prop)
    sa = prop.get("section_alignment", {})
    sections = sa.get("sections", [])
    gaps = sa.get("gaps_unlabeled", [])
    xidx = _xrep_index(prop)
    splits = _split_index(prop)

    # gap + song boundary times for near_gap
    gap_edges: list[float] = []
    for g in gaps:
        gap_edges += [float(g["t0"]), float(g["t1"])]

    out: list[tuple[dict, SegmentSignals]] = []
    for c in gt_chords:
        t0, t1 = float(c["t0"]), float(c["t1"])
        tmid = 0.5 * (t0 + t1)
        agr = agr_map.get(round(t0, 2))
        agr = float(agr) if agr is not None else -1.0

        sec = _section_of(tmid, sections)
        coverage_full = sec is not None
        near_gap = any(abs(t0 - e) < _GAP_TOL_S or abs(t1 - e) < _GAP_TOL_S
                       for e in gap_edges)

        # cross-rep consistency + divergence via the containing section's report
        xrep_consistent = True
        xrep_divergence = False
        if sec is not None:
            rep = xidx.get(sec["label"])
            if rep is not None:
                xrep_consistent = float(rep.get("var_after") or 0.0) <= _XREP_VAR_MAX
                for d in rep.get("divergences", []):
                    if d.get("label") == c["label"] and agr < _DIV_MEAN:
                        xrep_divergence = True
                        break

        # mid-span split match (secondary substitution signal)
        split_fire = False
        split_margin = 0.0
        sub_suggestion = None
        for f in splits:
            if (abs(float(f["t0"]) - t0) < 0.25 and abs(float(f["t1"]) - t1) < 0.25
                    and f.get("charted") == c["label"]):
                split_fire = True
                split_margin = float(f.get("second_agr", 0.0)) - float(f.get("charted_2nd", 0.0))
                sub_suggestion = f.get("second_chord")
                break

        sig = SegmentSignals(
            t0=t0, t1=t1, label=c["label"],
            agreement=agr, agreement_norm=float(np.clip(agr / ceiling, 0.0, 1.0)),
            beat_lock=beat_lock.span_lock(t0, t1),
            octave_locked=beat_lock.octave_locked,
            coverage_full=coverage_full, near_gap=near_gap,
            transpose_margin=tmargin,
            xrep_consistent=xrep_consistent, xrep_divergence=xrep_divergence,
            split_fire=split_fire, split_margin=split_margin,
            frac_low_song=frac_low,
            substitution_suggestion=sub_suggestion,
        )
        out.append((c, sig))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Stage 4: harvest -> rows
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class HarvestResult:
    song_id: str
    audio_path: str
    clean_rows: list[dict] = field(default_factory=list)
    review_rows: list[dict] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)         # {t0,t1,label,reason,confidence}
    stats: dict = field(default_factory=dict)


def harvest_song(song: dict, *, gate_cfg: GateConfig = GateConfig(),
                 source: str = "ireal+youtube", force: bool = False,
                 proposal: Optional[dict] = None,
                 beat_lock: Optional[BeatLock] = None) -> HarvestResult:
    """Align + gate one song into clean / review / dropped segments.

    ``proposal`` / ``beat_lock`` may be pre-supplied (calibration reuses them);
    otherwise they are computed (and cached).
    """
    if proposal is None:
        proposal = run_aligner(song, force=force)
    prop = proposal["proposal"]
    beat_period = float(prop.get("beat_this", {}).get("beat_period") or 0.5)
    audio_rel = proposal.get("audio_path", song.get("audio", ""))
    audio_abs = audio_rel if Path(audio_rel).is_absolute() else str(REPO / audio_rel)
    if beat_lock is None:
        beat_lock = compute_beat_lock(Path(audio_abs), prior_period=beat_period,
                                      force=force)

    sigs = build_segment_signals(proposal, beat_lock)
    res = HarvestResult(song_id=song["song_id"], audio_path=audio_rel)

    # decide every segment; group contiguous CLEAN chords into rows
    decided: list[tuple[dict, SegmentSignals, GateDecision]] = [
        (c, s, gate_segment(s, gate_cfg)) for c, s in sigs]

    clean_run: list[tuple[dict, GateDecision]] = []

    def flush_clean():
        if not clean_run:
            return
        chords = [dict(t0=c["t0"], t1=c["t1"], label=c["label"],
                       root_pc=c.get("root_pc"), bass_pc=c.get("bass_pc"))
                  for c, _ in clean_run]
        conf = round(float(np.mean([d.confidence for _, d in clean_run])), 4)
        res.clean_rows.append(dict(
            audio_path=audio_rel, t0=chords[0]["t0"], t1=chords[-1]["t1"],
            chords=chords, confidence=conf, song_id=song["song_id"], source=source))
        clean_run.clear()

    for c, s, d in decided:
        if d.bucket is Bucket.CLEAN:
            clean_run.append((c, d))
            continue
        flush_clean()
        if d.bucket is Bucket.REVIEW:
            res.review_rows.append(dict(
                audio_path=audio_rel, t0=c["t0"], t1=c["t1"],
                chart_suggestion=c["label"],
                substitution_suggestion=s.substitution_suggestion,
                divergence_evidence=dict(
                    reasons=list(d.reasons), agreement=round(s.agreement, 3),
                    beat_lock=round(s.beat_lock, 3), split_margin=round(s.split_margin, 3),
                    xrep_divergence=s.xrep_divergence),
                confidence=d.confidence, song_id=song["song_id"], source=source,
                status="unlabeled"))
        else:
            res.dropped.append(dict(t0=c["t0"], t1=c["t1"], label=c["label"],
                                    reason=d.reasons[0] if d.reasons else "",
                                    confidence=d.confidence))
    flush_clean()

    total = sum(float(c["t1"]) - float(c["t0"]) for c, _ in sigs)
    clean_dur = sum(r["t1"] - r["t0"] for r in res.clean_rows)
    review_dur = sum(r["t1"] - r["t0"] for r in res.review_rows)
    n_clean_chords = sum(len(r["chords"]) for r in res.clean_rows)
    res.stats = dict(
        n_segments=len(sigs), n_clean_chords=n_clean_chords,
        n_clean_rows=len(res.clean_rows), n_review=len(res.review_rows),
        n_dropped=len(res.dropped),
        labeled_dur_s=round(total, 1), clean_dur_s=round(clean_dur, 1),
        review_dur_s=round(review_dur, 1),
        kept_fraction=round(clean_dur / total, 3) if total else 0.0)
    return res


# ═════════════════════════════════════════════════════════════════════════════
# Manifest writers
# ═════════════════════════════════════════════════════════════════════════════

def write_manifests(results: list[HarvestResult], *,
                    clean_path: Path = CLEAN_MANIFEST,
                    review_path: Path = REVIEW_MANIFEST) -> dict:
    """Write JSONL manifests (clean GT + review/substitution queue). Audio stays
    on disk; rows only reference it."""
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    with clean_path.open("w") as fh:
        for r in results:
            for row in r.clean_rows:
                fh.write(json.dumps(row) + "\n")
    with review_path.open("w") as fh:
        for r in results:
            for row in r.review_rows:
                fh.write(json.dumps(row) + "\n")
    n_pairs = sum(r.stats["n_clean_chords"] for r in results)
    minutes = sum(r.stats["clean_dur_s"] for r in results) / 60.0
    return dict(clean_manifest=str(clean_path), review_manifest=str(review_path),
                n_songs=len(results), n_clean_pairs=n_pairs,
                n_review=sum(len(r.review_rows) for r in results),
                clean_minutes=round(minutes, 2))


__all__ = ["run_aligner", "compute_beat_lock", "BeatLock", "build_segment_signals",
           "harvest_song", "HarvestResult", "write_manifests",
           "CLEAN_MANIFEST", "REVIEW_MANIFEST", "DATASET_DIR", "REPO"]
