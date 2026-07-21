"""Independent-GT beat / downbeat / octave-lock harness (Phase 2, rewrite plan).

WHY THIS EXISTS (rewrite_execution_plan_2026_07_21.md §1, the GT-provenance
trap): beat / downbeat / bar-grid accuracy is measured **only** against
INDEPENDENT ground truth — never against `aligned_corpus`, whose per-chord
timestamps ARE the alignment output (scoring alignment against alignment is
circular). The independent GT used here is:

  * **POP909** `beat_midi.txt` — column 1 = beat times (s), column 3 = downbeat
    flag. This is exact GT for the MIDI, and we render that same MIDI to audio,
    so the rendered-audio beat timing == the GT beat timing by construction.
    Both a **beat GT** and a **downbeat GT**, for 909 songs. (CLAUDE.md: prefer
    `is_downbeat` / `downbeat_times` over any audio-only downbeat detector.)
  * RWC-AIST beats would be the other independent source, but **no RWC beat
    annotations are present on disk** (`data/cache/rwc/` has audio + feature
    caches only, no `.BEAT` files), so this harness uses POP909, which supplies
    both beats and downbeats for hundreds of songs.

What it measures on the CURRENT pipeline beat path (chord_pipeline_v1, the live
`beat_backend="beatthis"` default, with `librosa` scored alongside for the
octave-lock reconciliation):

  * **beat F-measure** (mir_eval, ±70 ms — the MIREX standard tolerance; first
    5 s trimmed per MIREX convention).
  * **native downbeat F-measure** (same matcher, ±70 ms) — Beat This!'s raw
    native downbeats. This is an INPUT-QUALITY UPPER BOUND (~0.751): the
    pipeline computes these but DISCARDS them (`chord_pipeline_v1`
    `beatthis_downbeats` at ~L4241 is dead code, never read). Kept as the
    headroom ceiling, NOT the shipped output.
  * **pipeline downbeat F-measure** (the CANONICAL shipped metric, ~0.29) —
    scores what the live path ACTUALLY produces: a rigid single-phase bar grid.
    The pipeline collapses the native downbeats to ONE circular-mean bar phase
    over a uniform grid (`bar_period = 4*period`) via
    `sota_downbeat_phase` → `_flux_downbeat_phase` → `_structure_anchor_phase`
    (`downbeat_anchor.py` / `beat_grid.py`, chosen at `chord_pipeline_v1`
    ~L3777-3812). This metric reproduces that single-phase choice and scores it
    on the real (drift-free) tracked beat grid — valid because the display layer
    snaps bars to real beats. Ceiling if the BEST of the 4 phases were picked is
    reported alongside (~0.53) — the gap 0.29→0.53 is intra-phase headroom, and
    0.53→0.751 is the further headroom a native-per-bar (non-collapsed) grid
    could reach. See PHASE 2 STEP 3/4 in docs/known_issues.md.
  * **octave-error rate** — fraction of songs whose detected tempo is ~2x or
    ~0.5x the GT tempo (the "octave-lock" the 2026-07-19 audit flagged). NOTE
    (STEP 4): these errors are BIDIRECTIONAL — both 2x-fast and 0.5x-slow occur.

This module is MEASUREMENT ONLY. It does not modify the pipeline. Each fix is a
separate, orchestrator-dispatched change.

Usage:
    python -m harmonia.eval.beat_alignment_gt --selftest        # no audio, unit-checks the metrics
    python -m harmonia.eval.beat_alignment_gt --song 002 --backend both
    python -m harmonia.eval.beat_alignment_gt --n 24 --backend both   # baseline table
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
POP909_DIR = REPO / "data" / "pop909" / "POP909"
SOUNDFONT = REPO / "data" / "soundfonts" / "MuseScore_General.sf2"
RENDER_SR = 22050  # mono render; enough for beat tracking, small on disk

BEAT_TOL_S = 0.07       # ±70 ms — MIREX standard beat tolerance
MIREX_TRIM_S = 5.0      # MIREX convention: ignore the first 5 s (count-in)


# --------------------------------------------------------------------------- #
# Ground truth (POP909 beat_midi.txt)
# --------------------------------------------------------------------------- #
def load_pop909_gt(song_id: str) -> "tuple[np.ndarray, np.ndarray, float]":
    """(beat_times, downbeat_times, gt_tempo_bpm) from beat_midi.txt.

    beat_midi.txt: col1 = beat time (s), col2 = half-bar flag, col3 = downbeat
    flag. gt_tempo = 60 / median(inter-beat-interval).
    """
    p = POP909_DIR / song_id / "beat_midi.txt"
    times: list[float] = []
    downs: list[bool] = []
    for line in p.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            t = float(parts[0])
        except ValueError:
            continue
        times.append(t)
        downs.append(len(parts) >= 3 and float(parts[2]) >= 0.5)
    beats = np.asarray(times, dtype=float)
    is_db = np.asarray(downs, dtype=bool)
    gt_tempo = 60.0 / float(np.median(np.diff(beats)))
    return beats, beats[is_db], gt_tempo


def all_gt_tempos() -> "dict[str, float]":
    """GT tempo for every POP909 song with a usable beat file (no audio)."""
    out: dict[str, float] = {}
    for d in sorted(POP909_DIR.iterdir()):
        if not (d.is_dir() and d.name.isdigit()):
            continue
        try:
            beats, _, tempo = load_pop909_gt(d.name)
        except Exception:
            continue
        if len(beats) >= 8:
            out[d.name] = tempo
    return out


def stratified_sample(n: int, force=("002",)) -> list[str]:
    """Deterministic tempo-stratified sample of POP909 song ids.

    Sort by GT tempo, take every (N//n)-th so slow/octave-prone ballads and
    fast songs are both represented; always include `force` ids (002 = the
    CLAUDE.md octave-lock calibration anchor).
    """
    tempos = all_gt_tempos()
    ordered = sorted(tempos, key=lambda s: tempos[s])
    step = max(1, len(ordered) // n)
    picked = ordered[::step][:n]
    for f in force:
        if f not in picked and f in tempos:
            picked.append(f)
    # keep sorted-by-tempo order, unique
    return sorted(set(picked), key=lambda s: tempos[s])


# --------------------------------------------------------------------------- #
# Render + beat detection (replicates the live chord_pipeline_v1 beat path)
# --------------------------------------------------------------------------- #
def render_midi(song_id: str, out_wav: Path) -> Path:
    """Render POP909 MIDI to a wav via fluidsynth + MuseScore_General.sf2."""
    mid = POP909_DIR / song_id / f"{song_id}.mid"
    if not mid.exists():
        raise FileNotFoundError(mid)
    cmd = [
        "fluidsynth", "-ni", "-g", "0.8",
        "-F", str(out_wav), "-r", str(RENDER_SR),
        str(SOUNDFONT), str(mid),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_wav


_DBN_F2B = None  # harness-local File2Beats(dbn=True); NEVER the pipeline's cache


def _get_beatthis_dbn():
    """SCREEN-ONLY Beat This! with the DBN postprocessor (dbn=True).

    Kept entirely inside the harness — it does NOT touch the pipeline's
    `_get_beatthis()` (which stays dbn=False for the live default). Beat This!'s
    DBN postprocessor imports madmom, which is broken on this Python 3.12
    (`MutableSequence` moved to collections.abc); the repo's own compat shim
    (`rhythm._ensure_madmom_compat`) restores it. Applied here only when a
    dbn=True run is requested.
    """
    global _DBN_F2B
    if _DBN_F2B is not None:
        return _DBN_F2B
    from harmonia.models.rhythm import _ensure_madmom_compat, _patch_madmom_downbeat_argmax
    _ensure_madmom_compat()
    try:
        _patch_madmom_downbeat_argmax()
    except Exception:  # noqa: BLE001 — best-effort second shim
        pass
    from beat_this.inference import File2Beats
    _DBN_F2B = File2Beats(device="cpu", dbn=True)
    return _DBN_F2B


def detect_beatthis(wav: Path, dbn: bool = False) -> "tuple[np.ndarray, np.ndarray, float]":
    """Beat This! → (beat_times, downbeats, tempo_bpm).

    dbn=False mirrors the LIVE default (chord_pipeline_v1.infer_chords_v1
    beat_backend='beatthis', lines ~4224-4243, via _get_beatthis()). dbn=True is
    a SCREEN-ONLY variant (harness-local instance) to A/B the DBN postprocessor.
    tempo = 60/median(diff(beats)).
    """
    if dbn:
        f2b = _get_beatthis_dbn()
    else:
        from harmonia.models.chord_pipeline_v1 import _get_beatthis
        f2b = _get_beatthis()
    if f2b is None:
        raise RuntimeError("Beat This! unavailable")
    bts, dbs = f2b(str(wav))
    bts = np.asarray(bts, dtype=float)
    dbs = np.asarray(dbs, dtype=float)
    tempo = 60.0 / float(np.median(np.diff(bts))) if len(bts) >= 2 else 0.0
    return bts, dbs, tempo


def detect_librosa(wav: Path) -> "tuple[np.ndarray, np.ndarray, float]":
    """Legacy/fallback path: librosa.beat.beat_track (no native downbeats)."""
    import librosa
    import soundfile as sf
    y, sr = sf.read(str(wav))
    y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
    tempo_arr, frames = librosa.beat.beat_track(y=y, sr=sr)
    bts = librosa.frames_to_time(frames, sr=sr)
    tempo = float(np.atleast_1d(tempo_arr)[0])
    return bts, np.asarray([]), tempo


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def beat_f(ref: np.ndarray, est: np.ndarray, tol: float = BEAT_TOL_S,
           trim: bool = True) -> float:
    """mir_eval beat F-measure (±tol s). MIREX trims the first 5 s."""
    import mir_eval.beat as B
    if len(ref) == 0 or len(est) == 0:
        return 0.0
    r = B.trim_beats(np.sort(ref), MIREX_TRIM_S) if trim else np.sort(ref)
    e = B.trim_beats(np.sort(est), MIREX_TRIM_S) if trim else np.sort(est)
    if len(r) == 0 or len(e) == 0:
        return 0.0
    return float(B.f_measure(r, e, f_measure_threshold=tol))


def octave_class(est_tempo: float, gt_tempo: float) -> str:
    """'ok' | 'octave' | 'other'.  'octave' == detected ~2x or ~0.5x GT.

    Ratio bands (±~15% around the octave / unison points):
      ok      : 0.87 <= r <= 1.15
      octave  : 1.70 <= r <= 2.30   or   0.435 <= r <= 0.585
      other   : anything else (a non-octave tempo error)
    """
    if gt_tempo <= 0 or est_tempo <= 0:
        return "other"
    r = est_tempo / gt_tempo
    if 0.87 <= r <= 1.15:
        return "ok"
    if (1.70 <= r <= 2.30) or (0.435 <= r <= 0.585):
        return "octave"
    return "other"


# --------------------------------------------------------------------------- #
# Pipeline-REAL downbeat metric (the shipped rigid single-phase bar grid)
# --------------------------------------------------------------------------- #
# WHY (docs/known_issues.md PHASE 2 STEP 3 — the measurement trap): the native
# `downbeat_f` above scores Beat This!'s raw downbeats, which the pipeline NEVER
# ships (`beatthis_downbeats` is dead code). The pipeline collapses them to a
# single circular-mean bar phase over a rigid uniform grid. The two functions
# below reproduce that phase choice (faithful to `chord_pipeline_v1` ~L3777-3812:
# sota if confident → flux → structure tie-break when flux comb weak) and score
# it drift-free on the real tracked beat grid. Mirrors the Bug-2 subagent's
# gate_phase.py `old_phase`/`abs_to_realidx`; calibrated to reproduce its
# gate_31.json (mean pipeline dbF 0.292, ceiling 0.534).

def _sota_phase(downbeats: np.ndarray, conf: float,
                bar_period: float, beat: float) -> "int | None":
    """Pipeline's `sota_downbeat_phase`: circular mean of native downbeats mod
    bar_period → one integer beat phase in [0,4). Abstains (None) below the
    live `min_confidence=0.85` / `<5 downbeats` gate (downbeat_anchor.py:105)."""
    if len(downbeats) < 5 or conf < 0.85:
        return None
    ang = 2 * np.pi * (downbeats % bar_period) / bar_period
    phase_s = float((np.angle(np.mean(np.exp(1j * ang))) % (2 * np.pi))
                    * bar_period / (2 * np.pi))
    return int(round(phase_s / beat)) % 4


def _abs_phase_to_real_idx(phi_beats: int, bts: np.ndarray, beat: float,
                           bar_period: float, t_end: float) -> int:
    """Map the pipeline's abs-time bar phase (phi in beats) to the real-beat
    subsample index p ∈ [0,4) whose bts[p::4] best matches the uniform bar grid.
    Drift-free: the display layer snaps bars to real beats, so scoring the real
    beat subsample at the chosen phase is faithful to what ships."""
    g = np.arange(phi_beats * beat, t_end + bar_period, bar_period)
    return int(np.argmax([beat_f(bts[p::4], g) for p in range(4)]))


def pipeline_downbeat_scores(
    wav: Path, bts: np.ndarray, det_tempo: float, gt_downs: np.ndarray,
) -> "tuple[float | None, float | None]":
    """(pipeline_downbeat_f, ceiling_f) — the SHIPPED single-phase bar grid vs GT.

    pipeline_downbeat_f: dbF at the phase the live pipeline actually picks.
    ceiling_f:           dbF if the best of the 4 real-beat phases were picked
                         (intra-phase headroom the collapse throws away).
    Returns (None, None) when the beat grid is too short to grid. Faithful to
    chord_pipeline_v1's sota→flux→structure selection; structure is only
    computed when it would matter (flux comb ratio < 1.05), which is
    result-identical to the pipeline.
    """
    if len(bts) < 8 or len(gt_downs) == 0 or det_tempo <= 0:
        return None, None
    from harmonia.models.downbeat_anchor import beat_this_downbeats
    from harmonia.models.beat_grid import (
        flux_downbeat_phase, structure_anchor_phase)
    from harmonia.models import nnls_features as nf
    from harmonia.models.chord_pipeline_v1 import _note_name_to_pc
    from harmonia.theory.key_profiles import infer_key

    period = 60.0 / det_tempo
    bar_period = 4.0 * period
    beat = bar_period / 4.0
    t_end = float(bts[-1])

    downbeats, conf = beat_this_downbeats(wav)
    phi_sota = _sota_phase(np.asarray(downbeats, dtype=float), conf, bar_period, beat)

    arr, times = nf.extract_bothchroma(wav)
    phi_flux, ratio_flux = flux_downbeat_phase(arr, times, bar_period, audio_path=wav)

    if phi_sota is not None:
        phi = phi_sota
    else:
        phi = phi_flux
        if ratio_flux < 1.05:  # weak comb → structure-crispness tie-break
            heads = nf.get_heads()
            feat = nf.pool_beats(arr, times, bts)
            beat_proba = heads.root_proba(feat)
            try:
                tonic_pc = _note_name_to_pc(
                    infer_key(feat[:, 12:].sum(0)).key_name.split()[0])
            except Exception:  # noqa: BLE001
                tonic_pc = None
            phi, _ = structure_anchor_phase(beat_proba, tonic_pc=tonic_pc)

    realbeat = {p: beat_f(gt_downs, bts[p::4]) for p in range(4)}
    p_chosen = _abs_phase_to_real_idx(phi, bts, beat, bar_period, t_end)
    return realbeat[p_chosen], max(realbeat.values())


@dataclass
class SongResult:
    song_id: str
    gt_tempo: float
    backend: str
    det_tempo: float
    octave: str
    beat_f: float
    downbeat_f: float | None          # native (Beat This! raw) — input-quality ceiling
    pipeline_downbeat_f: float | None  # CANONICAL: the shipped single-phase grid
    pipeline_downbeat_ceil: float | None  # best-of-4-phases headroom ceiling
    n_gt_beats: int
    n_gt_downbeats: int
    n_det_beats: int


def run_song(song_id: str, backend: str, wav: Path) -> SongResult:
    gt_beats, gt_downs, gt_tempo = load_pop909_gt(song_id)
    pipe_db: float | None = None
    pipe_ceil: float | None = None
    if backend.startswith("beatthis"):
        est_beats, est_downs, det_tempo = detect_beatthis(wav, dbn=backend.endswith("dbn"))
        db_f: float | None = beat_f(gt_downs, est_downs) if len(gt_downs) else None
        # CANONICAL shipped metric — only for the LIVE default (beatthis, dbn=False);
        # the dbn screen variant ships nothing.
        if backend == "beatthis" and len(gt_downs):
            pipe_db, pipe_ceil = pipeline_downbeat_scores(
                wav, est_beats, det_tempo, gt_downs)
    else:
        est_beats, est_downs, det_tempo = detect_librosa(wav)
        db_f = None  # librosa has no downbeats
    return SongResult(
        song_id=song_id, gt_tempo=round(gt_tempo, 1), backend=backend,
        det_tempo=round(det_tempo, 1), octave=octave_class(det_tempo, gt_tempo),
        beat_f=round(beat_f(gt_beats, est_beats), 4),
        downbeat_f=None if db_f is None else round(db_f, 4),
        pipeline_downbeat_f=None if pipe_db is None else round(pipe_db, 4),
        pipeline_downbeat_ceil=None if pipe_ceil is None else round(pipe_ceil, 4),
        n_gt_beats=len(gt_beats), n_gt_downbeats=len(gt_downs),
        n_det_beats=len(est_beats),
    )


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=24, help="stratified sample size")
    ap.add_argument("--song", default=None, help="single POP909 id, e.g. 002")
    ap.add_argument("--backend",
                    choices=["beatthis", "beatthis-dbn", "librosa", "both", "ab"],
                    default="both",
                    help="'both'=beatthis+librosa; 'ab'=beatthis(dbn=False)+beatthis-dbn")
    ap.add_argument("--selftest", action="store_true",
                    help="unit-check metrics on synthetic data (no audio)")
    ap.add_argument("--out", default=None, help="write per-song JSON here")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    if args.backend == "both":
        backends = ["beatthis", "librosa"]
    elif args.backend == "ab":
        backends = ["beatthis", "beatthis-dbn"]
    else:
        backends = [args.backend]
    song_ids = [args.song] if args.song else stratified_sample(args.n)
    scratch = Path("/private/tmp/beat_gt_wav"); scratch.mkdir(exist_ok=True)

    results: list[SongResult] = []
    skipped: list[str] = []
    for sid in song_ids:
        wav = scratch / f"{sid}.wav"
        try:
            render_midi(sid, wav)
        except Exception as exc:
            print(f"[skip] {sid}: render failed: {exc}", file=sys.stderr)
            skipped.append(sid)
            continue
        for be in backends:
            try:
                r = run_song(sid, be, wav)
                results.append(r)
                _pdb = ("--" if r.pipeline_downbeat_f is None
                        else f"{r.pipeline_downbeat_f:.3f}")
                _pceil = ("--" if r.pipeline_downbeat_ceil is None
                          else f"{r.pipeline_downbeat_ceil:.3f}")
                print(f"{sid} gt={r.gt_tempo:6.1f} {be:8s} det={r.det_tempo:6.1f} "
                      f"{r.octave:6s} beatF={r.beat_f:.3f} "
                      f"native_dbF={'--' if r.downbeat_f is None else f'{r.downbeat_f:.3f}'} "
                      f"pipe_dbF={_pdb} (ceil {_pceil})")
            except Exception as exc:
                print(f"[skip] {sid}/{be}: {exc}", file=sys.stderr)
        # delete wav immediately (disk discipline — verify path first)
        if wav.exists() and wav.parent == scratch and wav.suffix == ".wav":
            wav.unlink()

    _summary(results, skipped)
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"results": [asdict(r) for r in results], "skipped": skipped}, indent=2))
        print(f"\nwrote {args.out}")
    return 0


def _summary(results: list[SongResult], skipped: list[str]) -> None:
    print("\n" + "=" * 64)
    for be in sorted({r.backend for r in results}):
        rs = [r for r in results if r.backend == be]
        if not rs:
            continue
        bf = np.mean([r.beat_f for r in rs])
        dbs = [r.downbeat_f for r in rs if r.downbeat_f is not None]
        pdbs = [r.pipeline_downbeat_f for r in rs if r.pipeline_downbeat_f is not None]
        pceils = [r.pipeline_downbeat_ceil for r in rs if r.pipeline_downbeat_ceil is not None]
        oct_rate = np.mean([r.octave == "octave" for r in rs])
        other_rate = np.mean([r.octave == "other" for r in rs])
        ok_rate = np.mean([r.octave == "ok" for r in rs])
        print(f"[{be}] N={len(rs)}  beatF={bf:.3f}  "
              f"tempo-ok={ok_rate:.2f} octave-lock={oct_rate:.2f} other={other_rate:.2f}")
        # CANONICAL shipped downbeat metric next to the native input-quality ceiling
        print(f"       pipeline_downbeatF={'n/a' if not pdbs else f'{np.mean(pdbs):.3f}'} "
              f"(SHIPPED, canonical)   "
              f"best-phase ceiling={'n/a' if not pceils else f'{np.mean(pceils):.3f}'}   "
              f"native_downbeatF={'n/a' if not dbs else f'{np.mean(dbs):.3f}'} "
              f"(input-quality upper bound; DISCARDED by pipeline)")
        octs = [r.song_id for r in rs if r.octave == "octave"]
        if octs:
            print(f"       octave-lock (bidirectional) songs: {octs}")
    if skipped:
        print(f"skipped (render fail): {skipped}")


def _selftest() -> int:
    """No-audio unit checks of the metrics (CLAUDE.md #1: pin the load-bearing
    assumptions against an external reference)."""
    ok = True

    # 1. octave classifier fires on the CLAUDE.md song-002 scenario and its
    #    inverse, and does NOT fire on a correct match.
    cases = [
        (128.0, 64.0, "octave"),   # tracker locks 2x fast
        (63.0, 129.0, "octave"),   # the literal CLAUDE.md "63 vs 129" pair
        (64.0, 64.0, "ok"),
        (74.0, 70.0, "ok"),
        (95.0, 64.0, "other"),     # ~1.5x — a real (non-octave) error
    ]
    for est, gt, want in cases:
        got = octave_class(est, gt)
        flag = "OK " if got == want else "XX "
        if got != want:
            ok = False
        print(f"{flag}octave_class(est={est}, gt={gt}) = {got}  (want {want})")

    # 2. beat_f: perfect match == 1.0; a half-tempo estimate (every other beat)
    #    scores well below 1 (fewer matched beats) — sanity that the matcher
    #    isn't trivially saturating.
    ref = np.arange(0, 60, 0.5)             # 120 bpm, 60 s
    perfect = beat_f(ref, ref.copy())
    half = beat_f(ref, ref[::2])            # only downbeat-rate beats present
    print(f"{'OK ' if abs(perfect-1.0)<1e-6 else 'XX '}beat_f(perfect)={perfect:.3f} (want 1.000)")
    print(f"{'OK ' if half<0.8 else 'XX '}beat_f(half-rate est)={half:.3f} (want <0.8)")
    if abs(perfect - 1.0) > 1e-6 or half >= 0.8:
        ok = False

    # 2b. pipeline-downbeat drift-free subsample (CLAUDE.md #1: pin the
    #     load-bearing math with no audio). A 120-bpm beat grid whose GT
    #     downbeats sit on beat-index 1 of every 4 must score ~1.0 at phase 1
    #     and near-0 at the other phases; the abs→real-idx mapper must recover 1.
    bts = np.arange(0.0, 60.0, 0.5)
    gt_db = bts[1::4]
    sub = {p: beat_f(gt_db, bts[p::4]) for p in range(4)}
    period = 0.5
    picked = _abs_phase_to_real_idx(1, bts, period, 4 * period, float(bts[-1]))
    pdb_ok = (sub[1] > 0.99 and max(sub[0], sub[2], sub[3]) < 0.2 and picked == 1)
    print(f"{'OK ' if pdb_ok else 'XX '}pipeline-dbF subsample: phase1={sub[1]:.3f} "
          f"others<={max(sub[0], sub[2], sub[3]):.3f} mapper->{picked} (want ~1.0/<0.2/1)")
    ok = ok and pdb_ok

    # 3. GT load: song 002 tempo is ~64 (three POP909 annotations agree — NOT
    #    129; see the harness header). This pins that our GT octave is right.
    try:
        _, downs, tempo = load_pop909_gt("002")
        good = 60 <= tempo <= 68 and len(downs) > 10
        print(f"{'OK ' if good else 'XX '}pop909 002 gt_tempo={tempo:.1f} "
              f"n_downbeats={len(downs)} (want ~64, >10 downbeats)")
        ok = ok and good
    except Exception as exc:
        print(f"XX pop909 002 gt load failed: {exc}")
        ok = False

    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
