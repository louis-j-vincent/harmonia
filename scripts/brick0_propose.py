"""Brick 0 — per-song frozen-GT *PROPOSAL* builder (batch 1).

This is the PROPOSAL half of Brick 0 (the scorer/schema half is
`harmonia/eval/accuracy_score.py` + `golden/brick0/SCHEMA.md`, commit 759643d).
It emits, per song, a `golden/brick0/<song>.gt.json` with **`verified=false`**
plus a self-contained ear-verification HTML under `docs/brick0_review/`. It
NEVER freezes/verifies — a human (Louis) hand-verifies transpose/form/bar-1
anchor by ear and flips `verified` afterwards.

THE NON-CIRCULARITY CONTRACT (docs/known_issues.md STEP 11 / BRICK 0 GREENLIT,
golden/brick0/SCHEMA.md). No GT field is ever derived from the model's own
chord predictions:

  * Chord labels        <- the iReal chart ONLY (trusted; keep `/bass` -> the
                           SOUNDING bass via `corpus_schema.sounding_bass_pc`).
  * Beat/downbeat grid  <- an INDEPENDENT Beat This! pass on the audio
                           (`downbeat_anchor.beat_this_downbeats`), NOT the
                           production chord-decode's grid.
  * Transpose proposal  <- raw-audio librosa chroma vs the chart's own
                           chord-tone profile (acoustic), NOT model chords.
  * Form proposal       <- the chart's written form tiled to fill the audio.
  * Bar-1 anchor        <- the independent Beat This! downbeats (phase is a
                           GUESS — Beat This! mis-places downbeats, EAR NEEDED).

Every proposed field carries {confidence in [0,1], top alternative, EVIDENCE}
with real numbers from the run (chroma-cosine margins, downbeat regularity,
duration-fit ratio). The per-song aggregate confidence is the MINIMUM of the
three machine-guessed fields (transpose/form/anchor) — weakest-link, since one
wrong field breaks the GT — and the verification queue is sorted ascending by
that aggregate so Louis spends ear-time on the ambiguous songs first.

Run:  .venv/bin/python scripts/brick0_propose.py
"""
from __future__ import annotations

import base64
import contextlib
import io
import json
import logging
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Make `harmonia` importable when run as `python scripts/brick0_propose.py`
# (sys.path[0] would otherwise be scripts/, not the repo root).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("brick0_propose")

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "golden" / "brick0"
REVIEW = REPO / "docs" / "brick0_review"
MANIFEST = REPO / "data" / "real_audio_benchmark" / "brick0_batch1.json"

NOTE_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# ── Batch 1: on-disk audio (docs/audio/*.m4a) x iReal chart (data/ireal/*.txt).
# 5 jazz + 3 pop = 62.5% jazz (STEP 11 named autumn_leaves, blue_bossa x2,
# ray_charles_georgia, bein_green). NO downloads this batch.
BATCH1 = [
    dict(song_id="autumn_leaves", title="Autumn Leaves",
         audio="docs/audio/autumn_leaves.m4a",
         ireal_file="jazz1460", tune_title="Autumn Leaves"),
    dict(song_id="blue_bossa", title="Blue Bossa",
         audio="docs/audio/blue_bossa.m4a",
         ireal_file="jazz1460", tune_title="Blue Bossa"),
    dict(song_id="blue_bossa_backing", title="Blue Bossa (150bpm backing track)",
         audio="docs/audio/blue_bossa_150bpm_backing_track.m4a",
         ireal_file="jazz1460", tune_title="Blue Bossa"),
    dict(song_id="georgia_on_my_mind", title="Georgia On My Mind (Ray Charles)",
         audio="docs/audio/ray_charles_georgia_on_my_mind_official_video.m4a",
         ireal_file="jazz1460", tune_title="Georgia On My Mind"),
    dict(song_id="bein_green", title="Bein' Green",
         audio="docs/audio/bein_green.m4a",
         ireal_file="jazz1460", tune_title="Bein' Green"),
    dict(song_id="close_to_you", title="Close To You (Carpenters)",
         audio="docs/audio/carpenters_close_to_you.m4a",
         ireal_file="pop400", tune_title="Close To You (They Long To Be)"),
    dict(song_id="stand_by_me", title="Stand By Me (Ben E. King)",
         audio="docs/audio/ben_e_king_stand_by_me_audio.m4a",
         ireal_file="pop400", tune_title="Stand By Me"),
    dict(song_id="every_breath_you_take", title="Every Breath You Take (The Police)",
         audio="docs/audio/the_police_every_breath_you_take_official_music_video.m4a",
         ireal_file="pop400", tune_title="Every Breath You Take"),
]

# ── iReal quality token -> shipped schema quality vocabulary ─────────────────
# The shipped pipeline emits `harmonia/models/musx_bass._MUSX_Q_TO_SEV` VALUES:
#   maj min 7 maj7 min7 dim dim7 hdim7 aug sus2 sus4 7sus4 9 min9 maj9 dom11 dom13
# GT tokens are chosen from that set so strict/sevenths comparisons are fair.
# Built after grepping the ACTUAL tokens in the 7 batch-1 charts (CLAUDE.md #1):
#   '' add9 -7 7 ^7 - -9 6 69 h7 7b9 7sus -6 7b13 7#5 9 h ^ -^7  (+ note-letter
#   slash basses; NO no-chord/repeat tokens present).
IREAL_Q_TO_SCHEMA: dict[str, str] = {
    "": "maj", "add9": "maj", "2": "maj", "5": "maj",
    "^": "maj7", "^7": "maj7", "^9": "maj9", "^13": "maj7",
    "6": "6", "69": "6",
    "-": "min", "-7": "min7", "-9": "min9", "-6": "min6", "-69": "min6",
    "-^7": "minmaj7", "-^9": "minmaj7",
    "h": "hdim7", "h7": "hdim7", "h9": "hdim7",
    "o": "dim", "o7": "dim7",
    "7": "7", "9": "9", "11": "11", "13": "13",
    "7b9": "7", "7#9": "7", "7b5": "7", "7#5": "7", "7b13": "7",
    "7#11": "7", "7alt": "7", "alt": "7", "at": "7", "7#9#5": "7", "7b9b5": "7",
    "+": "aug", "7+": "7",
    "sus": "sus4", "7sus": "7sus4", "9sus": "7sus4", "13sus": "7sus4",
}

# Chord-tone pitch-class sets (root-relative) for the acoustic transpose match.
QUALITY_INTERVALS: dict[str, list[int]] = {
    "maj": [0, 4, 7], "min": [0, 3, 7], "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11], "min7": [0, 3, 7, 10], "dim": [0, 3, 6],
    "dim7": [0, 3, 6, 9], "hdim7": [0, 3, 6, 10], "aug": [0, 4, 8],
    "sus2": [0, 2, 7], "sus4": [0, 5, 7], "7sus4": [0, 5, 7, 10],
    "6": [0, 4, 7, 9], "maj6": [0, 4, 7, 9], "min6": [0, 3, 7, 9],
    "9": [0, 4, 7, 10, 2], "maj9": [0, 4, 7, 11, 2], "min9": [0, 3, 7, 10, 2],
    "11": [0, 4, 7, 10, 2, 5], "13": [0, 4, 7, 10, 2, 9], "minmaj7": [0, 3, 7, 11],
}

_NOTE_TO_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
# root(1) + optional quality(2) + optional slash-bass(3), anchored at end.
_CHORD_RE = re.compile(r"([A-G][b#]?)([^A-G]*?)(?:/([A-G][b#]?))?$")


def note_to_pc(name: str) -> int:
    pc = _NOTE_TO_PC[name[0]]
    for c in name[1:]:
        if c == "#":
            pc += 1
        elif c == "b":
            pc -= 1
    return pc % 12


# ── chart parsing ────────────────────────────────────────────────────────────

@dataclass
class BarChord:
    beat_offset: int          # 0-indexed beat within the bar
    root_pc: int              # written-key root pc
    quality: str              # schema quality token
    bass_pc: int              # written-key sounding-bass pc
    bass_name: str | None     # note letter for a slash bass, else None
    raw: str                  # raw iReal token (for debugging)


@dataclass
class Chart:
    title: str
    key: str
    beats_per_bar: int
    tempo: int
    bars: list[tuple[str, list[BarChord]]]   # (section_label, [BarChord ...])
    unmapped: list[str] = field(default_factory=list)

    @property
    def n_bars(self) -> int:
        return len(self.bars)

    @property
    def section_runs(self) -> list[tuple[str, int]]:
        runs: list[list] = []
        for label, _ in self.bars:
            if not runs or runs[-1][0] != label:
                runs.append([label, 1])
            else:
                runs[-1][1] += 1
        return [(lab, n) for lab, n in runs]


def parse_token(tok: str) -> BarChord | None:
    """Parse one iReal token -> BarChord (written key), or None for no-chord."""
    tok = tok.strip()
    if tok in ("n", "N.C.", "p", "W", ""):
        return None
    tok = re.sub(r"N\d", "", tok)
    tok = re.sub(r"[UQSrl]+$", "", tok)
    m = _CHORD_RE.match(tok)
    if m is None:
        return BarChord(0, 0, "?" + tok, 0, None, tok)
    root, qual, bass = m.groups()
    qual = (qual or "").replace("W", "").strip()
    root_pc = note_to_pc(root)
    schema_q = IREAL_Q_TO_SCHEMA.get(qual)
    if schema_q is None:
        schema_q = "?" + qual
    if bass:
        bass_pc = note_to_pc(bass)
        bass_name = bass
    else:
        bass_pc = root_pc
        bass_name = None
    return BarChord(0, root_pc, schema_q, bass_pc, bass_name, tok)


def parse_chart(ireal_file: str, tune_title: str) -> Chart:
    """Parse a tune from a data/ireal/*.txt playlist into a Chart (written key).

    Reuses ireal_corpus.tune_to_mma for the (well-tested) section labelling +
    repeat expansion + within-bar beat distribution, then re-labels each slot's
    quality with the shipped-vocab mapping above.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):   # pyRealParser prints "Parsed <t>"
        from harmonia.data.ireal_corpus import load_playlist, tune_to_mma
        tunes = {t.title: t for t in load_playlist(REPO / "data" / "ireal" / f"{ireal_file}.txt")}
        tune = tunes[tune_title]
        mma = tune_to_mma(tune)

    bars: list[tuple[str, list[BarChord]]] = []
    unmapped: list[str] = []
    for _barno, label, slots in mma.timeline:
        chords: list[BarChord] = []
        for beat_offset, ireal_tok, _mma in slots:
            bc = parse_token(ireal_tok)
            if bc is None:
                continue
            bc.beat_offset = beat_offset
            if bc.quality.startswith("?"):
                unmapped.append(bc.quality[1:] or ireal_tok)
            chords.append(bc)
        bars.append((label, chords))
    return Chart(title=mma.title, key=mma.key, beats_per_bar=mma.beats_per_bar,
                 tempo=mma.tempo, bars=bars, unmapped=unmapped)


def chart_profile(chart: Chart) -> np.ndarray:
    """Duration-weighted chord-tone pitch-class histogram of the WRITTEN chart."""
    prof = np.zeros(12)
    bpb = chart.beats_per_bar
    for _label, chords in chart.bars:
        if not chords:
            continue
        offs = [c.beat_offset for c in chords] + [bpb]
        for i, c in enumerate(chords):
            dur = max(offs[i + 1] - offs[i], 1)
            ivs = QUALITY_INTERVALS.get(c.quality, [0, 4, 7])
            for iv in ivs:
                prof[(c.root_pc + iv) % 12] += dur
            prof[c.bass_pc % 12] += dur   # emphasise the sounding bass
    n = prof.sum()
    return prof / n if n else prof


# ── audio: Beat This! grid + librosa chroma ─────────────────────────────────

def decode_wav(audio: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / (audio.stem + ".wav")
    if not wav.exists():
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio),
                        "-ac", "1", "-ar", "22050", str(wav)], check=True)
    return wav


def audio_duration(audio: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(audio)], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def beat_this_full(wav: Path) -> dict:
    """INDEPENDENT Beat This! pass -> full beat grid + downbeats.

    Design (per downbeat_anchor.py's own framing: "the beat tracker works great,
    the problem is finding beat 1"): take the bar PERIOD and local bar placement
    from the reliable BEAT grid (imposing the chart's 4/4 meter = every 4th
    beat), and use the DOWNBEAT track only for the bar-1 PHASE guess (its known
    weak spot -> human-verified). Reuses the module's lazy model singleton +
    regularity metric so this is the same independent tracker, not the model
    chord-decode's grid.
    """
    from beat_this.inference import load_audio
    from harmonia.models import downbeat_anchor as da

    a2f, pp = da._get_beat_this()
    signal, sr = load_audio(str(wav))
    beat_logits, downbeat_logits = a2f(signal, sr)
    beats, downbeats = pp(beat_logits, downbeat_logits)
    beats = np.asarray(beats, dtype=float)
    downbeats = np.asarray(downbeats, dtype=float)
    beat_period = float(np.median(np.diff(beats))) if len(beats) > 1 else 0.5
    return dict(beats=beats, downbeats=downbeats, beat_period=beat_period,
                beat_regularity=da._regularity(beats),
                downbeat_regularity=da._regularity(downbeats))


def audio_chroma(wav: Path) -> np.ndarray:
    """Mean CQT chroma over the whole track (12-vec, L1-normalised). Independent
    of the harmonia model — plain librosa on the raw audio."""
    import librosa
    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    ch = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=2048)
    v = ch.mean(axis=1)
    # librosa chroma index 0 == C, matching our pc convention.
    s = v.sum()
    return v / s if s else v


# ── proposals (each returns value + confidence + alternative + evidence) ────

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


_NOTATED_KEY_PRIOR = 0.015   # cosine-scale bonus to t=0 (see note below)


def propose_transpose(chart: Chart, chroma: np.ndarray) -> dict:
    prof = chart_profile(chart)
    # score(t): chart transposed UP by t semitones vs the audio chroma (cosine).
    scores = np.array([
        float(np.dot(np.roll(prof, t), chroma) /
              ((np.linalg.norm(prof) * np.linalg.norm(chroma)) or 1.0))
        for t in range(12)
    ])
    # Fifth-related keys (t vs t+/-5, t+/-7) share 6/7 diatonic notes, so chroma
    # near-ties a key against its dominant. Charts are transcribed in a REAL
    # key, so break genuine near-ties toward the notated key (t=0). The prior is
    # tiny (0.015): it only flips ties, never a clear acoustic shift (verified:
    # Georgia margin 0.064 and Every Breath 0.036 survive it; the dead-tied
    # backing-track/Close-To-You flip to notated).
    adj = scores.copy()
    adj[0] += _NOTATED_KEY_PRIOR
    best = int(np.argmax(adj))
    raw_order = list(np.argsort(scores)[::-1])
    second = next(int(t) for t in raw_order if t != best)
    z = (scores - scores.mean()) / (scores.std() or 1.0)
    conf = float(_softmax(2.0 * z)[best])
    top2gap = float(scores[raw_order[0]] - scores[raw_order[1]])
    fifth_amb = (top2gap < 0.02 and
                 (raw_order[0] - raw_order[1]) % 12 in (5, 7))
    if best == 0 and raw_order[0] != 0 and scores[raw_order[0]] - scores[0] < _NOTATED_KEY_PRIOR:
        note = (f"chroma near-tie (t{raw_order[0]:+d}={scores[raw_order[0]]:.3f} "
                f"vs notated t+0={scores[0]:.3f}, gap {scores[raw_order[0]]-scores[0]:.3f}"
                f"{'; fifth-ambiguity' if fifth_amb else ''}) -> notated key preferred by prior")
        conf = min(conf, 0.45)
    else:
        note = (f"chroma-cos top={scores[best]:.3f} (t={best:+d}) vs "
                f"2nd={scores[second]:.3f} (t={second:+d}), gap {top2gap:.3f}"
                f"{'; FIFTH-AMBIGUITY (key vs dominant)' if fifth_amb else ''}")
    alts = [(int(t), round(float(scores[t]), 3)) for t in raw_order[:3]]
    return dict(
        transpose_semitones=best,
        confidence=round(conf, 3),
        alternative=dict(transpose=second, chroma_cos=round(float(scores[second]), 3)),
        evidence=f"{note}; chart key {chart.key}; top3(cos)={alts}; EAR NEEDED",
        _scores=[round(float(s), 4) for s in scores],
    )


def propose_anchor(downbeats: np.ndarray, regularity: float, bar_period: float) -> dict:
    if len(downbeats) == 0:
        return dict(bar1_anchor_time=0.0, confidence=0.05, alternative=None,
                    evidence="no downbeats detected")
    anchor = float(downbeats[0])
    alt = float(downbeats[1]) if len(downbeats) > 1 else anchor + bar_period
    # phase is Beat This!'s known weakness -> capped, always EAR NEEDED.
    conf = round(min(0.65, 0.30 + 0.40 * regularity), 3)
    return dict(
        bar1_anchor_time=round(anchor, 3),
        confidence=conf,
        alternative=dict(bar1_anchor_time=round(alt, 3),
                         note="next detected downbeat (covers a 1-bar count-in)"),
        evidence=(f"Beat This! 1st downbeat t={anchor:.2f}s; downbeat regularity "
                  f"{regularity:.2f} (n={len(downbeats)}); PHASE is a guess — "
                  f"Beat This! mis-places downbeats, EAR NEEDED"),
    )


def propose_form(chart: Chart, bar_period: float, anchor: float,
                 audio_dur: float, beat_regularity: float) -> dict:
    n_bars = chart.n_bars
    avail_bars = (audio_dur - anchor) / bar_period if bar_period > 0 else 0.0
    ratio = avail_bars / n_bars if n_bars else 0.0            # choruses that fit
    n_chor = max(1, int(round(ratio)))
    frac = abs(ratio - round(ratio))                          # distance to integer
    # steady beat grid (high beat_regularity) -> the bar count is trustworthy.
    conf = round(float(np.clip((1.0 - 2.0 * frac) * (0.55 + 0.45 * beat_regularity),
                               0.2, 0.95)), 3)
    alt_chor = max(1, (n_chor + 1) if ratio > n_chor else (n_chor - 1))
    runs = " ".join(f"{lab}{n}" for lab, n in chart.section_runs)
    return dict(
        n_choruses=n_chor,
        bars_per_chorus=n_bars,
        section_order=[lab for lab, _ in chart.section_runs],
        repeat_counts={lab: sum(n for l2, n in chart.section_runs if l2 == lab)
                       for lab, _ in chart.section_runs},
        one_chorus_form=runs,
        intro_bars=0,
        outro_bars=0,
        confidence=conf,
        alternative=dict(n_choruses=alt_chor),
        evidence=(f"(dur {audio_dur:.0f}s - anchor {anchor:.1f}s)/bar {bar_period:.2f}s "
                  f"= {avail_bars:.1f} bars / {n_bars} per chorus = {ratio:.2f} choruses; "
                  f"nearest int {n_chor}, frac-to-int {frac:.2f}; intro/outro=0 (GUESS, "
                  f"repeat count + intro need the ear)"),
    )


# ── GT timeline construction ─────────────────────────────────────────────────

def _beat_time(beats: np.ndarray, a0: int, k: int, beat_period: float,
               anchor: float) -> float:
    """Time of the (a0+k)-th beat, extrapolating past the tracked grid."""
    j = a0 + k
    if 0 <= j < len(beats):
        return float(beats[j])
    if len(beats):
        return float(beats[-1]) + (j - (len(beats) - 1)) * beat_period
    return anchor + k * beat_period


def build_gt_chords(chart: Chart, n_chor: int, beats: np.ndarray, anchor: float,
                    beat_period: float, transpose: int, audio_dur: float) -> list[dict]:
    """Tile the chart form n_chor times and place each bar/beat on the audio
    clock via the reliable Beat This! BEAT grid (chart 4/4 meter = every 4th
    beat from the anchor beat), applying the proposed transpose. The anchor's
    phase is the human-verified guess. Adjacent identical chords are merged."""
    from harmonia.data.corpus_schema import sounding_bass_pc

    bpb = chart.beats_per_bar
    a0 = int(np.argmin(np.abs(beats - anchor))) if len(beats) else 0

    raw: list[dict] = []
    for k in range(n_chor):
        for bi, (_label, chords) in enumerate(chart.bars):
            gidx = k * chart.n_bars + bi           # global bar index from anchor
            b0 = _beat_time(beats, a0, gidx * bpb, beat_period, anchor)
            if b0 >= audio_dur:
                break
            if not chords:
                continue
            offs = [c.beat_offset for c in chords] + [bpb]
            for i, c in enumerate(chords):
                t0 = _beat_time(beats, a0, gidx * bpb + offs[i], beat_period, anchor)
                t1 = _beat_time(beats, a0, gidx * bpb + offs[i + 1], beat_period, anchor)
                t1 = min(t1, audio_dur)
                if t1 <= t0:
                    continue
                root_pc = (c.root_pc + transpose) % 12
                bass_pc = (c.bass_pc + transpose) % 12
                root_name = NOTE_SHARP[root_pc]
                if c.bass_name is not None and bass_pc != root_pc:
                    label = f"{root_name}:{c.quality}/{NOTE_SHARP[bass_pc]}"
                else:
                    label = f"{root_name}:{c.quality}"
                sb = sounding_bass_pc(label, root_pc)
                raw.append(dict(t0=round(t0, 3), t1=round(t1, 3), root_pc=root_pc,
                                quality=c.quality,
                                bass_pc=(sb if sb is not None else root_pc),
                                label=label))
    # merge adjacent identical (root_pc, quality, bass_pc, label) spans
    merged: list[dict] = []
    for c in raw:
        if merged and (merged[-1]["root_pc"], merged[-1]["quality"],
                       merged[-1]["bass_pc"], merged[-1]["label"]) == \
                (c["root_pc"], c["quality"], c["bass_pc"], c["label"]) and \
                abs(merged[-1]["t1"] - c["t0"]) < 1e-3:
            merged[-1]["t1"] = c["t1"]
        else:
            merged.append(dict(c))
    return merged


# ── verification HTML (self-contained, embedded m4a data URI) ───────────────

def build_html(song: dict, gt: dict, prop: dict, agg: dict,
               bar_starts: list[dict], audio_b64: str) -> str:
    payload = json.dumps(dict(
        title=song["title"], gt_chords=gt["gt_chords"], bar_starts=bar_starts,
        transpose=prop["transpose"], form=prop["form"], anchor=prop["anchor"],
        aggregate=agg,
    ))
    tr, fo, an, grid = prop["transpose"], prop["form"], prop["anchor"], prop["grid"]
    return _HTML_TEMPLATE.replace("__TITLE__", song["title"]) \
        .replace("__AGG__", f"{agg['aggregate_confidence']:.2f}") \
        .replace("__WEAK__", agg["weak_field"]) \
        .replace("__GRID__", grid) \
        .replace("__TR__", f"{tr['transpose_semitones']:+d} (conf {tr['confidence']:.2f}) — {tr['evidence']}") \
        .replace("__FO__", f"{fo['one_chorus_form']} &times;{fo['n_choruses']} (conf {fo['confidence']:.2f}) — {fo['evidence']}") \
        .replace("__AN__", f"t={an['bar1_anchor_time']:.2f}s (conf {an['confidence']:.2f}) — {an['evidence']}") \
        .replace("__NCH__", str(len(gt["gt_chords"]))) \
        .replace("__AUDIO_B64__", audio_b64) \
        .replace("__PAYLOAD__", payload)


_HTML_TEMPLATE = r"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Brick0 verify — __TITLE__</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#14161b;color:#e6e8ec}
 header{padding:14px 20px;background:#1c1f27;border-bottom:1px solid #2a2e39;position:sticky;top:0;z-index:5}
 h1{margin:0 0 6px;font-size:19px}
 .agg{font-size:14px;margin:4px 0 10px}
 .agg b{color:#ffcf5c}
 .field{font-size:12px;line-height:1.5;margin:3px 0;color:#c3c7d1}
 .field span{color:#7fd1ff;font-weight:600}
 .weak{color:#ff8a8a!important;font-weight:700}
 #now{font-size:64px;font-weight:800;text-align:center;padding:18px;letter-spacing:1px}
 #nowsub{text-align:center;color:#9aa0ad;margin-top:-12px;font-size:13px}
 #ribbon{display:flex;overflow-x:auto;gap:2px;padding:10px;background:#0f1116}
 .cell{flex:0 0 auto;min-width:58px;padding:6px 8px;border-radius:5px;background:#232733;
   text-align:center;font-size:13px;cursor:pointer;border:1px solid transparent}
 .cell.active{background:#2f6df6;border-color:#7fb0ff;color:#fff}
 .cell small{display:block;color:#8b91a0;font-size:10px}
 .cell.active small{color:#cfe0ff}
 .controls{padding:10px 20px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
 audio{width:100%;max-width:640px}
 label{font-size:13px;user-select:none}
 .barflash{display:inline-block;width:14px;height:14px;border-radius:50%;background:#333;margin-left:8px;vertical-align:middle}
 .barflash.on{background:#ffcf5c;box-shadow:0 0 10px #ffcf5c}
</style></head><body>
<header>
 <h1>Brick 0 — verify by ear: __TITLE__</h1>
 <div class="agg">aggregate confidence <b>__AGG__</b> &nbsp; weak field: <span class="weak">__WEAK__</span> &nbsp; | &nbsp; __NCH__ GT chords &nbsp; <span class="barflash" id="flash"></span></div>
 <div class="field"><span>grid</span> __GRID__</div>
 <div class="field"><span>transpose</span> __TR__</div>
 <div class="field"><span>form</span> __FO__</div>
 <div class="field"><span>bar-1 anchor</span> __AN__</div>
</header>
<div class="controls">
 <audio id="au" controls src="data:audio/mp4;base64,__AUDIO_B64__"></audio>
 <label><input type="checkbox" id="click" checked> metronome click at bar downbeats (accent = form start)</label>
</div>
<div id="now">—</div>
<div id="nowsub">click a cell to seek • listen: do the chords land on the music at this transpose+form+anchor?</div>
<div id="ribbon"></div>
<script>
const D = __PAYLOAD__;
const au = document.getElementById('au');
const ribbon = document.getElementById('ribbon');
const now = document.getElementById('now');
const flash = document.getElementById('flash');
const clickBox = document.getElementById('click');
// build ribbon
D.gt_chords.forEach((c,i)=>{
  const el=document.createElement('div'); el.className='cell'; el.dataset.i=i;
  el.innerHTML=c.label.replace(':','')+'<small>'+c.t0.toFixed(1)+'s</small>';
  el.onclick=()=>{au.currentTime=c.t0+0.01; au.play();};
  ribbon.appendChild(el);
});
const cells=[...ribbon.children];
// WebAudio click
let actx=null;
function ping(accent){
  if(!clickBox.checked) return;
  actx=actx||new (window.AudioContext||window.webkitAudioContext)();
  const o=actx.createOscillator(), g=actx.createGain();
  o.frequency.value=accent?1760:880; o.connect(g); g.connect(actx.destination);
  const t=actx.currentTime; g.gain.setValueAtTime(0.001,t);
  g.gain.exponentialRampToValueAtTime(accent?0.5:0.28,t+0.001);
  g.gain.exponentialRampToValueAtTime(0.001,t+0.06);
  o.start(t); o.stop(t+0.07);
}
let bi=0, ci=-1;
function resync(){
  const t=au.currentTime;
  bi=D.bar_starts.findIndex(b=>b.t>t); if(bi<0) bi=D.bar_starts.length;
}
au.addEventListener('seeking',resync);
au.addEventListener('play',()=>{resync();});
function frame(){
  const t=au.currentTime;
  while(bi<D.bar_starts.length && D.bar_starts[bi].t<=t){
    ping(D.bar_starts[bi].accent);
    flash.className='barflash on'; setTimeout(()=>flash.className='barflash',90);
    bi++;
  }
  let idx=-1;
  for(let i=0;i<D.gt_chords.length;i++){ if(D.gt_chords[i].t0<=t && t<D.gt_chords[i].t1){idx=i;break;} }
  if(idx!==ci){
    if(ci>=0) cells[ci].classList.remove('active');
    if(idx>=0){ cells[idx].classList.add('active');
      now.textContent=D.gt_chords[idx].label.replace(':','');
      cells[idx].scrollIntoView({inline:'center',block:'nearest',behavior:'smooth'});
    } else now.textContent='—';
    ci=idx;
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
</script></body></html>
"""


# ── main ─────────────────────────────────────────────────────────────────────

def process(song: dict, workdir: Path) -> dict:
    audio = REPO / song["audio"]
    log.info("=== %s (%s) ===", song["song_id"], song["title"])
    chart = parse_chart(song["ireal_file"], song["tune_title"])
    if chart.unmapped:
        log.warning("%s: unmapped quality tokens %s", song["song_id"], set(chart.unmapped))
    dur = audio_duration(audio)

    # Decode m4a -> mono WAV once (beat_this.load_audio + soundfile can't read
    # m4a; ffmpeg is deterministic). Feeds BOTH the beat pass and the chroma.
    wav = decode_wav(audio, workdir)

    # INDEPENDENT Beat This! pass (never the model's decode grid). Bar period +
    # local placement come from the reliable BEAT grid @ chart 4/4; the DOWNBEAT
    # track supplies only the bar-1 phase guess.
    bt = beat_this_full(wav)
    beats, downbeats = bt["beats"], bt["downbeats"]
    beat_period = bt["beat_period"]
    bar_period = beat_period * chart.beats_per_bar
    log.info("  Beat This!: %d beats (reg %.2f) / %d downbeats (reg %.2f); "
             "beat=%.3fs bar=%.3fs (~%.0f BPM); dur=%.1fs",
             len(beats), bt["beat_regularity"], len(downbeats),
             bt["downbeat_regularity"], beat_period, bar_period,
             60.0 / beat_period if beat_period else 0, dur)

    chroma = audio_chroma(wav)

    tr = propose_transpose(chart, chroma)
    an = propose_anchor(downbeats, bt["downbeat_regularity"], bar_period)
    anchor = an["bar1_anchor_time"]
    fo = propose_form(chart, bar_period, anchor, dur, bt["beat_regularity"])

    gt_chords = build_gt_chords(chart, fo["n_choruses"], beats, anchor,
                                beat_period, tr["transpose_semitones"], dur)

    # aggregate = weakest of the three machine-guessed fields
    fields = {"transpose": tr["confidence"], "form": fo["confidence"],
              "bar1_anchor": an["confidence"]}
    weak = min(fields, key=fields.get)
    agg = dict(aggregate_confidence=round(min(fields.values()), 3),
               weak_field=weak, field_confidences=fields)

    # bar-start grid used for the GT (= every 4th beat from the anchor beat);
    # also the schema's downbeat_times + the HTML metronome.
    a0 = int(np.argmin(np.abs(beats - anchor))) if len(beats) else 0
    total_bars = fo["n_choruses"] * chart.n_bars
    bar_starts = []
    for g in range(total_bars + 1):
        t = _beat_time(beats, a0, g * chart.beats_per_bar, beat_period, anchor)
        if t >= dur:
            break
        bar_starts.append(dict(t=round(t, 3), accent=bool(g % chart.n_bars == 0)))

    # ---- write golden/brick0/<song>.gt.json (verified=false) ----
    gt_json = dict(
        song_id=song["song_id"], title=song["title"], audio_path=song["audio"],
        chart_source=dict(ireal_file=song["ireal_file"], index=None,
                          tune_title=song["tune_title"]),
        transpose_semitones=tr["transpose_semitones"],
        form=dict(section_order=fo["section_order"], repeat_counts=fo["repeat_counts"],
                  intro_bars=fo["intro_bars"], outro_bars=fo["outro_bars"],
                  n_choruses=fo["n_choruses"], bars_per_chorus=fo["bars_per_chorus"],
                  one_chorus_form=fo["one_chorus_form"]),
        bar1_anchor_time=an["bar1_anchor_time"],
        downbeat_times=[b["t"] for b in bar_starts],
        gt_chords=gt_chords,
        verified=False,
        proposal=dict(  # per-field confidence/alt/evidence for the human reviewer
            aggregate=agg,
            transpose=tr, form={k: v for k, v in fo.items()},
            bar1_anchor=an,
            beat_this=dict(n_beats=len(beats), n_downbeats=len(downbeats),
                           beat_regularity=round(bt["beat_regularity"], 3),
                           downbeat_regularity=round(bt["downbeat_regularity"], 3),
                           beat_period=round(beat_period, 3),
                           bar_period=round(bar_period, 3),
                           est_bpm=round(60.0 / beat_period, 1) if beat_period else None),
            audio_duration_s=round(dur, 2),
            unmapped_quality_tokens=sorted(set(chart.unmapped)),
            builder="scripts/brick0_propose.py",
        ),
    )
    gt_path = GOLDEN / f"{song['song_id']}.gt.json"
    gt_path.write_text(json.dumps(gt_json, indent=2))
    log.info("  wrote %s (%d chords)", gt_path.relative_to(REPO), len(gt_chords))

    # grid context (surfaces double/half-time detections, e.g. a 187-BPM lock)
    est_bpm = 60.0 / beat_period if beat_period else 0.0
    warn = ""
    if est_bpm > 175:
        warn = " ⚠ high BPM — possible DOUBLE-TIME lock (chords/clicks may be 2x too fast)"
    elif est_bpm < 58:
        warn = " ⚠ low BPM — possible HALF-TIME lock (chords/clicks may be 2x too slow)"
    grid_str = (f"~{est_bpm:.0f} BPM (beat {beat_period:.2f}s, bar {bar_period:.2f}s @ "
                f"{chart.beats_per_bar}/4) • beat-reg {bt['beat_regularity']:.2f} • "
                f"downbeat-reg {bt['downbeat_regularity']:.2f}{warn}")

    # ---- write verification HTML ----
    audio_b64 = base64.b64encode(audio.read_bytes()).decode()
    html = build_html(
        song, dict(gt_chords=gt_chords),
        dict(transpose=tr, form=fo, anchor=an, grid=grid_str), agg, bar_starts, audio_b64)
    html_path = REVIEW / f"{song['song_id']}.html"
    html_path.write_text(html)
    log.info("  wrote %s (%.1f MB)", html_path.relative_to(REPO),
             len(html) / 1e6)

    return dict(
        song_id=song["song_id"], title=song["title"], audio_path=song["audio"],
        ireal_file=song["ireal_file"], tune_title=song["tune_title"],
        transpose=tr, form={k: fo[k] for k in
                            ("n_choruses", "one_chorus_form", "confidence",
                             "alternative", "evidence")},
        bar1_anchor=an, aggregate=agg, n_chords=len(gt_chords),
        audio_duration_s=round(dur, 2), grid=grid_str,
        gt_path=str(gt_path.relative_to(REPO)),
        html_path=str(html_path.relative_to(REPO)),
    )


def main() -> int:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    REVIEW.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="brick0_"))
    rows = []
    for song in BATCH1:
        try:
            rows.append(process(song, workdir))
        except Exception as exc:  # noqa: BLE001
            log.exception("FAILED %s: %s", song["song_id"], exc)
    rows.sort(key=lambda r: r["aggregate"]["aggregate_confidence"])  # low -> high

    MANIFEST.write_text(json.dumps(dict(
        batch="brick0_batch1", n_songs=len(rows), verified=False,
        note="ALL verified=false — machine PROPOSALS for human ear-verification. "
             "Sorted ascending by aggregate confidence (weakest first).",
        songs=rows), indent=2))
    log.info("wrote manifest %s", MANIFEST.relative_to(REPO))

    write_queue_md(rows)
    return 0


def write_queue_md(rows: list[dict]) -> None:
    lines = ["# Brick 0 — batch 1 verification QUEUE (verified=false)",
             "",
             "Machine PROPOSALS for Louis to hand-verify by ear. **Nothing is "
             "frozen/verified.** Sorted ascending by aggregate confidence "
             "(= min of transpose/form/anchor) — **do the top ones first**; the "
             "weak field is where your ear is most needed.",
             "",
             "Open each `<song>.html` (self-contained, embedded audio + downbeat "
             "metronome). Check: at the shown transpose+form+anchor, do the chord "
             "cells land on the music?",
             ""]
    for i, r in enumerate(rows, 1):
        a = r["aggregate"]
        tr, fo, an = r["transpose"], r["form"], r["bar1_anchor"]
        lines += [
            f"## {i}. {r['title']}  —  aggregate **{a['aggregate_confidence']:.2f}**"
            f"  (weak: **{a['weak_field']}**)",
            f"- audio {r['audio_duration_s']:.0f}s • {r['n_chords']} GT chords • "
            f"HTML `{r['html_path']}` • GT `{r['gt_path']}`",
            f"- grid: {r['grid']}",
            f"- **transpose** {tr['transpose_semitones']:+d}, conf {tr['confidence']:.2f}; "
            f"alt {tr['alternative']['transpose']:+d} — {tr['evidence']}",
            f"- **form** {fo['one_chorus_form']} &times;{fo['n_choruses']}, "
            f"conf {fo['confidence']:.2f}; alt &times;{fo['alternative']['n_choruses']} — {fo['evidence']}",
            f"- **bar-1 anchor** t={an['bar1_anchor_time']:.2f}s, conf {an['confidence']:.2f}; "
            f"alt t={an['alternative']['bar1_anchor_time']:.2f}s — {an['evidence']}",
            "",
        ]
    (REVIEW / "_QUEUE.md").write_text("\n".join(lines))
    log.info("wrote %s", (REVIEW / "_QUEUE.md").relative_to(REPO))


if __name__ == "__main__":
    raise SystemExit(main())
