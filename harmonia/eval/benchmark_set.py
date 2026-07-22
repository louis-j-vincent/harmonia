"""harmonia/eval/benchmark_set.py — the FROZEN PARITY BENCHMARK manifest.

This is the fixed, small song set that the staged-rewrite parity/accuracy nets
are gated on (rewrite_execution_plan_2026_07_21.md §0, §1, Phase 0). It is
data + provenance ONLY — no capture logic lives here (see parity.py for that).

════════════════════════════════════════════════════════════════════════════
WHY THIS EXISTS / THE TWO-NET + GT-PROVENANCE RULE (plan §0, §1)
════════════════════════════════════════════════════════════════════════════
Two different questions, never to be conflated:

  * PARITY net  — "did a refactor CHANGE behaviour?"  new-code output vs the
    CURRENT code's output on frozen audio, label-for-label / vector-for-vector.
    Needs NO ground truth. This is the load-bearing safety net for the rewrite.
  * ACCURACY net — "is the output RIGHT?"  new-code output vs real ground truth.
    Only valid where INDEPENDENT ground truth exists.

And the GT-provenance trap (plan §1 — the single most important measurement
subtlety in this project):

  | GT source        | valid for CHORD-LABEL eval | valid for BEAT/ALIGNMENT eval |
  | RWC-Popular      | yes (AIST chord labels)    | yes (AIST beats) — **but audio gone** |
  | POP909 beat_midi | yes (functional-root only) | yes — **independent downbeat GT**     |
  | aligned_corpus   | labels OK, **at self-derived timestamps** | NO — circular (its timing IS the alignment output) |

════════════════════════════════════════════════════════════════════════════
WHAT SURVIVES ON DISK (the hard constraints that shaped this set)
════════════════════════════════════════════════════════════════════════════
  * RWC full-song AUDIO IS GONE (licensed) — only docs/audio/rwc_rwc_p001.m4a
    survives, plus cached RWC features (data/cache/rwc/rwc_bp48.npz). So RWC can
    only ever contribute chord-accuracy-from-cached-features, never an
    audio-dependent stage.  It is NOT in this benchmark (feature-only, and the
    chord-head-on-cached-features capture is deferred — see CHORD_LABEL_REFS).
  * POP909 is MIDI-synth: beat_midi.txt carries INDEPENDENT beat+downbeat GT,
    but NO rendered audio is on disk and rendering wav is forbidden this
    session. So POP909 songs are BEAT/ALIGNMENT-net GT REFERENCES only — their
    audio-pipeline golden capture is deferred (see BEAT_ALIGN_REFS).
  * docs/audio/*.m4a (real YouTube pop/jazz) + their Basic-Pitch stem caches
    (data/cache/pitch/<slug>.npz) are the only AVAILABLE AUDIO that can drive
    the full pipeline. These carry the PARITY net (see PARITY_SONGS). None of
    them has independent chord-label OR beat GT, so they serve PARITY ONLY.

════════════════════════════════════════════════════════════════════════════
THE FROZEN PARITY ORACLE CONFIG
════════════════════════════════════════════════════════════════════════════
The current LIVE production path (scripts/harmonia_server.py `_run_analysis`,
== harmonia.pipeline.PipelineConfig.live_defaults()) is:

    infer_chords_v1(wav, feature_frontend="nnls24", bass_frontend="musx",
                    quality_frontend="musx", segment_source="nnls",
                    beat_backend="beatthis", beat_period_mode="bestfit", ...)

That is the parity oracle frozen by the golden capture. Verified deterministic
across two independent processes (2026-07-22 screen). NB: infer_chords_v1 loads
audio via soundfile.read, which cannot decode .m4a — the pipeline is fed a
decoded WAV (matches the server, which transcodes before analysis).
"""

from __future__ import annotations

import json
from pathlib import Path

BENCHMARK_VERSION = "frozen_parity_v1_2026_07_22"

# Repo root (this file is harmonia/eval/benchmark_set.py → parents[2] == repo).
REPO = Path(__file__).resolve().parents[2]
AUDIO_DIR = REPO / "docs" / "audio"
PITCH_CACHE_DIR = REPO / "data" / "cache" / "pitch"
POP909_DIR = REPO / "data" / "pop909" / "POP909"
ALIGNED_CORPUS_NPZ = REPO / "data" / "cache" / "aligned_corpus" / "aligned_corpus.npz"

# The live production parity-oracle kwargs to infer_chords_v1 (== live_defaults).
LIVE_ORACLE_KWARGS: dict = {
    "feature_frontend": "nnls24",
    "bass_frontend": "musx",
    "quality_frontend": "musx",
    "segment_source": "nnls",
    "beat_backend": "beatthis",
    "beat_period_mode": "bestfit",
}

# ── PARITY net: real-audio songs with a Basic-Pitch stem cache ────────────────
# Chosen from docs/audio for genre diversity + short-to-medium length (keeps
# golden capture tractable & disk-safe). Each has both an .m4a and a stem pitch
# cache. NONE has independent chord/beat GT → PARITY net ONLY.
# duration_s is measured from the stem-cache frame count / 86.1328125 Hz.
PARITY_SONGS: list[dict] = [
    {"song_id": "yesterday_remastered_2009",              "genre": "beatles-pop", "duration_s": 125.5},
    {"song_id": "muppets_kermit_its_not_easy_being_green_original", "genre": "ballad", "duration_s": 130.2},
    {"song_id": "the_ronettes_be_my_baby_music_video",    "genre": "motown",      "duration_s": 160.1},
    {"song_id": "nina_simone_feeling_good_lyric_video",   "genre": "jazz-soul",   "duration_s": 176.5},
    {"song_id": "ben_e_king_stand_by_me_audio",           "genre": "soul",        "duration_s": 177.2},
    {"song_id": "bein_green",                             "genre": "jazz-ballad", "duration_s": 178.8},
    {"song_id": "ray_charles_georgia_on_my_mind_official_video", "genre": "jazz-soul", "duration_s": 217.1},
    {"song_id": "let_it_be_remastered_2009",              "genre": "beatles-pop", "duration_s": 242.6},
]

# ── CHORD-LABEL net: aligned_corpus jazz standards (cached feat24 + GT) ────────
# Valid for chord-label accuracy ONLY "at aligned timestamps" (plan §1: the
# per-chord t0/t1 ARE the alignment output → NOT usable for beat/alignment).
# Source is the cached 24-dim chroma table, not audio — so these are audio-
# independent. Capture of a chord-head prediction vs GT on feat24 is DEFERRED
# this session (the audio-pipeline golden is the load-bearing deliverable);
# recorded here so the chord-label net's members + provenance are frozen.
CHORD_LABEL_REFS: list[dict] = [
    {"song_id": "Autumn Leaves"},
    {"song_id": "A Night In Tunisia"},
    {"song_id": "Falling"},
    {"song_id": "Fifty Ways To Leave Your Lover"},
]

# ── BEAT/ALIGNMENT net: POP909 songs with INDEPENDENT downbeat GT ─────────────
# beat_midi.txt col1 = beat times (s), col3 = downbeat flag — independent of any
# alignment (plan §1). tempo_gt = 60 / median(diff(beat times)). Song 002 is the
# CLAUDE.md octave-lock canary: GT ~64 BPM, librosa doubles it to ~129.
# NO rendered audio on disk + rendering forbidden this session → audio-pipeline
# golden capture is DEFERRED; these anchor the beat/alignment ACCURACY net and
# the song-002 calibration pin.
BEAT_ALIGN_REFS: list[dict] = [
    {"song_id": "001", "tempo_gt_bpm": 90.0},
    {"song_id": "002", "tempo_gt_bpm": 64.0},  # octave-lock canary (NOT 129)
    {"song_id": "008", "tempo_gt_bpm": 65.0},
]


def parity_songs() -> list[dict]:
    """PARITY-net songs, each augmented with resolved audio/cache paths + flags.

    `audio_path` is the .m4a; `capturable` is True iff both the .m4a and the
    stem pitch cache exist on disk (the golden capture needs the audio; the
    stem cache makes the bp48 feature stage disk-free).
    """
    out = []
    for s in PARITY_SONGS:
        sid = s["song_id"]
        m4a = AUDIO_DIR / f"{sid}.m4a"
        pit = PITCH_CACHE_DIR / f"{sid}.npz"
        out.append({
            **s,
            "nets": ["parity"],
            "gt": {"chord_label": None, "beat_alignment": None},
            "gt_note": "no independent GT — PARITY net only",
            "audio_path": str(m4a),
            "pitch_cache": str(pit),
            "capturable": m4a.exists() and pit.exists(),
        })
    return out


def chord_label_refs() -> list[dict]:
    """CHORD-LABEL-net reference entries (aligned_corpus feat24 rows)."""
    return [{
        **s,
        "nets": ["chord_label"],
        "source": str(ALIGNED_CORPUS_NPZ),
        "gt_provenance": "aligned_corpus feat24 + root/quality/bass GT",
        "valid_for": "chord-label accuracy AT ALIGNED TIMESTAMPS ONLY (plan §1)",
        "not_valid_for": "beat/alignment (circular — timing IS the alignment output)",
        "capturable": False,
        "capture_status": "deferred (cached-feature chord-head capture not built this session)",
    } for s in CHORD_LABEL_REFS]


def beat_align_refs() -> list[dict]:
    """BEAT/ALIGNMENT-net reference entries (POP909 independent downbeat GT)."""
    return [{
        **s,
        "nets": ["beat_alignment"],
        "beat_midi": str(POP909_DIR / s["song_id"] / "beat_midi.txt"),
        "gt_provenance": "POP909 beat_midi.txt col1 beats / col3 downbeats (INDEPENDENT)",
        "valid_for": "beat / downbeat / alignment accuracy (independent GT, plan §1)",
        "audio_status": "MISSING — POP909 is MIDI-synth; no rendered wav on disk; render forbidden",
        "capturable": False,
        "capture_status": "deferred (audio-pipeline golden needs rendered audio)",
    } for s in BEAT_ALIGN_REFS]


def all_entries() -> list[dict]:
    return parity_songs() + chord_label_refs() + beat_align_refs()


def manifest() -> dict:
    """The full frozen manifest as a JSON-serializable dict."""
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "oracle_config": LIVE_ORACLE_KWARGS,
        "oracle_config_note": "== harmonia.pipeline.PipelineConfig.live_defaults(); "
                              "infer_chords_v1 fed a decoded WAV (sf.read can't read m4a).",
        "nets": {
            "parity": "new-code vs CURRENT-code output on frozen audio (no GT). Load-bearing.",
            "chord_label": "output vs GT chord labels — aligned_corpus, AT ALIGNED TIMESTAMPS only.",
            "beat_alignment": "beat/downbeat vs INDEPENDENT GT — POP909 downbeats only (never aligned_corpus).",
        },
        "parity_songs": parity_songs(),
        "chord_label_refs": chord_label_refs(),
        "beat_align_refs": beat_align_refs(),
    }


def write_manifest_json(path: Path | str | None = None) -> Path:
    path = Path(path) if path else (REPO / "harmonia" / "eval" / "golden" /
                                    "frozen_parity" / "benchmark_manifest.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest(), indent=2))
    return path


def _print_provenance_table() -> None:
    print(f"FROZEN PARITY BENCHMARK — {BENCHMARK_VERSION}")
    print(f"oracle: infer_chords_v1({LIVE_ORACLE_KWARGS})\n")
    print(f"{'song_id':<52} {'net':<15} {'capturable':<11} gt")
    print("-" * 100)
    for e in all_entries():
        net = ",".join(e["nets"])
        gt = e.get("gt_note") or e.get("valid_for") or e.get("gt_provenance", "")
        print(f"{e['song_id']:<52} {net:<15} {str(e['capturable']):<11} {gt[:40]}")


if __name__ == "__main__":  # pragma: no cover
    _print_provenance_table()
    p = write_manifest_json()
    print(f"\nmanifest → {p}")
