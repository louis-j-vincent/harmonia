"""Phase-7 EXIT GATE: no new inline feature-extraction sites.

Phase 7 routed every reachable feature-extraction call-site onto the canonical
bricks:

    * audio file -> model activations : `harmonia.core.features.FeatureExtractor`
                                        (`.create("bp48"|"nnls24"|"musx")`)
    * (y, sr) waveform -> CQT chroma  : `harmonia.core.chroma`
                                        (`chroma_cqt` / `ltas_normalize` /
                                         `chroma_cqt_ltas`)

This test greps the repo and fails when a NEW inline site appears — i.e. a
direct `stage1_pitch.PitchExtractor` use, or a raw `librosa.feature.chroma_*` /
`librosa.cqt` call — anywhere outside the explicit allowlist below.

If you are adding a feature-extraction call: use the brick. If you genuinely
need a new allowlist entry, add it here WITH a reason, so the exemption is a
deliberate, reviewed decision rather than silent drift.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Directories that are not part of the reviewed source surface.
SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "scratchpad", "docs", "data",
    "node_modules", ".pytest_cache", ".mypy_cache", "third_party", "build",
    "dist", ".ruff_cache",
}

# ─────────────────────────────────────────────────────────────────────────────
# Patterns. Deliberately match only ACTIONABLE code forms (an import of the
# symbol, or a call), never prose — otherwise every docstring mentioning
# "PitchExtractor" would trip the gate.
# ─────────────────────────────────────────────────────────────────────────────
PITCH_EXTRACTOR_RE = re.compile(
    r"from\s+harmonia\.models\.stage1_pitch\s+import\s+[^\n]*\bPitchExtractor\b"
    r"|\bPitchExtractor\s*\("
)
INLINE_CHROMA_RE = re.compile(
    r"librosa\.feature\.chroma_(?:cqt|stft|cens)\s*\("
    r"|librosa\.(?:cqt|hybrid_cqt|pseudo_cqt)\s*\("
)

# ─────────────────────────────────────────────────────────────────────────────
# ALLOWLISTS — each entry is a legitimate exemption with a stated reason.
# ─────────────────────────────────────────────────────────────────────────────
PITCH_EXTRACTOR_ALLOW: dict[str, str] = {
    # The brick itself and the class it wraps.
    "harmonia/core/features.py":
        "THE canonical wrapper — its own delegation to PitchExtractor",
    "harmonia/models/stage1_pitch.py":
        "defines PitchExtractor",
    # Guarded library sites: live/shipped paths, deliberately not rerouted in
    # Phase 7 (owned by other lanes; reroute needs its own parity gate).
    "harmonia/pipeline.py":
        "guarded library site (Phase-7 skip: shipped pipeline)",
    "harmonia/models/chord_pipeline_v1.py":
        "guarded library site (Phase-7 skip: shipped chord pipeline)",
    "harmonia/serving/analysis.py":
        "guarded library site (Phase-7 skip: serving lane)",
    # Corpus builders whose source audio is gone from disk — cannot be run,
    # so cannot be gated by an extraction-vs-extraction bit-identity proof.
    "scripts/build_rwc_corpus.py": "audio-gone builder (RWC audio purged)",
    "scripts/build_jaah_corpus.py": "audio-gone builder (JAAH audio purged)",
    "scripts/train_billboard_batched.py": "audio-gone builder (Billboard audio purged)",
    "scripts/train_billboard_chord_model.py": "audio-gone builder (Billboard audio purged)",
    # Tests that exercise PitchExtractor directly — that IS their subject.
    "tests/test_stage1_pitch.py": "unit-tests PitchExtractor itself",
    "tests/test_stage1_pitch_characterization.py": "pins PitchExtractor's numeric output",
    "tests/test_no_inline_feature_sites.py": "this gate (patterns appear as literals)",
}

INLINE_CHROMA_ALLOW: dict[str, str] = {
    # The brick itself.
    "harmonia/core/chroma.py": "THE canonical chroma brick — the one real call",
    # Other-lane library modules: outside Phase 7's edit scope.
    "harmonia/align/bass_salience.py": "align lane (raw CQT for bass salience, not chroma)",
    "harmonia/models/section_structure.py": "models lane (raw CQT for structure SSM)",
    "harmonia/data/ireal_youtube_align.py": "data lane (iReal/YouTube alignment)",
    "harmonia/data/yt_chord_corpus.py": "data lane (YouTube corpus builder)",
    "harmonia/dataset/harvest.py": "dataset lane (harvest)",
    # Audio-gone builders (see above).
    "scripts/build_jaah_corpus.py": "audio-gone builder (JAAH audio purged)",
    "scripts/train_billboard_batched.py": "audio-gone builder (Billboard audio purged)",
    "scripts/train_billboard_chord_model.py": "audio-gone builder (Billboard audio purged)",
    "tests/test_no_inline_feature_sites.py": "this gate (patterns appear as literals)",
}


def _python_files() -> list[Path]:
    out = []
    for p in REPO.rglob("*.py"):
        if any(part in SKIP_DIRS for part in p.relative_to(REPO).parts):
            continue
        out.append(p)
    return out


def _offenders(pattern: re.Pattern, allow: dict[str, str]) -> list[str]:
    bad = []
    for path in _python_files():
        rel = path.relative_to(REPO).as_posix()
        if rel in allow:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                bad.append(f"{rel}:{i}: {line.strip()}")
    return bad


def test_no_inline_pitch_extractor_sites():
    """Feature extraction must go through core.features.FeatureExtractor."""
    bad = _offenders(PITCH_EXTRACTOR_RE, PITCH_EXTRACTOR_ALLOW)
    assert not bad, (
        "New inline PitchExtractor site(s) — use "
        '`FeatureExtractor.create("bp48", cache_dir=...)` from '
        "harmonia.core.features instead (fields: .activations / .onsets / "
        ".frame_times). If the site is a genuine exception, add it to "
        "PITCH_EXTRACTOR_ALLOW with a reason:\n  " + "\n  ".join(bad)
    )


def test_no_inline_librosa_chroma_sites():
    """CQT chroma must go through core.chroma."""
    bad = _offenders(INLINE_CHROMA_RE, INLINE_CHROMA_ALLOW)
    assert not bad, (
        "New inline librosa chroma/CQT site(s) — use "
        "`chroma_cqt` / `chroma_cqt_ltas` from harmonia.core.chroma instead. "
        "If the site is a genuine exception, add it to INLINE_CHROMA_ALLOW "
        "with a reason:\n  " + "\n  ".join(bad)
    )


@pytest.mark.parametrize(
    "allow", [PITCH_EXTRACTOR_ALLOW, INLINE_CHROMA_ALLOW], ids=["pitch", "chroma"]
)
def test_allowlist_entries_exist(allow: dict[str, str]):
    """An allowlist entry pointing at a deleted file is silent rot — catch it.

    (Deliberately does NOT require the pattern to still be present: removing
    an inline site is an improvement and must never fail the suite. Only a
    missing FILE is an error.)
    """
    missing = [rel for rel in allow if not (REPO / rel).exists()]
    assert not missing, (
        "Allowlist references file(s) that no longer exist — remove the stale "
        "entr(ies):\n  " + "\n  ".join(missing)
    )
