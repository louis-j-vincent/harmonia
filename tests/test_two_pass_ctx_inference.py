"""Two-pass 801d key-relative ctx inference wiring (#20/#23, volet 2).

The bootstrap A/B established that a 117d key-relative local-key block lifts
held-out family accuracy (+4.3pp, minor-family +7.6pp) — but computed with a
GT-quality context (an upper bound). Making that gain realizable needs a
*two-pass* inference scheme, wired in chord_pipeline_v1:

  1. pass 1 classifies quality with no local-key feature;
  2. the raw-v2 continuity teacher reads a local key per chord off that
     predicted sequence;
  3. pass 2 re-runs the 801d model with the resulting 117d block.

These tests pin the load-bearing pieces of that wiring:
  * the (root, Harte-quality) → iReal-token converter routes every quality
    through the teacher's functional class correctly;
  * the inference-side windowed block matches the training-side one bit-for-bit
    (train_ctx_model_v2._localkey_ctx_onehots) — the feature the model's weights
    expect;
  * the whole pass-1 → local-key step is transpose-invariant (degree-relative).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from harmonia.models import chord_pipeline_v1 as cp
from harmonia.theory.local_key import quality_class, parse_token


def test_token_converter_routes_functional_class():
    """Each Harte quality must map to a token the teacher reads as the right
    functional class (quality_class)."""
    cases = {
        "maj": "maj", "maj7": "maj", "6": "maj",
        "min": "min", "min7": "min", "minmaj7": "min", "min6": "min",
        "7": "dom", "9": "dom", "13": "dom",
        "hdim7": "m7b5", "dim": "dim", "dim7": "dim",
        "sus4": "sus",
    }
    for sev, want in cases.items():
        tok = cp._sev_to_localkey_token(0, sev)
        got = quality_class(parse_token(tok)[1])
        assert got == want, f"{sev} -> {tok!r} classified {got}, want {want}"


def test_token_converter_root_spelling():
    for root in range(12):
        tok = cp._sev_to_localkey_token(root, "7")
        assert parse_token(tok)[0] == root


def test_window_block_matches_training():
    """The inference windowed block must equal the training-side builder
    (_localkey_ctx_onehots) for the same per-chord (degree, mode) sequence."""
    import train_ctx_model_v2 as tr

    rng = np.random.default_rng(0)
    n = 15
    lk_pos = [(int(rng.integers(0, 12)), int(rng.integers(0, 2))) for _ in range(n)]
    # training builder consumes records carrying lk_degree / lk_mode
    records = [{"lk_degree": d, "lk_mode": m} for (d, m) in lk_pos]
    train_block = tr._localkey_ctx_onehots(records, k=4)  # (n, 117)
    for i in range(n):
        infer_block = cp._localkey_window_block(lk_pos, i, k=4)
        assert infer_block.shape == (117,)
        np.testing.assert_array_equal(infer_block, train_block[i])


def test_localkey_track_transpose_invariant():
    """Reading a local key off a predicted (root, quality) sequence is
    degree-relative: transposing every root by the same offset leaves the
    (degree, mode) labels unchanged."""
    roots = [0, 7, 9, 5, 0, 2, 7, 0]      # C G Am F C Dm G C
    sevs = ["maj", "maj", "min", "maj", "maj", "min", "7", "maj"]
    base = cp._localkey_track_from_qualities_v2(roots, sevs, 0, "major")
    for shift in range(1, 12):
        roots_t = [(r + shift) % 12 for r in roots]
        # home tonic transposes with the song
        shifted = cp._localkey_track_from_qualities_v2(
            roots_t, sevs, shift % 12, "major"
        )
        assert shifted == base, f"shift {shift}: {shifted} != {base}"


def test_let_it_be_am_reads_as_vi_of_C():
    """The motivating case: C–G–Am–F. The Am must read as degree 9 (the vi) of a
    C-major local key, mode major — i.e. a minor chord expected diatonically,
    NOT a modulation to A. This is the signal the 801d model uses to stop the
    'A major where La mineur is expected' flip."""
    roots = [0, 7, 9, 5]
    sevs = ["maj", "maj", "min", "maj"]
    lk = cp._localkey_track_from_qualities_v2(roots, sevs, 0, "major")
    # every chord in the local key of C major (mode bit 0)
    assert all(m == 0 for (_d, m) in lk)
    assert lk[2][0] == 9   # Am is the 9th degree (vi) of C
