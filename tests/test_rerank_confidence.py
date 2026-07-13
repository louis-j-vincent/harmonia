"""Red-first tests for the stale-confidence bug (audit 2026-07-13, build-order step 1a).

The 8a/8b second-pass rerankers in ``infer_chords_v1`` flip a segment's quality
but carry the PRE-rerank acoustic confidence into ``labeled`` — the confidence
shown no longer describes the decision shown.  Fix contract tested here: both
rerank functions accept ``return_post=True`` and return ``(qualities, posts)``
where ``posts[i]`` is the normalized posterior probability of the decision the
reranker actually made at a *flipped* position, and ``None`` where the label
was left unchanged (acoustic conf still describes an unchanged label; global
recalibration is step 1b's job, not this fix's).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.models import chord_pipeline_v1 as P  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

_ENC = P._get_progression_encoder()
needs_encoder = pytest.mark.skipif(_ENC is None, reason="progression_encoder.pt unavailable")


def _lk_model_or_skip():
    if P._get_local_key_seq_model() is None:
        pytest.skip("local_key_seq_gru.pt not available")


# ── apply_diatonic_prior ──────────────────────────────────────────────────────

def test_diatonic_prior_posterior_on_flip():
    """vi mis-called major in C major, uncertain acoustics → flips to min, and
    the returned posterior is the 2-way normalized winner probability."""
    conf, boost = 0.4, 4.0
    sev, post = P.apply_diatonic_prior(
        9, "maj", conf, 0, "major", 0.9,
        diatonic_boost=boost, return_post=True,
    )
    assert sev == "min"
    expected = ((1 - conf) * boost) / ((1 - conf) * boost + conf)  # 0.857…
    assert post == pytest.approx(expected, abs=1e-6)
    assert post != conf, "posterior must not be the stale acoustic conf"


def test_diatonic_prior_no_flip_posterior_is_none():
    # confident acoustics → gate closed → pass-through, no posterior
    sev, post = P.apply_diatonic_prior(
        9, "maj", 0.95, 0, "major", 0.9, return_post=True,
    )
    assert sev == "maj" and post is None
    # already-diatonic call → pass-through, no posterior
    sev, post = P.apply_diatonic_prior(
        9, "min", 0.4, 0, "major", 0.9, return_post=True,
    )
    assert sev == "min" and post is None


def test_diatonic_prior_backward_compatible():
    """Without return_post the function still returns a bare quality string."""
    out = P.apply_diatonic_prior(9, "maj", 0.4, 0, "major", 0.9)
    assert isinstance(out, str)


# ── rerank_local_key_qualities ────────────────────────────────────────────────

def test_local_key_rerank_posts_align_with_flips():
    _lk_model_or_skip()
    roots = [0, 7, 9, 5] * 3
    sev = ["maj"] * 12
    confs = [0.5] * 12
    out, posts = P.rerank_local_key_qualities(
        roots, sev, confs, global_tonic=0, boost=4.0, return_post=True,
    )
    assert len(out) == len(posts) == 12
    assert out[2] == "min"                       # the Georgia/Let-It-Be flip
    for i in range(12):
        if out[i] != sev[i]:
            assert posts[i] is not None and 0.0 < posts[i] <= 1.0
            assert posts[i] != confs[i]
        else:
            assert posts[i] is None


# ── rerank_progression_qualities ──────────────────────────────────────────────

@needs_encoder
def test_progression_rerank_posts_align_with_flips():
    roots = [2, 7, 0, 2, 7, 0]
    sevs = ["min7", "maj7", "maj7", "min7", "7", "maj7"]   # index 1 mislabelled
    confs = [0.9, 0.35, 0.9, 0.9, 0.9, 0.9]
    out, posts = P.rerank_progression_qualities(
        roots, sevs, confs, weight=1.0, return_post=True,
    )
    assert out[1] == "7"                          # recovered dominant
    assert posts[1] is not None and 0.0 < posts[1] <= 1.0
    assert posts[1] != confs[1], "posterior must not be the stale acoustic conf"
    for i in range(6):
        if out[i] == sevs[i]:
            assert posts[i] is None


@needs_encoder
def test_progression_rerank_backward_compatible():
    roots = [2, 7, 0]
    sevs = ["min7", "7", "maj7"]
    confs = [0.95, 0.95, 0.95]
    out = P.rerank_progression_qualities(roots, sevs, confs, weight=0.5)
    assert isinstance(out, list) and all(isinstance(s, str) for s in out)


def test_progression_rerank_empty_with_post():
    out, posts = P.rerank_progression_qualities([], [], [], weight=0.5, return_post=True)
    assert out == [] and posts == []
