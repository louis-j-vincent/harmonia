"""Tests for the default-OFF `function_family` brick (maj<->dom function fix).

Nothing here needs the music-x-lab clone: the `s7` posterior is supplied as a
plain array, which is exactly the contract `musx_redecode.frame_posteriors`
returns (index 2 of its list).

The load-bearing properties, in order of how badly a silent break would hurt:
  1. OFF is an EXACT no-op — the same list object comes back.
  2. The frame grid matches the vendored one (CLAUDE.md #1 calibration pin).
  3. The seventh-degree head's column order matches complex_chord.SeventhTypes.
  4. Root and bass are NEVER rewritten (the brick's whole safety claim, and the
     reason mirex_root / bass_root move by exactly 0.00 pp on the benchmark).
  5. Neither witness can act alone in the direction the other opposes at the
     default weight.
"""
from __future__ import annotations

import numpy as np
import pytest

from harmonia.models import function_family as ff
from harmonia.models import musx_redecode as mr


# ── 1. calibration pins ─────────────────────────────────────────────────────

def test_frame_grid_matches_musx_redecode():
    """Both modules index the same posterior array; a drift here would sample
    the wrong frames and produce plausible-but-wrong numbers (error-pattern #1)."""
    assert ff.FRAME_DT == pytest.approx(mr.FRAME_DT, abs=1e-12)
    assert ff.FRAME_DT == pytest.approx(512 / 22050, abs=1e-12)


def test_s7_column_order_matches_the_vendored_enum():
    """complex_chord.SeventhTypes: none=0, add_7=1, add_b7=2, add_bb7=3."""
    assert (ff.S7_NONE, ff.S7_ADD7, ff.S7_ADDB7, ff.S7_ADDBB7) == (0, 1, 2, 3)


# ── helpers ─────────────────────────────────────────────────────────────────

def _s7(n=200, p=(0.1, 0.1, 0.7, 0.1)):
    a = np.zeros((n, 4), dtype=np.float32)
    a[:] = np.asarray(p, dtype=np.float32)
    return a


def _chords(*labels, dur=1.0):
    return [{"label": l, "start_s": i * dur, "end_s": (i + 1) * dur}
            for i, l in enumerate(labels)]


# ── 2. OFF is an exact no-op ────────────────────────────────────────────────

def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv(ff.ENV_ENABLE, raising=False)
    assert ff.enabled() is False


def test_off_returns_the_same_object(monkeypatch):
    monkeypatch.delenv(ff.ENV_ENABLE, raising=False)
    ch = _chords("C:maj", "F:maj", "G:maj")
    out = ff.apply(ch, _s7())
    assert out is ch                      # identity, not just equality


def test_off_env_zero_is_also_a_no_op(monkeypatch):
    monkeypatch.setenv(ff.ENV_ENABLE, "0")
    ch = _chords("G:maj", "C:maj")
    assert ff.apply(ch, _s7()) is ch


def test_missing_posteriors_is_a_no_op(monkeypatch):
    monkeypatch.setenv(ff.ENV_ENABLE, "1")
    ch = _chords("G:maj", "C:maj")
    assert ff.apply(ch, None) is ch


def test_apply_to_audio_degrades_to_a_no_op(monkeypatch, tmp_path):
    """No clone / no such file must not raise — musx_bass's contract."""
    monkeypatch.setenv(ff.ENV_ENABLE, "1")
    ch = _chords("G:maj", "C:maj")
    assert ff.apply_to_audio(ch, tmp_path / "does_not_exist.m4a") is ch


# ── 3. the two witnesses ────────────────────────────────────────────────────

def test_both_witnesses_agree_flips_maj_to_dom(monkeypatch):
    """G before C resolves down a fifth AND the head votes b7 -> G7."""
    monkeypatch.setenv(ff.ENV_ENABLE, "1")
    ch = _chords("G:maj", "C:maj")
    out = ff.apply(ch, _s7(p=(0.1, 0.1, 0.7, 0.1)))
    assert out[0]["label"] == "G:7"
    assert out[0]["function_family_flip"] == "maj->7"


def test_context_alone_cannot_flip_when_the_head_opposes(monkeypatch):
    """G->C resolves, but the head is sure there is no b7: at w_ctx=0.5 the
    context must not be able to overturn it."""
    monkeypatch.setenv(ff.ENV_ENABLE, "1")
    ch = _chords("G:maj", "C:maj")
    out = ff.apply(ch, _s7(p=(0.9, 0.05, 0.03, 0.02)))
    assert out is ch


def test_head_alone_cannot_flip_when_the_context_opposes(monkeypatch):
    """C -> G does NOT resolve down a fifth (it goes UP a fifth).  A head that
    leans b7 by less than w_ctx nats must not be enough on its own.
    p = (.20,.21,.55,.04) -> log(.55/.41) = +0.29 nat, under the 0.5 the context
    subtracts."""
    monkeypatch.setenv(ff.ENV_ENABLE, "1")
    ch = _chords("C:maj", "G:maj")
    out = ff.apply(ch, _s7(p=(0.20, 0.21, 0.55, 0.04)))
    assert out[0]["label"] == "C:maj"


def test_dom_becomes_maj_when_it_does_not_resolve(monkeypatch):
    """A printed C7 that goes UP a fifth to G, with no b7 evidence, is a plain
    major chord — the dom->maj direction."""
    monkeypatch.setenv(ff.ENV_ENABLE, "1")
    ch = _chords("C:7", "G:maj")
    out = ff.apply(ch, _s7(p=(0.6, 0.3, 0.05, 0.05)))
    assert out[0]["label"] == "C:maj7"


def test_repeated_root_is_skipped_when_looking_ahead(monkeypatch):
    """G G C — the second G still resolves to C; a naive next-symbol lookup
    would see G->G and call it unresolved."""
    monkeypatch.setenv(ff.ENV_ENABLE, "1")
    ch = _chords("G:maj", "G:maj", "C:maj")
    out = ff.apply(ch, _s7(p=(0.1, 0.1, 0.7, 0.1)))
    assert [c["label"] for c in out[:2]] == ["G:7", "G:7"]


def test_resolves_down_a_fifth_semantics():
    assert ff.resolves_down_a_fifth([7, 0], 0) is True       # G -> C
    assert ff.resolves_down_a_fifth([0, 7], 0) is False      # C -> G
    assert ff.resolves_down_a_fifth([7, 7, 0], 0) is True    # repeat skipped
    assert ff.resolves_down_a_fifth([7], 0) is False         # end of chart
    assert ff.resolves_down_a_fifth([None, 0], 0) is False   # N span


# ── 4. what the brick must never touch ──────────────────────────────────────

def test_never_rewrites_the_root_or_the_bass(monkeypatch):
    monkeypatch.setenv(ff.ENV_ENABLE, "1")
    ch = _chords("G:maj/D", "C:maj")
    out = ff.apply(ch, _s7(p=(0.1, 0.1, 0.7, 0.1)))
    assert out[0]["label"] == "G:7/D"        # root G and bass /D preserved


@pytest.mark.parametrize("label", ["A:min7", "D:hdim7", "B:dim", "C:sus4",
                                   "F:7sus4", "N", "X"])
def test_other_families_are_untouched(monkeypatch, label):
    """Only tokens whose ONLY difference is the seventh degree may be rewritten:
    min/hdim/dim/sus and no-chord must pass through byte-identical."""
    monkeypatch.setenv(ff.ENV_ENABLE, "1")
    ch = _chords(label, "C:maj")
    out = ff.apply(ch, _s7(p=(0.1, 0.1, 0.7, 0.1)))
    assert out[0]["label"] == label


def test_no_chord_span_does_not_crash_the_lookahead(monkeypatch):
    monkeypatch.setenv(ff.ENV_ENABLE, "1")
    ch = _chords("G:maj", "N", "C:maj")
    out = ff.apply(ch, _s7(p=(0.1, 0.1, 0.7, 0.1)))
    assert out[0]["label"] == "G:7"          # the N is skipped, G still resolves
    assert out[1]["label"] == "N"


# ── 5. the knob ─────────────────────────────────────────────────────────────

def test_weight_default_and_env_override(monkeypatch):
    monkeypatch.delenv(ff.ENV_W, raising=False)
    assert ff.weight() == pytest.approx(ff.DEFAULT_W_CTX)
    monkeypatch.setenv(ff.ENV_W, "1.25")
    assert ff.weight() == pytest.approx(1.25)
    monkeypatch.setenv(ff.ENV_W, "not-a-float")
    assert ff.weight() == pytest.approx(ff.DEFAULT_W_CTX)


def test_w_ctx_zero_makes_it_the_head_alone(monkeypatch):
    """With the context switched off, a b7-leaning head flips a non-resolving
    major — which is exactly the regime measured at -0.50 pp per song, i.e. the
    reason the context is in the brick at all."""
    monkeypatch.setenv(ff.ENV_ENABLE, "1")
    ch = _chords("C:maj", "G:maj")
    out = ff.apply(ch, _s7(p=(0.20, 0.21, 0.55, 0.04)), w_ctx=0.0)
    assert out[0]["label"] == "C:7"


def test_s7_log_odds_is_a_segment_mean():
    """Half the segment screams b7, half screams none -> the MEAN posterior is
    balanced, so the log-odds must be ~0, not the midpoint frame's value."""
    a = np.zeros((100, 4), dtype=np.float32)
    a[:50] = [0.0, 0.0, 1.0, 0.0]
    a[50:] = [1.0, 0.0, 0.0, 0.0]
    z = ff.s7_log_odds(a, 0.0, 100 * ff.FRAME_DT)
    assert abs(z) < 1e-6


# ── 6. wiring invariant (parity-neutrality) ─────────────────────────────────
# The brick is wired default-ON in the live path, but the frozen parity oracle
# pins it OFF so its committed goldens (captured before the brick) stay
# byte-identical.  If this split ever collapses, the parity net either goes
# silently blind (oracle ON == goldens re-baked with the brick) or throws
# spurious failures (live ON vs OFF goldens) — so it is load-bearing.

def test_live_default_is_on_but_oracle_is_pinned_off():
    from harmonia.stages.chord_head import ChordHeadConfig
    from harmonia.eval.benchmark_set import LIVE_ORACLE_KWARGS
    from harmonia.eval.accuracy_score import SHIPPED_CONFIG

    # live/server default: ON (this is what reaches Louis's phone)
    assert ChordHeadConfig().function_family is True
    assert ChordHeadConfig.live_defaults().function_family is True
    # frozen parity oracle: pinned OFF (goldens predate the brick)
    assert ChordHeadConfig.from_infer_kwargs(
        **LIVE_ORACLE_KWARGS).function_family is False
    # benchmark config: ON, so the scoreboard reflects production
    assert ChordHeadConfig.from_infer_kwargs(
        **SHIPPED_CONFIG).function_family is True
