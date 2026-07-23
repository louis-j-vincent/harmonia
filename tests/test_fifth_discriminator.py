"""Contract tests for the fifth_discriminator brick.

Load-bearing: disabled = exact no-op. Plus the one-note (4th-degree) decision and the
major-key / tonic / fifth-apart / confidence gates.
"""
import os

import numpy as np

from harmonia.models.fifth_discriminator import discriminate_fifth_root


def _chroma(pc, mass=0.6):
    c = np.full(12, (1.0 - mass) / 11.0)
    c[pc % 12] = mass
    return c


def test_disabled_is_noop():
    os.environ.pop("HARMONIA_FIFTH_DISC", None)
    # C(0) final vs G(7) nnls; F-natural present would favor C — but OFF -> unchanged
    assert discriminate_fifth_root(0, 7, _chroma(5)) == 0


def test_fnatural_picks_tonic_C_over_G():
    # candidates C(0) & G(7): R_low=0. F-natural=(0+5)=5 present -> disc>0 -> R_low=C.
    out = discriminate_fifth_root(7, 0, _chroma(5), tonic_pc=0, is_minor=False, force=True)
    assert out == 0  # switched G(final) -> C(nnls)


def test_fsharp_picks_dominant_G_over_C():
    # F#=(0+6)=6 present -> disc<0 -> R_high=G.
    out = discriminate_fifth_root(0, 7, _chroma(6), tonic_pc=0, is_minor=False, force=True)
    assert out == 7  # switched C(final) -> G(nnls)


def test_abstains_on_minor_key():
    out = discriminate_fifth_root(7, 0, _chroma(5), tonic_pc=0, is_minor=True, force=True)
    assert out == 7  # minor -> abstain, keep final


def test_abstains_when_tonic_not_a_candidate():
    # tonic D(2) is neither C nor G -> abstain
    out = discriminate_fifth_root(7, 0, _chroma(5), tonic_pc=2, is_minor=False, force=True)
    assert out == 7


def test_abstains_when_not_a_fifth():
    # C(0) & E(4) are a M3, not a fifth -> abstain
    out = discriminate_fifth_root(0, 4, _chroma(5), tonic_pc=0, is_minor=False, force=True)
    assert out == 0


def test_abstains_below_threshold():
    # flat chroma -> |disc| ~ 0 < thr -> abstain
    flat = np.full(12, 1.0 / 12)
    out = discriminate_fifth_root(7, 0, flat, tonic_pc=0, is_minor=False, force=True, thr=0.05)
    assert out == 7


def test_no_tonic_gate_allows_major_fifth():
    out = discriminate_fifth_root(7, 0, _chroma(5), tonic_pc=None, is_minor=False, force=True)
    assert out == 0


def test_env_enable():
    os.environ["HARMONIA_FIFTH_DISC"] = "1"
    try:
        assert discriminate_fifth_root(7, 0, _chroma(5), tonic_pc=0, is_minor=False) == 0
    finally:
        os.environ.pop("HARMONIA_FIFTH_DISC", None)
