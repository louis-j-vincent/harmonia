"""Contract tests for the root_resolve brick (harmonia/models/root_resolve.py).

Load-bearing safety property (brick pattern): disabled = exact no-op. Plus the
fifth-gate / confidence / key-diatonic decision logic under force=True.
"""
import os

import numpy as np

from harmonia.models.root_resolve import (
    key_diatonic_pcs, resolve_root,
)


def _p(root12, mass):
    p = np.full(12, (1.0 - mass) / 11.0)
    p[root12] = mass
    return p


def test_disabled_is_noop():
    os.environ.pop("HARMONIA_ROOT_RESOLVE", None)
    # G (7) final, C (0) nnls: a P5 apart, confident — but brick OFF -> unchanged
    assert resolve_root(7, 0, _p(0, 0.9)) == 7


def test_force_swaps_confident_diatonic_fifth():
    # final G(7), nnls C(0): (7-0)%12=7 (P5). C confident, diatonic to C major.
    out = resolve_root(7, 0, _p(0, 0.9), key_diatonic_pcs(0, False),
                       force=True, tau=0.35, margin=0.10)
    assert out == 0


def test_force_keeps_when_not_a_fifth():
    # final E(4), nnls C(0): (4-0)%12=4 (M3), not a fifth -> unchanged
    out = resolve_root(4, 0, _p(0, 0.9), key_diatonic_pcs(0, False), force=True)
    assert out == 4


def test_force_keeps_when_low_confidence():
    out = resolve_root(7, 0, _p(0, 0.20), key_diatonic_pcs(0, False),
                       force=True, tau=0.35, margin=0.10)
    assert out == 7  # nnls mass 0.20 < tau 0.35


def test_force_keeps_when_nnls_not_diatonic():
    # nnls root C#(1) not diatonic to C major -> keep final
    out = resolve_root(8, 1, _p(1, 0.9), key_diatonic_pcs(0, False), force=True)
    assert out == 8


def test_force_no_key_allows_swap():
    out = resolve_root(7, 0, _p(0, 0.9), None, force=True, tau=0.35, margin=0.10)
    assert out == 0


def test_env_enable():
    os.environ["HARMONIA_ROOT_RESOLVE"] = "on"
    try:
        assert resolve_root(7, 0, _p(0, 0.9), key_diatonic_pcs(0, False)) == 0
    finally:
        os.environ.pop("HARMONIA_ROOT_RESOLVE", None)


def test_key_diatonic_c_major():
    assert key_diatonic_pcs(0, False) == {0, 2, 4, 5, 7, 9, 11}


def test_key_diatonic_a_minor():
    assert key_diatonic_pcs(9, True) == {9, 11, 0, 2, 4, 5, 7}
