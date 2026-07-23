"""Contract tests for the no-chord-policy brick (harmonia/models/no_chord_policy.py).

The load-bearing safety property (CLAUDE.md brick pattern): the env flag OFF is a
behavioural no-op — the wrapped mask is bit-identical to the input, so wiring the
hook into _infer_nnls24 changes nothing until explicitly enabled.
"""
import os

import numpy as np
import pytest

from harmonia.models.no_chord_policy import gated_no_chord_mask, no_chord_policy


def test_off_is_identity():
    m = np.array([True, False, True, False])
    out = no_chord_policy(m, mode="off")
    assert np.array_equal(out, m)


def test_suppress_is_all_false():
    m = np.array([True, True, False])
    out = no_chord_policy(m, mode="suppress")
    assert out.dtype == bool
    assert not out.any()
    assert out.shape == m.shape


def test_intersect_requires_both():
    musx = np.array([True, True, False, False])
    nnls = np.array([True, False, True, False])
    out = no_chord_policy(musx, nnls, mode="intersect")
    assert np.array_equal(out, np.array([True, False, False, False]))


def test_intersect_without_nnls_falls_back_to_off():
    musx = np.array([True, False, True])
    out = no_chord_policy(musx, None, mode="intersect")
    assert np.array_equal(out, musx)


def test_unknown_mode_passthrough():
    m = np.array([True, False])
    assert np.array_equal(no_chord_policy(m, mode="banana"), m)


def test_env_gate_default_off_is_noop():
    os.environ.pop("HARMONIA_NC_POLICY", None)
    m = np.array([True, False, True])
    assert np.array_equal(gated_no_chord_mask(m), m)


def test_env_gate_suppress():
    os.environ["HARMONIA_NC_POLICY"] = "suppress"
    try:
        m = np.array([True, False, True])
        assert not gated_no_chord_mask(m).any()
    finally:
        os.environ.pop("HARMONIA_NC_POLICY", None)
