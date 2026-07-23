"""Unit tests for the fusion aligner's pure, audio-free core (harmonia.align.fusion).

These drive the numpy-only fusion math with synthetic streams — the reliability-
weighted harmony curve, the bass/harmonic-rhythm curves, the bar-pointer Viterbi
MAP, the forward-backward posterior confidence, and the anticipation-aware downbeat
refinement. The audio front-end (``align_fusion`` / ``extract_features``) is covered
by the scratchpad validation harness against the frozen goldens, not here."""
import numpy as np
import pytest

from harmonia.align import fusion as F
from harmonia.align.fusion import _centre_norm


# ── a tiny synthetic section type (matches the duck-typed brick0 Section) ──────
class FakeSection:
    def __init__(self, template, label, content_key=None, beat_chords=None):
        self.template = np.asarray(template, float)
        self.n_beats = len(self.template)
        self.label = label
        self.content_key = content_key or (label,)
        self.beat_chords = beat_chords


def _triad(pcs):
    v = np.zeros(12)
    v[list(pcs)] = 1.0
    return v


def _planted_grid(N, seed=0):
    """A grid with C-major planted at 0,8,16 and G-major at 4,12,20."""
    rng = np.random.default_rng(seed)
    cn = rng.standard_normal((N, 12)) * 0.05
    A = np.tile(_triad([0, 4, 7]), (4, 1))
    B = np.tile(_triad([7, 11, 2]), (4, 1))
    for p in (0, 8, 16):
        cn[p:p + 4] = _centre_norm(A)
    for p in (4, 12, 20):
        cn[p:p + 4] = _centre_norm(B)
    return cn, A, B


# ═══════════════════════════ stream 1: harmony ════════════════════════════════

def test_fused_agreement_curve_peaks_where_planted():
    cn, A, B = _planted_grid(24)
    ha = F.fused_agreement_curve(A, cn)
    hb = F.fused_agreement_curve(B, cn)
    assert int(np.argmax(ha)) == 0
    assert int(np.argmax(hb)) == 4
    assert set(np.argsort(ha)[::-1][:3]) == {0, 8, 16}
    assert set(np.argsort(hb)[::-1][:3]) == {4, 12, 20}


def test_uniform_salience_equals_unweighted():
    """With uniform salience the reliability-weighted curve == the plain mean —
    the property that guarantees the frozen alignments reproduce."""
    cn, A, _ = _planted_grid(24)
    plain = F.fused_agreement_curve(A, cn)
    unif = F.fused_agreement_curve(A, cn, salience=np.ones(len(cn)))
    fin = np.isfinite(plain)
    assert np.allclose(plain[fin], unif[fin])


def test_salience_downweights_blind_beats():
    """A spuriously-matching beat with ~0 salience should not inflate the score —
    the Autumn 'cacahuète' fix mechanism."""
    cn, A, _ = _planted_grid(24)
    # corrupt beat 2 to a strong but WRONG match while zeroing its salience
    cn2 = cn.copy()
    cn2[2] = _centre_norm(_triad([1, 5, 8])[None, :])[0]
    sal = np.ones(len(cn2))
    sal[2] = 0.0
    weighted = F.fused_agreement_curve(A, cn2, salience=sal)
    unweighted = F.fused_agreement_curve(A, cn2, salience=None)
    # down-weighting the corrupt beat recovers a HIGHER agreement at the true start
    assert weighted[0] > unweighted[0]


def test_agreement_curve_out_of_range_is_neg_inf():
    cn, A, _ = _planted_grid(24)
    ha = F.fused_agreement_curve(A, cn)
    assert ha[-1] == float("-inf")           # start too late to fit L beats
    assert np.isfinite(ha[24 - 4])


# ═══════════════════════════ stream 3: bass ═══════════════════════════════════

def test_bass_match_curve_matches_and_reliability_gates():
    N = 12
    # audio bass pc = [0,0,0,0, 7,7,7,7, 0,0,0,0]; chart section bass = [0,0,0,0]
    audio_bass = np.array([0, 0, 0, 0, 7, 7, 7, 7, 0, 0, 0, 0])
    rel = np.ones(N)
    curve = F.bass_match_curve([0, 0, 0, 0], N, audio_bass, rel)
    assert curve[0] == pytest.approx(1.0)     # perfect match at 0
    assert curve[4] == pytest.approx(0.0)     # G span, chart says C -> no match
    assert curve[8] == pytest.approx(1.0)


def test_bass_match_zero_reliability_is_zero():
    N = 8
    audio_bass = np.zeros(N, int)
    rel = np.zeros(N)                          # walking/absent bass -> no weight
    curve = F.bass_match_curve([0, 0, 0, 0], N, audio_bass, rel)
    assert curve[0] == pytest.approx(0.0)      # den==0 -> 0 evidence, not a crash


def test_bass_match_skips_none_beats():
    N = 8
    audio_bass = np.array([0, 3, 0, 3, 0, 3, 0, 3])
    rel = np.ones(N)
    # only beats 0 and 2 carry a chart bass (0); beats 1,3 are no-chord (None)
    curve = F.bass_match_curve([0, None, 0, None], N, audio_bass, rel)
    assert curve[0] == pytest.approx(1.0)      # both charted beats match


# ═══════════════════════════ stream 4: harmonic rhythm ════════════════════════

def test_hr_match_curve_rewards_changes_on_flux():
    N = 12
    flux = np.zeros(N)
    flux[[0, 4, 8]] = 1.0                       # audio chord-changes at 0,4,8
    # a 4-beat section that changes at offset 0 only
    curve = F.hr_match_curve([0], 4, N, flux)
    assert curve[0] == pytest.approx(1.0)       # change lands on a flux peak
    assert curve[1] == pytest.approx(0.0)


def test_hr_match_no_changes_is_flat_zero():
    N = 8
    flux = np.ones(N)
    curve = F.hr_match_curve([], 4, N, flux)     # section with no chord changes
    assert curve[0] == pytest.approx(0.0)
    assert curve[4] == pytest.approx(0.0)


# ═══════════════════════════ fuse + Viterbi ═══════════════════════════════════

def test_fuse_emissions_weights_and_preserves_neg_inf():
    harm = np.array([0.5, 0.3, float("-inf")])
    bass = np.array([1.0, 0.0, 1.0])
    hr = np.array([1.0, 1.0, 1.0])
    out = F.fuse_emissions(harm, bass, hr, w_bass=0.1, w_hr=0.1)
    assert out[0] == pytest.approx(0.5 + 0.1 + 0.1)
    assert out[1] == pytest.approx(0.3 + 0.0 + 0.1)
    assert out[2] == float("-inf")


def test_viterbi_places_sections_contiguously():
    cn, A, B = _planted_grid(24)
    secA, secB = FakeSection(A, "A"), FakeSection(B, "B")
    order, emits = [], []
    for c in range(4):
        for si, s in ((0, secA), (1, secB)):
            order.append((c, si, s))
            emits.append(F.fused_agreement_curve(s.template, cn))
    pls = F.viterbi_bar_pointer(order, emits, len(cn), 0, min_gap_beats=6)
    assert [p.start_beat for p in pls] == [0, 4, 8, 12, 16, 20]
    assert [p.label for p in pls] == ["A", "B", "A", "B", "A", "B"]


def test_viterbi_opens_a_large_gap_over_a_vamp():
    """A real multi-beat vamp with no chart material forces the large-gap branch."""
    rng = np.random.default_rng(1)
    N = 20
    A = np.tile(_triad([0, 4, 7]), (4, 1))
    cn = rng.standard_normal((N, 12)) * 0.05
    # A at 0-3 and again at 12-15 (an 8-beat vamp gap between them, no chart match)
    for p in (0, 12):
        cn[p:p + 4] = _centre_norm(A)
    secA = FakeSection(A, "A")
    order = [(0, 0, secA), (1, 0, secA)]
    emits = [F.fused_agreement_curve(secA.template, cn)] * 2
    pls = F.viterbi_bar_pointer(order, emits, N, 0, min_gap_beats=4, gap_open=0.1)
    assert [p.start_beat for p in pls] == [0, 12]     # the second A jumped the vamp


def test_viterbi_drops_noise_tail():
    N = 24
    A = np.tile(_triad([0, 4, 7]), (4, 1))
    rng = np.random.default_rng(3)
    cn = rng.standard_normal((N, 12)) * 0.05
    for p in (0, 4, 8):                         # 3 contiguous matching A's, rest noise
        cn[p:p + 4] = _centre_norm(A)
    secA = FakeSection(A, "A")
    order = [(c, 0, secA) for c in range(6)]
    emits = [F.fused_agreement_curve(secA.template, cn)] * 6
    pls = F.viterbi_bar_pointer(order, emits, N, 0, min_gap_beats=6)
    assert [p.start_beat for p in pls] == [0, 4, 8]     # noise tail dropped
    assert all(p.emission >= F._MIN_FIT for p in pls)


# ═══════════════════════════ forward-backward confidence ══════════════════════

def test_confidence_high_when_peaked_low_when_flat():
    cn, A, B = _planted_grid(24)
    secA, secB = FakeSection(A, "A"), FakeSection(B, "B")
    order, emits = [], []
    for c in range(3):
        for si, s in ((0, secA), (1, secB)):
            order.append((c, si, s))
            emits.append(F.fused_agreement_curve(s.template, cn))
    pls = F.viterbi_bar_pointer(order, emits, len(cn), 0, min_gap_beats=6)
    eby = {i: emits[i] for i in range(len(pls))}
    confs = F.forward_backward_confidence(pls, eby, len(cn))
    assert min(confs) > 0.6                       # all well-determined -> high

    # a FLAT emission (no information) -> diffuse posterior -> low confidence
    flat = {i: np.zeros(len(cn)) for i in range(len(pls))}
    confs_flat = F.forward_backward_confidence(pls, flat, len(cn))
    assert max(confs_flat) < min(confs)           # strictly less determined


def test_confidence_empty_placements():
    assert F.forward_backward_confidence([], {}, 10) == []


# ═══════════════════════════ downbeat refinement ══════════════════════════════

def test_downbeat_agreement_is_confident():
    r = F.refine_downbeat_phase(form_phase=1, acoustic_phase=1, acoustic_conf=0.8)
    assert r.phase == 1
    assert not r.anticipation and not r.flagged
    assert r.confidence > 0.8


def test_downbeat_anticipation_prefers_form():
    """Acoustic one beat off form = the pop push (Every Breath): keep the FORM
    phase, mark anticipation, do NOT flag."""
    r = F.refine_downbeat_phase(form_phase=1, acoustic_phase=2, acoustic_conf=0.5)
    assert r.phase == 1                            # form wins
    assert r.anticipation and not r.flagged


def test_downbeat_halfbar_disagreement_flags():
    r = F.refine_downbeat_phase(form_phase=0, acoustic_phase=2, acoustic_conf=0.5)
    assert r.flagged
    assert r.phase == 0                            # form kept as best guess


def test_downbeat_wraparound_is_anticipation():
    """acoustic 0 vs form 3 is a 1-beat (wrap) difference, not a half-bar."""
    r = F.refine_downbeat_phase(form_phase=3, acoustic_phase=0, acoustic_conf=0.5)
    assert r.anticipation and not r.flagged
