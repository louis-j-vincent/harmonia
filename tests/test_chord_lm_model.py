"""Tests for the chord-LM transformer itself (shapes, masking, positions)."""
from __future__ import annotations

import torch

from harmonia_min.chord_lm import vocab
from harmonia_min.chord_lm.model import ChordLM, LMConfig

SMALL = dict(d_model=64, n_layers=2, n_heads=4, d_ff=128, max_len=64)


def test_forward_shapes_both_position_schemes():
    for rope in (True, False):
        net = ChordLM(LMConfig(rope=rope, **SMALL))
        out = net(torch.randint(0, vocab.N_CHORD, (3, 17)))
        assert out.shape == (3, 17, vocab.VOCAB_SIZE)


def test_rope_is_translation_equivariant():
    """The whole point of rotary positions: the same 12 slots must score the
    same whether they sit at bar 1 or bar 21. That is what lets one attention
    pattern match a repeated section at any offset."""
    net = ChordLM(LMConfig(rope=True, causal=True, **SMALL)).eval()
    seq = torch.randint(0, vocab.N_CHORD, (1, 12))
    with torch.no_grad():
        a = net(seq, pos_ix=torch.arange(12).unsqueeze(0))
        b = net(seq, pos_ix=(torch.arange(12) + 40).unsqueeze(0))
    assert torch.allclose(a, b, atol=1e-5)


def test_absolute_positions_are_not_translation_equivariant():
    """Control for the test above — with a learned table the shift MUST matter,
    otherwise the equivariance result would be vacuous."""
    net = ChordLM(LMConfig(rope=False, causal=True, **SMALL)).eval()
    seq = torch.randint(0, vocab.N_CHORD, (1, 12))
    with torch.no_grad():
        a = net(seq, pos_ix=torch.arange(12).unsqueeze(0))
        b = net(seq, pos_ix=(torch.arange(12) + 40).unsqueeze(0))
    assert not torch.allclose(a, b, atol=1e-3)


def test_causal_model_cannot_see_the_future():
    net = ChordLM(LMConfig(rope=True, causal=True, **SMALL)).eval()
    x = torch.randint(0, vocab.N_CHORD, (1, 10))
    y = x.clone()
    y[0, 7] = (x[0, 7] + 5) % vocab.N_CHORD      # change a LATER slot
    with torch.no_grad():
        a, b = net(x), net(y)
    assert torch.allclose(a[0, :7], b[0, :7], atol=1e-5), "future leaked backwards"
    assert not torch.allclose(a[0, 7:], b[0, 7:], atol=1e-3)


def test_masked_model_does_see_the_future():
    net = ChordLM(LMConfig(rope=True, causal=False, **SMALL)).eval()
    x = torch.randint(0, vocab.N_CHORD, (1, 10))
    y = x.clone()
    y[0, 7] = (x[0, 7] + 5) % vocab.N_CHORD
    with torch.no_grad():
        a, b = net(x), net(y)
    assert not torch.allclose(a[0, 2], b[0, 2], atol=1e-3)


def test_padding_does_not_change_real_positions():
    """A padded batch must give the same logits as the unpadded sequence."""
    net = ChordLM(LMConfig(rope=True, causal=True, **SMALL)).eval()
    seq = torch.randint(0, vocab.N_CHORD, (1, 9))
    padded = torch.cat([seq, torch.full((1, 5), vocab.PAD)], dim=1)
    with torch.no_grad():
        a = net(seq)
        b = net(padded)
    assert torch.allclose(a, b[:, :9], atol=1e-5)


def test_from_checkpoint_defaults_legacy_cfg_to_absolute_positions(tmp_path):
    """A checkpoint saved before rotary existed has no `rope` key. Loading it
    with the current default (True) would silently give it a position signal it
    was never trained on."""
    net = ChordLM(LMConfig(rope=False, **SMALL))
    cfg = {k: v for k, v in net.cfg.__dict__.items() if k != "rope"}
    path = tmp_path / "legacy.pt"
    torch.save({"cfg": cfg, "state": net.state_dict()}, path)
    loaded = ChordLM.from_checkpoint(path)
    assert loaded.cfg.rope is False


def test_generate_respects_length():
    net = ChordLM(LMConfig(rope=True, causal=True, **SMALL))
    out = net.generate([vocab.chord_id(2, "min"), vocab.REP], 12)
    assert len(out) == 14
    assert vocab.PAD not in out
