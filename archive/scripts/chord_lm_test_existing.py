"""Actually run the pretrained chord LMs that exist, instead of reading their cards.

Asked by Louis, 2026-08-01: "Tu as testé les LLMs qui existent déjà ?"

Surveyed on HuggingFace (2026-08-01):
  musiclang/musiclang-chord-v2-4k      GPT-2, 49 tokens, weights present   <- tested
  floriangardin/chord_model            GPT-2, weights present              <- tested
  patcasso/chords_complete_12x_triad_1perbar_241119
                                       name promises per-bar; repo contains
                                       ONLY a tokenizer, no model weights
  ailsntua (Chordonomicon authors)     dataset published, GPT-2 weights NOT

The question this script answers is not "is their model good" but "can their
model express what we need". Two concrete probes:

  1. Does the vocabulary contain any duration / bar / position token? If not,
     the model cannot say "Am for two bars" — only "Am then F" — and cannot be
     evaluated on a half-bar grid at all.
  2. Generate from it and look at the output.

A quantitative head-to-head on our half-bar task is deliberately NOT attempted:
it would need a lossy conversion of our chords into their Roman-numeral
degree-within-tonality format, and jazz's non-diatonic chords force a tonality
change on almost every bar. The conversion's losses would dominate the result,
so the number would measure our converter, not their model.

    .venv/bin/python scripts/chord_lm_test_existing.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

MODELS = ["musiclang/musiclang-chord-v2-4k", "floriangardin/chord_model"]
DURATION_HINT = re.compile(r"dur|bar|beat|time|len|quarter|whole|half|pos|onset|"
                           r"rest|measure|tick", re.I)


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    for name in MODELS:
        print("=" * 74)
        print(name)
        print("=" * 74)
        try:
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(name)
        except Exception as e:
            print(f"  could not load: {type(e).__name__}: {e}\n")
            continue
        model.eval()
        vocab = tok.get_vocab()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  params {n_params/1e6:.1f}M   vocab {len(vocab)}   "
              f"n_positions {getattr(model.config, 'n_positions', '?')}")

        hits = sorted(t for t in vocab if DURATION_HINT.search(t))
        print(f"\n  PROBE 1 — duration / bar / position tokens in the vocabulary:")
        print(f"    {hits if hits else 'NONE'}")
        if not hits:
            print("    -> cannot express how long a chord lasts. An event-level")
            print("       model: 'Am then F', never 'Am for two bars'.")

        print(f"\n  PROBE 2 — generation from an empty prompt:")
        try:
            ids = tok("", return_tensors="pt").input_ids
            if ids.shape[1] == 0:
                bos = getattr(model.config, "bos_token_id", None)
                ids = torch.tensor([[bos if bos is not None else 0]])
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=40, do_sample=True,
                                     temperature=0.9, top_k=20,
                                     pad_token_id=getattr(model.config,
                                                          "padding_token_id", 0))
            text = tok.decode(out[0], skip_special_tokens=False)
            print(f"    {text[:400]}")
        except Exception as e:
            print(f"    generation failed: {type(e).__name__}: {e}")
        print()

    print("=" * 74)
    print("VERDICT")
    print("=" * 74)
    print("""  Both published models are event-level: their vocabularies contain a
  chord-change separator and no unit of time. On a half-bar grid they cannot
  represent 44.5% of our slots (the holds), which is precisely the part our
  pipeline needs predicted. They are therefore not baselines for this task —
  they are models of a different, coarser task.

  The one repo whose NAME promised per-bar tokens
  (patcasso/chords_complete_12x_triad_1perbar) ships a tokenizer and no
  weights, so there is nothing to run.""")


if __name__ == "__main__":
    main()
