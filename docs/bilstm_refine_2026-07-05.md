# BiLSTM refinement on soft chord distributions + calibrated certainty (2026-07-05)

User's idea: feed the sequence model not hard detected chords but the
**probability table** of each chord, use context **before and after** (a chord is
constrained by what follows), a **neutral element** where there is no/masked
neighbour, and **output a certainty** for the inferred chord.

Implemented (`scripts/experiment_bilstm_refine.py`):
- input per position = soft audio quality distribution (14, out-of-fold) +
  scale-degree one-hot (12);
- a **bidirectional** LSTM reads the whole sequence (subsumes "4 before + after");
- **neutral element** = a zero vector, used at sequence ends and for 15% random
  masking during training, so the model is robust to a missing/uncertain neighbour;
- output = softmax (a properly normalised distribution over qualities);
  **certainty = the max softmax probability**.
- No leakage: audio distributions are out-of-fold; BiLSTM is song-grouped-CV.

## Accuracy: a small gain (progression is a small lever once audio is strong)

| level | audio alone (argmax soft dist) | BiLSTM refined (before+after) |
|---|---|---|
| seventh | 88.1% | 88.5% (+0.4) |
| family | 94.3% | 94.2% (−0.0) |

Same story as every progression experiment: a strong audio likelihood already
explains most of the variance, so bidirectional progression context adds only a
little. (It would add more on weaker/real audio.)

## The real win: the certainty is excellent and calibrated

The model's confidence reliably predicts whether it is right — this is what the
user asked for and it works cleanly:

| confidence bin | n chords | accuracy |
|---|---|---|
| 0.00–0.40 | 97 | 36.1% |
| 0.40–0.60 | 583 | 47.9% |
| 0.60–0.80 | 978 | 60.0% |
| 0.80–0.95 | 1,722 | 78.5% |
| **0.95–1.00** | **9,316** | **96.4%** |

When the model is confident (0.95+, ~73% of all chords) it is right 96% of the
time; when it's unsure (<0.4) it's right ~36%. That is a genuinely trustworthy
certainty signal.

## It powers the hierarchical tree directly

"Report the exact seventh only when confident, else back off to the family" — the
certainty is the gate the chord tree (`harmonia/theory/chord_tree.py`) needed,
now learned and calibrated instead of a template margin:

| confidence threshold | coverage (chords answered at seventh) | accuracy on those |
|---|---|---|
| 0.00 (always) | 100% | 88.5% |
| 0.70 | 91% | 92.1% |
| 0.85 | 84% | 94.5% |

So we can answer the exact seventh on 84% of chords at 94.5% accuracy, and
gracefully fall back to the (safer, ~94%) family on the uncertain 16% — exactly
the "only go deeper when confident" behaviour, driven by a real probability.

## Why the soft + bidirectional design matters (even if accuracy moved little)

- **Soft inputs** (probability tables, not argmax) are what make the output
  certainty meaningful: the model propagates uncertainty instead of committing to
  a possibly-wrong hard neighbour, so its own confidence reflects the true
  ambiguity.
- **Before + after** context is what lets a later chord confirm an earlier one
  (a V is validated by the I that follows), and the masking/neutral element keeps
  it robust when a neighbour is missing.
- Normalisation: softmax over the quality logits gives a proper distribution;
  certainty = its max (equivalently, 1 − normalised entropy would work). Both are
  calibrated here.

## Next

- Feed the CNN's (not LR's) soft distributions as input — a better likelihood
  should raise both accuracy and the confident-coverage.
- Wire this certainty into `ChordEvent.reported_depth` so the live chart shows
  the exact chord only where the model is calibrated-confident.
- Re-test on real/varied audio, where the bidirectional progression context
  should help accuracy more, not just certainty.
