# Scoping: extending the chord tree beyond 7ths (9 / 11 / 13 / alt)

**Question.** The chord tree currently descends family → base7 → *exact*, where
"exact" tops out at 7th-level qualities (`maj7`, `dom7`, `m7b5`, `dim7`, `6`,
`sus`, …). The renderer's level menu therefore stops at "exact". Can we add a
level-4 "extensions" node (9 / 11 / 13 / b9 / #9 / #11 / b13 / alt) and expose
it in the chart?

This doc is the cheap premise-screen (process rule #2), not an implementation.

## 1. Label supply (accomp DB, 110,250 chord instances)

Only **10.7%** of chord instances carry any extension, and the classes are
severely imbalanced:

| tension | count | share |
|---------|------:|------:|
| 9       | 3684  | 3.34% |
| b9      | 3272  | 2.97% |
| #11     | 1883  | 1.71% |
| 13      | 1110  | 1.01% |
| #9      | 1073  | 0.97% |
| b13     |  649  | 0.59% |
| add     |  620  | 0.56% |
| 11      |  588  | 0.53% |

So ~89% of chords have *no* extension, and the informative tail (13/#9/11/b13)
is each <1%. Any level-4 classifier is a rare-event, long-tailed problem — it
must default hard to "no extension" and only fire on strong evidence.

Reproduce: `scripts/probe_extensions.py`.

## 2. Audio recoverability — the load-bearing question

An extension adds essentially **one pitch class** to the chord (a 9th adds the
2nd degree, a 13th the 6th, etc.). The question is whether that single added pc
rises above the chroma noise floor. Measured on the root-rotated feature cache
(`data/cache/audio_chord_features.npz`), for chords that carry **no** extension,
the mean energy at each tension pc as a fraction of the chord-tone level:

| feature | 9th (pc2) | 11th (pc5) | #11 (pc6) | 13th (pc9) | b9 (pc1) |
|---------|----------:|-----------:|----------:|-----------:|---------:|
| onset (dom7)  | 33% | 58% | **16%** | 23% | 38% |
| treble (dom7) | 37% | 80% | **9%**  | 10% | 15% |
| bass (dom7)   | 34% | 49% | 21% | 31% | 51% |

(The smeared `note` = Basic-Pitch activation feature is nearly flat — chord
tones only 0.085 vs 0.083 uniform — and is useless for this; `onset`/`treble`
separate chord tones at ~2× uniform and are what matter.)

**Read-out.** The natural tensions (9, 11, 13) already sit at a **30–58%
incidental floor** in un-extended chords — passing tones, melody and reverb put
energy there anyway — so a *deliberate* 9/11/13 is only a modest bump above that
floor and will be hard to separate from audio. In contrast **#11 has a clean
floor (~9–21%)**: an altered tension is a strong, unusual chroma signature and
*is* audio-recoverable. b9 is intermediate.

## 3. Conclusion / recommended scope

Extensions split into two regimes:

- **Altered tensions (#11, #9, b9, alt)** — cleaner chroma signature, worth an
  audio classifier. #11 especially.
- **Natural extensions (9, 11, 13, add)** — near the noise floor; **predict
  these from a symbolic prior, not audio.** The pieces already exist:
  `scripts/train_progression_lm.py` (P(quality | prev, root-motion)),
  the key prior, and `scripts/experiment_style_grammar.py`. A 9/13 is largely a
  *voicing/style* choice, so a style-conditioned prior is the right instrument
  — consistent with the project's finding that priors carry quality where audio
  degrades (`docs/priors_when_audio_degrades_2026-07-05.md`).

**Proposed implementation (when prioritized):**
1. Add an `ext` label per feature row in `build_audio_chord_features.py`
   (9/11/13/b9/#9/#11/b13/alt/none), derived from the iReal token that already
   backs each row. ~one afternoon; unblocks everything else.
2. Train a level-4 classifier **only on the altered set** (audio) + a
   **style/progression prior** for natural extensions; combine as the existing
   levels do. Gate with the same confidence threshold.
3. The renderer is **already extension-ready**: `infer_song` emits a per-chord
   `levels` dict and the chart/interactive code keys off it. Adding
   `levels["extended"] = {ireal, conf}` and one more `<option>` in the level
   menu is all the UI needs — no typesetting work (the `_typeset_quality` /
   `chord_html` path already renders `7b9`, `^9`, `13#11`, etc.).

**Risk to flag (process rule #4):** with only 0.5–1% support for the tail
classes, a level-4 model will look good on aggregate accuracy while being
essentially a "no-extension" predictor. Report per-class recall on the tail, not
just overall accuracy, or the win is illusory.
