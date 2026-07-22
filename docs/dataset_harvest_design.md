# High-Precision Chord-Training Dataset — harvest design (2026-07-23 overnight mandate)

**Louis's overnight mandate:**
1. **Best downbeat tracker** — clean, modular, integrates with the new refactoring.
2. **Continue alignment.**
3. **A training dataset of song-PARTS 100%-aligned to GT chords**, harvested via our alignment
   tools. **PRECISION-FIRST**: very few false positives — a FP is a wrong chord label that poisons
   training. Some FPs are inevitable and that's fine (we'll catch them by analyzing where our model
   errs). Better 40% of a song at ~98% label-precision than 100% at 85%.
4. **Clean modular pipeline to add songs on demand** — iRealb charts + YouTube audio = a
   near-inexhaustible source.

## Principle: precision >> recall
Emit ONLY spans the alignment is CONFIDENT about; drop everything ambiguous. The dataset's value is
CLEAN (audio-segment → chord) pairs, not coverage.

## The confidence gate (the crux) — emit a segment as GT only if ALL hold (non-circular signals)
- harmonic agreement r over the span ≥ a HIGH, calibrated threshold;
- beat/downbeat lock high (Stage-1 drum tracker reliability + octave-locked; downbeat model once built);
- cross-repetition consistency (same section scores consistently across repeats);
- NO divergence flag (chart≠recording), NO gap/low-coverage in the span;
- transpose chosen with a clear margin.
**Calibrate the threshold on the FROZEN songs** (known-good alignment) → measure the gate's PRECISION
(fraction of emitted spans actually correctly labelled). Report the precision/recall tradeoff; pick the
operating point at very-high precision.

## Pipeline (modular) — `harmonia/dataset/`
- `ingest.py` — add a song = iReal chart (via `irealb_fetcher`, READ-ONLY import) + YouTube audio
  (clean standalone yt-dlp fetcher in this package) → cache.
- `harvest.py` — align (reuse the existing aligner, import don't edit) → confidence-gate per segment →
  emit (audio-segment ref, chord-sequence, metadata, confidence).
- `gate.py` — the strict per-segment confidence gate + its calibration.
- output: a dataset MANIFEST (segment refs + chord labels + confidence) — audio stays on disk, manifest
  references offsets (no giant blobs committed).
- entry: `add_song(chart_ref, youtube_ref)` — one call to grow the dataset.

## Boundary (concurrent-session safety — real risk per CLAUDE.md)
The serving/ refactor + `irealb_fetcher.py` + `harmonia_server.py` + `output/*` show CONCURRENT edits
from another session. I will NOT edit them. I build clean NEW modules (`harmonia/align/`,
`harmonia/dataset/`) with clean APIs + documented integration points; WIRING into the refactor is a
reconcile step for when the lanes converge — flagged for Louis, not done blind.

## Overnight deliverables (targets, honest)
1. **Downbeat model** `harmonia/align/downbeat.py` — fuse drum strong-beat-pair (Stage 1) + bass
   root-on-1 + chart/form periodicity + harmonic rhythm; validated downbeat accuracy on the 5 frozen +
   Autumn/Let It Be. Clean + modular + tested. (Blocked on the bass + 0b-bis premise results — in flight.)
2. **Dataset pipeline** `harmonia/dataset/` — ingest + harvest + gate, gate PRECISION calibrated on the
   frozen songs, demo harvest on available audio, manifest emitted.
3. **Morning summary** + honest precision numbers + what's done vs pending.

## Substitutions are a FEATURE, not noise — 3-bucket gate + ear-training loop (Louis, 2026-07-23)
Chart≠recording substitutions (Georgia's charted F#dim played as B7) are VALUABLE — a corpus to
learn reharmonizations/alterations, interesting work in itself. So the gate is **3-way**, not KEEP/DROP:
1. **Confident + chart MATCHES** → clean GT manifest (chart label). Precision-first bulk.
2. **Divergence — time-aligned but chart≠recording** (cross-rep uniformly-low agreement / mid-span
   split detector fire on a WELL-PLACED span) → `review_manifest.jsonl` (substitution/to-decide queue):
   {audio, t0, t1, chart_suggestion, divergence_evidence, confidence}. NEVER labelled with the wrong
   chart chord; queued for ear-labelling. NOT counted against bucket-1 precision.
3. **Low-confidence / misaligned** (low agreement not attributable to a clean substitution; gaps,
   ambiguous timing, weak transpose) → drop.
Discriminator 2-vs-3 = the EXISTING divergence detectors (well-placed-but-harmonically-off =
substitution; poorly-placed = drop). The bucket-2 queue feeds a future **EAR-TRAINING active-learning
mode**: Harmonia surfaces the chords it's unsure about + suggestions, the human decides, labels flow
back to improve the model — closing the loop from alignment → dataset → model → human → better model.
