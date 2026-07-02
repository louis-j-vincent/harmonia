# Handoff: key inference — full technical detail (2026-07-02)

This is the detailed reference document behind the top-priority item in
`docs/known_issues.md` #0. It exists so nothing from this session's
investigation gets lost in summarization — the prompt that points here is
intentionally short; this file has the numbers, the exact reasoning chains,
and the dead ends, in full. Read this before starting, not just the prompt.

Order of discovery matters here — each finding led to the next, and
understanding *why* we went looking for the next thing is as useful as the
finding itself.

---

## 1. Starting point: chasing "why did key_prior_per_beat hurt song 001"

Session context: `docs/known_issues.md` issue #1 (chord-change temporal
resolution) had just gone through three rejected fix candidates (A: emission
preprocessing, B: explicit-duration decoding, C: periodicity folding), all
converging on the same diagnosis — per-beat emission evidence can't reliably
discriminate chord *qualities* that share most of their template (root/majmin
metrics stayed roughly stable across the whole set of experiments, but
majmin/tetrads consistently collapsed).

While zooming into a diagnostic plot to investigate this further (see §2),
two independent, more foundational bugs were found and fixed first:

### 1a. `BASIC_PITCH_FRAME_RATE` off by exactly 2x

`harmonia/models/stage1_pitch.py` had `BASIC_PITCH_FRAME_RATE = 43.066`.
Basic Pitch's own package constants (`basic_pitch.constants`):
`AUDIO_SAMPLE_RATE = 22050`, `FFT_HOP = 256`, giving a real frame rate of
`22050/256 = 86.1328125` — our constant was exactly half.

Confirmed empirically before fixing: for
`data/renders/pop909/001/001_v000_prog0.wav` (real duration, via
`soundfile.info()`: 198.12s), the pipeline's computed `duration_s`
(`frame_times[-1]`) was **395.58s** — almost exactly double. Implied true
frame rate from `n_frames / real_duration = 17037 / 198.12 = 85.99 Hz`,
matching `86.1328125` almost exactly (small rounding).

**Impact:** `frame_times` were computed 2x too large everywhere, which
corrupted `harmonia/models/rhythm.py::BeatGrid.quantise_frames()` — beat
buckets are built from the beat grid's real-time boundaries
(`beat_times`, correctly derived from real audio by librosa), so any frame
whose (wrongly doubled) `frame_time` exceeded the beat grid's real max time
(~190s) could never fall inside any bucket. In practice this meant every
song's beat-level observations were built almost entirely from the *first
half* of that song's real audio (compressed 2x across the full beat-index
range), while the genuine second half of every song was silently dropped.

Fixed (`harmonia/models/stage1_pitch.py`): constant corrected to
`86.1328125`, with a comment tracing the derivation and the bug's history so
it can't silently regress. **Cleared `data/cache/*.npz`** — cached
`PitchActivations` had the wrong `frame_times` baked in, and the cache key
(`_cache_key`) never covered this constant, so stale cache would have
silently kept serving corrupted data even after the code fix.

Added `tests/test_stage1_pitch.py` — exactly the test that should have
existed from the start:
- `TestFrameRateConstant`: pure constants comparison against
  `basic_pitch.constants`, no audio/model needed, instant.
- `TestComputedDurationMatchesRealAudio`: one real audio file,
  `soundfile.info()` duration vs `PitchExtractor.extract().duration_s`,
  `pytest.approx` with 2s tolerance.

**Measured impact once combined with the soundfont fix below:** 5-song mean
root accuracy 21.5% → 35.5%, majmin 15.4% → 27.1%. Bigger than anything the
issue #1 A/B/C investigation achieved.

### 1b. Mislabeled soundfont

`data/soundfonts/GeneralUser.sf2` — running `strings` on it revealed its
real embedded name: `"Vintage Dreams Waves v 2.0"` (Ian Wilson, 1996,
307KB) — not the real GeneralUser GS the filename claimed. Every render used
throughout the whole project had been synthesized with this low-fidelity
file. Downloaded a real GM soundfont
(`https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf2`,
215MB, MuseScore's own, well-regarded) → `data/soundfonts/MuseScore_General.sf2`.
Re-rendered all 5 songs: `data/renders/pop909/{001..005}/{id}_v005_musescoregeneral.wav`.
**Use these renders going forward, not `_v000_prog0.wav`.**

### 1c. GT chord overlay plotting bug (minor, cosmetic, but relevant to §2)

`scripts/plot_inference_diagnostics.py::plot_beat_note_probs()` was using
`ev.start_beat` (which — known gotcha — is actually **seconds**, not a beat
index, in `POP909Song.ChordEvent`) directly as an x-axis beat-index pixel
position. At this song's tempo (~89 BPM, 0.68s/beat), beats accumulate
faster than seconds, so this systematically misplaced every GT overlay line
and, critically, silently dropped every line past roughly 2/3 of the way
through the plot (once the wrong "beat index" — literally the rounded
seconds value — exceeded the axis width). Fixed: convert via
`np.searchsorted(beat_times, ev.start_beat)` (the real beat grid), not
`round(ev.start_beat)`.

---

## 2. The third-weakness finding

With both bugs above fixed, a new diagnostic script was built:
`scripts/plot_note_probs_vs_gt.py` — a 3-panel figure (note-probability
heatmap zoomed to a piano key range, a chroma [12 pitch-class] summary
panel, and a GT chord label panel), all sharing a real time x-axis with
gridlines for direct vertical comparison. See
`docs/plots/inference/pop909_001/note_probs_vs_gt_C2_C5.png`.

Using this, inspected a single, clean, unambiguous beat (t=12.10s, GT chord
= `F#:maj`, well inside a long stable region) and looked at raw salience for
the full 88-key range. Top hits were overwhelmingly F#/C#/Bb (=A#) across
octaves — i.e. the F# major triad (root, fifth, third respectively) — with
root and fifth strongly dominant and the third present but visibly weaker.

To check this wasn't a one-off, checked 6 separate instances of the same
`F#:maj` chord at different points in the song, folding each beat's 88-key
salience into a 12-pitch-class chroma and reading off root/third/fifth
energy directly (root/third pitch classes computed via `(root_pc + interval)
% 12` for `[0, 4, 7]` = major triad intervals):

| Instance | root | third | fifth | third/root |
|---|---|---|---|---|
| 1 | 7.99 | 2.49 | 5.43 | 0.31 |
| 2 | 10.42 | 4.41 | 10.33 | 0.42 |
| 3 | 14.44 | 3.78 | 14.29 | 0.26 |
| 4 | 10.08 | 2.44 | 9.94 | 0.24 |
| 5 | 18.17 | 5.04 | 16.56 | 0.28 |
| 6 | 13.40 | 4.97 | 10.74 | 0.37 |

**The third is consistently 24-42% of the root's salience, every single
time.** Root and fifth are comparably strong to each other. This directly
explains the whole-session pattern from the issue #1 investigation: major
vs. minor differ *only* in the third (major = root+4 semitones, minor =
root+3), so the one acoustic feature that would let the model tell them
apart is systematically the weakest, noisiest part of the signal. Root
accuracy (which doesn't depend on the third at all) stayed roughly stable
across every experiment in the whole investigation; majmin/tetrads
(quality-sensitive) kept collapsing. This is very likely just how these
piano voicings are played/rendered — root and fifth carry more energy
(stronger low harmonics, more commonly doubled), thirds are often a softer
inner voice. **Not yet confirmed on minor chords or other songs** — the 6
examples above are all the same major chord in one song. Worth checking
generality if it becomes relevant again.

---

## 3. `key_prior_per_beat`: the fix, and the puzzle it created

**The architectural gap:** `build_key_prior()` in `harmonia/models/chord_hmm.py`
correctly computes a diatonic-quality-boosted log-prior over all
(root, quality) pairs for a given key — e.g. confirmed directly: for F#
major (`tonic=6`), `F#maj` gets `log(3.0) = 1.0986`, `F#min` gets `log(1.0)
= 0.0`, a genuine 3x boost, computed correctly. But this prior
(`log_key_prior` / `log_init`) was **only ever used as the Viterbi initial
distribution** — `viterbi_mat[0] = log_init + log_emission[0]` — voting on
the first beat of a structural segment and never again, despite segments
running 10-40 beats. Grepped to confirm: `log_init` only appears in
`viterbi()`'s `t=0` initialization and `viterbi_duration_aware()`'s `s==0`
base case, nowhere else.

**The fix:** added `ChordInferrer(key_prior_per_beat: bool, key_prior_weight:
float)`. When enabled, the key prior is added as an ongoing per-beat bias to
`log_obs` for beats `1..T-1` (deliberately *not* beat 0, which already gets
it via `log_init` — this makes `key_prior_weight=0` an exact no-op relative
to the flag being off, matching the no-op behavior of every other tunable
weight in this class, e.g. `periodicity_weight`). Now the `HarmoniaPipeline`
default (`key_prior_per_beat=True`).

Implementation detail worth knowing: the weight saturates almost
immediately — `key_prior_weight = 1, 3, 10` all gave byte-identical decoded
paths on song 001. This isn't a dial to fine-tune; it's closer to a binary
"does forcing diatonic-quality assumptions help or hurt for this song's
actual harmony" switch, at least at the tested scale.

**Result across the 5-song set** (both fixes from §1 already applied):

| | boundary F | root | majmin | 7ths |
|---|---|---|---|---|
| baseline (post frame-rate + soundfont fix) | 0.275 | 35.5% | 27.1% | 12.2% |
| + `key_prior_per_beat` (w=1) | 0.276 | 33.0% | **29.6%** | **17.9%** |

Net positive for majmin/7ths (the metrics this whole line of investigation
has been trying to move), flat on boundary F, but root dropped ~2.5pp on
average — **driven almost entirely by song 001**, which regressed sharply:

| song | metric | baseline | +key_prior_per_beat |
|---|---|---|---|
| 001 | root | 33.3% | 22.6% |
| 001 | majmin | 34.0% | 21.9% |
| 001 | 7ths | 4.4% | 12.4% |

Songs 002-005 all improved or stayed flat on every metric simultaneously —
only song 001 regressed on root/majmin.

**First (wrong) hypothesis:** song 001's GT harmony includes `Bb:min` and
`Eb:min`, which look non-diatonic to F# major at first glance. **This was
checked and is wrong** — `Bb:min` = `A#:min` (same pitch class, enharmonic
spelling) = the iii chord of F# major; `Eb:min` = `D#:min` = the vi chord.
Verified directly against `build_key_prior()`'s diatonic dictionary:
`(tonic+4)%12` (iii, =10=Bb/A# for tonic=6) is boosted for
`[MINOR, MIN7]` ✓, `(tonic+9)%12` (vi, =3=Eb/D#) is boosted for
`[MINOR, MIN7]` ✓. All four chords seen in song 001's repeating GT
progression (`F#:maj`→`C#:maj`→`Bb:min`→`Eb:min` cycle, i.e. I-V-iii-vi) are
**exactly diatonic** to F# major. So "non-diatonic harmony" does not explain
the regression — this needs a different explanation, which is what led to §4.

---

## 4. The key-inference calibration bug — top priority, not yet fixed

Since song 001's harmony genuinely is diatonic to F# major, the next
question was whether the *detected* key was actually F# major, consistently,
for every segment (rather than trusting the single global `global_key`
field the pipeline reports). Checked directly:

```python
# harmonia/theory/key_profiles.py::infer_key(), called per-segment
for seg in segments:
    kp = infer_key(seg.chroma)
    print(seg.start_time_s, seg.end_time_s, seg.n_beats, kp.key_name, kp.confidence)
```

All 16 structural segments of song 001 resolved to `"F# major"` — with
**bit-for-bit identical confidence, `0.043`**, across segments with
different beat counts (11 to 35 beats) and genuinely different content.
`0.043 ≈ 1/24 = 0.04167` — suspiciously close to what you'd get from a
perfectly uniform distribution over all 24 candidate keys.

Printed the full posterior for one segment (the 35-beat one,
38.6s-62.1s) to check:

```
chroma (12-pc, L1-normalized): [0.0079, 0.2111, 0.0197, 0.1004, 0.0146,
                                 0.0540, 0.2588, 0.0094, 0.0769, 0.0185,
                                 0.1215, 0.1071]
# indices: C=0.008 C#=0.211 D=0.020 D#=0.100 E=0.015 F=0.054
#          F#=0.259 G=0.009 G#=0.077 A=0.019 A#=0.121 B=0.107

full posterior, sorted:
  F# major     0.04296   <- picked (correctly — matches key_audio.txt, see below)
  C# major     0.04249
  B major      0.04239
  D# minor     0.04236
  F# minor     0.04229
  A# minor     0.04226
  C# minor     0.04201
  G# minor     0.04189
  B minor      0.04188
  E major      0.04172
  ... (24 keys total, monotonically down to:)
  C major      0.04077   <- "worst" key of all 24
```

The chroma is genuinely informative — C# (0.211) and F# (0.259) are clearly
the two dominant pitch classes, correctly matching F# major's tonic and
dominant. But the resulting posterior spans only **0.04077 to 0.04296 —
about a 5% relative spread across all 24 keys.** F# major "wins" by a margin
of 0.00047 over the runner-up. This is a coin flip that happened to land
right, not a confident inference, and it's happening on *every* segment of
*every* song, not just this one.

### Root cause

`harmonia/theory/key_profiles.py::infer_key()`:

```python
log_likelihood = KEY_PROFILES @ chroma_norm   # shape (24,)
log_likelihood = log_likelihood * (1.0 + alpha * total / 12.0)
```

`KEY_PROFILES` rows are L1-normalized Krumhansl-Schmuckler profiles (sum to
1); `chroma_norm` is the L1-normalized input chroma (also sums to 1). Their
dot product is a *convex combination* of the profile's 12 entries — bounded
to whatever range the profile itself spans. For `KS_MAJOR`, normalized
values range roughly 0.06 to 0.16. **This bounded correlation score is then
treated directly as a log-likelihood and exponentiated.** `exp(0.16) /
exp(0.06) ≈ 1.10` — a hard mathematical ceiling of about 10% relative
concentration between the best- and worst-fitting key, no matter how clean
or ambiguous the actual input is. This is the primary bug.

**Secondary, compounding bug:** the `(1 + alpha * total / 12)` term is
clearly meant to be a Dirichlet-style confidence-sharpening factor — more
total acoustic evidence in a segment should produce a more peaked posterior.
But `total = chroma.sum()` is computed from `chroma`, which by the time it
reaches `infer_key()` has *already been L1-normalized to sum to 1* by the
caller — `harmonia/models/structure.py::_make_segment()`:

```python
chroma = np.zeros(12, dtype=np.float32)
...  # accumulate raw salience per pitch class
total = chroma.sum()
if total > 0:
    chroma /= total   # <-- normalized here, BEFORE infer_key() is ever called
```

So `total` inside `infer_key()` is always ≈1.0 regardless of how much real
evidence the segment had, making the confidence-scaling term collapse to a
near-constant `(1 + 1×1/12) ≈ 1.083` for every segment. The magnitude
information that would let this term actually work is destroyed before
`infer_key()` ever sees it.

### The agreed fix direction

Proper multinomial (or Dirichlet-multinomial, for smoothing) log-likelihood:
treat each `KEY_PROFILES[k]` row as a probability distribution over the 12
pitch classes for key `k` (it already sums to 1, so it already qualifies as
one), and compute the log-likelihood of the **unnormalized** chroma
(pseudo-)counts under it:

```python
log_likelihood[k] = sum_i chroma_raw[i] * log(KEY_PROFILES[k, i])
```

This is the log-likelihood of observing `chroma_raw` (or something
proportional to counts derived from it) under a multinomial distribution
with parameters `KEY_PROFILES[k]`. It naturally scales with the total
magnitude of `chroma_raw` — more evidence, same shape, produces a more
negative-but-more-separated total log-likelihood relative to competing
keys, without needing any ad hoc temperature constant — and it fixes the
secondary bug for free, since it operates on raw magnitudes throughout
rather than a pre-normalized vector.

**This requires `structure.py::_make_segment()` to stop normalizing
`chroma` before it's used**, or to preserve the raw magnitude alongside the
normalized version. Check both call sites of `infer_key()`:
1. Per-segment: `harmonia/pipeline.py`, `infer_key(seg.chroma)`.
2. Global-track: `harmonia/pipeline.py`, `infer_key(activations.chroma())`
   — `PitchActivations.chroma()` in `harmonia/models/stage1_pitch.py` calls
   `activations_to_chroma()` in `key_profiles.py`, which also normalizes
   (`chroma /= total` at the end) — same issue, needs the same fix or an
   explicit raw-chroma variant.

### An unused, directly relevant GT resource: `key_audio.txt`

POP909 ships real annotated key ground truth per song, not yet used
anywhere in this project until now:

```
$ cat data/pop909/POP909/001/key_audio.txt
2.670294784580499	191.9825850340136	Gb:maj
```

Format: `start_time_s  end_time_s  key_label`. For song 001 it's a single
line spanning almost the whole track. `Gb:maj` = `F#:maj` enharmonically
(same pitch class, 6) — **this independently confirms our F# major
detection for song 001 is correct**, which is useful (rules out "wrong key
entirely" as an explanation for anything) but also means song 001 alone
can't tell you whether the *confidence* fix actually improves *correctness*
elsewhere — you need multiple songs' `key_audio.txt` files for that,
across songs where the current near-arbitrary argmax might not have gotten
lucky. Not yet checked for songs 002-005 — do that as part of validating
the fix (see the prompt for the suggested incremental order).

---

## 5. Files changed / created this session (for orientation, all committed)

- `harmonia/models/stage1_pitch.py` — frame rate fix, `raw_activations()`
  method, cache key now includes threshold params.
- `harmonia/models/chord_hmm.py` — `key_prior_per_beat`/`key_prior_weight`
  on `ChordInferrer`; also (from earlier in the session, unrelated to key
  inference) `normalize_emission`, `compress_emission`, `duration_prior` +
  `viterbi_duration_aware()`, `periodicity_weight` + folded-view support.
- `harmonia/pipeline.py` — wires all of the above through
  `HarmoniaPipeline`.
- `harmonia/models/periodicity.py`, `harmonia/theory/duration_prior.py` —
  from the (on-hold) issue #1 A/B/C investigation.
- `tests/test_stage1_pitch.py`, plus extensive additions to
  `tests/test_chord_hmm.py` (`TestKeyPriorPerBeat`,
  `TestEmissionPreprocessing`, `TestFoldedViews`, `TestViterbiDurationAware`,
  `TestDurationAwareChordInferrer`), `tests/test_duration_prior.py`,
  `tests/test_periodicity.py`.
- `scripts/plot_note_probs_vs_gt.py`,
  `scripts/plot_note_probs_with_chord_timeline.py`,
  `scripts/plot_periodicity_diagnostic.py`,
  `scripts/experiment_issue1.py` — diagnostic/experiment scripts, all
  reusable.
- `docs/known_issues.md` — the authoritative living tracker, keep it
  updated.
- `docs/blog/01-` through `03-...md` — narrative devlog, keep it updated
  (see `[[feedback_blog_devlog]]` memory note if you have access to
  auto-memory).
- Data: `data/soundfonts/MuseScore_General.sf2` (215MB, gitignored),
  `data/renders/pop909/{001..005}/{id}_v005_musescoregeneral.wav`
  (gitignored) — both real, regenerable if missing (see §1b for the
  download URL and render command pattern in `harmonia/data/midi_renderer.py`).

Test suite as of this handoff: 84 passing (`pytest tests -q --no-cov`).
