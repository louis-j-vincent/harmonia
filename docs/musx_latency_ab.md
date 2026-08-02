# musx latency selector — audit follow-up, A/B report (2026-08-02)

Answers the "BUG RÉEL" flagged in `docs/harmonia_min_audit_questions.md`
(2026-08-01 audit): a claim that the musx latency selector in
`harmonia_min/musx.py` is structurally biased toward 0 ms.

## One-line verdict

**The bug is real and now fixed, but it is not why every song picks 0 ms —
fixing it changes nothing on all 5 songs tested. Ship the fix (it is a
genuine correctness bug and costs nothing), but drop the expectation that it
"unlocks" non-zero latencies.**

## What the bug actually is

`path_loglik` scores a latency candidate by rebuilding a per-frame chord tag
array from its decoded segments, then summing the emission log-prob at each
frame's tag. A segment whose start time lands before t=0 once you shift the
whole decode by `latency` gets clamped to `t0=0` — but the code that
re-derives frame indices for scoring doesn't know that clamp happened, so it
computes a start index *later* than the segment's real start. The result: the
first ~`latency`-worth of frames in the file are never written into the tag
array. They keep the array's default value, index 0.

**Correction to the audit's description**: index 0 is not "`C:min/b7`, first
line of the vocabulary file." `XHMMDecoder.__init_known_chord_names` (the
vendored decoder, `extractors/xhmm_ismir.py:41`) always prepends a literal
`"N"` (no-chord) entry as index 0, regardless of which chord-list file is
loaded. So the "artificial penalty" is really: *the first `latency` worth of
frames get silently scored as no-chord*, not as a specific low-probability
chord.

## Measured magnitude — confirmed real, but three orders of magnitude too small

I reproduced the mechanism directly and measured its size against the actual
range of scores the selector compares, on all 5 corpus songs with cached
musx posteriors + beat grids (`harmonia_min/state/charts/*.json` ×
`data/cache/musx_probs/*.npz` × `harmonia_min/state/beats/*.json`):

| song | loglik(0ms) | loglik(280ms) | span (0→280ms) | max bug size (nats) | bug / span |
|---|---:|---:|---:|---:|---:|
| ben_e_king_stand_by_me_audio | -6112.2 | -7088.5 | 976.3 | 12.56 | 1.29% |
| carpenters_close_to_you | -14646.3 | -17902.5 | 3256.2 | 7.82 | 0.24% |
| let_it_be_remastered_2009 | -9948.1 | -18696.5 | 8748.4 | 112.39 | 1.29% |
| maroon_5_she_will_be_loved | -13241.0 | -18614.0 | 5373.0 | 33.39 | 0.62% |
| maroon_5_this_love | -15706.7 | -21914.5 | 6207.8 | 13.86 | 0.22% |

The bug never accounts for more than ~1.3% of the gap between candidates. It
cannot be why every song lands on 0 ms.

I also validated the fix does what it's supposed to (i.e. it isn't a
no-op) on a constructed toy case where the true first chord is *not*
silence: without the guard, a 3-frame-latency candidate was penalised by
14.97 nats relative to L=0 despite the two having **identical** true fit;
with the guard, the gap is exactly 0.000.

## What actually drives the 0 ms preference

For all 5 songs, `path_loglik` decreases *monotonically* from L=0 as latency
grows — no song has a peak away from zero. The mechanism: `make_beat_arr`
shifts the *legal-transition grid* later by `latency`, which forces the
Viterbi decode to hold each chord's previous label for `latency` extra frames
at **every** transition (dozens to ~125 per song), not just the first. If the
beat grid feeding the decode is already on-time (no systematic lag between
beat and chord onset), this uniformly makes the fit worse everywhere, and the
loss compounds with every transition in the song — which is exactly the
large, monotonic effect measured above.

This corroborates a note already logged 2026-07-31 in
`docs/minimal_pipeline_log.md`: *"All 3 songs selected musx latency 0 ms (the
old 7-song study measured 60–340 ms optima)... different beat backend feeding
the grid may explain it."* `harmonia_min` feeds the decode with `beatthis`
beats (the project's verified-accurate tracker); the original external study
that measured +46…+289 ms optima almost certainly ran on a different (less
accurate) beat source, whose latency compensation was masking a beat-grid
timing offset, not a genuine musx-emission lag. That hypothesis is plausible
and consistent with the evidence gathered here, but it is **not proven** by
this investigation — it would need a same-song comparison across beat
backends to confirm, which is out of scope for this audit follow-up.

## The fix

`harmonia_min/musx.py`: `path_loglik` gained a `guard_frames` parameter that
excludes that many frames from **both ends** of the emission sum.
`redecode()` computes `guard_frames` once from `max(latency_grid)` (13 frames
≈ 302 ms for the shipped 8-value grid) and passes the *same* value to every
candidate — including L=0, so no candidate is ever compared on a frame
window another candidate doesn't also get scored on. This is the "score only
the common frame window" scheme from the mission brief, applied symmetrically
at both ends per the brief's guidance (no tail-side clamping bug was found in
practice, but the symmetric guard is free and defensive).

## Per-song A/B table

Protocol: for each of the 5 cached songs, ran the current (unguarded)
selector and the fixed (guarded) selector back to back on the same cached
musx posteriors + cached beatthis beat grid (no audio re-analysis, no beat
re-tracking), then diffed the resulting chord-event lists.

| song | latency_old | latency_new | n_chords | boundaries_moved | median\|shift\| | label_changes |
|---|---:|---:|---:|---:|---:|---:|
| ben_e_king_stand_by_me_audio | 0 ms | 0 ms | 38 → 38 | 0 | 0 ms | 0 |
| carpenters_close_to_you | 0 ms | 0 ms | 71 → 71 | 0 | 0 ms | 0 |
| let_it_be_remastered_2009 | 0 ms | 0 ms | 127 → 127 | 0 | 0 ms | 0 |
| maroon_5_she_will_be_loved | 0 ms | 0 ms | 104 → 104 | 0 | 0 ms | 0 |
| maroon_5_this_love | 0 ms | 0 ms | 120 → 120 | 0 | 0 ms | 0 |

**5/5 songs pick 0 ms both before and after the fix. Output is byte-identical
in every case — 0 boundaries moved, 0 label changes.** This is the "≥15%
label changes → hold" sanity gate from the mission brief inverted in the
other direction: not just under threshold, exactly zero.

Sanity anchor check: the mission flagged "pinning at the max candidate for
most songs" as suspicious. That did not happen — nothing moved at all, which
given the measured mechanism (a monotonic, ~1%-sized correction against a
much larger monotonic real effect) is the expected outcome, not a red flag.

## Musical diff for spotlight songs (This Love, She Will Be Loved)

Requested by the mission for the largest-change songs. There is no diff to
show: both songs kept latency 0 ms before and after the fix, so every chord
boundary and every chord label is identical. No "bar 12, the Fm arrived half
a beat late" story exists here — nothing moved.

## What this does NOT solve

- **`harmonia/models/musx_redecode.py` (the old pipeline) has the identical
  bug**, confirmed by direct code comparison (`_tags_to_lab`/`path_loglik` are
  byte-for-byte the same clamp-then-re-derive pattern). It is **not fixed**
  here — out of scope per the mission brief. If anyone later measures a
  latency-selection question on the old pipeline, this fix does not apply
  there; it would need the identical `guard_frames` treatment ported over.
- **Why beatthis's grid needs no latency compensation while the original
  study measured +46…+289 ms is not established here** — only proposed as
  the more likely explanation than "the selector is broken." A same-song,
  cross-beat-backend comparison would be needed to confirm it.
- **The `:7772` server was not restarted.** The fix is live in the source
  tree but the running process (if any) will keep using its already-loaded
  module until the next restart or the next fresh `redecode()` call in a new
  process. `:7771` was not touched.

## Recommendation

**Ship the code fix.** It removes a real, now-quantified correctness bug at
zero measured cost (byte-identical output on all 5 cached charts, so no
regression risk to review) and stops any future latency-grid change (wider
range, different step) from silently inheriting a larger version of the same
bias. Do not expect it to change any currently-shipped chart, and do not
treat "0 ms everywhere" as resolved — that appears to be a real property of
this beat backend + this corpus, not an artifact of the fixed bug.
