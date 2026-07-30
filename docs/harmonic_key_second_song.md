# Harmonic-key colour tracker: second-song study (2026-07-30)

Mission: generalise the v4.1 scale-colour tracker beyond This Love and report
whether its calibration is This-Love-shaped (error-pattern #5: single-song
findings are hypotheses).

Scripts (new, scratchpad-only; the This Love originals were left untouched):
- `scratchpad/colour_hmm_song.py <slug> [--tonic N]` — generic tracker,
  tonic/mode/bpb from the baked payload, all arithmetic tonic-relative.
- `scratchpad/colour_chart_song.py <slug>` — generic lead-sheet PNG,
  bar anchor computed per song.
- `scratchpad/colour_transpose_check.py` — transposition unit test.
- `scratchpad/colour_inventory.py` — the candidate scan below.

## Verdict in one line

The numeric constants (Q, GAIN, LAMBDA, transitions, HELD_W, INFL) survive
the trip; what is This-Love-shaped is everything **upstream** of them — the
assumption that the payload's home key is right (wrong on both new songs),
that the tonic never moves, that four minor colours span the song's harmony,
and that non-diatonic chords are rare enough to be individually suspicious
(5% on This Love, 64–73% on the new songs — the audit layer drowns).

## Inventory: minor-key + real audio + nnls cache

Scan of 57 baked charts (`docs/plots/inferred_*.html`); criteria: payload
`home.mode == "minor"`, `docs/audio/<slug>.m4a` present, nnls cache present.

| slug | keyName | tonic | bpb | nBars | chords |
|---|---|---|---|---|---|
| aretha_franklin_chain_of_fools_official_lyric_video | A minor | 9 | 4 | 78 | 100 |
| carpenters_close_to_you | C minor | 0 | 4 | 83 | 59 |
| maroon_5_this_love | C minor | 0 | 4 | 80 | 120 |

Only two candidates besides This Love. 14 more minor-key charts exist but
lack audio and/or nnls cache (all ireal_* charts, Billie Jean, Blue Lights,
Beat It, Happy, ...). Both candidates were taken: Chain of Fools (soul,
tonic ≠ C — exercises transposition) and Close to You (orchestral pop).
No jazz candidate exists with audio + cache.

## Reproduction + transposition checks (both PASS)

- Generic script on This Love reproduces v4.1 **exactly**: folded counts
  natural=83 harmonic=32 dorian=5 melodic=0; identical flags (6), audits
  (6, same chords, same proposals, same scores), identical form string.
  Chart script reproduces bar anchor offset 2 (74/117 onsets).
- Synthetic transposition (This Love chroma rolled +3 semitones, roots and
  tonic shifted to Eb): 0/120 decode diffs unfolded AND folded, identical
  form — the tonic-relative rotation and the vocab-fold machinery are both
  transposition-clean.

## Song 1 — Chain of Fools (payload says A minor; the audio says C minor)

`infer_key` on the song's summed chroma: **C minor** (payload: A minor).
The chart's chord histogram is C-rooted (C7 ×23, C ×13, Eb ×14, C- ×6...).
The payload tonic is simply wrong, so both runs are reported.

| run | folded colours | switches | flags | audits |
|---|---|---|---|---|
| payload tonic A | natural=100 | 0 | 4 | 59/100 |
| tonic forced to C | natural=100 | 0 | 9 | 72/100 |

Findings:
1. **The audit layer becomes a tonic detector.** Under the wrong A-minor
   tonic, 59 of 100 chords are challenged and the proposals overwhelmingly
   name C-rooted chords (C, C-, C^7). The tracker cannot say "wrong key",
   but its failure shape says it loudly.
2. **The blues third is outside the state space.** Under the correct C
   tonic the chart's C7/C/E/E- chords (E natural = major third, 46% of the
   chart) are non-diatonic to every minor colour, so 72/100 chords get
   audited. The four colour states only model the 6th and 7th degrees; a
   tonic-major/blues 3rd axis does not exist. This never mattered on This
   Love, which has no blue thirds.
3. Colour content is honest: a soul vamp has no harmonic/melodic minor;
   natural=100 with 0 switches is the right answer. Evidence shares are
   mushy though (7th-degree decisive on only 46% of gated chords, vs 90%
   on This Love) — Q=0.85 gets little to bite on.
4. The vocab fold slots 100/100 chords but fragments the form into 21
   terms (`A×4 B A×11 C B D A E A F A G×2 ...`) — vamp songs give the
   section vocabulary nothing to hold onto. The fold still behaved (no
   pathological pooling observed).

## Song 2 — Close to You (payload says C minor; the song is major)

`infer_key` agrees with the payload ("C minor") — and both are wrong about
the mode: the chart is 34 major triads + maj7s (C, G7, E-7, Db, Ab, Db^7)
against a single C-. Folded decode: **natural=28 harmonic=0 dorian=3
melodic=28**, 3 switches, 0 flags, 18 audits.

Findings:
1. **The melodic state is a closet major-mode detector.** The entire
   C-major body of the song (0:12–1:35) decodes "melodic minor" — raised
   6th AND 7th, i.e. C major minus a third the state space cannot see.
   Evidence is the crispest of all three songs (raised-share decisive on
   95–97% of gated chords). The melodic state was stone dead on This Love
   (0 chords in every version since v3); here it carries half the song.
   LAMBDA and the asymmetric transitions do NOT suppress a true raised
   signal — good news for the constants, bad news for the mode premise.
2. **The tracker paints the modulation.** From ~1:38 the chart lives in
   Db-land (Db, Ab, F-7, Db^7); tonic-relative to C these supply Ab and Bb,
   so the decode flips melodic→natural and stays there. The colour band is
   accidentally a key-change map. All four late Db^7s (2:59, 3:10, 3:21,
   3:32) get challenged →F-/Ab: with a fixed C tonic, a modulation is
   indistinguishable from a wrong chart.
3. **The audit layer challenges high-confidence chords on thin margins.**
   On This Love every audited chord had chart confidence 0.34–0.58; here it
   challenges E-7 at conf 0.91–0.97 with 1.25–1.3× score ratios. The open
   v4 item "confidence-scaled challenge thresholds" is now demonstrably
   needed, not just plausible.

## Calibration audit — constant by constant

| constant | verdict | evidence |
|---|---|---|
| Q=0.85 | keep | crisp evidence where the mode fits (decisive share 0.90 This Love, 0.95–0.97 Close); on blues material shares are mush (0.46/0.30) and no Q would fix that |
| GAIN=25, HELD_W=0.5 | keep | held rates 23% / 2–19% / 27% across songs — sane spread, no cliff |
| LAMBDA=0.25, trans 0.94/0.02 & 0.85/0.12/0.015 | keep, under-tested | did not suppress Close's genuine 28-chord melodic block, did not latch anything on Chain; but no new song stressed a genuinely ambiguous 6th/7th the way This Love does |
| INFL thresholds (1.0, 0.15) | keep | 0–11 flags per song, all interpretable; on Close the layer is bypassed (disagreeing chords are non-diatonic → routed to audits) |
| CHALLENGE_MARGIN=1.25 | broken premise, not broken value | eligibility ("non-diatonic to every colour") is 5% of This Love vs **64–73%** of the new songs — the rule assumes non-diatonicity is rare; blues thirds and modulations make it the norm. Needs a global guard (e.g. if eligible-rate > ~20%, audit the HOME KEY, not the chords) + confidence scaling |
| vocab_sections thresholds | better than feared | 100/100 and 59/59 chords slotted (This Love: 119/120); vamp form fragments (21 terms) but the fold stays harmless |

Ranked This-Love-shaped assumptions (all upstream of the constants):
1. `home.tonic`/`home.mode` from the payload is correct — wrong on BOTH new
   songs (Chain: tonic A vs sounding C; Close: mode minor vs sounding
   major). `infer_key`'s confidence is saturated at 1.0000 on all three
   songs, so it cannot arbitrate.
2. One fixed tonic for the whole song — Close modulates and the tracker
   can only express that as colour flips + an audit storm.
3. Four minor colours span the harmony — no blues-third axis (Chain), no
   major mode (Close).
4. Non-diatonic chords are rare and individually suspicious — the audit
   layer's founding premise, true only on This Love.

Emergent positive worth keeping: the tracker **fails informatively**. Wrong
tonic → audit proposals converge on the true tonic; wrong mode → melodic
occupancy explodes. A v5 could feed these diagnostics (audit-eligible rate,
melodic occupancy, proposal-root histogram) back as a home-key auditor
before any per-chord verdict is issued.

## What was NOT verified (no ear available)

- Whether Chain of Fools' band ever sounds the major third (chart: C7 ×23;
  infer_key + audit proposals: C-). Both sides are model output; no iReal
  source for either song.
- Whether Close to You's key really moves at ~1:38, and whether its body
  is major (near-certain from 34 major-family chords + crisp raised
  shares, but still a chart-derived claim).
- Every individual challenge proposal (D for B-7, F- for Db^7, ...).
  Proposal machinery was already flagged in the investigation doc as
  triad-biased; nothing here upgrades its trustworthiness.

## Ear-check questions for Louis

Chain of Fools (`docs/audio/aretha_franklin_chain_of_fools_official_lyric_video.m4a`):
1. 0:00–0:45 and 0:40–1:00: is the vamp chord C minor all the way (my
   read), or does the band really sound a major third (the chart wrote C7
   23 times and C major 13 times)?
2. 1:17 (chart: "B") and 1:27 (chart: "F#"): anything real there, or chart
   hallucination during the breakdown? The colour tracker flags both as
   fake-dorian moments.
3. Confirm the tonic is C, not A (payload says A minor — I say that's a
   pipeline home-key bug worth a known_issues entry).

Close to You (`docs/audio/carpenters_close_to_you.m4a`):
4. 0:17 / 0:20 / 0:39 / 0:41: chart writes B-7 then E-7; audits propose
   D / E- major. What's really under "why do birds suddenly appear"?
5. 1:38–1:45: does the key move up a half-step here (chart enters Db-land
   and never leaves), or does the famous modulation come later? At 1:41
   the chart writes C major and the colour prior challenges it →C-.
6. Sanity: the body (0:12–1:35) is a MAJOR key, yes? The payload calls the
   whole song C minor — if your ear agrees it's major, both the pipeline
   home-key call and `infer_key` (conf pinned at 1.0) are wrong here.

## Artifacts

- `scratchpad/colour_hmm_<slug>.png` — decode band + per-degree evidence
  (three songs; Chain also as `_tonic0` diagnostic).
- `scratchpad/colour_chart_<slug>.png` — lead-sheet colour charts. Close
  to You's chart reads as a modulation map; Chain of Fools' tonic-C chart
  is a wall of red audit frames (the blues-third problem, visually).
