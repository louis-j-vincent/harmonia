# Louis's ear verdicts on the frozen benchmark — 2026-07-27

Returned from the triage page (`docs/research_sessions/gt_triage_2026-07-27.html`).
68 regions judged. **These are verdicts on the REFERENCE (the GT), not on our model.**

| | regions | playing time |
|---|---|---|
| reference is right (`ok`) | 16 | 134 s |
| **reference is WRONG (`bad`)** | **23** | **201 s** |
| can't tell (`idk`) | 29 | 241 s |

## Per song

| song | ok | bad | idk | bad time |
|---|---|---|---|---|
| **every_breath_you_take** | 2 | **8** | 1 | **73.3 s** |
| **bein_green** | 0 | **5** | 0 | **45.0 s** |
| **georgia_on_my_mind** | 0 | **4** | 0 | **37.7 s** |
| blue_bossa | 3 | 3 | **25** | 25.1 s |
| close_to_you | 1 | 2 | 2 | 16.3 s |
| blue_bossa_backing | 0 | 1 | 0 | 3.2 s |
| **stand_by_me** | **10** | **0** | 1 | **0.0 s** |

## What this establishes

1. **~201 s of the 27.6-min benchmark (≈12%) has a confirmed-wrong reference.** Every accuracy
   number this project has reported was measured partly against it. The affected songs are
   `every_breath_you_take` (worst — the whole 79–136 s block plus 177–202 s), `bein_green`
   (every judged region bad), `georgia_on_my_mind` (every judged region bad), `close_to_you`.

2. **Stand By Me's reference is GOOD — 10/10 ok.** This settles the two-instrument conflict
   logged in `e00fd5c`: the raw-chroma reading of −1.15 s was the WRONG witness; the
   music-x-lab posterior instrument (median −0.02 s) was right. Chroma is unreliable exactly
   where the chords share tones (A/F#m/D share 2 of 3), as predicted. **Consequence:** the drift
   Louis hears on Stand By Me is OUR MODEL's error, not a reference error — a real target.

3. **Blue Bossa is largely unadjudicable by ear** (25 `idk` of 31): a 9-minute jam. Those regions
   should be marked UNSCOREABLE rather than counted as either pass or fail.

4. **The `drift` bucket split cleanly**: 12 bad / 20 idk, 0 ok — the drift detector was pointing at
   real problems (or at least never at a healthy region), but on Blue Bossa its regions are
   exactly the unadjudicable ones.

## Consequences to act on

- **Re-score the benchmark excluding the 23 bad regions** (and reporting the `idk` time separately
  as unscoreable) — until then, every headline number carries an unquantified reference error.
- **Repair the reference** on every_breath / bein_green / georgia / close_to_you. These GT files
  are `golden/brick0/*` — owned by the alignment lane; this is a proposal to them, not an edit.
- **Calibrate the triage instruments** with these 68 labels: chroma over-reported problems on
  Stand By Me, posteriors were right. Reweight accordingly before the next triage round.
