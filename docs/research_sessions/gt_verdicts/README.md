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

## 2026-07-27 — Louis's diagnosis behind the `bad` verdicts (decisive)

He explained WHAT is wrong in the regions he rejected. Three distinct defects, not one:

**(1) The reference is TIME-SHIFTED, the model is right.** "Sur tous les autres le modèle semble
avoir juste et c'est le ground truth qui semble être décalé." This confirms the lattice finding
(`67389e8`): GT onsets sit on a perfect constant-tempo grid while real players drift. Already
measured, per song: bein_green +0.22 s/min, close_to_you +0.22 s/min, georgia −0.245 s/min
(crossing zero mid-song), blue_bossa a constant +0.28 s anchor offset.
**Non-circular fix:** re-lay the reference's chord onsets on the DETECTED beat grid instead of a
synthetic constant-tempo lattice. The beat tracker is an instrument independent of the chord
model, so this does not correct the reference with the thing being measured.

**(2) The recording DIVERGES HARMONICALLY from the chart** — georgia, bein_green, blue_bossa
live: "il y a des variations d'accord par rapport à la grille". The players reharmonise; the
chart-derived reference then asserts chords nobody plays. A chart≠recording detector already
exists (built for Georgia's F#dim→B7 case, cross-repetition uniformly-low ⇒ flag) — run it over
these regions rather than assuming the model is wrong.

**(3) *** HARMONY IMPLIED, NOT STATED *** — the category we were missing.**
Louis: on Stand By Me, and in the Blue Bossa live bass-solo section, **only the bass is playing.**
"It is enough to deduce the chord from the general harmony but no chord is being played."
So these passages contain NO sounded chord at all — the harmony is inferred by a musician from
the bass line plus the form.

**This re-reads our biggest logged "failure".** The two worst slices in the whole benchmark were
`no chord` printed over 28.6 s of Blue Bossa and 10 s of Stand By Me (`1018ee2`). We filed that as
a chord-vs-no-chord discrimination bug. **It is not.** The model is acoustically CORRECT that no
chord is sounding; the reference asserts the *implied* harmony. Suppressing N (`no_chord_policy`,
+2.10pp) therefore scores better for the wrong reason — it forces a name where the benchmark
convention wants one, without any evidence that it is the right name.

**Consequences:**
- Regions must be labelled **chord STATED vs chord IMPLIED**. Implied regions cannot be scored as
  acoustic chord detection — they test form/bass inference, a different capability.
- The only honest way to get implied regions right is **bass note + form/repetition prior** — i.e.
  exactly Louis's repetition direction ([[project_repetition_prior]]). Acoustic chord evidence is
  absent by construction there, which is why every per-instant signal failed on them.
- Blue Bossa's bass solo should be **removed from the benchmark** ("à virer, ça perturbe à chaque
  fois"); Stand By Me's bass-only passages should be marked IMPLIED, not deleted (its reference was
  judged 10/10 correct).
