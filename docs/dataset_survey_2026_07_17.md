# Dataset Survey — trustworthy chord+audio(+bass) sources beyond current corpora

**Data-sourcing research thread, 2026-07-17.** Question from the user: *"what
does Chord AI use?"* plus — more usefully — *what other properly-aligned,
trustworthy chord+audio(+bass) datasets has this project not yet tried?*

Motivation: this session has been burned repeatedly by training-data trust
(YouTube/Billboard audio misalignment up to 6.9 s, ~25 % wrong-edit; a
fabricated NNLS-JAAH result caught by audit; POP909 GT discarding `/bass`).
So the axis that matters here is **alignment trust**, then bass/inversion
coverage, then jazz/pop relevance, then accessibility (given the past
"Billboard audio blocked" lesson and current disk pressure).

---

## Part 1 — Chord AI (and peer commercial products): what's knowable

**Short answer: Chord AI's training data and methodology are NOT publicly
disclosed.** This was already reverse-engineered in
`docs/chord_ai_reverse_engineering.md` (Agent 3, 2026-07-15) and nothing new
has surfaced. Recap so this doc is self-contained:

- **Chord AI** (chordai.net, iOS/Android, closed-source). Its own pages publish
  only marketing: *"in-house state-of-the-art deep learning models trained on
  thousands of songs that we have meticulously labelled over the years,"*
  on-device. **No architecture, no feature spec, no benchmark numbers, and no
  named training corpus.** The "beyond trained human level" accuracy claim is
  unfalsifiable — ignore it. What it *does* disclose (feature list): same
  **Basic Pitch** front-end as Harmonia, 4-stem separation incl. a **bass
  stem**, Whisper lyrics, and a Pro vocabulary with **slash chords (C/E, Am/C)**
  → i.e. a real bass/inversion model, first-class, not discarded. Everything
  else is *inferred* from the SOTA lineage it belongs to (McFee&Bello 2017 →
  BTC 2019 → ChordFormer 2025): structured output heads (root / bass / quality
  bitmap / 7-9-11-13), octave-preserving CQT features, learned emissions,
  heavy pitch-shift augmentation. Treat as reconstruction, not leak.

- **Chordify** — the one commercial team that *does* publish. They released
  **CASD** (Chordify Annotator Subjectivity Dataset, 50 songs × 4 expert
  annotators, JNMR paper) and co-authored work on annotator subjectivity. But
  CASD is a *label-agreement* study, not their production training set; their
  actual model/corpus is undisclosed. Their public stance is notable for the
  project: it treats chord GT as **inherently multi-annotator / subjective**,
  which is the same "GT is a measurement too" lesson in CLAUDE.md, quantified.

- **Moises / Yousician / others** — no engineering disclosures found on training
  data. Dead end; don't speculate.

**Verdict on Part 1:** nothing to build on from commercial disclosures beyond
what Agent 3 already logged. The transferable facts are (a) they all keep
octave resolution + a bass stem, and (b) Chordify's public position that chord
GT is multi-annotator. No corpus names to chase.

---

## Part 2 — Additional academic datasets (ranked by trust × relevance × access)

Already in use / evaluated (see memory `reference_ground_truth_datasets.md` and
`docs/literature_review_nnls_bass.md`): **McGill Billboard** (~890, pop, Harte
+inversions, duration-match sourcing works), **Isophonics** (~210 Beatles/Queen
etc., Harte +inversions), **JAAH** (113 jazz, beat-level Harte +inversions,
best genre fit), **RWC-Pop** (100, high-friction CD sourcing), **POP909**
(current primary, GT discards `/bass`). Below are the ones NOT yet tried.

Alignment-trust legend: **SELF-CONTAINED** = audio bundled or generated so GT
timestamps are trusted directly (no re-sourcing); **DURATION-MATCH** = audio
must be re-sourced but absolute-timestamp GT lets us duration-verify (the
established trick); **CHART-ONLY / NO AUDIO** = labels only, worst case.

### Tier A — real downloadable audio, alignment self-contained (highest trust)

1. **GuitarSet** — *premise-check PASSED (Zenodo record 3371780, CC-BY-4.0).*
   360 excerpts (~30 s), solo + comping, 6 players × 3 progressions × 2 styles.
   Files: `annotation.zip` 39 MB, `audio_mono-mic.zip` 657 MB (also mono-pickup
   mix 683 MB, hex-pickup ~3.2–3.6 GB). JAMS annotations: **performed chord
   labels + beats + downbeats + per-string note/fret** (so the sounding **bass
   note is recoverable per frame** even though there's no separate bass stem).
   Audio and GT ship together → **zero alignment risk**. Genre: solo/comping
   acoustic-electric **guitar only** (jazz/bossa/rock/funk/pop-ish
   progressions), no full band, no vocals. *Best available zero-risk chord+audio
   set for immediate use; small enough to fit (~700 MB) even at 97 % disk.*

2. **AAM — Artificial Audio Multitracks** — *premise-check PASSED (Zenodo
   record 5794629, CC-BY-4.0).* **3,000 synthetic tracks**, algorithmically
   composed from real instrument samples. Annotations (8.6 MB/third):
   **chords, beats, downbeats, keys, tempo, segments, onsets, pitches,
   instruments**, plus **isolated instrument stems incl. bass** and the source
   MIDI. Because audio is *generated from* the annotations, alignment is
   **perfect by construction — the ideal answer to this project's entire
   alignment-trust problem.** The isolated **bass stems + perfectly-aligned
   chord/bass GT are directly relevant to the parallel bass-model thread**
   (train/validate a bass head with zero label noise). Caveats: (a) **synthetic
   timbre + algorithmic (non-jazz-idiomatic) composition** — a domain gap vs
   real recordings, exactly the concern the "artificially generated audio"
   paper (arXiv 2508.05878, 2025) studies (it finds synthetic pretrain helps
   but real-audio fine-tune is still needed); (b) **SIZE: full mixes ≈ 44 GB
   (3×14.7 GB), multitracks ≈ 165 GB, full set ≈ 220 GB** — impossible at
   current **7 GiB free / 97 % disk**. Only a subset (e.g. mixes for 1000
   tracks = 14.7 GB, or a few-hundred-track slice) is feasible, and only after
   freeing disk (see known_issues #15).

### Tier B — labels only, no audio (pair with duration-match sourcing)

3. **USPop2002 + RWC-Pop chord annotations** (`tmc323/Chord-Annotations`,
   GitHub) — manually-annotated Harte-format chord labels (start/end sec) for
   195 US-pop songs + the RWC-Pop 100. **No audio** (copyright). USPop audio is
   hard to source cleanly; RWC needs the CD-purchase route already flagged
   low-priority. Useful mainly as an *additional label source* if audio is
   duration-matched. Overlaps genre with Billboard/Isophonics — low marginal
   value.

4. **Robbie Williams dataset** — key+chord Harte annotations for 5 albums (~65
   songs), **no audio**. This is the "RobbieWilliams" set referenced in the
   BTC / music-x-lab literature (BTC reports 83.9 % WCSR on
   Isophonics+RobbieWilliams). Audio must be re-sourced (famous, so
   duration-match is plausible). Pop/rock — same genre family as existing
   corpora, low marginal value beyond enlarging the pop pool.

5. **ChoCo — Chord Corpus** (Nature Sci Data 2023, `smashub/choco`) —
   **aggregates 18 chord sources into one JAMS/Harte harmony corpus** (incl.
   Billboard, Isophonics, JAAH, Weimar, iReal, biab...), ~20 k+ progressions.
   **Annotations only, NO audio.** Value: a single normalized pipeline to pull
   Harte-with-inversions labels for many of the above at once — a convenience
   layer, not a new audio source.

6. **CASD** (Chordify, 50 songs × 4 annotators) — multi-annotator labels, **no
   audio**. Value is methodological (quantifies GT subjectivity), not a
   training corpus.

### Tier C — jazz-relevant but audio-blocked or off-format

7. **Weimar Jazz Database (WJazzD)** — 456 **monophonic solo** transcriptions
   (Jazzomat, ODbL, downloadable SQLite/CSV). Has beats, chord-change
   annotations, structure. **BUT: chords are cloned from lead sheets (only
   sometimes transcribed from the actual rhythm section), and it references
   original audio via MusicBrainz ID + solo start/end — audio NOT bundled.**
   It's a *solo-line* dataset, not a chord-from-mix dataset; the chord track is
   lead-sheet-derived, not sounding-verified. Low fit for our
   chord-recognition-from-audio task despite the perfect jazz genre. (Related:
   **JSD**, Jazz Structure Dataset, TISMIR 2022 — structure segments over the
   same recordings; structure not chords.)

8. **Schubert Winterreise Dataset (SWD)** — 24-song cycle, chord+key+structure
   annotations across 9 performances, **but only 2 performances' audio are
   bundled (copyright); the other 7 are re-source-only.** Classical art-song,
   piano+voice — **genre mismatch** for a jazz/pop chord project. Excellent
   alignment discipline (multi-performance, measure-aligned) but wrong domain.
   Skip unless testing cross-genre robustness.

---

## Part 3 — Recommendation

**No full build-out is warranted right now — disk is the binding constraint
(7 GiB free, 97 %), and the two genuinely new zero-alignment-risk sources split
into "small but narrow" (GuitarSet) and "ideal but huge" (AAM).**

Ranked actionable recommendations:

1. **GuitarSet — adopt as an immediate zero-risk chord+audio validation set
   (~700 MB, fits today).** It is the only new source that ships aligned audio,
   is small enough for the current disk, and gives *performed* chord labels plus
   per-string notes from which a sounding bass note is recoverable — useful as a
   clean sanity/eval corpus for the emission and (partially) the bass work,
   caveated by guitar-only timbre.

2. **AAM — flag to the bass-model thread as the highest-value target, but
   BLOCKED on disk.** Its perfectly-aligned chord+**isolated-bass-stem** GT is
   the cleanest possible bass-head training signal and structurally solves the
   alignment-trust problem — but only a subset is feasible, and only after
   freeing ≥20 GB (known_issues #15). Recommend: once disk is freed, pull a
   ~1000-track *mixes* slice (14.7 GB) or a few-hundred-track multitrack slice
   for bass stems, and treat it as **synthetic pretrain to be real-audio
   fine-tuned** (per arXiv 2508.05878), not as a standalone corpus.

3. **ChoCo — use as the label plumbing layer** if/when expanding the
   Harte-with-inversions label pool across Billboard/Isophonics/JAAH; it's a
   convenience, not new audio, so no trust risk and negligible cost.

4. **Skip** WJazzD (lead-sheet-cloned chords, solo-line, no bundled audio),
   SWD (wrong genre, mostly audio-blocked), USPop/RobbieWilliams (audio-blocked,
   genre-redundant with existing pop corpora).

**Trust-bar note:** only GuitarSet and AAM meet this project's post-Billboard
"alignment must be self-contained, not re-sourced" bar without the
duration-match dance. Everything in Tier B/C reintroduces the exact re-sourcing
step that produced the 6.9 s offsets — so prefer the Tier-A pair, and if pop
label-count must grow, do it through ChoCo + duration-match with eyes open.

## Sources

- Chord AI — https://chordai.net/ ; recap in `docs/chord_ai_reverse_engineering.md`
- Chordify CASD — https://github.com/chordify/CASD ; https://chordify.net/pages/subjectivity-in-chord-recognition/
- GuitarSet — https://zenodo.org/records/3371780 (CC-BY-4.0) ; https://github.com/marl/guitarset
- AAM — https://zenodo.org/records/5794629 (CC-BY-4.0) ; https://asmp-eurasipjournals.springeropen.com/articles/10.1186/s13636-023-00278-7
- Training on artificially generated audio — https://arxiv.org/abs/2508.05878
- USPop/RWC chord labels — https://github.com/tmc323/Chord-Annotations
- Robbie Williams annotations — https://www.researchgate.net/publication/260399240
- ChoCo — https://www.nature.com/articles/s41597-023-02410-w
- Weimar Jazz DB — https://jazzomat.hfm-weimar.de/download/download.html
- JSD (Jazz Structure Dataset) — https://transactions.ismir.net/articles/10.5334/tismir.131
- Schubert Winterreise — https://zenodo.org/records/5139893
