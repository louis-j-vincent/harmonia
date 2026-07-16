# Literature review — chord vocabulary, chart structure, and LM/attention approaches

**Delegated research pass, 2026-07-16.** Web literature synthesis (no code
changes). Format follows `chord_ai_reverse_engineering.md`: cited public results
+ explicit connection to what *this* project has already tried and found. The
deliverable is an opinionated "worth trying vs. re-discovers our own negatives"
verdict, not a bibliography.

Grounding facts from our own history (see `known_issues.md`):
- **Local neighbor-chord context is a triple-confirmed dead end for per-chord
  root/quality classification** (#21 bigram MARGINAL 63.8% < 70% gate; #27
  Mission 1 key-local bigram / encoder shallow-fusion / density-ratio fusion
  ALL net-negative on jazz, optimum λ→0; #31 Opus multi-head: neighbor-chroma
  context does NOT help root).
- **The bottleneck is front-end audio transcription, not modeling** (oracle
  boundaries → 86.8% root vs detected ~67%; #12).
- **P4/P5 (fifth) confusion is a genuine acoustic ambiguity**, 44–46% of root
  errors, confirmed on clean NNLS data across 5 seeds, broadly distributed
  (not a bad-song artifact) (#31). Diatonic/transition priors *reinforce* it
  because fifth-motion dominates real transitions (#31 Solution A FALSIFIED).
- **Bass information is the shared bottleneck** for both P4/P5 and dom→maj/min
  (#31 unified diagnosis).

---

## Thread 1 — Chord vocabulary representation

**The field has decisively moved to factored/compositional representations; flat
enumeration is considered the wrong tool for large vocabularies.** Our own
`architecture_extensions.md` §13 (drill-down rotor: root → family → 7th →
extensions) and the shipped 3-head factoring (#31) already sit on the correct
side of this — the literature independently converged on the same shape.

- **McFee & Bello, "Structured Training for Large-Vocabulary Chord Recognition"
  (ISMIR 2017)** — the foundational factored approach: decompose each chord into
  *root / pitch-class-presence / bass* and train a multitask model, rather than
  enumerating ~170 classes as flat softmax targets. This is the ancestor of our
  root/quality/7th sub-heads.
- **ChordFormer (Wu et al., arXiv 2502.11840, 2025)** — the current SOTA framing
  of exactly this. Represents every chord as a **6-dim structured vector**:
  dim 1 = root+triad, dims 2–6 = bass (inversion), 7th, 9th, 11th, 13th. A
  conformer (CNN+transformer) backbone with a multitask head per component.
  Reports **+2.3% frame-wise, +6% class-wise** over prior large-vocab baselines,
  and — the load-bearing claim — **explicitly better class balance across the
  long tail.** This is our own "separate binary is-degree-present sigmoids"
  (`architecture_extensions.md` / Q4 Expert Voicing) already implemented and
  benchmarked by someone else. **Do not re-derive it — adopt their exact 6-slot
  factoring as the target vocabulary schema.**
- **Vocabulary-size tradeoff — literature confirms our own finding.** The
  motivating problem in every large-vocab paper (ChordFormer, Jiang et al.
  "Chord Structure Decomposition" 2019, contrastive/noisy-student 2023) is the
  **long-tail distribution: rare chords are structurally under-sampled, and a
  flat large-vocab softmax collapses onto the head classes.** The *reason*
  factoring wins is precisely that it shares statistical strength across the
  tail (a 13th is just a maj7-triad + an extra sigmoid, not a brand-new class
  with 12 training examples). This is the field-level version of our repeated
  "rare classes are structurally under-supported by small real-audio corpora."
  Bigger *flat* vocab hurts common classes; bigger *factored* vocab does not.
- **Harte notation** (Harte et al. 2005, `C:maj7`, `A:hdim7/5`) remains the
  de-facto standard and is *not* being replaced — recent work (CHORDONOMICON,
  the `harte-library` music21 extension, 2024) doubles down on it, adding
  knowledge-graph links (Functional Harmony Ontology) rather than a new syntax.
  Takeaway: **keep Harte as the interchange format; the innovation is in the
  model's factored *target*, not the notation.**

## Thread 2 — Chart / harmonic (functional) structure

Distinct from our SSM boundary-timing work (already covered) — this is about how
chords relate *functionally* across a whole piece.

- **AugmentedNet (Nápoles López et al., ISMIR 2021; audio adaptation EvoMUSART
  2024)** — a CRNN doing full **Roman-numeral analysis** (key + degree + quality
  + inversion) via multitask learning with synthetic augmentation. The 2024 work
  adapts it to audio (chromagram/semitone-spectra input). Notably the paper
  claims it **outperforms transformer-based functional-harmony models** — a data
  point that attention is *not* automatically superior here.
- **"Attend to Chords" (Chen & Su, TISMIR 2021)** — transformer for symbolic
  harmonic analysis; the counterpoint that attention *does* help on *symbolic*
  (clean) input, where long-range functional dependency is the whole task.
- **"When in Rome" meta-corpus (TISMIR 2023)** and the **Functional Harmony
  Ontology** (via CHORDONOMICON, arXiv 2410.22046, 666k songs) provide the
  training substrate that didn't exist when this project started.

The honest read: functional-harmony models exist and work *on symbolic input*.
On *audio* they inherit exactly our bottleneck — you cannot do reliable
tonic/predominant/dominant reasoning on top of a 67%-accurate root stream.

## Thread 3 — Language models / attention (the user's primary interest)

### 3a. Attention for recognition (audio → chords)
- **BTC — Bi-directional Transformer for Chord Recognition (Park et al., ISMIR
  2019, arXiv 1907.02698)** — the reference architecture: bidirectional
  multi-head self-attention over a CQT frame sequence, single-phase training,
  self-attention providing an *adaptive receptive field* that implicitly
  segments chords (visible in attention maps). This is the "transformer replaces
  the HMM/CRF decoder" move. **Gains over CRNN baselines are real but modest
  (~1–2% WCSR)** — attention buys smoother segmentation, not better per-frame
  harmonic discrimination. Crucially for us: **BTC's contribution is temporal
  smoothing/segmentation, which is exactly the axis our semi-Markov duration
  decoder already owns (#27 Mission 2, shipped, +1–2pp).** A BTC-style encoder
  would largely re-implement a lever we already pulled.
- ChordFormer (above) is the current audio-attention SOTA and its win is the
  *factored head*, not the attention per se.

### 3b. Chord sequences AS a language (symbolic chord LMs)
- Yes, this exists and is mature: **CHORDONOMICON (2410.22046, 666k songs)** is
  the GPT-scale chord-progression corpus; **The Chordinator (2024)** and
  **Chord-Transformer (2024)** are seq2seq/autoregressive transformers that
  *generate* and *predict-next* chord progressions, analogous to Music
  Transformer for melody. They model `P(next chord | history)` well **for
  generation**.
- **But this is precisely the "generic transition grammar" our decoder already
  found saturated** (#27 Mission 1; mission_5_llm_priors_research.md §2). A
  better `P(chord|context)` LM does not help our decoder because that slot is
  full — a generic progression prior over *our* noisy emissions is denoising a
  signal whose errors (P4/P5, dom→maj/min) are *acoustic*, not *grammatical*,
  and fifth-motion priors actively reinforce the fifth-confusion (#31).

### 3c. Modern generative LLMs as a correction layer
- **"Enhancing ACR through LLM Chain-of-Thought Reasoning" (arXiv 2509.18700,
  2025)** — the single most on-point new paper. GPT-4o as a **post-hoc
  correction layer** over a 301-class acoustic model, in 5 CoT stages: source-
  separation selection, **bass correction**, key correction, anomaly detection,
  beat alignment. Reported **MIREX gains: +1.06% (UsPop), +1.23% (IdolSongsJp),
  +2.77% (in-house).**
- **Read this critically.** (1) The biggest gain (2.77%) is on their *in-house*
  set — the classic weak-baseline/dataset-favorable pattern to distrust. (2)
  Public-set gains are ~1%. (3) **Their "bass correction" stage sometimes
  *decreased* accuracy** due to sustained bass notes — i.e. they independently
  re-discovered our exact bass-is-the-bottleneck problem (#31), and an LLM did
  not solve it. This paper is strong evidence for the *architecture* (LLM as a
  structured post-processor over MIR-tool outputs, fed *stems* not audio) and
  weak evidence that it moves the number much.

---

## Synthesis — what is genuinely worth trying vs. what re-discovers our negatives

### Would just re-confirm our own negatives (do NOT spend on these)
1. **A BTC-style self-attention encoder as the decoder.** Its win is temporal
   segmentation/smoothing — the semi-Markov duration decoder already owns that
   axis (#27 M2). New attention here re-implements a pulled lever.
2. **A symbolic chord LM (Chordonomicon/Chordinator-style) as a transition/
   grammar prior** over our emissions. The generic-grammar slot is triple-
   confirmed saturated (#21, #27 M1). Fancier branding (transformer vs. trigram)
   of `P(chord|neighbors)` will re-hit the same λ→0 wall — *worse*, because
   fifth-motion priors reinforce our dominant error mode (#31 Solution A).
3. **Functional-harmony / Roman-numeral net on top of the current root stream.**
   Tonic-predominant-dominant reasoning is only as good as its root input;
   at 67% audio root it will amplify, not fix, the acoustic errors.

### Genuinely worth trying (meaningfully different from what's ruled out)
1. **Adopt ChordFormer's 6-slot factored vocabulary as the target schema
   (highest value, lowest risk).** It is the *representation* answer to Thread 1,
   it directly attacks the long-tail problem we keep hitting, and it generalizes
   our already-shipped 3-head factoring to the full extension space (9/11/13,
   inversions) without a combinatorial-class blowup. This is not "more context";
   it's a better output parameterization — orthogonal to the dead context axis.
2. **LLM-CoT correction *conditioned on song identity/chart*, NOT as a grammar
   prior (the one LM lever with a real, distinct payoff).** The 2509.18700 paper
   + our own mission_5 agree: the LLM's unique value is **song-specific**
   structure a corpus bigram cannot access — "this is Autumn Leaves in Gm, these
   two 8-bar strains are identical, the bar-13 V is a 7b13." mission_5 already
   scoped this correctly (lead with key/mode + repeat-structure assertion;
   `P(q|root,pos)` softly; never a generic grammar). The literature now
   *validates that framing* and warns of its failure mode (bass correction can
   backfire). **Recommendation: build the LLM path as mission_5 specced — a
   confidence-gated automated-annotator factor keyed on key/mode + asserted
   repeats — and explicitly do NOT let it touch root via a fifth-biased grammar.**
3. **A sequence-level LM as a RE-RANKER over top-k per-segment candidates,
   tested only for the ONE error class where it is a-priori plausible:
   dom→maj/min quality (not root).** Our negative results were all about *root*
   and about *local* neighbor voting; quality has a genuine functional signal
   (V wants dom7). The distinction the user asks about *is* real in principle —
   but note #31 already found *learned trigram context helps quality* (bal
   0.714→0.735) when digested as rotated root-*posteriors*, and it did NOT help
   root. So: the promising residual is a **quality-only re-ranker**, and it is
   already half-built. A whole-sequence LM re-ranking *root* is very likely to
   re-discover the fifth-reinforcement negative — bound the experiment to
   quality, with a cheap premise check first (CLAUDE.md rule #2).

**Bottom line for the user:** the LM/attention literature's genuine, non-
redundant offerings for this project are (a) ChordFormer's **factored output
representation** (a vocabulary/representation win, not a context win) and (b) an
**LLM song-specific correction layer** (already correctly scoped in mission_5,
now literature-validated). Everything shaped like "a better `P(chord|neighbor
chords)` grammar" — whether trigram, transformer, or GPT — will re-confirm the
project's triple-confirmed local-context and saturated-grammar negatives, and
risks *worsening* the P4/P5 root error it would claim to fix.

## Sources
- ChordFormer: https://arxiv.org/abs/2502.11840
- McFee & Bello, Structured Training (ISMIR 2017): https://brianmcfee.net/papers/ismir2017_chord.pdf
- Jiang et al., Chord Structure Decomposition (2019): https://zenodo.org/records/3527892
- BTC (ISMIR 2019): https://arxiv.org/abs/1907.02698
- LLM Chain-of-Thought ACR (2025): https://arxiv.org/html/2509.18700v1
- CHORDONOMICON (2024): https://arxiv.org/pdf/2410.22046
- AugmentedNet (ISMIR 2021): https://archives.ismir.net/ismir2021/paper/000050.pdf ; audio adaptation (EvoMUSART 2024): https://link.springer.com/chapter/10.1007/978-3-031-56992-0_10
- Attend to Chords (TISMIR 2021): https://transactions.ismir.net/articles/10.5334/tismir.65
- The Chordinator (2024): https://hal.science/hal-04289026/document
- Harte notation / harte-library: https://github.com/andreamust/harte-library
</content>
</invoke>
