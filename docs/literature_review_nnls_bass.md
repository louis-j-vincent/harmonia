# Literature review — bass/root detection front-ends & runnable pretrained tools (2026-07-17)

Scope narrowed per session direction: focus on (a) what is **actionable for our
specific bottleneck** — the ~0.65 bass-pc inversion ceiling and the −41pp
root-on-inversion penalty on RWC BP48 — and (b) **existing, runnable pretrained
tools**, not paper reimplementations. "Online pre-existing tools seem to be big
levers for us" (user framing).

## 1. Where our gap actually is (recap, so the review stays targeted)
- Functional-root head on RWC BP48: root-position 69%, **inversion 28%** (−41pp);
  root errors on inversions land on the **sounding bass** 54.8% of the time.
- Dedicated bass-pc 12-way head: **~66% on true inversions** (chance 8.3%) — a real
  NEW capability, but the **inversion *detector* precision (~20%) is the bottleneck**
  for using it to fix root. Temporal/GRU input did NOT help (negative result, 07-16).
- So our gap is specifically: **detect inversions reliably, and read the sounding
  bass** — i.e. bass/inversion output, which our functional-root+quality vocabulary
  structurally discards.

## 2. Runnable pretrained tools — landscape (the priority)

| tool | front-end | model | vocabulary | **bass/inversion?** | weights runnable? |
|---|---|---|---|---|---|
| **autochord** (cjbayron, PyPI) | **NNLS-Chroma VAMP** | Bi-LSTM-CRF (TF) | 25 (maj/min/N) | **No** | Yes, auto-downloads (~67% test) |
| **madmom** `chords` | Deep Chroma / CNN | CNN + CRF (`CNNChordRecognition`) | 25 (maj/min/N) | **No** | Yes, ships weights |
| madmom `DeepChromaProcessor` | learned deep chroma | Korzeniowski&Widmer 2016 | (chroma, not chords) | n/a (a *feature*) | Yes — a **learned chroma front-end** we could feed our own heads |
| **BTC-ISMIR19** (jayg996) | CQT | Bi-directional Transformer | `voca=True` **large-vocab (~170)** incl. **bass/inversion** | **YES** | weights via repo (Google Drive); needs own audio |
| **music-x-lab / ISMIR2019 Large-Vocab Chord Structure Decomposition** (Jiang et al.) | CQT | structure-decomposition net | large-vocab, **root+bass+quality factored** | **YES** | **ships pretrained models** |
| **ChordFormer** (2025, arXiv 2502.11840) | CQT 36 bins/oct | Conformer + CRF | 301, **6 structured slots** (root+triad, **bass**, 7,9,11,13) | **YES** | code/weights not clearly released yet (Feb 2025) |

**Takeaways for us:**
1. The two "big" chroma-CRF tools everyone reaches for (autochord, madmom) are
   **maj/min-only (25-class)** — they do **not** address our gap at all. autochord
   is notable only because it already wraps the **real NNLS-Chroma VAMP plugin** we
   separately confirmed builds/runs here (#NNLS entries) — but its head is maj/min.
2. The tools that DO output bass/inversion are **BTC (large-voca)** and the
   **music-x-lab structure-decomposition** model, both of which **ship pretrained
   weights** and both trained on Isophonics/UsPop/RobbieWilliams (pop/rock — same
   genre family as our RWC-Pop primary corpus, so a fairer transfer than jazz).
3. **ChordFormer is the architectural north star** (structured slots incl. an
   explicit bass slot, CQT 36 bins/oct, reweighted loss → class-wise 38.8% on a
   301-chord vocab) but is not yet a drop-in tool (no released weights as of the
   review). It *validates our own direction*: the bridge-doc Q4 "voicing bitmap"
   head and the dedicated bass head are exactly ChordFormer's factoring.

**Cheapest high-value experiment (run-tool-and-compare, not reimplement):** run
**BTC large-voca** (or the music-x-lab decomposition model) on a shared RWC/JAAH
audio subset for which we already have Harte GT, and score its **bass/inversion**
output against our GT with the same `sounding_bass_pc()` derivation. This directly
tests "does an off-the-shelf large-vocab model already beat our 66%/28% bass &
inversion numbers?" — far cheaper than training, and the answer reshapes Phase 2.
Caveat: BTC/music-x-lab need their own audio ingestion (CQT), and we are disk-tight
(~6 GB free) — one song at a time, delete after, as the NNLS-VAMP runs did.

## 3. Source separation as a bass front-end — literature is *cautionary*
The specific "Demucs-bass → chroma/pitch, then chord recognition" pipeline this
project flagged as untested has a **published negative datapoint**:
- **Ko (UW-Madison), "Automatic Chord Recognition by Music Source Separation":**
  training ACR on source-separated stems (combined *other*+*bass*, drums/vocals
  removed) performed **worse on average** than on the original mix, attributed to
  **separation artifacts** ("ranging from unnoticeable to barely listenable").
  No per-stem accuracy deltas given; author explicitly unsure if it's model-specific.
- **But** for the *narrow* bass-pitch task (not full ACR), separation+monophonic
  tracker **does** work: Demucs bass stem → **pYIN** (resolves sub-bass intervals
  where spectrogram trackers fail) or **CREPE** gives strong monophonic bass
  transcription — on FiloBass (double-bass), CREPE hit **72% F-measure, +10% over
  Basic Pitch**. So separation helps *monophonic bass-note read*, which is exactly
  our sounding-bass sub-problem, even though it hurt *full-mix chroma ACR*.

**Implication:** don't feed separated stems into the existing chroma/BP48 ACR head
(Ko's negative). Instead the promising, untested-here lever is narrow:
**Demucs bass stem → monophonic pitch tracker (pYIN/CREPE) → sounding-bass pc**, used
ONLY to (a) improve the inversion detector's precision (the confirmed bottleneck)
and (b) supply the bass-pc directly, bypassing the muddy BP48 bass block. This
sidesteps Ko's artifact problem because a monophonic tracker on the bass stem is
robust to the artifacts that wreck a 12-bin chroma.

## 4. Learned-chroma front-ends (NNLS successors)
- **NNLS-Chroma** (Mauch & Dixon 2010): our confirmed-sharp but trained-head-neutral
  front-end (see #NNLS entries). Real VAMP plugin builds/runs here.
- **Deep Chroma Extractor** (Korzeniowski & Widmer, ISMIR 2016; `madmom.DeepChromaProcessor`):
  a CNN trained to output *chord-relevant* chroma robust to interference — a learned,
  **runnable** alternative to NNLS as our root/bass feature. Untested here; a
  low-cost front-end swap (it is just a feature extractor, feeds our existing heads).
- **MERT** (self-supervised acoustic music model, 2023): general music representation;
  heavier, less targeted, deprioritized for our narrow bass task.

## 5. Recommendation ranking for Phase 2 (informed by our error patterns — see companion known_issues entry)
1. **Demucs bass stem → pYIN/CREPE → sounding-bass pc**, evaluated against our
   bass-pc GT and as an **inversion-detector precision** booster. Targets the
   confirmed 20%-precision bottleneck with a monophonic tracker (robust to the
   artifacts that sank Ko's full-mix approach). Highest expected value.
2. **Run BTC large-voca / music-x-lab decomposition on shared RWC subset, compare
   bass/inversion output to GT** — cheap "does an off-the-shelf tool already beat
   us" check before we build anything.
3. **Deep Chroma (madmom) as a front-end swap** for the root/bass heads — low cost,
   directly comparable to the NNLS-vs-BP48 head-to-heads already in the repo.

Sources: Ko (ko28.github.io/chord-transcription); ChordFormer (arXiv 2502.11840);
BTC-ISMIR19 (github jayg996); music-x-lab ISMIR2019-Large-Vocabulary-Chord-Recognition;
Korzeniowski & Widmer 2016 (arXiv 1612.05065); autochord (PyPI/github cjbayron);
madmom (CPJKU); FiloBass/CREPE bass-transcription (ISMIR).
