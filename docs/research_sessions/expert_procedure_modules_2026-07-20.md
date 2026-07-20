# Expert-procedure modules — bass veto / top-2 referee / phase-robust sections — 2026-07-20

Continuation. Budget 3h (start ~08:32). Spec `docs/expert_procedure_louis.md`.
Three coordinator addenda arrived mid-work; all converge on section-similarity.

## Module 1/2 premise check (Let It Be Am target) — FALSIFIED on the live path
The brief: NNLS root head prefers C in Am bars → Am misread. **But the live path
uses MUSX root, not the NNLS head.** Evidence (`scratchpad/premise_bass.py`,
`lib_align.py`, direct segment dump):
- Every `A:min` segment has `mx_root=A`, routed bass `A` (17.4/40.1/69.2/79.9/
  148.4/165.5/190.3/218.1). Module 1's trigger (C-labeled, low-margin, bass-A)
  fires on **0 segments**.
- First verse decodes **`C G Am F | C G F C`** exactly; 16 Am segments evenly
  spaced across the whole song (one per verse/chorus phrase) = matches GT.
- The "Am 1/16, per-slot acc 0.30" from uniform time-fraction scoring is a
  time-REGISTRATION artifact (same confound as Bein' Green), not a chord error.

**Conclusion**: Let It Be's Am is ALREADY correct on the live path. The C-over-Am
bias exists only in the NNLS head, which the live path doesn't use for root; it only
affected loop-DETECTION pooling (Mission 1, structure). Modules 1/2 target a
non-problem on this song → I do not force them; I redirect to the evidence-backed
section-drift addendum and report Module 1/2 honestly.

## Section-drift addendum (user D8-bis / D9) — CONFIRMED real, the actual target
`chart_model._sections_by_largest_unit._sim` compares L-blocks strictly position-by-
position (`zip(a,b)`) on the uniform-grid bar-root sequence. A 1-bar phase drift
collapses the match. Evidence (`scratchpad/sec_diag.py`, Let It Be L=8 blocks):
identical-but-drifted blocks score **strict 0.00 → phase-tolerant 1.00** (blk0 vs
blk4/9/12). Current A×15 survives only via fragile single-linkage chains; the seed
block (blk0) has strict 0.00 vs most later A-blocks. → make block matching
phase-tolerant (±1 bar) so drifted repetitions merge directly.

Addendum params (D9): compare at FAMILY level (maj≡maj7≡6, min≡min7, DOM distinct,
dim/aug/sus) — current `_sim` is root-ONLY, already immune to maj7 wobble; keep
root-based (dominant-distinction refinement deferred, it only ADDS splits = regression
risk). 8-bar A prior kept as tie-break. Position-2 discriminator considered vs
over-merge risk.

## Phase-tolerant section matching — implemented, gated, SHIPPED (default ON)
`chart_model._sections_by_largest_unit._sim` (kill-switch `HARMONIA_SECTION_PHASE_TOL=0`):
- Equal-length blocks: strict position match (unchanged) as `base`.
- A small bar LAG (±1) may recover a drifted repetition, but a shifted match is
  trusted ONLY if it clears `_PHASE_STRICT=0.80` (real drift aligns ~1.00; two
  different sections only find a weak ~0.6 coincidental slide → NOT trusted → no
  over-merge). This is the double-sided G15 constraint (don't under-merge via
  drift-blindness, don't over-merge via weak coincidence) in miniature.
- Unequal-length (trailing partial block): compared on its min-length overlap
  instead of the hard len-≠ → 0.0 reject, so a truncated final repeat merges into
  its phrase instead of minting a false letter.

### Union-find evidence (the decisive debug)
Bein' Green uses L=16: blk3 is an 8-of-16 TRAILING PARTIAL block = truncated A-phrase
prefix (matches blk1/blk2 at 0.62 overlap); its OFF "B" was a length-artifact, NOT the
real bridge (the bridge's G#/C# material sits INSIDE blk1/blk2). Let It Be L=8: identical-
but-1-bar-drifted blocks strict 0.00 → phase 1.00 (blk0 vs blk4/9/12).

### Matched-set gate (9 charts, all deterministic / 2-run stable, live /api/chart-model confirmed)
| song | OFF | ON | verdict |
|---|---|---|---|
| Let It Be | Intro·A×15·B | **A×18** | trailing-B + Intro (a verse) absorbed; verse≈chorus harmonically → one phrase (user's largest-phrase principle) |
| Billie Jean (validated) | A·B·A×2·C | **A·B·A×3** | trailing-C→A; **bridge B PRESERVED** ✓ |
| Stand By Me | Intro·A×2·B | Intro·A×3 | trailing-B→A; song is one I-vi-IV-V progression throughout |
| Bein' Green | Intro·A×2·B | Intro·A×3 | trailing partial-block artifact merged |
| abba, Commodores, aretha, Autumn, Georgia | — | **SAME** | unchanged |
| henny, just-aint (validated) | — | **SAME** | unchanged |

**Gate: GREEN.** Validated forms (henny/just-aint SAME, Billie Jean bridge preserved);
every merge is a TRAILING/drift artifact, no genuine middle section destroyed; 2-run
stable through the live serve path; decode/anti-crush untouched (display-layer only).

### Honest scope / not-solved
- Does NOT recover the verse↔chorus split where they're harmonically near-identical
  (Let It Be) — that needs GRID-ANCHORED blocks (option a, real-beat pooling) so the
  position-2 discriminator can act on non-rotated blocks; deferred. The blocks are cut
  on a fixed grid, so an off-grid phrase downbeat makes each block a ROTATION → the
  position-1/2 discriminator is unreliable until the grid is anchored.
- F13 note (spec update): the exception criterion = regularity of recurrence ACROSS
  PASSES at a cycle position, bass-arbitrated — generalizes Mission-1 phrase-pooling to
  exception arbitration too. Future round.
