# Production Bridge Strategy — synthesis + real-audio retraining plan

*2026-07-15. Synthesizes `docs/feature_domain_bridge_nnls_to_bp48.md` (synthetic
BP48 bridge experiment) and `docs/known_issues.md` #31 chain (McGill-Billboard
NNLS root/quality/7th heads) into one shippable plan. Also corrects a scoping
error in the mission brief this doc responds to (see "Premise correction"
below) before any numbers are quoted.*

## Premise correction — "real audio Billboard" does not exist

The mission brief asks for real-audio Billboard training in BP48 space. This
is **not possible with what's on disk or obtainable**: McGill-Billboard
(`~/mir_datasets/billboard`) ships only pre-extracted Chordino/NNLS chroma
(`bothchroma.csv`) and chord/salami/mirex label files — **zero audio**, by the
dataset's own design (copyright). This is independently confirmed three times:
`docs/known_issues.md` #31 ("BP48/BP12/transfer BLOCKED — no Billboard audio"),
the bridge doc's Q3 section ("McGill Billboard ships no audio we can push
through Basic Pitch"), and a fresh check this session (`tar tzf
billboard-2.0-chordino.tar.gz` → only `bothchroma.csv`/`tuning.csv`, no `.wav`).
**Nothing in this codebase or dataset makes Billboard audio appear** — any
report claiming "Billboard, audio format FLAC, 890 tracks" (found in a stale
`/private/tmp/billboard_mission_summary.txt` from a separate prior session)
is fabricated and should not be trusted; it does not match the actual
tarball contents.

**Substitution used below:** the bridge doc's own recommendation is to treat
Billboard/NNLS as "a teacher for the architecture and the root/majmin priors
only, never a weight source," and retrain natively on real audio using
"Mission-2 real corpus_50 + YouTube/iRealb." That cached corpus
(`data/cache/yt_corpus/corpus_50.npz`) no longer exists on disk (likely
cleared during an earlier full-disk incident — `data/cache/yt_audio/` is
currently 0B). This mission rebuilds it via the existing, already-tested
pipeline (`scripts/build_yt_corpus.py` + `harmonia/data/yt_chord_corpus.py`):
real YouTube recordings of jazz standards, aligned to iReal Pro ground truth,
features extracted with the production `extract_beat_features` (BP48) path.
This is real audio; it is not Billboard.

## Findings synthesis (validated, by domain — read the domain column)

| Finding | Domain | Result | Source |
|---|---|---|---|
| Register (bass/treble split) beats folding to 12-dim | synthetic BP48 (accomp_db, 60 songs) | +2.2pp over channel-duplication at equal 24-dim; 12→48 buys +6.3pp | `feature_domain_bridge_nnls_to_bp48.md` Q1 |
| Root-relative (functional root→C) beats key-relative | synthetic BP48 | **+15pp** (0.968 vs 0.819 bal acc); key-relative barely beats raw | `feature_domain_bridge_nnls_to_bp48.md` Q2 |
| Native BP48 training beats porting NNLS-trained weights | synthetic BP48 vs McGill NNLS | native 0.923, ported-raw 0.765, ported+z-norm 0.673 (z-norm alignment **hurts** −9pp) | `feature_domain_bridge_nnls_to_bp48.md` Q3 |
| Root architecture: nonlinear MLP(24→128→64→12) on bass+treble | McGill NNLS, oracle boundary | 89.0% (vs 84.0% linear) | known_issues #31 Addendum 4 |
| Quality: root-relative rotation + learned trigram context (neighbor root-posteriors as features, not a λ-blended prior) | McGill NNLS, oracle boundary | bal 0.735, dom recall 0.698 (oracle root); top-k marginalization + dom-weight×1.8 → dom recall **0.776** at bal 0.710 | known_issues #31 Addendum 4 |
| Harmonic/transition/trigram **priors** (λ-blended into logits) | McGill NNLS | consistently **negative** — reinforces majority classes, tanks dom recall | known_issues #31 Phase 2A, 2C |
| 7th head: flat 5-way beats AND-reassembled base3×has-7th | McGill NNLS | flat dom 0.697 > reassembled 0.642; ship flat, keep base3 as prior only | known_issues #31 Addendum 4 |

**What is NOT yet validated:** every one of the above numbers involving BP48
is on *clean synthetic MMA piano renders*, not real audio (bridge doc's own
"CRITICAL CAVEAT — synthetic ceiling": "Absolute values are optimistic (dom
0.95 here vs real-audio dom 0.21 in #19)"). Every number involving the
root/quality architecture and the 89%/0.776 headline is on **McGill NNLS
chroma with oracle chord-boundary spans**, not BP48, not the production
segmentation. The two experiments were never run on the same data — no paired
NNLS↔BP48 comparison exists because Billboard has no audio (this is *why* Q3
above uses ported weights, not a real transfer test).

## What this mission does

Retrains the validated *recipe* (register-preserving BP48, root-relative
normalization, MLP root head, root-relative+trigram quality head, flat 5-way
7th head) **natively on real audio** for the first time, using the rebuilt
YouTube+iReal corpus. This is the first BP48 number on real (not synthetic,
not oracle-boundary) audio for this architecture family. Expect the synthetic
ceiling caveat to bite — absolute numbers will very likely fall well short of
the synthetic 0.92–0.97 range and the McGill-oracle 0.89/0.776, per the
already-observed real-audio dom recall of 0.21 in issue #19. Results, honest
gap analysis, and shippability verdict are in
`docs/production_deployment_checklist.md` (or, if training does not clear the
bar, the corresponding entry appended to `docs/known_issues.md`).

## Do NOT

- Do not port NNLS-trained weights into the BP48 production path (Q3: net
  −16pp vs native, and z-norm alignment is an additional −9pp on top).
- Do not wire a trigram/transition prior into quality decoding via λ-blending
  (Phase 2A/2C: negative on every measured λ).
- Do not report raw/overall accuracy on the imbalanced quality label set —
  balanced accuracy + per-class recall only (dom recall is the binding
  constraint, per #19/#31).
- Do not claim a number is "real audio" unless it was measured on audio that
  was actually pushed through Basic Pitch this session — verify this before
  quoting, given the fabricated-Billboard-audio incident above.
