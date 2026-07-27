# Session 2026-07-27 — P1 (function_family wired) + P2 (second lattice)

Continues `docs/handoff_2026-07-27_post_clock_discovery.md`. A second session was
editing `docs/known_issues.md` live, so this session's findings are logged here
instead; fold the SHIPPED entries into `known_issues.md` once that settles.

## P1 — `function_family` wired default-ON (SHIPPED)

The maj↔dom function fix (`harmonia/models/function_family.py`) is now live: it
rewrites **only** the quality token on same-root maj/dom segments, from
music-x-lab's b7 posterior tie-broken by a "root resolves down a fifth" witness.
Never touches root, bass, or timing.

**Where it is wired:** the very end of `NNLS24ChordHead.run_full`
(`harmonia/stages/chord_head.py`), on the FINAL chord list, before the UI
progress callback so the phone and the returned chart agree. Threaded through
`infer_chords_v1 → _infer_nnls24 → ChordHeadConfig(function_family=…)`.

**Default split (mirrors `segment_source`):**

| path | value | why |
|---|---|---|
| live / server (`live_defaults`, `SHIPPED_CONFIG`) | **ON** | ships the gain to the phone |
| frozen parity oracle (`LIVE_ORACLE_KWARGS`) | **OFF** | goldens predate the brick → stay byte-identical |
| kill switch | `HARMONIA_FUNCTION_FAMILY=0` | rollback with no code change |

**Benchmark (7 frozen songs, family-level partial), ON vs OFF:**

| reference | OFF | ON | Δ |
|---|---|---|---|
| raw | 0.6698 | 0.6967 | **+2.69 pp** |
| repaired (L4) | 0.7114 | 0.7426 | **+3.12 pp** |

`mirex_root` and `bass_root` move by exactly **+0.00** (the brick never rewrites
root/bass — the whole safety claim). **Zero per-song regressions** on both
references: blue_bossa_backing +10.33, blue_bossa +2.18, georgia +1.25, the
other four exactly 0.00.

**Wiring proven faithful through the real pipeline** (not just the offline
harness — the no_chord_policy lesson): on blue_bossa_backing (16 flips),
close_to_you (0 flips), georgia (9 flips), the wired-ON labels are
byte-identical to `function_family.apply(wired-OFF, s7)`, and the
`HARMONIA_FUNCTION_FAMILY=0` kill switch reproduces the OFF chart. Real flips
are musical: `G:maj → G:7` (resolves a fifth down to C = dominant),
`C#:7 → C#:maj7` (does not resolve = major), `G:maj/D → G:7/D` (bass preserved).

**Stop criterion (P1): met.** ≥ +2.0 pp on both references, zero per-song
regressions.

**Tests:** `tests/test_function_family.py` +1 invariant (live ON / oracle OFF /
shipped ON); 937 passed in the fast suite. One **pre-existing, unrelated** red
remains — `test_no_inline_feature_sites.py::test_no_inline_librosa_chroma_sites`
flags `harmonia/models/harmonic_texture.py:175` (inline `librosa.cqt`), shipped
earlier today in 49c0ea1; not touched here.

**To reach the phone: restart the server** (`scripts/harmonia_server.py`) — it
has no reloader.

## P2 — second lattice in the Occam post-pass (FIXED)

**Premise confirmed cheaply first** (Stand By Me, `grid` mode, distance of each
emitted chord boundary to the nearest detected beat):

| config | boundary → nearest beat |
|---|---|
| off + Occam ON (shipped) | 107 ms |
| grid + Occam ON (before fix) | 107 ms — **bit-identical to off** |
| grid + Occam OFF | **0.0 ms** |
| snap + Occam ON | **0.0 ms** |

Mechanism (read from the code, not the prose): when the Occam simplify pass
fires on a clean loop (Stand By Me's D–A vamp, `cov=0.97`), it **re-emits every
span on its own uniform bar grid** (`_apply_occam_to_coalesced`, using
`_flux_anchored_bar_root`'s `np.arange` bars), throwing the real-beat decode
away. `snap` mode already lands at 0.0 ms because its beat-snap runs *after*
Occam.

**Fix:** run that same `snap_chord_times_to_beats` in `grid` mode too (one gate:
`mode in ("snap","grid")`). It re-lays the re-gridded boundaries back onto the
real beats while keeping Occam's label cleanup. Default `off`, so production and
the parity net are byte-identical.

**Stop criterion (Stand By Me collar gain) — met.** `collar gain` = accuracy
with the ±0.25 s around every reference boundary blanked, minus full accuracy
(big = error piled at boundaries):

| reference | off (=grid before fix) | grid (fixed) | reduction |
|---|---|---|---|
| retimed (real timing) | +3.53 | +1.52 | **+2.01 pp** |
| as-is (synthetic grid) | +2.25 | +2.84 | −0.59 |

The fix changes Stand By Me's collar behaviour decisively (before, grid==off →
zero reduction). Against the honest *retimed* reference the grid brick now
reduces boundary error by +2.01 pp (root +2.76). Against the *as-is* reference
it does not — expected, because that reference is itself a synthetic metronomic
grid (§6 of the handoff): moving the model onto real beats moves it away from a
fake reference. P3's collar test is "matched-clock", i.e. the retimed reference.

`beat_grid.py` mechanism verified after 25 s; `mirex_root`/production paths
untouched (grid mode is a research/measurement flag).

## P3 — decide `real_beat_grid` default (in progress)
Criterion: flip only if matched-clock partial gains ≥ +1.0 pp **and** the ±0.25 s
collar gain drops on ≥ 5 of 7 songs (against the retimed reference).
