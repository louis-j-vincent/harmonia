# madmom re-inference — results (2026-07-14)

**TL;DR — negative finding.** madmom is now installed and wired into the Gen-2
pipeline as an opt-in beat backend, but **it does not fix the tempo octave-lock**
(issue #1's premise). On the 10-song `docs/audio` corpus, where madmom diverges
from librosa it drops to *half*-tempo, and on every reference-anchored song it
lands **further** from the true tempo than librosa. Downbeat detection fails
entirely in this env (numpy-2.x). librosa stays the default; madmom kept as
opt-in infrastructure. Consistent with known_issues #9 ("beat tracking not the
bottleneck").

## 1. Install status

- **Installed:** `madmom 0.16.1` into `.venv` (the only release; no py3.12 wheel).
- **Build:** needed `Cython` present + `--no-build-isolation` (madmom's `setup.py`
  imports numpy/Cython at build time). Wheel built and cached.
- **Runtime shim required.** madmom 0.16.1 predates Python 3.10+ (ABCs moved out
  of `collections`) and numpy 1.24+/2.x (removed `np.float` etc.). Its pure-Python
  modules hit these in ~98 places. Rather than patch site-packages, a small
  idempotent compat shim `_ensure_madmom_compat()` (in `harmonia/models/rhythm.py`)
  restores the missing names **before** madmom imports. It only sets aliases numpy
  no longer provides — numpy 2.x re-added `np.bool` as its own scalar type, and
  clobbering it breaks masked arrays (learned the hard way).
- **Downbeat detection is broken in this env.** `DBNDownBeatTrackingProcessor`
  raises `inhomogeneous shape (2,2)` under numpy 2.x on every song → `ndownbeats=0`
  everywhere, time-signature falls back to 4/4. madmom's metre/downbeat advantage
  is therefore unavailable; only its beat/tempo track is usable.

## 2. Wiring

- `infer_chords_v1(..., beat_backend="librosa"|"madmom")` — new opt-in param
  (`harmonia/models/chord_pipeline_v1.py`). Default **librosa** (unchanged
  production behaviour). When `"madmom"`, madmom's tempo + beats feed the same
  tempo-grid de-jitter. Falls back to librosa with a warning if madmom is
  unavailable.
- `RhythmAnalyser(prefer_madmom=True)` (frozen `HarmoniaPipeline`) now actually
  gets madmom too, via the same shim.

> **Env gotcha uncovered (important).** The editable install maps `harmonia` →
> the **stale clone** `~/harmonia/harmonia` (CLAUDE.md's "never work there"). When
> a script is run as a file (`python scripts/foo.py`), `scripts/` is `sys.path[0]`
> and cwd is *not* on the path, so `import harmonia` silently resolves to the stale
> clone — **not** this repo. Edits here are invisible to file-run scripts and the
> server unless the canonical root is forced onto `sys.path`. All scripts here
> insert `_REPO_ROOT` at `sys.path[0]`; a bare `-c` from the repo root also works
> (cwd wins). **The server (`scripts/harmonia_server.py`), if launched as a file,
> imports the stale clone — so the madmom wiring will NOT reach the server until
> either the editable install is repointed at this repo, or the server is launched
> with this repo on `PYTHONPATH`.**

## 3. Tempo comparison (faithful production path: 44.1 kHz mono wav, `sf.read`)

Data: `docs/tempo_comparison_madmom.json` · Plot: `docs/plots/tempo_comparison_madmom.png`
(sorted by librosa↔madmom octave disagreement).

| song | librosa | madmom | ref (src) | madmom vs ref | verdict |
|---|---:|---:|---|---|---|
| blue_bossa_150bpm_backing_track | 99.4 | **75.0** | 150 (filename, **exact**) | **exactly ½×** | madmom worse |
| a_foggy_day | 129.2 | 72.3 | ~150 (approx) | ~½× (−1.05 oct) | madmom worse |
| blue_bossa | 172.3 | 88.2 | ~150 (approx) | ~½× (−0.77) | madmom worse |
| kermit (being green) | 136.0 | 66.7 | — | — | ½× of librosa |
| let it be | 139.7 | 70.6 | — | — | ½× of librosa |
| nina simone feeling good | 114.8 | 75.9 | — | — | ~⅔× of librosa |
| ghost_of_a_chance | 123.0 | 113.2 | ~60 (ballad) | ~2× (both doubled) | both wrong, agree |
| autumn_leaves | 184.6 | 187.5 | ~120 (approx) | ~1.5× (both high) | both wrong, agree |
| airegin | 136.0 | 133.3 | ~220 (bebop) | ~0.6× (both low) | both wrong, agree |
| adele hello | 156.6 | 157.9 | — | — | agree |

**Reading it.** Neither backend hits the octave reliably. The only *exact*
reference (blue_bossa backing track, 150 BPM from the filename): librosa 99.4
(0.66×), madmom **75.0 = exactly half**. Across all reference-anchored songs
(backing track, a_foggy_day, blue_bossa), madmom is *further* from the reference
than librosa — it consistently prefers the slower (half-note) pulse. Where the
two agree (ghost, autumn, airegin, adele), they agree at the **same wrong octave**.

**Octave-lock "fixed" count: 0 / 10.** madmom corrects the octave on no song and
regresses it on the reference-anchored ones.

**Aside (a real second finding):** librosa's own octave choice is
preprocessing-sensitive — autumn_leaves reads **92 BPM at 22.05 kHz** but **185
BPM at 44.1 kHz** (production sr). Any tempo comparison must match the production
load path (this table does); a 22 kHz read understates librosa↔madmom agreement.

## 4. Before/after re-inference (3 representative songs, cached Basic Pitch)

Charts written for both backends (`docs/plots/reinferred_{madmom,librosa}_*.html`).

| song | librosa tempo / #chords | madmom tempo / #chords | note |
|---|---|---|---|
| blue_bossa_150bpm_backing_track | 99.4 / 176 | 75.0 / 157 | key C minor both (correct) |
| ghost_of_a_chance | 123.0 / 212 | 113.2 / 200 | both doubled → both over-segment |
| autumn_leaves | 184.6 / 464 | 187.5 / 472 | both doubled → ~470 chords for a ~32-bar tune |

Chord counts barely move where tempos agree (autumn 464 vs 472) and move only
because the grid density changes where they differ (blue_bossa 176 vs 157) — not
because of a better beat placement. No qualitative improvement from madmom.

## 5. Corpus

- Corpus size: **10** songs (`docs/audio/*.m4a`), all tempo-measured on both
  backends. **3** fully re-inferred to charts per backend (Basic Pitch is the
  slow leg; the tempo finding already settles the question, so the remaining 7
  were not re-charted).
- Tempo accuracy vs GT/annotation: only one hard reference exists (blue_bossa
  backing track = 150). librosa 0.66×, madmom 0.50× — **neither within ±5%**.

## 6. Recommendation

- **Keep librosa as the default** `beat_backend`. madmom provides no octave
  correction here and is worse on anchored songs; downbeats are unavailable.
- **Keep madmom installed + wired** as opt-in infrastructure (it may still help
  on genuinely swung jazz with strong downbeat cues once the downbeat processor
  is fixed for numpy 2.x).
- The real octave-lock lever is **not** the tracker choice — it's an octave
  *disambiguation* step (e.g. pick the octave whose beat count best matches the
  bar-level harmonic-rhythm / chord-change rate, or a min/max-BPM prior from
  detected style). Both trackers land in the [55, 215] band and simply pick the
  wrong multiple; a downstream chooser would fix both.
- Fix the madmom downbeat crash (numpy-2.x ragged array in
  `DBNDownBeatTrackingProcessor`) before relying on madmom's metre.
