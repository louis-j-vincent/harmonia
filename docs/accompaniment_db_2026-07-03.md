# Accompaniment database: (song_structure, chords, bass, midi) — 2026-07-03

Goal: a reliable pipeline that takes structured chord sequences from real songs,
renders a musically plausible accompaniment to MIDI, and stores aligned
`(song_structure, chords, bass, midi)` tuples as training data — i.e. ground truth
where the *chart is the label* and the *rendered audio-ready MIDI is the input*,
the exact inverse of Harmonia's inference direction.

## Options scoped (each smoke-tested before deciding)

| option | verdict | evidence |
|---|---|---|
| **iReal Pro charts via pyRealParser** | ✅ chart source | Parses 1,859/1,914 charts from three public corpora (jazz1460, pop400, blues50) with title, composer, style, key, time signature, section markers (`*A/*B`), repeats/codas flattened. |
| **MMA (Musical MIDI Accompaniment)** | ✅ renderer | v25.05.3, pure Python, runs offline from `data/tools/`. 2,022 grooves (Swing, Ballad, BossaNova, JazzWaltz, SlowBlues…). Emits **named tracks** (`Bass`, `Chord`, `Drum`, `Chord-Guitar`) — bass is extractable *by construction*. 163-quality chord table covers the full iReal vocabulary incl. `7alt`, `13sus`, `m11b5`. Deterministic with `RndSeed`. ~0.2 s/song. |
| **pretty_midi hand-rolled comping** | ✅ works, kept as fallback/augmenter | Trivial to generate root-fifth bass + block comping; validated in a 20-line test. Far poorer musical variety than MMA's grooves — use later for augmentation variety, not as primary. |
| **music21 chord realization** | viable but redundant | Same ceiling as the pretty_midi fallback; no groove/style engine. Not pursued. |
| **GarageBand** | ❌ ruled out | No CLI, no AppleScript dictionary, no batch export. GUI-only — cannot be a pipeline component. |
| **iReal Pro app itself (audio export)** | ❌ ruled out | The app renders accompaniment but has no programmatic/batch interface; its *format* is the valuable part, which pyRealParser covers. |

**Decision: iReal corpora (pyRealParser) → MMA → pretty_midi post-processing.**

## Validation performed

1. **Smoke test through every step first** (scratchpad, 5 jazz standards):
   parse → sectionize → chord-map → MMA render → MIDI load → bass extraction.
   Zero unmapped chord tokens; correct forms (All The Things You Are = `A8 B8 C8 D12`,
   All Of Me = `A8 B8 A8 C8`); bossa bass hit the chart's chord root on 100% of
   barlines (swing walking bass is humanized off-grid, handled by tolerance).
2. **Unit tests** (`tests/test_ireal_corpus.py`, 39 tests): chord-token splitting
   (concatenated bars like `Eh7A7b9`, slash basses, glued `n`/`W` symbols),
   iReal→MMA quality mapping validated against MMA's own chord table,
   groove mapping, section-label survival through repeat expansion.
3. **Corpus-scale artifacts found and fixed**: pyRealParser's annotation cleanup
   mangles `alt`→`at` (mapped back); repeat-ending markers (`N2`) can leak into
   chord tokens (stripped); `W` bass-only symbols rendered as power chords;
   section sentinels can glue to the previous bar when repeat expansion drops a
   bar separator (mid-measure sentinel labels the *next* bar).

## Pipeline

```
bash scripts/fetch_accompaniment_deps.sh     # MMA → data/tools/, corpora → data/ireal/
.venv/bin/python scripts/build_accompaniment_db.py
```

- `harmonia/data/ireal_corpus.py` — parsing, sectionized flattening, chord mapping,
  style→groove table, `.mma` chart emission (`MMAChart` dataclass).
- `scripts/build_accompaniment_db.py` — renders each chart, extracts the bass
  track by name, computes a per-song bass-root agreement sanity metric, writes
  `data/accomp_db/db.jsonl` + `midi/<corpus>/*.mid` + `mma/<corpus>/*.mma`.

## Database schema (`data/accomp_db/db.jsonl`, one JSON per song)

- identity/meta: `song_id, corpus, title, composer, style, key, time_signature`
- rendering: `groove, tempo, beats_per_bar, tracks, duration_sec, midi_path, mma_path`
- **structure**: `form` (e.g. `"A8 B8 C8 D12"`) + `section_per_bar` (label per bar)
- **chords**: `chord_timeline` — `{bar, beat, time, ireal, mma}` per chord slot
  (deterministic beat→seconds since tempo is fixed and MMA adds no count-in)
- **bass**: `bass_notes` — `[pitch, start, end, velocity]` from the named Bass track
- QA: `bass_root_agreement` (fraction of bar-initial bass notes matching the
  chart's root, ±15% of a beat tolerance), `unmapped_tokens`

## Results (full run, 2026-07-03, 134 s total)

| corpus | rendered | skipped | mean bass-root agreement |
|---|---|---|---|
| jazz1460 | 1458/1460 | 2 | 96.1% |
| pop400 | 344/345 | 1 | 88.2% |
| blues50 | 54/54 | 0 | 68.6% |

**1,856 songs, 78,928 bars, 110,250 chord events, 268,201 bass notes, 38 MB**
(`db.jsonl` + per-song `.mid`/`.mma`). Groove mix: Swing 1016, Ballad 328,
JazzRock 124, JazzWaltz 116, BossaNova 108, 8Beat 65, Blues 42, 68Swing 21, ….
Most common jazz form: `A16 B8 A8` (AABA with contiguous A's merged, 376 songs).
Blues agreement is lowest by design — blues bass riffs on 3rds/5ths, which is
content, not error. Residual unmapped tokens across the entire corpus: 9
occurrences of one corrupted notation (`C*-^*`) in a single song.

Skipped songs: 1 chart pyRealParser can't parse (coda edge case, "Tequila"),
1 chart with 5 chords in a 4/4 bar (MMA max is 4/line, "Scatterbrain"),
2 jazz charts with empty flattened bodies.

## Known limitations (v1)

- **Mid-tune time-signature changes ignored** (`T34` inside a 4/4 chart) — the
  bar grid assumes one meter per song. Affects a minority of pop400 charts;
  their `chord_timeline` bar/beat times would drift after the change. Detectable
  later by scanning chord strings for mid-tune `T\d\d` markers.
- **One rendering per song** (fixed groove + tempo + seed). Augmentation
  (tempo jitter, alternate grooves per style, seed variation, the pretty_midi
  fallback comper) is a flag away — `tune_to_mma(tempo=…, rnd_seed=…)`.
- **`p` (repeat-previous-chord) mapped to rest** rather than sustaining the
  previous chord — rare in flattened output since pyRealParser fills most slashes.
- Charts that pyRealParser cannot parse at all (~55/1,914, mostly coda edge
  cases) are dropped, not repaired.
- Licensing: chord charts of standards are user-contributed transcriptions;
  corpora stay out of git (`data/ireal/` gitignored) and are fetched from the
  public `ireal-musicxml` test data.

## Relation to the rest of Harmonia

This gives the *forward* direction (chart → MIDI) that POP909 doesn't:
jazz vocabulary (ii-V-I, alt dominants — see `docs/architecture_extensions.md`
risk "POP909 is pop, not jazz"), explicit section structure (POP909 has none),
and a clean bass stem (POP909's piano has no separate bass). Downstream uses:
render MIDI→audio via the existing `MIDIRenderer`/FluidSynth for end-to-end
pipeline evaluation with known ground truth; train bass/chord-change priors on
jazz-flavored progressions; validate the form-clustering work (item #11) against
real section labels, which POP909 could never provide.
