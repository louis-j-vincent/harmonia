# UG-tab pre-flight — new candidate songs (2026-07-30)

## PASSED (3) — use these

| slug | cached html |
|---|---|
| `let_it_be_remastered_2009` | `scratchpad/ug_cache/let_it_be_17427.html` |
| `katy_perry_hot_n_cold_official_music_video` | `scratchpad/ug_cache/hot_n_cold_733932.html` |
| `the_police_every_breath_you_take_official_music_video` | `scratchpad/ug_cache/every_breath_1087239.html` |

Plus one **conditional**: `ben_e_king_stand_by_me_audio`
(`scratchpad/ug_cache/stand_by_me_1724608.html`) — fails only the literal
`[key]` gate, and that failure is demonstrably an `infer_key` bug, not a
transposed tab (evidence below). Louis's call whether to use it.

---

## Results table

| slug | tab id | rating | votes | tonality | capo | `[key]` | verdict | boundary_contrast | cost_contrast | unsupported_frac | contradicted spans | root agree vs our chart | result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `let_it_be_remastered_2009` | 17427 | 4.811★ | 14002 | C | 0 | **MATCH** (C vs C major) | ok | 0.745 | **21.36σ** | 0.005 | none | 65.2% | **PASS** |
| `the_police_every_breath_you_take_official_music_video` | 1087239 | 4.814★ | 3582 | Ab | **1** | **MATCH** (Ab vs G# major) | ok | 0.768 | **12.20σ** | 0.275 | none | 73.0% | **PASS** |
| `katy_perry_hot_n_cold_official_music_video` | 733932 | 4.858★ | 629 | *(none declared)* | **5** | MATCH (vacuous — see note) | ok | 0.743 | **11.41σ** | 0.010 | none | 72.5% | **PASS** |
| `ben_e_king_stand_by_me_audio` | 1724608 | 4.839★ | 8443 | A | **2** | **MISMATCH** (A vs "C# minor") | ok | 0.399 | 7.46σ | 0.023 | none | **96.0%** | **FAIL on key gate only** — false alarm, see below |

Every other gate passes on all four: rating > 4.7 ✓, verdict `ok` (no `--asr`
needed on any of them) ✓, `unsupported_frac` < 0.45 ✓.

### Capo call-outs (per the This Love precedent)

Three of the four tabs have a capo. In all three the aligner's capo handling is
verified correct by comparing the **sounding** chord vocabulary against our own
chart's:

| slug | capo | written shapes | sounding chords after capo | our chart's chords | verdict |
|---|---|---|---|---|---|
| Hot N Cold | 5 | D/A/G/Em/Bm | G, D, C, Am, Em | G, D, C, Am, Em | identical |
| Stand By Me | 2 | G/Em/C/D | A, F#m, D, E | A, F#m, D, E | identical |
| Every Breath | 1 | G-shapes | Ab-centred (tab declares Ab) | — | `[key]` MATCH |

Note the UG `tonality` field is the **sounding** key, not the shape key
(Stand By Me: G shapes + capo 2, tonality field says "A"). `ug_align.py`
transposes the *chords* by the capo and leaves `tonality` alone, which is the
correct pairing.

---

## Stand By Me: the `[key]` MISMATCH is `infer_key`'s fault, not the tab's

The gate says MISMATCH ⇒ the tab is transposed or the video is pitch-shifted.
Neither is true here, and three independent checks say so:

1. **Chroma energy peak is A** — the tab's declared tonality. Ranked treble
   chroma mass: A 1.00, C# 0.896, E 0.894, F# 0.852, G# 0.783, … The A-major
   scale degrees occupy the top 5 slots.
2. **The capo-2 sounding chords are A, F#m, D, E**, and our own chart —
   computed from the audio with no knowledge of the tab — contains exactly
   `A, F#m(Gb-), D, E` and nothing else.
3. **Root agreement is 96.0%**, the highest of any song run through this
   aligner so far (This Love 82.8%, Close to You 72.5%).

`infer_key` returned "C# minor, conf 1.00" for an A-major song — it picked the
**mediant**, not just the wrong mode. This is a new instance of the known
`infer_key` limitation recorded in `docs/ug_alignment_brick.md` ("infer_key
cannot arbitrate mode … the cross-check therefore compares tonic only"), but
worse: here the *tonic itself* is wrong, so tonic-only comparison does not save
it. Worth logging separately in `docs/known_issues.md`.

Note also Stand By Me's `boundary_contrast` is 0.399 — the lowest of the four,
as expected for a song that is one I–vi–IV–V loop end to end. It is still well
above the 0.12 underdetermined threshold and 7.46σ is a solid cost contrast, so
harmony *can* time this tab.

---

## Every Breath You Take: elevated but passing `unsupported_frac`

0.275 is the highest of the four (cf. Close to You 0.217, Chain of Fools 0.229)
but under the 0.45 gate, and it is **scattered rather than clustered** — the
script printed no `[support]` contradicted runs at all, so there is no
alternate-ending-style block of tab material missing from the recording. The
likely source is the tab's 22 distinct chord tokens over 138 events (it spells
out the riff's `Aadd9`/`Bbsus2`-type voicings), which the 24-d chord-tone
template scores loosely.

---

## STEP 1 — full candidate pool

18 slugs have all three of a baked payload, `.m4a` audio, and an NNLS cache:

```
abba_chiquitita_official_lyric_video
aretha_franklin_chain_of_fools_official_lyric_video      (already done)
bein_green
ben_e_king_stand_by_me_audio                             ← run
blue_bossa
blue_bossa_150bpm_backing_track
carpenters_close_to_you                                  (already done)
land_of_1000_dances
let_it_be_remastered_2009                                ← run, PASS
maroon_5_this_love                                       (already done)
mayer_hawthorne_just_ain_t_gonna_work_out_official_video
muppets_kermit_its_not_easy_being_green_original
nina_simone_feeling_good_lyric_video
ray_charles_georgia_on_my_mind_official_video
the_commodores_easy_1977
the_police_every_breath_you_take_official_music_video    ← run, PASS
the_ronettes_be_my_baby_music_video
yesterday_remastered_2009
```

**The NNLS cache is not actually a gating requirement.** The native
NNLS-Chroma VAMP plugin is installed and working in this env, so
`extract_bothchroma()` generates `data/cache/nnls_infer/<slug>.npz` on first
use (~90 s for a 4-minute track). Hot N Cold had no cache and was run anyway —
it is now cached. The real requirement is only payload + audio, which widens
the pool to **54 slugs** (every `docs/audio/*.m4a` that has a baked payload).

Louis's other two suggestions, `maroon_5_she_will_be_loved_official_music_video`
and `maroon_5_misery_official_music_video`, also have payload + audio and can be
run the same way; not run here because three songs already passed.

### Unused search pages already cached (no refetch needed)

`_search_yesterday.html` — best Beatles Yesterday chords tab is
**4.88★ / 12721 votes, id 887610, tonality F** (runner-up 4.86★ / 12838,
id 17450). A strong next candidate if a 4th song is wanted.

---

## Reproduce

```
.venv/bin/python scratchpad/ug_align.py let_it_be_remastered_2009 \
    scratchpad/ug_cache/let_it_be_17427.html
.venv/bin/python scratchpad/ug_align.py katy_perry_hot_n_cold_official_music_video \
    scratchpad/ug_cache/hot_n_cold_733932.html
.venv/bin/python scratchpad/ug_align.py the_police_every_breath_you_take_official_music_video \
    scratchpad/ug_cache/every_breath_1087239.html
.venv/bin/python scratchpad/ug_align.py ben_e_king_stand_by_me_audio \
    scratchpad/ug_cache/stand_by_me_1724608.html
```

No `--asr` run was needed: no song came back `harmony-underdetermined`.
Outputs are in `scratchpad/ug_align_<slug>.{json,png}`.
