# Bar snapping ↔ section detection: end-to-end coherence audit — 2026-07-30

Brief (Louis): *"be very wary of how we separate bars and sections, this is where we
lose the most, check from one end to the other of the pipeline if our bar snapping
and section detections are coherent, I suspect we might be losing a lot here."*

Audit only — **no file under `harmonia/` was modified**. All scratch scripts live in
the session scratchpad; the three failing checks are pasted below so they can be
copied into `tests/`.

**Working-tree note (read before acting).** A concurrent session has landed
`harmonia/models/section_vocab.py` (**untracked**, `git status` shows `??`) plus
uncommitted WIP in `harmonia/output/chart_display.py` (+69 lines) and
`harmonia/output/chart_model.py` (+394 lines). `section_vocab` is now THE section
detector on the regrid path — `pattern_cover.py`/`pattern_slide.py` in `scratchpad/`
are its prototypes and are referenced by nothing in `harmonia/`. Everything below was
measured against that working tree.

---

## 1. Verdict

**The bar grid and the section detector are internally coherent in their bookkeeping,
and the loss is almost entirely in ONE place: the recovered bar length is the wrong
metrical octave on roughly a third of the corpus, and the section detector inherits
that error at 2× granularity.** The handoffs themselves are clean — `barRanges` is
inclusive at all ~30 producer and consumer sites (no off-by-one anywhere), `bar1` is
exclusive in `vocab_sections` and converted correctly, `reps × d_bars` exactly equals
each section's bar count on 36/36 songs (no bars silently dropped by the fold), and
`apply_rigid_grid`'s half-open binning is correct. There is **no tempo drift on This
Love** — the rigid grid fits the downbeat onsets to 6 ms RMS, so the earlier
0.481→0.507→0.617 s "drift" was purely the mid-bar mixture confound, and chasing it
would have been wasted work. What *is* being lost: `rigid_grid_for` recovers the
correct bar length on only **7/11 songs measured against a trustworthy reference
(16/35 against any reference)**; on a 2×-too-long bar, `section_vocab`'s "half-bar"
slot becomes a whole real bar, the half-bar resolution its own docstring calls
load-bearing disappears, patterns can only start on even real bars, and the shipped
chart degenerates (Close To You: `A×4 B` — two sections for a 3½-minute song). Two
smaller but real incoherences sit alongside it: a 0.15-bar edge backoff in
`_build_bounds` makes `beat == 0` unreachable after every regrid (75 of This Love's
downbeat chords are labelled beat 1), and section `spans` in seconds overlap by up to
3.78 s where `barRanges` are perfectly contiguous — the frontend drives the playhead
from `spans`, so that one is audible. Louis's suspicion is **confirmed in substance
but mislocated**: the money is in the octave, not in the boundary bookkeeping.

Blast-radius calibration: `HARMONIA_REGRID` is set nowhere outside tests, so the
regrid path (and therefore items #1, #2, #3, #5) is **opt-in and currently OFF**.
These are blockers for turning the flag on, not live production bugs. Item #4 is in
the default path today.

---

## 2. Concrete incoherences, most severe first

| # | `file:line` | What is wrong | Failing input (demonstrated) | Consequence |
|---|---|---|---|---|
| **1** | `harmonia/models/rigid_grid.py:172` `_structural_spacing_bar`, surfaced by `rigid_grid_for:230` | The recovered bar is the wrong metrical octave. Also: `rigid_grid_for` **never** returned `None` on 52/52 payloads, so the documented "defer, no regression" gate never engages. | `docs/plots/inferred_carpenters_close_to_you.html` → bar **5.409 s** vs iReal Pro 89 BPM → **2.697 s** (ratio 2.01). Also `jorja_smith_blue_lights` 5.217 vs 2.609 (2.00), `the_jackson_5_abc` 2.053 vs 2.553 (0.80), `ireal_falling` 0.857 vs 1.714 (0.50). | `section_vocab.SLOTS_PER_BAR = 2` then makes a "half-bar" slot **one whole real bar** (2.705 s on Close To You), so a 2-chords-per-bar chorus can no longer show a 2-bar loop; `_claim`'s stride `SLOTS_PER_BAR` means patterns may only start on **even real bars**. Shipped chart with `REGRID=1`: Close To You = **2 sections** `A×4 B`, 42 bars, 30 printed (vs 12 sections / 83 bars with regrid off); Blue Lights = `A×6 B ×5`; Billie Jean = `A B A B A B A B A B A` with nothing folded. |
| **2** | `harmonia/models/rigid_grid.py:206-227` (`_build_bounds`: `eps = 0.15*bar`; `edges = anchor + k*bar - eps`) vs `rigid_grid.py:83-84` (`frac = (t0-bounds[k])/span`, `beat = round(frac*bpb)`) | Every bar **edge** is moved 15 % of a bar earlier, but `beat` is measured from that same moved edge and never compensated. So `beat` carries a constant +0.6-beat bias and **beat 0 is unreachable**. | This Love after regrid: beat histogram `{1: 75, 2: 1, 3: 43}` — **0 of 34** full-bar (downbeat) chords are labelled beat 0, although 75 chords sit exactly on an anchored downbeat. | (a) `section_vocab.build_slots:140` puts a chord in the 2nd half-bar when `beat >= 2`, i.e. at true position **0.225** of the bar instead of 0.500 — a chord 0.57–1.26 s late in its bar lands in the wrong half-bar. Headroom before *every* downbeat flips into slot #2 is only 0.225 bar (0.57 s). (b) Any consumer testing `beat == 0` silently gets nothing: `scratchpad/section_merge_declined.py:113` (`_bar_times` anchors), `harmonia/tab_renderer.py:148`. (c) The accepted This Love form depends on this constant: backoff 0.00 → 18 sections; 0.05/0.10 → correct beats `{0:75, 2:39}` but a different final section; 0.15 → the accepted 14; 0.30 → 18 sections and collapse. |
| **3** | `harmonia/output/chart_display.py:355-360` (`_section_from_vocab` held-bar backfill: `dict(held, beat=0)` copies `t0`/`t1` unclipped) → `chart_model.py:526` `_span_of` | Section `spans` (seconds) overlap. `barRanges` (bars) stay perfectly contiguous, so bar bookkeeping hides it. | This Love, `REGRID=1`: `C2` (bars 22-23) ends **64.20 s**, `A3` (bars 24-27) starts **60.42 s** → 3.78 s overlap. Same at `C5`→`A6` (114.70 / 110.92). Bonus: `A3`'s own two passes overlap each other by 3.16 s (ends 74.30, next pass starts 71.14). | The frontend takes section identity from bar scanning but the **playhead from `spans`** (`app_shell.html:1330-1339`, `hitAtTime:2261`), and a folded section replays each pass at `spans[k][0] - spans[0][0]`. So from 60.42–64.20 s two sections claim the same instant and the highlight can read "A" while the bridge's G7 is still sounding; the overlapping *pass* offsets mis-position folded chords. |
| **4** | `scripts/render_youtube_chart.py:437-439` — `eff_beat = abs_beat - off_c; bar = max(0, eff_beat // bpb); beat = eff_beat % bpb` | The **bar** is clamped to 0 for a pre-grid pickup but the **beat** is not, so every chord with `eff_beat < 0` collapses into bar 0 with `beat = eff_beat % bpb`, colliding with the genuine chord at that beat. The comment above the code asserts the opposite ("collision-free (bar, beat) pair"). | This Love's baked payload: the true first downbeat (`G`, t0=1.080) and the next bar's `Cm` (t0=3.600) **both** carry `(bar 0, beat 3)`; the N.C. at t0=0.440 gets `(0, 2)`. Corpus: **12/52 songs have ≥1 duplicate `(bar, beat)`, 46 chords total.** | `(bar, beat)` is the annotation sidecar's correction key (`chart_model.py:142`, sidecar schema §5.1) — a human ear-correction on one of those chords lands on the wrong chord or on both. Separately, This Love has **1 chord on beat 0 in 80 bars** (corpus median: 0.55 beat-0 anchors per bar), i.e. the default grid's phase is one beat late on this song; 2/52 songs have <2 anchors, which silently degrades `section_merge_declined._bar_times` to uniform interpolation. |
| **5** | `harmonia/output/chart_model.py:287` (WIP) — `if _rd is not None and 2 <= len(_rd[0]) <= 40 and 2 <= _n_letters <= 8` (the committed version was `<= 10`) | The plausibility gate counts sections and letters. It cannot distinguish "clean form" from "grid octave wrong", because an octave-wrong grid produces *fewer* sections, not more. | Close To You → 2 sections / 2 letters → **passes**. Blue Lights → 11 / 2 → passes. Billie Jean → 11 / 2 → passes. | Turning `HARMONIA_REGRID=1` on today ships a degenerate chart on the ~10-12 octave-double songs while looking "clean" to the gate. The gate is currently the only thing standing between #1 and the user. |
| **6** | `docs/known_issues.md:31` | Documents a This Love output the code no longer produces: `form = "A×4 B A×3 B A C B×3"` and "C bridge found (8 bars) **with its ending**". | Actual working-tree output (`REGRID=1`, through `_chart_model_for`): `A×4 B×3 C A×3 B×3 C A D B×3 C B×3 E B×3 E`, 14 sections, 5 letters, and **no section carries an `endings` field at all**. | The authoritative issue tracker disagrees with the code; the next session will debug against the wrong baseline. (Cause: the concurrent session's `section_vocab.py` replaced the detector; its own docstring documents the 14-section form as accepted, so this is doc drift, not a regression.) |
| **7** | `scratchpad/pattern_cover.py:271` (`letters = "ABCDEFGHIJ"`, `letters[len(pats)]`) | Unguarded index → `IndexError: string index out of range` past 10 patterns. `section_vocab.py:341` guards it (`len(pats) >= len(_LETTERS)`) but then **silently stops covering**. | This Love with the grid phase shifted +0.25 bar: `pattern_cover` raises; `section_vocab` does not. | Low severity, but the prototype and the shipped twin diverge in failure mode, so a scratchpad experiment cannot reproduce a production truncation. |
| **8** *(hypothesis — not demonstrated to break a specific song here)* | `harmonia/output/chart_display.py:165` — `blocks = [(i, min(i+P, n_bars)) for i in range(0, n_bars, P)]` | Period without phase (CLAUDE.md #4): the fallback block clusterer always cuts P-bar blocks **from bar 0**, so a form whose first phrase starts at bar 1 is unreachable. Also `_seq_match` uses `k = min(len(a), len(b))`, so a 1-bar trailing partial block matches any full block on its first bar, inflating `coverage`. | Not isolated to a song — `section_vocab` now runs first and this path only fires when `vocab_sections` returns `None`. | Latent. Worth a red test before it becomes live again. |

### What is coherent (checked, found clean — do not spend time here)

- **`barRanges` is INCLUSIVE at every one of ~30 producer and consumer sites**
  (`chart_model.py` ×25, `chart_display.py` ×3, `ssm_block_sections.py` ×3,
  `app_shell.html` ×2, tests ×7). Zero disagreements. `ssm_block_sections.py:236`
  even documents it. `chart_interactive.py` is not a `barRanges` consumer at all
  (it renders the static `docs/plots/*.html` pages from a flat per-bar
  `P.sections[bar]` array); the real consumer is `harmonia/output/app_shell.html`.
- **`vocab_sections` returns `bar1` EXCLUSIVE** (`section_vocab.py:310`) and
  `_section_from_vocab` converts with `y - 1` — correct.
- **No bars are lost in the fold.** `reps × d_bars == bar1 - bar0` on **36/36**
  songs; 0 bars total never rendered. This Love's 14 `barRanges` blocks tile
  0-79 exactly (16+6+2+12+6+2+4+8+6+2+6+2+6+2 = 80).
- **`apply_rigid_grid`'s binning is correct.** `bisect_right(bounds, t0) - 1` is
  proper half-open assignment; sanity check on This Love: 2.5240 s × 80 = 201.9 s
  vs a 205.6 s chord span (the residue is the last chord's tail past the final
  edge, as expected) and vs iReal Pro's 95 BPM → 2.526 s, a 0.08 % match.
- **A whole-bar grid-phase error does not exist as a distinct perturbation** — see §5.

---

## 3. Runnable failing checks (top 3)

All three fail on the working tree; the 4th parametrisation of check 1
(`maroon_5_this_love`) **passes**, which is what makes the check discriminating
rather than just red. Verified: `5 failed, 1 passed in 0.59s`.

```python
"""Red-first checks for the 2026-07-30 grid/section coherence audit."""
import os, re, json, collections
from pathlib import Path
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]          # repo root


def _payload(html: Path) -> dict:
    """Extract the baked `const P = {...}` chart payload — brace matcher copied
    verbatim from scratchpad/section_merge_declined._load_payload."""
    txt = html.read_text(encoding="utf-8", errors="ignore")
    i = txt.index("{", re.search(r"const\s+P\s*=\s*", txt).end())
    depth, j, instr, esc = 0, i, False, False
    while j < len(txt):
        c = txt[j]
        if instr:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                instr = False
        else:
            if c == '"':
                instr = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
        j += 1
    return json.loads(txt[i:j + 1])


# ── CHECK 1 — the grid octave, against iReal Pro (CLAUDE.md trust rank #1) ─────
@pytest.mark.parametrize("slug,ireal_bpm", [
    ("carpenters_close_to_you", 89),                  # FAILS: 5.409 s vs 2.697
    ("jorja_smith_blue_lights_a_colors_show", 92),    # FAILS: 5.217 s vs 2.609
    ("the_jackson_5_abc", 94),                        # FAILS: 2.053 s vs 2.553
    ("maroon_5_this_love", 95),                       # PASSES: 2.524 vs 2.526
])
def test_rigid_grid_bar_matches_ireal_tempo(slug, ireal_bpm):
    from harmonia.models.rigid_grid import rigid_grid_for
    P = _payload(REPO / f"docs/plots/inferred_{slug}.html")
    g = rigid_grid_for(P["chords"], tonic_pc=int((P.get("home") or {}).get("tonic", 0)))
    assert g is not None, f"{slug}: rigid_grid_for deferred"
    bar = float(np.median(np.diff(g)))
    ref = (P.get("bpb") or 4) * 60.0 / ireal_bpm
    assert abs(np.log2(bar / ref)) < np.log2(1.10), (
        f"{slug}: recovered bar {bar:.3f}s is {bar/ref:.2f}x the iReal-Pro bar "
        f"{ref:.3f}s ({ireal_bpm} BPM) -> wrong metrical octave; section_vocab's "
        f"half-bar slot is {bar/2/ref:.2f} real bars")


# ── CHECK 2 — beat 0 must be reachable after a regrid ─────────────────────────
def test_regrid_puts_downbeat_chords_on_beat_zero():
    from harmonia.models.rigid_grid import rigid_grid_for, apply_rigid_grid
    P = _payload(REPO / "docs/plots/inferred_maroon_5_this_love.html")
    bpb = P.get("bpb") or 4
    g = rigid_grid_for(P["chords"], tonic_pc=int((P.get("home") or {}).get("tonic", 0)))
    chords, _ = apply_rigid_grid(P["chords"], g, beats_per_bar=bpb,
                                 drop_before_grid=True)
    bar = float(np.median(np.diff(g)))
    per_bar = collections.Counter(c["bar"] for c in chords)
    downbeats = [c for c in chords
                 if per_bar[c["bar"]] == 1 and (c["t1"] - c["t0"]) >= 0.8 * bar]
    assert downbeats
    on_zero = sum(1 for c in downbeats if c["beat"] == 0)
    assert on_zero == len(downbeats), (
        f"only {on_zero}/{len(downbeats)} full-bar chords are labelled beat 0; "
        f"beat histogram = "
        f"{dict(sorted(collections.Counter(c['beat'] for c in chords).items()))}. "
        f"Cause: _build_bounds subtracts eps=0.15*bar ({0.15*bar:.3f}s) from every "
        f"edge but apply_rigid_grid measures `frac` from that same backed-off edge, "
        f"so a true downbeat sits at frac=0.15 -> round(0.6) = beat 1.")
    # FAILS TODAY: "only 0/34 ...; beat histogram = {1: 75, 2: 1, 3: 43}"


# ── CHECK 3 — section `spans` (seconds, the playhead's source of truth) ───────
def test_section_spans_do_not_overlap():
    os.environ["HARMONIA_REGRID"] = "1"
    from harmonia.output.chart_model import to_chart_model
    P = _payload(REPO / "docs/plots/inferred_maroon_5_this_love.html")
    m = to_chart_model(P, filename="inferred_maroon_5_this_love.html")
    flat = sorted((float(a), float(b), s["id"])
                  for s in m["sections"] for (a, b) in s.get("spans", []))
    bad = [(flat[i], flat[i + 1]) for i in range(len(flat) - 1)
           if flat[i + 1][0] < flat[i][1] - 1e-6]
    assert not bad, "section spans overlap in TIME:\n" + "\n".join(
        f"  {a[2]} ends {a[1]:.2f}s but {b[2]} starts {b[0]:.2f}s "
        f"(overlap {a[1]-b[0]:.2f}s)" for a, b in bad)
    # FAILS TODAY: C2 ends 64.20s / A3 starts 60.42s (3.78s);
    #              A3 ends 74.30s / A3 starts 71.14s (3.16s);
    #              C5 ends 114.70s / A6 starts 110.92s (3.78s)
```

---

## 4. The drift answer: **there is no tempo drift on This Love**

Three ways, all agreeing. (This-Love-only; not a corpus claim.)

| test | n | fitted period | residual RMS | max abs residual | quadratic curvature *c* | 95 % CI on *c* | verdict |
|---|---|---|---|---|---|---|---|
| chord onsets, **downbeat class only** (sole onset in its recovered bar, duration ≥ 0.8 bar) | 34 | 2.52483 s / bar | **6.2 ms** | 15 ms (0.6 % of a bar) | −2.98 × 10⁻⁶ s/bar² | [−9.02 × 10⁻⁶, +3.07 × 10⁻⁶] | contains 0 → **not significant** |
| **beatthis beat times** (independent of the chord decoder and of any mid-bar mixture) | 308 | 0.63122 s / beat | 8.5 ms | 121 ms | +2.72 × 10⁻⁸ s/beat² | [−1.08 × 10⁻⁷, +1.62 × 10⁻⁷] | contains 0 → **not significant** |
| linear residual by song third (the honest version of the earlier measurement) | 34 | — | **5 ms / 5 ms / 5 ms** | — | — | — | **flat** |

Implied tempo change end-to-end: **95.05 → 95.05 BPM** (0.00 %). 0 of 34 downbeat
chords have a residual larger than a quarter-bar (631 ms), so no chord can be pushed
into the wrong half-bar slot by drift. The 6 ms RMS is at the payload's own onset
quantisation (`t0` values are multiples of 0.02 s), i.e. the fit is as tight as the
data allows.

**The earlier 0.481 → 0.507 → 0.617 s trend was entirely the mid-bar mixture
confound**, exactly as suspected in the brief: a chord sitting mid-bar is ~1.26 s from
an edge by construction, and This Love's later thirds are chorus-heavy (2 chords per
bar), so the *mixture* shifted, not the tempo. Restricting to downbeat-class onsets
removes the trend completely. **Do not spend budget on drift correction for this
song.** Cross-check: iReal Pro says 95 BPM → 2.526 s/bar; the fit says 2.52483 s
(0.08 % apart).

---

## 5. Corpus octave table for `rigid_grid_for`

Reference hierarchy, and why each tier is justified:

- **Tier A — symbolic-exact.** `inferred_ireal_*.html` payloads were baked from a
  symbolic chart, so their own `(bar, beat)` label grid *is* the true grid. Verified
  per song, not assumed: the residual of `t0` against a uniform fit of
  `bar*bpb + beat` is 0.00–0.17 beats. `bar_ref = bpb × fitted beat duration`.
- **Tier B — iReal Pro tempo.** A real `docs/plots/irealb_<slug>.html` (tempo ≠ the
  120 default, > 4 bars). CLAUDE.md puts iReal Pro first in the trust order.
  `bar_ref = bpb × 60/tempo`.
- **Tier C — beat tracker.** `bpb × median(diff(raw_beat_times_v2))`. **WEAK**, and
  proven so here: on `bein_green` tier B says 75 BPM while the tracker says 148, so
  a tier-C classification would have called a correct grid `OCTAVE_DOUBLE`. Report
  separately; do not pool with A/B.

### Tier A + B (the defensible measurement): **7/11 OK**

| song | tier | reference bar | recovered bar | ratio | class |
|---|---|---|---|---|---|
| `maroon_5_this_love` | B (iReal 95 BPM) | 2.526 | 2.524 | 1.00 | **OK** |
| `bein_green` | B (iReal 75 BPM) | 3.200 | 3.213 | 1.00 | **OK** |
| `ireal_autumn_leaves` | A | 1.714 | 1.718 | 1.00 | **OK** |
| `ireal_bein_green` | A | 1.996 | 2.001 | 1.00 | **OK** |
| `ireal_billie_jean` | A | 1.714 | 1.716 | 1.00 | **OK** |
| `ireal_i_can_t_help_it` | A | 1.714 | 1.716 | 1.00 | **OK** |
| `ireal_i_m_a_fool_to_want_you` | A | 3.429 | 3.429 | 1.00 | **OK** |
| `carpenters_close_to_you` | B (iReal 89 BPM) | 2.697 | 5.409 | 2.01 | **OCTAVE_DOUBLE** |
| `jorja_smith_blue_lights_a_colors_show` | B (iReal 92 BPM) | 2.609 | 5.217 | 2.00 | **OCTAVE_DOUBLE** |
| `ireal_falling` | A | 1.714 | 0.857 | 0.50 | **OCTAVE_HALF** |
| `the_jackson_5_abc` | B (iReal 94 BPM) | 2.553 | 2.053 | 0.80 | **OTHER** |

### Tier C (weak reference, reported for shape only): 9/24 OK

`OCTAVE_DOUBLE` 8 (`maroon_5_misery`, `mayer_hawthorne_just_ain_t_gonna_work_out`,
`mayer_hawthorne_maybe_so_maybe_no`, `mayer_hawthorne_the_walk`,
`michael_jackson_beat_it`, `michael_jackson_billie_jean`, `sade_like_a_tattoo`,
`the_police_every_breath_you_take`) · `OCTAVE_HALF` 1
(`sam_smith_i_m_not_the_only_one`) · `QUAD` 2 (`abba_chiquitita`,
`jorja_smith_on_my_mind` — 7.046 s bars) · `OTHER` 4 · `OK` 9.

### Totals

| category | count |
|---|---|
| classified against **any** reference | **16/35 OK (46 %)** — 10 DOUBLE, 2 HALF, 2 QUAD, 5 OTHER |
| classified against a **trustworthy** reference (A+B) | **7/11 OK (64 %)** |
| no independent reference available | 17 |
| symbolic payloads without `t0`/`t1` (excluded) | 5 |
| `rigid_grid_for` returned `None` (deferred) | **0/52** |

**On `docs/this_love_mistakes_and_fixes.md` item 6's "11/18 recover the octave":
not reproducible as stated.** I could not recover the denominator 18 from any
defensible subset of the corpus. The measurement is 16/35 (46 %) against any
reference and 7/11 (64 %) against the trustworthy ones — the same ballpark as 61 %,
so the *claim* is directionally right, but the number should be replaced with one of
these, with its reference tier named.

**Second, unreported finding on the same line: `rigid_grid_for` never defers.** Its
docstring promises "Returns `None` on any failure or too-few chords (defer, no
regression)"; over 52 payloads it returned a grid every single time, including on
every octave-wrong song. There is no confidence gate at all — the only gate
downstream is `chart_model.py:287`, which #5 shows cannot see this failure.

---

## 6. Grid-phase sensitivity (±1 bar, ±½ bar) — This Love, shipped path

Method: rebuild the rigid grid at a shifted phase **over the same time span** (so the
bar count is free to change), monkeypatch `rigid_grid_for` to return it, run
`chart_model.to_chart_model(HARMONIA_REGRID=1)`, derive per-bar labels from
`barRanges`, and compare **boundary times in seconds** against the hand GT boundaries
(bars 16 / 24 / 36 / 44 / 48 / 56 on the base grid = 41.1 / 61.3 / 91.6 / 111.8 /
121.9 / 142.1 s), tolerance ¾ bar.

| grid phase error | bars | GT boundary recall | boundaries found | extra | form |
|---|---|---|---|---|---|
| **0** | 80 | **6/6** | 13 | 7 | `A×4 B×3 C A×3 B×3 C A D B×3 C B×3 E B×3 E` (14 terms) |
| **−1 bar** | 80 | **6/6** | 13 | 7 | *byte-identical to 0* |
| **+1 bar** | 80 | **6/6** | 13 | 7 | *byte-identical to 0* |
| **−½ bar** | 81 | **6/6** | 21 | **15** | `A×4 B×3 C A×3 B D B C A E A F B D B C B×3 C B D×2 C G` (22 terms) |
| **+½ bar** | 81 | **6/6** | 21 | **15** | *identical to −½ bar* |

Read it this way:

1. **A whole-bar phase error is a non-event.** Shifting a uniform grid by exactly one
   bar over a fixed span produces literally the same set of edges, so form,
   boundaries and times are byte-identical. (My first attempt appeared to show ±1 bar
   destroying the form; that was an artifact of the shifted grid covering a
   *different* time span, caught by printing the boundary times — CLAUDE.md #1.
   Recording it so nobody re-derives the false version.)
2. **The boundaries are robust to a half-bar error; the vocabulary is not.** All 6
   true section boundaries survive a ½-bar phase error — Louis's new
   claim-at-any-bar-aligned-peak design really is phase-tolerant where the old
   fixed-phase walk was not. But extra boundaries go **7 → 15** and the form goes
   **14 → 22 terms**: `B×3` shatters into `B D B C`, and the `×N` folding — the whole
   point of a lead sheet — collapses. So a half-bar grid error does not move the
   cuts, it destroys the *naming*.
3. **The `_build_bounds` backoff constant is a third, hidden phase knob** with only
   0.225 bar of headroom (item #2): 0.00 → 18 sections, 0.05/0.10 → 14 sections but a
   different tail, 0.15 (shipped) → the accepted 14, 0.30 → 18 sections. The accepted
   This Love output sits on a narrow plateau of an untested constant.

---

## 7. Ranked fix list, with try-order

Order is by expected recovered form-quality per hour, not by severity alone.

1. **Break the grid octave with an external cue, and make `rigid_grid_for` defer
   when it can't.** This is the whole ballgame: 4/11 songs wrong on the best
   reference, and every one of them ships a degenerate chart. Cheapest premise-check
   first (CLAUDE.md #2, ~30 min): for the 11 tier-A/B songs, does
   `median(diff(raw_beat_times_v2)) × 4` already discriminate the correct octave
   *once you know whether the tracker itself is octave-locked*? Test on `bein_green`,
   the one case where the tracker is provably 2× fast. If the beat tracker alone
   can't do it, the next cue is the drum stem — `docs/research_sessions/rhythm_ssm_2026-07-30.md`
   already has demucs cached and a working per-bar onset-energy feature, and
   *kick-on-1* is a much easier question than *fill detection*. Acceptance: tier-A/B
   OK rate from 7/11 to ≥ 10/11, and `rigid_grid_for` returning `None` on the rest
   instead of a confident wrong answer. Nothing below this line is worth doing first.
2. **Fix the `_build_bounds` / `apply_rigid_grid` phase leak (item #2).** ~15 min,
   mechanical: either subtract `eps` back out before computing `frac`, or pass the
   un-backed-off anchor separately so the drop-pickup test and the beat labelling use
   different references (they are two different jobs sharing one constant today).
   Then re-verify This Love's form. Note this WILL move the output — at backoff
   0.05/0.10 the final section reads `C` not `E` — so it needs Louis's ear on which
   is right, and it must be done before any threshold in `section_vocab` is retuned,
   or the thresholds get calibrated against a biased slot grid.
3. **Clip the backfilled held chord in `_section_from_vocab` (item #3).** ~10 min:
   the copied chord dict needs `t0` raised to the bar's own start (and `t1` to its
   end) instead of inheriting the previous section's times. Audible fix — validate by
   playing This Love from 58 s and watching the section highlight, not by re-reading
   the numbers.
4. **Make the plausibility gate see the octave (item #5).** The current
   section-count/letter-count gate is blind by construction. A gate that *can* see it:
   reject when the recovered bar disagrees with an independent tempo cue by more than
   10 %, or when the fold prints fewer than ~⅓ of the song's bars *and* fewer than 3
   letters. Do this after (1) or it will just mask the symptom.
5. **Fix the `(bar, beat)` pickup collision in `render_youtube_chart.py` (item #4).**
   Affects 12/52 songs and the human-correction key, which is the one thing that must
   never silently mis-target. Cheap: allow bar 0's pickup chords a distinct key
   (negative beat, or a dedicated pickup bar) instead of clamping the bar while
   leaving the beat modular. Red test: assert `(bar, beat)` is unique across every
   baked payload.
6. **Update `docs/known_issues.md` for the `section_vocab` swap (item #6)** and
   record that `endings` no longer ships. Five minutes, and it is the difference
   between the next session debugging the code and debugging the doc. **Coordinate
   with the concurrent session that owns `section_vocab.py` before editing** — that
   file is untracked and `chart_model.py`/`chart_display.py` carry its uncommitted WIP.
7. **Guard `pattern_cover`'s letter alphabet (item #7)** and add a red test for
   `_segment_at`'s phase-0 blocking (item #8) so the latent fallback can't quietly
   become live again.

### Explicitly NOT worth doing

- **Tempo-drift correction on This Love** (§4): there is no drift to correct. If
  drift work is wanted, first re-run §4's downbeat-class + quadratic-CI protocol on
  a song with real drift; do not generalise from the discredited thirds measurement.
- **Hunting `barRanges` off-by-ones** (§2): the convention is inclusive at all ~30
  sites and I could not find a single disagreement.
- **Auditing the fold for dropped bars**: `reps × d_bars` is exact on 36/36 songs.

---

## Files

- Scratch scripts (session scratchpad, not committed): `a1_sanity.py`,
  `b1_corpus.py`, `b2_octave.py`, `c1_drift.py`, `d1_sensitivity.py`, `d2_sens.py`,
  `test_grid_section_coherence.py`.
- Code read (never modified): `harmonia/models/rigid_grid.py`,
  `harmonia/models/section_vocab.py` *(untracked WIP)*,
  `harmonia/output/chart_model.py` *(WIP)*, `harmonia/output/chart_display.py`
  *(WIP)*, `harmonia/output/app_shell.html`, `harmonia/serving/render.py`,
  `scripts/render_youtube_chart.py`, `scratchpad/pattern_slide.py`,
  `scratchpad/pattern_cover.py`, `scratchpad/section_merge_declined.py`.
