# Target output spec — This Love (from Louis's hand-written lead sheet, 2026-07-29)

The reference lead sheet Louis handed me ("this is what I'm expecting"). Writing
it down to confirm I read it correctly and to pin the acceptance target.

## The chart

- **Title**: This Love.
- **One uniform 4/4 bar grid** (~2.52 s/bar). Harmonic-rhythm differences are
  shown as **1 vs 2 chords per bar**, NOT as different bar lengths. This is the
  crux I was getting wrong: "B is 2× faster" means **B has 2 chords per bar**,
  same bar length as A — the standard iReal convention.

- **A** (verse) — 4 bars, **1 chord/bar**, played **×3**:
  ```
  | G | Cm | Fm | Dø |            ×3
  ```

- **B** (chorus) — 4 bars, **2 chords/bar** (the "2× faster" rate). The **last
  two bars change on the 2nd pass** (1st/2nd ending); bars 1–2 are identical both
  times:
  ```
  bars 1-2 (both passes):   | Cm Fm | Bb Eb |
  bars 3-4, 1st pass ① :    | Cm Fm | Bb Eb |
  bars 3-4, 2nd pass ② :    | C  F  | Ab G  |
  ```
  So the 1st pass is `Cm Fm | Bb Eb | Cm Fm | Bb Eb` (a 2-bar loop ×2); the 2nd
  pass replaces the last two bars with `C F | Ab G`. My pipeline already emits
  the 1st pass exactly.

- **C** (bridge) — 4 bars, 1 chord/bar, SAME 1st/2nd-ending shape as B (last two
  bars change on the 2nd pass; bars 1–2 identical both times):
  ```
  bars 1-2 (both passes):   | Fm | Eb▵ |
  bars 3-4, 1st pass ① :    | G7 | Cm  |
  bars 3-4, 2nd pass ② :    | G  | G7  |
  ```
  Unrolled: `Fm Eb▵ G7 Cm  Fm Eb▵ G G7`.

- **Form** (written out, compact):
  ```
  A×3  B  A×3  B  A  C  B×3
  ```

## The five principles this encodes (the general spec, not just This Love)

1. **Uniform bar length**; a faster section = **more chords per bar** (≤2), never
   a shorter bar. (My earlier per-section-bar-length detour was wrong.)
2. **Fold repeats** with `×N` badges (A×3, B×3) — don't print a loop N times.
3. **Distinct sections A/B/C** cut cleanly; the **bridge C must be found**
   (I had cut it off before).
4. **1st/2nd endings** where a repeated section diverges only at the tail.
5. **Form as a compact string** at the bottom.

## Where my pipeline stands vs this (2026-07-29)

Matches structurally: A = `G7 Cm Fm7 Ddim` (≈ G Cm Fm Dø); B already renders
`Cm Fm | Bb Eb | Cm Fm | Bb Eb` (2 chords/bar, via the rigid grid + chart_model's
≤2-per-bar); C bridge = `Fm7 Ebmaj7 G7 Cm …`. Rendered proof:
`scratchpad/this_love_leadsheet.png`.

Remaining gaps to the target: fold the repeats into `A×3 … B×3` with badges;
detect 1st/2nd endings on B; drop the leading intro pickup + the mid-bar passing
`Fdim` so A reads a clean `Dø`; recover the compact form string automatically.
