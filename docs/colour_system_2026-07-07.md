# Design: circle-of-fifths colour system

**Goal.** The interactive chart should use colour as harmonic information, not
decoration. A key colour must be stable, related keys should look related, and
the transpose control should eventually feel like the same object as the scale
highlight: a musical circle mapped onto a colour circle.

## Principle

Map the **circle of fifths** onto hue:

```text
C -> G -> D -> A -> E -> B -> Gb -> Db -> Ab -> Eb -> Bb -> F -> C
```

Neighbouring keys are neighbouring hues. Distant keys are distant hues. Relative
major/minor share one collection colour because they share the same seven pitch
classes; minor is a slightly deeper value, and melodic minor is a more saturated
parallel variant.

This is why the current implementation derives hue from `tonic * 7 mod 12`.
Multiplication by 7 walks pitch classes by fifths, then spreads the 12 positions
around the 360-degree hue wheel.

## Kandinsky-style reading

The useful Kandinsky idea is not "paint it like Kandinsky"; it is that colour can
carry force and direction. The circle-of-fifths map gives every key a location,
then lets the UI express harmonic pull visually:

- Adjacent fifth relations produce small colour moves.
- Chromatic or remote substitutions produce larger colour moves.
- Tonicizations appear as local colour bands inside the broader form.
- Transposition rotates the whole system rather than recolouring arbitrarily.

## Three near-orthogonal anchors

A 12-key circle can be divided into three major-axis anchors four fifth-steps
apart. Those anchors are close to orthogonal in harmonic function and can carry
the three visual primaries:

| axis | fifth-cycle positions | visual anchor | musical reading |
|---|---|---|---|
| I-axis | C / E / Ab family | red-family | stable centre / home pull |
| V-axis | G / B / Eb family | yellow-family | dominant / forward pull |
| IV-axis | F / A / Db family | blue-family | subdominant / side pull |

This is a design metaphor, not a theory claim. The implementation should stay
mechanical and reproducible: hue from circle position, lightness/saturation from
mode/type.

## Transpose wheel

The transpose dropdown should eventually become a compact chord wheel:

- 12 segments ordered by fifths, not semitones.
- Segment fill uses the same `colOf(scale)` colour as highlighted scale bands.
- The current home key is marked as the zero point.
- Dragging or clicking a segment applies that semitone offset.
- The selected transposition rotates labels but does not mutate the underlying
  inferred chart.

That keeps the mental model clean: scale highlight, key legend, and transpose are
three views of the same harmonic coordinate system.

## Implementation status

Current shipped piece: deterministic scale colours in
`harmonia/output/chart_interactive.py`, with relative major/minor sharing colour
and melodic minor using a deeper variant.

Still to build: the actual wheel control. The existing `transpose` select remains
the low-risk interface until the wheel can be tested on desktop and phone.
