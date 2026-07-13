# 19 — The app, and the two bugs it flushed out

A design agent handed over a full UI/UX pass on the front-end: a launcher, a
search-first import, an iReal-style chart with Read/Analyse/Annotate modes, a
circular "Compass" chord editor, and the collaborative loop (confirm a chord →
re-infer → see what your fix sharpened). Five surfaces, all vanilla JS, no
build step. The handoff's own framing was the important part:

> **The real inference is messy and under-structured. Do NOT feed it raw.** The
> UI expects one clean, normalised shape. Build a single adapter —
> `to_chart_model(inference) -> ChartModel` — and hand the UI *only* that.

That instruction is what turned a UI integration into a modelling audit.

## The adapter is a measurement instrument

A first attempt at this handoff (another session) had produced a set of
standalone HTML files in `docs/` with mock data and no route pointing at them.
It looked like an app and was never wired to anything. The lesson I took: the
integration surface *is* the adapter, so build that first and point it at real
charts immediately.

`harmonia/output/chart_model.py` reads the payload baked into every
`inferred_*.html` (`const P = {…}`) and normalises it: roots as pitch classes,
one key per tune, sections as real spans, repeats folded to `×N`, at most two
chords per bar, honest calibrated confidence, `t0`/`t1` in real seconds.

Then I ran it over all 17 charts in the library before writing a single line of
UI. That is the whole trick, and it cost about ninety seconds.

## Bug one: every major song was secretly minor

`_parse_home_key` turns a key string into `(tonic, mode)`. It served two
dialects — the iReal DB format (`"Ab"`, `"G-"`) and the pipeline's own
`global_key` (`"G# major"`) — and decided the mode like this:

```python
mode = "minor" if "-" in key[i:] or "m" in key[i:] else "major"
```

The word **"major" contains an "m."**

So `_parse_home_key("C major")` returned `(0, "minor")`, and every real-audio
chart we have ever rendered was baked with the wrong mode. The true-minor songs
came out right *by luck*, because "minor" also contains an "m" — which is
precisely why nobody ever noticed. The output was plausible. It's the project's
error pattern #1, for the fifth time: a low-level string bug producing numbers
that look fine all the way downstream.

And it was not cosmetic. The chart's client JS computes its relative-major
reference from that field:

```js
const maj = h.mode === "major" ? h.tonic : mod(h.tonic + 3, 12);
```

Wrong mode → reference tonic three semitones off → the function/scale colouring
on 9 of our 17 charts has been keyed to the wrong home this whole time. The
colour system that is supposed to teach you *why* a chord does what it does was
quietly lying on every major-key song.

Fixed (test the "maj" prefix before the bare "m"), with a red-first test, and
the already-baked charts repaired in place — the raw key string still lives in
each chart's subhead, so `scripts/fix_chart_home_mode.py` recovers the mode
without re-running inference on anything.

## Bug two: the form was the key name

The adapter next needed the song's form. `P.sections` is a per-bar list of
section labels, so grouping runs of equal labels gives you A/B/C blocks.

Except on real-audio charts, where `section_per_bar` is filled with the *local
key*, so all 330 bars of Autumn Leaves are labelled `"G# major"`. Group that
and you get one section, spanning the entire tune, named "G# major" — and the
chart still renders, and the app still works, and the form ribbon just shows
one chip. It would have shipped.

The actual segmentation is in `P.sectionChips`. One field, two jobs, depending
on which pipeline filled it (#31).

## Bug three: our own ruler was wrong

Every doc in this repo — including the migration guide — says to verify phone
layout with:

```
google-chrome --headless --screenshot=x.png --window-size=390,844 URL
```

On macOS Chrome refuses to make a window narrower than 500px. It renders the
page at **500 CSS px** and scales the screenshot down to 390. You get a
convincing picture of a phone that is not a phone.

I found this because the first chart screenshot showed a tidy 2-column grid
when the code says `repeat(4, 1fr)`. Measuring inside the page:

```
gridClient: 352   gridScroll: 699   cellW: 174   cols: 2
```

Grid items default to `min-width: auto`, so instead of shrinking the type to
fit four bars in 352px, the columns grew to 174px each and **bars 3 and 4 of
every row were clipped off the edge**. In the fake-390 screenshot, this was
invisible: the layout had room at 500px.

So the verification instrument was itself the bug (pattern #6, applied to
tooling). `scripts/phone_screenshot.py` now drives Chrome over the DevTools
Protocol with `Emulation.setDeviceMetricsOverride` — a real 390px mobile
viewport — and can run JS in the page and tap by label, so layout claims get
*measured* (`scrollWidth == clientWidth`) instead of eyeballed.

## What actually shipped

The app is at `/` on the server. It reads `/api/library`, `/api/chart-model`,
`/api/yt-search`, `/api/analyze`, `/api/reinfer` — real endpoints, no mocks
anywhere. The Compass orbits the model's *own* `sug` candidates (Autumn Leaves
bar 25: B♭m 48%, B♭ 24%, Gm 14%), sized by confidence and angled by circle of
fifths. Annotate prints the calibrated confidence on every chord and collapses
the shaky ones to family-only.

Two deliberate deviations from the handoff:

**Playback is the local cached audio, not a YouTube iframe.** The server
already keeps the audio it downloaded for inference, and the shipped chart
already plays it back locally — a decision made specifically to sidestep the
iframe's origin/playsinline/embedding-disabled failures on iOS. The design's
transport UI is kept; it just points at `/audio/<slug>.m4a`.

**The re-infer spinner tells the truth.** The mock implied ~2 seconds. A real
re-infer is two full decodes of the whole track (the constrained one, and the
unconstrained baseline it diffs against) — **19 seconds** for a 7-minute song.
So the overlay says "about 20 seconds" and shows a running clock. The loop
works: confirming `G:7` at 6.7s moved a neighbouring chord from `G:hdim7` to
`C:min7` on its own, which is exactly the propagation story the design is
selling.

## The pattern, again

Three bugs, all found in the first ninety minutes, none of them found by
looking for bugs. They were found by (a) refusing to feed the UI raw model
output, (b) running the adapter over the whole corpus before building on it,
and (c) not trusting a screenshot I hadn't measured.

The handoff said "build one adapter and hand the UI only that." It's good UI
advice. It turns out to be better debugging advice: a normalisation layer is a
place where every unstated assumption about your data has to be written down,
and the moment you write them down, you can test them.
