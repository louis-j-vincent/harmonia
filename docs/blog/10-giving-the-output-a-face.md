# Part 10 — Giving the output a face

For nine parts this project spoke in accuracy numbers. `family 94%`, `seventh 88%`,
a confidence gate that's "honest at every level." All true, all invisible. If you
can't *see* the chart the model heard, you can't feel the difference between a
confident `C^7` and a nervous `C`.

So I built the thing the whole pipeline was secretly for: an iReal-Pro-style lead
sheet. Warm paper, four bars to a row, boxed section letters, double barlines where
a section turns over, and chord symbols typeset the way a jazz musician expects them
— big root, small quality, a real △ for major-7, ø for half-diminished, ° for
diminished, flats and sharps raised where they belong. Two renderers share one
typesetter: a matplotlib PNG and a DOM twin, so the ground-truth iReal charts and the
model's inferred charts come out looking identical. That parity is the point — you
read them side by side.

## Certainty, in colour

The model already knew how sure it was; now you can see it. Every chord is coloured on
a red→green certainty scale, and the label sits at the depth the confidence gate is
willing to defend: a green `Bb^7` where it's sure, an amber `Am` where it backed off
from the seventh to the triad rather than lie. The interactive version lets you pull
the gate threshold yourself and watch the chart descend and retreat — the honesty of
Part 9, made tactile.

## Two requests, and what they taught me

Then came two feature asks that turned into small research questions.

**"Show which sections are in which key."** Easy to say. My first pass estimated one
key per section from its chord tones with the Krumhansl profiles in the repo, and
tinted each section. It validated at a sobering 70% against the corpus' own key labels
— though the misses were nearly all relative-major/minor pairs, which *share a scale*,
so for a scale-colour they're arguably right. But the real problem wasn't the accuracy;
it was the *granularity*. A jazz musician looked at Anthropology and said, correctly,
that a section-wide "Bb major" is a lie: bar 1's `G7` tonicizes ii, bars 5–6 are
secondary ii-V's through Eb and Ab, and the whole bridge is a cycle of dominants —
D7→G7→C7→F7 — each a V pointing a fifth down to a *different* key. One tint per section
can't say any of that.

So the section estimate became a *per-chord* functional analysis. Each chord is
labelled with the key it belongs to locally, the way you read a chart: a dominant is a
V and points a fifth down to its target (so the bridge steps through G, C, F, Bb, one
per chord); a ii binds to its V; a maj7 is a I. Now every chord wears its own
highlight, and Anthropology's A section reads exactly as a player hears it — home,
brief tonicization, home, a climb through three keys, home. The lesson was the one this
project keeps re-learning from the musician in the room: the honest unit of analysis is
rarely the one that's easy to compute.

**"Let me transpose it."** The trick wasn't the transposition — shift twelve
pitch classes, respell for the target key's flats or sharps. It was that I'd been
storing chords as pre-rendered symbols. To transpose live I had to store them
*structurally* — root pitch-class plus quality tail, typeset at draw time — which,
it turned out, is just the better design. The interactive chart now rewrites itself
into any of the twelve keys on a dropdown.

## The one I said no to

The last ask was "9ths and beyond." Before writing a model I ran the premise-check
that this project has learned to run first. Two numbers settled it. Only 10.7% of
chords in the corpus carry any extension, and the tail (13, #9, 11) is under 1% each.
And in the audio itself, the tension pitch classes already carry 30–58% of the
chord-tone energy in chords that *have no extension* — passing tones and reverb put
them there. A deliberate natural 9th is a whisper above that floor. The exception is
the altered tensions: #11 sits at a clean 9–21%, so an alteration is a loud, unusual
signature the audio *can* hear.

So the scope wrote itself: altered tensions are an audio problem, natural extensions
are a symbolic one — a voicing choice, better predicted by the style and progression
priors than by chroma. I didn't build it. I wrote down exactly what building it would
cost, and why the naïve version would score well while learning nothing. The map is
still the map: check where the hard part is before you walk toward it.

## One more layer: suggestion, not inference

The interactive chart now has two deliberately separate transformations. **Fuse
repeats** is a reading aid: adjacent bars with the same displayed harmony collapse
into a wider measure with a repeat count, without changing the underlying chord
sequence. **Jazzify** is a creative layer: a 0–5 cursor adds colour tones, diatonic
extensions, secondary ii-Vs, tritone substitutions, and altered dominants. Jazzified
symbols are marked and the caption says plainly that they are suggestions, not audio
evidence.

The important implementation detail is that both transforms run after the model's
structured output and before DOM rendering. The original inferred chart is still one
drag or checkbox away, transposition still composes cleanly, and the scale bands are
recomputed on the displayed harmony. If a secondary ii-V appears, the highlight has
to explain that local tonicization too; otherwise the feature would be a trick rather
than a musical object.

The colour system got the same cleanup. Scale colours are now deterministic by
circle-of-fifths position, so related keys sit near one another on the hue wheel and
relative major/minor share a collection colour. The transpose control is now a
compact chromatic wheel — keys arranged by semitone for easy navigation, but coloured
by their circle-of-fifths position so you can see harmonic relationships at a glance.

## Making it playful

Two more features turned the chart from a display into an instrument.

**Jazzify with diversity.** The 0–5 slider now samples from possibilities rather than
applying one deterministic transform. At level 3, it might insert just the V or the
full ii–V. At level 4, it chooses between tritone substitution, backdoor dominant, or
keeping the original. A "Re-roll" button resamples with a new seed — same intensity,
different choices. And clicking any bar while Jazzify is active cycles that bar's
level independently, so you can dial in more spice on the bridge while leaving the
head clean.

**Motif mode.** A toggle that turns the chart into a pattern-marking tool. Drag across
bars to select a range, and a popover offers to name it — suggesting "ii-V" or
"dom-cycle" if the shape detector recognizes it. Click "Find similar" to highlight
every occurrence of that pattern (in the same key or transposed, your choice), then
"Group all" to label them with matching brackets. The brackets stack visually when
patterns nest, and clicking one lets you rename it, ungroup just that instance, or
delete the whole motif. It's the compression the motif detector was doing
automatically, but now a human can do it by hand — and it feels like annotating a
score rather than configuring a data structure.
