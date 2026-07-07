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

**"Show which sections are in which key."** Easy to say. I estimate a local key per
section from its chord tones with the Krumhansl profiles already in the repo, and tint
each section by its key. Validating against the corpus' own key labels gave a
sobering 70% — until I looked at the misses and found they were almost all
relative-major/minor pairs, which *share a scale*. For a feature that colours by scale,
E-minor and G-major wanting the same tint is correct, not wrong. The whole-song number
undersells it anyway: the real payoff is watching "All The Things You Are" light up in
four different keys across its four sections, which no single key label can show.

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
