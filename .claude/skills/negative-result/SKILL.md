---
name: negative-result
description: Use whenever an experiment, rule, metric or strategy has just been measured and FAILED — before reporting that failure to Louis. Turns "X does not work" into "X does not work, here is why, here is the fix, it is implemented and measured". Also use when about to write words like "échec", "ne marche pas", "measured false", "worse than", "does not help" in a reply.
---

# A negative result is never the end of a message

Louis, 2026-08-07:

> « Quand tu arrives à ce genre de conclusion, sois toujours force de
> proposition et propose une solution que tu implémentes — ici clairement il
> faut un score moins biaisé. Fais-toi en un skill pour que ce soit un réflexe
> dès que tu me donnes un résultat qui n'a pas marché, sauf si on peut prouver
> logiquement qu'on est arrivés à une impasse nette. »

He is not asking for optimism. He is asking that the *diagnosis* be carried one
step further: a measurement that says "this failed" almost always also says
**why**, and the why is a design constraint for the next attempt. Stopping at
the failure throws that away and hands him the work.

## The rule

Before sending a message that reports a failed experiment, do all four:

1. **State the failure plainly, with the number.** No softening. This part does
   not change.
2. **Name the MECHANISM.** Not "it did not work" but "the cost pays for the
   intro's bars, so it always wants the intro shortest". The mechanism is the
   deliverable — it constrains every future attempt.
3. **Propose the fix that the mechanism implies.** The mechanism usually names
   its own cure: a biased objective needs debiasing, a saturating threshold
   needs a per-song scale, a metric that rewards explaining less needs a term
   that pays quantity.
4. **Implement it and measure it before sending.** A proposal he has to
   authorise is still work handed back to him. Send the *result* of the fix, or
   say explicitly that it is running.

## The one exemption

Skip steps 3–4 only when the dead end can be **proved**, not merely observed.
Proof means an argument that no variant of the approach can work, e.g.:

* the target is unrepresentable — a 1-bar intro cannot be written on a 2-bar
  grid, whatever the threshold;
* the channel carries no information — the harmony's median pair similarity is
  0.906 on Let It Be, so it cannot separate anything at any threshold;
* the criterion is monotone in the wrong direction by construction — the mean of
  retained scores rises as the count falls, so its maximum is always the
  emptiest hypothesis.

In that case say *"impasse prouvée"* and give the proof in one sentence. That is
a real result too, and it is what stops the same idea being retried in a month.

## Worked examples from this project

| failure | mechanism | fix that followed |
|---|---|---|
| intro chosen by global writing cost, 10–20 % correct | intro bars are paid for in the cost, so shorter is always cheaper | score only the body, normalised, with a flat "there is an intro" term instead of a per-bar one |
| `moyenne × part` identical to `moyenne` everywhere | coverage barely moves with the threshold, so the multiplier is near-constant and cannot move the argmax | a term that actually varies — the sum, or a per-bar normalisation |
| vocal-recurrence maximised over a free window, 4/10 | maximised freely it finds where singing is most repetitive, i.e. mid-verse | use it as an arbiter between two legal candidates, not as an objective — 7/10 |
| letters collapse to one on Let It Be | median pair similarity 0.906 vs a 0.90 threshold: over half of all pairs pass | per-song threshold, and a channel with no contrast abstains |

## What this is not

Not a licence to pad a failure with speculation. If the fix is not yet
implementable, say what would make it implementable — the missing measurement,
the missing annotation — in one line. One concrete next step beats three
plausible ones.
