# Lock propagation: from dead code to a measured feature in one session

*2026-07-30, branch `feat/chord-context-prior`, commits d36e182→368eea4*

Louis's brief: locking a chord in annotation mode is supposed to re-infer its
neighbours, it "doesn't work at all", and the fix should be a model that
answers *given the chords before and after, what belongs in the middle* —
learning generic patterns (ii-V-I, tritone subs), not songs by heart.

## What we found

- Propagation was **dead by unreachable branch**, not by bad model: every
  confirms-only `/api/reinfer` took a Billboard-backend path that decodes
  from scratch and patches only the locked chord's label. The one component
  designed to propagate was unreachable from the lock UI and targeted the
  retired bp48 frontend anyway.
- Five prior generations of progression-prior work (all pre-deprecation) had
  failed at *unconditional* decoding — but their own diagnosis ("grammar
  needs trustworthy neighbours") is exactly what a user lock supplies.
  Conditional infilling anchored on a lock is a different premise
  (cf. SERENADE, arXiv:2310.11165).

## What we built

1. **Context prior** (`harmonia/models/chord_context_prior.py`): trigram
   counts normalized by the candidate chord — neighbours as (root interval
   from candidate, quality maj/min/dom/hdim/dim) — Witten-Bell-style backoff,
   trained on 3,826 local symbolic songs. Held-out infilling 28.5%/57.6%
   top-1/3 (vs 20.2%/45.7% one-sided). Genre matters: a jazz-only table
   lifts jazz to 39.4% top-1 and finally surfaces the tritone sub (Db7 #6)
   and backdoor (Bb7 #7) in ii-V-I contexts → tables routed by chart
   provenance, never blended (interpolation always loses).
2. **Differential re-score** (`/api/context_rescore`): pool musx frame
   posteriors per displayed span, second-order lattice (acoustic +
   λ·context), decode twice — no-lock baseline vs locked — and apply only
   the differences. Warm round-trip 0.04s.
3. **Tuning on real audio through the production decode** (19 songs): shipped
   λ=2, δ=0.5, K=6. +0.10 neighbours fixed per lock; locking a correct chord
   is safe (0.02 corruptions); locking a fix still corrupts 0.20 (2:1
   against) — the eval has almost no jazz ii-V-I material (JAAH has no
   audio), so this is likely a floor for the app's actual jazz charts.

## Process notes (the verify-don't-trust rule earned its keep three times)

- `pop909_parser.parse_harte_label` silently maps unknown qualities to MAJOR
  (logged, bypassed).
- musx's 73-wide triad posterior packs root-fast; the naive reshape
  transposed root/type — agreement 0.000 until caught by the mandatory
  chart-agreement check.
- The first tuning round's "8.7 corruptions per lock" was ~all pre-existing
  churn conflated into the metric; the differential re-analysis shrank
  magnitudes 5–45×.

## Open

- Jazz-audio eval (the missing measurement): JAAH audio or ear-validated
  iReal-matched charts would test the prior where it's strongest.
- `chart_interactive.py` client still posts to old `/api/reinfer`; port
  pending.
- Margin gate exists as a request knob (`margin_gate`), ships off — improved
  the corruption ratio (2.0→1.33) but never past the bar.
