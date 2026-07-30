# Lock propagation demo: end-to-end artifact

Chart: `inferred_maroon_5_this_love.html`  |  operating point: lam=2.0, delta=0.5, K=6, margin_gate=0.0, table=pooled — **this IS the shipped default** (`harmonia/serving/api.py` `_CTX_RESCORE_DEFAULT_*`; see `docs/lock_propagation_tuning.md` sections 8-9 for how it was chosen, including why the margin gate — tested, real, request-overridable — ships OFF by default)

Acoustic backend: `musx_probs` (cache_hit=True)

Generated via `scripts/demo_lock_propagation.py`, which POSTs a real
HTTP-shaped request to `/api/context_rescore/inferred_maroon_5_this_love.html`
through the Flask test client (no server restarted). The endpoint runs
`span_rescore.differential_rescore`: a baseline (no-lock) decode and the
locked decode at IDENTICAL settings, reporting ONLY spans where locking made
a difference relative to that baseline — not spans where the rescore merely
disagrees with what's displayed regardless of the lock (see the tuning doc
for why that distinction matters: an earlier, non-differential version of
this same demo, at a different illustrative config, showed 30 spans changing
from one lock, almost all of it baseline noise unrelated to the lock).

## The lock

Span 39 (bar 29.1, 71.8s): displayed **G7**, but the model's OWN acoustic evidence favours **Cm7** by 2.93 nats (no independent ground truth for this app chart -- this is the 'my ear says this chord is wrong' scenario). Simulated user action: lock this span to Cm7.


## Chart excerpt BEFORE the lock (+/-4 spans)

| span | bar | time | chord |
|---|---|---|---|
| 35 | bar 26.1 | 64.2s | C:min |
| 36 | bar 27.1 | 66.7s | F:min |
| 37 | bar 28.1 | 69.3s | D:hdim7 |
| 38 | bar 28.4 | 71.1s | A#:maj7 |
| 39 | bar 29.1 | 71.8s | G:7/B <- ABOUT TO LOCK |
| 40 | bar 30.1 | 74.3s | C:min |
| 41 | bar 31.1 | 76.8s | F:min |
| 42 | bar 32.1 | 79.4s | D:hdim7 |
| 43 | bar 32.4 | 81.2s | A#:maj |

## Chart excerpt AFTER the lock (+/-4 spans)

| span | bar | time | before | after |
|---|---|---|---|---|
| 35 | bar 26.1 | 64.2s | C:min | C:min |
| 36 | bar 27.1 | 66.7s | F:min | F:min |
| 37 | bar 28.1 | 69.3s | D:hdim7 | D:dim <- CHANGED |
| 38 | bar 28.4 | 71.1s | A#:maj7 | A#:maj |
| 39 | bar 29.1 | 71.8s | G:7/B | C:min <- LOCKED |
| 40 | bar 30.1 | 74.3s | C:min | C:maj <- CHANGED |
| 41 | bar 31.1 | 76.8s | F:min | F:min |
| 42 | bar 32.1 | 79.4s | D:hdim7 | D:hdim7 |
| 43 | bar 32.4 | 81.2s | A#:maj | A#:maj |

## All changed spans (3 total) with the context prior's top-3

| span | bar | before | after | prior top-3 (given final neighbours) |
|---|---|---|---|---|
| 37 | bar 28.1 | D:hdim7 | D:dim | Fmaj (0.186), D#maj (0.156), A#m7 (0.128) |
| 39 | bar 29.1 | G:7/B | C:min | Fmaj (0.192), Gm7 (0.128), D#maj (0.121) |
| 40 | bar 30.1 | C:min | C:maj | F7 (0.154), G#maj (0.127), D#maj (0.119) |
