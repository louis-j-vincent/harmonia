# Mission 3 — UI contract for `/api/reinfer` (re-analyze with user corrections)

*Handoff for the design-focused UI session, 2026-07-13. The model layer and the
server endpoint are built, tested, and committed; this document is the contract
the UI builds against. It is self-contained — you do not need the reasoning
trail, only what is written here. Everything below is verified against a running
server on Autumn Leaves.*

## 0. What this feature is

The annotator already captures **chord edits** and **section merges** (see
`docs/annotation_sidecar_schema.md`). Mission 3 turns those from display-only
overrides into **inference factors**: a re-decode where a user's confirmed chord
is a hard constraint that **propagates** to its neighbours through the joint
decoder, and a merge pools two sections' acoustic evidence. The payoff the UI
must surface: *"you confirmed one chord; here are the N other chords your
correction also changed."*

One new route does the work: `POST /api/reinfer/<chart-filename>`.

---

## 1. Endpoint

```
POST /api/reinfer/<filename>
Content-Type: application/json
```

`<filename>` is the chart HTML filename the UI already uses everywhere else
(e.g. `inferred_autumn_leaves.html`) — the same key as `/api/annotations/<filename>`.

**Preconditions.** Re-inference runs the real decoder on the song's cached local
audio, so it only works for charts that have cached audio (analyzed via the app's
YouTube path — the same songs that have `window.HARM_AUDIO_URL`). If there is no
cached audio the route returns **404** with an `error` string. The first call
per server process is ~15–20 s (cold Basic-Pitch); the constrained re-decode
itself is a stage-1 **cache hit (~2 s)**. Show a spinner; do not block the UI.

### 1.1 Request body

```jsonc
{
  "confirms": [                 // chord-confirm / edit factors (may be [])
    { "t0": 6.178, "t1": 6.818, // span in SECONDS (see §3 for bar→time)
      "root": 2,                // pitch-class 0..11 of the confirmed chord
      "q5": 2 }                 // optional: model quality 0..4
                                //   0 maj · 1 min · 2 dom · 3 hdim · 4 dim
      // — OR, instead of q5, send the sidecar's iReal tail:
      // "q": "7"               // e.g. "-7", "^7", "7", "-7b5", "o7"; server maps it
  ],
  "merges": [                   // section-merge factors (may be [])
    { "spans": [ [6.178, 20.0], [20.0, 34.0] ] }   // ≥2 equal-length SECOND spans
  ]
}
```

- At least one of `confirms` / `merges` must be non-empty, else **400**.
- **Spans are in seconds, not bars.** The UI holds `window.P` with per-chord
  `t0`/`t1`; convert bar/beat addresses to seconds from those (§3). This keeps
  the server stateless — it never re-parses the chart payload.
- `merges`: each group's spans must be **equal musical length** (equal number of
  beats). Unequal spans are rejected server-side and that merge is skipped (a
  warning is logged; the response still returns). v1 supports the common
  two-identical-sections case.

### 1.2 Response (200)

```jsonc
{
  "key": "G# major",
  "tempo_bpm": 112.3,
  "n_changed": 2,
  "chords": [                   // the FULL re-decoded chart, in order
    { "index": 4, "label": "D:7",
      "start_s": 6.037, "end_s": 6.687, "duration_beats": 1,
      "confidence": 0.0,        // CALIBRATED P(correct) 0..1 — see §2
      "confidence_raw": 0.971,  // pre-calibration max-softmax (diagnostic only)
      "changed": true },        // differs from the same-config UNCONSTRAINED decode
    // ... one entry per chord ...
  ],
  "diff": [                     // ONLY the chords that changed — what to highlight
    { "index": 4, "start_s": 6.037, "end_s": 6.687,
      "old_label": "G:hdim7", "new_label": "D:7",
      "old_confidence": 0.3291, "new_confidence": 0.0 },
    { "index": 5, "start_s": 6.687, "end_s": 8.638,
      "old_label": "G:hdim7", "new_label": "G:7",
      "old_confidence": 0.3291, "new_confidence": 0.1818 }
  ]
}
```

- `label` is `"<root-note>:<quality>"` where quality ∈
  `{maj, maj7, min, min7, 7, hdim7, dim, dim7}` (model Harte; map to your iReal
  display tail — e.g. `maj7`→`^7`, `min7`→`-7`, `7`→`7`, `hdim7`→`-7b5`,
  `dim7`→`o7`, `maj`→``, `min`→`-`, `dim`→`o`).
- `chords` is the complete new timeline; `diff` is the **subset** that changed
  vs the *same-configuration unconstrained* decode. `diff` is the propagation
  story to surface; `chords` is what to render.
- `index` is positional within this response's `chords` array only. Because a
  re-decode can re-segment, do **not** assume `index` aligns 1:1 with your
  existing `P.chords` grid — match by **time overlap** (`start_s`/`end_s`), which
  is stable. (This is the §5.1 positional-identity caveat from the sidecar
  schema, and why the response carries explicit spans.)

### 1.3 Errors

| status | body | meaning |
|---|---|---|
| 400 | `{"error": "No corrections to apply."}` | empty confirms + merges |
| 404 | `{"error": "No cached audio …"}` | chart has no local audio to re-infer from |
| 500 | `{"error": "Audio transcode failed: …"}` | ffmpeg problem |

---

## 2. Semantics the UI must respect

- **`confidence` is calibrated `P(correct)`, 0..1.** It is the decoder's
  posterior for that chord after isotonic calibration — a real "how sure is the
  model" number, not a raw softmax. **Drive visual uncertainty from it**
  (opacity/hedged typography/"?" affordance where low; full commitment where
  high). Prefer `confidence` over `confidence_raw` (the latter is a diagnostic).
- **A confirm is a HARD constraint (a clamp).** The re-decode pins the confirmed
  `(root, quality)` on that span. Note a subtlety visible in the example: chord
  #4's returned `confidence` is `0.0` because the user confirmed `D:7` where the
  acoustics wanted `G:hdim7` — **the returned confidence reflects acoustic
  agreement, not the user's certainty.** So the UI must treat a user-confirmed
  chord as *certain by fiat* (render it committed, e.g. the existing `.corrected`
  ✓ pin), and must **not** use the returned low `confidence` to hedge a chord the
  user themselves asserted. Use the returned `confidence` only for chords the
  user did *not* confirm.
- **A merge asserts two sections are the same material.** The re-decode pools the
  two spans' per-beat acoustic evidence (more observations → lower variance) and
  both spans come back with **one shared decoded chord sequence**. The UI should
  present a merge as "these are the same," and after re-infer both spans will
  read identically. (Merge pooling is gated by the user's assertion — it is never
  a blind average.)
- **The `diff` is the highlight target.** After a re-infer, briefly highlight the
  `diff` chords and tell the user "your correction also changed N chords" — that
  visible propagation *is* the collaborative feature. Match diff entries to your
  rendered chords by time overlap.

---

## 3. Sidecar → request mapping (what's existing vs new)

**Already in the sidecar** (`docs/annotation_sidecar_schema.md`, persisted via
`POST /api/annotations/<filename>`):
- `chords[]`: `{bar, beat, root, bass, q, old, ts}` — chord corrections keyed by
  `(bar, beat)`, quality as the iReal tail `q`.
- `merges[]`: `{id, spans, label, ts}` where `spans` are inclusive **bar ranges**
  `[[startBar, endBar], ...]`.

**New for re-infer** (the UI computes these from the payload it already holds):
- Convert each correction's `(bar, beat)` to its chord's **`t0`/`t1` seconds**
  by finding the `P.chords` entry with that `(bar, beat)` (they carry `t0`/`t1`).
  Emit `{t0, t1, root, q}` (send `q` as-is; the server maps `q`→`q5`).
- Convert each merge's bar ranges to **second spans**: `t0 = min t0` over
  `P.chords` with `bar` in the range, `t1 = max t1`. Emit `{spans: [[t0,t1],…]}`.
- Corrections/merges on chords lacking `t0`/`t1` (older charts) can't be sent —
  skip them (the payload flags timing presence).

No sidecar schema change is required. The re-infer route is **read-only** w.r.t.
the sidecar — it does not persist anything; keep using `/api/annotations` to save.

---

## 4. Working curl example (Autumn Leaves)

Server must be running with cached audio for the chart (Autumn Leaves has it):

```bash
# start a scratch server (the live one is on 7771; use another port to test)
.venv/bin/python scripts/harmonia_server.py --no-open --port 7799 &

# confirm the chord at 6.178–6.818 s as D7 (root=2, iReal q="7")
curl -s -X POST http://127.0.0.1:7799/api/reinfer/inferred_autumn_leaves.html \
  -H 'Content-Type: application/json' \
  -d '{"confirms":[{"t0":6.178,"t1":6.818,"root":2,"q":"7"}],"merges":[]}'
```

Returns (abridged) — the confirmed chord plus one propagated neighbour:

```json
{ "key":"G# major", "tempo_bpm":112.3, "n_changed":2,
  "chords":[ ... 255 entries ... ],
  "diff":[
    {"index":4,"start_s":6.037,"end_s":6.687,
     "old_label":"G:hdim7","new_label":"D:7",
     "old_confidence":0.3291,"new_confidence":0.0},
    {"index":5,"start_s":6.687,"end_s":8.638,
     "old_label":"G:hdim7","new_label":"G:7",
     "old_confidence":0.3291,"new_confidence":0.1818} ] }
```

Chord #4 is the confirmed one; chord #5 (`G:hdim7` → `G:7`) is the model
**re-deciding a neighbour** because of the confirm — the propagation to show.

---

## 5. Notes / boundaries

- The endpoint lives in `scripts/harmonia_server.py` (`api_reinfer`). The
  constraint model layer is `harmonia/models/user_constraints.py` +
  `harmonia/models/joint_decode.py` (clamp/pool) — you should not need to touch
  either; drive everything through the route.
- Propagation strength: confirms are re-decoded with the progression transition
  factor **on** (it is off in the default production decode because unanchored it
  over-smooths jazz; a user anchor is exactly the reliable evidence that makes it
  pay off). This is a server-side decision — the UI does not set it.
- Empirically (jazz held-out) confirming the k lowest-confidence chords improves
  majmin on the *other* chords by ~+0.5pp overall and up to +6pp on the immediate
  neighbours — i.e. the `diff` is usually small and local, which is the right UX
  shape (a correction sharpens its neighbourhood, it does not reshuffle the song).
```
