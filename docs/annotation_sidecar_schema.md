# Annotation sidecar schema

Design doc for the per-song annotation sidecar produced by the professional
annotator tool (`docs/architecture_extensions.md` §13). Data model only — no
UI, no re-inference. Everything here is derived from the *existing* payload and
server code; friction with that code is called out explicitly in §5.

Cross-references (read these before editing the schema):
- `harmonia/output/chart_interactive.py::render_interactive()` — builds the
  `payload` dict serialized into `%%PAYLOAD%%` / `window.P` (lines ~165–251).
- `harmonia/output/chart_interactive.py` JS — `motifState` (line ~1852),
  `saveMotif`/`renderMotifBrackets` (~2130–2180), `pickLevel` (~848).
- `scripts/harmonia_server.py` — `_YT_IDS_FILE`/`_load_yt_video_ids`/
  `_remember_video_id` (166–184), `serve_chart` (~1348).

---

## 1. What the existing data model gives us

The payload the chart is built from (`window.P`):

```
P.chords : [ {                         // one entry per detected chord
    root : int,          // pitch-class 0..11
    bass : int,          // pitch-class 0..11, or -1 if no explicit /bass
    bar  : int,          // 0-based bar index
    beat : number,       // beat-in-bar (offset, sorted within a bar)
    lv   : { family : {q,c}, seventh : {q,c}, exact : {q,c} },
                         //   q = iReal quality tail (e.g. "-7", "^7", "7b9")
                         //   c = confidence 0..1 at that descent level
    t0   : number?,      // start_s, present only if inference emitted it
    t1   : number?       // end_s,   present only if inference emitted it
} , ... ]

P.sections     : [ "A","A","B", ... ]  // per-bar section LETTER (may be "")
P.sectionChips : [ {label, start_s}, ... ]  // A/B/C navigator row
P.nBars, P.bpb, P.home, P.motifs        // grid geometry + auto-motifs
```

Two facts drive the whole schema:

- **Chord identity is positional.** No chord carries a stable id. Its array
  index `i` is implicit (used only to mint the DOM id `chord-{i}`); its
  *logical* address is the pair `(bar, beat)`, which `render_interactive`
  guarantees is unique per song (chords are bucketed `by_bar` then sorted by
  `beat`). We key corrections on `(bar, beat)`, not `i` — see §5.1.
- **A "section" is not an object, it's a run of equal letters** in
  `P.sections`. There is no section id to merge. The only durable handle on a
  span is a **bar range `[start, end]`** — which is exactly what the manual
  motif tagger (`motifState.motifs[*].bars : [[s,e], ...]`) already uses. So a
  merge group is structurally a motif entry with ≥2 bar ranges (§3).

---

## 2. Sidecar file: location & top-level shape

One file per chart, named by the chart's HTML filename (the same key the
server already uses for `_yt_video_ids` / `_yt_audio_meta`):

```
docs/plots/annotations/<chart-filename>.html.json
   e.g. docs/plots/annotations/inferred_autumn_leaves_remastered.html.json
```

(Living under `PLOTS_DIR/annotations/` keeps it next to the charts and the
existing `.yt_*` registries; per-song files rather than one aggregate dict —
see §5.4 for why this diverges slightly from the existing pattern and why
that's fine.)

```jsonc
{
  "schema": 1,                        // bump on any breaking shape change
  "chart": "inferred_autumn_leaves_remastered.html",
  "annotator": "Louis Vincent",       // plain name string, from localStorage
  "modified": "2026-07-12T14:03:22Z", // ISO-8601 UTC, last write
  "chords": [ /* §3 chord corrections */ ],
  "merges": [ /* §4 merge groups     */ ]
}
```

| field | rationale (tied to existing code) |
|---|---|
| `schema` | Version guard. `_load_*` in the server swallows malformed JSON and returns `{}`; a version int lets the client refuse-and-warn instead of silently misapplying a stale shape. |
| `chart` | The chart HTML filename — the exact key `serve_chart(filename)` receives and that `_yt_video_ids`/`_yt_audio_meta` are keyed by. Redundant with the filename but makes the file self-describing if copied. |
| `annotator` | Plain string, entered once, held in `localStorage`, replayed into the POST body. One annotator per song (decided) → a single top-level field, no per-edit identity. |
| `modified` | Single last-write timestamp. The spec explicitly rejects a revision history ("current best label per bar plus merge groups, not a revision history"), so one timestamp for the file suffices. |
| `chords` | List of corrections (§3). A **list**, not a map, because the natural key is a compound `(bar, beat)` — a JSON object can't key on a pair without stringifying it, and a list of `{bar, beat, ...}` records mirrors `P.chords` itself. |
| `merges` | List of merge groups (§4), each a set of equivalent bar ranges. |

---

## 3. Chord corrections

```jsonc
"chords": [
  {
    "bar": 12,                // matches P.chords[*].bar
    "beat": 0,                // matches P.chords[*].beat  → (bar,beat) is the key
    "root": 9,                // NEW root pitch-class 0..11
    "bass": -1,               // NEW bass pc, or -1 for none (mirror P.chords.bass)
    "q": "-7",                // NEW iReal quality tail, same vocabulary as lv.*.q
    "old": { "root": 9, "bass": -1, "q": "-" },   // guess at correction time
    "ts": "2026-07-12T14:01:55Z"                  // per-edit timestamp
  }
]
```

| field | rationale |
|---|---|
| `bar`, `beat` | The logical key, matching `P.chords` addressing. Applying a correction at render time is "find the chord entry whose `(bar,beat)` equals this" — robust across re-renders in a way the raw array index `i` is not (§5.1). |
| `root`, `bass`, `q` | The corrected chord, stored in exactly the three fields the display reads: `root` (pc), `bass` (pc or −1), and `q` (the iReal quality tail already produced by `parse_token` and consumed by `typesetQuality`/`jazzify`). The rotor UI (§13.1) emits root → family → 7th → extensions, which collapse cleanly into a single `q` tail plus `root`/`bass`. No new vocabulary. |
| `old` | The model's guess at the moment of correction (its displayed `{root,bass,q}`). Kept as training signal — §13's whole thesis is "turn every correction into training signal" — and lets the UI show "was X → now Y" without re-running inference. Not load-bearing for rendering; drop-safe. |
| `ts` | Per-edit timestamp. The file-level `modified` covers "when was this touched"; `ts` gives the ordering/recency of individual edits cheaply, still within the "no full revision history" constraint (one value, not a log). |

**Why one flat `q` instead of the three-level `lv` structure:** a human
correction is a single certain answer, not a family/seventh/exact confidence
ladder. Collapsing to one `q` is deliberate and forces a decision about the
level slider — see §5.2.

---

## 4. Merge groups

A merge group asserts "these bar ranges are the same underlying material."
Structurally identical to a `motifState` entry with multiple `bars` ranges,
so it reuses that shape rather than inventing one:

```jsonc
"merges": [
  {
    "id": 1,                      // stable within this file (mirror motifState.nextId)
    "spans": [ [0, 7], [16, 23] ],// ≥2 inclusive bar ranges [start,end], as motifState.bars
    "label": "A",                 // optional human label (section letter or free text)
    "ts": "2026-07-12T14:02:40Z"
  }
]
```

| field | rationale |
|---|---|
| `spans` | List of inclusive `[startBar, endBar]` ranges — byte-for-byte the shape of `motifState.motifs[*].bars` (`[[s,e],...]`), which `renderMotifBrackets` already iterates and `saveMotif` already overlap-checks. The merge UI is "a lightweight extension of the motif-bracket UI"; using its exact range shape means the client can seed `motifState` directly from `merges` and reuse bracket rendering. Bar ranges (not section ids) because a section has no id — §1. |
| `id` | Stable per-file identifier, mirroring `motifState.nextId`. Lets the re-score step and the UI refer to a group without positional fragility. |
| `label` | Optional display name — typically the shared section letter (from `P.sections`) or annotator free text, matching `motifState.motifs[*].name`. Optional because equivalence is carried by `spans`, not the label. |
| `ts` | Per-merge timestamp, same rationale as chord `ts`. |

`spans` is the *only* thing the sidecar records about a merge — deliberately.
The local re-score (pooled chroma) is a runtime action; its **input** is
"which spans are equivalent," and that is fully captured here. Where the chroma
comes from is the separate in-flight investigation and is out of scope, as
instructed.

---

## 5. Friction with the existing model — named, not papered over

### 5.1 Chord index vs `(bar, beat)` — positional identity is fragile
`P.chords` has no stable id; the DOM uses raw array index `i` (`chord-{i}`).
Keying corrections on `i` would be simplest to apply (direct `P.chords[i]`)
but breaks the instant inference is re-run and emits a different chord count or
ordering — a merge's local re-score can itself change how many chord events a
span has. `(bar, beat)` is the more durable key and matches how a human thinks
("bar 12"), but it is **not bulletproof**: if a re-score changes the *beat
offset* of an event, the key silently misses. Mitigation to flag for
implementation: apply corrections by nearest-`(bar,beat)` within a bar, and
surface an "unmatched correction" if none lands — do not silently drop.

### 5.2 The level slider has no defined behavior for a corrected chord
Display picks family/seventh/exact per chord via `pickLevel(d, mode, th)`
against the "Sure ≥" threshold slider. A human correction is a single certain
chord with no confidence ladder. **Unresolved design question the schema
forces into the open:** when a corrected chord is rendered, should it (a) pin
to a synthetic "exact/confidence 1.0" and ignore the slider entirely, or (b)
populate all three `lv` levels with the same corrected `q`? The schema stores
only one `q` (option a is cleaner and matches "annotator disposes"), but the
overlay code must special-case corrected chords so the slider can't demote a
human answer back to a vaguer family label. This is a real integration seam,
not a stored-data problem — flagged so it isn't discovered at render time.

### 5.3 Multiple chords per bar — `beat` must be trusted, and it may be float
`by_bar` allows several chords in one bar, disambiguated only by `beat`. The
key therefore *requires* `beat` to be stable and comparable. `beat` is
`c.get("beat", 0)` upstream and may be a float; equality on floats is brittle.
Store `beat` exactly as it appears in `P.chords` and compare with a small
tolerance, or (better, if upstream allows) quantize `beat` to a known grid
before it enters the payload. Named so the key choice isn't assumed exact.

### 5.4 Per-song file vs the existing aggregate-dict pattern
The existing registries are **single files holding a dict keyed by filename**
(`.yt_video_ids.json` = `{filename: vid}`), not per-song files. The decided
design is per-song sidecars. This is a minor, intentional divergence: per-song
files avoid one growing global dict, isolate corruption to one song, and let a
sidecar travel with its chart. The server-side *mechanics* still mirror the
pattern exactly — a `_load_annotations(filename)` reader that tolerates
missing/malformed JSON (returns `{}` / `None` like `_load_yt_video_ids`) and a
`_remember_annotation(filename, doc)` writer that writes on every update. What
does **not** carry over is the "load everything once at module import into a
module-level dict" step — with per-song files you read the one file lazily in
`serve_chart` / the POST handler. Flag: don't blindly copy the module-level
`_yt_video_ids = _load...()` line; there's no single file to preload.

### 5.5 Merge overlap semantics differ from motif overlap
`saveMotif` *drops* ranges that overlap any existing motif (motifs partition
the bar line). Merge groups have different semantics: two spans in one group
must be allowed to be non-adjacent, and a bar could plausibly belong to both a
motif tag and a merge group. Reusing `motifState`'s `bars` *shape* is right;
reusing its overlap-rejection logic is **not** — the merge feature needs its
own containment rules. Flagged because "extend the motif UI" invites copying
`saveMotif` wholesale.

---

## 6. How it gets applied (recommended, consistent with existing conventions)

Follow the `HARM_AUDIO_URL` / `YT_VIDEO_ID` injection pattern in
`serve_chart` verbatim.

**Read path (server → page).** In `serve_chart(filename)`, after the existing
audio/video injections, read the sidecar and inject it as one more JS global:

```
window.HARM_ANNOTATIONS = { annotator, modified, chords:[...], merges:[...] };
```

via the same `content.replace("</head>", "<script>…</script></head>", 1)`
mechanism already used three times in that function. One global, JSON-encoded
with `json.dumps`, exactly like `HARM_AUDIO_URL`. The chart's own JS then, at
init: (a) overlays `HARM_ANNOTATIONS.chords` onto `P.chords` by `(bar,beat)`
before first render (respecting §5.2), and (b) seeds `motifState.motifs` from
`HARM_ANNOTATIONS.merges` so brackets render with no extra code path.

**Write path (page → server).** A single POST endpoint, e.g.
`POST /api/annotations/<filename>`, whose handler validates the body and calls
`_remember_annotation(filename, doc)` — the write-on-update twin of
`_remember_video_id`, writing `PLOTS_DIR/annotations/<filename>.json`. The
client posts the whole current annotation doc (name from `localStorage` +
current `chords` + `merges`) on each change; last-write-wins, no merge/conflict
logic (single annotator, decided). Keep the endpoint dumb: it stamps
`modified` server-side and persists — no re-inference in the request path
(local re-score is a separate client-triggered flow).

Why a global rather than a separate fetch: the three existing injections prove
the convention is "compute per-request server-side, hand the page a
`window.*` constant." A separate `fetch('/api/annotations/…')` on load would
add a round-trip and a race against first render for zero benefit, since
`serve_chart` already has the filename and disk access in hand.
