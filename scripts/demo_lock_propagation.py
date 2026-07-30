"""scripts/demo_lock_propagation.py — task 5 of the lock-propagation
final-integration brief: an inspectable end-to-end artifact.

Decodes a real app chart via the exact production path, picks a span where
the model's own acoustic evidence most strongly disagrees with what's
displayed (the realistic "my ear says this is wrong" case), simulates the
user locking it to that alternative via a REAL HTTP-shaped call through the
Flask test client to /api/context_rescore/<filename>, and writes
docs/lock_propagation_demo.md showing the chart before/after with changed
spans marked and the context prior's own top-3 for each changed span.

No server is started or restarted -- Flask test client only (per the brief's
rules: never restart/kill any running server).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from harmonia.eval.accuracy_score import SHIPPED_CONFIG, _decode_to_wav  # noqa: E402
from harmonia.models import span_rescore  # noqa: E402
from harmonia.models.chord_context_prior import (  # noqa: E402
    chord_name,
    parse_harte_lite,
    top_candidates,
)

WORK_DIR = REPO / "data" / "cache" / "tune_lock_propagation"
WAV_DIR = WORK_DIR / "wav"


def decode_chart(audio_path: Path):
    from harmonia.models.chord_pipeline_v1 import infer_chords_v1
    wav = _decode_to_wav(audio_path, WAV_DIR)
    chart = infer_chords_v1(wav, cache_dir=REPO / "data/cache", **SHIPPED_CONFIG)
    chords = [{"start_s": c["start_s"], "end_s": c["end_s"], "label": c["label"]}
             for c in chart.chords if c["end_s"] > c["start_s"]]
    return chords, chart.tempo_bpm, wav


def bar_of(cum_beats: float, beats_per_bar: int = 4) -> str:
    bar = int(cum_beats // beats_per_bar) + 1
    beat = int(cum_beats % beats_per_bar) + 1
    return f"bar {bar}.{beat}"


def pick_lock_span(chords, spans, acoustic_logp):
    """Span where the model's OWN acoustic evidence disagrees most strongly
    with what's displayed -- the "my ear says this chord is wrong" case,
    absent independent GT for this chart."""
    best = None
    for i, c in enumerate(chords):
        displayed = parse_harte_lite(c["label"])
        if displayed is None:
            continue
        d_idx = span_rescore.idx_of(*displayed)
        row = acoustic_logp[i]
        order = row.argsort()[::-1]
        top_idx = int(order[0])
        if top_idx == d_idx:
            top_idx = int(order[1])
        gap = float(row[top_idx] - row[d_idx])
        if best is None or gap > best[0]:
            best = (gap, i, displayed, span_rescore.token_of(top_idx))
    return best  # (gap, index, displayed_token, alt_token)


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--chart", default="inferred_maroon_5_this_love.html")
    ap.add_argument("--audio", default=str(REPO / "docs/audio/maroon_5_this_love.m4a"))
    ap.add_argument("--table", default="pooled")
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--delta", type=float, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--margin-gate", type=float, default=0.0)
    ap.add_argument("--out", default=str(REPO / "docs/lock_propagation_demo.md"))
    args = ap.parse_args()

    print(f"Decoding {args.audio} via production path...")
    raw_chords, tempo_bpm, wav = decode_chart(Path(args.audio))
    # Drop any span whose label parse_harte_lite can't map to (root, qual5) --
    # e.g. a "N"/no-chord segment -- keeping chords/spans/displayed in lockstep,
    # same convention as scripts/tune_lock_propagation.py's precompute_song.
    chords = [c for c in raw_chords if parse_harte_lite(c["label"]) is not None]
    n_dropped = len(raw_chords) - len(chords)
    spans = [(c["start_s"], c["end_s"]) for c in chords]
    print(f"{len(chords)} spans ({n_dropped} unparseable dropped), tempo={tempo_bpm:.1f} bpm")

    result = span_rescore.compute_acoustic_logp(wav, spans, fallback_audio=wav)
    print(f"acoustic backend: {result['backend']} cache_hit={result['cache_hit']}")

    gap, lock_i, displayed_tok, alt_tok = pick_lock_span(chords, spans, result["logp"])
    print(f"Lock candidate: span {lock_i} ({chords[lock_i]['label']}) -- acoustic "
         f"evidence favours {chord_name(*alt_tok)} by {gap:.2f} nats over the "
         f"displayed {chord_name(*displayed_tok)}")

    # cumulative beat position for readable bar labels (approx: 4/4, cumulative
    # nominal beat count from t=0 using the CHART's own beat grid isn't carried
    # in this raw chords list, so we fall back to tempo_bpm-derived beats --
    # documented as approximate in the output doc).
    cum_beats = []
    acc = 0.0
    for c in chords:
        cum_beats.append(acc)
        acc += (c["end_s"] - c["start_s"]) * tempo_bpm / 60.0

    # ── build the request payload exactly as the client would ──────────────
    def q5_of(tok):
        return {"root": tok[0], "q5": tok[1]}

    req_chords = []
    for i, c in enumerate(chords):
        tok = parse_harte_lite(c["label"])
        req_chords.append({"t0": c["start_s"], "t1": c["end_s"],
                           "root": tok[0], "q5": tok[1]})
    lock_chord = req_chords[lock_i].copy()
    lock_chord["root"], lock_chord["q5"] = alt_tok

    import harmonia_server
    app = harmonia_server.create_app()
    app.testing = True
    client = app.test_client()

    payload = {"chords": req_chords, "confirms": [lock_chord],
              "lam": args.lam, "delta": args.delta, "K": args.K, "table": args.table,
              "margin_gate": args.margin_gate}
    resp = client.post(f"/api/context_rescore/{args.chart}", json=payload)
    print(f"POST /api/context_rescore/{args.chart} -> {resp.status_code}")
    body = resp.get_json()
    if resp.status_code != 200:
        print("ERROR:", body)
        return
    print(f"table={body.get('table')} acoustic_backend={body.get('acoustic_backend')} "
         f"n_changed={body['n_changed']}")

    out_chords = body["chords"]
    changed_idxs = sorted(d["index"] for d in body["diff"])
    print("changed spans:", changed_idxs)

    # ── write the markdown artifact ─────────────────────────────────────────
    lo = max(0, lock_i - 4)
    hi = min(len(chords), lock_i + 5)

    def fmt_row(i, before_label, after_label, changed, locked):
        marker = " <- LOCKED" if locked else (" <- CHANGED" if changed else "")
        return (f"| {i} | {bar_of(cum_beats[i])} | {chords[i]['start_s']:.1f}s | "
               f"{before_label} | {after_label}{marker} |")

    lines = []
    lines.append("# Lock propagation demo: end-to-end artifact\n")
    lines.append(f"Chart: `{args.chart}`  |  operating point: lam={args.lam}, "
                f"delta={args.delta}, K={args.K}, margin_gate={args.margin_gate}, "
                f"table={args.table}\n")
    lines.append(f"Acoustic backend: `{result['backend']}` (cache_hit={result['cache_hit']})\n")
    lines.append(f"\n## The lock\n")
    lines.append(f"Span {lock_i} ({bar_of(cum_beats[lock_i])}, "
                f"{chords[lock_i]['start_s']:.1f}s): displayed **"
                f"{chord_name(*displayed_tok)}**, but the model's OWN acoustic "
                f"evidence favours **{chord_name(*alt_tok)}** by {gap:.2f} nats "
                f"(no independent ground truth for this app chart -- this is the "
                f"'my ear says this chord is wrong' scenario). Simulated user "
                f"action: lock this span to {chord_name(*alt_tok)}.\n")
    lines.append("\n## Chart excerpt BEFORE the lock (+/-4 spans)\n")
    lines.append("| span | bar | time | chord |")
    lines.append("|---|---|---|---|")
    for i in range(lo, hi):
        marker = " <- ABOUT TO LOCK" if i == lock_i else ""
        lines.append(f"| {i} | {bar_of(cum_beats[i])} | {chords[i]['start_s']:.1f}s | "
                    f"{chords[i]['label']}{marker} |")

    lines.append("\n## Chart excerpt AFTER the lock (+/-4 spans)\n")
    lines.append("| span | bar | time | before | after |")
    lines.append("|---|---|---|---|---|")
    for i in range(lo, hi):
        before = chords[i]["label"]
        after = out_chords[i]["label"]
        changed = out_chords[i]["changed"]
        locked = (i == lock_i)
        lines.append(fmt_row(i, before, after, changed, locked))

    lines.append(f"\n## All changed spans ({len(changed_idxs)} total) with the "
                f"context prior's top-3\n")
    lines.append("| span | bar | before | after | prior top-3 (given final neighbours) |")
    lines.append("|---|---|---|---|---|")
    from harmonia.models.chord_context_prior import load_context_prior
    ctx_model = load_context_prior(corpus=args.table)
    chosen_toks = [(parse_harte_lite(oc["label"])) for oc in out_chords]
    for i in changed_idxs:
        prev_tok = chosen_toks[i - 1] if i > 0 else None
        next_tok = chosen_toks[i + 1] if i < len(chosen_toks) - 1 else None
        top3 = top_candidates(prev_tok, next_tok, k=3, model=ctx_model)
        top3_str = ", ".join(f"{t['name']} ({t['prob']:.3f})" for t in top3)
        lines.append(f"| {i} | {bar_of(cum_beats[i])} | {chords[i]['label']} | "
                    f"{out_chords[i]['label']} | {top3_str} |")

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
