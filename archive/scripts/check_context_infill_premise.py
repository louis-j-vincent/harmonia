#!/usr/bin/env python3
"""Phase 1 evals for the chord-context prior (docs/design_chord_context_prior.md).

Eval 1 — masked infilling (cheap premise check, no audio): on held-out songs,
hide every interior chord and score all 60 (root, quality) candidates from
its (prev, next) neighbours; report top-1/top-3 recall against three
same-candidate-space baselines.

Eval 2 — theory sanity panel: ranked top-8 candidates for 6 hand-picked
harmonic contexts (ii-V-I, minor ii-V-i, etc.), human-readable.

Writes docs/context_prior_phase1_results.md with both plus corpus stats.

Usage: .venv/bin/python scripts/check_context_infill_premise.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from harmonia.models.chord_context_prior import (  # noqa: E402
    cand_index,
    load_all_corpus_sequences,
    load_context_prior,
    score_candidates,
    top_candidates,
)
from harmonia.models.progression_encoder import QUAL5, QUAL5_IDX  # noqa: E402

MAJ, MIN, DOM, HDIM, DIM = (QUAL5_IDX[q] for q in ("maj", "min", "dom", "hdim", "dim"))


def family_of(song_id: str) -> str:
    return "jazz" if song_id.startswith("accomp:") else "pop"


# ── Eval 1: masked infilling ─────────────────────────────────────────────────

def run_infill_eval(model: dict, songs: dict) -> dict:
    heldout_ids = [sid for sid in model["heldout_ids"] if sid in songs]
    print(f"Eval 1: {len(heldout_ids)} held-out songs (of {len(model['heldout_ids'])} in split)")

    # counters[metric][group] = [hits, total]
    counters: dict[str, dict[str, list[int]]] = {}

    def bump(metric: str, group: str, hit: bool):
        d = counters.setdefault(metric, {})
        rec = d.setdefault(group, [0, 0])
        rec[1] += 1
        if hit:
            rec[0] += 1

    n_positions = 0
    for sid in heldout_ids:
        fam = family_of(sid)
        for seg in songs[sid]:
            for i in range(1, len(seg) - 1):
                prev, true_tok, nxt = seg[i - 1], seg[i], seg[i + 1]
                true_idx = cand_index(*true_tok)
                true_q = QUAL5[true_tok[1]]
                n_positions += 1

                # --- model: full context ---
                probs = score_candidates(prev, nxt, model=model)
                order = np.argsort(probs)[::-1]
                top1_idx = int(order[0])
                top3_idx = set(int(x) for x in order[:3])

                exact1 = top1_idx == true_idx
                exact3 = true_idx in top3_idx
                root1 = (top1_idx // 5) == true_tok[0]
                root3 = any((idx // 5) == true_tok[0] for idx in top3_idx)
                qual1 = (top1_idx % 5) == true_tok[1]
                qual3 = any((idx % 5) == true_tok[1] for idx in top3_idx)

                bump("model_exact_top1", "ALL", exact1)
                bump("model_exact_top3", "ALL", exact3)
                bump("model_root_top1", "ALL", root1)
                bump("model_root_top3", "ALL", root3)
                bump("model_qual_top1", "ALL", qual1)
                bump("model_qual_top3", "ALL", qual3)

                bump("model_exact_top1", f"family:{fam}", exact1)
                bump("model_exact_top3", f"family:{fam}", exact3)
                bump("model_exact_top1", f"qual:{true_q}", exact1)
                bump("model_exact_top3", f"qual:{true_q}", exact3)

                # --- baseline (a): repeat prev ---
                bump("baseline_repeat_prev_top1", "ALL", prev == true_tok)

                # --- baseline (b): repeat next ---
                bump("baseline_repeat_next_top1", "ALL", nxt == true_tok)

                # --- baseline (c): context-free candidate prior (prev-only
                #     directed bigram alone, i.e. score_candidates with no
                #     "next" -- same candidate space, same smoothing machinery,
                #     just missing the second neighbour) ---
                probs_c = score_candidates(prev, None, model=model)
                order_c = np.argsort(probs_c)[::-1]
                bump("baseline_prevbigram_top1", "ALL", int(order_c[0]) == true_idx)
                bump("baseline_prevbigram_top3", "ALL", true_idx in set(int(x) for x in order_c[:3]))

    print(f"  {n_positions} masked interior positions scored")
    return counters


def _recall(counters, metric, group="ALL"):
    rec = counters.get(metric, {}).get(group)
    if not rec or rec[1] == 0:
        return None, 0
    return rec[0] / rec[1], rec[1]


def render_eval1_markdown(counters: dict) -> str:
    lines = []
    lines.append("## Eval 1 — masked infilling recall (held-out songs)\n")
    lines.append("Model vs same-candidate-space baselines, all on the identical masked positions.\n")
    lines.append("**Caveat on the two \"repeat a neighbour\" baselines**: every segment has had "
                 "consecutive-identical tokens collapsed (change-to-change transitions only, per "
                 "the design doc), so by construction `prev != true_tok != next` at every masked "
                 "position. \"Repeat prev\"/\"repeat next\" therefore score EXACTLY 0% — not a "
                 "real result, a structural consequence of the dedup, kept only because the brief "
                 "named them. The meaningful baseline is \"prev-bigram only\" (uses real context, "
                 "just one-sided).\n")
    lines.append("| method | top-1 recall | top-3 recall | n |")
    lines.append("|---|---|---|---|")

    def row(label, m1, m3):
        r1, n1 = _recall(counters, m1)
        r3, n3 = _recall(counters, m3) if m3 else (None, 0)
        r1s = f"{r1:.1%}" if r1 is not None else "-"
        r3s = f"{r3:.1%}" if r3 is not None else "-"
        lines.append(f"| {label} | {r1s} | {r3s} | {n1} |")

    row("context prior P(c\\|prev,next)", "model_exact_top1", "model_exact_top3")
    row("repeat prev chord", "baseline_repeat_prev_top1", None)
    row("repeat next chord", "baseline_repeat_next_top1", None)
    row("prev-bigram only (no next)", "baseline_prevbigram_top1", "baseline_prevbigram_top3")

    lines.append("\n### Root-correct / quality-correct, independently (model, full context)\n")
    lines.append("| axis | top-1 recall | top-3 recall | n |")
    lines.append("|---|---|---|---|")
    row("root correct (any quality)", "model_root_top1", "model_root_top3")
    row("quality correct (any root)", "model_qual_top1", "model_qual_top3")

    lines.append("\n### Exact-match recall by TRUE candidate quality (model)\n")
    lines.append("| quality | top-1 recall | top-3 recall | n |")
    lines.append("|---|---|---|---|")
    for q in QUAL5:
        r1, n1 = _recall(counters, "model_exact_top1", f"qual:{q}")
        r3, n3 = _recall(counters, "model_exact_top3", f"qual:{q}")
        r1s = f"{r1:.1%}" if r1 is not None else "-"
        r3s = f"{r3:.1%}" if r3 is not None else "-"
        lines.append(f"| {q} | {r1s} | {r3s} | {n1} |")

    lines.append("\n### Exact-match recall by corpus family (model)\n")
    lines.append("| family | top-1 recall | top-3 recall | n |")
    lines.append("|---|---|---|---|")
    for fam in ("jazz", "pop"):
        r1, n1 = _recall(counters, "model_exact_top1", f"family:{fam}")
        r3, n3 = _recall(counters, "model_exact_top3", f"family:{fam}")
        r1s = f"{r1:.1%}" if r1 is not None else "-"
        r3s = f"{r3:.1%}" if r3 is not None else "-"
        lines.append(f"| {fam} | {r1s} | {r3s} | {n1} |")

    return "\n".join(lines) + "\n"


# ── Eval 2: theory sanity panel ─────────────────────────────────────────────

PANEL = [
    ("Dm7 -> ? -> Cmaj  (ii-V-I; expect G7 dominant high, Db7 tritone sub in top-8)",
     (2, MIN), (0, MAJ)),
    ("Dm7b5 -> ? -> Cm  (minor ii-V-i; expect G7)",
     (2, HDIM), (0, MIN)),
    ("Cmaj -> ? -> Dm7  (expect diatonic passing / A7 = V-of-ii)",
     (0, MAJ), (2, MIN)),
    ("Cmaj -> ? -> G7  (expect Dm7 ii, or D7 V-of-V)",
     (0, MAJ), (7, DOM)),
    ("Fmaj -> ? -> Cmaj  (expect G7, Bb7 backdoor, Fm iv, Ab/G dim passing)",
     (5, MAJ), (0, MAJ)),
    ("Am7 -> ? -> Dm7  (expect A7 or E7-family secondary dominants)",
     (9, MIN), (2, MIN)),
]


def render_eval2_markdown(model: dict) -> str:
    lines = ["## Eval 2 — theory sanity panel (ranked top-8, all contexts in C)\n"]
    for desc, prev, nxt in PANEL:
        lines.append(f"### {desc}\n")
        lines.append("| rank | chord | quality | P |")
        lines.append("|---|---|---|---|")
        for rank, c in enumerate(top_candidates(prev, nxt, k=8, model=model), start=1):
            lines.append(f"| {rank} | {c['name']} | {c['q5']} | {c['prob']:.3f} |")
        lines.append("")
    lines.append(
        "**Observations (reported as measured, not cherry-picked)**: ii-V-I, minor "
        "iiø-V-i, and V-of-V (Cmaj->?->G7 surfaces Dm7 #1, D7 #7) all match the stated "
        "expectation. Two clear misses: (1) the Db7 tritone sub does **not** appear in "
        "the Dm7->?->Cmaj top-8 at all — tritone subs are a real but rare voicing choice "
        "in the training corpora, so the model has little to learn it from; (2) in "
        "Fmaj->?->Cmaj the model's top-2 candidates are **Gmaj/Bbmaj (plain triads)**, "
        "not the expected **G7/Bb7 (dominant)** — the pooled corpus is pop-heavy, and "
        "pop's IV-V-I / IV-bVII-I idioms are overwhelmingly plain triads, diluting the "
        "jazz-specific dominant reading (the design doc's own genre-pooling caveat, "
        "confirmed here). Am7->?->Dm7's top candidate is D7 (same root as the *next* "
        "chord, quality shifting dom->min) rather than the expected A7/E7 secondary "
        "dominant — worth a follow-up look at whether same-root quality-shift "
        "transitions are systematically over-weighted by the root-evidence term.\n"
    )
    return "\n".join(lines) + "\n"


# ── corpus stats markdown ───────────────────────────────────────────────────

def render_corpus_stats_markdown(model: dict) -> str:
    stats = model["stats"]
    lines = ["## Corpus stats\n"]
    lines.append(f"Train songs: **{len(model['train_ids'])}**  |  "
                 f"Held-out songs: **{len(model['heldout_ids'])}**  "
                 f"(deterministic 1-in-10 split by stable hash of song id)\n")
    lines.append("| source | records seen | songs kept | songs dropped (short) | "
                 "songs dropped (other) | tokens kept |")
    lines.append("|---|---|---|---|---|---|")
    for src in ("accomp_db", "pop909", "choco"):
        st = stats.get(src, {})
        if "error" in st:
            lines.append(f"| {src} | MISSING: {st['error']} | - | - | - | - |")
            continue
        other = st.get("n_skipped_unmapped_token_songs", 0) + st.get("n_unreadable", 0) + st.get("n_no_chord_ns", 0)
        lines.append(
            f"| {src} | {st.get('n_records')} | {st.get('n_kept_songs')} | "
            f"{st.get('n_dropped_short')} | {other} | {st.get('n_tokens')} |"
        )
    lines.append("")

    lines.append("### Dropped-label counters (top 15 per source)\n")
    for src in ("accomp_db", "pop909", "choco"):
        st = stats.get(src, {})
        dropped = st.get("dropped_labels", {})
        if not dropped:
            continue
        top = Counter(dropped).most_common(15)
        total = sum(dropped.values())
        lines.append(f"**{src}** (total dropped label instances shown here: top-15 of "
                     f"the recorded set; grand total dropped = {total}):\n")
        lines.append("| label | count |")
        lines.append("|---|---|")
        for lab, n in top:
            lines.append(f"| `{lab}` | {n} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    model = load_context_prior()
    songs, load_stats = load_all_corpus_sequences()
    assert songs is not None

    counters = run_infill_eval(model, songs)
    eval1_md = render_eval1_markdown(counters)
    print("\n" + eval1_md)

    eval2_md = render_eval2_markdown(model)
    print(eval2_md)

    corpus_md = render_corpus_stats_markdown(model)

    out = REPO / "docs" / "context_prior_phase1_results.md"
    out.write_text(
        "# Chord context prior — Phase 1 results\n\n"
        "Branch `feat/chord-context-prior`. See `docs/design_chord_context_prior.md` "
        "for the design and `harmonia/models/chord_context_prior.py` for the model.\n\n"
        + corpus_md + "\n" + eval1_md + "\n" + eval2_md
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
