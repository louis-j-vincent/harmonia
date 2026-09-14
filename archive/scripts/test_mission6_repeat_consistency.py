"""Cheap premise check for Mission 6 (CLAUDE.md rule 2): does Signal 1
(repeat_consistency) separate an *aligned* chart from a *slipped* one on
REAL audio, before we build the 20-song eval harness?

Signal 1 (docs/mission_6_elastic_matching_design.md):
    from a candidate alignment, collect the INFERRED chord content mapped under
    each iReal section instance, fingerprint each instance as an L2-normalized
    mean [root one-hot | quality one-hot] vector, then
        within = mean cosine over same-label section-instance pairs
        cross  = mean cosine over diff-label section-instance pairs
        repeat_consistency = within - cross            # want > 0, ideally > 0.10
A slipped repeat (failure #3) destroys the within-label agreement.

We test on real inferred content two ways per song:
  (a) CLEAN   — the aligner's own candidate alignment.
  (b) SLIPPED — inject a 1-section slip (rotate each section instance's inferred
                content onto its neighbour, labels unchanged) and recompute.
If clean_Δ >> slipped_Δ on real audio, the premise holds.

Single-use script. Run: PYTHONPATH=. python scripts/test_mission6_repeat_consistency.py
"""
from __future__ import annotations
import io, contextlib, json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

with contextlib.redirect_stdout(io.StringIO()):
    from harmonia.data.ireal_corpus import load_playlist, tune_to_mma
    from harmonia.irealb_aligner import align_irealb_to_inferred, dedup_inferred
    from harmonia.theory.local_key import parse_token

SCRATCH = Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-Code-harmonia"
               "/d7ec820d-b659-4940-aa06-eebbf6fd5927/scratchpad")


# ── extract p_chords from a saved inferred_*.html (embeds `const P = {...}`) ──
def _extract_brace_object(text: str, start_key: str) -> dict:
    i = text.find(start_key)
    if i < 0:
        raise ValueError(f"{start_key} not found")
    j = text.find("{", i)
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[j:k + 1])
    raise ValueError("unbalanced braces")


def pchords_from_html(path: Path):
    html = path.read_text(encoding="utf-8")
    P = _extract_brace_object(html, "const P = {")
    return P["chords"], P.get("bpb", 4)


def pchords_from_json(path: Path):
    d = json.loads(path.read_text())
    return d["chords"], 4


# ── fingerprint of a list of inferred (root, quality) segments ───────────────
_QUAL_IDX: dict[str, int] = {}
def _qidx(q: str) -> int:
    return _QUAL_IDX.setdefault(q, len(_QUAL_IDX))


def fingerprint(segments, n_pitches=12, n_qual=24) -> np.ndarray:
    """L2-normalized mean [root one-hot(12) | quality one-hot] over inferred
    segments (root absolute mod 12 — fine, all sections share one song key)."""
    if not segments:
        return np.zeros(n_pitches + n_qual)
    feats = []
    for root, q in segments:
        v = np.zeros(n_pitches + n_qual)
        if root is not None and root >= 0:
            v[root % n_pitches] = 1.0
        qi = _qidx(q)
        if qi < n_qual:
            v[n_pitches + qi] = 1.0
        n = np.linalg.norm(v)
        feats.append(v / n if n > 1e-9 else v)
    fp = np.mean(feats, axis=0)
    n = np.linalg.norm(fp)
    return fp / n if n > 1e-9 else fp


# ── split the aligned result into contiguous section instances, gather the
#    inferred segments that fall under each instance's time span ──────────────
def section_instances(result_chords, inferred):
    """-> list of (label, [(root,q),...]) — one entry per contiguous section run."""
    # inferred: list of InferredChord (dedup_inferred) w/ pc, quality, t0, t1
    inf_mid = [(0.5 * (c.t0 + c.t1), c.pc, c.quality) for c in inferred]
    instances = []
    cur_label = None
    cur_t0 = None
    cur_t1 = None

    def flush(lbl, t0, t1):
        if lbl is None:
            return
        segs = [(pc, q) for (mid, pc, q) in inf_mid if t0 <= mid <= t1]
        instances.append((lbl, segs))

    for ch in result_chords:
        lbl = ch.get("section", "?")
        t0 = float(ch.get("t0") or 0.0)
        t1 = float(ch.get("t1") or t0)
        if lbl != cur_label:
            flush(cur_label, cur_t0, cur_t1)
            cur_label, cur_t0, cur_t1 = lbl, t0, t1
        else:
            cur_t1 = max(cur_t1, t1)
    flush(cur_label, cur_t0, cur_t1)
    return instances


def repeat_consistency(labels, fps):
    S = np.array([[float(a @ b) for b in fps] for a in fps])
    within, cross = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            (within if labels[i] == labels[j] else cross).append(S[i, j])
    w = float(np.mean(within)) if within else float("nan")
    c = float(np.mean(cross)) if cross else float("nan")
    return w, c, (w - c if within and cross else float("nan"))


def section_family_fracs(result_chords):
    by = {}
    run = 0
    last = None
    for ch in result_chords:
        lbl = ch.get("section", "?")
        if lbl != last:
            run += 1
            last = lbl
        key = f"{lbl}#{run}"
        m = ch.get("match", "")
        by.setdefault((lbl, key), []).append(1 if m in ("exact", "family") else 0)
    return {k[1]: float(np.mean(v)) for k, v in by.items() if v}


# ── pilots ───────────────────────────────────────────────────────────────────
PILOTS = [
    dict(name="Autumn Leaves", tune="autumn leaves", playlist="jazz1460",
         src=("html", ROOT / "docs/plots/inferred_autumn_leaves.html"),
         expected="aligned", note="known-good (mission framing); real inference"),
    dict(name="Let It Be", tune="let it be", playlist="pop400",
         src=("html", ROOT / "docs/plots/inferred_the_beatles_the_beatles_let_it_be_official_music_video_remas.html"),
         expected="slipped", note="documented #22 cycle-shift slip; real inference"),
    dict(name="Ghost of a Chance", tune="a ghost of a chance", playlist="jazz1460",
         src=("json", SCRATCH / "ghost_pchords.json"),
         expected="aligned", note="issue-#20 pilot, fresh inference; phase1b best-aligned"),
]


def load_tunes():
    tunes = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for pl in ("jazz1460", "pop400"):
            for t in load_playlist(ROOT / f"data/ireal/{pl}.txt"):
                tunes[(t.title or "").lower()] = t
    return tunes


def main():
    tunes = load_tunes()
    rows = []
    for p in PILOTS:
        print("=" * 66)
        print(f"Pilot: {p['name']}  (expected: {p['expected']})")
        print(f"  {p['note']}")
        print("=" * 66)
        kind, path = p["src"]
        p_chords, bpb = (pchords_from_html(path) if kind == "html"
                         else pchords_from_json(path))
        tune = tunes.get(p["tune"]) or next(
            (t for k, t in tunes.items() if p["tune"] in k), None)
        mma = tune_to_mma(tune)

        result = align_irealb_to_inferred(mma, p_chords)
        inferred = dedup_inferred(p_chords)
        insts = section_instances(result.chords, inferred)
        # drop empty-content instances
        insts = [(lbl, segs) for lbl, segs in insts if segs]
        labels = [lbl for lbl, _ in insts]
        fps = [fingerprint(segs) for _, segs in insts]

        uniq = sorted(set(labels))
        n_same = sum(labels.count(u) * (labels.count(u) - 1) // 2 for u in uniq)
        print(f"  chart sections   : {sorted(set(s for _,s,_ in mma.timeline))}  "
              f"transpose=+{result.transpose_semitones} n_repeats={result.n_repeats}")
        print(f"  section instances: {len(insts)}  labels={labels}")
        print(f"  same-label pairs : {n_same}   distinct labels: {len(uniq)}")

        if n_same == 0 or len(uniq) < 2:
            print("  -> ABSTAIN (need >=2 labels and a repeated label). UNVERIFIABLE.\n")
            rows.append((p["name"], p["expected"], None, None, None, None, None, None,
                         "UNVERIFIABLE"))
            continue

        # (a) CLEAN
        w0, c0, d0 = repeat_consistency(labels, fps)

        # (b) SLIPPED — inject a LOCALIZED slip (failure #3, exactly the design
        #     doc's test): take one repeated-label instance and overwrite its
        #     content with a DIFFERENT-label instance's content, leaving the rest
        #     intact.  This makes that instance an outlier and must drop `within`.
        #     (A whole-cycle content rotation is a no-op on a periodic form —
        #     within/cross are rotation-invariant — so we corrupt a single slot.)
        rep_label = next(u for u in uniq if labels.count(u) >= 2)  # e.g. "A"
        donor_label = next(u for u in uniq if u != rep_label)      # e.g. "B"
        victim = labels.index(rep_label)                            # first A instance
        donor = labels.index(donor_label)
        fps_slip = list(fps)
        fps_slip[victim] = fps[donor]                               # A#1 now holds B content
        w1, c1, d1 = repeat_consistency(labels, fps_slip)

        # Per-instance outlier score: each repeated-label instance's mean cosine
        # to its same-label siblings. A localized slip makes the victim an
        # outlier even when the GLOBAL within-mean barely moves (dilution).
        def sibling_mean(fp_list, idx):
            sibs = [fp_list[j] @ fp_list[idx] for j in range(len(labels))
                    if j != idx and labels[j] == labels[idx]]
            return float(np.mean(sibs)) if sibs else float("nan")
        victim_clean = sibling_mean(fps, victim)
        victim_slip = sibling_mean(fps_slip, victim)
        # z-score of the victim vs the distribution of same-label sibling-means (clean)
        same_idx = [i for i in range(len(labels)) if labels[i] == rep_label]
        sib_means_clean = [sibling_mean(fps, i) for i in same_idx]
        sib_means_slip = [sibling_mean(fps_slip, i) for i in same_idx]
        mu, sd = np.mean(sib_means_clean), np.std(sib_means_clean) + 1e-9
        z_victim_slip = (sib_means_slip[same_idx.index(victim)] - mu) / sd
        print(f"  localization: victim {rep_label}#1 sibling-mean "
              f"clean={victim_clean:.4f} -> slip={victim_slip:.4f}  "
              f"(z vs clean siblings = {z_victim_slip:+.2f})")

        fam = section_family_fracs(result.chords)
        min_fam = min(fam.values()) if fam else float("nan")

        verdict_clean = "OK" if d0 > 0.05 else "SLIPPED"
        print(f"  CLEAN   within={w0:.4f} cross={c0:.4f}  Δ={d0:+.4f}  -> {verdict_clean}")
        print(f"  SLIPPED within={w1:.4f} cross={c1:.4f}  Δ={d1:+.4f}  "
              f"-> {'OK' if d1 > 0.05 else 'SLIPPED'}")
        print(f"  drop (clean-slip Δ): {d0 - d1:+.4f}   min section family_frac={min_fam:.3f}")
        print(f"  expected: {p['expected']}\n")
        rows.append((p["name"], p["expected"], w0, c0, d0, w1, c1, d1, verdict_clean))

    # summary
    print("\n" + "#" * 66)
    print("SUMMARY")
    print("#" * 66)
    hdr = f"{'Pilot':22s} {'exp':8s} {'clean Δ':>9s} {'slip Δ':>9s} {'sep(drop)':>10s} {'verdict':>10s}"
    print(hdr)
    for r in rows:
        name, exp = r[0], r[1]
        if r[4] is None:
            print(f"{name:22s} {exp:8s} {'--':>9s} {'--':>9s} {'--':>10s} {'UNVERIF':>10s}")
        else:
            d0, d1, v = r[4], r[7], r[8]
            print(f"{name:22s} {exp:8s} {d0:>+9.4f} {d1:>+9.4f} {d0-d1:>+10.4f} {v:>10s}")

    json.dump([{"pilot": r[0], "expected": r[1], "clean": r[2:5], "slip": r[5:8],
                "verdict": r[8]} for r in rows],
              open(SCRATCH / "m6_premise_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
