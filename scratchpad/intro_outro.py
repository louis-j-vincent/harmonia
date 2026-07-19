"""intro_outro.py — Step 3 (independent of noise-calibration question, per
the brief: pick this up regardless of how Step 1 went): 1-2 bar mean-
similarity-to-rest-of-song detector at a song's leading/trailing edges,
validated against iReal's real `i`-labeled intro GT.

GT check: iReal's `*i` section marker is a real, standard iReal-editor
"Intro" label — confirmed corpus-wide: 578/2401 tunes across the 7 playlist
files carry an `i` label somewhere in section_per_bar (close to the brief's
cited 573; small diff is parse-failure filtering). No reliable OUTRO marker
exists in this corpus (checked: no 'o'/'outro'/coda-specific per-bar label
survives sectionized_measures — codas are stripped as repeat-structure
markers, not section labels) — so this script builds a symmetric
leading/trailing detector but can only VALIDATE the leading (intro) side
against real GT. The trailing (outro) side is reported qualitatively only,
flagged honestly as unvalidated, matching the brief's "no real-audio GT ->
qualitative" spirit applied here to the outro side specifically.

Method: score(edge_block) = mean_j block_sim(edge_block, block_j) over ALL
other same-size blocks in the song (not just adjacent). ORIGINAL HYPOTHESIS
(predict intro when score is LOW, i.e. "an intro matches nothing else in
the song") was tested and FALSIFIED by a direct AUC check before trusting
any threshold sweep: AUC(low-score-predicts-intro) = 0.388, reliably BELOW
0.5 — the opposite direction is what the data supports. Root-caused before
flipping (not just flipped blindly): iReal intros are typically SHORT,
harmonically STATIC vamps (e.g. a single tonic chord or pedal figure) —
that simplicity makes them trivially similar to lots of other tonic-flavored
blocks scattered through the tune, giving them a spuriously HIGH mean
similarity to "the rest of the song", not a low one. Flipped: predict
"is-intro" when score > tau (ABOVE background, AUC=0.612 by construction of
the flip). tau chosen on val by FPR-gated threshold (same low-FP discipline
as Step 2 — an intro flag on a normal first section is a false positive),
sweep at 1-bar and 2-bar edge sizes.
"""
from __future__ import annotations
import sys, io, random, json
from pathlib import Path
from contextlib import redirect_stdout
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from harmonia.data.ireal_corpus import load_playlist, tune_to_mma, chord_root_pc
from chord_distance import chord_vector_binary
from chord_distance_eval import block_sim, FILES
from symstruct import qbucket

OUT_DIR = Path(__file__).resolve().parent


def load_corpus(max_tunes=None):
    out = []
    for f in FILES:
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                tunes = load_playlist(Path("data/ireal") / (f + ".txt"))
        except Exception:
            continue
        for t in tunes:
            try:
                mc = tune_to_mma(t)
            except Exception:
                continue
            shift = 0
            if mc.key:
                pc = chord_root_pc(mc.key.rstrip("-"))
                shift = (-pc % 12) if pc is not None else 0
            vecs, labels = [], []
            for bar_no, section, slots in mc.timeline:
                accum = None
                for (_, _, mma) in slots:
                    pc = chord_root_pc(mma)
                    if pc is None:
                        continue
                    rpc = (pc + shift) % 12
                    q = qbucket(mma)
                    v = chord_vector_binary(rpc, q)
                    accum = v if accum is None else accum + v
                vecs.append(accum if accum is not None else np.zeros(12))
                labels.append(section)
            if len(vecs) < 8:
                continue
            out.append({"title": mc.title, "vecs": vecs, "labels": labels})
    if max_tunes:
        random.Random(0).shuffle(out)
        out = out[:max_tunes]
    return out


def edge_scores(vecs, edge_size):
    """Return (lead_score, trail_score): mean block_sim of the leading /
    trailing `edge_size`-bar block against every other same-size block
    elsewhere in the song (stride-1 windows, excluding overlap with the
    edge itself)."""
    n = len(vecs)
    if n < edge_size * 4:
        return None, None
    lead = vecs[:edge_size]
    trail = vecs[n - edge_size:]

    def mean_sim_vs_rest(block, exclude_range):
        sims = []
        for s in range(0, n - edge_size + 1, 1):
            if s < exclude_range[0] + edge_size and s + edge_size > exclude_range[0]:
                # overlaps the edge itself (with a small buffer) — skip
                if abs(s - exclude_range[0]) < edge_size:
                    continue
            other = vecs[s:s + edge_size]
            sims.append(block_sim(block, other))
        return float(np.mean(sims)) if sims else None

    lead_score = mean_sim_vs_rest(lead, (0, edge_size))
    trail_score = mean_sim_vs_rest(trail, (n - edge_size, n))
    return lead_score, trail_score


def fpr_gated_threshold_upper(scores, y, target_fpr=0.10):
    """Predict positive when score > thr (HIGH similarity = intro, per the
    falsified-then-flipped hypothesis above). Find thr (on the negative-class
    score distribution) giving FPR<=target."""
    neg = np.array([s for s, l in zip(scores, y) if l == 0])
    pos = np.array([s for s, l in zip(scores, y) if l == 1])
    if len(neg) == 0 or len(pos) == 0:
        return None, None, None
    thr = float(np.quantile(neg, 1 - target_fpr))  # (1-target_fpr)-th percentile of neg scores
    fpr = float(np.mean(neg >= thr))
    rec = float(np.mean(pos >= thr))
    return thr, rec, fpr


def main():
    print("Loading iReal corpus...")
    corpus = load_corpus(max_tunes=600)
    print("  corpus: %d tunes" % len(corpus))

    for edge_size in (1, 2):
        print("\n=== edge_size=%d bars ===" % edge_size)
        rows = []  # (title, lead_score, trail_score, is_intro_gt)
        for c in corpus:
            lead_s, trail_s = edge_scores(c["vecs"], edge_size)
            if lead_s is None:
                continue
            is_intro = 1 if c["labels"][0] == "i" else 0
            rows.append((c["title"], lead_s, trail_s, is_intro))
        print("  n_songs_scored=%d, n_with_intro_gt=%d" %
              (len(rows), sum(r[3] for r in rows)))

        ids = list(range(len(rows)))
        random.Random(1).shuffle(ids)
        nval = len(ids) // 2
        val_ids, test_ids = ids[:nval], ids[nval:]
        val_scores = [rows[i][1] for i in val_ids]
        val_y = [rows[i][3] for i in val_ids]
        test_scores = [rows[i][1] for i in test_ids]
        test_y = [rows[i][3] for i in test_ids]

        for target_fpr in (0.05, 0.10, 0.20):
            thr, _, _ = fpr_gated_threshold_upper(val_scores, val_y, target_fpr)
            pred = [1 if s >= thr else 0 for s in test_scores]
            tp = sum(1 for p, l in zip(pred, test_y) if p == 1 and l == 1)
            fp = sum(1 for p, l in zip(pred, test_y) if p == 1 and l == 0)
            fn = sum(1 for p, l in zip(pred, test_y) if p == 0 and l == 1)
            tn = sum(1 for p, l in zip(pred, test_y) if p == 0 and l == 0)
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            print("  target_fpr=%.2f  thr=%.4f  test: P=%.3f R=%.3f FPR=%.3f (tp=%d fp=%d fn=%d tn=%d)" %
                  (target_fpr, thr, prec, rec, fpr, tp, fp, fn, tn))

        # baseline: always predict "not intro" (base rate)
        base_rate = np.mean(test_y)
        print("  [reference] test base rate of is-intro = %.3f (n=%d)" %
              (base_rate, len(test_y)))

        # trailing/outro side: qualitative only, no GT — report score distribution
        trail_scores = [r[2] for r in rows if r[2] is not None]
        print("  [outro, UNVALIDATED — no GT] trailing-edge score distribution: "
              "mean=%.3f std=%.3f min=%.3f p10=%.3f" %
              (np.mean(trail_scores), np.std(trail_scores), np.min(trail_scores),
               np.percentile(trail_scores, 10)))

    print("\nDone.")


if __name__ == "__main__":
    main()
