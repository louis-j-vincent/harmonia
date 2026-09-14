"""Does a PHASE search make merging better, or just easier to fool?

Arm 8 showed that on our own charts 24% of refused same-letter pairs become
mergeable if one occurrence slides 1-4 bars. Before recommending that, measure
what the extra freedom COSTS: giving the gate 9 chances to pass instead of 1 is
exactly how a false-merge rate gets inflated.

Policy `align+len+phase(tau)`: a span joins a group if, for SOME shift
delta in {0, +-b, +-2b, +-3b, +-4b} (b = the track's median chord duration, a
proxy for one bar), the shifted chroma agrees at >= tau. The block is then
written at the winning phase, and harm is scored against that same shifted
template — no free lunch in the scoring.

Compared head-to-head with the no-phase policy on the same 889 tracks.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
from fold_criteria_billboard import MIN_SPAN, base_letter, chordtone_seq, slot_cos  # noqa
from fold_policy_tradeoff import chroma_span, harm_of, span_track                   # noqa

OUT = os.path.join(HERE, "scratchpad", "fold_criteria")


def main():
    import mirdata
    import pandas as pd
    bb = mirdata.initialize("billboard")
    TAUS = [0.75, 0.80, 0.85, 0.87, 0.90, 0.93]
    acc = {}

    def add(name, w, h, tot):
        a = acc.setdefault(name, [0.0, 0.0, 0.0])
        a[0] += w
        a[1] += h
        a[2] += tot

    done = 0
    for tid in bb.track_ids:
        tr_ = bb.track(tid)
        try:
            sec, ch = tr_.sections, tr_.chords_full
        except Exception:
            continue
        if sec is None or ch is None:
            continue
        V, iv, nm = chordtone_seq(ch.labels, ch.intervals)
        if len(V) < 4:
            continue
        try:
            raw = pd.read_csv(tr_.bothchroma_path, header=None).values
            t = raw[:, 1].astype(float)
            X = raw[:, 2:26].astype(float)
            X = X[:, 12:] + X[:, :12]
            X = X / np.maximum(X.sum(1, keepdims=True), 1e-9)
            dt = float(np.median(np.diff(t)))
        except Exception:
            continue
        bar = float(np.median(iv[:, 1] - iv[:, 0]))      # ~one harmonic unit
        if not np.isfinite(bar) or bar <= 0:
            continue
        T_end = float(t[-1])
        spans = []
        for (a, b), lab in zip(sec.intervals, sec.labels):
            if b - a < MIN_SPAN or not lab or not lab[0].isalpha():
                continue
            if len(base_letter(lab)) > 1 or base_letter(lab) == "Z":
                continue
            cs = chroma_span(X, dt, a, b)
            if cs is None:
                continue
            spans.append([base_letter(lab), a, b, span_track(V, iv, a, b), cs])
        if len(spans) < 2:
            continue
        done += 1
        shifts = [k * bar for k in (0, 1, -1, 2, -2, 3, -3, 4, -4)]
        for tau in TAUS:
            groups = []
            for s in spans:
                joined = False
                for g in groups:
                    tm = g[0]
                    if tm[0] != s[0]:
                        continue
                    d1, d2 = tm[2] - tm[1], s[2] - s[1]
                    if min(d1, d2) / max(d1, d2) < 0.95:
                        continue
                    best = (-1.0, 0.0)
                    for d in shifts:
                        a2, b2 = s[1] + d, s[2] + d
                        if a2 < 0 or b2 > T_end:
                            continue
                        c2 = chroma_span(X, dt, a2, b2)
                        if c2 is None:
                            continue
                        v = slot_cos(tm[4], c2)
                        if v > best[0]:
                            best = (v, d)
                    if best[0] >= tau:
                        # the member is written AT ITS SHIFTED PHASE
                        g.append([s[0], s[1] + best[1], s[2] + best[1],
                                  span_track(V, iv, s[1] + best[1],
                                             s[2] + best[1]), None])
                        joined = True
                        break
                if not joined:
                    groups.append([s])
            w = h = tot = 0.0
            for g in groups:
                lab, t0, t1, trk, _ = g[0]
                w += t1 - t0
                for (_, a, b, _, _) in g:
                    tot += b - a
                    h += (b - a) * harm_of(trk, t1 - t0, a, b, V, iv)
            add(f"align+len+phase{tau}", w, h, tot)
        if done % 200 == 0:
            print(f"  {done} tracks", flush=True)
    res = {k: {"compression": v[0] / v[2], "harm": v[1] / v[2]}
           for k, v in acc.items()}
    old = json.load(open(os.path.join(OUT, "policy.json")))
    old.update(res)
    json.dump(old, open(os.path.join(OUT, "policy.json"), "w"), indent=1)
    print(f"\n{done} tracks")
    print(f"{'policy':<26}{'written':>10}{'harm':>9}   vs no-phase at same tau")
    for k in sorted(res, key=lambda k: res[k]["harm"]):
        tau = k.split("phase")[1]
        base = old.get(f"align+len{tau}")
        cmp_ = (f"   {base['compression']:.3f} / {base['harm']:.3f}"
                if base else "")
        print(f"{k:<26}{res[k]['compression']:>10.3f}{res[k]['harm']:>9.3f}{cmp_}")


if __name__ == "__main__":
    main()
