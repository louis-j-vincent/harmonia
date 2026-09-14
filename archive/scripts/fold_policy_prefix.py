"""Candidate criterion #10: PREFIX agreement instead of whole-span agreement.

Motivation (measured 2026-08-02): the length term refuses most merges because
same-letter occurrences differ in length — a 16-bar A and a 12-bar A. But if
the 12-bar one is the 16-bar one minus its tail, the right chart writes the
16-bar block ONCE and says the short pass stops early. Nothing is destroyed and
nothing is duplicated.

So: template = the LONGEST occurrence in the group, written at real time scale.
A shorter occurrence joins if the template's PREFIX (unstretched, same seconds
from the start) reproduces it. Harm is scored against that prefix, not a tiling.

Compared against the tiling policies from fold_policy_tradeoff.py.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
from fold_criteria_billboard import MIN_SPAN, base_letter, chordtone_seq  # noqa
from fold_policy_tradeoff import chroma_span, harm_of, span_track        # noqa

OUT = os.path.join(HERE, "scratchpad", "fold_criteria")
NG = 256


def prefix_agree(long_ch, short_ch, d_long, d_short):
    """Cosine agreement between the short span and the same NUMBER OF SECONDS
    at the start of the long span. Both are 64-slot resamples of their own
    span, so slice the long one to the first d_short/d_long of its slots."""
    n = len(long_ch)
    k = max(2, int(round(n * d_short / d_long)))
    A = long_ch[:k]
    idx = np.clip((np.linspace(0, len(short_ch), k, endpoint=False)).astype(int),
                  0, len(short_ch) - 1)
    B = short_ch[idx]
    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    ok = den > 1e-9
    return float(np.mean(num[ok] / den[ok])) if ok.any() else 0.0


def harm_prefix(tmpl_track, d_tmpl, t0, t1, V, iv, n=NG):
    """The block is written at real time scale and the member reads only its
    own first (t1-t0) seconds of it — no tiling, no stretching."""
    g = np.linspace(t0, t1, n, endpoint=False)
    frac = (g - t0) / d_tmpl
    ti = np.clip((frac * len(tmpl_track)).astype(int), 0, len(tmpl_track) - 1)
    k = np.clip(np.searchsorted(iv[:, 1], g, side="right"), 0, len(V) - 1)
    A, B = tmpl_track[ti], V[k]
    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    ok = den > 1e-9
    ag = np.zeros(n, bool)
    ag[ok] = num[ok] / den[ok] >= 0.99
    ag |= (np.linalg.norm(A, axis=1) < 1e-9) & (np.linalg.norm(B, axis=1) < 1e-9)
    return float(1 - ag.mean())


def main():
    import mirdata
    import pandas as pd
    bb = mirdata.initialize("billboard")
    TAUS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.93]
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
        spans = []
        for (a, b), lab in zip(sec.intervals, sec.labels):
            if b - a < MIN_SPAN or not lab or not lab[0].isalpha():
                continue
            if len(base_letter(lab)) > 1 or base_letter(lab) == "Z":
                continue
            cs = chroma_span(X, dt, a, b)
            if cs is None:
                continue
            spans.append((base_letter(lab), a, b, span_track(V, iv, a, b), cs))
        if len(spans) < 2:
            continue
        done += 1
        for tau in TAUS:
            # groups keyed by letter; template = longest member so far
            groups = []
            for s in spans:
                for g in groups:
                    tmpl = max(g, key=lambda x: x[2] - x[1])
                    if tmpl[0] != s[0]:
                        continue
                    lo, hi = sorted([tmpl, s], key=lambda x: x[2] - x[1])
                    if prefix_agree(hi[4], lo[4], hi[2] - hi[1],
                                    lo[2] - lo[1]) >= tau:
                        g.append(s)
                        break
                else:
                    groups.append([s])
            w = h = tot = 0.0
            for g in groups:
                tmpl = max(g, key=lambda x: x[2] - x[1])
                dT = tmpl[2] - tmpl[1]
                w += dT
                for (_, a, b, _, _) in g:
                    tot += b - a
                    h += (b - a) * harm_prefix(tmpl[3], dT, a, b, V, iv)
            add(f"prefix{tau}", w, h, tot)
        if done % 200 == 0:
            print(f"  {done} tracks", flush=True)
    res = {k: {"compression": v[0] / v[2], "harm": v[1] / v[2]}
           for k, v in acc.items()}
    old = json.load(open(os.path.join(OUT, "policy.json")))
    old.update(res)
    json.dump(old, open(os.path.join(OUT, "policy.json"), "w"), indent=1)
    print(f"\n{done} tracks\n{'policy':<16}{'written/total':>14}{'harm':>10}")
    for k in sorted(res, key=lambda k: res[k]["harm"]):
        print(f"{k:<16}{res[k]['compression']:>14.3f}{res[k]['harm']:>10.3f}")


if __name__ == "__main__":
    main()
