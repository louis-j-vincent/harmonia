"""Policy trade-off: what does each merge criterion COST and SAVE?

A criterion's AUC does not tell Louis what to pick. The decision the folder
actually makes is per LETTER GROUP: "write one block and repeat it, or write
the occurrences out separately?". So measure the two quantities that matter:

  compression = written time / total section time   (lower = more compact chart)
  harm        = fraction of section time whose SOUNDING chord differs from what
                the written block claims at that moment (GT labels = truth)

Policies compared on Billboard (890 tracks):
  write_out      every occurrence written separately          (harm 0, cost 1)
  letter         all same-letter occurrences share one block  (current pipeline)
  equal_len      same letter AND same length (+-5%)           (Louis's under-fold rule)
  gated(tau)     agglomerative on a CHROMA criterion          (the proposal)
  gated+len(tau) chroma criterion AND length agreement

Decisions use chroma (what the pipeline sees); harm is scored on GT labels.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
from fold_criteria_billboard import (MIN_SPAN, base_letter, chordtone_seq,  # noqa
                                     cos, slice_span, slot_cos)

OUT = os.path.join(HERE, "scratchpad", "fold_criteria")
NG = 256          # sample points per span when scoring harm


def span_track(V, iv, t0, t1, n=NG):
    """Chord-tone vector sampled on n equal points of [t0,t1]."""
    g = np.linspace(t0, t1, n, endpoint=False)
    k = np.clip(np.searchsorted(iv[:, 1], g, side="right"), 0, len(V) - 1)
    return V[k]


def harm_of(tmpl_track, tmpl_dur, t0, t1, V, iv, n=NG):
    """Tile the template (written at its own duration) over [t0,t1]; return the
    fraction of that time whose chord tones disagree with the truth."""
    g = np.linspace(t0, t1, n, endpoint=False)
    phase = ((g - t0) % tmpl_dur) / tmpl_dur
    ti = np.clip((phase * len(tmpl_track)).astype(int), 0, len(tmpl_track) - 1)
    k = np.clip(np.searchsorted(iv[:, 1], g, side="right"), 0, len(V) - 1)
    A, B = tmpl_track[ti], V[k]
    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    ok = den > 1e-9
    agree = np.zeros(n, bool)
    agree[ok] = num[ok] / den[ok] >= 0.99
    # both silent / "N" (zero chord-tone vector) is an AGREEMENT, not an error
    both_n = (np.linalg.norm(A, axis=1) < 1e-9) & (np.linalg.norm(B, axis=1) < 1e-9)
    agree |= both_n
    return float(1.0 - agree.mean())


def chroma_span(C, dt, t0, t1, n=64):
    a, b = int(t0 / dt), max(int(t0 / dt) + 1, int(t1 / dt))
    X = C[a:min(b, len(C))]
    if len(X) < 4:
        return None
    lo = np.clip(np.linspace(0, len(X), n + 1)[:-1].astype(int), 0, len(X) - 1)
    hi = np.clip(np.linspace(0, len(X), n + 1)[1:].astype(int), 1, len(X))
    return np.array([X[i:j].mean(0) for i, j in zip(lo, hi)])


def cv_pair(a, b):
    X = np.stack([a, b])
    return float(np.mean(np.sqrt(X.var(0)).mean(1)
                         / np.maximum(X.mean(0).mean(1), 1e-9)))


def group_by(spans, ok):
    """Greedy agglomeration in chronological order: a span joins the FIRST
    existing group whose template it agrees with (ok(template, span))."""
    groups = []
    for s in spans:
        for g in groups:
            if g[0][0] == s[0] and ok(g[0], s):    # same letter + criterion
                g.append(s)
                break
        else:
            groups.append([s])
    return groups


def score(groups, V, iv):
    """(written time, harmed time, total time) for a grouping."""
    w = h = tot = 0.0
    for g in groups:
        lab, t0, t1, tr, ch = g[0]
        w += t1 - t0
        for (_, a, b, _, _) in g:
            tot += b - a
            h += (b - a) * harm_of(tr, t1 - t0, a, b, V, iv)
    return w, h, tot


def main(limit=None, n_tracks_chroma=890):
    import mirdata
    import pandas as pd
    bb = mirdata.initialize("billboard")
    ids = list(bb.track_ids)[:limit] if limit else list(bb.track_ids)
    TAUS = [0.49, 0.60, 0.70, 0.75, 0.80, 0.83, 0.85, 0.87, 0.90, 0.93]
    # 0.49 = the SHIPPED CV_MAX=0.51 expressed as 1-CV; 0.80 = shipped
    # PERIOD_MIN_SCORE; 0.85 = shipped STACK_COHERENCE
    acc = {}

    def add(name, w, h, tot):
        a = acc.setdefault(name, [0.0, 0.0, 0.0, 0])
        a[0] += w
        a[1] += h
        a[2] += tot
        a[3] += 1

    done = 0
    for tid in ids:
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
        add("write_out", *score([[s] for s in spans], V, iv))
        add("letter", *score(group_by(spans, lambda p, q: True), V, iv))
        add("equal_len", *score(group_by(
            spans, lambda p, q: min(p[2] - p[1], q[2] - q[1])
            / max(p[2] - p[1], q[2] - q[1]) >= 0.95), V, iv))
        for tau in TAUS:
            add(f"align{tau}", *score(group_by(
                spans, lambda p, q, t=tau: slot_cos(p[4], q[4]) >= t), V, iv))
            add(f"align+len{tau}", *score(group_by(
                spans, lambda p, q, t=tau: slot_cos(p[4], q[4]) >= t
                and min(p[2] - p[1], q[2] - q[1]) / max(p[2] - p[1], q[2] - q[1])
                >= 0.95), V, iv))
            add(f"cv{tau}", *score(group_by(
                spans, lambda p, q, t=tau: 1.0 - cv_pair(p[4], q[4]) >= t), V, iv))
        if done % 100 == 0:
            print(f"  {done} tracks scored", flush=True)
        if done >= n_tracks_chroma:
            break
    res = {k: {"compression": v[0] / v[2], "harm": v[1] / v[2], "tracks": v[3]}
           for k, v in acc.items()}
    json.dump(res, open(os.path.join(OUT, "policy.json"), "w"), indent=1)
    print(f"\n{done} tracks\n{'policy':<18}{'written/total':>14}{'harm':>10}")
    for k in sorted(res, key=lambda k: res[k]["harm"]):
        print(f"{k:<18}{res[k]['compression']:>14.3f}{res[k]['harm']:>10.3f}")


if __name__ == "__main__":
    main(limit=int(sys.argv[1]) if len(sys.argv) > 1 else None)
