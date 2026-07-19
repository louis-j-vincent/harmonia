"""hierarchy_real.py — Task 2 deployment helper: MULTI-LEVEL structure for
REAL audio (bar-level root-probability arrays from real_root_proba.py), using
the probabilistic-root VARIABLE-SPAN encoder (keynorm_proba_varspan.pt) so
one single encoder can serve all three levels in-distribution:

  - "phrase"  level: fixed-phase learned-union at size=2  (mandated nuclear default)
  - "section" level: fixed-phase learned-union at size=8  (the validated
                      deployable level -- statistically tied with flat block8
                      per this session's multi-seed re-audit, kept as the
                      primary level for its demonstrated real-audio noise
                      robustness, Call 1 Stage B3a)
  - "form"    level: COARSENING of the section level -- re-cluster each
                      section label's representative span embedding at a
                      LOWER threshold, so e.g. a verse-like and chorus-like
                      section stay separate but two near-identical section
                      variants (a bridge repeat with a different turnaround,
                      etc.) can merge into one coarser "part".

No adaptive/free-form merge is used here (Task 2's finding: the free-form
agglomerative merge underperforms regardless of similarity source -- see
known_issues.md). This is a validated, deterministic 3-level STACK of the
same fixed-grid method, which is what "cluster at different structure
levels" is delivered as this session.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from symstruct_learned import MAXSPAN
from adaptive_proba import load_proba_encoder
from symstruct_proba import rotate13, estimate_tonic_pc


class ProbaBarsEmbedder:
    """Embeds arbitrary bar-spans of one ALREADY key-rotated real-audio
    bar_proba array (n_bars,13), with a cache."""
    def __init__(self, bar_proba, model, maxwidth=MAXSPAN):
        self.bp = bar_proba; self.model = model
        self.maxw = maxwidth; self.cache = {}

    def emb(self, spans):
        need = [sp for sp in spans if sp not in self.cache]
        if need:
            R, L = [], []
            for (s, e) in need:
                seg = self.bp[s:e]
                padded = np.zeros((self.maxw, 13), np.float32)
                l = min(len(seg), self.maxw)
                padded[:l] = seg[:self.maxw]
                R.append(padded); L.append(l)
            with torch.no_grad():
                z = self.model(torch.tensor(np.stack(R)), None,
                               torch.tensor(np.array(L)))
            for sp, v in zip(need, z):
                self.cache[sp] = v
        return torch.stack([self.cache[sp] for sp in spans])


def nuclear_spans(n, size):
    sp = [(s, min(s + size, n)) for s in range(0, n, size)]
    if len(sp) >= 2 and (sp[-1][1] - sp[-1][0]) < size / 2:
        s, e = sp.pop()
        sp[-1] = (sp[-1][0], e)
    return sp


def fixed_union_bars(bar_proba, model, size, tau, be=None):
    n = len(bar_proba)
    if n < size:
        return ["S0"] * n
    spans = nuclear_spans(n, size)
    if be is None:
        be = ProbaBarsEmbedder(bar_proba, model)
    E = be.emb(spans)
    S = (E @ E.t()).numpy()
    m = len(spans)
    parent = list(range(m))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[max(ra, rb)] = min(ra, rb)
    for i in range(m):
        for j in range(i + 1, m):
            if S[i, j] >= tau:
                union(i, j)
    remap = {}; lab = ["S0"] * n
    for k, (s, e) in enumerate(spans):
        r = find(k)
        if r not in remap: remap[r] = len(remap)
        for t in range(s, e):
            lab[t] = "S%d" % remap[r]
    return lab, spans


def coarsen_labels(bar_proba, model, section_labels, section_spans, tau_coarse, be=None):
    """Re-cluster section-level labels into fewer FORM-level groups by
    embedding each label's first representative span and union-finding at a
    lower threshold than the section level used."""
    if be is None:
        be = ProbaBarsEmbedder(bar_proba, model)
    # one representative span per distinct section label (its first occurrence)
    rep_span_for = {}
    for (s, e) in section_spans:
        lab = section_labels[s]
        if lab not in rep_span_for:
            rep_span_for[lab] = (s, e)
    labs_sorted = sorted(rep_span_for.keys(), key=lambda l: int(l[1:]))
    spans = [rep_span_for[l] for l in labs_sorted]
    E = be.emb(spans)
    S = (E @ E.t()).numpy()
    m = len(labs_sorted)
    parent = list(range(m))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[max(ra, rb)] = min(ra, rb)
    for i in range(m):
        for j in range(i + 1, m):
            if S[i, j] >= tau_coarse:
                union(i, j)
    remap = {}
    for k, lab in enumerate(labs_sorted):
        r = find(k)
        if r not in remap: remap[r] = "F%d" % len(remap)
    coarse = [remap[find(labs_sorted.index(l))] for l in section_labels]
    return coarse


def predict_multilevel(bar_proba, model, tau_phrase=0.80, tau_section=0.80,
                        tau_form=0.60):
    """bar_proba: (n_bars,13) RAW (not yet key-rotated) real pipeline output.
    Returns dict with 'phrase', 'section', 'form' label lists (len n_bars)
    and the estimated tonic pc used for key-normalization."""
    tonic = estimate_tonic_pc(bar_proba)
    shift = (-tonic) % 12
    bp = rotate13(bar_proba, shift)
    be = ProbaBarsEmbedder(bp, model)
    phrase, _ = fixed_union_bars(bp, model, 2, tau_phrase, be=be)
    section, sec_spans = fixed_union_bars(bp, model, 8, tau_section, be=be)
    form = coarsen_labels(bp, model, section, sec_spans, tau_form, be=be)
    return {"phrase": phrase, "section": section, "form": form,
            "est_tonic_pc": tonic}
