"""Rank the candidate merge criteria measured by fold_criteria_billboard.py
and emit an inspectable HTML report.

Headline per criterion (asymmetric loss — a false merge destroys music):
  * AUC positives vs ALL negatives, and vs HARD negatives only
  * tau@2%  = threshold whose false-merge rate on negatives is <= 2%
  * recall  = fraction of TRUE same-letter pairs that still merge at tau@2%
"""
from __future__ import annotations

import base64
import io
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(HERE, "scratchpad", "fold_criteria")
CLS = {0: "positive (A~A)", 1: "prime (A~A')", 2: "easy neg", 3: "HARD neg"}


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    x = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(x)) + 1.0
    # ties -> average ranks
    order = np.argsort(x)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = np.mean(r[order[i:j + 1]])
        i = j + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def tau_at_fpr(pos, neg, fpr=0.02):
    """Smallest threshold tau with mean(neg >= tau) <= fpr; recall = mean(pos>=tau)."""
    if len(neg) == 0:
        return float("nan"), float("nan")
    tau = float(np.quantile(neg, 1 - fpr))
    eps = 1e-9
    return tau + eps, float(np.mean(pos >= tau + eps))


def analyse(rows, keys, name):
    cls = np.array([r["cls"] for r in rows])
    out = []
    for k in keys:
        v = np.array([float(r[k]) for r in rows])
        pos, easy, hard, prime = v[cls == 0], v[cls == 2], v[cls == 3], v[cls == 1]
        allneg = np.concatenate([easy, hard])
        t2, rec2 = tau_at_fpr(pos, allneg, 0.02)
        th2, rech2 = tau_at_fpr(pos, hard, 0.02) if len(hard) else (np.nan, np.nan)
        t5, rec5 = tau_at_fpr(pos, allneg, 0.05)
        out.append({
            "arm": name, "criterion": k,
            "auc_all": auc(pos, allneg), "auc_hard": auc(pos, hard),
            "pos_med": float(np.median(pos)),
            "hard_med": float(np.median(hard)) if len(hard) else np.nan,
            "easy_med": float(np.median(easy)),
            "prime_med": float(np.median(prime)) if len(prime) else np.nan,
            "tau2": t2, "rec2": rec2, "tau_hard2": th2, "rec_hard2": rech2,
            "tau5": t5, "rec5": rec5,
            "n_pos": int(len(pos)), "n_hard": int(len(hard)),
            "n_easy": int(len(easy)), "n_prime": int(len(prime)),
        })
    out.sort(key=lambda r: -(0 if np.isnan(r["rec2"]) else r["rec2"]))
    return out


def combo(rows, pairs):
    """AND-rules: merge only if every member criterion passes its own tau.
    Thresholds are jointly tuned on a coarse grid to maximise recall at
    false-merge <= 2% over ALL negatives."""
    cls = np.array([r["cls"] for r in rows])
    res = []
    for ks in pairs:
        V = np.stack([np.array([float(r[k]) for r in rows]) for k in ks])
        pos = V[:, cls == 0]
        neg = V[:, (cls == 2) | (cls == 3)]
        hard = V[:, cls == 3]
        grids = [np.quantile(V[i], np.linspace(0.50, 0.999, 40)) for i in range(len(ks))]
        best = None
        for t0 in grids[0]:
            for t1 in grids[1]:
                t = np.array([t0, t1])[:, None]
                fp = float(np.mean(np.all(neg >= t, axis=0)))
                if fp > 0.02:
                    continue
                rc = float(np.mean(np.all(pos >= t, axis=0)))
                if best is None or rc > best[0]:
                    best = (rc, [float(t0), float(t1)], fp,
                            float(np.mean(np.all(hard >= t, axis=0))))
        if best:
            res.append({"criterion": " AND ".join(ks), "rec2": best[0],
                        "taus": best[1], "fp": best[2], "fp_hard": best[3]})
    res.sort(key=lambda r: -r["rec2"])
    return res


def hist_png(rows, k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cls = np.array([r["cls"] for r in rows])
    v = np.array([float(r[k]) for r in rows])
    fig, ax = plt.subplots(figsize=(4.1, 1.9), dpi=110)
    bins = np.linspace(min(0, v.min()), max(1.0, v.max()), 46)
    for c, col, lab in ((2, "#bbb", "easy neg"), (3, "#d1495b", "HARD neg"),
                        (0, "#2a9d8f", "positive")):
        s = v[cls == c]
        if len(s):
            ax.hist(s, bins=bins, density=True, alpha=.62, color=col, label=lab)
    ax.set_title(k, fontsize=8)
    ax.legend(fontsize=5.5, frameon=False)
    ax.tick_params(labelsize=6)
    ax.set_yticks([])
    fig.tight_layout(pad=.3)
    b = io.BytesIO()
    fig.savefig(b, format="png")
    plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


def examples_html(rows, n_each=5):
    """Real pairs Louis can read: the ones the criteria get wrong, with the
    actual chord sequences of both spans."""
    import sys
    sys.path.insert(0, os.path.join(HERE, "scripts"))
    import mirdata
    from fold_criteria_billboard import chordtone_seq, dedup, slice_span
    bb = mirdata.initialize("billboard")
    cache = {}

    def chords(tid, t0, t1):
        if tid not in cache:
            tr = bb.track(tid)
            cache[tid] = (tr,) + chordtone_seq(tr.chords_full.labels,
                                               tr.chords_full.intervals)
        tr, V, iv, nm = cache[tid]
        s = slice_span(V, iv, t0, t1, nm)
        if s is None:
            return tr, "?"
        dn, _ = dedup(s[2], s[1])
        return tr, " ".join(dn[:18]) + (" …" if len(dn) > 18 else "")

    pos = [r for r in rows if r["cls"] == 0 and "t" in r]
    hard = [r for r in rows if r["cls"] == 3 and "t" in r]
    picks = (
        [("SAME letter, very unequal lengths — the Norah shape", r)
         for r in sorted(pos, key=lambda r: r["len_agree"])[:n_each]]
        + [("DIFFERENT letters, high chord+time agreement — worst false merge", r)
           for r in sorted(hard, key=lambda r: -min(r["len_agree"], r["ct_align"]))[:n_each]]
        + [("SAME letter, clean merge — what we must keep", r)
           for r in sorted(pos, key=lambda r: -min(r["len_agree"], r["ct_align"],
                                                   r["seq_sim"]))[:n_each]])
    h = ("<tr><th>case<th>track<th>letters<th>dur<th>len_agree<th>ct_align"
         "<th>seq_sim<th>tile_cov<th>hr_sig<th>chords</tr>")
    b = []
    for tag, r in picks:
        a1, b1, a2, b2 = r["t"]
        tr, ca = chords(r["tid"], a1, b1)
        _, cb = chords(r["tid"], a2, b2)
        title = f"{getattr(tr,'artist','?')} — {getattr(tr,'title','?')}"
        b.append(
            f"<tr><td rowspan=2 class=k style='font-size:11px'>{tag}"
            f"<td rowspan=2 class=k>{title}<td>{r['l1']}<td>{r['d1']:.0f}s"
            f"<td rowspan=2>{fmt(r['len_agree'],2)}<td rowspan=2>{fmt(r['ct_align'],2)}"
            f"<td rowspan=2>{fmt(r['seq_sim'],2)}<td rowspan=2>{fmt(r['tile_cov'],2)}"
            f"<td rowspan=2>{fmt(r['hr_sig'],2)}"
            f"<td class=k style='font-size:10.5px'>{ca}</tr>"
            f"<tr><td>{r['l2']}<td>{r['d2']:.0f}s"
            f"<td class=k style='font-size:10.5px'>{cb}</tr>")
    return f"<table>{h}{''.join(b)}</table>"


def fmt(x, n=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{n}f}"


def harm_arm(chr_, keys, cov_key="tile_cov", cov_thr=0.95):
    """Reframed target (see log): the folder's real job is NOT to recover
    SALAMI letters — it is to avoid writing a block that does not reproduce
    what is played. Ground truth here = the LABEL-side tiling coverage
    (fraction of span 2's time the span-1 template actually gets right).
    Positive = coverage >= 0.95, negative = coverage < 0.80. Predictors are
    the CHROMA criteria only (what the system can actually see)."""
    y = np.array([r[cov_key] for r in chr_])
    keep = (y >= cov_thr) | (y < 0.80)
    rows = [r for r, k in zip(chr_, keep) if k]
    lbl = (y[keep] >= cov_thr).astype(int)
    out = []
    for k in keys:
        v = np.array([float(r[k]) for r in rows])
        pos, neg = v[lbl == 1], v[lbl == 0]
        t2, rec2 = tau_at_fpr(pos, neg, 0.02)
        t5, rec5 = tau_at_fpr(pos, neg, 0.05)
        out.append({"arm": "HARM", "criterion": k, "auc_all": auc(pos, neg),
                    "auc_hard": np.nan, "pos_med": float(np.median(pos)),
                    "hard_med": float(np.median(neg)), "easy_med": np.nan,
                    "prime_med": np.nan, "tau2": t2, "rec2": rec2,
                    "tau_hard2": t5, "rec_hard2": rec5,
                    "n_pos": int(len(pos)), "n_hard": int(len(neg)),
                    "n_easy": 0, "n_prime": 0})
    out.sort(key=lambda r: -r["rec2"])
    return out


def harm_combo(chr_, pairs, cov_key="tile_cov", cov_thr=0.95):
    y = np.array([r[cov_key] for r in chr_])
    keep = (y >= cov_thr) | (y < 0.80)
    rows = [r for r, k in zip(chr_, keep) if k]
    lbl = (y[keep] >= cov_thr).astype(int)
    res = []
    for ks in pairs:
        V = np.stack([np.array([float(r[k]) for r in rows]) for k in ks])
        pos, neg = V[:, lbl == 1], V[:, lbl == 0]
        grids = [np.quantile(V[i], np.linspace(0.40, 0.999, 45)) for i in range(2)]
        best = None
        for t0 in grids[0]:
            for t1 in grids[1]:
                t = np.array([t0, t1])[:, None]
                fp = float(np.mean(np.all(neg >= t, axis=0)))
                if fp > 0.02:
                    continue
                rc = float(np.mean(np.all(pos >= t, axis=0)))
                if best is None or rc > best[0]:
                    best = (rc, [float(t0), float(t1)], fp, float("nan"))
        if best:
            res.append({"criterion": " AND ".join(ks), "rec2": best[0],
                        "taus": best[1], "fp": best[2], "fp_hard": best[3]})
    res.sort(key=lambda r: -r["rec2"])
    return res


def policy_png():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = json.load(open(os.path.join(D, "policy.json")))
    fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=115)
    fam = {"align+len": ("#2a9d8f", "o"), "align": ("#457b9d", "s"),
           "cv": ("#e9c46a", "^")}
    for pre, (col, mk) in fam.items():
        pts = sorted((v["compression"], v["harm"], k) for k, v in P.items()
                     if k.startswith(pre) and not (pre == "align"
                                                   and k.startswith("align+")))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], mk + "-", color=col,
                ms=5, lw=1.4, label=pre + " (tau sweep)")
    for k, col, mk in (("letter", "#d1495b", "*"), ("equal_len", "#8338ec", "D"),
                       ("write_out", "#555", "P")):
        v = P[k]
        ax.plot(v["compression"], v["harm"], mk, color=col, ms=13 if mk == "*" else 8)
        ax.annotate(k, (v["compression"], v["harm"]), textcoords="offset points",
                    xytext=(8, 6), fontsize=9, color=col, weight="bold")
    ax.set_xlabel("chart length written  (fraction of the song's section time)")
    ax.set_ylabel("harm: fraction of time the written block gets the chord WRONG")
    ax.set_title("Fold policies on Billboard (889 tracks) — pick your point",
                 fontsize=11)
    ax.axhline(P["write_out"]["harm"], ls=":", c="#999", lw=1)
    ax.annotate("floor (span-edge quantisation)",
                (0.42, P["write_out"]["harm"]), fontsize=7.5, color="#777")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8.5, frameon=False)
    fig.tight_layout()
    b = io.BytesIO()
    fig.savefig(b, format="png")
    plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


def policy_table():
    P = json.load(open(os.path.join(D, "policy.json")))
    order = sorted(P, key=lambda k: P[k]["harm"])
    h = ("<tr><th>policy<th>chart length written<th>harm (chords wrong)"
         "<th>harm above floor</tr>")
    fl = P["write_out"]["harm"]
    b = "".join(f"<tr><td class=k>{k}<td>{100*P[k]['compression']:.1f}%"
                f"<td>{100*P[k]['harm']:.1f}%"
                f"<td class=hi>{100*(P[k]['harm']-fl):.1f}%</tr>" for k in order)
    return f"<table>{h}{b}</table>"


def ourcharts_html():
    q = os.path.join(D, "ourcharts.json")
    if not os.path.exists(q):
        return ""
    oc = json.load(open(q))
    import itertools
    h = ("<tr><th>our chart<th>letter<th>occurrence lengths (bars)"
         "<th>blocks the rule makes<th>bars written TODAY<th>bars under the rule</tr>")
    b, td, tr_, ta = "", 0, 0, 0
    npair = okpair = 0
    for stem, letters in oc.items():
        for L, d in letters.items():
            gs = "+".join("".join(str(i + 1) for i in g) for g in d["groups"])
            b += (f"<tr><td class=k>{stem.replace('_',' ')}<td>{L}"
                  f"<td class=k>{' '.join(str(x) for x in d['lengths'])}"
                  f"<td>{len(d['groups'])} ({gs})<td>{d['bars_today']}"
                  f"<td class=hi>{d['bars_rule']}</tr>")
            td += d["bars_today"]
            tr_ += d["bars_rule"]
            ta += sum(d["lengths"])
            for x, y in itertools.combinations(d["lengths"], 2):
                npair += 1
                okpair += min(x, y) / max(x, y) >= 0.95
    b += (f"<tr><td class=k colspan=4><b>TOTAL</b> (write-out would be {ta})"
          f"<td><b>{td}</b><td class=hi><b>{tr_}</b></tr>")
    return (f"<table>{h}{b}</table><p><b>Why:</b> {100*okpair/max(npair,1):.1f}% of "
            f"our charts' same-letter pairs agree in length within 5%, against "
            f"<b>61.7%</b> of Billboard's. Our sections are cut at inconsistent "
            f"lengths, so the length term refuses almost everything and the "
            f"harmony term never gets a say.</p>")


def norah_html():
    p = os.path.join(D, "norah.json")
    if not os.path.exists(p):
        return ""
    n = json.load(open(p))
    keys = [k for k in n["rows"][0] if k != "pair"]
    h = "<tr><th>pair of A occurrences" + "".join(f"<th>{k}" for k in keys) + "</tr>"
    b = ""
    for r in n["rows"]:
        ok = "A3" in r["pair"] and "A4" in r["pair"]
        b += (f"<tr style='background:{'#e8f5f2' if ok else '#fff'}'>"
              f"<td class=k>{r['pair']}"
              + "".join(f"<td>{fmt(r[k],3)}" for k in keys) + "</tr>")
    b += ("<tr><td class=k><b>MEDIAN</b>"
          + "".join(f"<td><b>{fmt(n['median'][k],3)}</b>" for k in keys) + "</tr>")
    return f"<table>{h}{b}</table>"


def main():
    lab = json.load(open(os.path.join(D, "pairs_label.json")))
    chr_ = json.load(open(os.path.join(D, "pairs_chroma.json")))
    LKEYS = ["len_agree", "nchord_agree", "seq_sim", "ct_bag_cos", "ct_align",
             "tile_cov", "hr_sig", "bigram_jac", "chordset_jac"]
    CKEYS = ["chroma_bag_cos", "chroma_align", "chroma_lag_unstretched",
             "chroma_cv_inv", "chroma_hr"]
    A = analyse(lab, LKEYS, "LABEL (GT chords)")
    B = analyse(chr_, CKEYS, "CHROMA (NNLS audio)")
    # len_agree is available to the live system without any chord content
    # (the section detector already knows each occurrence's bar count), so it
    # is a legitimate predictor in the harm arm alongside the chroma ones.
    H = harm_arm(chr_, CKEYS + ["len_agree"]) if chr_ and "tile_cov" in chr_[0] else []
    Hc = harm_combo(chr_, [("chroma_align", "len_agree"),
                           ("chroma_cv_inv", "len_agree"),
                           ("chroma_align", "chroma_cv_inv"),
                           ("chroma_lag_unstretched", "len_agree"),
                           ("chroma_align", "chroma_hr")]) if H else []
    C = combo(lab, [("len_agree", "ct_align"), ("len_agree", "seq_sim"),
                    ("len_agree", "tile_cov"), ("ct_align", "hr_sig"),
                    ("seq_sim", "ct_bag_cos"), ("tile_cov", "hr_sig"),
                    ("len_agree", "bigram_jac"), ("ct_align", "len_agree"),
                    ("seq_sim", "hr_sig"), ("tile_cov", "len_agree")])
    Cc = combo(chr_, [("chroma_align", "chroma_hr"),
                      ("chroma_align", "chroma_cv_inv"),
                      ("chroma_lag_unstretched", "chroma_hr")])
    json.dump({"label": A, "chroma": B, "harm": H, "harm_combo": Hc,
               "combo_label": C, "combo_chroma": Cc},
              open(os.path.join(D, "summary.json"), "w"), indent=1)

    def table(rows):
        h = ("<tr><th>criterion<th>AUC vs all neg<th>AUC vs HARD<th>median pos"
             "<th>median HARD<th>median A~A'<th>&tau; @2% FP<th><b>recall @2%</b>"
             "<th>&tau; @2% on HARD only<th>recall there</tr>")
        b = "".join(
            f"<tr><td class=k>{r['criterion']}<td>{fmt(r['auc_all'])}"
            f"<td>{fmt(r['auc_hard'])}<td>{fmt(r['pos_med'])}<td>{fmt(r['hard_med'])}"
            f"<td>{fmt(r['prime_med'])}<td>{fmt(r['tau2'])}"
            f"<td class=hi>{fmt(100*r['rec2'],1)}%<td>{fmt(r['tau_hard2'])}"
            f"<td>{fmt(100*r['rec_hard2'],1)}%</tr>" for r in rows)
        return f"<table>{h}{b}</table>"

    imgs = "".join(f'<img src="data:image/png;base64,{hist_png(lab,k)}">'
                   for k in LKEYS)
    imgs_c = "".join(f'<img src="data:image/png;base64,{hist_png(chr_,k)}">'
                     for k in CKEYS)
    def ctable(rows):
        h = ("<tr><th>AND rule<th>thresholds<th>false merge (all neg)"
             "<th>false merge (HARD)<th><b>recall</b></tr>")
        b = "".join(f"<tr><td class=k>{r['criterion']}<td>"
                    + " / ".join(fmt(t) for t in r["taus"])
                    + f"<td>{fmt(100*r['fp'],1)}%<td>{fmt(100*r['fp_hard'],1)}%"
                    f"<td class=hi>{fmt(100*r['rec2'],1)}%</tr>" for r in rows)
        return f"<table>{h}{b}</table>"

    ex = examples_html(lab)
    n_tracks = len({r["tid"] for r in lab})
    css = """body{font:14px/1.5 -apple-system,Segoe UI,sans-serif;margin:26px;
      max-width:1180px;color:#222} table{border-collapse:collapse;margin:10px 0 22px;
      font-size:13px} th,td{border:1px solid #ddd;padding:4px 8px;text-align:right}
      th{background:#f4f4f4;font-weight:600;font-size:11.5px}
      td.k{text-align:left;font-family:ui-monospace,monospace}
      td.hi{background:#e8f5f2;font-weight:700} img{margin:2px}
      h2{margin-top:30px;border-bottom:2px solid #eee;padding-bottom:4px}
      .note{background:#fffbe6;border-left:4px solid #f0c040;padding:8px 12px;
      margin:12px 0} code{background:#f4f4f4;padding:1px 4px}"""
    html = f"""<!doctype html><meta charset=utf-8>
<title>Fold merge criteria — Billboard</title><style>{css}</style>
<h1>Which criterion says "these two spans are the same section"?</h1>
<p>Billboard GT, {n_tracks} tracks, {len(lab)} span pairs
({sum(1 for r in lab if r['cls']==0)} positive, {sum(1 for r in lab if r['cls']==3)}
HARD negative, {sum(1 for r in lab if r['cls']==2)} easy negative,
{sum(1 for r in lab if r['cls']==1)} A~A' prime pairs).
Positive = same SALAMI letter. HARD negative = different letter whose chord-label
sets overlap &ge; 0.6 Jaccard (verse vs chorus over the same loop).</p>
<div class=note><b>How to read the headline column.</b> <code>recall @2%</code> =
of all pairs that GT says are the same section, what fraction would still be
merged if we set the threshold so that only 2% of genuinely different pairs get
merged. Higher = writes the chart more compactly without destroying music.</div>
<h2>Arm 1 — criteria on GT chord labels (upper bound)</h2>
{table(A)}
{imgs}
<h2>Arm 2 — same shapes on real NNLS chroma (McGill bothchroma, no decoder)</h2>
<p>This is the substrate the shipped constants live on:
<code>chroma_lag_unstretched</code> is <code>section_period</code>'s
<code>Vb[b]@Vb[b+P]</code> (PERIOD_MIN_SCORE=0.80),
<code>chroma_align</code> is STACK_COHERENCE (0.85),
<code>chroma_cv_inv</code> is 1&minus;CV (CV_MAX=0.51 &rArr; 0.49 here).</p>
{table(B)}
{imgs_c}
<h2>Arm 3 — AND-rules (two criteria, thresholds jointly tuned at 2% false merge)</h2>
{ctable(C)}
{ctable(Cc)}
<h2>Arm 4 — the reframed target: "does the block reproduce what is played?"</h2>
<div class=note>SALAMI letters are the wrong ground truth for a CHORD chart. A
verse and a chorus over the identical loop are different letters but the same
chords — writing them once loses nothing on a chord sheet. So here the target is
not the letter: positive = the span-1 template, tiled, reproduces &ge;95% of
span 2's chord-tone TIME (GT labels); negative = &lt;80%. Predictors are the
chroma criteria only — what the pipeline can actually see.
Columns <code>&tau;@2%</code>/<code>recall</code> as before;
the last two columns are the 5% operating point.</div>
{table(H)}
{ctable(Hc)}
<h2>The Norah verdict — Don't Know Why, letter A, 6 occurrences</h2>
<p>Written as ONE 4-bar block covering 8/6/12/12/16/8 bars. Green row = the only
pair that is genuinely the same length and the same music (A3 vs A4). Every
criterion above its Billboard &tau;@2% would MERGE the pair; below, refuse.</p>
{norah_html()}
<h2>What each policy COSTS — the choice, on 889 Billboard tracks</h2>
<div class=note>x = how long the chart is (1.0 = every occurrence written out).
y = how much of the song's harmony the written chart gets WRONG.
Down-left is better. The current pipeline (<code>letter</code>: all same-letter
occurrences share one block) is the red star.</div>
<img src="data:image/png;base64,{policy_png()}" style="max-width:100%">
{policy_table()}
<h2>Norah, as a policy choice</h2>
<pre style="background:#f7f7f7;padding:10px;font-size:12.5px">
policy for Don't Know Why, letter A (6 occurrences, 62 bars)   bars written   harm
SHIPPED  minimal_fold block = [bar0 bar1 bar0 bar1]                   4       72.6%
"letter" first occurrence (8 bars) tiled over all six                 8       37.1%
equal_len  /  chroma_align >= 0.85   (A3+A4 merge, rest apart)       50        0.0%
write_out  every occurrence separately                               62        0.0%
</pre>
<h2>The same rule on OUR six charts — the finding that inverts the recommendation</h2>
<div class=note>Rule = <code>chroma_align &ge; 0.85 AND len_agree &ge; 0.95</code>,
chroma taken from each song's own audio on its own bar grid. On Billboard this
rule is a good trade. On our charts it barely folds — because the spans it is
asked to compare are cut at inconsistent lengths.</div>
{ourcharts_html()}
<h2>Examples you can read</h2>
<p>Real Billboard pairs near the decision boundary, chord sequences shown.</p>
{ex}
"""
    out = os.path.join(HERE, "docs", "fold_criteria_billboard.html")
    open(out, "w").write(html)
    print("wrote", out)
    print(json.dumps({"label_top": A[:4], "chroma_top": B[:3],
                      "combo_top": C[:4]}, indent=1))


if __name__ == "__main__":
    main()
