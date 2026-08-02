"""Concrete worked examples: two passages, their chords, every metric, the verdict.

Louis, 2026-08-02: "Je ne comprends pas, donne des exemples concrets."
Abstract AUC tables do not tell you what a criterion DOES. This prints real
chord sequences from Billboard next to what each metric says about them.
"""
import json
import numpy as np
import mirdata

rows = json.load(open("scratchpad/fold_criteria/pairs_chroma.json"))
lab = json.load(open("scratchpad/fold_criteria/pairs_label.json"))
ds = mirdata.initialize("billboard")

SHOW = ["len_agree", "chroma_align", "chroma_cv_inv", "seq_sim",
        "ct_bag_cos", "hr_sig"]

def chords_between(tid, t0, t1, maxn=10):
    tr = ds.track(tid)
    c = tr.chords_full
    out, prev = [], None
    for (a, b), l in zip(c.intervals, c.labels):
        if b <= t0 or a >= t1 or l == "N":
            continue
        if l != prev:
            out.append(l.replace(":maj", "").replace(":min", "m")
                        .replace(":", "").replace("hdim7", "ø7"))
            prev = l
    return " ".join(out[:maxn]) + (" …" if len(out) > maxn else "")

def show(rec, lrec, title, truth):
    tid = rec["tid"]
    t = lrec["t"]
    print(f"\n=== {title}")
    try:
        name = ds.track(tid).title
    except Exception:
        name = f"track {tid}"
    print(f"    {name}  (Billboard {tid})")
    print(f"    section {lrec['l1']}  {t[0]:6.1f}-{t[1]:6.1f}s : {chords_between(tid,t[0],t[1])}")
    print(f"    section {lrec['l2']}  {t[2]:6.1f}-{t[3]:6.1f}s : {chords_between(tid,t[2],t[3])}")
    print(f"    VERITE : {truth}")
    print("    " + "".join(f"{k.replace('chroma_','').replace('_',' '):>15}" for k in SHOW))
    print("    " + "".join(f"{rec[k]:>15.2f}" for k in SHOW))

# a clean positive: same letter, both metrics high
pos = [(r, l) for r, l in zip(rows, lab)
       if r["cls"] == 0 and r["chroma_align"] > .93 and r["len_agree"] > .97]
# the trap: DIFFERENT letters, chord-identical
trap = [(r, l) for r, l in zip(rows, lab)
        if r["cls"] == 3 and r["ct_bag_cos"] > .99 and r["hr_sig"] > .95]
# same letter but the chroma says no (the case a bag-of-chords would merge blind)
subtle = [(r, l) for r, l in zip(rows, lab)
          if r["cls"] == 2 and r["ct_bag_cos"] > .97 and r["chroma_align"] < .6]

show(*pos[0], "VRAIE REPRISE — meme lettre", "il FAUT fusionner")
show(*trap[0], "PIEGE — lettres DIFFERENTES, memes accords", "il ne faut PAS fusionner")
show(*subtle[0], "PIEGE 2 — memes accords, son different", "il ne faut PAS fusionner")

print("\n\n=== ce que chaque critere ferait sur les 3 cas ci-dessus")
print(f"{'critere':<18}{'vraie reprise':>15}{'piege 1':>12}{'piege 2':>12}   verdict")
taus = {}
cls = np.array([r["cls"] for r in rows])
for k in SHOW:
    v = np.nan_to_num(np.array([float(r[k]) for r in rows]))
    neg = v[(cls == 2) | (cls == 3)]
    taus[k] = float(np.quantile(neg, .98))
for k in SHOW:
    a, b, c = pos[0][0][k], trap[0][0][k], subtle[0][0][k]
    tau = taus[k]
    ok = "OK" if (a >= tau and b < tau and c < tau) else "se trompe"
    print(f"{k:<18}{a:>15.2f}{b:>12.2f}{c:>12.2f}   seuil {tau:.2f} -> {ok}")
