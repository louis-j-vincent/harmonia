"""Le score validé, confronté à la vérité de Louis. Et le seuil qui en sort.

    python scripts/score_eval.py [<stem> ...]

Louis, 2026-08-07 : « validé pour le score normalisation ancre + harmonie », puis
la stratégie qu'il a décrite plus tôt :

  « On calcule un score global (à maximiser) qui est la moyenne des scores
    normalisés (par rapport au score du pic d'origine) qu'on a. Chaque seuil nous
    crée une hypothèse de recouvrement du morceau par section, et donc un score
    global ; on prend le seuil qui maximise notre score global. »

CE QUE CE SCRIPT VÉRIFIE, ET POURQUOI IL FAUT LE VÉRIFIER. La règle « prendre le
seuil qui maximise la moyenne des scores retenus » a un défaut de construction
qu'il vaut mieux constater que subir : **plus le seuil est haut, moins on garde
de pics, et plus la moyenne des survivants est bonne.** Maximiser cette moyenne
pousse donc mécaniquement vers un seuil qui ne garde qu'un seul pic parfait et
n'explique rien. C'est le même piège que le ratio de couverture qu'on a déjà
retiré de `blocks_flex` en août : une clé qui récompense d'expliquer moins.

On mesure donc DEUX choses sur le même balayage de seuils :

  * le **score global de Louis** — la moyenne des scores normalisés retenus ;
  * l'**accord avec sa vérité** — les frontières trouvées contre les siennes, et
    l'accord des lettres paire de mesures par paire de mesures.

Si les deux courbes culminent au même endroit, sa règle se suffit à elle-même et
on peut la brancher. Sinon on saura de combien elle se trompe, et dans quel sens.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from pattern_lanes import load                                            # noqa: E402
import vocal_anchor as VA                                                 # noqa: E402
import blocks8 as B8                                                      # noqa: E402
import melody_ssm as MS                                                   # noqa: E402
import melody_check as MC                                                 # noqa: E402
import vocal_melody as VM                                                 # noqa: E402
import voice_first as VF                                                  # noqa: E402

BLOCK, UNIT = VF.BLOCK, VF.UNIT
TOL = 2            # « au bon endroit » = à une double-mesure près
API = "http://127.0.0.1:7772/api/sections"


def truth():
    """Les annotations de Louis, telles qu'il les a posées."""
    with urllib.request.urlopen(API, timeout=10) as r:
        return json.load(r)


def anchor_runs(S, M, n, vstart, thr, max_blocks=12, BLOCK=BLOCK):
    """L'ancrage de `voice_first`, mais sur le score validé et à seuil imposé."""
    from scipy.signal import find_peaks
    claimed = np.zeros(n, bool)
    runs, cursor = [], vstart
    while cursor + BLOCK <= n and len(runs) < max_blocks:
        if claimed[cursor:cursor + BLOCK].any():
            cursor += UNIT
            continue
        cur_m = MC.slide_on(M, cursor, BLOCK, n)
        cur_h = MC.slide_on(S, cursor, BLOCK, n)
        sc = VF.block_score(cur_m, cur_h, cursor, n)
        idx, _ = find_peaks(sc, height=thr, distance=UNIT)
        cand = [int(p) for p in idx
                if (p - cursor) % UNIT == 0 and abs(p - cursor) >= BLOCK
                and p + BLOCK <= n and not claimed[p:p + BLOCK].any()]
        occ = []
        for p in sorted(cand, key=lambda p: -sc[p]):
            if any(abs(p - q) < BLOCK for q in occ):
                continue
            occ.append(p)
        occ.sort()
        runs.append({"b0": cursor, "occ": occ,
                     "scores": [float(sc[p]) for p in occ]})
        for c in [cursor] + occ:
            claimed[c:min(n, c + BLOCK)] = True
        cursor += BLOCK
    return runs, claimed


def labels_of(runs, n, vstart, block=BLOCK):
    """Une lettre par mesure ; -1 = pas expliqué."""
    lab = np.full(n, -1)
    for i, r in enumerate(runs):
        for c in [r["b0"]] + r["occ"]:
            lab[c:min(n, c + block)] = i
    lab[:vstart] = -1
    return lab


def truth_labels(secs, n):
    lab = np.full(n, -1)
    names = {}
    for s in secs:
        if s["label"] in ("intro", "outro", "queue"):
            continue
        names.setdefault(s["label"], len(names))
        lab[max(0, s["b0"]):min(n, s["b1"] + 1)] = names[s["label"]]
    return lab


def bounds(lab):
    return {int(i) for i in range(1, len(lab)) if lab[i] != lab[i - 1]}


def f1(pred, ref, tol=TOL):
    if not pred or not ref:
        return 0.0
    hit_p = sum(1 for p in pred if any(abs(p - r) <= tol for r in ref))
    hit_r = sum(1 for r in ref if any(abs(p - r) <= tol for p in pred))
    pr, rc = hit_p / len(pred), hit_r / len(ref)
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def pairwise(pred, ref):
    """Deux mesures de même lettre chez Louis le sont-elles chez nous ?"""
    ok = (ref >= 0) & (pred >= 0)
    idx = np.where(ok)[0]
    if len(idx) < 2:
        return 0.0
    P = pred[idx][:, None] == pred[idx][None, :]
    R = ref[idx][:, None] == ref[idx][None, :]
    iu = np.triu_indices(len(idx), 1)
    P, R = P[iu], R[iu]
    tp = float((P & R).sum())
    if not P.sum() or not R.sum():
        return 0.0
    pr, rc = tp / P.sum(), tp / R.sum()
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def main():
    T = truth()
    stems = sys.argv[1:] or [k for k, v in T.items() if v.get("sections")]
    grid = np.round(np.arange(0.30, 0.96, 0.02), 2)
    table = {}
    for st in stems:
        if not (HERE / f"docs/audio/{st}.m4a").exists():
            print(f"  !! {st} introuvable"); continue
        S, n, g = load(st)
        voc = VA.separate_vocals(HERE / f"docs/audio/{st}.m4a")
        onset, *_ = B8.sing_onset(voc)
        vstart = MS.voice_start(VA.bar_of(g, onset), n) or 0
        tt, ff, vv, rr = VM.track_f0(voc)
        notes, _ = VM.melody_notes(tt, ff, vv, rr)
        notes, _ = VM.clean(notes)
        M, mute = MS.melody_bars(notes, g, n)
        ref = truth_labels(T[st]["sections"], n)
        rb = bounds(ref)
        rows = []
        for thr in grid:
            runs, _ = anchor_runs(S, M, n, vstart, float(thr))
            pred = labels_of(runs, n, vstart)
            sc = [s for r in runs for s in r["scores"]]
            rows.append({
                "thr": float(thr),
                "louis": float(np.mean(sc)) if sc else 0.0,   # sa règle
                "nrep": len(sc),
                "fb": f1(bounds(pred), rb),
                "fp": pairwise(pred, ref),
            })
        table[st] = rows
        best_l = max(rows, key=lambda r: (r["louis"], r["nrep"]))
        best_t = max(rows, key=lambda r: r["fb"] + r["fp"])
        print(f"\n{st[:46]}  ({n} mesures)")
        print(f"   sa règle (moyenne max) → seuil {best_l['thr']:.2f} · "
              f"{best_l['nrep']} reprises · frontières {best_l['fb']:.2f} · "
              f"lettres {best_l['fp']:.2f}")
        print(f"   sa vérité   (accord max) → seuil {best_t['thr']:.2f} · "
              f"{best_t['nrep']} reprises · frontières {best_t['fb']:.2f} · "
              f"lettres {best_t['fp']:.2f}")

    print("\n" + "=" * 74)
    print("MOYENNE CORPUS — quel seuil unique marche le mieux ?")
    print(f"{'seuil':>6} {'reprises':>9} {'frontières':>11} {'lettres':>8} "
          f"{'règle de Louis':>15}")
    for i, thr in enumerate(grid):
        rs = [table[s][i] for s in table]
        print(f"{thr:>6.2f} {np.mean([r['nrep'] for r in rs]):>9.1f} "
              f"{np.mean([r['fb'] for r in rs]):>11.2f} "
              f"{np.mean([r['fp'] for r in rs]):>8.2f} "
              f"{np.mean([r['louis'] for r in rs]):>15.2f}")
    out = HERE / "harmonia_min/state/reports/score_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=1))
    print(f"\nécrit {out.relative_to(HERE)}")


if __name__ == "__main__":
    main()
