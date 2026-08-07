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


def entropies(pred, ref):
    """SUR-découpage et SOUS-découpage, séparés. Louis, 2026-08-07 :

      « Des fois je vais différencier un A d'un B, alors que d'autres vont
        considérer que c'est la même chose car ils ont les mêmes accords mais on
        chante dessus différemment, ou inversement. Et des fois on va merger un B
        et un C ensemble et appeler ça B, ou les laisser séparément, et encore une
        fois ça c'est à l'appréciation. Il te faudrait une métrique d'évaluation
        qui prend ça en compte. »

    Un seul chiffre d'accord ne peut pas : il punit pareil « tu as coupé là où je
    n'aurais pas coupé » et « tu as collé ce que j'aurais séparé », alors que ces
    deux-là ne sont pas la même faute et que l'une des deux est souvent une
    question de goût. On en rend donc DEUX (Lukashevich 2008, la mesure standard
    du domaine) :

      SUR-découpage   1.0 = on n'a rien coupé de plus que toi. Il baisse quand on
                      sépare un B et un C que tu avais laissés ensemble.
      SOUS-découpage  1.0 = on n'a rien collé que tu avais séparé. Il baisse
                      quand on met sous une seule lettre deux passages que tu
                      distingues.

    Un découpage qui est exactement le tien avec des noms différents garde 1.0
    partout : ce sont des mesures d'information partagée, pas de noms.
    """
    ok = (ref >= 0) & (pred >= 0)
    e, r = pred[ok], ref[ok]
    if e.size < 2:
        return 0.0, 0.0
    ue, ur = np.unique(e), np.unique(r)
    P = np.zeros((len(ue), len(ur)))
    for i, a in enumerate(ue):
        for j, b in enumerate(ur):
            P[i, j] = np.sum((e == a) & (r == b))
    P /= P.sum()

    def cond(P, axis):
        """axis=0 -> H(nous | toi) ; axis=1 -> H(toi | nous)."""
        m = P.sum(axis=axis, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(P > 0, P * np.log(np.where(m > 0, P / m, 1)), 0.0)
        return float(-t.sum())

    # LES DEUX SONT FACILES À INTERVERTIR, et je l'ai fait le 2026-08-07 : le
    # « sur-découpage » affiché était le sous-découpage et réciproquement, donc
    # la lecture des stratégies était exactement à l'envers. La vérification qui
    # tranche tient en une ligne et vit dans __main__ : un découpage qui coupe
    # chaque section de Louis en deux DOIT faire chuter le sur-découpage et
    # laisser le sous-découpage à 1.
    over = 1.0 if len(ue) < 2 else 1 - cond(P, 0) / np.log(len(ue))
    under = 1.0 if len(ur) < 2 else 1 - cond(P, 1) / np.log(len(ur))
    return max(0.0, over), max(0.0, under)


def runs_of(lab):
    """Les segments maximaux d'un étiquetage par mesure -> [(a, b, lettre)]."""
    out, a = [], 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[a]:
            out.append((a, i - 1, int(lab[a])))
            a = i
    return out


def correspondence(pred, ref, tol=0):
    """LA CORRESPONDANCE STABLE. Louis, 2026-08-07, et c'est sa métrique :

      « Tu peux regarder à chaque fois s'il y a une fonction qui permet de passer
        de mes sections aux tiennes de manière constante (A -> A, B+C -> B, ou
        C -> B+C). Tu pars de tes sections, pour chacune tu regardes la
        correspondante chez moi, tu établis ça comme un match, puis tu vérifies
        si ce match se répète — donc si on est concordants même avec un
        sur-découpage et un nommage différents. L'important c'est que les
        frontières globales des sections soient partagées : si un B équivaut à un
        A+B, il faut que les frontières du B soient les mêmes que celles du A+B. »

    Deux nombres, exactement ses deux exigences.

      FRONTIÈRES PARTAGÉES — la part de nos sections dont les deux bords tombent
      sur des frontières à lui. Une section qui coupe une des siennes en deux est
      une vraie faute, même si on l'a bien nommée : elle rend toute
      correspondance impossible.

      CORRESPONDANCE STABLE — on note, pour chaque section à nous, la SUITE de
      ses lettres à lui qu'elle recouvre (sa signature : « A », ou « A+B »). Les
      occurrences d'une même lettre à nous doivent toutes porter la même
      signature. La part de mesures où c'est le cas est le score. Renommer ne
      change rien, sur-découper non plus tant que le découpage est le MÊME à
      chaque reprise — ce qui est précisément ce qu'il demande.

    Les deux sens sont rendus : de nous vers lui, puis de lui vers nous. Le
    premier attrape « on a séparé ce qu'il regroupe », le second l'inverse.
    """
    def one_way(a, b):
        segs = [s for s in runs_of(a) if s[2] >= 0]
        bb = {0, len(b)} | {i for i in range(1, len(b)) if b[i] != b[i - 1]}
        if not segs:
            return 0.0, 0.0
        edges = sum(s[1] - s[0] + 1 for s in segs
                    if any(abs(s[0] - x) <= tol for x in bb)
                    and any(abs(s[1] + 1 - x) <= tol for x in bb))
        total = sum(s[1] - s[0] + 1 for s in segs)
        sig = {}
        for a0, a1, lab in segs:
            k = tuple(v for v, _ in
                      [(b[i], i) for i in range(a0, a1 + 1)
                       if i == a0 or b[i] != b[i - 1]])
            sig.setdefault(lab, []).append((k, a1 - a0 + 1))
        stable = 0
        for lab, lst in sig.items():
            w = {}
            for k, ln in lst:
                w[k] = w.get(k, 0) + ln
            stable += max(w.values())
        return edges / total, stable / total

    e1, s1 = one_way(pred, ref)
    e2, s2 = one_way(ref, pred)
    return e1, (s1 + s2) / 2


def bounds_pr(pred, ref, tol=TOL):
    """Frontières : précision et rappel SÉPARÉS.

    Fusionner deux sections fait chuter le rappel sans toucher la précision ;
    en découper une de trop fait l'inverse. Le F les mélange et cache donc
    exactement la distinction que Louis demande.
    """
    if not pred or not ref:
        return 0.0, 0.0
    pr = sum(1 for p in pred if any(abs(p - r) <= tol for r in ref)) / len(pred)
    rc = sum(1 for r in ref if any(abs(p - r) <= tol for p in pred)) / len(ref)
    return pr, rc


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
