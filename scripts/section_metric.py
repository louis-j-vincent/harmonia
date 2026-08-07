"""La distance entre deux découpages en sections. Celle que Louis a spécifiée.

    from section_metric import compare
    r = compare(nos_sections, ses_sections, n)   ->  r["score"] dans [0, 1]

Louis, 2026-08-07 au soir :

  « Vérifie que ta métrique de correction d'annotation est bonne : distance
    entre 2 annotations faite avec l'appairage des sections, puis vérification
    que chaque section couvre bien le même espace que son équivalent dans
    l'autre grille ; on perd des points quand il y a un désalignement, mais
    tenir compte des queues, qui peuvent parfois être considérées comme la même
    section, parfois une section différente. »

Quatre exigences, et chacune tue une métrique plus simple.

  APPAIRAGE — comparer les noms directement ne veut rien dire : son A peut être
  notre B. On apparie donc, dans les deux sens, et on ne compare qu'ensuite.

  MÊME ESPACE — un accord sur les lettres seul est aveugle au cas le plus
  fréquent chez nous : ses deux A de 8 mesures écrits comme un seul A de 16.
  Mesure par mesure les lettres sont identiques, et pourtant ce n'est pas le
  même découpage. Il faut donc apparier les SEGMENTS, pas seulement les lettres.

  DÉSALIGNEMENT — une frontière posée une mesure trop loin doit coûter, et
  coûter proportionnellement au nombre de mesures qu'elle déplace.

  QUEUES — les deux ou trois mesures de fin d'une section (le turnaround, la
  reprise instrumentale) sont parfois une section à part, parfois collées à la
  précédente, et les deux lectures sont défendables. Ce désaccord-là ne doit pas
  coûter le même prix qu'une vraie erreur.

CE QUI SÉPARE UNE QUEUE D'UN DÉCALAGE, et c'est le seul point délicat du
fichier. Les deux se ressemblent : quelques mesures attribuées ailleurs, contre
une frontière. La différence est ce qu'il y a en face.

  * QUEUE : les mesures contestées tombent dans un segment de l'autre grille que
    PERSONNE n'a apparié — une section en trop, créée par un seul des deux
    annotateurs. C'est le désaccord de goût. On la fait payer `TAIL_W`.
  * DÉCALAGE : elles tombent dans le VOISIN, qui a lui-même son partenaire.
    Les deux grilles ont les mêmes sections, la frontière n'est pas au même
    endroit. Plein tarif.

Sans cette distinction, tout décalage d'une mesure passerait pour une queue et
la métrique ne verrait plus les erreurs de frontière — exactement ce que Louis
demande de mesurer.

DEUX NOMBRES, et le score est leur moyenne harmonique (pas arithmétique : une
métrique de confiance ne doit pas pouvoir se rattraper sur l'autre moitié).

  spans    l'appairage des SEGMENTS. Voit les longueurs, les frontières, les
           fusions de reprises adjacentes.
  letters  l'appairage des LETTRES, constant sur tout le morceau. Voit
           « tu as appelé B ce que j'appelle B et C », et est insensible au
           renommage.

Les deux sont rendus dans les deux sens (`_p2r` = de nous vers lui, `_r2p` =
l'inverse) parce que sur-découper et sous-découper ne sont pas la même faute :
le premier fait chuter le sens nous→lui, le second l'autre.

VÉRIFIÉ PAR `tests/test_section_metric.py`, et il faut que ça le reste : les
deux sens sont triviaux à intervertir et la faute est silencieuse — elle est
déjà arrivée sur les entropies le 2026-08-07, où toute une table de comparaison
se lisait à l'envers pendant une soirée.
"""
from __future__ import annotations

TAIL = 2        # une queue fait au plus deux mesures (la grille des sections
                # est en 2 : `UNIT = 2` dans voice_sections)
TAIL_W = 0.35   # ce qu'on fait payer une queue contestée, contre 1.0 une erreur


# ── représentations ─────────────────────────────────────────────────────────

def segs(sections, n):
    """[{b0,b1,label}] -> [(b0, b1, label)] borné à n, dans l'ordre.

    On garde les segments TELS QU'ILS SONT ÉCRITS : deux A adjacents restent
    deux A. Les fusionner effacerait la reprise, qui est justement l'objet.
    """
    out = []
    for s in sections:
        b0, b1 = max(0, int(s["b0"])), min(n - 1, int(s["b1"]))
        if b1 >= b0:
            out.append((b0, b1, str(s["label"])))
    return sorted(out)


def bars(sections, n):
    """Une étiquette par mesure ; None si la grille ne couvre pas la mesure."""
    lab = [None] * n
    for b0, b1, l in segs(sections, n):
        for i in range(b0, b1 + 1):
            lab[i] = l
    return lab


def _runs(idx):
    """[3,4,7,8,9] -> [[3,4],[7,8,9]]."""
    out = []
    for i in sorted(idx):
        if out and i == out[-1][-1] + 1:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def _edges(sections, n):
    """Les frontières d'une grille, 0 et n compris."""
    e = {0, n}
    for b0, b1, _ in segs(sections, n):
        e.add(b0)
        e.add(b1 + 1)
    return e


# ── l'appairage des segments ────────────────────────────────────────────────

def _best(A, B):
    """Pour chaque segment de A, l'indice du segment de B qui le recouvre le
    plus (−1 si aucun ne le touche)."""
    out = []
    for a0, a1, _ in A:
        k, ov = -1, 0
        for j, (b0, b1, _) in enumerate(B):
            o = min(a1, b1) - max(a0, b0) + 1
            if o > ov:
                ov, k = o, j
        out.append(k)
    return out


def _span_one(A, B, n, tail=TAIL, tail_w=TAIL_W, systematic=frozenset()):
    """Sens A -> B : la part des mesures de A que son partenaire couvre aussi.

    C'est la distance de Hamming dirigée (Abdallah 2005), plus les deux remises.

    `systematic` = les lettres de B qui sont un sur-découpage CONSTANT (au moins
    deux occurrences, toutes de même nature). Sans cette entrée, découper chacun
    de ses B en deux de la MÊME façon coûterait autant que le découper une seule
    fois — la métrique dirait que la cohérence ne sert à rien, alors que c'est
    exactement ce que Louis demande de récompenser.
    """
    if not A or not B:
        return 0.0, 0
    best = _best(A, B)
    paired = {k for k in best if k >= 0}
    cost, forgiven = 0.0, 0
    for (a0, a1, _), k in zip(A, best):
        if k < 0:
            cost += a1 - a0 + 1
            continue
        b0, b1, _ = B[k]
        miss = [i for i in range(a0, a1 + 1) if not (b0 <= i <= b1)]
        for r in _runs(miss):
            # dans quels segments de B ces mesures tombent-elles ?
            owners = {j for j, (c0, c1, _) in enumerate(B)
                      if not (c1 < r[0] or c0 > r[-1])}
            orphan = owners.isdisjoint(paired)          # une section en trop
            flush = r[0] == a0 or r[-1] == a1           # collée à un bord
            # UN SOUS-DÉCOUPAGE CONSTANT, pas une reprise oubliée. Les deux se
            # ressemblent — des mesures qui débordent dans un segment que
            # personne n'a apparié — et il faut absolument les séparer :
            #   « son B écrit B puis C »  : l'orphelin porte un AUTRE nom, c'est
            #     la convention d'écriture que Louis dit être à l'appréciation.
            #   « ses deux A écrits un seul A » : l'orphelin porte le MÊME nom,
            #     c'est une reprise qu'on n'a pas su séparer, et c'est notre
            #     défaut le plus fréquent. Plein tarif.
            const = bool(owners) and all(
                B[j][2] in systematic and B[j][2] != B[k][2] for j in owners)
            if orphan and flush and (len(r) <= tail or const):
                cost += len(r) * tail_w
                forgiven += len(r)
            else:
                cost += len(r)
    return max(0.0, 1.0 - cost / n), forgiven


# ── l'appairage des lettres ─────────────────────────────────────────────────

def _sig(a0, a1, other, tail=TAIL):
    """La SIGNATURE d'un segment : la suite des lettres de l'autre grille qu'il
    recouvre, les échardes de bord retirées.

    C'est ce qui autorise le sur-découpage constant que Louis demande : notre B
    qui recouvre toujours son « B puis C » a pour signature ('B','C') à chaque
    reprise, donc il est stable, donc il ne coûte rien. Il ne devient une faute
    que s'il recouvre ('B','C') ici et ('B',) là.

    Les échardes : deux mesures qui dépassent d'un bord ne changent pas la
    signature. Sans ça, une frontière posée une mesure trop loin serait comptée
    deux fois — une fois comme désalignement (`spans`, à raison) et une fois
    comme changement de nature (ici, à tort).
    """
    parts = []
    for i in range(a0, a1 + 1):
        # « ∅ » = l'autre grille ne dit rien de cette mesure. Ça arrive quand
        # les deux découpages ne couvrent pas exactement le même nombre de
        # mesures ; c'est une lettre comme une autre, jamais une absence.
        l = other[i] if other[i] is not None else "∅"
        if parts and parts[-1][0] == l:
            parts[-1][1] += 1
        else:
            parts.append([l, 1])
    # UNE écharde par bord, pas plus : en boucle, un segment entièrement fait de
    # tranches courtes se ferait raboter jusqu'à une seule lettre choisie au
    # hasard, et un découpage absurde (une lettre toutes les deux mesures)
    # deviendrait « stable ». Une queue, c'est un bord, pas tout le segment.
    if len(parts) > 1 and parts[0][1] <= tail:
        parts.pop(0)
    if len(parts) > 1 and parts[-1][1] <= tail:
        parts.pop()
    return tuple(p[0] for p in parts)


def _letter_one(A, other, n, tail=TAIL):
    """Sens A -> autre : la part des mesures où la correspondance est CONSTANTE.

    Louis, 2026-08-07 : « tu peux regarder à chaque fois s'il y a une fonction
    qui permet de passer de mes sections aux tiennes de manière constante
    (A -> A, B+C -> B) ». Une lettre a droit à une signature, et une seule :
    les reprises qui n'ont pas la signature majoritaire sont ce qu'on perd.

    Renommer ne coûte rien. Sur-découper ne coûte rien tant qu'on sur-découpe
    PAREIL à chaque reprise. Appeler C ici ce qu'on appelle A là coûte les
    mesures de la minorité — et c'est l'autre sens qui l'attrape.
    """
    if not A:
        return 0.0, {}
    sig = {}
    for a0, a1, l in A:
        k = _sig(a0, a1, other, tail)
        sig.setdefault(l, {})
        sig[l][k] = sig[l].get(k, 0) + (a1 - a0 + 1)
    stable = sum(max(d.values()) for d in sig.values())
    total = sum(a1 - a0 + 1 for a0, a1, _ in A)
    keep = {l: max(d, key=lambda k: d[k]) for l, d in sig.items()}
    return (stable / total if total else 0.0), keep


def _systematic(A, other, n, tail=TAIL):
    """Les lettres de A qui sont un sur-découpage CONSTANT.

    Deux occurrences au moins, toutes de même signature. Une occurrence unique
    ne compte pas : elle est « stable » par défaut, faute de contradicteur, et
    l'accepter reviendrait à excuser n'importe quelle section inventée.
    """
    sig = {}
    for a0, a1, l in A:
        sig.setdefault(l, []).append(_sig(a0, a1, other, tail))
    return {l for l, ks in sig.items() if len(ks) >= 2 and len(set(ks)) == 1}


# ── le pair-à-pair, garde-fou ───────────────────────────────────────────────

def _pairwise(pred, ref, n):
    """Deux mesures de même lettre chez lui le sont-elles chez nous ?

    Ce n'est pas dans sa spécification ; c'est le garde-fou. L'appairage par
    argmax a une faille connue : tout mettre sous une seule lettre rend le sens
    lui→nous parfait. Le pair-à-pair, lui, s'effondre dans ce cas (rappel 1,
    précision au plancher) comme dans son symétrique (une lettre par mesure).
    On le calcule donc toujours, et le test `test_degenere` s'appuie dessus.
    """
    idx = [i for i in range(n) if pred[i] is not None and ref[i] is not None]
    if len(idx) < 2:
        return 0.0
    tp = fp = fn = 0
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            p, r = pred[i] == pred[j], ref[i] == ref[j]
            tp += p and r
            fp += p and not r
            fn += r and not p
    if not tp:
        return 0.0
    pr, rc = tp / (tp + fp), tp / (tp + fn)
    return 2 * pr * rc / (pr + rc)


# ── la mesure ───────────────────────────────────────────────────────────────

def _h(a, b):
    """Moyenne harmonique. Nulle dès que l'un des deux est nul."""
    return 2 * a * b / (a + b) if a + b > 0 else 0.0


def compare(pred, ref, n, tail=TAIL, tail_w=TAIL_W):
    """Notre découpage contre le sien. Tout est dans [0, 1], 1 = identique.

    LE SCORE EST UN PRODUIT, `spans × letters`, et pas leur moyenne. Une moyenne
    laisse une moitié racheter l'autre : mettre tout le morceau sous une seule
    lettre donne des lettres parfaitement « constantes » (une lettre, une
    signature, rien à contredire) et une moyenne de 0,50 pour un découpage qui
    ne dit rien. Le produit le laisse à 0,33, et il faut que les deux tiennent
    pour qu'il monte. C'est vérifié par `test_degenere`.

    Renvoie `score`, ses deux moitiés, les quatre sens séparément, le garde-fou
    `pairwise`, la correspondance trouvée et le nombre de mesures excusées.
    """
    A, B = segs(pred, n), segs(ref, n)
    lp, lb = bars(pred, n), bars(ref, n)
    sys_p, sys_r = _systematic(A, lb, n, tail), _systematic(B, lp, n, tail)
    sp, sf = _span_one(A, B, n, tail, tail_w, sys_r)
    sr, srf = _span_one(B, A, n, tail, tail_w, sys_p)
    tp_, mp = _letter_one(A, lb, n, tail)
    tr_, mr = _letter_one(B, lp, n, tail)
    spans, letters = _h(sp, sr), _h(tp_, tr_)
    return {
        "score": spans * letters,
        "spans": spans, "letters": letters,
        "span_p2r": sp, "span_r2p": sr,
        "letter_p2r": tp_, "letter_r2p": tr_,
        "pairwise": _pairwise(lp, lb, n),
        "mapping": {k: "+".join(v) for k, v in mp.items()},
        "forgiven": sf + srf,
        "n_pred": len(A), "n_ref": len(B), "n": n,
    }
