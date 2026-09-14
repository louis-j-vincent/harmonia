"""Une voie qui dit « tout se ressemble » ne dit rien : elle ne vote pas.

Louis, 2026-08-07 :

  « Sur Let It Be on détecte des sections en disant qu'elles sont similaires
    alors qu'elles ne le sont pas — à fixer, je te laisse généraliser à toutes
    les chansons. »

LE DÉFAUT, MESURÉ. On déclarait deux passages identiques quand leur similarité
harmonique dépassait 0.90, un nombre absolu, le même pour tous les morceaux.
Voici la similarité harmonique **médiane** entre deux passages quelconques de
huit mesures, morceau par morceau :

    Let It Be          0.906        The Walk           0.917
    This Love          0.624        Bein Green         0.486
    Norah              0.580        Grenade            0.616
    She Will Be Loved  0.555

Sur Let It Be et The Walk, **plus de la moitié de toutes les paires possibles
franchissent le seuil**. Ce n'est pas un réglage trop bas : ces deux morceaux
tournent sur les mêmes quatre accords du début à la fin, donc l'harmonie y est
littéralement incapable de distinguer un couplet d'un refrain. Elle répond
« pareil » à toutes les questions, et on prenait cette réponse pour une
information. D'où « A A A A A A A A » sur Let It Be.

CE QU'ON FAIT À LA PLACE. Deux corrections, qui ne sont pas des seuils réglés à
la main mais des conséquences de la mesure ci-dessus.

1. **Le seuil est relatif au morceau, pas absolu.** « Se ressembler » veut dire
   *être dans le haut du panier de CE morceau* — au-dessus du neuvième décile de
   toutes les paires de même longueur. Sur les morceaux où l'harmonie sépare
   bien, ce décile tombe sous l'ancien 0.90 et rien ne change ; sur Let It Be il
   monte à ~0.985 et le tri redevient sélectif.

2. **Une voie muette perd son droit de vote.** Le contraste d'une voie — de
   combien sa meilleure paire dépasse sa paire médiane — sépare les morceaux en
   deux familles sans aucun recouvrement :

       harmonie utile     0.374 … 0.510   (This Love, Bein Green, Norah, …)
       harmonie morte     0.082 … 0.093   (The Walk, Let It Be)

   Le trou entre 0.09 et 0.37 est vide : le seuil à 0.20 tombe au milieu de
   nulle part, il ne départage aucun cas limite. Sous ce contraste, la voie
   s'abstient complètement — mieux vaut ne rien dire que dire « pareil » de tout.

Ce qui reste alors sur Let It Be, c'est le chant, et le chant a la réponse : les
blocs chantés se répondent tous entre 0.63 et 0.84, le solo instrumental des
mesures 29 à 44 tombe à 0.11–0.36. La voie qui sait parle, celle qui ne sait pas
se tait. Sur The Walk, où les deux voies sont faibles, c'est le chant qui isole
le second groupe {25, 81} — la substitution du C#m7 de la vérité de Louis.

CE QUE ÇA NE RÉGLE PAS. Sur Let It Be le chant regroupe tous les passages
chantés en une seule famille : il ne sépare pas le couplet du refrain, il sépare
le chanté de l'instrumental. C'est moins faux qu'avant, ce n'est pas encore la
structure. Et si les DEUX voies s'abstiennent sur une paire, on ne fusionne
pas — deux lettres différentes veulent alors dire « on ne sait pas », pas
« différent ».
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
import harmonia_min.harmonic_sections as HS       # noqa: E402

DEAD = 0.20        # contraste sous lequel une voie s'abstient (trou mesuré : 0.09→0.37)
Q = 0.90           # « se ressembler » = neuvième décile des paires du morceau
FLOOR_H = 0.90     # l'ancien seuil absolu reste un plancher : jamais moins exigeant
FLOOR_M = 0.40
MUTE_MAX = 0.50    # au-delà, le chant n'a rien entendu et s'abstient sur la paire
STEP = 2


def pair_values(X, n, L, step=STEP):
    """Le fond du morceau : toutes les paires de passages de longueur L."""
    out = []
    for a in range(0, n - L + 1, step):
        for b in range(a + L, n - L + 1, step):
            out.append(HS.diag_match(X, a, b, L))
    return np.asarray(out, float)


def contrast(vals):
    """De combien la meilleure paire dépasse la paire médiane."""
    if vals.size == 0:
        return 0.0
    return float(np.max(vals) - np.median(vals))


class Voice:
    """Une voie de similarité et son droit de vote sur CE morceau."""

    def __init__(self, name, X, n, L, floor, mute=None, dead=DEAD, q=Q):
        self.name, self.X, self.mute = name, X, mute
        vals = pair_values(X, n, L)
        self.background = vals
        self.contrast = contrast(vals)
        self.alive = self.contrast >= dead
        self.thr = max(floor, float(np.quantile(vals, q))) if vals.size else 1.0

    def says(self, a, b, L):
        """« pareil » / « différent » / None quand la voie n'a pas d'avis."""
        if not self.alive:
            return None
        if self.mute is not None:
            share = float(np.mean([self.mute[a + k] or self.mute[b + k]
                                   for k in range(L)
                                   if a + k < len(self.mute) and b + k < len(self.mute)]))
            if share > MUTE_MAX:
                return None
        return HS.diag_match(self.X, a, b, L) >= self.thr

    def __repr__(self):
        return (f"<{self.name} {'vote' if self.alive else 'muette'} "
                f"contraste={self.contrast:.3f} seuil={self.thr:.3f}>")


def voices(S, M, n, L, mute=None):
    """Les deux voies du morceau, avec leur droit de vote déjà tranché."""
    return [Voice("harmonie", S, n, L, FLOOR_H),
            Voice("chant", M, n, L, FLOOR_M, mute=mute)]


def same_pair(vs, a, b, L):
    """La première voie qui SAIT tranche. Priorité, pas vote à la majorité.

    Essayé d'abord en OU — « pareil » dès qu'une voie le dit — et mesuré faux :
    Grenade passait de `A A B C′ A A B C′ D B C′`, une vraie structure, à
    `A A A A A A A A A A A`, et She Will Be Loved pareil. Le chant a une queue de
    distribution longue ; son neuvième décile tombe vers 0.54, et beaucoup de
    paires sans rapport le franchissent. En OU il noyait une harmonie qui, sur
    ces morceaux-là, avait la bonne réponse.

    Ce que la mesure autorise est plus étroit et c'est tout ce qu'on fait :
    l'harmonie décide quand elle discrimine, le chant prend le relais **là où
    l'harmonie est morte** — Let It Be, The Walk. La voie qui sait parle ; celle
    qui ne sait pas se tait ; on ne les additionne pas.
    """
    for v in vs:
        said = v.says(a, b, L)
        if said is not None:
            return said
    return False


def group(vs, starts, lens):
    """Regroupe les blocs par proche-en-proche (lien simple).

    J'ai d'abord mis un **lien complet** ici — deux groupes ne fusionnent que si
    toutes leurs paires croisées se ressemblent — pour empêcher le chaînage : A
    rejoint B, B rejoint C pour une autre raison, et C se retrouve avec A sans
    que personne ne les ait comparés. C'était une parade au OU de `same_pair`,
    et le OU a disparu ; la parade coûtait plus qu'elle ne rapportait. Mesuré :

        The Walk           lien simple 2 lettres (la vérité de Louis)
                           lien complet 5 lettres
        She Will Be Loved  lien simple 4 lettres, identique à l'ancien code
                           lien complet 6 lettres

    Avec une voie unique et un seuil déjà propre au morceau, le chaînage ne se
    produit plus ; le lien complet ne faisait qu'émietter. Gardé en note pour ne
    pas y revenir.
    """
    k = len(starts)
    parent = list(range(k))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(k):
        for j in range(i + 1, k):
            L = min(lens[i], lens[j])
            if L >= 2 and same_pair(vs, starts[i], starts[j], L):
                parent[find(i)] = find(j)
    return [find(i) for i in range(k)]


if __name__ == "__main__":            # caractérisation, pas un test
    sys.path.insert(0, str(HERE / "scripts"))
    from pattern_lanes import load
    import vocal_anchor as VA, blocks8 as B8, melody_ssm as MS, vocal_melody as VM
    for st in (sys.argv[1:] or B8.DEFAULT):
        if not (HERE / f"docs/audio/{st}.m4a").exists():
            continue
        S, n, grid = load(st)
        voc = VA.separate_vocals(HERE / f"docs/audio/{st}.m4a")
        tt, ff, vv, rr = VM.track_f0(voc)
        notes, _ = VM.melody_notes(tt, ff, vv, rr)
        notes, _ = VM.clean(notes)
        M, mute = MS.melody_bars(notes, grid, n)
        vs = voices(S, M, n, 8, mute)
        print(f"{st[:34]:<36} " + "  ".join(str(v) for v in vs))
