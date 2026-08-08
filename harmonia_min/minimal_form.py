"""harmonia_min/minimal_form.py — la FORME MINIMALE d'une section.

Louis, 2026-08-08 : « pour chaque section, utilise l'analyse en 2 ou 4 mesures
pour voir quelle est la forme minimale — 4 mesures qui bouclent, 4 mesures qui
bouclent avec les 2 dernières qui varient, 8 mesures qui bouclent avec les 2
dernières qui varient… La règle d'or : la représentation doit être la plus
compacte possible, MAIS toutes les variations doivent être représentées. Si on
affiche deux fois les 8 mêmes mesures, c'est un loupé. »

C'est une COMPRESSION SANS PERTE. Deux propriétés, toutes deux testées dans
`tests/test_minimal_form.py`, et la seconde n'est pas négociable :

  (1) compacité — aucune suite de mesures n'est écrite deux fois ;
  (2) reconstruction — `reconstruct()` rend exactement la suite d'accords
      d'origine, mesure par mesure.

Sans (2), « compact » finit par vouloir dire « faux » : c'est exactement ce qui
est arrivé au repli livré, qui a écrit 4 mesures × 3 pour un A de Norah Jones
qui joue 22 / 8 / 20 mesures, dont 13 jamais rendues (docs/known_issues.md,
`folding.minimal_fold`).

────────────────────────────────────────────────────────────────────────────
LE MODÈLE
────────────────────────────────────────────────────────────────────────────
Une lettre (A, B, refrain…) a des OCCURRENCES : les passages où elle joue.
Chaque occurrence est une suite de mesures, découpée en REPRISES d'une CELLULE
de `period` mesures. Une reprise est la cellule, éventuellement corrigée par des
RETOUCHES : des tranches de mesures qui remplacent celles de la cellule. Une
retouche qui va jusqu'au bout de la cellule est une TERMINAISON — la « 2ᵉ fois »
d'un musicien, le cas que Louis cite. Ce qui n'entre dans aucune reprise est
écrit LITTÉRALEMENT.

    Bloc(period=4, cellule=[Bb7 | Eb^7 D | G-7 C7 | F7 Bb],
         retouches=[R(pos 3, [F7sus4]), R(pos 1, [Eb D])],
         reprises=[…10 reprises, chacune avec ses retouches…])

Écrit : 4 + 1 + 1 = 6 mesures. Joué : 40. Compression 0,15.

Une retouche NON terminale (au milieu de la cellule) n'était pas dans l'énoncé
de Louis, mais il l'exige sans le dire : « toutes les variations doivent être
représentées ». Les variations réelles ne tombent pas toutes en fin de section
— sur ce corpus, 54 % d'entre elles tombent ailleurs. Un modèle qui ne sait
écrire que des queues doit écrire tout le reste au long, et ne comprime plus.

────────────────────────────────────────────────────────────────────────────
CE QUI DÉCIDE, ET POURQUOI IL N'Y A PAS DE SEUIL
────────────────────────────────────────────────────────────────────────────
Une occurrence rejoint un bloc si, et seulement si, la rejoindre écrit MOINS de
mesures que l'écrire au long. Rien d'autre. Une occurrence qui ne ressemble pas
à la cellule a besoin d'autant de retouches qu'elle a de mesures : le test la
refuse tout seul, et elle repart dans son propre bloc — écrite à la longueur
qu'elle joue vraiment.

C'est la règle « under-fold, never over-fold » de Louis (2026-07-30) obtenue
sans constante à régler, et c'est aussi l'arbitrage qu'il demandait d'expliciter
contre « le plus compact possible ». Le repli livré tranchait en SÉPARANT les
occurrences de longueurs différentes, parce que son bloc était étiré
proportionnellement : 8 mesures écrites posées sur une occurrence de 4 faisaient
courir le curseur à double vitesse (`tests/test_section_fold_unequal.py`). Ici
la question ne se pose plus : une reprise fait exactement `period` mesures de
temps réel et la carte reprise → mesures est exacte, donc une occurrence de 4
mesures est UNE reprise et une de 8 en fait DEUX. Rien n'est étiré. Deux
occurrences de longueurs différentes partagent donc une cellule sans qu'aucune
ne soit déformée : on gagne la compacité sans payer le défaut que la règle
protégeait. Ce qui reste séparé, et doit l'être, c'est ce qui diverge par le
CONTENU, pas par la longueur.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

# Périodes candidates, en mesures. 3/6/12 pour les formes ternaires et le blues.
PERIODS = (1, 2, 3, 4, 6, 8, 12, 16)

# Une cellule n'est retenue que si elle est jouée au moins deux fois : écrire
# une « boucle » vue une seule fois n'économise rien et invente une régularité.
MIN_RENDITIONS = 2

# Un bloc est une FORME, pas un dictionnaire : la cellule écrite doit rendre à
# elle seule PLUS DE LA MOITIÉ de ce que le bloc fait entendre. C'est la seule
# contrainte qui ne vienne pas du coût, et sans elle le coût seul dégénère.
#
# Mesuré (2026-08-08, les 27 morceaux) : le A de Bein Green, une section de 8
# mesures jouée 4 fois, sortait en cellule d'UNE mesure (Bb^7) suivie de quinze
# variantes d'une mesure. 16 mesures écrites pour 32 jouées — moins cher que la
# lecture en 8 mesures (18/32), et pourtant ce n'est pas une forme : ni période,
# ni reprise, rien que Louis puisse rejouer de tête. Sa cellule ne rend que 5
# des 32 mesures ; la règle la refuse. La lecture en 8 mesures en rend 21, elle
# passe. Idem pour le A de Goodbye Yellow Brick Road.
#
# Deux formulations plus faibles ont été essayées et écartées (mesuré) :
#   * plafonner la part retouchée reprise par reprise — une seule passe bruitée
#     éjectait toute l'occurrence, écrite au long ensuite : 0,374 → 0,481 ;
#   * plafonner la part retouchée en moyenne — la dégénérescence revenait par
#     les littéraux (53 blocs d'une mesure), le dictionnaire déguisé.
# Compter ce que la cellule REND, littéraux inclus, ferme les deux portes.
CELL_SHARE = 0.5

# Départage à compacité ÉGALE seulement : une boucle de 4 mesures se lit mieux
# qu'une de 3, et une de 8 mieux qu'une de 6. N'influence jamais le choix quand
# les coûts diffèrent.
_PERIOD_RANK = {4: 0, 8: 0, 2: 0, 16: 0, 12: 1, 6: 1, 3: 1, 1: 2}


# ── types ───────────────────────────────────────────────────────────────────

Bar = tuple            # signature d'une mesure (voir bar_signature)


@dataclass(frozen=True)
class Patch:
    """Les mesures [pos, pos+len) de la cellule, remplacées."""
    pos: int
    bars: tuple[Bar, ...]

    def is_ending(self, period: int) -> bool:
        """Une retouche qui va jusqu'au bout : la « 2ᵉ fois » du musicien."""
        return self.pos + len(self.bars) == period


@dataclass(frozen=True)
class Rendition:
    """Une reprise de la cellule, posée sur `period` mesures à partir de b0."""
    b0: int                        # décalage DANS l'occurrence
    occ: int                       # index de l'occurrence
    patches: tuple[int, ...] = ()  # indices dans Block.patches


@dataclass(frozen=True)
class Literal:
    """Des mesures écrites telles quelles (ce qui n'entre dans aucune boucle)."""
    b0: int
    occ: int
    bars: tuple[Bar, ...]


@dataclass
class Block:
    """Une cellule, ses retouches, et toutes les reprises qu'elle explique."""
    period: int
    cell: tuple[Bar, ...]
    patches: list[Patch] = field(default_factory=list)
    renditions: list[Rendition] = field(default_factory=list)
    literals: list[Literal] = field(default_factory=list)
    alias_of: str | None = None    # « identique au bloc X » (dédup inter-lettres)

    @property
    def written(self) -> int:
        """Les mesures que la page dessine pour ce bloc.

        Un bloc en ALIAS ne redessine pas sa cellule — la page renvoie à celle
        du bloc jumeau (« A² = A¹, autre fin ») — mais ses retouches, elles,
        sont bien écrites : c'est par elles qu'il en diffère.
        """
        n = sum(len(p.bars) for p in self.patches)
        if self.alias_of is None:
            n += self.period
        # Un littéral coûte ses mesures À CHAQUE FOIS. Il est posé à un endroit
        # précis d'une occurrence, donc la page le redessine à chaque endroit —
        # le compter une fois mentirait sur ce qu'elle montre, et surtout ça
        # rendait GRATUIT un mauvais choix de période. Mesuré : Let It Be
        # sortait en boucle de 2 mesures avec `G | A- F` écrit HUIT fois (le
        # couplet est une boucle de 4) ; Stand By Me en boucle de 6 avec
        # `A | A~` écrit neuf fois (la boucle fait 8).
        return n + sum(len(lit.bars) for lit in self.literals)

    @property
    def played(self) -> int:
        return (len(self.renditions) * self.period
                + sum(len(l.bars) for l in self.literals))

    @property
    def start(self) -> tuple[int, int]:
        pts = [(r.occ, r.b0) for r in self.renditions]
        pts += [(l.occ, l.b0) for l in self.literals]
        return min(pts) if pts else (0, 0)

    def units(self) -> list[tuple[str, tuple[Bar, ...]]]:
        """Les suites de mesures que ce bloc pose sur la page, sans doublon."""
        out: list[tuple[str, tuple[Bar, ...]]] = []
        if self.period and self.alias_of is None:
            out.append(("cell", self.cell))
        for p in self.patches:
            out.append(("ending" if p.is_ending(self.period) else "patch",
                        p.bars))
        seen = set()
        for lit in self.literals:
            if lit.bars not in seen:
                seen.add(lit.bars)
                out.append(("literal", lit.bars))
        return out


@dataclass
class LetterForm:
    """La forme minimale d'une lettre : un ou plusieurs blocs."""
    label: str
    blocks: list[Block]

    @property
    def written(self) -> int:
        return sum(b.written for b in self.blocks)

    @property
    def played(self) -> int:
        return sum(b.played for b in self.blocks)

    def name(self, i: int) -> str:
        """A, ou A¹ A² … quand la lettre a dû se scinder (under-fold)."""
        if len(self.blocks) == 1:
            return self.label
        return self.label + "¹²³⁴⁵⁶⁷⁸⁹"[min(i, 8)]


@dataclass
class SongForm:
    letters: list[LetterForm]
    aliases: list[tuple[str, str]] = field(default_factory=list)

    @property
    def written(self) -> int:
        return sum(l.written for l in self.letters)

    @property
    def played(self) -> int:
        return sum(l.played for l in self.letters)

    @property
    def ratio(self) -> float:
        return self.written / self.played if self.played else 1.0


# ── signature d'une mesure ──────────────────────────────────────────────────

def bar_signature(bar, loose: bool = False) -> Bar:
    """Ce qu'un musicien LIT dans une mesure : quel accord, sur quel temps.

    Hors signature dans les deux modes : `c` (confiance), `n_obs`, `colour`,
    `sug`, `t0`/`t1`, `bar`, `pickup` — de la métadonnée, pas de la musique.

    `loose=True` retire en plus ce sur quoi le décodeur HÉSITE sans que la
    lecture change vraiment : la tenue (`carry`, une mesure qui prolonge
    l'accord précédent) et l'enrichissement au-delà de la triade (`Eb^7` et
    `Eb` deviennent le même Eb). Ce mode ne sert PAS à écrire le chart — il
    mesure combien de notre non-compacité vient du tremblement du décodage
    plutôt que de la musique. Les deux modes sont comprimés par le même code et
    passent le même test de reconstruction, chacun contre SA signature.
    """
    out = []
    for c in bar:
        q = str(c.get("q", ""))
        if loose:
            q = "-" if q.startswith("-") else ("o" if q.startswith("o") else "")
        out.append((int(c.get("beat", 0)), int(c["root"]), q,
                    -1 if loose else int(c.get("bass", -1)),
                    bool(c.get("nc")),
                    False if loose else bool(c.get("carry"))))
    if loose:                      # une tenue seule = « rien de nouveau »
        out = [t for i, t in enumerate(out)
               if not (i and out[i - 1][1:] == t[1:])]
    return tuple(out)


def signatures(bars, loose: bool = False) -> list[Bar]:
    return [bar_signature(b, loose) for b in bars]


# ── le cœur : comprimer une lettre ──────────────────────────────────────────

def _runs(chunk: tuple[Bar, ...], cell: tuple[Bar, ...]) -> list[Patch]:
    """Les tranches CONTIGUËS où la reprise s'écarte de la cellule."""
    out, i, P = [], 0, len(cell)
    while i < P:
        if chunk[i] == cell[i]:
            i += 1
            continue
        j = i
        while j < P and chunk[j] != cell[j]:
            j += 1
        out.append(Patch(i, tuple(chunk[i:j])))
        i = j
    return out


def _consensus(occs, P: int) -> tuple[Bar, ...] | None:
    """La cellule qui minimise les écarts : mesure par mesure, la majorité.

    Prise sur TOUTES les tranches pleines de toutes les occurrences restantes.
    Choisir la première occurrence comme référence (ce que fait le repli livré)
    revient à parier que la première passe est la plus propre ; le vote ne parie
    rien et coûte moins de retouches.
    """
    cols: list[Counter] = [Counter() for _ in range(P)]
    for _, seq in occs:
        for k in range(len(seq) // P):
            for j in range(P):
                cols[j][seq[k * P + j]] += 1
    if any(not c for c in cols):
        return None
    return tuple(c.most_common(1)[0][0] for c in cols)


def _try_form(occs, cell: tuple[Bar, ...]):
    """Construit le bloc de cellule `cell`. Rend (coût/mesure, bloc, retenues).

    Une occurrence rejoint le bloc si la rejoindre écrit STRICTEMENT moins de
    mesures que l'écrire au long. Une tranche qui ne partage RIEN avec la
    cellule n'est pas une variante mais une autre musique : elle sort en
    littéral, sans faire sortir toute l'occurrence — la refuser en bloc coûtait
    0,086 de compression sur le corpus (0,374 → 0,460, mesuré), pour un défaut
    qui ne concernait qu'une passe.
    """
    P = len(cell)
    patches: list[Patch] = []
    index: dict[Patch, int] = {}
    rends: list[Rendition] = []
    lits: list[Literal] = []
    used: list[int] = []
    covered = plain = 0
    for oi, seq in occs:
        n = len(seq)
        if n < P:
            continue
        plan, cost, cov, npl, nre = [], 0, 0, 0, 0
        for k in range(n // P):
            chunk = tuple(seq[k * P:(k + 1) * P])
            runs = _runs(chunk, cell)
            if any(len(r.bars) == P for r in runs):     # rien de commun
                cost += P
                plan.append((k * P, None, chunk))
                continue
            cov += sum(len(r.bars) for r in runs)
            nre += 1
            npl += not runs
            for pt in runs:
                if pt not in index:
                    cost += len(pt.bars)
            plan.append((k * P, runs, None))
        rest = n - (n // P) * P
        tail = tuple(seq[n - rest:]) if rest else ()
        cost += rest
        if cost >= n or not nre:         # rejoindre ne fait pas gagner
            continue
        used.append(oi)
        covered += cov
        plain += npl
        for off, runs, chunk in plan:
            if runs is None:
                lits.append(Literal(off, oi, chunk))
                continue
            got = []
            for pt in runs:
                if pt not in index:
                    index[pt] = len(patches)
                    patches.append(pt)
                got.append(index[pt])
            rends.append(Rendition(off, oi, tuple(got)))
        if rest:
            lits.append(Literal(n - rest, oi, tail))
    if len(rends) < MIN_RENDITIONS or not plain:
        return None
    blk = Block(P, cell, patches, rends, lits)
    explained = len(rends) * P - covered          # mesures rendues par la cellule
    if explained <= CELL_SHARE * blk.played:
        return None                               # un dictionnaire, pas une forme
    return blk.written / max(1, blk.played), plain, blk, used


def compress_letter(label: str, occurrences: list[list[Bar]]) -> LetterForm:
    """La forme minimale d'une lettre : des blocs, dans l'ordre chronologique.

    Chaque tour retient la cellule dont le coût par mesure JOUÉE est le plus bas
    (départage : période musicale, puis période longue, puis moins de
    retouches). Les occurrences qu'elle n'explique pas repartent au tour
    suivant ; ce qui ne boucle jamais est écrit littéralement.
    """
    todo = [(i, list(s)) for i, s in enumerate(occurrences) if s]
    blocks: list[Block] = []
    while todo:
        best = None
        for P in PERIODS:
            cells = []
            cs = _consensus(todo, P)
            if cs is not None:
                cells.append(cs)
            for _, seq in todo:                # la cellule telle qu'elle est
                if len(seq) >= P:              # jouée, si le vote la manque
                    cells.append(tuple(seq[:P]))
            for cell in dict.fromkeys(cells):
                got = _try_form(todo, cell)
                if got is None:
                    continue
                r, plain, blk, used = got
                # `-plain` : à coût égal, la cellule que le plus de reprises
                # jouent TELLE QUELLE. Sans ce départage, Let It Be écrivait
                # une boucle jouée par 4 reprises sur 12 et mettait les 8
                # autres en « variante » — aussi compact, illisible.
                key = (round(r, 6), _PERIOD_RANK[P], -P, -plain,
                       len(blk.patches))
                if best is None or key < best[0]:
                    best = (key, blk, used)
        if best is None:
            break
        _, blk, used = best
        blocks.append(blk)
        drop = set(used)
        todo = [(i, s) for i, s in todo if i not in drop]
    for oi, seq in todo:                        # ce qui ne boucle pas
        blocks.append(Block(0, (), [], [], [Literal(0, oi, tuple(seq))]))
    blocks.sort(key=lambda b: b.start)
    return LetterForm(label, blocks)


# ── reconstruction : le garde-fou ───────────────────────────────────────────

def reconstruct_block(blk: Block) -> dict[int, list[Bar]]:
    """{occurrence: suite de mesures}, telle que ce bloc la décrit."""
    parts: dict[int, list[tuple[int, list[Bar]]]] = {}
    for r in blk.renditions:
        bars = list(blk.cell)
        for pi in r.patches:
            pt = blk.patches[pi]
            bars[pt.pos:pt.pos + len(pt.bars)] = list(pt.bars)
        parts.setdefault(r.occ, []).append((r.b0, bars))
    for lit in blk.literals:
        parts.setdefault(lit.occ, []).append((lit.b0, list(lit.bars)))
    return {oi: [b for _, chunk in sorted(p) for b in chunk]
            for oi, p in parts.items()}


def reconstruct(form: LetterForm) -> list[list[Bar]]:
    """La suite d'accords de CHAQUE occurrence, rejouée depuis la forme."""
    parts: dict[int, list[tuple[int, list[Bar]]]] = {}
    for blk in form.blocks:
        for r in blk.renditions:
            bars = list(blk.cell)
            for pi in r.patches:
                pt = blk.patches[pi]
                bars[pt.pos:pt.pos + len(pt.bars)] = list(pt.bars)
            parts.setdefault(r.occ, []).append((r.b0, bars))
        for lit in blk.literals:
            parts.setdefault(lit.occ, []).append((lit.b0, list(lit.bars)))
    if not parts:
        return []
    return [[b for _, chunk in sorted(parts.get(i, [])) for b in chunk]
            for i in range(max(parts) + 1)]


# ── le morceau entier ───────────────────────────────────────────────────────

def occurrences_by_letter(sections: list[dict], sigs: list[Bar]):
    """{lettre: [occurrences]} + l'ordre d'apparition, depuis le ChartModel."""
    by: dict[str, list[list[Bar]]] = {}
    order: list[str] = []
    for s in sections:
        L = s["label"]
        if L not in by:
            order.append(L)
            by[L] = []
        for b0, b1 in s["barRanges"]:
            by[L].append(sigs[b0:b1 + 1])
    return by, order


def occurrence_ranges(sections: list[dict]) -> dict[str, list[tuple[int, int]]]:
    """{lettre: [(première mesure, dernière mesure)]}, dans l'ordre du morceau.

    Le pendant de `occurrences_by_letter` côté TEMPS : une reprise `(occ, b0)`
    d'un bloc retombe sur les mesures `ranges[occ][0] + b0 …`, donc sur des
    secondes de `barGrid`. C'est ce qui rend la forme jouable.
    """
    out: dict[str, list[tuple[int, int]]] = {}
    for s in sections:
        for b0, b1 in s["barRanges"]:
            out.setdefault(s["label"], []).append((b0, b1))
    return out


def compress_song(sections: list[dict], bars, loose: bool = False) -> SongForm:
    """Forme minimale d'un morceau.

    `sections` : la liste du ChartModel AVANT repli d'affichage (une occurrence
    par entrée, `label` + `barRanges`). `bars` : la liste mesure par mesure.
    """
    by, order = occurrences_by_letter(sections, signatures(bars, loose))
    letters = [compress_letter(L, by[L]) for L in order]
    aliases = dedupe_across_letters(letters)
    return SongForm(letters, aliases)


def dedupe_across_letters(letters: list[LetterForm]) -> list[tuple[str, str]]:
    """« Si on affiche 2 fois les 8 mêmes mesures, c'est un loupé. »

    Deux blocs dont la CELLULE est la même music bouclent sur la même musique,
    quelles que soient leurs terminaisons. Le second devient un alias du premier
    — la page écrit « B = A, autre fin » au lieu de redessiner la boucle — et ne
    dessine plus que ce par quoi il diffère. Rend la liste des alias posés.

    La clé est la cellule SEULE, et pas (cellule, retouches) : deux lettres qui
    bouclent pareil et finissent autrement, c'est justement le cas où Louis voit
    « 2 fois les mêmes mesures ». Mesuré sur les 27 morceaux : la clé complète
    n'attrape que 2 doublons de boucle, la clé courte en attrape 6.
    """
    seen: dict[tuple, str] = {}
    out = []
    for lf in letters:
        for i, blk in enumerate(lf.blocks):
            if not blk.cell:
                continue
            nm = lf.name(i)
            if blk.cell in seen and seen[blk.cell] != nm:
                blk.alias_of = seen[blk.cell]
                out.append((nm, seen[blk.cell]))
            else:
                seen.setdefault(blk.cell, nm)
    return out


# ── audit : ce que la règle d'or exige ──────────────────────────────────────

def duplicate_units(form: SongForm, min_bars: int = 1):
    """Les suites de mesures écrites DEUX FOIS sur la page (règle d'or).

    Rend [(mesures, [où])]. Une page correcte n'en a aucune.
    """
    where: dict[tuple[Bar, ...], list[str]] = {}
    for lf in form.letters:
        for i, blk in enumerate(lf.blocks):
            if blk.alias_of is not None:
                continue
            nm = lf.name(i)
            for kind, bb in blk.units():
                where.setdefault(bb, []).append(f"{nm}·{kind}")
    return [(bb, w) for bb, w in where.items()
            if len(w) > 1 and len(bb) >= min_bars]


def check_reconstruction(form: SongForm, sections, bars, loose: bool = False):
    """Rend les lettres dont la forme ne rejoue PAS la musique d'origine."""
    by, _ = occurrences_by_letter(sections, signatures(bars, loose))
    bad = []
    for lf in form.letters:
        got = reconstruct(lf)
        want = by[lf.label]
        if [list(x) for x in got] != [list(x) for x in want]:
            bad.append(lf.label)
    return bad
