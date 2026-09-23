"""irealb:// → ChartModel : importer une grille écrite par un musicien.

Le sens INVERSE de `irealb_export.py`. Ce module existait dans l'ancienne app
(:7771) et n'avait jamais été porté — `irealb_fetcher.py` dit lui-même, en
tête, qu'il a laissé « la conversion irealb:// → ChordChart » derrière lui.
Résultat : l'onglet « iReal Pro » de la recherche trouvait des morceaux et
n'en ouvrait aucun (`importIrealChart` était appelé sans exister).

Pourquoi ne pas utiliser `pyRealParser.Tune.measures_as_strings` : parce
qu'il APLATIT. `Tune._get_measures` enlève les repères de section (`*A`),
déroule les reprises (`{...}`) et fusionne les 1re/2e fins — c'est-à-dire
exactement ce que Louis demande de conserver (« les mêmes accords ET les
mêmes sections que iRealb »). On ne garde de `Tune` que le désembrouillage
(l'involution par blocs de 50) et les métadonnées, et on relit nous-mêmes la
chaîne BRUTE, celle où `*A`, `{`, `}`, `N1`, `N2` sont encore là.

Ce que ce module ne fait PAS :
  * deviner un tempo réel — un chart iReal n'a pas d'audio. Il pose une
    grille de temps NOMINALE (voir `NOMINAL_BPM`) pour que tout l'aval
    (empans de section, `barSpans`, prompteur) soit défini et monotone
    plutôt que rempli de zéros. `audio_url` reste vide : l'app sait déjà
    qu'un chart sans audio ne se joue pas.
  * lire les D.C./D.S. al Coda — `S`, `Q`, `U` sont ignorés. Le chart écrit
    ce qui est écrit, sans dérouler les sauts (`docs/known_issues.md`).
  * toucher au fichier : voir `harmonia/server/routes/irealb.py` pour l'écriture.
"""
from __future__ import annotations

import logging
import re
import unicodedata
import urllib.parse

from pyRealParser.pyRealParser import Tune

log = logging.getLogger("harmonia.integrations.irealb_import")

#: Tempo nominal d'un chart importé. Il ne prétend rien sur le morceau : il
#: n'existe que pour donner des `t0/t1` monotones à chaque accord. Sans lui
#: tous les empans valent [0,0] et l'aval (Practise, boucle A-B, éditeur de
#: sections) travaille sur des durées nulles — `chordAt` ne trouve jamais
#: d'accord quand `t0 === t1`.
NOMINAL_BPM = 120.0

NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

#: Les barres de mesure d'iReal, toutes équivalentes pour le découpage :
#: `|` simple, `LZ`/`K` (nouvelle mesure), `[`/`]` (double), `{`/`}`
#: (reprise), `Z` (fin). Seules `{` et `}` portent une information en plus.
_BARLINES = {"|", "LZ", "K", "[", "]", "{", "}", "Z"}

#: `x` et `cl` disent la MÊME chose — « rejoue la mesure précédente » (le
#: signe `%`). Les deux encodages coexistent dans le corpus iReal : *Autumn
#: Leaves* des standards écrit `Kcl`, celui du forum écrit `|x`. N'en
#: reconnaître qu'un faisait disparaître la mesure, et décalait tout ce qui
#: suit d'un cran.

#: L'alphabet d'une queue d'accord iReal : `^` maj, `-` min, `+` aug,
#: `o` dim, `h` demi-dim, `alt`, `sus`, `add`, les chiffres et les
#: altérations. Le fermer est ce qui permet de distinguer `G-6XyQKcl`
#: (accord + remplissage + barre + reprise) d'un seul long symbole.
_QUAL = r"(?:alt|sus|add|[\^\-\+oh#b0-9])*"

_TOKEN = re.compile(
    r"(?P<comment><[^>]*>)"
    r"|(?P<alt>\([^)]*\))"
    r"|(?P<ts>T\d\d)"
    r"|(?P<section>\*[A-Za-z])"
    r"|(?P<ending>N\d)"
    r"|(?P<bar>LZ|XyQ|[|\[\]{}KZ])"
    r"|(?P<rep1>cl|x)"
    r"|(?P<rep2>r)"
    r"|(?P<slash>p)"
    r"|(?P<nc>n)"
    r"|(?P<pedal>W/[A-G][#b]?)"
    rf"|(?P<chord>[A-G][#b]?{_QUAL}(?:/[A-G][#b]?)?)"
    r"|(?P<skip>.)"
)
# `XyQ` (cellule vide) est rangé avec les barres de mesure À DESSEIN : il ne
# sépare pas deux mesures, mais il ne doit jamais être avalé par la queue de
# l'accord qui le précède. Le découpage en mesures ne le compte pas (voir
# `_measures`), il ne sert qu'à borner le symbole.


def parse_root(sym: str) -> int:
    """'Bb' -> 10, 'F#' -> 6, 'C' -> 0."""
    pc = NOTE_PC[sym[0].upper()]
    for ch in sym[1:]:
        if ch in "#s":
            pc += 1
        elif ch in "b!":
            pc -= 1
    return pc % 12


def raw_chord_string(tune_string: str) -> str:
    """La chaîne d'accords DÉSEMBROUILLÉE mais non nettoyée.

    `Tune.__init__` enchaîne désembrouillage puis `_cleanup_chord_string`
    puis `_remove_annotations` ; le second efface `*A`, le troisième déroule
    `{...}`. On refait ici les deux premières lignes de `__init__`, rien de
    plus — le découpage des champs reste celui de pyRealParser pour qu'il ne
    puisse pas diverger.
    """
    parts = re.split(r"=+", tune_string)
    offset = 0 if parts[4].index(Tune._chords_prefix) == 0 else 1
    scrambled = parts[4 + offset].split(Tune._chords_prefix)[1]
    return Tune._unscramble_chord_string(scrambled)


# ── la relecture de la chaîne brute ──────────────────────────────────────────

class _Bar:
    """Une mesure telle qu'iReal l'écrit, avec ses marques de structure."""

    __slots__ = ("chords", "held", "marker", "repeat_start", "repeat_end",
                 "ending", "double", "meter")

    def __init__(self):
        self.chords: list[dict] = []   # {root,q,bass,nc,var}
        self.held = False              # `cl`/`x`/`r` : la mesure précédente tient
        self.marker: str | None = None  # la lettre d'un repère `*A`
        self.repeat_start = False
        self.repeat_end = False
        self.ending: int | None = None
        #: une double barre `[` ou `]` s'ouvre sur cette mesure. iReal ne
        #: s'en sert qu'aux frontières de section — c'est le seul indice de
        #: structure des grilles qui n'ont aucun repère `*X` (le refrain de
        #: « Autumn Leaves In Our Key! », par exemple).
        self.double = False
        #: LE CHIFFRAGE DE CETTE MESURE-LÀ (2026-09-18, Louis sur *Virtual
        #: Insanity*) : « un passage à la fin qui ne fait que 2 temps, noté
        #: comme du 2/4, en fait c'est une demi-barre et puis ça reprend, très
        #: jamiroquai — il faut le noter ». iReal change de chiffrage EN COURS
        #: de morceau ; on ne lisait que le premier et toutes les mesures
        #: duraient pareil. Trois demi-barres disparaissaient sur ce seul
        #: morceau, et le repli fusionnait une fin de 2 temps avec une de 4.
        self.meter: tuple[int, int] = (4, 4)

    def empty(self) -> bool:
        return not self.chords and not self.held


def _chord(sym: str) -> dict:
    """'Bb^7/D' -> {root:10, q:'^7', bass:2}. La queue est recopiée TELLE
    QUELLE : `q` est déjà, dans tout Harmonia, la queue iReal (voir
    `harmonia/labels.py` et la table `TOK` du rendu). Importer, ici, c'est
    ne rien traduire."""
    head, _, bass = sym.partition("/")
    m = re.match(r"^([A-G][#b]?)(.*)$", head)
    root, q = parse_root(m.group(1)), m.group(2)
    return {"root": root, "q": q, "bass": parse_root(bass) if bass else -1,
            "nc": False}


def _measures(raw: str) -> tuple[list[_Bar], int, int]:
    """La chaîne brute → une liste de mesures + le chiffrage (num, den).

    Un seul passage, un seul automate : les marques de structure (`*A`,
    `{`, `}`, `N1`) se posent sur la mesure qu'elles ouvrent ou ferment,
    les accords s'empilent dans la mesure courante. Les mesures vides —
    les cellules de remplissage de mise en page d'iReal — disparaissent.
    """
    bars: list[_Bar] = []
    cur = _Bar()
    dernier: dict | None = None   # le dernier accord VRAIMENT écrit (pour `W/`)
    pending_marker: str | None = None
    pending_start = False
    pending_double = False
    pending_ending: int | None = None
    ts = (4, 4)
    ts_premier: tuple[int, int] | None = None
    slots: list[str] = []   # 'c' (accord) ou 'p' (barre oblique) dans l'ordre

    def flush():
        nonlocal cur, slots, pending_marker, pending_start, pending_double
        nonlocal pending_ending
        if not cur.empty():
            cur.meter = ts
            _place_beats(cur, slots, _pulsations(ts))
            bars.append(cur)
        else:
            # Une cellule vide n'est pas une mesure, mais elle ne doit pas
            # perdre les marques qu'on venait d'y poser (« *B[ » : le repère
            # arrive avant la double barre, donc avant la vraie mesure).
            if cur.marker:
                pending_marker = cur.marker
            pending_start = pending_start or cur.repeat_start
            pending_double = pending_double or cur.double
            if cur.ending is not None:
                pending_ending = cur.ending
            if cur.repeat_end and bars:
                bars[-1].repeat_end = True
        cur = _Bar()
        cur.marker, pending_marker = pending_marker, None
        cur.repeat_start, pending_start = pending_start, False
        cur.double, pending_double = pending_double, False
        cur.ending, pending_ending = pending_ending, None
        slots = []

    for m in _TOKEN.finditer(raw):
        kind, tok = m.lastgroup, m.group()
        if kind == "ts":
            # Les chiffrages d'iReal sont des CODES à deux chiffres, pas un
            # numérateur suivi d'un dénominateur : `T44` = 4/4, `T34` = 3/4,
            # `T68` = 6/8 — et `T12` = 12/8, le seul où la lecture naïve
            # (1/2) est fausse. pyRealParser fait cette erreur-là ; 21 des
            # 2207 standards en dépendent.
            ts = (12, 8) if tok == "T12" else (int(tok[1]), int(tok[2]))
            # Le chiffrage du CHART est le premier : un morceau en 4/4 avec
            # une mesure de 2/4 au milieu (`Virtual Insanity`) rendait
            # autrefois le DERNIER vu, et la grille entière basculait sur
            # une mesure de passage. Les suivants continuent de servir, mais
            # localement : `_place_beats` les lit pour cette mesure-là.
            if ts_premier is None:
                ts_premier = ts
        elif kind == "section":
            cur.marker = tok[1]
        elif kind == "ending":
            cur.ending = int(tok[1])
        elif kind == "bar":
            if tok == "XyQ":
                continue                       # cellule vide : mise en page
            if tok == "{":
                flush()
                cur.repeat_start = True
            elif tok == "}":
                cur.repeat_end = True
                flush()
                if bars:
                    bars[-1].repeat_end = True
            else:
                flush()
                if tok in ("[", "]"):
                    cur.double = True
        elif kind == "rep1":
            cur.held = True
            flush()
        elif kind == "rep2":
            cur.held = True
            flush()
            cur.held = True
            flush()
        elif kind == "slash":
            slots.append("p")
        elif kind == "nc":
            cur.chords.append({"root": 0, "q": "", "bass": -1, "nc": True})
            slots.append("c")
        elif kind == "pedal":
            # `W/G♭` = LE MÊME ACCORD, UNE AUTRE BASSE — la basse qui marche
            # sous un accord tenu (le refrain de *Virtual Insanity* :
            # `E♭-7 · /G♭ · A♭-11 · /B♭`). On le lisait comme un accord de
            # sol♭ MAJEUR, c'est-à-dire un accord que personne n'a écrit :
            # 877 occurrences dans 154 des 2207 standards, *Round Midnight*
            # compris. `W` n'apparaît jamais sans barre oblique (vérifié).
            basse = parse_root(tok[2:])
            if dernier is None:
                # Rien au-dessus : la basse est tout ce qu'on sait.
                cur.chords.append({"root": basse, "q": "", "bass": -1, "nc": False})
            else:
                cur.chords.append({"root": dernier["root"], "q": dernier["q"],
                                   "bass": basse, "nc": False,
                                   # Le rendu n'écrira que « /G♭ », comme
                                   # iReal — l'accord complet reste ici.
                                   "pedal": True})
            slots.append("c")
        elif kind == "chord":
            dernier = _chord(tok)
            cur.chords.append(dict(dernier))
            slots.append("c")
        elif kind == "alt":
            # `(D-7)` — l'accord optionnel, petit au-dessus. Le rendu
            # l'attend depuis 2026-08-17 sous le nom `var` (chart.js).
            inner = tok[1:-1].strip()
            if re.match(rf"^[A-G][#b]?{_QUAL}(?:/[A-G][#b]?)?$", inner):
                v = _chord(inner)
                if cur.chords:
                    cur.chords[-1]["var"] = {**v, "c": 1.0, "n": 1}
        # comment / skip : rien (les repères S, Q, U, s, l, f n'ont pas de
        # sens sur une grille sans lecture — voir la docstring du module).
    flush()
    premier = ts_premier or (4, 4)
    return bars, premier[0], premier[1]


def _pulsations(meter: tuple[int, int]) -> int:
    """Le nombre de PULSATIONS d'une mesure — ce qui fait sa durée et le
    nombre de colonnes qu'elle occupe. Un 6/8 se bat à 2, un 12/8 à 4 ;
    partout ailleurs c'est le numérateur."""
    num, den = meter
    if den == 8 and num % 3 == 0:
        return num // 3
    return num if num in (2, 3, 4, 5, 6, 7, 9, 12) else 4


def _place_beats(bar: _Bar, slots: list[str], bpb: int) -> None:
    """Le temps de chaque accord DANS sa mesure.

    Les barres obliques d'iReal (`p`) sont une information de placement :
    `C7 p p F7` dit bien « C7 sur 1, F7 sur 4 ». Quand il n'y en a pas, les
    accords se répartissent également — ce que iReal affiche de toute façon
    pour une mesure à deux accords.
    """
    n = len(slots) or 1
    i = 0
    for k, s in enumerate(slots):
        if s != "c":
            continue
        if i < len(bar.chords):
            bar.chords[i]["beat"] = int(round(k * bpb / n))
        i += 1
    for j, c in enumerate(bar.chords):   # filet : pas de `slots` exploitable
        c.setdefault("beat", int(round(j * bpb / max(1, len(bar.chords)))))


# ── mesures → sections (repères, reprises, 1re/2e fins) ──────────────────────

def _sections(bars: list[_Bar]) -> list[list[_Bar]]:
    """Découper la suite de mesures en sections, comme iReal les montre.

    Une section commence à un repère `*A`, à une ouverture de reprise `{`,
    ou juste après la fermeture d'une reprise. L'exception qui compte :
    une mesure de 2e fin (`N2`) reste dans la section de la 1re, sinon le
    crochet « 2. » deviendrait une section à lui tout seul.
    """
    out: list[list[_Bar]] = []
    cur: list[_Bar] = []
    prev_close = False
    for b in bars:
        boundary = bool(cur) and (
            b.marker is not None or b.repeat_start or b.double
            or (prev_close and b.ending is None))
        if boundary:
            out.append(cur)
            cur = []
        cur.append(b)
        prev_close = b.repeat_end
    if cur:
        out.append(cur)
    return out


def _label_of(marker: str | None, fallback: str) -> str:
    if marker is None:
        return fallback
    if marker in ("i", "I"):
        return "intro"
    if marker in ("v", "V"):
        return "verse"
    return marker.upper()


def _propagate_endings(block: list[_Bar]) -> None:
    """`N1` ne marque que la PREMIÈRE mesure de la fin ; les suivantes en
    font partie jusqu'au marqueur suivant."""
    cur: int | None = None
    for b in block:
        if b.ending is not None:
            cur = b.ending
        elif cur is not None:
            b.ending = cur


def _bar_chords(b: _Bar) -> list[dict]:
    return [] if b.held else [dict(c) for c in b.chords]


def _block_to_section(block: list[_Bar], idx: int) -> dict:
    """Un bloc « logique » → UNE liste de blocs (une seule, sauf fins inégales).

    Ni étiquette ni temps encore — les deux se décident APRÈS le repliement
    (`_fold_blocks`), quand on sait quels blocs sont en fait le même, et
    combien de mesures le morceau fait en tout.
    """
    _propagate_endings(block)
    endings: dict[int, list[_Bar]] = {}
    prefix: list[_Bar] = []
    for b in block:
        if b.ending is None:
            if endings:
                # Une mesure sans marque APRÈS une fin appartient à la
                # dernière fin ouverte (iReal ne re-marque pas chaque mesure).
                endings[max(endings)].append(b)
            else:
                prefix.append(b)
        else:
            endings.setdefault(b.ending, []).append(b)

    has_repeat = any(b.repeat_start for b in block) or any(b.repeat_end for b in block)
    variants = [endings[k] for k in sorted(endings)]
    base = {"id": f"S{idx}", "tag": "", "marker": block[0].marker}

    if variants and len({len(v) for v in variants}) > 1:
        # DES FINS DE LONGUEURS DIFFÉRENTES NE SONT PAS DES FINS (2026-09-18).
        # iReal l'écrit (154 des 2207 standards : *A Ballad* a une 1re fin de
        # 5 mesures et une 2e de 6), mais le modèle d'Harmonia suppose une
        # queue de longueur UNIQUE : `bars` = un passage entier, et chaque
        # passage se joue à cette longueur-là. Les mélanger donnait un chart
        # qui ment sur son compte de mesures — et le rendu indexe `barSpans`
        # à `prefixLen + vi*tail + j`, donc il se décalait aussi.
        # On écrit donc un passage par fin, chacun à SA longueur : c'est la
        # règle « plutôt sous-replier que sur-replier » (Louis, 2026-07-30).
        return [{**base, "id": f"S{idx}_{i}", "reps": 1,
                 "bars": [_bar_chords(b) for b in prefix + v],
                 "beats": [_pulsations(b.meter) for b in prefix + v]}
                for i, v in enumerate(variants)]

    if variants:
        return [{**base, "reps": max(2, len(variants)),
                 "bars": [_bar_chords(b) for b in prefix + variants[0]],
                 "beats": [_pulsations(b.meter) for b in prefix + variants[0]],
                 "endings": {"tail": len(variants[0]),
                             "variants": [{"passes": [],
                                           "bars": [_bar_chords(b) for b in v],
                                           "beats": [_pulsations(b.meter) for b in v]}
                                          for v in variants]}}]
    return [{**base, "reps": 2 if has_repeat else 1,
             "bars": [_bar_chords(b) for b in block],
             "beats": [_pulsations(b.meter) for b in block]}]


# ── le repliement : une lettre = un seul bloc écrit ──────────────────────────

#: Au-delà de deux mesures, une queue qui diffère n'est plus une 1re/2e fin,
#: c'est une autre musique. Même constante que `folding._ENDING_TAIL_MAX`,
#: pour que les deux chemins (audio et iReal) ne suspendent pas des crochets
#: selon deux règles différentes.
_ENDING_TAIL_MAX = 2


def _famille(q: str) -> str:
    """La FAMILLE d'un accord : son 3e et son type, sans la 7e ni les extensions.

    Pourquoi cette table existe (Louis, 2026-09-18, « choisis une écriture
    unique ») : le transcripteur d'iReal écrit le même accord de plusieurs
    façons dans le même morceau. Sur *Virtual Insanity*, la même mesure
    s'écrit `Cø7 B△7`, `Cø7 B△` et `Cø B△` — trois lignes quasi identiques
    dans la grille pour une seule musique. La famille sert à DÉCIDER que
    c'est le même passage ; l'écriture retenue, elle, est choisie par un vote
    (`_ecriture_unique`), jamais inventée ici.

    Ce que la famille EFFACE, donc : la 7e, les 9/11/13, les altérations.
    Ce qu'elle GARDE : la tierce, la quinte diminuée ou augmentée, le sus.
    `C` et `C7` restent deux familles différentes (majeur contre dominante) —
    c'est la distinction qui porte la fonction harmonique.
    """
    if q.startswith("h") or q.startswith("-7b5"):
        return "h"                                    # demi-diminué
    if q.startswith("o"):
        return "o"                                    # diminué
    if "sus" in q:
        return "sus"
    if q.startswith("-"):
        return "-"                                    # mineur (avec ou sans 7e)
    if q.startswith("+"):
        return "+" if q == "+" else "7"      # `+` triade augmentée, `+7` dominante
    if q.startswith("^") or q == "" or q[:1] in ("6", "2", "5") or q.startswith("add"):
        return "^"                                    # majeur
    return "7"                                        # dominante


def _sig_famille(bars, beats) -> tuple:
    """La signature qui sert à DÉCIDER (familles + durées), pas à écrire.

    Les PULSATIONS en font partie au même titre que les accords : une mesure
    de 2 temps et une de 4 ne sont pas le même passage, même avec les mêmes
    accords. C'est la demi-barre de *Virtual Insanity* (Louis, 2026-09-18).
    """
    return tuple((tuple((c["root"], _famille(c["q"]), c.get("bass", -1),
                         bool(c.get("nc"))) for c in bar), n)
                 for bar, n in zip(bars, beats))


def _signature(sec: dict) -> tuple:
    """La signature d'un bloc entier : son tronc et ses fins.

    Les PULSATIONS en font partie, au même titre que les accords : une mesure
    de 2 temps et une de 4 ne sont pas le même passage, même avec les mêmes
    accords. C'est la demi-barre de *Virtual Insanity* (Louis, 2026-09-18) —
    le repli l'effaçait en fusionnant une fin de 2 temps avec une de 4.
    """
    e = sec.get("endings") or {}
    return (_sig_famille(sec["bars"], sec["beats"]),
            tuple(_sig_famille(v["bars"], v["beats"])
                  for v in e.get("variants", ())))


def _ecriture_unique(lots: list[tuple[list, int]]) -> list:
    """Une seule graphie par case : la plus fréquente l'emporte.

    `lots` = les passages à fusionner, chacun `(mesures, poids)` où le poids
    est le nombre de fois que ce passage se joue. Deux règles, dans l'ordre :
    la graphie que le morceau emploie le PLUS souvent gagne ; à égalité,
    celle qui arrive en premier. On ne choisit donc jamais à la place du
    transcripteur — on tranche son hésitation avec ses propres statistiques.

    Tous les lots ont forcément la même forme (mêmes mesures, même nombre
    d'accords par mesure) : ils ont été groupés sur leur signature de
    FAMILLE, qui compte les accords. Seule la queue `q` varie.
    """
    modele = [[dict(c) for c in bar] for bar in lots[0][0]]
    for r, bar in enumerate(modele):
        for i in range(len(bar)):
            votes: dict[str, int] = {}
            rang: dict[str, int] = {}
            for ordre, (bars, poids) in enumerate(lots):
                q = bars[r][i]["q"]
                votes[q] = votes.get(q, 0) + max(1, poids)
                rang.setdefault(q, ordre)
            gagnant = max(votes, key=lambda q: (votes[q], -rang[q]))
            if gagnant != bar[i]["q"]:
                log.info("ireal: écriture unique — %r l'emporte sur %r", gagnant,
                         bar[i]["q"])
            bar[i]["q"] = gagnant
    return modele


def _fold_blocks(blocks: list[dict]) -> list[dict]:
    """Replier les blocs identiques — le geste central d'un chart Harmonia.

    iReal écrit un chart LINÉAIRE : son refrain marqué `*B` est réécrit en
    entier à chacune de ses quatre reprises, et c'est ce qu'on voyait — quatre
    blocs « B » l'un sous l'autre (*Virtual Insanity*, 2026-09-18). Harmonia
    écrit la musique UNE fois et compte les passages.

    Deux étages, dans cet ordre :

    1. **Même musique → un seul bloc.** « Même » se juge sur les FAMILLES
       d'accords (`_famille`), pas sur la graphie : le transcripteur écrit
       `Cø7 B△7` ici et `Cø B△` trois lignes plus bas pour le même passage.
       La graphie retenue sort d'un vote entre les passages
       (`_ecriture_unique`). Le repère d'iReal ne sépare plus rien : 131 des
       2207 standards nomment `*A` et `*C` deux passages identiques.
    2. **Même longueur, même début, queue différente → 1re/2e fin.** C'est
       le crochet « 1. / 2. » que le treillis dessine déjà, et le contrat de
       `folding._ireal_endings`.

    CE QUE ÇA NE REPLIE PAS, ET POURQUOI : deux passages de LONGUEURS
    DIFFÉRENTES, même quand le court est le début exact du long. Louis,
    2026-07-30 : « I'd rather under fold than over fold every time ». Un
    passage de 4 mesures écrit dans un bloc de 7 est un chart qui ment sur sa
    propre longueur, et le curseur de lecture traverse alors la section à la
    mauvaise vitesse. `folding._merge_coupe` se le permet sur le chemin
    audio, où le découpage est une ESTIMATION ; ici la longueur vient du
    transcripteur, il n'y a rien à rattraper.
    """
    groupes: list[dict] = []
    for blk in blocks:
        blk["_sig"] = _signature(blk)
        variants = (blk.get("endings") or {}).get("variants", ())
        blk["_passes"] = [p % (len(variants) or 1)
                          for p in range(max(1, blk["reps"]))]
        hote = next((g for g in groupes if g["_sig"] == blk["_sig"]), None)
        if hote is None:
            hote = dict(blk)
            hote["_markers"] = [blk.get("marker")]
            hote["_lots"] = [(blk["bars"], blk["reps"])]
            hote["_starts"] = []       # rempli par `_lay_out_time`
            if variants:
                # On DÉTACHE les fins du bloc d'origine : `dict(blk)` est une
                # copie de surface, et l'étage 2 va muter `endings`.
                hote["endings"] = {
                    "tail": blk["endings"]["tail"],
                    "variants": [{"bars": v["bars"], "beats": v["beats"],
                                  "_lots": [(v["bars"], blk["reps"])]}
                                 for v in variants]}
            groupes.append(hote)
        else:
            hote["_markers"].append(blk.get("marker"))
            hote["_lots"].append((blk["bars"], blk["reps"]))
            for hv, bv in zip((hote.get("endings") or {}).get("variants", ()),
                              variants):
                hv["_lots"].append((bv["bars"], blk["reps"]))
            hote["reps"] += blk["reps"]
        blk["_group"] = hote

    groupes = _fold_endings(groupes, blocks)
    for g in groupes:
        _voter_la_graphie(g)
    return groupes


def _voter_la_graphie(g: dict) -> None:
    """Le vote, appliqué au tronc du bloc puis à chacune de ses fins.

    Le tronc et les queues votent SÉPARÉMENT, chacun sur ses propres mesures :
    c'est tout l'intérêt d'une 1re et d'une 2e fin — elles ne disent pas la
    même chose, et elles n'ont pas forcément le même nombre d'accords.
    """
    endings = g.get("endings")
    tail = endings["tail"] if endings else 0
    prefix_len = len(g["bars"]) - tail
    tronc = _ecriture_unique([(bars[:prefix_len], w) for bars, w in g["_lots"]])
    if not endings:
        g["bars"] = tronc
        return
    for v in endings["variants"]:
        v["bars"] = _ecriture_unique(v["_lots"])
    # `bars` = un passage ENTIER : son tronc, puis la queue du PREMIER variant.
    g["bars"] = tronc + [list(b) for b in endings["variants"][0]["bars"]]


def _fold_endings(groupes: list[dict], blocks: list[dict]) -> list[dict]:
    """Étage 2 : deux blocs de MÊME LONGUEUR qui ne diffèrent que sur leur
    queue deviennent un seul bloc à 1re et 2e fin.

    Sur *Virtual Insanity*, deux passages de 4 mesures se terminent l'un par
    `B♭7♯5` et l'autre par `B♭7♯5 E♭m7` : c'est une 1re et une 2e fin, pas
    deux sections. Les deux passages gardent leurs 4 mesures — la règle
    « jamais sur-replier » (2026-07-30) tient, puisque personne n'est écrit à
    une longueur qu'il ne joue pas.
    """
    sortie: list[dict] = []
    for g in groupes:
        tail = hote = None
        if not g.get("endings"):
            for h in sortie:
                t = _queue_diff(h, g)
                if t is not None:
                    hote, tail = h, t
                    break
        if hote is None:
            sortie.append(g)
            continue
        prefix_len = len(hote["bars"]) - tail
        if not hote.get("endings"):
            hote["endings"] = {"tail": tail, "variants": [
                {"bars": [list(b) for b in hote["bars"][prefix_len:]],
                 "beats": list(hote["beats"][prefix_len:]),
                 "_lots": [(bars[prefix_len:], w) for bars, w in hote["_lots"]]}]}
        vi = len(hote["endings"]["variants"])
        hote["endings"]["variants"].append(
            {"bars": [list(b) for b in g["bars"][prefix_len:]],
             "beats": list(g["beats"][prefix_len:]),
             "_lots": [(bars[prefix_len:], w) for bars, w in g["_lots"]]})
        hote["_lots"] += g["_lots"]
        hote["_markers"] += g["_markers"]
        hote["reps"] += g["reps"]
        for blk in (b for b in blocks if b.get("_group") is g):
            blk["_group"] = hote
            blk["_passes"] = [vi] * len(blk["_passes"])
        log.info("ireal: deux passages de %d mesures ne diffèrent que sur "
                 "leurs %d dernières → une 1re et une 2e fin",
                 len(hote["bars"]), tail)
    return sortie


def _queue_diff(a: dict, b: dict) -> int | None:
    """La longueur de queue qui sépare `a` de `b`, ou None s'ils ne sont pas
    « même longueur, même début, queue différente »."""
    sa = _sig_famille(a["bars"], a["beats"])
    sb = _sig_famille(b["bars"], b["beats"])
    if len(sa) != len(sb) or sa == sb:
        return None
    prefixe = 0
    for x, y in zip(sa, sb):
        if x != y:
            break
        prefixe += 1
    tail = len(sa) - prefixe
    if prefixe < 2 or not (1 <= tail <= _ENDING_TAIL_MAX):
        return None
    if a.get("endings") and a["endings"]["tail"] != tail:
        return None
    # UNE VRAIE 2e FIN CHANGE D'ACCORD OU DE DURÉE, PAS DE COULEUR — même
    # exigence que `folding._ireal_endings` (la queue doit changer de
    # FONDAMENTALE), plus la durée : une fin de 2 temps contre une de 4 est
    # une vraie 2e fin même sur les mêmes accords. C'est le cas de figure que
    # Louis appelle « très jamiroquai ».
    ra = tuple((tuple(c[0] for c in bar), n) for bar, n in sa[prefixe:])
    rb = tuple((tuple(c[0] for c in bar), n) for bar, n in sb[prefixe:])
    return tail if ra != rb else None


def _label_blocks(groupes: list[dict]) -> None:
    """A, B, C… DANS L'ORDRE OÙ LE MORCEAU LES JOUE.

    Louis, 2026-09-18 : « l'ordre des sections c'est TOUJOURS ABCDE… ». On
    n'hérite donc PAS des repères d'iReal : ils ne sortent pas dans l'ordre
    (sur *Virtual Insanity*, son `*A` arrive en 3e position et son `*B` en 5e,
    entre deux passages qu'il ne nomme pas du tout), et une grille lue de haut
    en bas doit pouvoir se citer « A, B, C ». Une section qui revient reprend
    sa lettre — c'est le repliement qui le garantit, pas cette fonction.

    La seule exception est l'intro SANS accord : elle garde son nom, parce que
    `chart.js` la réduit en lecture à une bande « intro … » et que c'est le
    bon rendu pour une mesure qui n'a rien à lire. Un `*i` qui porte de vrais
    accords, lui, prend une lettre comme les autres — sinon ses mesures
    disparaîtraient du chart.
    """
    lettres = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for g in groupes:
        muet = not any(c for bar in g["bars"] for c in bar if not c.get("nc"))
        if muet and any(r in ("i", "I") for r in g["_markers"]):
            g["label"] = "intro"
            continue
        g["label"] = next(lettres, "?")


# ── la grille horaire nominale ───────────────────────────────────────────────

def _lay_out_time(blocks: list[dict], groupes: list[dict],
                  pulse: float) -> tuple[list[float], list[int], int]:
    """Poser `spans`, `barRanges`, `barSpans`, `meters` et les `t0/t1`.

    Les mesures sont posées dans l'ORDRE DE JEU (`blocks`, la suite linéaire
    qu'iReal écrit) ; les passages sont ensuite rangés sous le bloc REPLIÉ qui
    les représente (`groupes`). Un refrain joué quatre fois à quatre endroits
    du morceau est donc écrit une fois, avec quatre passages dispersés dans le
    temps — et `chart.js:formRuns` les retrouve en triant les `barRanges`.

    TOUTES LES MESURES N'ONT PAS LA MÊME DURÉE (2026-09-18). iReal change de
    chiffrage en cours de morceau, et *Virtual Insanity* s'en sert trois fois
    pour une demi-barre de 2/4. La grille se construit donc en accumulant les
    PULSATIONS de chaque mesure, jamais en multipliant un index par une durée
    fixe : `grid[i]` est l'unique source des temps, ici comme en aval.

    Le contrat est celui de `harmonia.folding._ireal_endings`, à la lettre :
    `bars` = un passage ENTIER, `endings.variants[i].bars` = la queue de ce
    variant, `barSpans` listé « les mesures du tronc commun, puis la queue de
    chaque variant ». `meters` suit exactement le même ordre.
    """
    # 1. l'ordre de jeu pose les mesures, chacune avec SES pulsations.
    #    `blk["_passes"][p]` dit quelle fin ce passage-là joue — posé par
    #    `_fold_blocks`, qui sait s'il est devenu la 2e fin d'un autre.
    battues: list[int] = []
    for blk in blocks:
        endings = blk.get("endings")
        prefix_len = len(blk["bars"]) - (endings["tail"] if endings else 0)
        for v in blk["_passes"]:
            blk["_group"]["_starts"].append((len(battues), v))
            battues += blk["beats"][:prefix_len]
            if endings:
                battues += endings["variants"][v]["beats"]

    n_bars = len(battues)
    grid = [0.0]
    for n in battues:
        grid.append(round(grid[-1] + n * pulse, 4))

    # 2. chaque bloc replié range ses passages
    for s in groupes:
        starts = s.pop("_starts")
        for cle in ("_sig", "_markers", "_lots", "marker", "_passes",
                    "_group", "_variant_offset"):
            s.pop(cle, None)
        L = len(s["bars"])
        s["reps"] = len(starts)
        s["spans"] = [[grid[b0], grid[b0 + L]] for b0, _v in starts]
        s["barRanges"] = [[b0, b0 + L - 1] for b0, _v in starts]

        endings = s.get("endings")
        tail = endings["tail"] if endings else 0
        prefix_len = L - tail
        rows = [[[grid[b0 + r], grid[b0 + r + 1]] for b0, _v in starts]
                for r in range(prefix_len)]
        # `meters` : les pulsations de chaque mesure ÉCRITE, dans l'ordre
        # d'émission du rendu (tronc, puis la queue de chaque variant).
        meters = list(s["beats"][:prefix_len])

        for r, bar in enumerate(s["bars"][:prefix_len]):
            b = starts[0][0] + r
            _stamp(bar, b, grid[b], grid[b + 1], s["beats"][r])

        if endings:
            for vi, v in enumerate(endings["variants"]):
                # `passes` indexe `spans` : c'est ainsi que le rendu sait
                # quels passages jouent CETTE fin-là.
                v["passes"] = [i for i, (_b0, vv) in enumerate(starts) if vv == vi]
                if not v["passes"]:
                    v["passes"] = [min(vi, len(starts) - 1)]
                base = starts[v["passes"][0]][0] + prefix_len
                meters += v["beats"]
                for r, bar in enumerate(v["bars"]):
                    b = base + r
                    _stamp(bar, b, grid[b], grid[b + 1], v["beats"][r])
                    rows.append([[grid[starts[p][0] + prefix_len + r],
                                  grid[starts[p][0] + prefix_len + r + 1]]
                                 for p in v["passes"]])
                # `bars` = le passage entier : sa queue est celle du 1er variant.
                if vi == 0:
                    s["bars"][prefix_len:] = [list(b) for b in v["bars"]]
                    s["beats"][prefix_len:] = list(v["beats"])
        s["barSpans"] = rows
        s["meters"] = meters
        s.pop("beats", None)
        for v in (s.get("endings") or {}).get("variants", ()):
            v.pop("_lots", None)
            v.pop("beats", None)
    return grid, battues, n_bars


def _stamp(bar: list[dict], bar_no: int, t0: float, t1: float,
           beats: int) -> None:
    """Les temps d'une mesure, et les deux champs qu'un chart ÉCRIT porte.

    `t0`/`t1` sont les bornes de CETTE mesure, prises dans la grille : une
    demi-barre de 2/4 dure deux fois moins qu'une mesure de 4/4, et ses
    accords se placent dans cet intervalle-là.

    `c=1.0` : un accord relevé par un musicien n'a pas d'hésitation — sans lui
    `confColor(undefined)` peint tout le chart en rouge « très faux » dès
    qu'on passe en Annotate. `n=0` : aucun compte d'occurrences à annoncer, et
    `playedLabel` se tait pour 0 (jamais pour 1).
    """
    duree = t1 - t0
    for c in bar:
        beat = c.get("beat", 0)
        suivants = [x for x in bar if x.get("beat", 0) > beat]
        fin = min(x["beat"] for x in suivants) if suivants else beats
        c["bar"] = bar_no
        c["t0"] = round(t0 + beat * duree / beats, 4)
        c["t1"] = round(t0 + fin * duree / beats, 4)
        c["c"] = 1.0
        c["n"] = 0


def _prompter(sections: list[dict], grid: list[float],
              battues: list[int]) -> list[dict]:
    """La liste plate {t0,t1,root,q,bass,nc,c} du prompteur, morceau DÉROULÉ.

    Le prompteur défile dans le temps : il ne se replie pas. Un bloc joué
    quatre fois y apparaît quatre fois, à ses quatre places — donc on repart
    des `barRanges` (un par passage) et on trie par le temps.

    Les temps sortent de `grid`, jamais d'un produit index × durée : une
    demi-barre de 2/4 dure deux fois moins qu'une mesure de 4/4.
    """
    flat: list[dict] = []
    for s in sections:
        L = len(s["bars"])
        endings = s.get("endings")
        tail = endings["tail"] if endings else 0
        prefix_len = L - tail
        variants = endings["variants"] if endings else []
        for p, (b0, _b1) in enumerate(s["barRanges"]):
            corps = list(s["bars"][:prefix_len])
            if variants:
                v = next((v for v in variants if p in v["passes"]), variants[0])
                corps = corps + list(v["bars"])
            for r, bar in enumerate(corps):
                t0, t1 = grid[b0 + r], grid[b0 + r + 1]
                pulses = battues[b0 + r]
                for c in bar:
                    beat = c.get("beat", 0)
                    suivants = [x for x in bar if x.get("beat", 0) > beat]
                    fin = min(x["beat"] for x in suivants) if suivants else pulses
                    flat.append({"t0": round(t0 + beat * (t1 - t0) / pulses, 4),
                                 "t1": round(t0 + fin * (t1 - t0) / pulses, 4),
                                 "root": c["root"], "q": c["q"],
                                 "bass": c.get("bass", -1), "nc": bool(c.get("nc")),
                                 "c": 1.0})
    flat.sort(key=lambda d: (d["t0"], d["t1"]))
    return flat


# ── clé, titre, point d'entrée ───────────────────────────────────────────────

def parse_key(key_str: str) -> dict:
    """'G-' -> {tonic:7, mode:'minor'} ; 'Bb' -> {tonic:10, mode:'major'}."""
    k = (key_str or "C").strip()
    m = re.match(r"^([A-G][#b]?)(-?)", k)
    if not m:
        return {"tonic": 0, "mode": "major"}
    return {"tonic": parse_root(m.group(1)),
            "mode": "minor" if m.group(2) == "-" else "major"}


NOTE_NAMES = ["C", "C#", "D", "E♭", "E", "F", "F#", "G", "A♭", "A", "B♭", "B"]


def key_name(key: dict) -> str:
    return f"{NOTE_NAMES[key['tonic'] % 12]} {key['mode']}"


def slugify(text: str) -> str:
    """Un stem de fichier sûr et lisible, sans accent ni espace."""
    s = unicodedata.normalize("NFKD", text or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return (s or "sans_titre")[:60]


def tune_string_to_chart(tune_string: str, *, file_key: str = "",
                         bpm: float = NOMINAL_BPM) -> dict:
    """Le segment irealb d'UN morceau → un ChartModel prêt à servir.

    `tune_string` est le corps d'une URL `irealb://` déjà déquotée, pour un
    seul morceau (`irealb_fetcher._result_dict` en produit exactement un par
    résultat de recherche — jamais la playlist entière).
    """
    tune = Tune(tune_string)
    raw = raw_chord_string(tune_string)
    bars, num, _den = _measures(raw)   # le PREMIER chiffrage du morceau
    if not bars:
        raise ValueError("cette grille iReal ne contient aucune mesure")

    # `bpb` = le nombre de colonnes d'une mesure dans la grille. C'est le
    # chiffrage PRINCIPAL du morceau ; les mesures qui en changent portent le
    # leur dans `meters` (voir `_lay_out_time`).
    bpb = _pulsations((num, _den))
    blocks = [b for i, blk in enumerate(_sections(bars))
              for b in _block_to_section(blk, i)]
    # Replier D'ABORD (qui est le même bloc ?), nommer ENSUITE (les lettres
    # d'iReal d'abord servies), poser les temps EN DERNIER — dans cet ordre,
    # parce que chaque étape a besoin du résultat de la précédente.
    sections = _fold_blocks(blocks)
    _label_blocks(sections)

    # UNE PULSATION, pas une mesure : c'est l'unité qui survit à un changement
    # de chiffrage en cours de morceau.
    pulse = 60.0 / (bpm or NOMINAL_BPM)
    grid, battues, n_bars = _lay_out_time(blocks, sections, pulse)
    key = parse_key(tune.key)
    title = (tune.title or "").strip() or "Sans titre"
    stem = file_key or f"ireal_{slugify(title)}"

    # Les temps de chaque pulsation, mesure par mesure — une demi-barre de 2/4
    # n'en apporte que deux.
    beat_times = [0.0]
    for i, n in enumerate(battues):
        pas = (grid[i + 1] - grid[i]) / n
        beat_times += [round(grid[i] + (k + 1) * pas, 4) for k in range(n)]
    return {
        "file": stem, "title": title, "video_id": "",
        # Pas d'audio : c'est le fait central d'un chart importé, et toute
        # l'app le lit ici (`S.audioUrl`, `hasAudio` de /api/library).
        "audio_url": "",
        "key": key, "keyName": key_name(key),
        "keySegments": [{"i0": 0, "i1": max(0, n_bars - 1), "t0": 0.0,
                         "t1": grid[-1], "tonic": key["tonic"],
                         "mode": key["mode"], "x3": 0.0,
                         "colours": {"natural": n_bars}}],
        "bpb": bpb, "nBars": n_bars,
        "barGrid": grid, "beatTimes": beat_times,
        "bar1": None, "form": None, "fold": {},
        "sections": sections,
        "prompter": {"chords": _prompter(sections, grid, battues)},
        "meta": {"bpm": round(bpm or NOMINAL_BPM, 2),
                 "n_segments": sum(len(b) for s in sections for b in s["bars"]),
                 "engine": "irealb", "raw": False, "pending": [],
                 "source": "irealb",
                 "composer": (tune.composer or "").strip(),
                 "style": (tune.style or "").strip(),
                 "timeSig": f"{num}/{_den}",
                 # Les mesures qui SORTENT du chiffrage principal — la
                 # demi-barre de 2/4 de *Virtual Insanity*. Un compte, pas une
                 # liste : la position vit dans `sections[].meters`.
                 "meterChanges": sum(
                     1 for s_ in sections for n in s_["meters"] if n != bpb),
                 # Le tempo est NOMINAL, pas mesuré : le dire dans le modèle
                 # évite qu'un lecteur plus tard le prenne pour une mesure.
                 "nominalTempo": True},
    }


def url_to_chart(irealb_url: str, *, file_key: str = "",
                 bpm: float = NOMINAL_BPM) -> dict:
    """`irealb://…` (une URL, un morceau) → ChartModel."""
    decoded = urllib.parse.unquote(irealb_url or "")
    m = re.match(r"irealb://(.+)", decoded, re.S)
    body = m.group(1) if m else decoded
    seg = re.split("===", body)[0]
    if not seg.strip():
        raise ValueError("URL irealb:// vide")
    return tune_string_to_chart(seg, file_key=file_key, bpm=bpm)
