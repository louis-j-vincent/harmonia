"""Le rôle d'un accord dans une cadence, et l'équivalence de deux accords qui
tiennent le même rôle.

Louis, 2026-09-15 : « il va falloir faire une fonction ou un mapping pour
tester si deux accords sont équivalents au sein d'une cadence : sur un 2-5-1
en C majeur, G7 a le même rôle que F/G (qu'on considérera comme un Gsus),
que G7b9 et alternativement que Db7 … afin de permettre un repliement
logique de voisins équivalents si une section a plusieurs passes avec des
variations (commun en jazz / soul / neo soul / funk). »

Quatre règles, dans cet ordre. Les trois premières ne regardent pas la
tonalité ; la quatrième en a besoin.

  1. LA COULEUR NE CHANGE PAS LE RÔLE. Même fondamentale fonctionnelle et
     même famille → équivalents. Les familles : majeure (C, C^7, C6, C^9),
     mineure (D-, D-7, D-9, D-6), dominante (G7, G9, G13, G7b9, G7#9, Gsus4,
     G7sus4, G+), demi-diminuée (Dh7), diminuée (voir 3).
  2. UNE BASSE ÉTRANGÈRE FAIT LA FONDAMENTALE. X/Y où Y n'est PAS une note
     de X est un accord « sus » sur Y : F/G = Gsus = G7sus4, D-7/G = G11,
     Bb/C = C9sus. Une basse qui EST une note de X est un renversement, et
     ne change rien : C/E ≡ C, D-/F ≡ D-.
  3. LE TRITON FAIT LA DOMINANTE. Deux dominantes qui partagent le même
     triton sont substituables : G7 (si-fa) ≡ Db7 (fa-dob). Un accord
     diminué est une dominante b9 sans fondamentale : Bo7 = Bo ≡ G7b9.
  4. AVEC LA TONALITÉ (trois règles candidates, vues sur la page
     `tools/roles_page`, à arbitrer) : un accord majeur sans septième sur
     le Ve degré est une dominante (G en C ≡ G7) ; un demi-diminué sur la
     sensible est un V9 sans fondamentale (Dh7 en Eb mineur ≡ Bb9 —
     Virtual Insanity) ; une septième mineure sur le Ier degré est une
     couleur du tonique (A7 ≡ A^7 sur The Walk), pas une dominante. Sans
     tonalité, rien de tout ça — on ne devine pas.

Et une identité qui ne demande aucune théorie : deux accords qui SONNENT les
mêmes notes sont le même accord (D-7 = F6, A-7 = C6, Dh7 = F-6).

Ce que ce module ne fait PAS (encore) : lire le contexte (ce qui précède et
ce qui suit) — un E- sur le IIIe degré et un C^7 sur le Ier tiennent
souvent le même rôle tonique, mais pas toujours ; la règle 1 les garde
distincts, et Louis tranchera sur la page `tools/roles_page`. `role` et
`equivalent` ne sont pas branchés dans le repli : c'est une brique à valider à
l'oreille avant d'en faire une loi.

Ce que le repli utilise, lui, vit en bas de ce module : `meme_lecture`, la
SEULE comparaison « même accord » de `harmonia.folding` (2026-10-01).
"""
from __future__ import annotations

from harmonia.labels import chord_pcs

NOTES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

#: Les notes sonnées par chaque queue iReal (intervalles depuis la
#: fondamentale). Complète `labels.chord_pcs` pour les queues que le
#: décodeur n'émet pas mais que l'annotation ou iReal écrivent.
_PCS = {
    "": (0, 4, 7), "^7": (0, 4, 7, 11), "6": (0, 4, 7, 9), "69": (0, 4, 7, 9, 2),
    "^9": (0, 4, 7, 11, 2), "add9": (0, 4, 7, 2), "^7#11": (0, 4, 7, 11, 6),
    "-": (0, 3, 7), "-7": (0, 3, 7, 10), "-9": (0, 3, 7, 10, 2), "-6": (0, 3, 7, 9),
    "-11": (0, 3, 7, 10, 5), "-^7": (0, 3, 7, 11),
    "7": (0, 4, 7, 10), "9": (0, 4, 7, 10, 2), "13": (0, 4, 7, 10, 2, 9),
    "7b9": (0, 4, 7, 10, 1), "7#9": (0, 4, 7, 10, 3), "7#11": (0, 4, 7, 10, 6),
    "7b13": (0, 4, 7, 10, 8), "7alt": (0, 4, 10, 1, 8), "+": (0, 4, 8),
    "7#5": (0, 4, 8, 10), "sus4": (0, 5, 7), "7sus4": (0, 5, 7, 10),
    "9sus4": (0, 5, 7, 10, 2), "sus2": (0, 2, 7),
    "o": (0, 3, 6), "o7": (0, 3, 6, 9), "h7": (0, 3, 6, 10),
}

#: Queue iReal → famille de rôle.
_FAMILY = {
    "maj": ("", "^7", "6", "69", "^9", "add9", "^7#11", "sus2"),
    "min": ("-", "-7", "-9", "-6", "-11", "-^7"),
    "dom": ("7", "9", "13", "7b9", "7#9", "7#11", "7b13", "7alt", "+", "7#5",
            "sus4", "7sus4", "9sus4"),
    "dim": ("o", "o7"),
    "hdim": ("h7",),
}
_FAMILY_OF = {q: fam for fam, qs in _FAMILY.items() for q in qs}


def pcs(chord: dict) -> frozenset[int]:
    """Les notes que l'accord sonne, basse comprise."""
    root, q = int(chord["root"]), chord.get("q") or ""
    rel = _PCS.get(q)
    out = {(root + r) % 12 for r in rel} if rel else set(chord_pcs(root, q))
    bass = chord.get("bass", -1)
    if bass is not None and bass >= 0:
        out.add(int(bass) % 12)
    return frozenset(out)


def family(q: str) -> str:
    """majeure / mineure / dominante / diminuée / demi-diminuée."""
    if q in _FAMILY_OF:
        return _FAMILY_OF[q]
    if q.startswith("-"):
        return "min"
    if q.startswith("^") or q == "":
        return "maj"
    if q.startswith("h"):
        return "hdim"
    if q.startswith("o"):
        return "dim"
    return "dom"


def role(chord: dict, key: tuple[int, str] | None = None) -> tuple:
    """Le rôle : ("dom", (triton)) pour une dominante, (famille, fondamentale)
    sinon. Deux rôles égaux = deux accords équivalents dans une cadence.

    `key` = (tonique, "major"|"minor") ou None ; ne sert qu'à la règle 4.
    """
    root, q = int(chord["root"]) % 12, chord.get("q") or ""
    bass = chord.get("bass", -1)
    fam = family(q)
    fr = root
    # 2. basse étrangère → accord sus sur la basse, famille dominante
    if bass is not None and 0 <= bass < 12 and bass != root:
        upper = {(root + r) % 12 for r in (_PCS.get(q) or chord_pcs(root, q))}
        if bass not in upper:
            fr, fam = int(bass), "dom"
    # 3. diminué = dominante b9 sans fondamentale (Bo7 → G7b9 : fr = r − 4)
    if fam == "dim":
        fr, fam = (fr - 4) % 12, "dom"
    # 4. avec la tonalité (candidates, à arbitrer sur tools/roles_page) :
    if key is not None:
        tonic = int(key[0]) % 12
        # 4a. majeur sans septième sur le Ve degré = dominante (G en C ≡ G7)
        if fam == "maj" and q in ("", "sus2") and fr == (tonic + 7) % 12:
            fam = "dom"
        # 4b. demi-diminué sur la sensible = V9 sans fondamentale (Dh7 en Eb
        #     mineur ≡ Bb9 : viiø7 → V). Sur le IIe degré, il reste un iiø.
        elif fam == "hdim" and fr == (tonic + 11) % 12:
            fr, fam = (fr - 4) % 12, "dom"
        # 4c. septième mineure sur le Ier degré = une couleur du tonique
        #     (A7 ≡ A^7 ≡ A sur le I de The Walk), pas une dominante.
        elif fam == "dom" and q in ("7", "9", "13") and fr == tonic:
            fam = "maj"
    if fam == "dom":
        return ("dom", tuple(sorted(((fr + 4) % 12, (fr + 10) % 12))))
    return (fam, fr)


def equivalent(a: dict, b: dict, key: tuple[int, str] | None = None) -> bool:
    """Vrai si les deux accords tiennent le même rôle (règles 1-4), ou
    sonnent exactement les mêmes notes (D-7 = F6)."""
    if a.get("nc") or b.get("nc"):
        return bool(a.get("nc")) and bool(b.get("nc"))
    return role(a, key) == role(b, key) or pcs(a) == pcs(b)


def role_name(r: tuple, key: tuple[int, str] | None = None) -> str:
    """Un nom lisible : « V (G7 / Db7) », « min D », « maj C »."""
    if r[0] == "dom":
        t = r[1]
        roots = sorted({(t[0] - 4) % 12, (t[1] - 4) % 12})
        names = " / ".join(NOTES[x] + "7" for x in roots)
        if key is not None and (int(key[0]) + 7) % 12 in roots:
            return f"V ({names})"
        return f"dom ({names})"
    fam, fr = r
    return {"maj": "maj", "min": "min", "hdim": "ø"}.get(fam, fam) + " " + NOTES[fr]


def parse_key_name(name: str | None) -> tuple[int, str] | None:
    """« Eb major » → (3, "major") ; None si illisible."""
    if not name:
        return None
    parts = name.split()
    if len(parts) != 2:
        return None
    try:
        from harmonia.labels import parse_root
        return (parse_root(parts[0]), parts[1].lower())
    except (KeyError, IndexError):
        return None


# ── LA comparaison « même accord » du repli ───────────────────────────────
#
# Louis, 2026-10-01 : une mesure de similarité entre accords, générale et
# reproductible, viendra remplacer les règles écrites à la main (m7 ≡ 7sus4
# une quarte au-dessus, etc.). Pour qu'elle n'ait qu'UN endroit où se
# brancher, toutes les questions « ces deux accords sont-ils le même accord,
# ces deux passages le même passage ? » du repli (fins 1./2., boucle
# minimale, cascade iReal, fusion des lettres, petite lecture d'une variante)
# passent par `meme_lecture`. Son comportement aujourd'hui = les règles
# d'avant, réunies sans rien changer : `lectures` (Bb/C se lit aussi
# C7sus4, la basse d'un renversement ne compte pas), puis une granularité.

def famille_de(q: str | None) -> str:
    """maj / min / dim / aug — la granularité à laquelle `B-` et `B-7` sont le
    même moment musical, et `B-` et `B` deux moments différents."""
    q = q or ""
    if q.startswith(("o", "h")) or q.startswith("-7b5"):
        return "o"
    if q.startswith("-"):
        return "m"
    if q.startswith("+"):
        return "+"
    return "M"



def accord_sonnant(c: dict) -> tuple[int, str]:
    """(fondamentale, qualité) de l'accord tel qu'il SONNE sur sa basse.

    Louis, 2026-09-30, sur Claustrophobic : « t'as différencié un A où il y a
    un Bb/C d'un A où il y a un C7sus4 alors qu'essentiellement c'est la même
    chose ». Bb/C, c'est Bb-D-F sur un do : do, 9, 11, b7 — un C9sus4. Même
    chose pour F/G (G9sus4), Ab/Bb (Bb9sus4), G-7/C (C11).

    LA RÈGLE, par le contenu et non par une liste : une basse ÉTRANGÈRE à
    l'accord (pas un renversement), et, lues depuis cette basse, une septième
    mineure sans aucune tierce → un accord sus sur la basse, écrit `7sus4`
    (la 9 est une couleur : `9sus4` s'écrit `7sus4` aussi).

    CE QUE ÇA NE FAIT PAS, exprès : une basse étrangère qui fait entendre une
    tierce reste l'accord écrit — Eb7/C sonne C-7(b9), un accord MINEUR, et
    `roles.role` (règle 2, « toute basse étrangère = sus ») l'aurait appelé
    dominante. Un renversement (C/E) ne change rien. D/C (pas de b7 au-dessus
    du do) ne change rien. Si Louis veut Eb7/C ≡ C-7, c'est une autre règle,
    à arbitrer à part.
    """
    root, q = int(c["root"]) % 12, c.get("q") or ""
    if q == "9sus4":
        q = "7sus4"
    b = c.get("bass", -1)
    if c.get("nc") or b is None or b < 0 or b % 12 == root:
        return root, q
    b %= 12
    up = {(root + r) % 12 for r in (_PCS.get(q) or chord_pcs(root, q))}
    if b in up:
        return root, q
    rel = {(p - b) % 12 for p in up}
    if 10 in rel and 3 not in rel and 4 not in rel:
        return b, "7sus4"
    return root, q


def lectures(c: dict) -> frozenset:
    """Les façons LÉGITIMES de lire un accord : {(fondamentale, qualité)}.

    Un accord ordinaire n'en a qu'une — l'accord écrit (sa basse ignorée,
    comme avant : Bb/D ≡ Bb). Un sus-sur-basse (`accord_sonnant`) en a DEUX :
    la triade écrite ET le sus sur la basse — Bb/C se lit Bb (comme avant) OU
    C7sus4 (Louis, 2026-09-30). Deux accords se valent s'ils partagent une
    lecture. Ce n'est PAS une égalité (Bb ≡ Bb/C ≡ C7sus4 n'entraîne pas
    Bb ≡ C7sus4) : c'est voulu, et c'est pour ça qu'on compare par
    `_compat` et jamais par `==`.

    Conséquence mesurée (rapport d'or 2026-09-30) : une première version
    remplaçait la lecture écrite par la lecture sonnée ; elle CASSAIT un
    raccord qui marchait (9zEl-FQLI4A : `G/A` d'un passage contre `G^7` de
    l'autre, un A′ apparaissait). Avec deux lectures, l'équivalence ne fait
    qu'AJOUTER des accords — jamais en retirer un.
    """
    r, q = int(c["root"]) % 12, c.get("q") or ""
    if q == "9sus4":
        q = "7sus4"
    return frozenset({(r, q), accord_sonnant(c)})


#: Les granularités de `meme_lecture`, de la plus fine à la plus large.
#: Deux systèmes de familles coexistent, hérités : `famille_de` (M/m/o/+ —
#: une 7 de dominante ou une sus est une couleur du majeur) pour reconnaître
#: un même passage, et `family` (maj/min/dom/dim/hdim — une dominante n'est
#: pas un majeur) pour la petite lecture d'une variante. Les réunir changerait
#: des décisions : c'est le travail de la future mesure apprise, pas d'ici.
GRAINS = ("exact", "fonction", "famille", "racine")


def _cles(c: dict, grain: str) -> frozenset:
    ls = lectures(c)
    if grain == "exact":
        return ls
    if grain == "fonction":
        return frozenset((r, family(q)) for r, q in ls)
    if grain == "famille":
        return frozenset((r, famille_de(q)) for r, q in ls)
    if grain == "racine":
        return frozenset(r for r, _ in ls)
    raise ValueError(f"grain inconnu : {grain!r} (attendu : {GRAINS})")


def meme_lecture(a: dict, b: dict, contexte: dict | None = None,
                 grain: str = "famille") -> bool:
    """Deux accords sont-ils LE MÊME ACCORD pour le repli ?

    LE point d'entrée unique de toutes les comparaisons « même accord /
    même passage » de `harmonia.folding` (liste dans la docstring du
    module, section « La comparaison »). Aujourd'hui : deux N.C. se valent,
    un N.C. ne vaut aucun accord ; sinon les deux accords se valent s'ils
    partagent une lecture (`lectures`) à la granularité `grain` :

      - "exact"    : même fondamentale, même qualité (C7 ≠ C9, C ≠ C^7) —
                     sert à repérer une VRAIE différence (fins 1./2.) ;
      - "fonction" : même fondamentale, même `family` (C ≡ C^7, C ≢ C7) —
                     la petite lecture d'une variante ;
      - "famille"  : même fondamentale, même `famille_de` (C ≡ C^7 ≡ C7 ≡
                     C7sus4, D- ≡ D-7) — reconnaître un même passage ;
      - "racine"   : même fondamentale lisible — une fin qui change de
                     fondamentale quelque part.

    Ce n'est PAS une égalité (Bb ≡ Bb/C ≡ C7sus4 n'entraîne pas
    Bb ≡ C7sus4) : on compare toujours par paires, jamais par `==` sur des
    clés.

    `contexte` : réservé à la mesure apprise qui remplacera ces règles
    (tonalité, position dans la cadence, postérieures musx…). Ignoré
    aujourd'hui. CE QUE ÇA NE FAIT PAS (encore) : m7 ≡ 7sus4 une quarte
    au-dessus (D-7 ≡ G7sus4, Lost Without U) — Louis veut que ça sorte de la
    mesure apprise, pas d'une règle de plus.
    """
    del contexte
    na, nb = bool(a.get("nc")), bool(b.get("nc"))
    if na or nb:
        return na and nb
    return bool(_cles(a, grain) & _cles(b, grain))
