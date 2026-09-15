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
distincts, et Louis tranchera sur la page `tools/roles_page`. Rien ici
n'est branché dans le repli : c'est une brique à valider à l'oreille avant
d'en faire une loi.
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
