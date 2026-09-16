"""Confronter un diff de charts aux tabs Ultimate Guitar / iReal Pro.

Louis, 2026-09-16 : « si tu as un doute pour trancher, vas chercher les guitar
tabs ou irealpro tabs s'ils sont dispos et fais un matching pour comparer, en
mettant dans la bonne tona[lité] ».

L'ordre de confiance du projet reste iReal > tabs > modèle (CLAUDE.md), et la
tab ne tranche JAMAIS toute seule : elle apporte une pièce, Louis arbitre.

## La tonalité, sans la croire sur parole

Une tab de guitare est écrite en POSITIONS, pas en notes : quatre des seize
tabs récoltées sont capodastrées et trois autres simplement transposées par
leur auteur. Se fier au champ « capo » serait exactement l'erreur numéro 1 du
projet (une constante silencieusement fausse qui produit des chiffres
plausibles).

On ne lui fait donc pas confiance : on essaie **les douze rotations** et on
garde celle qui fait le mieux coïncider le vocabulaire d'accords de la tab
avec celui du chart. La rotation gagnante EST la transposition, et l'écart
avec la capo déclarée est un contrôle — s'ils divergent, on le signale.

## Ce que la tab peut dire, et ce qu'elle ne peut pas

Le diff du jour ne porte que sur des basses de slash (`G-` -> `G-/Bb`). Pour
ça, l'alignement mesure à mesure est inutile et fragile : la question est de
vocabulaire. Pour chaque accord litigieux on demande à la tab :

  * elle écrit CETTE basse           -> elle soutient le slash
  * elle écrit cet accord SANS basse, et elle utilise des slashes ailleurs
                                     -> elle soutient l'accord nu
  * elle écrit cet accord avec une AUTRE basse -> elle contredit les deux
  * elle n'utilise aucun slash       -> elle ne dit RIEN (une tab qui ignore
                                        les renversements n'est pas un témoin)

Ce dernier cas est le piège : une tab sans aucun slash n'est pas une tab qui
dit « pas de slash », c'est une tab qui ne s'exprime pas sur la question.

    .venv/bin/python -m tools.tab_match --before DIR --after DIR --out FILE
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from harmonia.settings import SETTINGS

NOTES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
PC = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "FB": 4,
      "E#": 5, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9,
      "A#": 10, "BB": 10, "B": 11, "CB": 11, "B#": 0}

#: `[ch]Am7/G[/ch]` — la balise d'Ultimate Guitar autour de chaque accord.
_CH = re.compile(r"\[ch\](.*?)\[/ch\]", re.S)
#: racine, altération, le reste, et la basse éventuelle après le slash
_SYM = re.compile(r"^([A-Ga-g])([#b]?)([^/\s]*)(?:/([A-Ga-g][#b]?))?$")


def parse_symbol(sym: str) -> dict | None:
    """« Am7/G » -> {root: 9, qual: 'm7', bass: 7}. None si illisible."""
    s = sym.strip().replace("♭", "b").replace("♯", "#")
    m = _SYM.match(s)
    if not m:
        return None
    root_name = (m.group(1) + m.group(2)).upper()
    if root_name not in PC:
        return None
    bass = None
    if m.group(4):
        bn = m.group(4).upper()
        if bn not in PC:
            return None
        bass = PC[bn]
    return {"root": PC[root_name], "qual": m.group(3), "bass": bass}


def famille(qual: str) -> str:
    """Ramener une qualité écrite à sa famille : maj / min / dom / dim.

    Grossier VOLONTAIREMENT : on compare une tab de guitare à un chart, et
    l'accord de passage à la septième n'est pas le désaccord qu'on cherche.
    """
    q = qual.lower().replace("maj", "^").replace("major", "^")
    if q.startswith(("m", "-")) and not q.startswith("maj"):
        return "min"
    if q.startswith(("dim", "o", "°")):
        return "dim"
    if q.startswith(("aug", "+")):
        return "aug"
    if re.match(r"^\d", q) or q.startswith(("7", "9", "11", "13")):
        return "dom"
    return "maj"


def tab_chords(raw: str) -> list[dict]:
    """La suite ORDONNÉE des accords d'une tab, symboles illisibles retirés."""
    out = []
    for sym in _CH.findall(raw or ""):
        p = parse_symbol(sym)
        if p:
            p["ecrit"] = sym.strip()
            out.append(p)
    return out


def chart_chords(model: dict) -> list[dict]:
    """La suite des accords d'un chart, en espace CHANSON (repli déplié)."""
    out = []
    for sec in model.get("sections") or []:
        for b0, b1 in sec.get("barRanges") or []:
            for bi, bar in enumerate(sec.get("bars") or []):
                if b0 + bi >= b1:
                    break
                for c in bar:
                    if c.get("carry") or c.get("nc"):
                        continue
                    r = int(c["root"]) % 12
                    b = c.get("bass", -1)
                    out.append({"root": r, "qual": c.get("q", ""),
                                "bass": (b % 12) if (b is not None and b >= 0) else None,
                                "bar": b0 + bi + 1})
    return out


def _qual_chart_to_famille(q: str) -> str:
    """Les qualités du chart (`-7`, `^7`, `7`, `ø`) vers les mêmes familles."""
    if q.startswith("-") or q.startswith("m"):
        return "min"
    if q.startswith("^"):
        return "maj"
    if q.startswith(("o", "ø")):
        return "dim"
    if q.startswith(("7", "9", "11", "13")):
        return "dom"
    return "maj"


def meilleure_rotation(tab: list[dict], chart: list[dict]) -> tuple[int, float, list[float]]:
    """La transposition qui fait le mieux coïncider les deux vocabulaires.

    On compare des SACS (root, famille) pondérés par le nombre d'occurrences,
    pas des suites : une tab écrit les couplets une fois et les répète par
    consigne, un chart les déplie. Score = recouvrement de Jaccard pondéré.
    """
    ct = Counter((c["root"], famille(c["qual"])) for c in tab)
    cc = Counter((c["root"], _qual_chart_to_famille(c["qual"])) for c in chart)
    if not ct or not cc:
        return 0, 0.0, [0.0] * 12
    scores = []
    for k in range(12):
        rot = Counter(((r + k) % 12, f) for (r, f), n in ct.items() for _ in range(n))
        inter = sum((rot & cc).values())
        union = sum((rot | cc).values())
        scores.append(inter / union if union else 0.0)
    best = max(range(12), key=lambda k: scores[k])
    return best, scores[best], scores


#: sous ce taux de slashes, la tab ne s'exprime pas sur les renversements.
#: Sam Smith écrit UN slash sur 150 accords (0,7 %) : compter ses 149 accords
#: nus comme autant de votes « pas de basse » serait lire du silence comme
#: une opinion. Les tabs qui notent vraiment les renversements sont à 15-22 %.
SLASH_MIN = 0.05


def interroger(tab: list[dict], k: int, root: int, fam: str, bass: int | None) -> dict:
    """Ce que la tab dit d'un accord donné, une fois transposée de `k`.

    `bass` est la basse que le chart veut écrire (None = accord nu).
    """
    mm = [c for c in tab
          if (c["root"] + k) % 12 == root and famille(c["qual"]) == fam]
    taux = (sum(1 for c in tab if c["bass"] is not None) / len(tab)) if tab else 0.0
    slashes_partout = taux >= SLASH_MIN
    if not mm:
        return {"verdict": "absent", "n": 0, "ecrits": [],
                "taux": round(taux, 3), "slashes": slashes_partout}
    bases = Counter(((c["bass"] + k) % 12) if c["bass"] is not None else None
                    for c in mm)
    ecrits = sorted({c["ecrit"] for c in mm})[:6]
    if not slashes_partout:
        v = "muette"                      # la tab n'écrit jamais de basse
    elif bass in bases and bass is not None:
        v = "soutient le slash"
    elif bass is not None and None in bases and len(bases) == 1:
        v = "soutient l'accord nu"
    elif bass is not None:
        v = "autre basse"
    elif any(b is not None for b in bases):
        v = "la tab met un slash"
    else:
        v = "soutient l'accord nu"
    return {"verdict": v, "n": len(mm), "ecrits": ecrits, "taux": round(taux, 3),
            "bases": {(NOTES[b] if b is not None else "—"): n for b, n in bases.items()},
            "slashes": slashes_partout}


#: l'index de la récolte : c'est lui qui porte la capo déclarée par la tab.
_INDEX = {r["chart_key"]: r for r in json.loads(
    (SETTINGS.repo / "scratchpad" / "tabrefs" / "_index.json").read_text(encoding="utf-8"))} \
    if (SETTINGS.repo / "scratchpad" / "tabrefs" / "_index.json").exists() else {}


def charger_ref(chart_key: str) -> dict | None:
    p = SETTINGS.repo / "scratchpad" / "tabrefs" / f"{chart_key}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--before", type=Path, required=True)
    ap.add_argument("--after", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import sys
    sys.path.insert(0, str(SETTINGS.repo / "tools"))
    from avant_apres import diff_song

    sorties = []
    for p in sorted(a.after.glob("min_*.json")):
        q = a.before / p.name
        if not q.exists():
            continue
        old = json.loads(q.read_text(encoding="utf-8"))
        new = json.loads(p.read_text(encoding="utf-8"))
        rows, _ = diff_song(old, new)
        if not rows:
            continue
        ref = charger_ref(p.stem)
        ug = (ref or {}).get("ug") or {}
        if not ug.get("found"):
            sorties.append({"song": p.stem, "title": new.get("title"),
                            "tab": None, "rows": rows})
            continue
        tab = tab_chords(ug.get("raw_content", ""))
        chart = chart_chords(new)
        k, score, tous = meilleure_rotation(tab, chart)
        capo = int(_INDEX.get(p.stem, {}).get("ug_capo_semitones") or 0)

        # chaque ligne de diff, passée à la tab
        lignes = []
        for bar, avant, apres in rows:
            for tok_a, tok_b in zip(avant.split(), apres.split()):
                if tok_a == tok_b:
                    continue
                pa, pb = parse_symbol(tok_a.strip("()")), parse_symbol(tok_b.strip("()"))
                if not pb:
                    continue
                fam = _qual_chart_to_famille(pb["qual"])
                lignes.append({
                    "bar": bar, "avant": tok_a, "apres": tok_b,
                    "accord": NOTES[pb["root"]] + pb["qual"],
                    "basse_apres": NOTES[pb["bass"]] if pb["bass"] is not None else None,
                    "basse_avant": NOTES[pa["bass"]] if pa and pa["bass"] is not None else None,
                    "tab": interroger(tab, k, pb["root"], fam, pb["bass"]),
                    # la tab soutient-elle plutôt l'AVANT ? C'est la phrase la
                    # plus actionnable pour Louis, et elle se lit sur les mêmes
                    # occurrences : inutile de la lui faire déduire.
                    "tab_avant": (interroger(tab, k, pa["root"],
                                             _qual_chart_to_famille(pa["qual"]), pa["bass"])
                                  if pa else None),
                })
        sorties.append({
            "song": p.stem, "title": new.get("title"),
            "tab": {"url": ug.get("url"), "rating": ug.get("rating"),
                    "votes": ug.get("votes"), "artist": ug.get("artist_name"),
                    "song_name": ug.get("song_name"),
                    "rotation": k, "score": round(score, 3),
                    "capo_declare": capo,
                    "accord_rotation_capo": (k == (capo % 12)) if capo else None,
                    "n_accords": len(tab),
                    "taux_slashes": round(
                        (sum(1 for c in tab if c["bass"] is not None) / len(tab))
                        if tab else 0.0, 3),
                    "utilise_des_slashes":
                        (sum(1 for c in tab if c["bass"] is not None) / len(tab)
                         if tab else 0.0) >= SLASH_MIN},
            "rows": rows, "lignes": lignes,
        })
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(sorties, ensure_ascii=False, indent=1), encoding="utf-8")
    n = sum(len(s.get("lignes") or []) for s in sorties)
    print(f"→ {a.out}  ·  {len(sorties)} morceaux, {n} accords confrontés")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
