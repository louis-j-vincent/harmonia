"""Le registre des corrections de Louis — ce que la machine disait, ce qu'il écrit.

Louis, 2026-09-18, avant deux semaines d'import et de correction : « je veux
que tu fasses quelque chose qui récupère automatiquement toutes les différences
d'annotation et les persiste quelque part, afin que quand on a récolté assez de
corrections un agent dédié puisse apprendre de mes corrections pour corriger
l'app ».

Ça n'existait pas. Le sidecar d'annotations (`harmonia.annotations`) garde ce
qu'il a ÉCRIT ; il ne garde rien de ce que la machine avait proposé. Or une
correction n'apprend rien sans son avant : « il a mis Fmaj7 » ne dit pas
grand-chose, « la machine disait F6 à 0,41 de confiance, ses trois suggestions
étaient F6 / Fmaj7 / Dm7, il a choisi la deuxième » est une leçon.

CE QUE LE REGISTRE GARDE, par correction :

  * OÙ — le morceau, la mesure, le temps, l'instant en secondes ;
  * AVANT — ce que la machine avait écrit, sa confiance, et ses suggestions
    classées, telles qu'elles étaient au moment où il a corrigé ;
  * APRÈS — ce qu'il a écrit ;
  * LE CONTEXTE — la tonalité, l'accord précédent et le suivant, l'étiquette
    de la section, la position dans la mesure. C'est ce qui permettra de
    chercher des régularités (« il corrige toujours la basse sur le 3e temps
    d'un ii-V ») plutôt que des cas isolés ;
  * QUAND.

FORME : un fichier JSONL par morceau, `state/human/corrections/<stem>.jsonl`,
en AJOUT SEUL. Jamais réécrit, jamais compacté. Une correction qu'il refait
trois fois laisse trois lignes, et c'est une information — il a hésité.

OÙ : dans la moitié de `state/` SUIVIE PAR GIT. Ces lignes sont de la vérité
terrain faite à la main, au même titre que `bass_verdicts.json` : une
annotation à la main a déjà été perdue une fois pour avoir vécu dans un
dossier ignoré (2026-08-12).

CE QUE CE MODULE NE FAIT PAS : apprendre. Il récolte. L'agent qui lira ces
lignes pour proposer une correction de l'app n'existe pas encore, et c'est
voulu — on ne conçoit pas la règle avant d'avoir les exemples.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from harmonia.settings import SETTINGS

log = logging.getLogger("harmonia.corrections")

DOSSIER = SETTINGS.human_dir / "corrections"
#: au-delà, la ligne est tronquée plutôt que refusée — mieux vaut une
#: correction incomplète qu'une correction perdue
MAX_OCTETS = 8 * 1024

NOMS = "C C# D Eb E F F# G Ab A Bb B".split()


def nom_accord(c: dict | None) -> str:
    """Un accord en toutes lettres — ce qu'un humain lirait sur le chart.

    Le registre garde aussi les champs bruts ; ce texte est là pour qu'une
    ligne se lise sans décodeur, y compris par quelqu'un qui ouvre le fichier
    dans six mois.
    """
    if not c:
        return "—"
    if c.get("nc"):
        return "N.C."
    try:
        s = NOMS[int(c.get("root", 0)) % 12]
    except (TypeError, ValueError):
        return "?"
    s += {"": "", "min": "m", "dom": "7", "hdim": "ø", "dim": "°"}.get(
        c.get("q"), str(c.get("q") or ""))
    b = c.get("bass", -1)
    try:
        if b is not None and int(b) >= 0 and int(b) != int(c.get("root", -1)):
            s += "/" + NOMS[int(b) % 12]
    except (TypeError, ValueError):
        pass
    return s


def _fichier(stem: str) -> Path:
    return DOSSIER / f"{Path(str(stem)).stem}.jsonl"


def noter(stem: str, quoi: str, ligne: dict) -> bool:
    """Ajoute une correction au registre. Rend False si elle n'a pas pu être
    écrite — et le DIT dans le journal, jamais en silence.

    `quoi` : "accord" ou "section". `ligne` : le reste, libre — chaque type de
    correction a ses champs, et figer un schéma obligerait à toucher ce module
    à chaque nouvelle sorte de correction.
    """
    try:
        DOSSIER.mkdir(parents=True, exist_ok=True)
        doc = {"quand": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "quoi": quoi, "morceau": stem, **ligne}
        texte = json.dumps(doc, ensure_ascii=False)[:MAX_OCTETS]
        with _fichier(stem).open("a", encoding="utf-8") as f:
            f.write(texte + "\n")
        return True
    except (OSError, TypeError, ValueError) as exc:
        log.warning("correction non enregistrée pour %s (%s) : %s",
                    stem, quoi, exc)
        return False


def lire(stem: str | None = None) -> list[dict]:
    """Les corrections d'un morceau, ou de toute la bibliothèque.

    Une ligne illisible est SAUTÉE et signalée : un fichier en ajout seul peut
    toujours porter une écriture interrompue, et perdre tout le registre pour
    une ligne serait pire que perdre la ligne.
    """
    fichiers = [_fichier(stem)] if stem else sorted(DOSSIER.glob("*.jsonl"))
    out = []
    for f in fichiers:
        if not f.exists():
            continue
        for n, brut in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not brut.strip():
                continue
            try:
                out.append(json.loads(brut))
            except ValueError:
                log.warning("registre %s ligne %d illisible, sautée", f.name, n)
    return out


def compter() -> dict:
    """Combien de corrections, de quel type, sur combien de morceaux.

    Sert à répondre à « en a-t-on assez pour qu'un agent en apprenne quelque
    chose ? » sans relire tout le registre à la main.
    """
    lignes = lire()
    par_type: dict[str, int] = {}
    for l in lignes:
        par_type[l.get("quoi", "?")] = par_type.get(l.get("quoi", "?"), 0) + 1
    return {"total": len(lignes), "par_type": par_type,
            "morceaux": len({l.get("morceau") for l in lignes if l.get("morceau")}),
            "fichiers": len(list(DOSSIER.glob("*.jsonl"))) if DOSSIER.exists() else 0}


# ── ce qui fabrique les lignes ──────────────────────────────────────────────

def _cle(bar, beat) -> tuple:
    try:
        return int(bar), round(float(beat or 0), 3)
    except (TypeError, ValueError):
        return bar, beat


def accords_du_chart(model: dict) -> dict:
    """{(mesure, temps) → accord} tel que la MACHINE l'a écrit.

    On lit le chart cuit, pas le modèle servi : le modèle servi a déjà reçu
    les corrections précédentes de Louis (`annotations.overlay`), donc il ne
    dirait plus ce que la machine pensait.
    """
    out = {}
    for sec in model.get("sections") or []:
        for bar in sec.get("bars") or []:
            for ch in bar:
                out[_cle(ch.get("bar"), ch.get("beat"))] = ch
    return out


def note_corrections_accords(stem: str, model: dict, fixes: list[dict]) -> int:
    """Compare ses accords confirmés à ce que la machine avait écrit.

    Rend le nombre de lignes ajoutées. Un accord qu'il confirme SANS le
    changer n'est pas une correction — c'est un accord validé, et ça n'apprend
    rien sur les erreurs de la machine ; on ne l'enregistre pas.
    """
    machine = accords_du_chart(model)
    ordre = sorted(machine)
    n = 0
    for fix in fixes or []:
        if fix.get("root") is None:
            continue
        cle = _cle(fix.get("bar"), fix.get("beat"))
        avant = machine.get(cle)
        apres = {"root": fix.get("root"), "q": fix.get("q", ""),
                 "bass": fix.get("bass", -1), "nc": bool(fix.get("nc"))}
        if avant is not None and (int(avant.get("root", -1)) == int(apres["root"])
                                  and (avant.get("q") or "") == (apres["q"] or "")
                                  and int(avant.get("bass", -1)) == int(apres["bass"])
                                  and bool(avant.get("nc")) == apres["nc"]):
            continue                       # confirmé à l'identique : rien appris
        i = ordre.index(cle) if cle in ordre else None
        n += noter(stem, "accord", {
            "mesure": cle[0], "temps": cle[1],
            "t0": (avant or {}).get("t0"), "t1": (avant or {}).get("t1"),
            "avant": None if avant is None else {
                "texte": nom_accord(avant),
                "root": avant.get("root"), "q": avant.get("q"),
                "bass": avant.get("bass"), "nc": bool(avant.get("nc")),
                "confiance": avant.get("c"),
                "suggestions": avant.get("sug")},
            "apres": {"texte": nom_accord(apres), **apres},
            "contexte": {
                "tonalite": (model.get("key") or {}).get("tonic"),
                "mode": (model.get("key") or {}).get("mode"),
                "bpm": (model.get("meta") or {}).get("bpm"),
                "precedent": nom_accord(machine.get(ordre[i - 1]))
                             if i not in (None, 0) else None,
                "suivant": nom_accord(machine.get(ordre[i + 1]))
                           if i is not None and i + 1 < len(ordre) else None,
                "section": _section_de(model, cle[0]),
            }})
    if n:
        log.info("registre : %d correction(s) d'accord notée(s) pour %s", n, stem)
    return n


def _section_de(model: dict, mesure) -> str | None:
    try:
        m = int(mesure)
    except (TypeError, ValueError):
        return None
    for sec in model.get("sections") or []:
        for a, b in (sec.get("barRanges") or []):
            if a <= m <= b:
                return sec.get("label")
    return None


def note_corrections_sections(stem: str, avant: list[dict],
                              apres: list[dict], n_mesures: int,
                              sources: dict | None = None) -> int:
    """Compare le découpage d'avant à celui qu'il vient de valider.

    On enregistre une ligne par MESURE dont l'étiquette change, groupée en
    plages contiguës : c'est la forme qui se lit et qui se compte, alors qu'une
    ligne par section rendrait incomparables deux découpages aux frontières
    différentes.
    """
    def par_mesure(secs):
        out = [None] * n_mesures
        for s in secs or []:
            for a, b in (s.get("barRanges") or []):
                for i in range(max(0, a), min(n_mesures, b + 1)):
                    out[i] = s.get("label")
        return out

    av, ap = par_mesure(avant), par_mesure(apres)
    n, i = 0, 0
    while i < n_mesures:
        if av[i] == ap[i]:
            i += 1
            continue
        j = i
        while j + 1 < n_mesures and av[j + 1] == av[i] and ap[j + 1] == ap[i]:
            j += 1
        n += noter(stem, "section", {
            "mesure_debut": i + 1, "mesure_fin": j + 1,
            "avant": av[i], "apres": ap[i],
            "source": (sources or {}).get(ap[i]),
        })
        i = j + 1
    if n:
        log.info("registre : %d correction(s) de section notée(s) pour %s",
                 n, stem)
    return n
