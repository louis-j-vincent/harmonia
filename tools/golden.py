"""Le rapport d'or : rejoue la bibliothèque avec un moteur et compare à une baseline.

UNE SEULE définition de « chart = f(entrées) », la même que
`scripts/rebake_library.py` (2026-08-20), qu'il remplacera au sprint 20 :
analyse complète avec la marque « Set bar 1 » de Louis rejouée (depuis le
sprint 15, `state/human/marks/<stem>.json` pour le moteur `harmonia` — voir
`jobs.bar1_for`), puis, quand un découpage de sections validé à la main
existe (`sections/<stem>.json` du moteur, `validated: true`), le repli est
refait sur CE découpage (`refold` + `sections_pour_chart`), les mesures non
couvertes recevant l'étiquette du détecteur (`_completer`).

Ce n'est PAS une porte qui bloque : c'est un rapport que Louis arbitre
(2026-09-14 : « a different chart doesn't mean you did bad work, you will have
to tell me what the differences are and I'll arbitrate »). Zéro mesure changée
= rien à arbitrer ; sinon `tools.avant_apres` fait la page AVANT/APRÈS.

Ce que ce script ne fait JAMAIS : calculer un cache froid. Battues, posteriors
musx, chroma NNLS et sections songformer doivent déjà être sur le disque pour
chaque morceau, sinon le morceau est marqué « froid » et sauté. Un chart d'or
qui dépendrait d'un modèle relancé ne serait pas comparable d'un sprint à
l'autre (et musx sur MPS n'est pas bit-à-bit reproductible).

Usage, depuis la racine du worktree :
    python -m tools.golden --engine harmonia --out …/A
    python -m tools.golden --engine harmonia --out …/B --baseline …/A
    python -m tools.golden --engine harmonia --out …/sprint07 --baseline …/baseline

Sortie : `<out>/<clé>.json` par morceau, `<out>/_report.json`, `<out>/_golden.log`.
Code de retour 1 seulement sur ERREUR (exception) ; froid ou différent = 0.

`--publish` (sprint 20, remplace `scripts/rebake_library.py`) : cuit comme
d'habitude dans `--out`, puis, chart par chart, déplace chaque réussite dans
la bibliothèque RÉELLE du moteur (`eng["charts"]`) — jamais tout d'un coup :
une interruption au milieu laissait sinon une bibliothèque mi-ancienne
mi-nouvelle, indistinguable d'un bug. Comme l'ancien script, refuse de
démarrer sans `--backup <dossier existant>` (une session concurrente peut
travailler sur les mêmes fichiers) sauf en `--dry-run`, qui cuit et affiche
sans rien publier.

    python -m tools.golden --engine harmonia --out state/cache/golden/pub \\
        --publish --backup docs/archive/charts.bak_20260914
    python -m tools.golden --engine harmonia --out /tmp/x --publish --dry-run
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import shutil
import sys
import time
import traceback
from pathlib import Path

from harmonia.settings import SETTINGS
from tools.avant_apres import diff_song

REPO = SETTINGS.repo
AUDIO = SETTINGS.audio_dir
DATA_CACHE = SETTINGS.data_cache
#: Le seul champ qui varie d'un run à l'autre sans que le chart change :
#: la latence choisie par la recherche de `musx.redecode` est chronométrée.
#: Champs ignorés par la comparaison. Vide depuis le 2026-09-15 : le seul
#: qu'il y avait (`meta.musx_latency_ms`) a disparu avec la recherche de
#: latence. Garder la mécanique : un prochain champ volatil se déclare ici.
VOLATILE: tuple = ()

log = logging.getLogger("golden")


def charger_moteur(nom: str) -> dict:
    """Les choses dont le rapport a besoin d'un moteur, et rien d'autre.

    `marks`/`new_cache` datent du sprint 15 : `harmonia` lit sa marque
    « Set bar 1 » dans `state/human/marks/` (la source de vérité, voir
    `jobs.bar1_for`) et vérifie ses caches via `harmonia.cache` (clé neuve +
    repli historique).

    Sprint 22 (2026-09-16) : il n'y a PLUS qu'un moteur. La branche
    `harmonia_min` est retirée plutôt que gardée morte — le paquet est
    supprimé, donc la garder ne produirait qu'un ImportError obscur au
    premier appel. La baseline du rapport d'or ne dépend pas d'elle : c'est
    du JSON gelé sur le disque (`state/cache/golden/baseline/`), pas un
    chart recuit par l'ancien code. Un vieux `--engine harmonia_min` doit
    donc échouer en DISANT pourquoi, pas planter à l'import.
    """
    if nom == "harmonia_min":
        raise SystemExit(
            "moteur « harmonia_min » supprimé au sprint 22 (2026-09-16) : le "
            "paquet n'existe plus. La baseline reste comparable — elle est "
            "gelée en JSON dans state/cache/golden/baseline/. "
            "Utilisez --engine harmonia.")
    if nom == "harmonia":
        from harmonia.pipeline import analyze
        from harmonia.refold import refold
        from harmonia.soudure import sections_pour_chart
        s = SETTINGS
        return {"analyze": analyze, "refold": refold,
                "sections_pour_chart": sections_pour_chart,
                "charts": s.charts_dir, "sections": s.sections_dir,
                "beats": s.beats_dir, "songformer": s.songformer_dir,
                "marks": s.marks_dir, "new_cache": True}
    raise SystemExit(f"moteur inconnu : {nom}")


def caches_manquants(eng: dict, stem: str, audio: Path) -> list[str]:
    if not audio.exists():
        return ["audio"]
    if eng["new_cache"]:
        # Passe par `harmonia.cache` — clé neuve `<stem>__<taille>`, repli sur
        # la clé historique — pour tester EXACTEMENT ce que le serveur voit,
        # pas une supposition de chemin réécrite ici en double.
        from harmonia import cache as _cache
        m = []
        if not _cache.exists("beats", audio):
            m.append("beats")
        if not _cache.exists("musx_probs", audio):
            m.append("musx")
        if not _cache.exists("nnls", audio):
            m.append("nnls")
        sf = _cache.load_json("songformer", audio)
        ok = bool(sf) and sf.get("taille") == audio.stat().st_size \
            and "segments" in sf
        if not ok:
            m.append("songformer")
        return m
    m = []
    if not (eng["beats"] / f"{stem}.json").exists():
        m.append("beats")
    if not (DATA_CACHE / "musx_probs" / f"{stem}.npz").exists():
        m.append("musx")
    if not (DATA_CACHE / "nnls_infer" / f"{stem}.npz").exists():
        m.append("nnls")
    sf = eng["songformer"] / f"{audio.name}.json"
    ok = False
    try:
        d = json.loads(sf.read_text(encoding="utf-8"))
        ok = d.get("taille") == audio.stat().st_size and "segments" in d
    except (OSError, ValueError):
        ok = False
    if not ok:
        m.append("songformer")
    return m


def sections_a_la_main(eng: dict, stem: str) -> list[dict] | None:
    """Le découpage validé à la main, ou None — copie de rebake_library."""
    f = eng["sections"] / f"{stem}.json"
    if not f.exists():
        return None
    try:
        doc = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not doc.get("validated"):
        return None
    secs = [{"label": x.get("label") or "?",
             "mesure_debut": int(x["b0"]) + 1,
             "mesure_fin": int(x["b1"]) + 1}
            for x in (doc.get("sections") or [])
            if isinstance(x, dict) and "b0" in x and "b1" in x]
    return secs or None


def completer(secs: list[dict], model: dict, n: int) -> tuple[list[dict], int]:
    """(découpage couvrant tout le morceau, sections ajoutées) — copie de rebake_library."""
    secs = [dict(x) for x in secs
            if 1 <= x["mesure_debut"] <= x["mesure_fin"] <= n]
    couvert = set()
    for x in secs:
        couvert |= set(range(x["mesure_debut"] - 1, x["mesure_fin"]))
    trous = sorted(set(range(n)) - couvert)
    if not trous:
        return secs, 0
    lab = {}
    for sec in model.get("sections") or []:
        for b0, b1 in sec.get("barRanges") or []:
            for b in range(b0, b1 + 1):
                lab[b] = sec.get("label") or "?"
    ajouts, debut = [], trous[0]
    for i, b in enumerate(trous):
        suivant = trous[i + 1] if i + 1 < len(trous) else None
        if suivant is None or suivant != b + 1 or lab.get(suivant) != lab.get(debut):
            ajouts.append({"label": lab.get(debut) or "?",
                           "mesure_debut": debut + 1, "mesure_fin": b + 1})
            if suivant is not None:
                debut = suivant
    out = sorted(secs + ajouts, key=lambda x: x["mesure_debut"])
    return out, len(ajouts)


def cuire(eng: dict, key: str, charts: Path, out: Path) -> dict:
    """Un chart = f(audio, caches, bar 1, sections à la main). Renvoie le compte rendu."""
    src = charts / f"{key}.json"
    old = json.loads(src.read_text(encoding="utf-8"))
    stem = Path(old.get("audio_url") or "").stem or key.removeprefix("min_")
    audio = AUDIO / f"{stem}.m4a"
    froid = caches_manquants(eng, stem, audio)
    if froid:
        return {"status": "froid", "manque": froid, "stem": stem}
    t0 = time.time()
    # Sprint 15 : la marque « Set bar 1 » se lit dans `state/human/marks/` —
    # jamais dans `old.get("bar1")`, qui ne serait que le champ d'un chart
    # régénérable (voir `jobs.bar1_for`). Le repli sur `old["bar1"]` reste
    # pour un moteur sans dossier de marques ; depuis le sprint 22 il n'y en
    # a plus, mais la mécanique ne coûte rien et documente la différence.
    #
    # MÊME MÉCANIQUE POUR ÷2/×2 (2026-09-17). Sans ça, un rebake/publish
    # effacerait silencieusement le réglage tempo de Louis exactement comme
    # `bar1` se perdait avant le sprint 15 (l'incident qui a motivé le fichier
    # de marque) : le chart cuit ici repartirait du tracker brut, deux fois
    # trop rapide, et Louis ne le découvrirait qu'en rouvrant le morceau.
    if eng.get("marks") is not None:
        try:
            mark = json.loads((eng["marks"] / f"{stem}.json")
                              .read_text(encoding="utf-8"))
        except (OSError, ValueError):
            mark = {}
        bar1 = mark.get("bar1")
        tempo_factor = mark.get("tempo_factor")
    else:
        bar1 = old.get("bar1")
        tempo_factor = None
    model = eng["analyze"](audio, title=old.get("title") or "", file_key=key,
                           audio_url=f"/audio/{audio.name}",
                           bar1_time=bar1, tempo_factor=tempo_factor)
    if not model.get("barGrid") or not model.get("sections"):
        return {"status": "erreur", "stem": stem, "raison": "chart vide"}
    if old.get("bar1") is not None and model.get("bar1") is None:
        return {"status": "erreur", "stem": stem,
                "raison": f"Set bar 1 ({old['bar1']}s) perdu"}
    note = ""
    a_la_main = sections_a_la_main(eng, stem)
    if a_la_main:
        n = model.get("nBars") or 0
        a_la_main, n_comble = completer(a_la_main, model, n)
        if a_la_main:
            bars, rap = eng["refold"](model, a_la_main, AUDIO)
            neuves = eng["sections_pour_chart"](model, a_la_main, bars=bars,
                                                fold_report=rap.get("rapport"))
            if neuves:
                model["sections"] = neuves
                model["fold"] = rap.get("rapport") or {}
                model["form"] = None
                note = f"sections à la main rejouées (+{n_comble} comblées)"
            else:
                note = "découpage à la main inutilisable"
        else:
            note = "découpage à la main inutilisable"
    (out / f"{key}.json").write_text(json.dumps(model, ensure_ascii=False),
                                     encoding="utf-8")
    return {"status": "ok", "stem": stem, "s": round(time.time() - t0, 1),
            "nBars": model.get("nBars"), "keyName": model.get("keyName"),
            "nSections": len(model.get("sections") or []), "note": note}


def _sans_volatile(m: dict) -> dict:
    m = copy.deepcopy(m)
    for chemin in VOLATILE:
        d = m
        for k in chemin[:-1]:
            d = d.get(k) or {}
        d.pop(chemin[-1], None)
    return m


def comparer(new: dict, base: dict) -> dict:
    rows, _ = diff_song(base, new)
    return {"identique": _sans_volatile(new) == _sans_volatile(base),
            "mesures_changees": len(rows),
            "keyName_avant": base.get("keyName"),
            "nSections_avant": len(base.get("sections") or [])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # Sprint 22 : un seul moteur. `harmonia_min` reste ACCEPTÉ par le parseur
    # pour que `charger_moteur` puisse expliquer sa disparition, au lieu du
    # « invalid choice » d'argparse qui n'apprend rien à qui rejoue une
    # vieille ligne de commande.
    ap.add_argument("--engine", required=True, choices=["harmonia", "harmonia_min"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--charts", type=Path, help="dossier source des min_*.json "
                    "(défaut : la bibliothèque du moteur)")
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--publish", action="store_true",
                    help="publie chaque chart réussi dans la bibliothèque du "
                    "moteur, un par un (remplace scripts/rebake_library.py)")
    ap.add_argument("--backup", type=Path,
                    help="dossier de sauvegarde EXISTANT de la bibliothèque — "
                    "requis avec --publish, sauf --dry-run")
    ap.add_argument("--dry-run", action="store_true",
                    help="avec --publish : cuit et affiche, ne publie rien")
    a = ap.parse_args(argv)

    if a.publish and not a.dry_run and not (a.backup and a.backup.is_dir()):
        print("refus : --publish exige --backup <dossier de copie existant> "
              "(ou --dry-run). Une bibliothèque se remplace avec un filet.")
        return 2

    a.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(a.out / "_golden.log", encoding="utf-8")])
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.WARNING)
    logging.getLogger().addHandler(console)

    eng = charger_moteur(a.engine)
    charts = a.charts or eng["charts"]
    keys = sorted(p.stem for p in charts.glob("min_*.json"))
    if a.only:
        keys = [k for k in keys if k in a.only]
    if a.limit:
        keys = keys[:a.limit]
    print(f"{len(keys)} chart(s) · moteur {a.engine} · sortie {a.out}"
          + (f" · baseline {a.baseline}" if a.baseline else ""), flush=True)

    report, n_err = {}, 0
    for k in keys:
        try:
            r = cuire(eng, k, charts, a.out)
        except Exception as exc:  # noqa: BLE001 — un morceau n'arrête pas la fournée
            log.exception("%s", k)
            r = {"status": "erreur", "raison": f"{type(exc).__name__}: {exc}",
                 "trace": traceback.format_exc()}
        if r["status"] == "ok" and a.baseline:
            b = a.baseline / f"{k}.json"
            if b.exists():
                r.update(comparer(json.loads((a.out / f"{k}.json").read_text(encoding="utf-8")),
                                  json.loads(b.read_text(encoding="utf-8"))))
            else:
                r["identique"], r["mesures_changees"] = None, None
        report[k] = r
        n_err += r["status"] == "erreur"
        tag = {"ok": "  ok   ", "froid": "  FROID", "erreur": "  ERREUR"}[r["status"]]
        extra = ""
        if r["status"] == "ok":
            extra = f"{r['s']}s · {r['nBars']} mes · {r['keyName']} · {r['nSections']} sect"
            if a.baseline and r.get("identique") is not None:
                extra += ("  · identique" if r["identique"]
                          else f"  · DIFFÈRE ({r['mesures_changees']} mesures"
                          + (f", tonalité {r['keyName_avant']}→{r['keyName']}"
                             if r["keyName_avant"] != r["keyName"] else "")
                          + (f", sections {r['nSections_avant']}→{r['nSections']}"
                             if r["nSections_avant"] != r["nSections"] else "") + ")")
            if r.get("note"):
                extra += f"  · {r['note']}"
        elif r["status"] == "froid":
            extra = "manque " + ", ".join(r["manque"])
        else:
            extra = r.get("raison", "")
        print(f"{tag} {k}: {extra}", flush=True)

    (a.out / "_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    oks = [r for r in report.values() if r["status"] == "ok"]
    froids = sum(r["status"] == "froid" for r in report.values())
    ligne = f"\n{len(oks)} cuit(s) · {froids} froid(s) · {n_err} erreur(s)"
    if a.baseline:
        ident = sum(1 for r in oks if r.get("identique"))
        chang = sum(r.get("mesures_changees") or 0 for r in oks)
        ligne += f" · identiques {ident}/{len(oks)} · mesures changées {chang}"
    print(ligne, flush=True)

    if a.publish:
        if a.dry_run:
            print(f"--dry-run : {len(oks)} chart(s) seraient publiés dans "
                  f"{eng['charts']}, la bibliothèque n'est pas touchée")
        else:
            moved = 0
            for k, r in report.items():
                if r["status"] != "ok":
                    continue
                shutil.move(str(a.out / f"{k}.json"), str(eng["charts"] / f"{k}.json"))
                moved += 1
            print(f"{moved} chart(s) publié(s) dans {eng['charts']}")
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
