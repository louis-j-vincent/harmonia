"""Régénère les charts de la bibliothèque avec la pipeline ACTUELLE.

Un chart est cuit une fois puis servi tel quel : changer une loi du repli ne
change donc RIEN à ce que Louis voit tant que les charts ne sont pas
refaits. C'est ce que fait ce script — et c'est la seule façon de mettre en
prod un changement de repli.

Prudence, deux règles payées par l'expérience :
  * il écrit dans un dossier de SORTIE séparé et ne remplace la bibliothèque
    qu'à la fin, morceau par morceau, quand le nouveau chart est valide :
    une interruption au milieu laissait sinon une bibliothèque mi-ancienne
    mi-nouvelle, impossible à distinguer d'un bug ;
  * il refuse de démarrer si `--backup` ne pointe pas sur une copie
    existante (`docs/…/state/charts.bak_*`) — une session concurrente peut
    travailler sur ces mêmes fichiers.

    python scripts/rebake_library.py --backup harmonia_min/state/charts.bak_20260812
    python scripts/rebake_library.py --only min_bein_green --dry-run
"""
from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CHARTS = REPO / "harmonia_min" / "state" / "charts"
AUDIO = REPO / "docs" / "audio"
SECTIONS = REPO / "harmonia_min" / "state" / "sections"


def _sections_a_la_main(stem: str) -> list[dict] | None:
    """Le découpage que Louis a validé à la main, ou None.

    `state/sections/<stem>.json` est la seule vérité terrain de sections du
    projet (23 morceaux au 2026-08-20). Le détecteur ne la lit pas : elle
    n'entre dans un chart que par l'outil de soudure. Un recuit qui
    l'ignorerait rendrait donc à ces 23 morceaux le découpage de la machine,
    en silence — la même faute que le `Set bar 1` effacé le 2026-08-13, et
    elle se répare pareil : on repasse la marque après l'analyse.
    """
    f = SECTIONS / f"{stem}.json"
    if not f.exists():
        return None
    try:
        doc = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not doc.get("validated"):
        return None            # un brouillon n'est pas une décision
    secs = [{"label": x.get("label") or "?",
             "mesure_debut": int(x["b0"]) + 1,
             "mesure_fin": int(x["b1"]) + 1}
            for x in (doc.get("sections") or [])
            if isinstance(x, dict) and "b0" in x and "b1" in x]
    return secs or None


def _completer(secs: list[dict], model: dict, n: int) -> tuple[list[dict], int]:
    """(découpage couvrant tout le morceau, nombre de sections ajoutées).

    Les mesures que l'annotation à la main ne couvre pas reçoivent l'étiquette
    que le DÉTECTEUR leur a donnée dans ce même recuit — découpée aux mêmes
    endroits que lui. On n'invente donc aucune frontière : on recolle deux
    sources, la sienne d'abord.
    """
    secs = [dict(x) for x in secs
            if 1 <= x["mesure_debut"] <= x["mesure_fin"] <= n]
    couvert = set()
    for x in secs:
        couvert |= set(range(x["mesure_debut"] - 1, x["mesure_fin"]))
    trous = sorted(set(range(n)) - couvert)
    if not trous:
        return secs, 0
    # étiquette du détecteur, mesure par mesure
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


def rebake(file_key: str, out_dir: Path) -> tuple[bool, str]:
    """Recuit un chart. Renvoie (ok, message)."""
    from harmonia_min.pipeline import analyze
    src = CHARTS / f"{file_key}.json"
    old = json.loads(src.read_text(encoding="utf-8"))
    stem = Path(old.get("audio_url") or "").stem or \
        file_key.removeprefix("min_")
    audio = AUDIO / f"{stem}.m4a"
    if not audio.exists():
        return False, f"{file_key}: audio absent ({stem}.m4a)"
    t0 = time.time()
    # La marque « Set bar 1 » de Louis se REPASSE (2026-08-13). Sans elle, le
    # rebake rendait la phase des mesures au tracker : chaque passage de
    # /ship effaçait silencieusement tous les recalages jamais posés, et le
    # chart revenait décalé sans que rien ne le dise.
    model = analyze(audio, title=old.get("title") or "", file_key=file_key,
                    audio_url=f"/audio/{audio.name}",
                    bar1_time=old.get("bar1"))
    # garde-fous : un chart vide ou sans grille ne remplace jamais l'ancien
    if not model.get("barGrid") or not model.get("sections"):
        return False, f"{file_key}: chart régénéré vide — ancien conservé"
    # …ni un chart qui aurait perdu la marque en route. Une annotation de
    # Louis qui disparaît doit ARRÊTER le recuit, pas passer inaperçue.
    if old.get("bar1") is not None and model.get("bar1") is None:
        return False, (f"{file_key}: le Set bar 1 ({old['bar1']}s) a disparu "
                       "du chart régénéré — ancien conservé")
    # ── le découpage fait à la main repasse par-dessus le détecteur ───────
    a_la_main = _sections_a_la_main(stem)
    note_sec = ""
    if a_la_main:
        n = model.get("nBars") or 0
        # SA VÉRITÉ OÙ IL L'A ÉCRITE, LA MACHINE AILLEURS (2026-08-20).
        # Première version : tout ou rien, refus dès qu'une mesure n'était pas
        # couverte. Elle a jeté 7 découpages sur 20 — pour un trou d'UNE mesure
        # sur Bein Green (la grille a gagné une mesure depuis l'annotation), de
        # deux sur Be My Baby, et pour la queue jamais annotée de Chain of
        # Fools. Jeter huit sections écrites à la main parce que la 53ᵉ mesure
        # manque, c'est perdre son travail pour préserver une symétrie.
        # On complète donc les trous avec ce que le détecteur a trouvé LÀ, et
        # rien de plus.
        a_la_main, n_comble = _completer(a_la_main, model, n)
        if not a_la_main:
            note_sec = ", découpage à la main inutilisable"
        else:
            from harmonia_min.refold import refold
            from harmonia_min.soudure import sections_pour_chart
            bars, rap = refold(model, a_la_main, AUDIO)
            neuves = sections_pour_chart(model, a_la_main, bars=bars)
            if neuves:
                model["sections"] = neuves
                model["fold"] = rap.get("rapport") or {}
                model["form"] = None
                note_sec = (f", {len(a_la_main) - n_comble} sections à la main "
                            f"rejouées"
                            + (f" (+{n_comble} comblée(s) par le détecteur)"
                               if n_comble else ""))
            else:
                note_sec = ", découpage à la main inutilisable"
    n_old = len(old.get("barGrid") or []) - 1
    n_new = len(model["barGrid"]) - 1
    (out_dir / f"{file_key}.json").write_text(
        json.dumps(model, ensure_ascii=False), encoding="utf-8")
    fold = model.get("fold") or {}
    folded = sum(1 for v in fold.values() if isinstance(v, dict)
                 and "n_obs" in v)
    return True, (f"{file_key}: {n_new} mesures (avant {n_old}), "
                  f"{folded}/{len(fold)} lettres repliées, "
                  + (f"bar 1 rejouée à {old['bar1']}s, "
                     if old.get("bar1") is not None else "")
                  + f"{time.time() - t0:.0f}s" + note_sec)


def main(argv):
    backup = argv[argv.index("--backup") + 1] if "--backup" in argv else None
    only = argv[argv.index("--only") + 1] if "--only" in argv else None
    dry = "--dry-run" in argv
    if not dry and (not backup or not Path(backup).is_dir()):
        print("refus : passe --backup <dossier de copie existant>. "
              "Une bibliothèque se remplace avec un filet.")
        return 2
    out_dir = REPO / "harmonia_min" / "state" / "charts_new"
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted(p.stem for p in CHARTS.glob("min_*.json"))
    if only:
        keys = [k for k in keys if k == only]
    print(f"{len(keys)} chart(s) à recuire · sortie {out_dir}", flush=True)
    ok = fail = 0
    for k in keys:
        try:
            good, msg = rebake(k, out_dir)
        except Exception as exc:  # noqa: BLE001 — un morceau ne doit pas
            good = False          # arrêter la fournée
            msg = f"{k}: ERREUR {type(exc).__name__}: {exc}"
            traceback.print_exc()
        print(("  ok   " if good else "  RATÉ ") + msg, flush=True)
        ok, fail = ok + int(good), fail + int(not good)
    print(f"\n{ok} réussite(s), {fail} échec(s)")
    if dry:
        print("--dry-run : la bibliothèque n'est pas remplacée")
        return 0
    moved = 0
    for p in sorted(out_dir.glob("min_*.json")):
        shutil.move(str(p), CHARTS / p.name)
        moved += 1
    print(f"{moved} chart(s) publié(s) dans {CHARTS}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
