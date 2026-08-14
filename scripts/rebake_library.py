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
                  + f"{time.time() - t0:.0f}s")


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
