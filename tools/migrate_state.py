"""Migre `harmonia_min/state/` (l'ancienne mise en page) vers `state/{human,cache}/`
(décision 10, plan du refactor, sprint 15) — COPIE, ne supprime jamais la source.

Louis, en revue du plan (2026-09-14) : « du travail à la main a été perdu une
fois (2026-08-12) parce qu'il vivait dans un dossier gitignored ». Le partage
état-humain / cache que ce script exécute, c'est la correction structurelle :

    state/human/   sections/ sections_draft/ annotations/ marks/
                   chart_meta.json folders.json      ← suivi par git
    state/cache/   charts/ beats/ songformer/         ← ignoré, régénérable

`marks/<stem>.json` (`{"bar1": <secondes>}`) est NEUF : avant ce sprint, la
marque « Set bar 1 » de Louis ne vivait QUE dans le champ `bar1` d'un chart
régénérable — ce qui l'a fait disparaître le 2026-08-13 au premier rebake qui
ne la repassait pas. Ce script est le SEUL endroit qui la fait naître à partir
d'un vieux chart ; à l'exécution, `server.jobs.bar1_for` ne lit plus jamais
`bar1` sur un chart.

`beats` et `songformer` changent aussi de CLÉ (`harmonia.cache`, sprint 15) :
`<stem>__<taille en octets>` au lieu du stem seul (battues) ou du nom de
fichier complet (songformer). La taille vient du fichier audio réel sur
`SETTINGS.audio_dir` ; si l'audio n'est plus là, le fichier est copié sous son
ANCIEN nom (clé historique) — `harmonia.cache` le retrouvera quand même par
son repli de lecture, et ce script le signale pour qu'on aille voir pourquoi
l'audio manque.

`data/cache/{musx_probs,musx_cqt,nnls_infer}` NE BOUGE PAS : lien symbolique
vers le cache PARTAGÉ du serveur vivant, disque plein à 96 % — leur renommage
en place est pour la bascule prod (sprint 21), pas ici (voir `harmonia.cache`,
qui les lit déjà par repli sur la clé historique).

IDEMPOTENT : chaque fichier est comparé PUIS écrit seulement s'il diffère, et
relu pour vérifier après coup. Une seconde exécution ne copie donc rien — les
deux runs (§gate) impriment le même tableau, sauf la ligne "copiés" à zéro.

Usage, depuis la racine du worktree :
    python -m tools.migrate_state
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from harmonia import cache as _cache
from harmonia.settings import SETTINGS

OLD = SETTINGS.repo / "harmonia_min" / "state"


@dataclass
class Ligne:
    etape: str
    copies: int = 0
    identiques: int = 0
    ignores: int = 0
    notes: list[str] = field(default_factory=list)

    def total(self) -> int:
        return self.copies + self.identiques + self.ignores


def _copy_verify(src: Path, dst: Path) -> str:
    """Copie octet pour octet SEULEMENT si différent, puis relit pour vérifier.
    Jamais de suppression de `src`."""
    data = src.read_bytes()
    if dst.exists() and dst.read_bytes() == data:
        return "identique"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    if dst.read_bytes() != data:
        raise RuntimeError(f"copie corrompue : {src} -> {dst}")
    return "copié"


def _write_json_verify(dst: Path, obj: dict) -> str:
    """Comme `_copy_verify` mais la source est un objet fabriqué ici (marks/),
    pas un fichier — sérialisation stable pour que la comparaison soit fiable."""
    if dst.exists():
        try:
            if json.loads(dst.read_text(encoding="utf-8")) == obj:
                return "identique"
        except (OSError, ValueError):
            pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    relu = json.loads(dst.read_text(encoding="utf-8"))
    if relu != obj:
        raise RuntimeError(f"écriture corrompue : {dst}")
    return "copié"


# ── état humain : sections, sections_draft, annotations, chart_meta, folders ─
def migrer_human() -> list[Ligne]:
    lignes = []
    for nom, dest_dir in (("sections", SETTINGS.sections_dir),
                          ("sections_draft", SETTINGS.sections_draft_dir),
                          ("annotations", SETTINGS.annotations_dir)):
        src_dir = OLD / nom
        l = Ligne(f"human/{nom}")
        if not src_dir.exists():
            l.notes.append("dossier source absent")
            lignes.append(l)
            continue
        for f in sorted(src_dir.glob("*.json")):
            r = _copy_verify(f, dest_dir / f.name)
            if r == "copié":
                l.copies += 1
            else:
                l.identiques += 1
        lignes.append(l)

    for nom, src, dest in (
            ("chart_meta.json", OLD / "chart_meta.json", SETTINGS.chart_meta_path),
            ("folders.json", OLD / "folders.json", SETTINGS.folders_path)):
        l = Ligne(f"human/{nom}")
        if not src.exists():
            l.notes.append("fichier source absent")
            lignes.append(l)
            continue
        r = _copy_verify(src, dest)
        if r == "copié":
            l.copies = 1
        else:
            l.identiques = 1
        lignes.append(l)
    return lignes


# ── marks/ : nées du champ `bar1` de chaque chart, jamais l'inverse ──────────
def migrer_marks() -> Ligne:
    l = Ligne("human/marks")
    charts_src = OLD / "charts"
    if not charts_src.exists():
        l.notes.append("dossier charts source absent")
        return l
    for f in sorted(charts_src.glob("min_*.json")):
        try:
            chart = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            l.ignores += 1
            l.notes.append(f"{f.name}: illisible ({exc})")
            continue
        bar1 = chart.get("bar1")
        if bar1 is None:
            continue          # pas de marque à créer — la majorité des charts
        stem = Path(chart.get("audio_url") or "").stem or \
            f.stem.removeprefix("min_")
        r = _write_json_verify(SETTINGS.marks_dir / f"{stem}.json",
                               {"bar1": float(bar1)})
        if r == "copié":
            l.copies += 1
        else:
            l.identiques += 1
    return l


# ── charts : min_*.json seulement, jamais old_/new_/pli_/raw_/retour_/... ───
def migrer_charts() -> Ligne:
    l = Ligne("cache/charts")
    src_dir = OLD / "charts"
    if not src_dir.exists():
        l.notes.append("dossier source absent")
        return l
    autres = [f for f in src_dir.glob("*.json") if not f.name.startswith("min_")]
    for f in sorted(src_dir.glob("min_*.json")):
        r = _copy_verify(f, SETTINGS.charts_dir / f.name)
        if r == "copié":
            l.copies += 1
        else:
            l.identiques += 1
    if autres:
        l.ignores += len(autres)
        l.notes.append(f"{len(autres)} fichier(s) hors min_*.json ignoré(s) "
                       "(old_/new_/pli_/raw_/retour_/…)")
    return l


# ── beats + songformer : copiés ET renommés sous la clé neuve de harmonia.cache
def migrer_cache_renomme(kind: str) -> Ligne:
    """`kind` est aussi le nom du dossier historique (`beats`, `songformer`)."""
    l = Ligne(f"cache/{kind}")
    src_dir = OLD / kind
    if not src_dir.exists():
        l.notes.append("dossier source absent")
        return l
    _, _, ext = _cache.KINDS[kind]
    dest_dir = _cache.folder(kind)
    n_repli = 0
    for f in sorted(src_dir.glob(f"*{ext}")):
        cle_historique = f.name[: -len(ext)] if f.name.endswith(ext) else f.stem
        stem = Path(cle_historique).stem       # "X" depuis "X" (beats) ou "X.m4a" (songformer)
        audio = SETTINGS.audio_dir / f"{stem}.m4a"
        if audio.exists():
            nouveau_nom = f"{stem}__{audio.stat().st_size}{ext}"
        else:
            nouveau_nom = f.name               # repli : clé historique inchangée
            n_repli += 1
        r = _copy_verify(f, dest_dir / nouveau_nom)
        if r == "copié":
            l.copies += 1
        else:
            l.identiques += 1
    if n_repli:
        l.notes.append(f"{n_repli} fichier(s) sans audio sur disque -> "
                       "copiés sous la clé HISTORIQUE (repli de lecture)")
    return l


def _df() -> str:
    try:
        out = subprocess.run(["df", "-h", str(SETTINGS.repo)],
                            capture_output=True, text=True, timeout=10)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"df indisponible : {exc}"


def _imprimer(lignes: list[Ligne]) -> None:
    print(f"{'étape':<22}{'copiés':>8}{'identiques':>12}{'ignorés':>9}  notes")
    for l in lignes:
        print(f"{l.etape:<22}{l.copies:>8}{l.identiques:>12}{l.ignores:>9}  "
             + "; ".join(l.notes))
    print(f"\ntotal copiés : {sum(l.copies for l in lignes)} · "
         f"identiques : {sum(l.identiques for l in lignes)} · "
         f"ignorés : {sum(l.ignores for l in lignes)}")


def main() -> int:
    print("df avant :\n" + _df() + "\n")
    lignes = []
    lignes += migrer_human()
    lignes.append(migrer_marks())
    lignes.append(migrer_charts())
    lignes.append(migrer_cache_renomme("beats"))
    lignes.append(migrer_cache_renomme("songformer"))
    _imprimer(lignes)
    print("\ndf après :\n" + _df())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
