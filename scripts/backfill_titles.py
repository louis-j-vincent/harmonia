"""Rend à chaque chart son artiste et son titre.

Louis, 2026-08-09 : « pour l'instant c'est n'importe quoi, on a des codes à la
place ». Onze charts de la bibliothèque s'appellent `J36Z7Anhvom`,
`Uw5Olnn7Uvm`… — l'identifiant YouTube passé en capitales de titre — et AUCUN
n'a d'artiste.

Deux situations, deux traitements, et la différence compte :

* **le stem EST un identifiant** (`J36z7AnhvOM`) — on interroge la vidéo
  directement. C'est exact, il n'y a rien à deviner.
* **le stem est un slug** (`ray_charles_georgia_on_my_mind_official_video`) — le
  tiret qui séparait l'artiste du titre a disparu au slugifiage. On cherche la
  vidéo sur YouTube et **on n'accepte sa réponse que si le titre trouvé se
  re-slugifie EXACTEMENT en notre stem**. Sinon on refuse et on laisse le champ
  vide. Un artiste faux se propage dans toute la bibliothèque ; un champ vide se
  corrige d'un tap dans l'app.

Écrit dans `harmonia_min/state/chart_meta.json`, le sidecar que `/api/library`
ressert déjà et que l'éditeur artiste/titre de l'app écrit. **Les charts
eux-mêmes ne sont pas touchés** — le modèle reste ce que le pipeline a produit.

    python scripts/backfill_titles.py             # dry-run, affiche la table
    python scripts/backfill_titles.py --apply     # écrit le sidecar
    python scripts/backfill_titles.py --apply --force   # écrase aussi ce qui
                                                        # a été édité à la main
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia_min.titles import (pretty_from_slug, slugify,  # noqa: E402
                                 split, strip_junk)

CHARTS = REPO / "harmonia_min" / "state" / "charts"
META = REPO / "harmonia_min" / "state" / "chart_meta.json"
CACHE = REPO / "data" / "cache" / "yt_meta"
YTDLP = Path(sys.executable).parent / "yt-dlp"
FIELDS = "%(id)s\t%(title)s\t%(artist)s\t%(track)s\t%(uploader)s"
IS_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _ytdlp(target: str, *, timeout: int = 90) -> dict | None:
    """Un appel yt-dlp en mode métadonnées seules → le premier résultat."""
    if not YTDLP.exists():
        raise SystemExit(f"yt-dlp introuvable ({YTDLP}) — rien à faire.")
    try:
        r = subprocess.run([str(YTDLP), "--skip-download", "--no-warnings",
                            "--no-playlist", "--print", FIELDS, target],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    line = next((ln for ln in r.stdout.splitlines() if ln.strip()), "")
    if not line:
        return None
    f = (line.split("\t") + [""] * 5)[:5]
    return {"id": f[0], "title": f[1], "artist": f[2], "track": f[3],
            "uploader": f[4]}


def _cached(key: str, target: str) -> dict | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{key}.json"
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return d or None
        except ValueError:
            pass
    d = _ytdlp(target)
    p.write_text(json.dumps(d or {}, ensure_ascii=False), encoding="utf-8")
    return d


def _same_song(found_title: str, stem: str) -> str:
    """La vidéo trouvée est-elle bien la nôtre ? → le motif de l'accord, ou "".

    Deux niveaux, tous deux vérifiables, aucun n'est une devinette :

    1. **le stem exact** — le titre trouvé se re-slugifie en notre stem. C'est
       littéralement la même chaîne de caractères, donc la même vidéo.
    2. **la même chanson, autre mise en ligne** — une fois les mentions de
       production enlevées des DEUX côtés, l'un contient l'autre sur des
       frontières de mots. `the_ronettes_be_my_baby_music_video` et
       `the_ronettes_be_my_baby_official_audio` deviennent la même chose ;
       `let_it_be` est contenu dans `the_beatles_let_it_be`, ce qui rend
       l'artiste qui manquait au nom de fichier.

    Le plancher de trois mots existe pour que « Sunny » ou « Yesterday » ne
    puissent pas s'accrocher au premier titre venu qui les contient.
    """
    if slugify(found_title)[:60] == stem[:60]:
        return "stem exact"
    a = slugify(strip_junk(found_title))
    b = slugify(strip_junk(stem.replace("_", " ")))
    if not a or not b:
        return ""
    lo, hi = (a, b) if len(a) <= len(b) else (b, a)
    if lo.count("_") + 1 < 3:                     # moins de trois mots : non
        return ""
    if re.search(rf"(^|_){re.escape(lo)}(_|$)", hi):
        return "même chanson"
    return ""


def resolve(stem: str) -> tuple[str, str, str]:
    """→ (artiste, titre, d'où ça vient)."""
    if IS_ID.match(stem):
        d = _cached(stem, f"https://www.youtube.com/watch?v={stem}")
        if not d:
            return "", pretty_from_slug(stem), "échec réseau"
        a, t = split(d["title"], uploader=d["uploader"],
                     artist=d["artist"], track=d["track"])
        return a, t, "id"

    # un slug : on cherche, et on VÉRIFIE avant d'accepter
    query = strip_junk(stem.replace("_", " "))
    d = _cached(stem, f"ytsearch1:{query}")
    why = _same_song(d["title"], stem) if d else ""
    if why:
        a, t = split(d["title"], uploader=d["uploader"],
                     artist=d["artist"], track=d["track"])
        # le nom de fichier peut être plus complet que le titre trouvé
        # (« elton_john_… » contre « Goodbye Yellow Brick Road ») : on garde le
        # titre trouvé, qui est le propre, et l'artiste vient d'avec.
        return a, t, why
    # refus : le titre trouvé n'est pas le nôtre. On garde le stem nettoyé.
    why = "refusé (%s)" % (slugify(d["title"])[:34] if d else "rien trouvé")
    return "", pretty_from_slug(stem), why


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="écrase aussi les entrées déjà présentes")
    ap.add_argument("--only", default="", help="un stem, pour tester")
    args = ap.parse_args()

    try:
        meta = json.loads(META.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}

    rows, changed = [], 0
    for p in sorted(CHARTS.glob("*.json")):
        key = p.stem                                   # min_<stem> / raw_<stem>
        stem = re.sub(r"^(min|raw)_", "", key)
        if args.only and args.only not in key:
            continue
        have = meta.get(key) or {}
        if have.get("title") and not args.force:
            rows.append((key, have.get("artist", ""), have["title"], "déjà là"))
            continue
        a, t, how = resolve(stem)
        rows.append((key, a, t, how))
        if t:
            meta[key] = {"artist": a, "title": t}
            changed += 1

    w = max((len(r[0]) for r in rows), default=10)
    print(f"\n{'chart':{w}}  {'artiste':22}  {'titre':34}  d'où")
    print("─" * (w + 68))
    for key, a, t, how in rows:
        print(f"{key:{w}}  {a[:22]:22}  {t[:34]:34}  {how}")
    n_art = sum(1 for r in rows if r[1])
    print(f"\n{len(rows)} charts · {n_art} avec artiste · {changed} à écrire")

    if args.apply and changed:
        META.parent.mkdir(parents=True, exist_ok=True)
        META.write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        print(f"écrit → {META}")
    elif changed:
        print("(dry-run — relance avec --apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
