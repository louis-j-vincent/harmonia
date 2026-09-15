"""Tout ce qui parle à YouTube : recherche (yt-dlp Python API) et
téléchargement (yt-dlp en sous-processus, avec retries et diagnostics).

Porté verbatim depuis `harmonia_min/server.py` (sprints 11-14) — ce sont les
morceaux « durement gagnés » que `CLAUDE.md` documente (le blocage anti-bot
YouTube du 2026-09-13) : trois pièces (cookies Firefox, solveur JS distant,
serveur PO Token local) dont aucune seule ne suffit.

Ce que ce module ne fait PAS : résoudre une URL d'analyse en fichier audio
(voir `jobs._resolve_audio`, qui appelle `_download_audio` d'ici) ; savoir où
vit la bibliothèque locale au-delà de `SETTINGS.audio_dir`.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from harmonia.settings import SETTINGS

log = logging.getLogger("harmonia.server.youtube")

AUDIO_DIR = SETTINGS.audio_dir

SEARCH_PER_PAGE = 12


def _local_matches(words):
    from harmonia_min import titles as _titles
    out = []
    for p in sorted(AUDIO_DIR.glob("*.m4a")):
        hay = p.stem.lower()
        if words and all(w in hay for w in words):
            out.append({"id": f"local:{p.stem}",
                        "title": _titles.pretty_from_slug(p.stem),
                        "uploader": "déjà téléchargé", "duration": None,
                        "thumb": "", "local": True})
    return out


def _youtube_search(q, page, per):
    """Real YouTube search via the yt_dlp PYTHON api (the CLI shells out and
    costs ~1 s extra per call). `extract_flat` skips per-video extraction, so a
    page comes back in about a second.

    yt-dlp has no cursor: `ytsearchN:` always returns the first N. Paging is
    therefore "ask for (page+1)*per and drop what we already showed" — the
    flat call is cheap enough that this is fine at the depths a human scrolls.
    """
    want = (page + 1) * per
    opts = {"quiet": True, "no_warnings": True, "skip_download": True,
            "extract_flat": "in_playlist"}
    import yt_dlp
    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(f"ytsearch{want}:{q}", download=False)
    entries = (info or {}).get("entries") or []
    out = []
    for e in entries[page * per:]:
        vid = e.get("id")
        if not vid:
            continue
        thumbs = e.get("thumbnails") or []
        out.append({
            "id": vid,
            "title": e.get("title") or vid,
            "uploader": e.get("uploader") or e.get("channel") or "YouTube",
            "duration": e.get("duration"),
            "thumb": (thumbs[0] or {}).get("url", "") if thumbs else "",
            "local": False,
        })
    return out, len(entries) >= want


def _ytdlp_bin():
    """The yt-dlp sitting next to THIS interpreter: the venv's bin/ is only on
    PATH when the server was launched from an activated shell, and a plain
    `python -m harmonia.server` otherwise reports "yt-dlp not installed"
    while it is right there."""
    cand = Path(sys.executable).parent / "yt-dlp"
    return str(cand) if cand.exists() else shutil.which("yt-dlp")


#: Ce que yt-dlp doit imprimer pour qu'on ait artiste et titre.
_META_PRINT = "%(title)s\t%(artist)s\t%(track)s\t%(uploader)s"


def _meta_from_print(stdout: str) -> tuple[str, str]:
    """La ligne `_META_PRINT` → (artiste, titre). Une seule fabrique : elle est
    lue à deux endroits (pendant le téléchargement, et sur un fichier déjà là)
    et deux analyseurs divergeraient."""
    from harmonia_min import titles as _titles
    line = next((x for x in stdout.splitlines() if "\t" in x), "")
    if not line:
        return "", ""
    f = (line.split("\t") + [""] * 4)[:4]
    return _titles.split(f[0], artist=f[1], track=f[2], uploader=f[3])


def _video_meta(ytdlp: str, url: str) -> tuple[str, str]:
    """(artiste, titre) d'une vidéo YouTube → ("", "") si on n'a rien pu lire.

    Louis, 2026-08-09 : « on a des codes à la place ». La cause était ici :
    quand yt-dlp télécharge, le fichier prend le nom de l'IDENTIFIANT de la
    vidéo, et le titre affiché était ce stem passé en capitales de titre —
    `J36Z7Anhvom`. Rien n'a jamais lu les métadonnées. Un appel de plus, deux
    secondes, et on a le vrai nom.

    Volontairement non fatal : l'analyse d'un morceau ne doit pas échouer parce
    qu'un champ de métadonnée manque. Le repli reste le stem, comme avant.
    """
    try:
        r = subprocess.run(
            [ytdlp, "--skip-download", "--no-warnings", "--no-playlist",
             "--print", _META_PRINT, url],
            capture_output=True, text=True, timeout=90)
        return _meta_from_print(r.stdout)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("metadata lookup failed for %s: %s", url, exc)
        return "", ""


#: Les clients d'API YouTube essayés, dans l'ordre. yt-dlp en choisit un tout
#: seul ; quand celui-là se fait jeter, la commande entière échoue alors que
#: le suivant aurait marché. Constaté le 2026-08-10 (Louis) : « android vr »
#: a rendu `HTTP Error 403: Forbidden`, et exactement la même URL est passée
#: à la reprise, sans rien changer. C'est intermittent et côté YouTube — donc
#: ça se réessaie, ça ne se diagnostique pas.
YTDLP_CLIENTS = (None, "web_safari", "android", "ios", "tv")


#: Vérifié 2026-09-13 : le blocage anti-bot YouTube (« Sign in to confirm
#: you're not a bot ») n'est plus un cas rare lié à une vidéo précise — 4
#: vidéos prises au hasard dans l'historique de Louis l'ont déclenché le
#: même jour, dont 3 déjà présentes dans sa bibliothèque. Le fix a trois
#: pièces, aucune seule ne suffit (mesuré en isolant chacune) :
#:   1. des cookies YouTube réels (`--cookies-from-browser`) pour passer
#:      la vérif anti-bot ;
#:   2. le solveur de challenge JS officiel de yt-dlp (`--remote-components
#:      ejs:github`, téléchargé une fois puis mis en cache) — sans lui,
#:      YouTube force le streaming SABR et yt-dlp ne récupère plus que les
#:      storyboards (aucun format audio/vidéo réel), même avec les cookies ;
#:   3. un serveur PO Token sur 127.0.0.1:4416
#:      (github.com/Brainicism/bgutil-ytdlp-pot-provider, installé dans
#:      ~/.local/share/bgutil-ytdlp-pot-provider ; le plugin pip
#:      `bgutil-ytdlp-pot-provider` est dans le venv) — sans lui, le point 2
#:      échoue quand même sur les vidéos qui exigent un PO Token valide.
#: `_ensure_pot_server()` relance ce process s'il n'est pas déjà debout —
#: sinon toute la chaîne casse silencieusement après un simple redémarrage
#: de la machine.
#: NE RÉSOUT PAS : une vraie vidéo privée reste bloquée (le fallback plus
#: bas le dit correctement) ; et si Firefox n'a jamais eu de session
#: YouTube connectée, `--cookies-from-browser` n'aide pas.
YTDLP_COOKIES_BROWSER = "firefox"
YTDLP_REMOTE_COMPONENTS = "ejs:github"
_POT_SERVER_URL = "http://127.0.0.1:4416"
_POT_SERVER_DIR = Path.home() / ".local" / "share" / "bgutil-ytdlp-pot-provider" / "server"


def _ensure_pot_server() -> None:
    """Démarre le serveur PO Token local s'il ne répond pas déjà.

    Best-effort : si l'installation est absente (autre machine, pas encore
    posée), on log et on continue — `_download_audio` retombera sur l'ancien
    comportement (marche pour les vidéos qui ne demandent pas de PO Token,
    échoue proprement sinon, avec le message anti-bot plus bas).
    """
    try:
        urllib.request.urlopen(f"{_POT_SERVER_URL}/ping", timeout=2)
        return
    except OSError:
        pass
    main_js = _POT_SERVER_DIR / "build" / "main.js"
    if not main_js.exists():
        log.warning("serveur PO Token absent (%s) — pas de retry auto pour "
                    "le blocage anti-bot YouTube.", main_js)
        return
    log.info("serveur PO Token éteint — redémarrage depuis %s", _POT_SERVER_DIR)
    subprocess.Popen(
        ["node", "build/main.js"], cwd=str(_POT_SERVER_DIR),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)
    for _ in range(15):
        time.sleep(1)
        try:
            urllib.request.urlopen(f"{_POT_SERVER_URL}/ping", timeout=2)
            log.info("serveur PO Token de retour.")
            return
        except OSError:
            continue
    log.warning("serveur PO Token relancé mais ne répond toujours pas après 15s.")


#: Au-delà de ça, un 403 sur TOUS les clients n'est plus « passager » : c'est
#: yt-dlp qui a pris du retard sur les signatures YouTube. Les deux fois où
#: Louis a vu l'écran d'échec (2026-08-10, 2026-08-20), la version installée
#: avait plus d'un mois et `pip install -U yt-dlp` a suffi. 21 jours : yt-dlp
#: publie environ toutes les deux semaines, donc trois semaines sans mise à
#: jour veut déjà dire qu'on a sauté une release.
YTDLP_STALE_DAYS = 21


def _ytdlp_staleness(ytdlp: str) -> str:
    """La phrase à afficher après un 403 : « réessaie » ou « mets à jour ».

    La version de yt-dlp EST une date (`2026.08.19`), donc son âge se lit sans
    réseau ni index PyPI. On ne le calcule que sur le chemin d'échec — un
    sous-processus de plus ne coûte rien quand plus rien ne marche.
    """
    import datetime as _dt
    try:
        r = subprocess.run([ytdlp, "--version"], capture_output=True,
                           text=True, timeout=30)
        ver = r.stdout.strip().splitlines()[0]
        y, m, d = (int(x) for x in ver.split(".")[:3])
        age = (_dt.date.today() - _dt.date(y, m, d)).days
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return "C'est peut-être passager : réessaie dans un moment."
    if age >= YTDLP_STALE_DAYS:
        return (f"yt-dlp date du {ver} ({age} jours) — c'est presque sûrement "
                "ça. Mets-le à jour : `.venv/bin/pip install -U yt-dlp`.")
    return "C'est passager : réessaie dans un moment."


def _download_audio(ytdlp: str, url: str, out: Path) -> tuple[str, str]:
    """Télécharge l'audio → (artiste, titre), en réessayant avec un autre client.

    LES MÉTADONNÉES VIENNENT D'ICI (2026-08-18). `_video_meta` était un SECOND
    aller-retour réseau, mesuré 2,5–3,1 s, posé sur le chemin critique juste
    avant les battues, pour un champ dont aucun accord ne dépend. `--print`
    avec `--no-simulate` imprime la même ligne pendant le téléchargement :
    même information, un appel au lieu de deux (4,9 s au total contre ~7,5 s
    mesurées le 2026-08-18 sur h_D3VFfhvs4).

    Lève une RuntimeError au message LISIBLE : l'échec précédent remontait
    jusqu'à l'écran de Louis sous la forme d'un `CalledProcessError` avec la
    ligne de commande complète, qui ne dit pas ce qui s'est passé ni quoi
    faire. La vraie cause (« 403 Forbidden ») était, elle, uniquement dans
    les logs du serveur.
    """
    _ensure_pot_server()
    errors = []
    for client in YTDLP_CLIENTS:
        cmd = [ytdlp, "-f", "bestaudio[ext=m4a]/bestaudio",
               "--extract-audio", "--audio-format", "m4a",
               "--retries", "5", "--fragment-retries", "5",
               "--cookies-from-browser", YTDLP_COOKIES_BROWSER,
               "--remote-components", YTDLP_REMOTE_COMPONENTS,
               # `--print` seul implique `--simulate` : sans `--no-simulate`
               # la commande n'écrirait plus aucun fichier.
               "--no-simulate", "--print", _META_PRINT,
               "-o", str(out)]
        if client:
            cmd += ["--extractor-args", f"youtube:player_client={client}"]
        cmd.append(url)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode == 0 and out.exists():
            if client:
                log.info("yt-dlp: réussi avec le client %s", client)
            return _meta_from_print(r.stdout)
        tail = (r.stderr or r.stdout or "").strip().splitlines()
        msg = tail[-1] if tail else f"code {r.returncode}"
        errors.append(f"{client or 'défaut'}: {msg}")
        log.warning("yt-dlp: client %s a échoué — %s", client or "défaut", msg)
        for junk in AUDIO_DIR.glob(out.name + "*.part"):
            junk.unlink(missing_ok=True)     # sinon la reprise repart de rien
    joined = " | ".join(errors)
    if "403" in joined or "Forbidden" in joined:
        raise RuntimeError(
            "YouTube a refusé le téléchargement (403) sur tous les clients "
            f"essayés. {_ytdlp_staleness(ytdlp)}")
    if "not a bot" in joined or "Sign in to confirm" in joined:
        # Vérifié 2026-09-13 sur nDUVEUjOKMw (Benny Sings, KCRW) : la vidéo
        # est publique (oEmbed répond 200) et yt-dlp télécharge d'autres
        # vidéos sans souci au même moment — ce n'est PAS la vidéo qui est
        # privée, c'est la vérif anti-bot de YouTube qui bloque CE
        # téléchargement précis, sur tous les clients essayés. L'ancien
        # message ("privée ou restreinte") attribuait le blocage à la vidéo
        # à tort — même bug que le pattern #1 de CLAUDE.md (message
        # plausible, mauvais diagnostic). Le vrai fix (cookies YouTube via
        # `--cookies-from-browser`) n'est pas branché ici.
        raise RuntimeError(
            "YouTube bloque ce téléchargement avec une vérification "
            "anti-bot — la vidéo n'est pas forcément privée (souvent une "
            "autre passe). Réessaie avec une autre vidéo pour l'instant.")
    if "Private video" in joined or "Sign in" in joined:
        raise RuntimeError("Cette vidéo demande une connexion (privée ou "
                           "restreinte) — elle ne peut pas être téléchargée.")
    if "Video unavailable" in joined:
        raise RuntimeError("Cette vidéo n'est pas disponible.")
    raise RuntimeError(f"Le téléchargement a échoué. Détail : {joined}")


# Ce que ce module ne fait PAS : décider comment une URL d'analyse (locale,
# "local:<stem>", ou YouTube) se résout en fichier audio — c'est
# `jobs._resolve_audio`, qui appelle `_download_audio` ci-dessus.
