"""Import YouTube sur téléphone (P3b, 2026-09-23) — PAS un fichier de `harmonia/`.

`harmonia/server/youtube.py` shelle vers un BINAIRE `yt-dlp` externe
(`subprocess.run([ytdlp, ...])`) avec `--cookies-from-browser firefox` et un
serveur PO-Token local sur 127.0.0.1:4416 : aucun des deux n'existe sur
Android (pas de Firefox natif à lire, pas de process Node local). Ce module
remplace les TROIS fonctions que `harmonia/server/jobs.py:_resolve_audio`
appelle (`_ytdlp_bin`, `_video_meta`, `_download_audio`) par un équivalent
EN PROCESSUS : `yt_dlp.YoutubeDL` importé comme bibliothèque Python (via le
paquet pip Chaquopy, pas un binaire), avec deux dépendances fournies par
l'app plutôt que par l'environnement Mac :

1. un fichier de cookies statique (`<filesDir>/youtube_cookies.txt`), copié
   depuis le Mac — pas de navigateur sur le téléphone pour les extraire ;
2. un runtime JS QuickJS embarqué en binaire natif
   (`<nativeLibraryDir>/libqjs.so`, voir docs/portage_mobile/p3b_*.md pour
   comment il a été compilé), pour résoudre le challenge de signature `n`
   que yt-dlp 2026.08 exige sur presque tous les clients — SANS PO-Token
   server, qui reste hors périmètre (voir "Ce qui reste ouvert" du rapport).

Installé par un monkeypatch explicite dans `mobile_entry.py` (`install()` ci-
dessous), PAS en modifiant `harmonia/server/youtube.py` — ce fichier n'est
jamais importé sur Mac, `harmonia/` reste identique sur les deux plateformes.
"""
from __future__ import annotations

import logging
from pathlib import Path

from flask import Blueprint, jsonify

log = logging.getLogger("mobile_youtube")

#: P3c (2026-09-23) : bouton "Connect YouTube" de prefs.js -> cette route
#: (montée par mobile_entry.py, n'existe pas sur Mac) -> `open_login_activity()`
#: ci-dessous. Voir docs/portage_mobile/p3c_youtube_session.md.
bp = Blueprint("mobile_youtube", __name__)


@bp.post("/api/mobile/youtube_login")
def _youtube_login_route():
    return jsonify({"ok": open_login_activity()})

#: Même format que le Mac (`harmonia/server/youtube.py`) — audio seul, m4a de
#: préférence. PAS de `--extract-audio`/`--audio-format` (postprocesseur
#: FFmpeg) : ffmpeg est interdit sur le téléphone (PLAN.md). Le flux "bestaudio
#: DASH m4a" que YouTube sert directement est donc gardé TEL QUEL — conteneur
#: valide mais sans l'atome de durée que FFmpeg aurait ajouté (`FixupM4a` sur
#: le Mac). Pas vérifié ici que MediaExtractor/MediaCodec s'en accommodent
#: (voir "Ce qui reste ouvert").
_FORMAT = "bestaudio[ext=m4a]/bestaudio"


def _app_context():
    from com.chaquo.python import Python  # noqa: PLC0415 (Android only)
    return Python.getPlatform().getApplication()


def _cookie_file() -> Path | None:
    """`<filesDir>/youtube_cookies.txt`, copié depuis le Mac par
    `adb push` + `run-as` (procédure : voir le rapport). Absent tant que
    Louis ne l'a pas poussé — traité comme "pas de cookies", pas une erreur."""
    try:
        files_dir = Path(_app_context().getFilesDir().getAbsolutePath())
    except Exception:
        return None
    p = files_dir / "youtube_cookies.txt"
    return p if p.exists() else None


def _qjs_path() -> str | None:
    """`<nativeLibraryDir>/libqjs.so` — binaire QuickJS-ng compilé pour
    arm64-v8a (voir docs/portage_mobile/p3b_youtube_cookies_qjs.md étape 1),
    embarqué comme lib native (extrait avec le bit exec par le système)."""
    try:
        info = _app_context().getApplicationInfo()
        p = Path(info.nativeLibraryDir) / "libqjs.so"
    except Exception:
        return None
    return str(p) if p.exists() else None


def open_login_activity() -> bool:
    """Lance `YouTubeLoginActivity` (WebView + bouton "Terminé", écrit
    `<filesDir>/youtube_cookies.txt` au format Netscape à la fermeture —
    voir YouTubeLoginActivity.kt) depuis un contexte Application : il faut
    `FLAG_ACTIVITY_NEW_TASK` (pas de tâche/Activity appelante ici, on part
    d'un thread Flask, pas d'un `onClick` Android)."""
    try:
        from java import jclass
    except ImportError:  # pas sur Android (tests desktop, etc.)
        return False
    ctx = _app_context()
    if ctx is None:
        return False
    try:
        intent_cls = jclass("android.content.Intent")
        intent = intent_cls()
        intent.setClassName(ctx.getPackageName(), "fr.harmonia.app.YouTubeLoginActivity")
        intent.setFlags(intent_cls.FLAG_ACTIVITY_NEW_TASK)
        ctx.startActivity(intent)
        return True
    except Exception:
        log.exception("mobile_youtube.open_login_activity failed")
        return False


def _ydl_opts() -> dict:
    opts: dict = {
        "format": _FORMAT,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": False,
        "retries": 5,
        "fragment_retries": 5,
        # Format exact requis par YoutubeDL._clean_js_runtimes :
        # {runtime: {config}}, PAS de clé "order" (ValueError mesuré ici le
        # 2026-09-23 avec {"order": [...], "quickjs": {...}} — corrigé après
        # lecture de yt_dlp/YoutubeDL.py:_js_runtimes). deno/node par défaut
        # n'existent pas sur Android, donc seul quickjs est déclaré ici — pas
        # besoin de "order" pour le préférer, c'est le seul candidat.
        "js_runtimes": {"quickjs": {"path": _qjs_path()}},
        # TROISIÈME dépendance découverte ici (2026-09-23), pas seulement les
        # deux du brief : le runtime JS exécute un SCRIPT solveur ("EJS") que
        # yt-dlp télécharge une fois puis met en cache — sans ce flag,
        # `--js-runtimes quickjs` déclare le moteur mais yt-dlp saute quand
        # même le téléchargement du script ("Remote component challenge
        # solver script (quickjs) was skipped"), donc le challenge `n` échoue
        # pareil. Sur Mac (Étape 0) ce cache existait déjà d'un test `deno`
        # précédent dans la même session — d'où le faux 3/3 "propre".
        "remote_components": ["ejs:github"],
    }
    cookies = _cookie_file()
    if cookies is not None:
        opts["cookiefile"] = str(cookies)
    else:
        log.warning("mobile_youtube: pas de cookies (%s absent) — YouTube "
                    "bloquera probablement (\"Sign in to confirm...\").",
                    "youtube_cookies.txt")
    return opts


def ytdlp_available() -> bool:
    try:
        import yt_dlp  # noqa: F401
        return True
    except ImportError:
        return False


def video_meta(url: str) -> tuple[str, str]:
    """(artiste, titre) — même contrat que
    `harmonia.server.youtube._video_meta`, non fatal."""
    from harmonia import titles as _titles
    import yt_dlp
    try:
        with yt_dlp.YoutubeDL({**_ydl_opts(), "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:  # yt_dlp.utils.DownloadError et alii
        log.warning("mobile_youtube.video_meta failed for %s: %s", url, exc)
        return "", ""
    return _titles.split(
        info.get("title") or "",
        artist=info.get("artist") or "",
        track=info.get("track") or "",
        uploader=info.get("uploader") or "",
    )


def download_audio(url: str, out: Path) -> tuple[str, str]:
    """Télécharge l'audio → (artiste, titre) — même contrat que
    `harmonia.server.youtube._download_audio`, mais une seule tentative (pas
    de boucle multi-client : c'est `player_client` côté extracteur qui variait
    sur le Mac, hors périmètre ici) et lève une RuntimeError lisible."""
    from harmonia import titles as _titles
    import yt_dlp
    opts = {**_ydl_opts(), "outtmpl": str(out.with_suffix("")) + ".%(ext)s"}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as exc:
        raise RuntimeError(
            f"téléchargement YouTube échoué : {exc}\n"
            "Cookies absents/expirés, ou le challenge JS a de nouveau changé "
            "(voir docs/portage_mobile/p3b_youtube_cookies_qjs.md « ce qui "
            "cassera »)."
        ) from exc
    if not out.exists():
        # yt-dlp a pu écrire sous un autre nom si `info["ext"]` != "m4a"
        # (bestaudio de repli) — pas rencontré aux 3 URL de test, documenté
        # comme ouvert.
        produced = out.with_suffix("." + (info.get("ext") or "m4a"))
        if produced.exists():
            produced.rename(out)
    artist, title = _titles.split(
        info.get("title") or "",
        artist=info.get("artist") or "",
        track=info.get("track") or "",
        uploader=info.get("uploader") or "",
    )
    return artist, title


def install() -> None:
    """Monkeypatch `harmonia.server.youtube` — appelé une fois par
    `mobile_entry.py`, AVANT toute route Flask susceptible d'appeler
    `_resolve_audio`. `harmonia/server/youtube.py` n'est pas modifié."""
    from harmonia.server import youtube as _yt

    def _ytdlp_bin():
        return "yt_dlp (in-process)" if ytdlp_available() else None

    _yt._ytdlp_bin = _ytdlp_bin
    _yt._video_meta = lambda ytdlp, url: video_meta(url)
    _yt._download_audio = lambda ytdlp, url, out: download_audio(url, out)
    log.info("mobile_youtube: harmonia.server.youtube patché (in-process, "
             "cookies=%s, qjs=%s)", _cookie_file(), _qjs_path())
