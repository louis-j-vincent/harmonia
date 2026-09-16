"""Un titre YouTube → un artiste et un titre de morceau.

Louis, 2026-08-09 : « fais aussi un fix pour qu'on extraie les bons titres +
artistes des chansons, pour l'instant c'est n'importe quoi, on a des codes à la
place ».

Les codes venaient de `server._resolve_audio` : quand yt-dlp télécharge une
vidéo, le fichier est nommé d'après l'IDENTIFIANT (`J36z7AnhvOM.m4a`) et le
titre affiché était `_pretty_title(stem)`, c'est-à-dire cet identifiant passé en
capitales de titre — `J36Z7Anhvom`. Rien n'interrogeait jamais les métadonnées.

Ce module ne fait QUE du texte : pas de réseau, pas de fichiers. Il est donc
testable sans YouTube, et c'est `scripts/backfill_titles.py` qui va chercher les
métadonnées.

Trois sources, dans cet ordre de confiance :

1. les balises YouTube Music (`artist` / `track`) quand la vidéo en a — c'est de
   la métadonnée d'éditeur, pas du texte saisi à la main ;
2. le titre de la vidéo, coupé sur son tiret (« Artiste - Morceau (Official
   Video) ») ;
3. la chaîne (`uploader`), débarrassée de « VEVO » / « - Topic », quand le titre
   ne porte pas de séparateur.

Ce que ce module NE résout PAS, et il faut le savoir : un *slug* de fichier
(`ray_charles_georgia_on_my_mind_official_video`) a perdu son tiret au
slugifiage. `split()` ne peut donc pas retrouver où finit l'artiste, et il
renvoie l'artiste vide plutôt que de couper au hasard. Pour ces morceaux-là,
`scripts/backfill_titles.py` retrouve la vidéo d'origine et n'accepte sa réponse
que si le titre trouvé se re-slugifie EXACTEMENT en le stem qu'on a — une
vérification, pas une devinette.
"""
from __future__ import annotations

import re

__all__ = ["slugify", "strip_junk", "split", "pretty_from_slug"]


def slugify(text: str) -> str:
    """La règle de nommage de `docs/audio` (identique à
    `harmonia.dataset.ingest.slugify`, recopiée pour ne pas faire dépendre le
    serveur du gros paquet)."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "untitled"


# Ce qui n'est pas le nom du morceau. Chaque motif est ancré en FIN de titre ou
# vit entre parenthèses/crochets : « Live » au milieu d'un titre (« Live and Let
# Die ») doit survivre, « (Live)» à la fin doit tomber.
_JUNK = r"""
      official\s+(music\s+)?video
    | official\s+(hd\s+)?(audio|video|lyric\s*s?\s*video|visuali[sz]er)
    | official\s+lyrics?
    | music\s+video | lyric\s*s?\s*video | lyrics? | visuali[sz]er
    | audio | hd | hq | 4k | 8k | mv | m/v
    | remaster(ed)?(\s+\d{4})? | \d{4}\s+remaster(ed)?
    | explicit | clean\s+version | radio\s+edit | extended\s+version
    | full\s+(song|album|version) | album\s+version | single\s+version
    | live (\s+(at|from|in|on)\s+.+)? | live\s+session | acoustic\s+session
    | video\s+oficial | clip\s+officiel | audio\s+officiel
    | a\s+colors\s+show | colors\s+show
"""
_JUNK_RE = re.compile(_JUNK, re.I | re.X)

# « (Official Video) », « [HD] », « - Official Video », « | Lyrics »
_BRACKET = re.compile(r"[\(\[\{]\s*(?:%s)\s*[\)\]\}]" % _JUNK, re.I | re.X)
_TRAILING = re.compile(r"(?:\s*[-–—|·:]\s*|\s+)(?:%s)\s*$" % _JUNK, re.I | re.X)

# Les séparateurs artiste/titre. Le tiret DOIT être entouré d'espaces :
# « Jay-Z » et « t-shirt » ne sont pas des séparateurs. `//` est la convention
# des chaînes de labels (« Mac DeMarco // "My Kind Of Woman" »).
_SEP = re.compile(r"\s+[-–—]\s+|\s+[|·]\s+|\s+::\s+|\s*//\s*")

# Un titre mis entre guillemets par la chaîne — les guillemets ne sont pas dans
# le nom du morceau.
_QUOTED = re.compile(r"""^\s*["“”'«]\s*(.+?)\s*["“”'»]\s*$""")

# Suffixes de chaîne qui ne font pas partie du nom d'artiste.
_CHANNEL_JUNK = re.compile(
    r"\s*(?:-\s*Topic|VEVO|Vevo|Official|Officiel|Music|Records|TV|HD|"
    r"Channel|Archive)\s*$")


def strip_junk(title: str) -> str:
    """Enlève les mentions de production, en boucle jusqu'au point fixe.

    « Let It Be (Remastered 2009) [Official Video] HD » → « Let It Be ».
    On boucle parce qu'un titre en empile souvent plusieurs, et on s'arrête dès
    que plus rien ne bouge — ou si tout allait disparaître, auquel cas on rend
    le titre d'avant (un morceau qui s'appelle vraiment « Lyrics » existe).
    """
    s = (title or "").strip()
    for _ in range(6):
        before = s
        s = _BRACKET.sub(" ", s)
        s = _TRAILING.sub("", s)
        m = _QUOTED.match(s)
        if m:
            s = m.group(1)
        s = re.sub(r"\s{2,}", " ", s).strip(" -–—|·:,")
        if s == before:
            break
        if not s:
            return before.strip()
    return s or (title or "").strip()


def _hollow(text: str) -> bool:
    """Vrai si ce morceau de texte n'est QUE de la mention de production.

    `strip_junk` refuse de rendre une chaîne vide (un morceau peut s'appeler
    « Live »), donc il ne peut pas répondre à cette question — d'où ce second
    passage, sans filet, qui sert uniquement à décider si un tiret séparait
    vraiment un artiste d'un titre.
    """
    s = (text or "").strip()
    for _ in range(6):
        before = s
        s = _BRACKET.sub(" ", s)
        s = _TRAILING.sub("", s)
        s = _JUNK_RE.sub(" ", s) if _JUNK_RE.fullmatch(s.strip()) else s
        s = re.sub(r"\s{2,}", " ", s).strip(" -–—|·:,")
        if s == before:
            break
    return not s


def _clean_channel(name: str) -> str:
    s = (name or "").strip()
    for _ in range(3):
        t = _CHANNEL_JUNK.sub("", s).strip(" -–—")
        if t == s:
            break
        s = t
    return s


def split(raw_title: str, *, uploader: str = "", artist: str = "",
          track: str = "") -> tuple[str, str]:
    """→ (artiste, titre). L'un ou l'autre peut être vide : mieux vaut un champ
    vide qu'une coupe inventée.

    `artist` / `track` sont les balises YouTube Music si la vidéo en a ; elles
    gagnent sur tout le reste. yt-dlp rend la chaîne « NA » quand le champ
    manque — c'est traité comme vide.
    """
    def _na(x):
        return "" if (x or "").strip().upper() in ("", "NA", "NONE") else x.strip()

    artist, track = _na(artist), _na(track)
    if artist and track:
        return _clean_channel(artist), strip_junk(track)

    raw = (raw_title or "").strip()
    # On coupe AVANT de nettoyer : « Artiste - Titre (Official Video) » garde son
    # tiret, alors qu'un nettoyage préalable peut le faire disparaître.
    parts = _SEP.split(raw, maxsplit=1)
    # « Yesterday - Remastered 2009 » : la droite du tiret n'est QUE du bruit,
    # donc ce tiret ne séparait pas un artiste d'un titre — couper là donnerait
    # l'artiste « Yesterday ». Même raisonnement à gauche.
    if (len(parts) == 2 and parts[0].strip() and parts[1].strip()
            and not _hollow(parts[0]) and not _hollow(parts[1])):
        a, t = _clean_channel(strip_junk(parts[0])), strip_junk(parts[1])
        # « The Beatles - The Beatles - Let It Be » : certaines chaînes
        # préfixent leur propre nom au titre déjà complet. On ne coupe qu'une
        # fois, donc l'artiste se retrouve en tête du titre — on l'enlève.
        while a and t:
            m = re.match(rf"{re.escape(a)}\s*(?:[-–—|·:]|//)\s*(.+)$", t, re.I)
            if not m:
                break
            t = strip_junk(m.group(1))
        if a and t:
            return a, t

    t = strip_junk(raw)
    return (artist or _clean_channel(uploader)), t


def pretty_from_slug(stem: str) -> str:
    """Le repli quand on n'a AUCUNE métadonnée : un stem de fichier rendu
    lisible. Sans artiste — un slug a perdu son séparateur, et couper au hasard
    donnerait « Ray » / « Charles Georgia On My Mind »."""
    words = strip_junk(stem.replace("_", " ")).split()
    return " ".join(w if w.isupper() else w.capitalize() for w in words)
