"""Placer les MOTS d'un texte connu sur l'audio : l'alignement forcé.

Louis, 2026-09-23 : « si on a déjà les paroles on peut les rentrer comme prior
d'un modèle speech to text non ? ». Oui, et c'est plus fort qu'un prior — le
texte devient une CONTRAINTE. Le modèle ne choisit plus QUELS mots ont été
chantés, seulement OÙ ils tombent.

COMMENT ÇA MARCHE, en une phrase par étage :

  1. un modèle acoustique (wav2vec2, CTC) découpe l'audio en trames de 20 ms et
     rend, pour chaque trame, la probabilité de chaque lettre ;
  2. `torchaudio.functional.forced_align` cherche le chemin monotone le plus
     probable qui épelle EXACTEMENT la suite de lettres qu'on lui donne ;
  3. on relit ce chemin pour retrouver, par mot, sa première et sa dernière
     trame — donc son début et sa fin en secondes.

POURQUOI ÇA MARCHE SUR DU CHANT ALORS QUE LA TRANSCRIPTION LIBRE ÉCHOUE. La
transcription doit explorer toutes les suites de mots possibles, et son modèle
de langue est entraîné sur de la parole : une voyelle tenue sur quatre temps,
un mélisme ou un chœur la font dérailler. Ici l'espace de recherche est réduit
à « où couper », ce qui est un problème beaucoup plus petit et beaucoup mieux
posé.

ON ALIGNE SUR LA VOIX SÉPARÉE, pas sur le mixage : `data/cache/stems/htdemucs/
<stem>/vocals.wav` existe déjà pour 60 morceaux (sorti de demucs par les
scripts de recherche). Un modèle de parole sur un mixage complet entend la
batterie et la basse comme du bruit ; sur la piste de voix il entend une voix.

CE QUE ÇA NE RÉSOUT PAS, et il faut le lire avant de croire un résultat :

  * **le texte doit être juste.** Là où le tab se trompe, saute une reprise ou
    invente une ligne, l'alignement tordra le reste pour caser ce qu'on lui a
    imposé — un alignement forcé ne peut pas dire « ce mot n'a pas été
    chanté ». Le score par mot est la seule alarme : il s'effondre là où le
    texte ne correspond pas, et c'est ce qu'il faut regarder en premier.
  * **les passages instrumentaux n'existent pas pour lui.** Entre deux couplets,
    il n'a aucune raison de laisser un trou, sauf si le texte en prévoit un.
  * **les chœurs.** demucs met tout ce qui chante dans `vocals.wav` ; deux voix
    qui disent des choses différentes en même temps ne peuvent pas être toutes
    les deux alignées sur un seul texte.
  * **le modèle est anglophone** (wav2vec2 base, 960 h de lecture en anglais).
    Sur un texte français il faudra le bundle multilingue, plus lourd.
"""
from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

#: Le modèle. Le bundle `MMS_FA` est multilingue mais pèse ~1,2 Go ; celui-ci
#: pèse ~360 Mo et suffit pour de l'anglais — le disque de Louis est à 2,4 Go
#: libres (2026-09-23), ce n'est pas un détail.
BUNDLE = "WAV2VEC2_ASR_BASE_960H"
#: sous ce score, le mot n'a probablement pas été chanté là — c'est l'alarme
#: qui dit « le texte ne colle pas », voir la docstring du module
SCORE_DOUTEUX = 0.30

_MOT = re.compile(r"[A-Za-z']+")
_BALISE = re.compile(r"\[/?(?:ch|tab)\]")
_CROCHET = re.compile(r"^\s*\[.*\]\s*$")

_MODELE = None


def _modele():
    """(modèle, étiquettes, images par seconde), chargé une fois par processus."""
    global _MODELE
    if _MODELE is not None:
        return _MODELE
    import torch
    import torchaudio.pipelines as P
    bundle = getattr(P, BUNDLE)
    mdl = bundle.get_model().eval()
    labels = list(bundle.get_labels())
    _MODELE = (mdl, labels, bundle.sample_rate)
    logger.info("paroles_align : %s chargé (%d étiquettes)", BUNDLE, len(labels))
    return _MODELE


def lignes_de_paroles(brut: str) -> list[dict]:
    """Les lignes de TEXTE d'un tab, avec leur numéro de ligne d'origine.

    On écarte les lignes d'accords (elles portent le balisage `[ch]`) et les
    en-têtes de section (`[Verse]`, `[Chorus]`) : ce sont des annotations du
    tabbeur, pas des mots chantés. Le numéro de ligne est gardé pour pouvoir
    rejoindre les accords posés par `tab_paroles.feuille`.
    """
    out = []
    for i, ligne in enumerate((brut or "").splitlines()):
        if "[ch]" in ligne:
            continue
        propre = _BALISE.sub("", ligne).strip()
        if not propre or _CROCHET.match(propre):
            continue
        if not _MOT.search(propre):
            continue
        out.append({"ligne": i, "texte": propre})
    return out


def _normalise(mot: str) -> str:
    """Le mot tel que le modèle sait l'épeler : sans accent, en majuscules.

    Le vocabulaire de wav2vec2 est A-Z plus l'apostrophe et le séparateur de
    mots. Tout le reste doit disparaître AVANT l'alignement, sinon le jeton est
    inconnu et la trame correspondante n'a aucune façon d'être expliquée.
    """
    sans = unicodedata.normalize("NFKD", mot)
    sans = "".join(c for c in sans if not unicodedata.combining(c))
    return "".join(c for c in sans.upper() if c.isalpha() or c == "'")


def mots_a_aligner(lignes: list[dict]) -> list[dict]:
    """[{mot, normalise, ligne, rang}] — la suite des mots, dans l'ordre."""
    out = []
    for l in lignes:
        for m in _MOT.finditer(l["texte"]):
            n = _normalise(m.group(0))
            if not n:
                continue
            out.append({"mot": m.group(0), "normalise": n, "ligne": l["ligne"],
                        "debut_car": m.start(), "rang": len(out)})
    return out


def aligner(vocals_wav, mots: list[dict]) -> list[dict]:
    """Chaque mot, avec son début et sa fin en secondes et son score.

    `[{rang, mot, ligne, t0, t1, score}]`. Le score est la probabilité moyenne
    du chemin sur les trames du mot : il s'effondre là où le texte ne
    correspond pas à ce qui est chanté, et c'est l'alarme la plus utile de tout
    ce module (voir `SCORE_DOUTEUX`).
    """
    import torch
    import torchaudio
    import torchaudio.functional as AF

    mdl, labels, sr_modele = _modele()
    # PAS `torchaudio.load` : depuis la 2.11 il délègue à TorchCodec, qui n'est
    # pas installé (et qu'on n'installera pas — le disque est à 2 Go libres).
    # soundfile est déjà là et lit le WAV de demucs sans rien tirer de plus.
    import numpy as np
    import soundfile as sf
    donnees, sr = sf.read(str(vocals_wav), dtype="float32", always_2d=True)
    onde = torch.from_numpy(np.ascontiguousarray(donnees.mean(1)))[None, :]
    if sr != sr_modele:
        onde = torchaudio.functional.resample(onde, sr, sr_modele)
    with torch.inference_mode():
        emission, _ = mdl(onde)
    log_probs = torch.log_softmax(emission, dim=-1)

    idx = {c: i for i, c in enumerate(labels)}
    inconnus = sorted({c for m in mots for c in m["normalise"] if c not in idx})
    if inconnus:
        # Jamais muet : un caractère hors vocabulaire fausse TOUT l'alignement
        # en aval, pas seulement son mot.
        logger.warning("paroles_align : caractères hors vocabulaire %s — "
                       "ils sont retirés", inconnus)
    jetons, bornes = [], []
    for m in mots:
        suite = [idx[c] for c in m["normalise"] if c in idx]
        if not suite:
            continue
        bornes.append((len(jetons), len(jetons) + len(suite), m))
        jetons.extend(suite)
    if not jetons:
        return []

    cible = torch.tensor([jetons], dtype=torch.int32)
    chemin, scores = AF.forced_align(log_probs, cible, blank=0)

    # `forced_align` rend L'ÉTIQUETTE DE LA LETTRE à chaque trame, PAS l'indice
    # du jeton dans la cible — se tromper là donne un alignement non monotone et
    # des mots qui sautent (mesuré le 2026-09-23 : 10 mots placés sur 171, un
    # « silence » de −120 s). `merge_tokens` est la fonction faite pour ça :
    # elle rend UNE portée par jeton de la cible, dans l'ordre.
    portees = AF.merge_tokens(chemin[0], scores[0].exp())
    if len(portees) != len(jetons):
        logger.warning("paroles_align : %d portée(s) pour %d jeton(s) — "
                       "l'alignement est suspect", len(portees), len(jetons))

    dt = onde.shape[1] / sr_modele / log_probs.shape[1]
    out = []
    for a, b, m in bornes:
        p = portees[a:b]
        if not p:
            continue
        sc = [float(x.score) for x in p]
        out.append({"rang": m["rang"], "mot": m["mot"], "ligne": m["ligne"],
                    "t0": round(p[0].start * dt, 3),
                    "t1": round(p[-1].end * dt, 3),
                    "score": round(sum(sc) / len(sc), 3)})
    return out


def avec_trous(align: list[dict], *, seuil: float = SCORE_DOUTEUX,
               silence_mini: float = 3.0) -> list[dict]:
    """L'alignement, avec des TROUS là où il ne faut pas faire confiance.

    Louis, 2026-09-23 : « rajoute une option pour pouvoir faire une pause /
    trou quand il n'y a pas de paroles ou qu'elles ne correspondent pas ».

    Un alignement forcé ne sait pas dire non : il place TOUS les mots qu'on lui
    donne, même ceux qui n'ont pas été chantés — là où le tab saute une reprise
    ou invente une ligne, il tord le reste pour les caser. Deux symptômes, deux
    trous :

      * **le score s'effondre** (`seuil`) — le mot est posé là faute de mieux ;
      * **un long silence** (`silence_mini`, en secondes) entre deux mots — un
        passage instrumental, que le modèle n'a aucune raison de nommer.

    Rend la même liste avec `trou: True` sur les mots douteux et, INSÉRÉES entre
    les mots, des entrées `{trou: True, mot: None, t0, t1}` pour les silences.
    Un lecteur karaoké peut alors se taire au lieu de surligner n'importe quoi —
    c'est la même loi que partout ici : un canal qui ne porte rien s'abstient,
    et il le DIT.
    """
    out = []
    for a, b in zip(align, align[1:] + [None]):
        m = dict(a)
        m["trou"] = a["score"] < seuil
        out.append(m)
        if b is not None and b["t0"] - a["t1"] >= silence_mini:
            out.append({"rang": None, "mot": None, "ligne": None,
                        "t0": a["t1"], "t1": b["t0"], "score": None,
                        "trou": True, "silence": True})
    return out
