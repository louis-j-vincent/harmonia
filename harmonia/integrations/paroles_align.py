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
#: SOUS CE SCORE MÉDIAN, LA LIGNE ENTIÈRE N'A PAS ÉTÉ CHANTÉE. Louis,
#: 2026-09-24 : « vire ce qui n'est pas le début de la chanson ». Pas besoin
#: d'une source extérieure pour ça — l'AUDIO le dit. Une ligne d'en-tête
#: (titre, artiste, accordage) ou de crédits ne correspond à rien dans le
#: chant, donc son score s'effondre. Mesuré le 2026-09-24 sur deux morceaux :
#:
#:     Stand By Me   lignes parasites 0,00 · 0,01 · 0,07   1er quartile 0,19
#:     This Love     lignes parasites 0,01 · 0,05 · 0,06   1er quartile 0,21
#:
#: 0,15 passe donc entre les deux populations. C'est un plancher volontairement
#: BAS : on préfère garder une ligne douteuse (elle se verra à ses mots
#: orange) que jeter une vraie ligne mal chantée.
LIGNE_MUETTE = 0.15

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


#: une ligne d'en-tête de tab : « Tuning: Standard », « Capo: 2nd fret »…
_ENTETE = re.compile(r"^[A-Za-z][A-Za-z /]{0,20}\s*:", re.I)


def lignes_de_paroles(brut: str) -> list[dict]:
    """Les lignes de TEXTE d'un tab, avec leur numéro de ligne d'origine.

    TROIS CHOSES SONT ÉCARTÉES, et la troisième a été payée :

      * les lignes d'accords (elles portent le balisage `[ch]`) ;
      * les en-têtes de section (`[Verse]`, `[Chorus]`) et les lignes de
        métadonnée (`Tuning:`, `Capo:`) — des annotations du tabbeur ;
      * **tout ce qui précède la PREMIÈRE ligne d'accords.** Louis,
        2026-09-24 : « au début tu as parsé une ou deux phrases qui ne sont pas
        les paroles de la chanson ». C'était l'en-tête du tab — titre, artiste,
        accordage — que l'alignement essayait ensuite de placer dans le chant,
        en tordant le début du morceau pour y arriver. Mesuré : 4 lignes
        parasites sur Stand By Me, 1 sur This Love.

    La règle du troisième point tient à la FORME d'un tab : les paroles y sont
    toujours sous une ligne d'accords. Rien avant la première n'est chanté.

    Le numéro de ligne est gardé pour pouvoir rejoindre les accords posés par
    `tab_paroles.feuille`.
    """
    lignes = (brut or "").splitlines()
    premier = next((i for i, l in enumerate(lignes) if "[ch]" in l), None)
    if premier is None:
        return []
    out = []
    section, saut = "", False
    for i, ligne in enumerate(lignes):
        propre = _BALISE.sub("", ligne).strip()
        # LA STRUCTURE HUMAINE EST DANS LE TAB, et on la gardait pas (Louis,
        # 2026-09-24 : « les retours à la ligne devraient correspondre à des
        # strophes, faire sens d'un point de vue humain »). Mesuré sur This
        # Love : 25 lignes vides et 7 en-têtes de section étaient jetés, d'où
        # 31 lignes indifférenciées à l'écran. Une ligne vide sépare deux
        # strophes ; un `[Chorus]` nomme ce qui suit. Les deux se retiennent au
        # lieu d'être perdus.
        if not propre:
            saut = True
            continue
        if _CROCHET.match(propre):
            section = propre.strip("[]").strip()
            saut = True
            continue
        if i < premier or "[ch]" in ligne or _ENTETE.match(propre):
            continue
        if not _MOT.search(propre):
            continue
        out.append({"ligne": i, "texte": propre, "section": section,
                    "strophe": saut})
        saut = False
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


def lignes_muettes(align: list[dict], *, plancher: float = LIGNE_MUETTE) -> set:
    """Les numéros de ligne dont AUCUN mot n'a vraiment été chanté.

    Le score médian de la ligne, comparé à `LIGNE_MUETTE`. On regarde la
    médiane et pas la moyenne : un seul mot bien placé par hasard ne doit pas
    sauver une ligne d'en-tête, et un mot raté ne doit pas condamner une vraie
    ligne.

    CE QUE ÇA NE RÉSOUT PAS : une ligne réellement chantée mais absente du
    tab reste absente — ce filtre ENLÈVE du faux, il n'ajoute pas du vrai.
    """
    import statistics as st
    par: dict = {}
    for m in align:
        if m.get("ligne") is None:
            continue
        par.setdefault(m["ligne"], []).append(float(m.get("score") or 0))
    return {k for k, v in par.items() if st.median(v) < plancher}


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


# ── juger l'alignement À L'OREILLE : la piste de temps ──────────────────────
# Louis, 2026-09-23 : « bien calée, faut pas qu'il y ait de décalage ». Un
# chiffre (171/171 mots placés, score médian 0,56) ne dit pas si un mot tombe
# sur la bonne syllabe — seule l'oreille le dit. D'où une page qui joue l'audio
# et fait défiler la piste sous une ligne fixe : si l'alignement est juste, le
# tic passe la ligne au moment où la syllabe sonne.
#
# ELLE N'ÉCRIT PAS LES PAROLES, VOLONTAIREMENT. Avec le texte à l'écran on lit
# et on croit entendre ce qu'on lit ; sans lui il ne reste que la question
# posée — ce tic tombe-t-il sur cette syllabe. C'est un instrument de mesure du
# temps, pas un afficheur de paroles.

import bisect  # noqa: E402
import html    # noqa: E402
import json    # noqa: E402
from pathlib import Path  # noqa: E402

from harmonia.settings import SETTINGS  # noqa: E402

#: la voix séparée par demucs, la seule entrée de l'alignement
def voix_de(stem: str) -> Path:
    """Le `vocals.wav` de ce morceau — qu'il existe ou non (à l'appelant de dire)."""
    return SETTINGS.data_cache / "stems" / "htdemucs" / str(stem) / "vocals.wav"


def index_courant(items: list[dict], t: float) -> int:
    """L'indice de l'élément qui SONNE à l'instant `t`, ou −1.

    −1 n'est pas une erreur : c'est un TROU. Entre deux mots il n'y a pas de
    mot, et un lecteur qui rendrait quand même le dernier (ou le suivant)
    mentirait sur le temps — exactement ce que cette page sert à détecter.
    Les éléments sont supposés triés par `t0` et non chevauchants.
    """
    if not items:
        return -1
    i = bisect.bisect_right([float(x["t0"]) for x in items], float(t)) - 1
    if i < 0:
        return -1
    return i if float(t) < float(items[i]["t1"]) else -1


def piste(align: list[dict]) -> dict:
    """La sortie d'`avec_trous` rangée pour la page : `{ticks, silences}`.

    Un tic par mot (son instant, son score, son doute, et LE MOT) ; les
    silences insérés par `avec_trous` partent dans leur propre liste — ce sont
    des ABSENCES, elles se dessinent comme des vides, pas comme des marques.

    Le mot voyage avec son tic depuis le 2026-09-24 (Louis : « le mode karaoké
    c'est juste pour debug, tu peux afficher les paroles »). Il vient du tab que
    SA machine est allée chercher ; rien ici ne l'écrit ni ne le stocke ailleurs
    que dans le cache de son serveur.
    """
    ticks, silences = [], []
    for e in align:
        if e.get("silence"):
            silences.append({"t0": float(e["t0"]), "t1": float(e["t1"])})
            continue
        ticks.append({"rang": e.get("rang"), "t0": float(e["t0"]),
                      "t1": float(e["t1"]), "score": float(e.get("score") or 0),
                      "trou": bool(e.get("trou")), "mot": e.get("mot")})
    return {"ticks": ticks, "silences": silences}


def resume(ticks: list[dict], *, seuil: float = SCORE_DOUTEUX) -> dict:
    """De quoi écrire le sous-titre : combien de mots, quelle confiance, quel trou.

    `plus_grand_trou` est le plus long intervalle SANS mot (fin d'un mot →
    début du suivant) : c'est la mesure la plus parlante de « là, il ne sait
    pas », qu'il s'agisse d'un solo ou d'un couplet que le tab a sauté.
    """
    if not ticks:
        return {"n": 0, "median": 0.0, "part_douteux": 0.0,
                "plus_grand_trou": 0.0}
    sc = sorted(t["score"] for t in ticks)
    n = len(sc)
    med = sc[n // 2] if n % 2 else (sc[n // 2 - 1] + sc[n // 2]) / 2
    trou = max((b["t0"] - a["t1"] for a, b in zip(ticks, ticks[1:])), default=0.0)
    return {"n": n, "median": round(med, 3),
            "part_douteux": round(sum(1 for s in sc if s < seuil) / n, 3),
            "plus_grand_trou": round(max(0.0, trou), 2)}


def accords_du_chart(chart: dict) -> list[dict]:
    """NOS accords, avec leurs instants et leur nom écrit.

    Le nom vient de `tab_chart.texte_du_chart` — la seule définition du
    nommage dans le projet ; en réécrire une ici donnerait deux noms pour le
    même accord.
    """
    from harmonia.integrations.tab_chart import texte_du_chart
    out = []
    for c in ((chart.get("prompter") or {}).get("chords") or []):
        out.append({"t0": round(float(c.get("t0") or 0), 3),
                    "t1": round(float(c.get("t1") or 0), 3),
                    "nom": texte_du_chart(c), "nc": bool(c.get("nc"))})
    return out


def accords_du_tab(pose: dict) -> list[dict]:
    """LES ACCORDS DU TAB, sur notre horloge et dans notre tonalité.

    `tab_chart.poser` a déjà fait le travail : ses `segments` portent l'instant
    où chaque accord du tab tombe dans NOTRE audio, et `texte` le nomme après
    la transposition qu'il a fallu pour l'y poser. On ne refait donc aucun
    alignement ici — on range.

    Les mettre sous les nôtres, sur la même ligne du temps, est la seule façon
    de voir OÙ on diverge plutôt que COMBIEN : un taux d'accord de 16 % ne dit
    pas si on se trompe partout un peu ou beaucoup à un seul endroit.
    """
    out = []
    for g in (pose or {}).get("segments") or []:
        out.append({"t0": round(float(g["t0"]), 3),
                    "t1": round(float(g["t1"]), 3),
                    "nom": g.get("texte") or "?", "nc": False})
    return out


# ── la page, UNE seule définition (même loi que `tab_paroles.page`) ─────────
# Contraintes iOS, payées cher et non négociables (voir la mémoire du projet
# et `harmonia/static/audio.js`) :
#   * PAS de boucle `requestAnimationFrame` — elle empêche le moteur audio de
#     WebKit de démarrer, et le son ne part jamais. L'horloge est l'événement
#     `timeupdate` du média, interpolé par un `setInterval` à 50 ms qui relit
#     `audio.currentTime` ;
#   * l'audio se charge par `fetch()` + `blob:` — le chargeur média de Safari
#     ne remplit parfois JAMAIS son tampon (tempête de 206), alors que
#     `fetch` marche. C'est le même geste que `swapToBlob()` du shell.

_GABARIT = r"""<!doctype html><html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>__TITRE__</title><style>
:root{--pap:#f6f3ec;--enc:#1c1c1c;--nous:#8c2f2f;--gris:#9a9287;--lig:#e2dcd0;
  --tic:#2f6b8c;--dout:#c98b2f;--sil:#e9e3d6}
@media(prefers-color-scheme:dark){:root{--pap:#16150f;--enc:#ece7dc;
  --nous:#e08a7a;--gris:#7d766b;--lig:#2b2820;--tic:#77b6d8;--dout:#e0b25e;
  --sil:#221f18}}
*{box-sizing:border-box}
html,body{overflow-x:hidden}
body{margin:0;padding:12px;background:var(--pap);color:var(--enc);
  font:15px/1.45 -apple-system,system-ui,sans-serif;max-width:46rem;
  margin:0 auto;-webkit-text-size-adjust:100%}
h1{font:600 19px/1.2 Georgia,serif;margin:0 0 2px}
.sous{color:var(--gris);font-size:12px;margin:0 0 10px}
.etat{font-size:12px;color:var(--gris);margin:0 0 8px}
.maintenant{display:flex;align-items:baseline;gap:10px;margin:0 0 6px;
  min-height:38px}
#acNow{font:700 30px/1 Georgia,serif;color:var(--nous)}
#acNext{font-size:13px;color:var(--gris)}
.piste{position:relative;height:128px;overflow:hidden;border:1px solid var(--lig);
  border-radius:10px;background:linear-gradient(var(--pap),var(--pap));
  touch-action:manipulation}
.ruban{position:absolute;top:0;left:0;height:100%;will-change:transform}
.ruban>div{position:absolute}
.g{top:0;height:100%;width:1px;background:var(--lig);opacity:.55}
.g5{opacity:1}
.sil{top:34px;height:74px;background:var(--sil);border-left:1px dashed var(--gris);
  border-right:1px dashed var(--gris)}
.ac{top:6px;height:24px;line-height:24px;border-left:2px solid var(--nous);
  padding-left:4px;font:600 13px/24px -apple-system,system-ui,sans-serif;
  color:var(--nous);white-space:nowrap;overflow:hidden;opacity:.5}
.ac.nc{border-left-color:var(--gris);opacity:.25}
.ac.on{opacity:1}
.act{top:32px;height:22px;line-height:22px;border-left:2px solid var(--eux);
  padding-left:4px;font:500 12px/22px -apple-system,system-ui,sans-serif;
  color:var(--eux);white-space:nowrap;overflow:hidden;opacity:.45}
.act.on{opacity:1}
.act.diff{color:var(--dout);border-left-color:var(--dout)}
.dur{bottom:18px;height:2px;background:var(--tic);opacity:.25}
.tic{bottom:22px;width:3px;border-radius:2px;background:var(--tic)}
.tic.dout{background:var(--dout)}
.tic.on{box-shadow:0 0 0 2px var(--pap),0 0 0 4px var(--tic)}
.mot{bottom:2px;height:16px;line-height:16px;font:12px/16px -apple-system,
  system-ui,sans-serif;color:var(--tic);white-space:nowrap;opacity:.45;
  transform:translateX(-50%);pointer-events:none}
.mot.dout{color:var(--dout)}
.mot.on{opacity:1;font-weight:700}
.ligne{position:absolute;left:50%;top:0;width:2px;height:100%;
  background:var(--nous);opacity:.85;pointer-events:none}
.ligne:after{content:"";position:absolute;left:-4px;top:32px;width:10px;
  height:10px;border-radius:50%;background:var(--nous)}
.barre{display:flex;align-items:center;gap:8px;margin:8px 0 6px}
button{border:1px solid var(--lig);background:transparent;color:var(--enc);
  border-radius:9px;padding:8px 13px;font:600 14px -apple-system,system-ui;
  cursor:pointer;-webkit-tap-highlight-color:transparent}
button:disabled{opacity:.35}
.z{font-size:12px;color:var(--gris);margin-left:auto}
.apercu{position:relative;height:30px;border:1px solid var(--lig);
  border-radius:8px;overflow:hidden;margin:0 0 8px;touch-action:manipulation}
.apercu>i{position:absolute;bottom:4px;width:1px;background:var(--tic);
  opacity:.65}
.apercu>i.dout{background:var(--dout)}
.apercu>u{position:absolute;top:0;bottom:0;background:var(--sil)}
#tete{position:absolute;top:0;bottom:0;width:2px;background:var(--nous)}
.lag{margin:0 0 10px;font:11.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
  color:var(--gris);white-space:pre-wrap}
.note{font-size:12px;color:var(--gris);border-top:1px solid var(--lig);
  padding-top:8px;margin:0}
b{color:var(--enc)}
</style>
<h1>__TITRE__</h1>
<p class="sous">__SOUS__</p>
<div class="etat" id="etat">chargement de l'audio…</div>
<div class="maintenant"><span id="acNow">—</span><span id="acNext"></span></div>
<div class="piste" id="piste"><div class="ruban" id="ruban"></div>
  <div class="ligne"></div></div>
<div class="barre"><button id="play" disabled>▶︎</button>
  <button id="moins">−</button><button id="plus">+</button>
  <span class="z" id="zoom"></span></div>
<div class="barre"><button id="dmoins">− 20 ms</button>
  <button id="dplus">+ 20 ms</button>
  <span class="z" id="dec"></span></div>
<div class="apercu" id="apercu"><div id="tete"></div></div>
<pre class="lag" id="lag"></pre>
<p class="note">Le bouton « ms » décale l'AFFICHAGE, pas le son : si
tous les traits passent la ligne un poil avant ce que vous entendez, le
retard est celui de votre sortie audio (Bluetooth, ~150 ms), pas celui de
l'alignement. Compensez jusqu'à ce que ça tombe, et lisez la valeur : ce
qui RESTE de dérive après ça est à nous.<br>
Un trait bleu = un mot placé, sa <b>hauteur</b> est la
confiance ; en orange, un mot dont le score s'est effondré. Les plages
grisées sont les silences, les vides sont des vides. Les paroles ne sont
pas écrites : la question est « le trait passe-t-il la ligne rouge quand la
syllabe sonne ? », et le texte à l'écran ferait entendre ce qu'on lit.</p>
<script>
var D = __DONNEES__;
var ruban=document.getElementById('ruban'), piste=document.getElementById('piste'),
    lag=document.getElementById('lag'), etat=document.getElementById('etat'),
    acNow=document.getElementById('acNow'), acNext=document.getElementById('acNext'),
    btn=document.getElementById('play'), tete=document.getElementById('tete'),
    apercu=document.getElementById('apercu'), zoom=document.getElementById('zoom');
var NIV=[30,60,110,200], iNiv=1, PPS=NIV[iNiv], CENTRE=piste.clientWidth/2;
var noeuds=[], acEls=[], actEls=[], ticEls=[], motEls=[];

function bloc(cls,t0,t1){ var d=document.createElement('div'); d.className=cls;
  d._t0=t0; if(t1!=null) d._t1=t1; ruban.appendChild(d); noeuds.push(d); return d; }

for(var s=0;s<=Math.ceil(D.duree);s++) bloc('g'+(s%5===0?' g5':''),s,null);
D.silences.forEach(function(x){ bloc('sil',x.t0,x.t1); });
D.accords.forEach(function(a){ var d=bloc('ac'+(a.nc?' nc':''),a.t0,a.t1);
  d.textContent=a.nc?'N.C.':a.nom; acEls.push(d); });
// LA VOIE DU TAB, juste dessous. En orange quand elle ne dit pas la même chose
// que nous au même instant : c'est ce désaccord-là qui est à écouter.
function notreA(t){ for(var i=0;i<D.accords.length;i++){
  if(D.accords[i].t0<=t && t<D.accords[i].t1) return D.accords[i].nom; } return null; }
(D.accords_tab||[]).forEach(function(a){
  var n=notreA(a.t0), diff=(n!==null && n!==a.nom);
  var d=bloc('act'+(diff?' diff':''),a.t0,a.t1);
  d.textContent=a.nom; actEls.push(d); });
D.ticks.forEach(function(w){
  bloc('dur',w.t0,Math.max(w.t1,w.t0+0.04));
  var d=bloc('tic'+(w.trou?' dout':''),w.t0,null);
  var sc=Math.max(0,Math.min(1,w.score));
  d.style.height=(10+52*sc)+'px'; d.style.opacity=(0.3+0.7*sc).toFixed(2);
  ticEls.push(d);
  // LE MOT SOUS SON TIC. C'est ce qui permet de juger la syllabe et pas
  // seulement l'instant : un tic peut tomber pile sur une attaque et
  // concerner le mot d'à côté.
  if(w.mot){ var m=bloc('mot'+(w.trou?' dout':''),w.t0,null);
    m.textContent=w.mot; motEls.push(m); } else { motEls.push(null); } });

// l'aperçu du morceau entier : pour sauter d'un couplet à l'autre
D.silences.forEach(function(x){ var u=document.createElement('u');
  u.style.left=(100*x.t0/D.duree)+'%'; u.style.width=Math.max(0.3,100*(x.t1-x.t0)/D.duree)+'%';
  apercu.appendChild(u); });
D.ticks.forEach(function(w){ var i=document.createElement('i');
  i.style.left=(100*w.t0/D.duree)+'%';
  i.style.height=(4+18*Math.max(0,Math.min(1,w.score)))+'px';
  if(w.trou) i.className='dout'; apercu.appendChild(i); });

function poser(){
  ruban.style.width=(D.duree*PPS)+'px';
  for(var k=0;k<noeuds.length;k++){ var e=noeuds[k];
    e.style.left=(e._t0*PPS)+'px';
    if(e._t1!=null) e.style.width=Math.max(1,(e._t1-e._t0)*PPS)+'px'; }
  zoom.textContent=PPS+' px/s · '+(piste.clientWidth/PPS).toFixed(1)+' s à l’écran';
}
function mesure(){ CENTRE=piste.clientWidth/2; poser(); peindre(); }

// L'indice de l'élément qui SONNE à t, −1 s'il n'y en a pas (un trou est un
// trou : rien à surligner). Jumeau de `paroles_align.index_courant`.
function courant(arr,t){ var lo=0,hi=arr.length-1,r=-1;
  while(lo<=hi){ var m=(lo+hi)>>1; if(arr[m].t0<=t){ r=m; lo=m+1; } else hi=m-1; }
  if(r<0) return -1; return t<arr[r].t1 ? r : -1; }
function dernier(arr,t){ var lo=0,hi=arr.length-1,r=-1;
  while(lo<=hi){ var m=(lo+hi)>>1; if(arr[m].t0<=t){ r=m; lo=m+1; } else hi=m-1; }
  return r; }

// ── l'horloge : `timeupdate` + un setInterval de 50 ms, JAMAIS de rAF ──────
var audio=new Audio(); audio.preload='auto'; audio.playsInline=true;
try{ audio.setAttribute('playsinline',''); audio.setAttribute('webkit-playsinline',''); }catch(e){}
var iAc=-2, iAct=-2, iTic=-2, tPeint=0, dtEcran=0, tTU=0, pret=false;
// COMPENSATION DE SORTIE AUDIO, en ms : elle décale l'AFFICHAGE, jamais le
// son. Sans elle on ne peut pas distinguer « notre alignement est en avance »
// de « la sortie Bluetooth de l'iPhone a 150 ms de retard » — deux causes
// opposées pour le même symptôme. Gardée par appareil (localStorage, dans un
// try : en navigation privée la lecture peut lever).
var dec=0; try{ dec=parseFloat(localStorage.getItem('harmKaraokeDecMs'))||0; }catch(e){}

function peindre(){
  var now=(window.performance&&performance.now)?performance.now():Date.now();
  if(tPeint) dtEcran=now-tPeint; tPeint=now;
  var brut=audio.currentTime||0, t=brut-dec/1000;
  ruban.style.transform='translate3d('+(CENTRE-t*PPS).toFixed(1)+'px,0,0)';
  tete.style.left=(100*Math.min(1,t/D.duree))+'%';
  var a=courant(D.accords,t);
  if(a!==iAc){ if(acEls[iAc]) acEls[iAc].classList.remove('on');
    if(acEls[a]) acEls[a].classList.add('on'); iAc=a;
    acNow.textContent=a>=0?(D.accords[a].nc?'N.C.':D.accords[a].nom):'—';
    var n=D.accords[(a<0?dernier(D.accords,t):a)+1];
    acNext.textContent=n?('puis '+(n.nc?'N.C.':n.nom)):''; }
  var kt=courant(D.accords_tab||[],t);
  if(kt!==iAct){ if(actEls[iAct]) actEls[iAct].classList.remove('on');
    if(actEls[kt]) actEls[kt].classList.add('on'); iAct=kt; }
  var w=courant(D.ticks,t);
  if(w!==iTic){ if(ticEls[iTic]) ticEls[iTic].classList.remove('on');
    if(motEls[iTic]) motEls[iTic].classList.remove('on');
    if(ticEls[w]) ticEls[w].classList.add('on');
    if(motEls[w]) motEls[w].classList.add('on'); iTic=w; }
  var j=dernier(D.ticks,t);
  var depuis=j>=0?(t-D.ticks[j].t0):0;
  lag.textContent=
    't = '+brut.toFixed(2)+' s   mot '+(j+1)+'/'+D.ticks.length+
    (w>=0?' (en cours)':'  (trou)')+'   +'+depuis.toFixed(2)+' s depuis son tic\n'+
    'écran '+Math.round(dtEcran)+' ms   horloge audio +'+
    Math.round(tTU?now-tTU:0)+' ms   accord '+(iAc+1)+'/'+D.accords.length+
    '   affichage '+(dec>=0?'+':'')+Math.round(dec)+' ms';
}
audio.addEventListener('timeupdate',function(){
  tTU=(window.performance&&performance.now)?performance.now():Date.now();
  peindre(); });
audio.addEventListener('seeked',peindre);
audio.addEventListener('play',function(){ btn.textContent='⏸'; });
audio.addEventListener('pause',function(){ btn.textContent='▶︎'; });
audio.addEventListener('loadedmetadata',function(){
  if(audio.duration&&isFinite(audio.duration)&&audio.duration>D.duree){
    D.duree=audio.duration; poser(); }
  peindre(); });
setInterval(function(){ if(!audio.paused) peindre(); },50);

// fetch + blob : le chargeur média de WebKit ne remplit parfois jamais son
// tampon (`harmonia/static/audio.js`, swapToBlob). On ne lui donne que des
// octets déjà locaux.
fetch(D.audio).then(function(r){ if(!r.ok) throw new Error('HTTP '+r.status);
    return r.blob(); })
  .then(function(b){ audio.src=URL.createObjectURL(b); audio.load(); pret=true;
    btn.disabled=false;
    etat.textContent='audio prêt ('+(b.size/1048576).toFixed(1)+' Mo en mémoire)'; })
  .catch(function(e){ etat.textContent='audio NON chargé : '+e+
    ' — la page ne peut rien juger sans le son.'; });

btn.onclick=function(){ if(!pret) return;
  if(audio.paused) audio.play().catch(function(e){
    etat.textContent='lecture refusée : '+e; }); else audio.pause(); };
document.getElementById('moins').onclick=function(){
  iNiv=Math.max(0,iNiv-1); PPS=NIV[iNiv]; poser(); peindre(); };
document.getElementById('plus').onclick=function(){
  iNiv=Math.min(NIV.length-1,iNiv+1); PPS=NIV[iNiv]; poser(); peindre(); };
function bouge(d){ dec=Math.max(-500,Math.min(500,dec+d));
  try{ localStorage.setItem('harmKaraokeDecMs',String(dec)); }catch(e){}
  document.getElementById('dec').textContent='affichage '+(dec>=0?'+':'')+dec+' ms';
  peindre(); }
document.getElementById('dmoins').onclick=function(){ bouge(-20); };
document.getElementById('dplus').onclick=function(){ bouge(20); };
bouge(0);
apercu.onclick=function(ev){ var r=apercu.getBoundingClientRect();
  try{ audio.currentTime=Math.max(0,D.duree*(ev.clientX-r.left)/r.width); }catch(e){}
  peindre(); };
piste.onclick=function(ev){ var r=piste.getBoundingClientRect();
  try{ audio.currentTime=Math.max(0,(audio.currentTime||0)+(ev.clientX-r.left-CENTRE)/PPS); }catch(e){}
  peindre(); };
window.addEventListener('resize',mesure);
mesure();
</script></html>
"""


def page(titre: str, sous: str, donnees: dict) -> str:
    """La page karaoké de ce morceau — HTML autonome, rien à charger.

    `donnees` : `{audio, duree, accords, ticks, silences}`. Tout est déjà
    calculé côté serveur ; la page ne fait que faire défiler et compter.
    """
    charge = json.dumps(donnees, ensure_ascii=False).replace("</", "<\\/")
    return (_GABARIT.replace("__TITRE__", html.escape(titre))
            .replace("__SOUS__", sous)
            .replace("__DONNEES__", charge))
