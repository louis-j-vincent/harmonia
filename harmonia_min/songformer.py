"""harmonia_min/songformer.py — les sections viennent d'un modèle pré-entraîné.

Louis, 2026-08-18, après avoir écouté `/plots/songformer.html` où sa propre
annotation, SongFormer et notre détecteur jouaient côte à côte sur dix-neuf
morceaux : « je suis d'accord avec lui partout, on le prend en prod ». C'est
donc le détecteur de sections par défaut (`HARMONIA_SECTIONS=songformer`), et
`voice` / `harmonic` / `chroma` restent joignables derrière la même variable.

CE QUE FAIT LE MODÈLE (lu dans `modeling_songformer.py`, pas dans son README).
SongFormer (ASLP-lab, octobre 2025) n'écoute pas les accords : il écoute le
SON. Le morceau est rééchantillonné à 24 kHz puis donné à DEUX encodeurs audio
pré-entraînés, MuQ et MusicFM, dont on prend la 10ᵉ couche cachée. Chacun voit
le morceau à deux échelles — par fenêtres de 30 s (le détail local) et par
fenêtres de 420 s (le morceau entier d'un coup) — soit quatre flux de vecteurs
concaténés. Un Transformer entraîné sur des morceaux annotés à la main lit
cette bande et sort, à 8,33 images par seconde, DEUX choses : la probabilité
qu'il y ait une frontière ici, et la probabilité de chaque étiquette (intro,
couplet, refrain, pont, instrumental, outro, silence, pré-refrain). Les
frontières sont les pics locaux de la première courbe ; l'étiquette d'un
segment est la moyenne de la seconde entre deux frontières, argmax. Quelques
règles finales recollent les segments d'une seconde en tête et en queue.

CE QUE ÇA CHANGE POUR NOUS. Nos trois détecteurs cherchaient une RÉPÉTITION
(mêmes accords, même mélodie, même chroma) et en déduisaient les lettres. Lui
RECONNAÎT le rôle d'un passage à sa texture, comme un auditeur qui entend
« ça, c'est un refrain » sans avoir besoin de l'avoir déjà entendu. Il trouve
donc les frontières qu'aucune répétition ne trahit, et il nomme.

TROIS FRICTIONS, toutes payées et documentées ici pour ne pas les repayer :

1. `AutoModel.from_pretrained` NE MARCHE PAS avec transformers 5.x. Le modèle
   est construit sous le device « meta » (tenseurs sans mémoire, matérialisés
   après coup) et le MuQ interne y mélange des tenseurs cpu et meta :
   « Tensor on device cpu is not on the expected device meta! », avec ou sans
   `low_cpu_mem_usage`. On importe donc la classe du dépôt à la main et on la
   construit nous-mêmes sous `torch.device("cpu")`.
2. LES POIDS NE SONT PAS DANS LE FICHIER QUI PORTE LE NOM DU MODÈLE.
   `SongFormer.safetensors` n'est que la copie EMA de la tête (62 tenseurs) ;
   `model.safetensors` porte tout (1052 tenseurs : MuQ + MusicFM + la tête).
3. Les dépendances du dépôt veulent faire monter torch de 2.12 à 2.13, ce qui
   casserait beatthis / demucs / musx. Elles ont été installées `--no-deps`
   avec `torchvision==0.27.*`. Ne pas relancer un `pip install` naïf ici.

COÛT, ET POURQUOI UN SOUS-PROCESSUS. CPU seulement (les noyaux de MuQ ne
passent pas tous sur MPS) : environ une minute de chargement, puis quelques
dizaines de secondes par morceau. Surtout, **il mange trop de mémoire pour
tourner dans le serveur** : sur une machine de 16 Go, `autumn_leaves` (422 s)
fait tuer le processus par l'OS (SIGKILL, code 137) — mesuré le 2026-08-19,
deux fois de suite, serveur perdu EN COURS DE JOB après publication du chart
brut. MuQ + MusicFM + le Transformer, et une fenêtre de 420 s, ne tiennent pas.

On l'exécute donc dans un **processus enfant jetable** (`_dans_un_enfant`).
Quand l'OS tue quelque chose, il tue l'enfant : le serveur, lui, reçoit une
erreur propre, la journalise en ERROR et retombe sur le détecteur `voice`. La
mémoire est rendue à chaque morceau au lieu de rester bloquée dans le serveur.
Le prix est le rechargement du modèle par morceau — payé une seule fois grâce
au cache disque (`state/songformer/`), qu'on peut remplir d'avance avec
`scripts/songformer_prechauffe.py`.

Un morceau de 125 s passe, un de 422 s ne passe pas ; le seuil exact n'est pas
mesuré (docs/known_issues.md, 2026-08-19). Tant qu'il ne l'est pas, le repli
sur `voice` est ce qui garde l'app debout sur les morceaux longs.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DEPOT = "ASLP-lab/SongFormer"
CACHE = Path(__file__).resolve().parent / "state" / "songformer"

# Ce que le modèle sait dire, en français, pour les journaux et les pages.
NOMS = {
    "intro": "intro", "verse": "couplet", "chorus": "refrain",
    "bridge": "pont", "inst": "instrumental", "outro": "outro",
    "silence": "silence", "prechorus": "pré-refrain",
    "pre-chorus": "pré-refrain", "solo": "solo", "break": "break",
    "transition": "transition", "instintro": "intro instrumentale",
}

# Ce qui, au tout début du morceau, s'appelle « intro » sur un chart.
DEBUTS = ("intro", "silence", "fadein", "rhythmlessintro", "instintro")
# …et à la toute fin, « outro ».
FINS = ("outro", "silence", "vocaloutro", "bigoutro", "fadeout")

DELAI = 1800          # secondes : au-delà, l'enfant est abandonné

_MODELE = None


# ── le modèle ───────────────────────────────────────────────────────────────
def _modele():
    """Le modèle, chargé une seule fois par processus. Lève si ça rate."""
    global _MODELE
    if _MODELE is not None:
        return _MODELE
    import importlib.util

    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    local = snapshot_download(repo_id=DEPOT, repo_type="model",
                              ignore_patterns=["SongFormer.pt", "figs/*"])
    os.environ["SONGFORMER_LOCAL_DIR"] = local
    # AJOUTÉ EN FIN DE `sys.path`, PAS EN TÊTE : le dépôt expose des noms très
    # génériques (`model`, `dataset`, `musicfm`, `postprocessing`). En tête ils
    # pourraient masquer un module à nous en silence ; en queue, le pire qui
    # puisse arriver est un ImportError bruyant. On l'enlève dès que la classe
    # est importée — `sys.modules` garde ce qu'il faut.
    sys.path.append(local)
    try:
        spec = importlib.util.spec_from_file_location(
            "msf", os.path.join(local, "modeling_songformer.py"))
        msf = importlib.util.module_from_spec(spec)
        sys.modules["msf"] = msf
        spec.loader.exec_module(msf)
    finally:
        if local in sys.path:
            sys.path.remove(local)
    cfg = msf.SongFormerConfig.from_pretrained(local)
    with torch.device("cpu"):          # voir friction 1 dans la docstring
        mdl = msf.SongFormerModel(cfg)
    miss, unexp = mdl.load_state_dict(       # voir friction 2
        load_file(os.path.join(local, "model.safetensors")), strict=False)
    logger.info("songformer : poids chargés (%d manquants, %d inattendus)",
                len(miss), len(unexp))
    _MODELE = mdl.to("cpu").eval()
    return _MODELE


def _cle(audio: str | Path) -> Path:
    return CACHE / f"{Path(audio).name}.json"


def segments(audio: str | Path, *, force: bool = False) -> list[dict]:
    """[{label, start, end}] en secondes. Mis en cache sur disque.

    Le cache est invalidé si le fichier audio change de taille — le nom seul ne
    suffit pas, un ré-encodage garderait le même nom.
    """
    audio = Path(audio)
    taille = audio.stat().st_size
    f = _cle(audio)
    if f.exists() and not force:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            if d.get("taille") == taille:
                return d["segments"]
            logger.info("songformer : %s a changé de taille, on recalcule",
                        audio.name)
        except Exception:
            logger.exception("songformer : cache illisible %s", f)
    segs = _dans_un_enfant(audio)
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"taille": taille, "segments": segs},
                            ensure_ascii=False), encoding="utf-8")
    return segs


def _dans_un_enfant(audio: Path) -> list[dict]:
    """Faire tourner le modèle dans un processus jetable, et rendre ses segments.

    Voir la docstring du module : sur un morceau long, l'OS tue le processus qui
    porte le modèle. S'il porte AUSSI le serveur, l'app perd le serveur au
    milieu d'un job. Ici c'est l'enfant qui meurt, et l'appelant reçoit une
    exception qu'il peut journaliser et rattraper.
    """
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "segments.json"
        r = subprocess.run(
            [sys.executable, "-m", "harmonia_min.songformer",
             str(audio), str(out)],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True, text=True, timeout=DELAI)
        if r.returncode != 0 or not out.exists():
            # Le cas qui compte : returncode 137 / -9 = SIGKILL, l'OS a manqué
            # de mémoire. On le nomme, sinon on cherchera un bug de code.
            tue = r.returncode in (137, -9)
            queue = (r.stderr or "").strip().splitlines()[-3:]
            raise RuntimeError(
                ("mémoire épuisée : l'OS a tué le processus SongFormer "
                 f"({audio.name})" if tue else
                 f"SongFormer a échoué sur {audio.name} (code {r.returncode})")
                + ("\n" + "\n".join(queue) if queue else ""))
        return json.loads(out.read_text(encoding="utf-8"))


def _enfant() -> None:
    """Le corps du processus enfant : un morceau, un fichier de segments."""
    import torch
    audio, sortie = sys.argv[1], sys.argv[2]
    mdl = _modele()
    with torch.no_grad():
        segs = mdl(str(audio))
    Path(sortie).write_text(
        json.dumps([{"label": str(s["label"]), "start": float(s["start"]),
                     "end": float(s["end"])} for s in segs],
                   ensure_ascii=False), encoding="utf-8")


# ── des secondes aux mesures ────────────────────────────────────────────────
def _sur_la_grille(segs: list[dict], grid: list[float]) -> list[dict]:
    """Chaque frontière du modèle est tirée sur la barre de mesure la + proche.

    Le modèle ne connaît pas notre grille : il coupe au dixième de seconde près,
    souvent une fraction de mesure avant le vrai départ. Une section de chart
    commence sur une barre — donc on arrondit, et on jette ce qui s'écrase à
    zéro mesure (une frontière de moins d'une mesure n'est pas une section).
    """
    import numpy as np
    g = np.asarray(grid, float)
    n = len(grid) - 1
    out: list[dict] = []
    for s in segs:
        b0 = int(np.argmin(np.abs(g[:n] - s["start"])))
        if out and b0 <= out[-1]["b0"]:
            continue                     # deux frontières dans la même mesure
        out.append({"b0": b0, "label": s["label"]})
    if not out:
        return [{"b0": 0, "b1": n - 1, "label": "verse"}]
    out[0]["b0"] = 0                     # la 1re section part de la 1re mesure
    for a, b in zip(out, out[1:]):
        a["b1"] = b["b0"] - 1
    out[-1]["b1"] = n - 1
    return out


def _fondre_muets(segs: list[dict]) -> list[dict]:
    """Le silence n'est pas une section : il rejoint son voisin."""
    out = [dict(s) for s in segs if s["label"] != "silence"] or [dict(segs[0])]
    if len(out) != len(segs):
        out[0]["b0"] = segs[0]["b0"]
        out[-1]["b1"] = segs[-1]["b1"]
        for a, b in zip(out, out[1:]):
            a["b1"] = b["b0"] - 1
    return out


def _lettres(segs: list[dict], *, avec_intro: bool = True) -> list[dict]:
    """intro / outro gardent leur nom ; le reste devient A, B, C… PAR RÔLE.

    Deux passages que le modèle appelle tous les deux « refrain » reçoivent la
    MÊME lettre, même s'ils ne font pas le même nombre de mesures. C'est tout
    l'intérêt du modèle : il reconnaît le rôle d'un passage, pas sa longueur.

    ET LA LONGUEUR ALORS ? « Under-fold, never over-fold » (Louis, 2026-07-30)
    dit qu'une section s'écrit à la longueur qu'elle JOUE, et que deux
    occurrences de longueurs différentes ne s'empilent pas. Cette règle est
    tenue EN AVAL, à l'endroit où elle a un effet, pas ici :

      * `soudure.sections_pour_chart` groupe par (lettre, longueur) — deux
        refrains de 8 et de 12 mesures donnent deux blocs, tous deux nommés B ;
      * `folding.fold_display` fait pareil depuis le 2026-08-18 (il groupait
        par lettre seule, et écrasait un passage de 8 mesures sur un bloc de
        4 — le bug de lecture d'Another Day) ;
      * `folding.fold_letter_groups(loop="occurrence")` n'empile entre elles
        que les occurrences de longueur majoritaire.

    Séparer ici aurait rendu la lettre au détecteur ce que le repli lui aurait
    déjà retiré : sur Another Day, un refrain joué cinq fois (9, 7, 8, 8 et 12
    mesures — la frontière du modèle tombe au demi-temps près) serait ressorti
    en QUATRE lettres différentes, et le chart aurait dit « quatre sections »
    là où l'oreille entend « le refrain, cinq fois ».
    """
    n = len(segs)
    ren: dict[str, str] = {}
    for i, s in enumerate(segs):
        lab = s["label"]
        # Le nom imposé (intro/outro) est distinct du rôle entendu : un
        # passage étiqueté « intro » AU MILIEU d'un morceau prend une lettre.
        if avec_intro and i == 0 and lab in DEBUTS:
            s["label"] = "intro"
            continue
        if i == n - 1 and n > 1 and lab in FINS:
            s["label"] = "outro"
            continue
        if lab not in ren:
            ren[lab] = chr(ord("A") + len(ren))
        s["label"] = ren[lab]
    return segs


def detect_sections(grid, audio, form_start=None, **_) -> list[dict]:
    """[{b0, b1, label}] sur les indices de mesure — contigu, couvrant.

    Même contrat que `voice_sections.detect_sections` et
    `harmonic_sections.detect_sections` : la sortie pave le morceau sans trou
    ni chevauchement, ce qui est vérifié avant de rendre.

    `form_start` (la marque « Set bar 1 » de l'utilisateur) : quand elle est
    passée, il n'y a PAS d'intro à trouver — la marque EST le début de la
    forme, tout ce qui précède a déjà été découpé par l'appelant.
    """
    n = len(grid) - 1
    if n < 2:
        return [{"b0": 0, "b1": max(0, n - 1), "label": "A"}]
    segs = segments(audio)
    noms = " ".join(NOMS.get(s["label"], s["label"]) for s in segs)
    segs = _lettres(_fondre_muets(_sur_la_grille(segs, grid)),
                    avec_intro=form_start is None)
    for a, c in zip(segs, segs[1:]):
        assert c["b0"] == a["b1"] + 1, f"trou/chevauchement : {a} -> {c}"
    assert segs[0]["b0"] == 0 and segs[-1]["b1"] == n - 1, "pavage incomplet"
    logger.info("sections=songformer : %d section(s) — %s", len(segs), noms)
    return [{"b0": s["b0"], "b1": s["b1"], "label": s["label"]} for s in segs]


if __name__ == "__main__":       # `python -m harmonia_min.songformer AUDIO OUT`
    _enfant()
