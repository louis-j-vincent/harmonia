"""Réglage : quel `.pte` de Beat This! charger (T5, 2026-10-27 → Réglages
généraux, 2026-10-28).

POURQUOI. Le premier test réel sur le téléphone de Louis (Samsung Galaxy
A32, bas de gamme) a trouvé « finding the beat » à ≥1 min sur un morceau de
~4 min — au-dessus du critère du projet (4 min de musique < 2 min de calcul
total). `docs/portage_mobile/t5prep_leviers_vitesse.md` a déjà mesuré deux
leviers sur Mac (small0, int8) SANS trancher — Louis veut comparer À
L'OREILLE sur son propre téléphone, pas sur un chiffre Mac. Ce module rend
ça possible sans reconstruire l'APK à chaque essai :
`BeatsRuntime.load()` (`BeatsRuntime.kt`) accepte déjà un nouveau chemin de
`.pte` à chaud (ne fait rien si le chemin est identique, sinon ferme
l'ancien module et charge le nouveau) — il ne manquait qu'un moyen de
l'appeler depuis l'UI.

PAS DANS `harmonia/` : exactement le patron de `mobile_debug.py` — propre à
la plateforme mobile, monté par `mobile_entry.start_server` APRÈS
`create_app()`, jamais importé par le code partagé.

CE QUE LE SWITCH FAIT, ET POURQUOI C'EST SUFFISANT. Il recharge
`BeatsRuntime` avec le `.pte` demandé (mêmes trois noms que
`t1c_beat_this_complet.md`/`t5prep_leviers_vitesse.md` : `final0`
production, `small0` 10x plus petit, `int8` quantifié) PUIS vide le cache
disque des battues (`state/cache/beats/*.json`) pour que la prochaine
analyse recalcule au lieu de rendre un résultat déjà en cache sous l'ancien
modèle — `harmonia.beats.track()` ne sait pas QUEL modèle a produit une
entrée de cache, seulement le `stem+taille` du fichier audio, donc sans ce
vidage un changement de modèle serait invisible sur un morceau déjà
analysé. Sans risque pour un autre morceau : ce cache est celui DU
TÉLÉPHONE (`state/` sous `filesDir`, jamais partagé avec le Mac de Louis).

CE QUE CE MODULE NE RÉSOUT PAS : il ne mesure rien lui-même — c'est
`mobile_timing.py` (déjà monté) qui chronomètre chaque brique du prochain
`/api/analyze`. Il ne touche pas non plus `backends_mobile._pte_path`, qui
ne sert plus une fois `beats_runtime="java"` installé (le chemin réel passe
par `beats_infer._chunk_runner`, posé une fois pour toutes par
`mobile_beats_runtime.install` — voir ce module) ; on le met à jour quand
même pour que les rapports de debug qui le lisent restent cohérents.

RÉGLAGES GÉNÉRAUX (2026-10-28, demande de Louis). `set_model()` persiste
désormais le choix via `mobile_settings.write_default_beat_model` — le
même bouton sert à la fois le sélecteur rapide d'un chart ouvert
(`beat_editor.js::openBeatModelDebug`) et l'écran Réglages
(`settings.js`), parce que Louis n'a demandé qu'UN seul réglage, pas deux
états (« ce que j'écoute maintenant » vs « le vrai défaut ») à tenir
synchronisés. `mobile_entry.py` lit ce fichier au démarrage pour choisir
le `.pte` initial.
"""
from __future__ import annotations

import logging
from pathlib import Path

from flask import Blueprint, jsonify, request

import mobile_settings
from harmonia import cache
from harmonia.settings import SETTINGS

log = logging.getLogger("harmonia.mobile.beatmodel")

bp = Blueprint("mobile_beatmodel", __name__)

#: nom affiché → fichier `.pte` dans `filesDir/harmonia_models` (copiés là
#: par `MainActivity.seedModelsOnce` depuis `assets/harmonia_models/`, les
#: trois fichiers déjà exportés par T1c et le levier 4 de t5prep). Même
#: trois noms que les rapports Mac, pour que Louis compare les mêmes choses.
#: Même dictionnaire que `mobile_settings.FILES` (dupliqué exprès, voir la
#: docstring de ce dernier).
_FILES = mobile_settings.FILES

#: Le nom actuellement chargé — PAS la vérité (BeatsRuntime.kt la détient),
#: juste ce que CE process a demandé, pour que GET réponde sans redemander
#: au runtime natif. Initialisé depuis le réglage persisté (et non plus
#: toujours "final0") : après le premier redémarrage qui a suivi un choix
#: dans Réglages, `mobile_entry.py` a déjà chargé CE modèle avant que ce
#: blueprint n'existe — sans ça, GET aurait menti sur ce qui tourne
#: vraiment jusqu'au premier appel à set_model().
_current = mobile_settings.read_default_beat_model(SETTINGS.human_dir) or "final0"


@bp.get("/api/settings/beat_model")
def get_model():
    """Le modèle actuellement chargé + les noms disponibles SUR CET
    APPAREIL (un nom listé dans `_FILES` dont le fichier n'a pas été copié
    serait un piège silencieux — on le dit explicitement plutôt que de
    laisser `POST /api/settings/beat_model` échouer sans qu'on sache
    pourquoi)."""
    from mobile.harmonia_mobile import backends_mobile as bm
    models_dir = Path(bm._models_dir)
    present = {name: (models_dir / fname).exists()
              for name, fname in _FILES.items()}
    return jsonify({"current": _current, "options": sorted(_FILES),
                    "present_on_device": present,
                    "models_dir": str(models_dir)})


@bp.post("/api/settings/beat_model")
def set_model():
    """Bascule `BeatsRuntime` sur `{"name": "final0"|"small0"|"int8"}`,
    PERSISTE ce choix comme le défaut de démarrage (`mobile_settings.
    write_default_beat_model`), et vide le cache de battues du téléphone
    pour forcer un recalcul.

    Ne lance PAS d'analyse elle-même : l'appelant (le sélecteur d'un chart
    ouvert, l'écran Réglages, ou Louis directement en requête HTTP) relance
    `/api/analyze` ensuite. Séparé exprès — un switch peut servir à préparer
    plusieurs morceaux avant de les écouter d'affilée, pas seulement le
    dernier analysé.
    """
    global _current
    name = (request.get_json(silent=True) or {}).get("name", "")
    if name not in _FILES:
        return jsonify({"error": f"name inconnu {name!r}, attendu un de "
                                 f"{sorted(_FILES)}"}), 400

    from mobile.harmonia_mobile import backends_mobile as bm
    models_dir = Path(bm._models_dir)
    pte_path = models_dir / _FILES[name]
    if not pte_path.exists():
        return jsonify({"error": f"modèle absent sur l'appareil : {pte_path} "
                                 f"— seedModelsOnce n'a peut-être pas encore "
                                 f"tourné, ou l'APK est antérieur à ce "
                                 f"modèle"}), 404

    try:
        import mobile_beats_runtime
        mobile_beats_runtime.install(str(pte_path), threads=0)
    except Exception as exc:                        # noqa: BLE001
        log.exception("beat_model: échec du chargement de %s", pte_path)
        return jsonify({"error": f"chargement du runtime natif échoué : "
                                 f"{exc}"}), 500

    bm._pte_path = pte_path  # cohérence des rapports de debug seulement
    _current = name
    mobile_settings.write_default_beat_model(SETTINGS.human_dir, name)
    n_cleared = _clear_beats_cache()
    log.info("beat_model: bascule sur %s (%s), cache battues vidé "
             "(%d fichier(s))", name, pte_path, n_cleared)
    return jsonify({"ok": True, "current": name, "pte_path": str(pte_path),
                    "cleared_cache_files": n_cleared})


def _clear_beats_cache() -> int:
    """Supprime le cache de battues DU TÉLÉPHONE (`state/cache/beats/*`,
    jamais le cache partagé du Mac — cette fonction tourne dans le process
    Android, sous `filesDir`). Rend le nombre de fichiers effacés."""
    d = cache.folder("beats")
    n = 0
    if d.exists():
        for f in d.glob("*.json"):
            try:
                f.unlink()
                n += 1
            except OSError as exc:
                log.warning("beat_model: échec suppression %s (%s)", f, exc)
    return n
