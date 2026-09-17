"""Le registre des tâches d'analyse : `_jobs`, `_run_job`, `_resolve_audio`.

Une analyse (YouTube, fichier local, ou micro) tourne dans un thread daemon et
publie son avancement dans `_jobs[job_id]`, que `/api/job/<id>` sert en
polling. Deux jalons plutôt qu'un (voir `_run_job`) : le chart BRUT dès qu'il
existe, puis `status="done"` quand le raffinement (sections, gammes) a
réécrit le même fichier.

Porté verbatim depuis `harmonia_min/server.py` (sprints 11-14), avec un ajout
qui manquait : `_JOBS_LOCK` autour de chaque écriture de `_jobs[job_id]` — le
dict n'était protégé par rien, alors que le thread du job et les requêtes
`GET /api/job/<id>` s'y croisent en continu.

Ce que ce module ne fait PAS : parler à yt-dlp (voir `youtube.py`, que
`_resolve_audio` appelle) ; savoir ce que le client doit afficher pour chaque
`stage`/`phase` (c'est `app_shell.html`).
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path

from harmonia.pipeline import analyze_steps
from harmonia.server import youtube
from harmonia.settings import SETTINGS

log = logging.getLogger("harmonia.server.jobs")

AUDIO_DIR = SETTINGS.audio_dir
CHARTS_DIR = SETTINGS.charts_dir
META_PATH = SETTINGS.chart_meta_path

_jobs: dict[str, dict] = {}
#: Le dict `_jobs` n'était protégé par rien : le thread du job et les
#: requêtes `GET /api/job/<id>` (et `POST /api/analyze` qui insère une
#: nouvelle entrée) s'y croisent en continu. Un verrou autour de chaque
#: lecture/écriture, pas seulement autour de la création — sinon la course
#: reste possible sur `job.update(...)` pendant qu'une requête lit `job`.
_JOBS_LOCK = threading.Lock()


def _load_chart_meta() -> dict:
    try:
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def bar1_for(file_key: str) -> float | None:
    """La marque « Set bar 1 » de Louis pour ce morceau, ou None.

    SEULE source à l'exécution (sprint 15, décision 10) : le fichier
    `state/human/marks/<stem>.json` (`{"bar1": <secondes>}`), écrit par
    `/api/bar1/<file>`. Avant ce sprint la marque ne vivait que dans le champ
    `bar1` du chart régénérable — perdue le 2026-08-13 dès que le chart était
    réécrit sans elle. Cette fonction ne lit JAMAIS ce champ : le repli sur
    le vieux chart n'existe que dans `tools/migrate_state.py`, une fois, à la
    migration — pas ici, sinon la même perte redeviendrait possible dès
    qu'un chart serait réécrit sans passer par cette fonction.

    `file_key` est celui du chart (`min_<stem>`) ; la marque, elle, est posée
    par AUDIO, donc sans le préfixe `min_`.
    """
    stem = file_key.removeprefix("min_")
    p = SETTINGS.marks_dir / f"{stem}.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d.get("bar1")


def tempo_factor_for(file_key: str) -> float | None:
    """La correction ÷2/×2 de Louis pour ce morceau, ou None.

    Même fichier-marque que `bar1_for` (`state/human/marks/<stem>.json`) — un
    seul sidecar de corrections humaines par morceau, pas un fichier par type
    de correction. Voir `harmonia.beats.apply_tempo_octave` pour ce que fait
    la valeur ; ce que ce garde ne résout pas (`check_grid` ne peut pas voir
    cette erreur, elle est invariante au tempo) est documenté là-bas.
    """
    stem = file_key.removeprefix("min_")
    p = SETTINGS.marks_dir / f"{stem}.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d.get("tempo_factor")


def _resolve_audio(url: str) -> tuple[Path, str, str]:
    """analyze URL → (audio path, title, artist). Three forms:
    'local:<stem>' (from our own search results, possibly wrapped in a
    youtube.com/watch?v= prefix by the untouched UI), a local stem typed
    directly, or a real YouTube URL (yt-dlp, if installed).

    L'artiste peut être vide : un slug de fichier a perdu le tiret qui séparait
    l'artiste du titre, et inventer la coupe serait pire que le champ vide (voir
    `harmonia.titles`). `scripts/backfill_titles.py` rattrape ces cas-là en
    retrouvant la vidéo d'origine.
    """
    from harmonia import titles as _titles

    m = re.search(r"local:([\w\-]+)", url)
    if m:
        p = AUDIO_DIR / f"{m.group(1)}.m4a"
        if p.exists():
            return p, _titles.pretty_from_slug(p.stem), ""
        raise FileNotFoundError(f"no local audio {m.group(1)}")
    stem = url.strip().strip("/")
    p = AUDIO_DIR / f"{Path(stem).stem}.m4a"
    if p.exists():
        return p, _titles.pretty_from_slug(p.stem), ""
    if re.search(r"youtu\.?be", url):
        ytdlp = youtube._ytdlp_bin()
        if not ytdlp:
            raise RuntimeError("yt-dlp not installed — paste a library song "
                               "name or install yt-dlp for YouTube links")
        vid = re.search(r"(?:v=|youtu\.be/)([\w\-]{6,})", url)
        out = AUDIO_DIR / (f"{vid.group(1)}.m4a" if vid else "download.m4a")
        if not out.exists():
            # `ytdlp`, pas "yt-dlp" : le chemin résolu juste au-dessus n'était
            # pas utilisé ici, ce qui annulait la raison d'être de _ytdlp_bin.
            artist, title = youtube._download_audio(ytdlp, url, out)
        else:
            # Fichier déjà sur le disque : personne n'a imprimé la ligne, il
            # faut donc bien l'aller-retour — mais on ne le paie plus sur le
            # cas qui compte, celui du morceau neuf.
            artist, title = youtube._video_meta(ytdlp, url)
        return out, (title or _titles.pretty_from_slug(out.stem)), artist
    raise FileNotFoundError(f"could not resolve {url!r} to audio")


def _run_job(job_id: str, url: str, bar1_time=None, tempo_factor=None):
    """Le chart BRUT est publié dès qu'il existe ; le raffinement continue après.

    Louis, 2026-08-07 : « dès que le chart brut est dispo tu l'affiches direct,
    et le reste (gammes, harmonies, sections) tu le fais en background ».

    Le job expose donc deux jalons plutôt qu'un :

        chart_url  écrit au premier rendu, `status` restant "running" ;
                   l'app bascule dessus et laisse le sondage tourner.
        status     passe à "done" quand le modèle raffiné a ÉCRASÉ le même
                   fichier — même `file_key`, donc rien ne se dédouble.

    `refining` liste ce qui manque encore (l'app l'affiche discrètement dans le
    chart) et `refined_at` change de valeur exactement une fois, ce qui donne à
    l'app un front sur lequel recharger.

    Les deux écritures sont sérialisées ici, dans le thread du job, entre deux
    reprises du générateur : `bars` est muté sur place par le repli et par
    l'analyse harmonique, donc le brut DOIT être en JSON avant que la suite ne
    tourne (voir pipeline.analyze_steps).

    La durée ffprobe calculée ici pour le job est transmise à `analyze_steps`
    (`duration_s`), qui ne la recalcule donc pas.
    """
    job = _jobs[job_id]

    def progress(stage, **kw):
        with _JOBS_LOCK:
            job["stage"] = max(job.get("stage", 0), stage)
            # L'ÉCRAN NE RECULE PAS (2026-08-18). Le chart de tête est rendu
            # avant que la passe complète ne commence, donc le `report(1,
            # phase="listening")` qui suit remettait l'écran sur le rond qui
            # tourne — et l'aperçu, déjà calculé, n'était jamais montré. La
            # phase reste "head" jusqu'au chart brut ; `stage` continue
            # d'avancer.
            if job.get("phase") == "head" and kw.get("phase") in ("listening",
                                                                  "decoding"):
                kw = {k: v for k, v in kw.items() if k != "phase"}
            job.update(kw)

    try:
        audio_path, title, artist = _resolve_audio(url)
        with _JOBS_LOCK:
            job["title"] = job.get("title") or title
            # Le titre qui GAGNE est celui du job (un enregistrement micro
            # arrive avec « Enregistrement du … » déjà posé) — sinon le
            # sidecar plus bas réécrivait la bibliothèque avec le slug du
            # fichier.
            title = job["title"]
            job["artist"] = artist
        dur = float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(audio_path)]).strip())
        with _JOBS_LOCK:
            job.update(stage=1, duration_s=dur)
        file_key = f"min_{audio_path.stem}"
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        dest = CHARTS_DIR / f"{file_key}.json"
        # Ré-analyser un morceau DÉJÀ dans la bibliothèque ne doit pas jeter
        # son « Set bar 1 » (2026-08-13). La marque est collante : on ne la
        # relit que si l'appel n'en apporte pas une nouvelle — et on la relit
        # de `state/human/marks/`, la seule source à l'exécution (voir
        # `bar1_for`), jamais du champ `bar1` de ce chart-ci.
        if bar1_time is None:
            bar1_time = bar1_for(file_key)
            if bar1_time is not None:
                log.info("job %s: reprise du Set bar 1 de %s (%.3fs)",
                         job_id, file_key, float(bar1_time))
        # Même logique collante pour la correction ÷2/×2 (2026-09-17) : une
        # ré-analyse d'un morceau déjà connu ne doit pas perdre le réglage
        # tempo de Louis, exactement comme pour bar1 juste au-dessus.
        if tempo_factor is None:
            tempo_factor = tempo_factor_for(file_key)
            if tempo_factor is not None:
                log.info("job %s: reprise du réglage tempo de %s (x%s)",
                         job_id, file_key, tempo_factor)
        # Le sidecar est ce que /api/library ressert : c'est là que l'artiste
        # doit atterrir pour être VU. On n'écrase jamais une saisie de Louis —
        # l'éditeur artiste/titre de l'app écrit dans le même fichier.
        if artist or title:
            _meta = _load_chart_meta()
            if not (_meta.get(file_key) or {}).get("title"):
                _meta[file_key] = {"artist": artist, "title": title}
                try:
                    META_PATH.parent.mkdir(parents=True, exist_ok=True)
                    META_PATH.write_text(
                        json.dumps(_meta, ensure_ascii=False, indent=1),
                        encoding="utf-8")
                except OSError as exc:
                    log.warning("chart-meta autosave failed: %s", exc)
        t0 = time.time()
        for kind, model in analyze_steps(
                audio_path, title=job["title"], file_key=file_key,
                audio_url=f"/audio/{audio_path.name}", progress=progress,
                bar1_time=bar1_time, tempo_factor=tempo_factor,
                duration_s=dur):
            if kind == "head":
                # LE CHART DE TÊTE NE TOUCHE PAS LE DISQUE (2026-08-18). C'est
                # un aperçu des ~45 premières secondes, dont 6 % des accords
                # changeront quand la passe complète arrivera : l'écrire dans
                # CHARTS_DIR le ferait apparaître dans la bibliothèque comme
                # un chart tronqué si le job mourait juste après. Il voyage
                # donc dans le job, rendu par le MÊME loadModel() que le brut,
                # et `chart_url` reste absent — donc pas de bouton « Play it
                # now » qui ouvrirait un fichier qui n'existe pas.
                with _JOBS_LOCK:
                    job.update(phase="head", raw_model=model,
                               n_bars=model["nBars"],
                               head_s=round(time.time() - t0, 2))
                log.info("job %s: chart de TÊTE en %.1f s → %d mesures",
                         job_id, time.time() - t0, model["nBars"])
                continue
            dest.write_text(json.dumps(model), encoding="utf-8")
            if kind == "raw":
                # UI refresh 2026-08-08 (§5): the raw ChartModel rides the job
                # itself — the loading screen renders it through the SAME
                # loadModel()/buildIReal() path the final chart uses, which is
                # what keeps the brut→final hand-off from moving a single
                # cell. phase="raw" therefore only ever appears with a
                # non-None raw_model (acceptance #1).
                n_chords = sum(1 for sec in model["sections"]
                               for bar in sec["bars"] for c in bar
                               if not c["nc"] and not c.get("carry"))
                with _JOBS_LOCK:
                    job.update(chart_url=f"/chart/{file_key}",
                               refining=list(model["meta"].get("pending") or []),
                               raw_s=round(time.time() - t0, 2),
                               phase="raw", raw_model=model,
                               n_bars=model["nBars"], n_chords=n_chords)
                log.info("job %s: CHART BRUT en %.1f s → %s (%d mesures) ; "
                         "raffinement en cours",
                         job_id, time.time() - t0, file_key, model["nBars"])
            else:
                with _JOBS_LOCK:
                    job.update(status="done", url=f"/chart/{file_key}",
                               chart_url=f"/chart/{file_key}", refining=[],
                               refined_at=round(time.time() - t0, 2), stage=6,
                               phase="done",
                               sections_found=len(model["sections"]))
                log.info("job %s done en %.1f s → %s (%d sections)",
                         job_id, time.time() - t0, file_key,
                         len(model["sections"]))
    except Exception as exc:
        log.exception("job %s failed", job_id)
        with _JOBS_LOCK:
            if job.get("chart_url"):
                # Le chart brut est DÉJÀ publié et l'app le montre peut-être
                # déjà : le passer en "error" effacerait un chart qui marche.
                # On dit la vérité — terminé, mais sans raffinement — et la
                # trace complète est dans le log ci-dessus (pas de repli
                # silencieux).
                job.update(status="done", url=job["chart_url"], refining=[],
                           refine_error=str(exc), stage=6,
                           phase="done", sections_found=1)
            else:
                job.update(status="error", error=str(exc))


def start_job(url: str, *, title: str = "", bar1_time=None,
              tempo_factor=None) -> str:
    """Crée un job, lance son thread, renvoie son id — l'usine commune à
    `/api/analyze`, `/api/bar1/<file>`, `/api/tempo/<file>` et
    `/api/record-analyze`."""
    job_id = uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        _jobs[job_id] = {"status": "running", "stage": 0, "created": time.time(),
                         "title": title}
    threading.Thread(target=_run_job, args=(job_id, url),
                     kwargs={"bar1_time": bar1_time,
                             "tempo_factor": tempo_factor}, daemon=True).start()
    return job_id


# Ce que ce module ne fait PAS : parler à yt-dlp directement (voir
# `youtube.py`) ; savoir ce que le client doit afficher pour chaque
# `stage`/`phase` (c'est `app_shell.html`) ; les routes elles-mêmes, qui
# vivent dans `routes/analyze.py`.
