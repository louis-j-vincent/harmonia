"""P2c — rejoue `harmonia.pipeline.analyze_steps` sur les 3 morceaux de seed,
une fois par chemin (Mac de référence / mobile), et compare les deux charts
mesure par mesure.

Deux sous-commandes, parce que les deux chemins vivent dans deux venvs :

    .venv/bin/python mobile/tools/parity_pipeline.py run --backend reference
    ~/harmonia/.venv-mobile/bin/python mobile/tools/parity_pipeline.py run --backend mobile
    .venv/bin/python mobile/tools/parity_pipeline.py compare

Ce que le script garantit et qui n'est PAS cosmétique :

* **caches séparés et FROIDS par chemin.** Les quatre caches par morceau
  (battues, `musx_probs`, `musx_cqt`, `nnls`) sont repointés sur
  `parity_out/<backend>/cache/` via `harmonia.cache.KINDS` — sans ça le
  chemin mobile lirait les postérieures que le chemin torch vient d'écrire et
  la parité serait une tautologie.
* **`head_s=0`.** Le chart de tête est un APERÇU (`pipeline._head_chart`,
  « CE QU'IL N'EST PAS : la vérité ») rendu sur un yield séparé ; il n'entre
  pas dans le chart final. On l'éteint pour ne pas payer trois analyses de
  45 s de plus, et on le dit.
* **`bar1_time=None`** : on veut exercer `harmonia.debut` (le seul autre
  consommateur d'audio décodé), pas court-circuiter par une marque.
* **1 thread** partout (`OMP/OPENBLAS/MKL/VECLIB_NUM_THREADS=1` posés par
  l'appelant, `torch.set_num_threads(1)` si torch est là).

La sortie par morceau : `<backend>/<stem>.json` (le ChartModel final, tel
quel) et `<backend>/<stem>.times.json` (le temps par étape).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AUDIO_DIR = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia/docs/audio")
OUT_ROOT = REPO / "mobile" / "tools" / "parity_out"

SONGS = ["maroon_5_this_love",
         "ben_e_king_stand_by_me_audio",
         "maroon_5_she_will_be_loved_official_music_video"]

#: Les quatre paquets qui n'ont PAS de roue Android (PLAN.md, « interdit sur le
#: téléphone »). La preuve que le chemin mobile tournera sous Chaquopy, c'est
#: qu'aucun des quatre n'est dans `sys.modules` à la fin.
FORBIDDEN = ("torch", "librosa", "vamp", "scipy")


def _setup_env(backend: str) -> Path:
    out = OUT_ROOT / backend
    (out / "cache").mkdir(parents=True, exist_ok=True)
    os.environ["HARMONIA_AUDIO_DIR"] = str(AUDIO_DIR)
    os.environ["HARMONIA_STATE_DIR"] = str(out / "state")
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    sys.path.insert(0, str(REPO))
    return out


def _isolate_caches(cache_dir: Path) -> None:
    """Repointe les 4 caches par morceau sur un dossier à nous (froid)."""
    from harmonia import cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    for kind, (_folder, legacy, ext) in list(cache.KINDS.items()):
        d = cache_dir / kind
        d.mkdir(parents=True, exist_ok=True)
        cache.KINDS[kind] = ((lambda _s, _d=d: _d), legacy, ext)


class _NoTimings:
    """Repli quand `harmonia.backends` n'existe pas encore — sert à produire
    les charts de RÉFÉRENCE avec le pipeline d'avant les coutures."""
    timings: dict = {}

    def reset_timings(self):
        pass


def run(backend: str, out_name: str | None = None) -> None:
    out = _setup_env(out_name or backend)
    _isolate_caches(out / "cache")

    if backend == "mobile":
        from mobile.harmonia_mobile import backends_mobile
        backends_mobile.install()

    from harmonia import pipeline
    try:
        from harmonia.backends import BACKENDS
    except ImportError:
        BACKENDS = _NoTimings()

    for stem in SONGS:
        audio = AUDIO_DIR / f"{stem}.m4a"
        BACKENDS.reset_timings()
        t0 = time.perf_counter()
        gen = pipeline.analyze_steps(audio, title=stem, file_key=stem,
                                     audio_url=f"/audio/{stem}.m4a",
                                     head_s=0.0, bar1_time=None)
        raw = final = None
        for kind, model in gen:
            if kind == "raw":
                raw = json.dumps(model, sort_keys=True)   # sérialisé tout de suite
            elif kind == "final":
                final = model
        total = time.perf_counter() - t0
        assert final is not None, f"pas de chart final pour {stem}"
        (out / f"{stem}.json").write_text(
            json.dumps(final, indent=1, sort_keys=True, ensure_ascii=False))
        (out / f"{stem}.raw.json").write_text(raw or "")
        times = dict(BACKENDS.timings)
        times["total"] = round(total, 3)
        times["reste"] = round(total - sum(v for k, v in times.items()
                                           if k != "total"), 3)
        (out / f"{stem}.times.json").write_text(json.dumps(times, indent=1))
        print(f"[{out_name or backend}] {stem}: {total:.1f}s  {times}", flush=True)

    leaked = sorted(m for m in FORBIDDEN if m in sys.modules)
    print(f"[{out_name or backend}] sys.modules interdits présents : {leaked or 'AUCUN'}")
    (out / "_sysmodules.json").write_text(json.dumps(
        {"forbidden_present": leaked,
         "n_modules": len(sys.modules)}, indent=1))


# ── comparaison ─────────────────────────────────────────────────────────────

def _bars(model: dict) -> list[list]:
    """Les mesures, à plat. Une section porte `bars` = liste de MESURES, et
    une mesure est elle-même une LISTE d'accords (une mesure peut en porter
    plusieurs : le quart de mesure est ouvert, `SETTINGS.quarter_bar`)."""
    out = []
    for sec in model.get("sections") or []:
        out.extend(sec.get("bars") or [])
    return out


#: Ce qui FAIT l'accord écrit sur la mesure — la note fondamentale (`root`,
#: 0=C), la qualité (`q`), le fait qu'il n'y ait pas d'accord (`nc`), la
#: position dans la mesure (`beat`) et la basse SONNANTE (`bass`, en classes
#: de hauteur, -1 = pas de basse). Le reste du dict (postérieures `c`,
#: cascade `casc`, candidats `sug`, couleur) est du diagnostic : on le compare
#: à part, il ne change pas ce que Louis LIT sur le chart.
_CH = ("root", "q", "nc", "beat", "bass")


def _chords_of(bar) -> tuple:
    return tuple(tuple(c.get(k) for k in _CH) for c in (bar or []))


def _chords_label(bar) -> list:
    return [("N" if c.get("nc") else
             f"{'CC#DD#EFF#GG#AA#B'[0:0] or ''}"
             f"{_PC[c.get('root') % 12] if c.get('root') is not None else '?'}"
             f"{c.get('q') or ''}" + (f"/{_PC[c['bass'] % 12]}"
                                      if c.get("bass", -1) >= 0 else ""))
            for c in (bar or [])]


_PC = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def compare(a_dir: Path, b_dir: Path) -> dict:
    rows = {}
    for stem in SONGS:
        fa, fb = a_dir / f"{stem}.json", b_dir / f"{stem}.json"
        if not (fa.exists() and fb.exists()):
            rows[stem] = {"erreur": "chart manquant"}
            continue
        ta, tb = fa.read_text(), fb.read_text()
        A, B = json.loads(ta), json.loads(tb)
        ba, bb = _bars(A), _bars(B)
        r = {"identique_octet": ta == tb,
             "n_mesures": (A.get("nBars"), B.get("nBars")),
             "n_mesures_ok": A.get("nBars") == B.get("nBars"),
             "n_battues": (len(A.get("beatTimes") or []),
                           len(B.get("beatTimes") or [])),
             "tonalite": (A.get("key"), A.get("keyName"),
                          B.get("key"), B.get("keyName")),
             "tonalite_ok": (A.get("key") == B.get("key")
                             and A.get("keyName") == B.get("keyName")),
             "sections": ([(s.get("label"), s.get("id"), len(s.get("bars") or []))
                           for s in (A.get("sections") or [])],
                          [(s.get("label"), s.get("id"), len(s.get("bars") or []))
                           for s in (B.get("sections") or [])]),
             "bpm": (A.get("meta", {}).get("bpm"), B.get("meta", {}).get("bpm")),
             }
        r["sections_ok"] = r["sections"][0] == r["sections"][1]
        pa, pb = A.get("beatTimes") or [], B.get("beatTimes") or []
        if len(pa) == len(pb) and pa:
            r["max_dt_battues_ms"] = round(
                1000 * max(abs(x - y) for x, y in zip(pa, pb)), 3)
        else:
            r["max_dt_battues_ms"] = None
        # accords mesure par mesure
        diffs = []
        n = min(len(ba), len(bb))
        for i in range(n):
            ca, cb = _chords_of(ba[i]), _chords_of(bb[i])
            if ca != cb:
                diffs.append({"mesure": i + 1,
                              "mac": _chords_label(ba[i]),
                              "mobile": _chords_label(bb[i])})
        r["n_mesures_comparees"] = n
        r["pct_mesures_identiques"] = (round(100.0 * (n - len(diffs)) / n, 2)
                                       if n else None)
        r["mesures_differentes"] = diffs[:40]
        r["n_mesures_differentes"] = len(diffs)
        # basse sonnante : le champ `bass` de chaque accord, s'il existe
        def _bass(bars):
            return [c.get("bass") for bar in bars for c in (bar or [])]
        Ba, Bb = _bass(ba), _bass(bb)
        r["basse_ok"] = Ba == Bb
        # Ce qui reste différent OCTET POUR OCTET, champ par champ : sert à
        # dire si un écart est musical (ce que Louis lit) ou seulement
        # numérique (postérieures affichées à 4 décimales).
        r["champs_differents"] = sorted(k for k in set(A) | set(B)
                                        if json.dumps(A.get(k), sort_keys=True)
                                        != json.dumps(B.get(k), sort_keys=True))
        r["n_basses"] = (len(Ba), len(Bb))
        rows[stem] = r
    return rows


# ── charts produits SUR LE DEVICE (P2d, 2026-09-23) ─────────────────────────

def pull(device_dir: str = "device", songs: list[str] | None = None) -> None:
    """Rapatrie les charts que l'ANDROID vient d'écrire, dans le même format
    de dossier que `run`, pour que `compare` les traite comme n'importe quel
    autre chemin.

    On lit le FICHIER, pas `/api/chart-model/<f>` : le fichier est exactement
    ce que `server/jobs.py` a écrit (`json.dumps(model)`), alors que la route
    reconstruit une réponse. Comparer des octets à des octets évite d'ajouter
    une couche entre le calcul et la mesure.

    `run-as` marche parce que l'APK est en debug ; sur un APK signé release il
    faudrait passer par la route. C'est noté dans le rapport, pas caché ici.
    """
    import subprocess
    out = OUT_ROOT / device_dir
    out.mkdir(parents=True, exist_ok=True)
    for stem in (songs or SONGS):
        remote = f"files/harmonia_state/cache/charts/min_{stem}.json"
        r = subprocess.run(
            ["adb", "exec-out", "run-as", "fr.harmonia.app", "cat", remote],
            capture_output=True)
        if r.returncode != 0 or not r.stdout:
            print(f"{stem}: ABSENT sur le device "
                  f"({r.stderr.decode(errors='replace').strip()[:120]})")
            continue
        (out / f"{stem}.json").write_bytes(r.stdout)
        print(f"{stem}: {len(r.stdout)} octets -> {out / f'{stem}.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--backend", choices=["reference", "mobile"], required=True)
    r.add_argument("--out-name", default=None)
    pl = sub.add_parser("pull", help="rapatrie les charts du device Android")
    pl.add_argument("--out", default="device")
    pl.add_argument("--songs", nargs="*", default=None)
    c = sub.add_parser("compare")
    c.add_argument("--a", default="reference")
    c.add_argument("--b", default="mobile")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.backend, a.out_name)
    elif a.cmd == "pull":
        pull(a.out, a.songs)
    else:
        rows = compare(OUT_ROOT / a.a, OUT_ROOT / a.b)
        print(json.dumps(rows, indent=1, ensure_ascii=False))
        (OUT_ROOT / f"compare_{a.a}_{a.b}.json").write_text(
            json.dumps(rows, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
