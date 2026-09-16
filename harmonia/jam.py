"""harmonia/jam.py — Jam Mode sur la pile de l'app : Beat This! + musx.

Une session = un tampon micro qui grandit. À chaque morceau de ~8 s uploadé,
on redécode TOUT le tampon et on cherche la boucle : la période (en temps) qui
se répète, puis un vote par case sur les tours complets — c'est ce qui fait que
la grille se nettoie à mesure qu'on rejoue la boucle.

CE QUI CHANGE PAR RAPPORT À `harmonia/models/jam_mode.py` (2026-07-20, l'autre
lane) : la détection de boucle et le vote par case sont repris tels quels
(`detect_loop_period`, `LoopVotes` — logique pure, sur des étiquettes), mais le
DÉCODAGE n'est plus librosa + NNLS-24. Il est celui de l'app : Beat This! pour
les temps (librosa est banni ici — il verrouille l'octave 2x sur un tiers du
disque) et les postérieures musx re-décodées sur nos temps. Le même décodeur
que le chart, donc les accords d'un jam et ceux d'un chart ne peuvent pas
diverger.

POURQUOI ON REDÉCODE TOUT À CHAQUE FOIS, jamais en incrémental : chaque passe
de Beat This! est indépendante et peut déplacer légèrement sa phase quand
l'audio s'allonge. Persister des votes à travers ça les rangerait peu à peu
dans la mauvaise case (piège de calibration silencieuse, règle #1 du CLAUDE.md).
Le prix : ça devient plus lent quand le jam dure. Mesuré, et le serveur SAUTE
un morceau plutôt que d'empiler les passes (voir `server.api_jam_chunk`).

CE QUE ÇA NE RÉSOUT PAS : la vraie basse latence (on est à ~10 s de retard, par
choix — Louis, 2026-07-20 : « on a le droit d'avoir un delta de quelques
secondes »), et le retour à une partie DÉJÀ jouée : un jam à 3 parties
détectera « une nouvelle partie » contre la partie courante au lieu de
reconnaître qu'on est revenu sur la première.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

from harmonia import beats as _beats
from harmonia import musx as _musx
from harmonia.labels import to_chord

logger = logging.getLogger(__name__)

#: Un changement de partie demande un désaccord SOUTENU, pas un accord de
#: passage — repris de jam_mode (2026-07-20).
PART_CHANGE_MIN_MISMATCH_S = 6.0
PART_CHANGE_MISMATCH_RATE = 0.6

#: La boucle est cherchée sur la QUEUE de la partie courante, pas depuis son
#: début : le rodage du jam (ou l'intro avant que le vamp se pose) dilue
#: l'autocorrélation sous le seuil alors que les 20 dernières secondes bouclent
#: proprement. Ça borne aussi le coût.
RECENT_WINDOW_BEATS = 64

#: musx chdir dans son clone (`_InMusxDir`), ce qui est visible par TOUT le
#: processus : deux décodages en parallèle se marcheraient dessus.
DECODE_LOCK = threading.Lock()

_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _pretty(label: str) -> str:
    """Étiquette musx (« C:maj », « A:min7 ») → symbole lisible (« C », « A-7 »).

    Même vocabulaire de queues que le chart (`labels.to_chord`), pour qu'un
    accord vu en jam s'écrive exactement comme il s'écrira dans le chart.
    """
    ch = to_chord(label)
    if ch is None:
        return "N"
    sym = _SHARP[ch["root"]] + ch["q"]
    if ch["bass"] >= 0 and ch["bass"] != ch["root"]:
        sym += "/" + _SHARP[ch["bass"]]
    return sym


def detect_loop_period(beat_labels: list[str], min_p: int = 4, max_p: int = 32,
                       min_score: float = 0.65) -> "tuple[int, float] | None":
    """Plus PETITE période P (Occam) dont l'autocorrélation à décalage P sur
    l'IDENTITÉ des étiquettes dépasse `min_score`. None = on s'abstient.

    Le seuil est à 0,65 et pas 0,75 : mesuré le 2026-07-20 sur un vrai vamp à
    2 accords, audiblement propre, la vraie période ne marquait que 0,696 — un
    seul accord ambigu (dim vs min) d'un tour à l'autre suffisait à passer
    dessous, et l'algo se verrouillait alors sur 2× la vraie période (une
    boucle juste, mais inutilement doublée — la même ambiguïté d'octave que
    nos doublements de tempo).
    """
    n = len(beat_labels)
    labels = np.asarray(beat_labels, dtype=object)
    for p in range(min_p, min(max_p, n // 2) + 1):
        score = float(np.mean(labels[:-p] == labels[p:]))
        if score >= min_score:
            return p, score
    return None


class LoopVotes:
    """Vote majoritaire par case sur les tours complets — le « ça se nettoie à
    mesure qu'on le rejoue ». Reconstruit à neuf à chaque passe (voir la
    docstring du module pour pourquoi on ne l'accumule jamais)."""

    def __init__(self, period: int):
        self.period = period
        self.slots: list[Counter] = [Counter() for _ in range(period)]
        self.n_beats = 0

    def add(self, beat_labels: list[str]) -> None:
        for bi, lbl in enumerate(beat_labels):
            self.slots[bi % self.period][lbl] += 1
        self.n_beats = len(beat_labels)

    def best(self) -> list[dict]:
        out = []
        for slot in self.slots:
            if not slot:
                out.append({"label": "N", "raw": "N", "confidence": 0.0})
                continue
            lbl, cnt = slot.most_common(1)[0]
            out.append({"label": _pretty(lbl), "raw": lbl,
                        "confidence": round(cnt / sum(slot.values()), 3)})
        return out

    def n_reps(self) -> int:
        return self.n_beats // self.period if self.period else 0


def decode_buffer(wav_path: Path) -> "dict | None":
    """Le tampon → {tempo_bpm, bt, beat_labels} avec la pile de l'app.

    None quand il n'y a pas encore de quoi conclure (trop court, trop calme
    pour que Beat This! rende une grille) — l'appelant traite ça comme
    « rien à montrer », pas comme une erreur.

    `use_cache=False` PARTOUT et c'est structurel : battues et postérieures
    sont cachées par STEM de fichier, et ici le même chemin porte un tampon
    qui grandit à chaque passe. Un cache chaud resservirait éternellement la
    première passe (le piège exact décrit dans la docstring de `jam_mode`).
    """
    with DECODE_LOCK:
        try:
            bd = _beats.track(wav_path, use_cache=False)
        except _beats.BeatTrackingError as exc:
            logger.info("jam: pas encore de grille (%s)", exc)
            return None
        beat_times, downbeats = bd["beats"], bd["downbeats"]
        if len(beat_times) < 8:
            return None
        probs = _musx.frame_posteriors(wav_path, use_cache=False)
        bpb = 4
        if len(downbeats) >= 3:
            _b = int(round(float(np.median(np.diff(downbeats)))
                           / float(np.median(np.diff(beat_times)))))
            bpb = _b if 2 <= _b <= 7 else 4
        segments, _lat = _musx.redecode(beat_times, probs,
                                        downbeat_times=downbeats,
                                        beats_per_bar=bpb,
                                        quarter_beats="all")
    if not segments:
        return None

    # Une étiquette PAR TEMPS : c'est l'unité sur laquelle la boucle se
    # cherche, un jam n'ayant pas de longueur de mesure connue d'avance.
    bt = np.asarray(beat_times, dtype=float)
    beat_labels: list[str] = []
    si = 0
    for t in bt[:-1]:
        while si < len(segments) - 1 and t >= segments[si][1]:
            si += 1
        beat_labels.append(segments[si][2])
    return {"tempo_bpm": bd["bpm"], "period_s": 60.0 / max(bd["bpm"], 1.0),
            "bt": bt, "beat_labels": beat_labels}


class JamSession:
    """Un jam : un tampon mono qui grandit + la boucle courante, découpée en
    « parties » figées dès qu'un motif neuf tient assez longtemps."""

    def __init__(self, sr: int = 44100):
        self.sr = sr
        self.buf = np.zeros(0, dtype=np.float32)
        self.created = time.time()
        self.parts: list[dict] = []
        self.part_boundary_s = 0.0
        self.cur: "dict | None" = None
        self.busy = False          # une passe de décodage est en cours

    def append(self, chunk: np.ndarray) -> None:
        self.buf = np.concatenate([self.buf, chunk.astype(np.float32)])

    def elapsed_s(self) -> float:
        return len(self.buf) / self.sr

    def update(self, tmp_wav: Path) -> dict:
        import soundfile as sf
        sf.write(tmp_wav, self.buf, self.sr)
        decoded = decode_buffer(tmp_wav)
        if decoded is None:
            return self._abstain()

        bt, beat_labels = decoded["bt"], decoded["beat_labels"]
        part_start_idx = int(np.searchsorted(bt[:-1], self.part_boundary_s))
        start_idx = max(part_start_idx, len(beat_labels) - RECENT_WINDOW_BEATS)
        cur_labels = beat_labels[start_idx:]

        period_info = detect_loop_period(cur_labels)
        if period_info is None:
            return self._abstain()

        period, _score = period_info
        votes = LoopVotes(period)
        votes.add(cur_labels)

        new_part_at: "float | None" = None
        if votes.n_reps() >= 2:
            window_beats = max(period, int(round(PART_CHANGE_MIN_MISMATCH_S
                                                 / decoded["period_s"])))
            tail_len = min(window_beats, len(cur_labels))
            tail_off = len(cur_labels) - tail_len
            best = votes.best()
            mismatches = sum(
                1 for i in range(tail_len)
                if cur_labels[tail_off + i] != best[(tail_off + i) % period]["raw"]
            )
            if tail_len and mismatches / tail_len >= PART_CHANGE_MISMATCH_RATE:
                new_part_at = float(bt[start_idx + tail_off])

        if new_part_at is not None:
            self.parts.append({
                "loop": votes.best(), "period_beats": period,
                "tempo_bpm": round(decoded["tempo_bpm"], 1),
                "n_reps": votes.n_reps(),
            })
            self.part_boundary_s = new_part_at
            self.cur = None
        else:
            self.cur = {"loop": votes.best(), "period_beats": period,
                        "n_reps": votes.n_reps(), "stale": False,
                        "tempo_bpm": round(decoded["tempo_bpm"], 1)}
        return self.state()

    def _abstain(self) -> dict:
        """Une passe qui ne retrouve pas la boucle NE VIDE PAS l'écran.

        Mesuré le 2026-08-20 sur un vamp de 2 mesures rejoué 8 fois : la
        boucle est trouvée à 32 s, perdue à 40 s, retrouvée à 48 s. Le score
        d'autocorrélation frôle le seuil (0,65) et un seul accord ambigu d'un
        tour à l'autre le fait passer dessous. Effacer les accords à ce
        moment-là donne un écran qui clignote alors que le jam, lui, n'a pas
        bougé — on garde donc la dernière boucle en la marquant `stale`, et
        l'app la montre en retrait au lieu de la remplacer par du vide.

        CE QUE ÇA NE RÉSOUT PAS : le clignotement lui-même. La boucle affichée
        pendant une abstention peut être PÉRIMÉE (le jam a vraiment changé et
        la détection n'a pas encore de quoi trancher). C'est un choix
        d'affichage, pas une détection plus stable.
        """
        if self.cur is not None:
            self.cur = {**self.cur, "stale": True}
        return self.state()

    def state(self) -> dict:
        return {"elapsed_s": round(self.elapsed_s(), 1),
                "parts": self.parts, "current": self.cur}
