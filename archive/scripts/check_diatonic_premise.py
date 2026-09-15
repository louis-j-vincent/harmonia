"""
Premise check pour issue #20 : quelle fraction des GT chords sont diatoniques
dans la clé du morceau ? Corpus : jazz1460 songs index >=70 (held-out, non vus
par beat_seq_model_v4).

Mesure le PLAFOND de gain d'un prior diatonique parfait : si la clé de section
était connue exactement, quelle fraction des accords GT la règle diatonique
(degre -> qualite attendue) couvrirait-elle ? Si ce plafond est bas, un prior
diatonique ne peut pas corriger grand-chose et il ne faut pas l'implementer.

La clé est estimée deux façons, croisées pour robustesse :
  (a) infer_key() sur un chroma symbolique pondéré par durée, construit à partir
      des notes d'accord GT (root + intervalles de la qualité 5-classes) — c'est
      la méthode demandée dans le spec de l'issue #20 (chroma -> infer_key).
  (b) le champ `key` annoté iReal du record (haute confiance) — sanity cross-check.

Chaque accord GT (root_pc, q5) est diatonique si son degré (root_pc - tonic) est
dans la gamme ET sa qualité correspond au degré. maj7/min7 tolérés comme variants
de maj/min (déjà fusionnés dans les 5 classes major/minor/dom7/maj7/dim).

Seuil de décision (spec nightly) : >= 60% diatonique -> implémenter, sinon STOP.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import json

from analyze_accomp_emission import song_chord_spans
from train_beat_seq_model_v3 import quality5, NOTE, HARTE_TO_PC
from harmonia.theory.key_profiles import infer_key

DB = REPO / "data" / "accomp_db" / "db.jsonl"

# q5 index -> chord-tone semitone offsets (relative to root) for symbolic chroma.
Q5_TONES = {
    0: (0, 4, 7),       # major
    1: (0, 3, 7),       # minor
    2: (0, 4, 7, 10),   # dom7
    3: (0, 4, 7, 11),   # maj7
    4: (0, 3, 6),       # dim
}

# --- diatonic tables: (degree semitone offset from tonic) -> allowed q5 names ---
Q5_NAMES = ["major", "minor", "dom7", "maj7", "dim"]

DIATONIC_MAJOR = {
    0:  {"major", "maj7"},   # I
    2:  {"minor"},           # ii
    4:  {"minor"},           # iii
    5:  {"major", "maj7"},   # IV
    7:  {"dom7", "major"},   # V (triad or dom7)
    9:  {"minor"},           # vi
    11: {"dim"},             # vii°
}
DIATONIC_MINOR = {
    0:  {"minor"},                    # i
    2:  {"dim"},                      # ii°
    3:  {"major", "maj7"},            # bIII
    5:  {"minor"},                    # iv
    7:  {"minor", "dom7", "major"},   # v (natural) / V (harmonic)
    8:  {"major", "maj7"},            # bVI
    10: {"dom7", "major"},            # bVII
    11: {"dim"},                      # vii° (harmonic-minor leading tone)
}


def is_diatonic(root_pc: int, q5_idx: int, tonic: int, mode: str) -> bool:
    deg = (root_pc - tonic) % 12
    table = DIATONIC_MAJOR if mode == "major" else DIATONIC_MINOR
    allowed = table.get(deg)
    if allowed is None:
        return False
    return Q5_NAMES[q5_idx] in allowed


def symbolic_chroma(spans) -> np.ndarray:
    """Duration-weighted pitch-class energy from GT chord tones (12,), raw."""
    v = np.zeros(12)
    for t0, t1, root, q in spans:
        q5 = quality5(q)
        if q5 is None:
            continue
        dur = t1 - t0
        for off in Q5_TONES[q5]:
            v[(root + off) % 12] += dur
    return v


def key_name_to_tonic(key_str: str) -> int | None:
    """iReal key annotation like 'Eb', 'F#', 'A-' (minor) -> tonic pc + mode."""
    s = key_str.strip()
    mode = "major"
    if s.endswith("-"):
        mode = "minor"
        s = s[:-1]
    tonic = HARTE_TO_PC.get(s)
    return (tonic, mode) if tonic is not None else None


def main() -> None:
    start, n = 70, 25
    recs = [json.loads(l) for l in open(DB)]
    jz = [r for r in recs if r.get("corpus") == "jazz1460"
          and r["beats_per_bar"] == 4 and (REPO / r["midi_path"]).exists()]
    held = jz[start:start + n]
    print(f"held-out jazz1460 songs: {len(held)} (index {start}..{start + n})\n")

    # per-song accumulators (weighted by duration)
    dia_infer_n = tot_n = 0
    dia_infer_dur = tot_dur = 0.0
    dia_annot_n = 0
    key_agree = key_total = 0
    per_song = []

    for rec in held:
        spans = [(t0, t1, r % 12, q) for t0, t1, r, q in song_chord_spans(rec)
                 if t1 > t0 and quality5(q) is not None]
        if not spans:
            continue
        ch = symbolic_chroma(spans)
        kp = infer_key(ch)
        it, im = kp.tonic, kp.mode

        annot = key_name_to_tonic(rec.get("key", ""))
        if annot is not None:
            key_total += 1
            if annot == (it, im):
                key_agree += 1

        s_dia_i = s_dia_a = s_n = 0
        s_dia_i_dur = s_dur = 0.0
        for t0, t1, root, q in spans:
            q5 = quality5(q)
            dur = t1 - t0
            s_n += 1
            s_dur += dur
            if is_diatonic(root, q5, it, im):
                s_dia_i += 1
                s_dia_i_dur += dur
            if annot is not None and is_diatonic(root, q5, annot[0], annot[1]):
                s_dia_a += 1

        dia_infer_n += s_dia_i; tot_n += s_n
        dia_infer_dur += s_dia_i_dur; tot_dur += s_dur
        dia_annot_n += s_dia_a
        per_song.append((rec["song_id"], rec.get("key", "?"),
                         f"{NOTE[it]} {im}", s_dia_i / s_n))

    print(f"{'song_id':<18} {'annot_key':<9} {'infer_key':<10} {'diatonic%':>9}")
    print("-" * 50)
    for sid, ak, ik, frac in per_song:
        print(f"{sid:<18} {ak:<9} {ik:<10} {frac:>8.1%}")

    print("\n=== PREMISE CHECK (issue #20) ===")
    print(f"songs scored              : {len(per_song)}")
    print(f"GT chord events           : {tot_n}")
    print(f"key infer==annot agreement: {key_agree}/{key_total} "
          f"({key_agree / max(key_total, 1):.1%})")
    print(f"diatonic %% (infer_key, count) : {dia_infer_n / tot_n:.1%}")
    print(f"diatonic %% (infer_key, dur)   : {dia_infer_dur / tot_dur:.1%}")
    print(f"diatonic %% (annot key, count) : {dia_annot_n / tot_n:.1%}")
    frac = dia_infer_n / tot_n
    print(f"\nDECISION (>=60%% -> implement): "
          f"{'PASS -> implement' if frac >= 0.60 else 'FAIL -> STOP, no impl'}")


if __name__ == "__main__":
    main()
