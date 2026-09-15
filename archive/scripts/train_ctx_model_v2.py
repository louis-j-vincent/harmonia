"""Online curriculum training — ctx family+root model v2.

Improvements over v1:
  1. Root-interval context (9 × 12-dim one-hot): explicit harmonic motion signal
     so the MLP can learn ii-V-I patterns and key-change detection directly.
  2. Beat-seq root logits (12-dim): MIDI piano-roll run through the robust
     beat_seq_model (88.3% root accuracy) added as input features.
  3. Dual-head architecture: shared trunk → family_head (5) + root_head (12).
     Loss = 0.6*CE_family + 0.4*CE_root — MIREX-aligned training objective.
  4. Metrics: family_acc, root_acc, MIREX_root proxy, MIREX_majmin proxy,
     maj_acc, min_acc — tracked every step on a held-out val set.
  5. Disk check every 50 steps; training pauses+warns if < 500 MB free.

Input features per segment (672d total):
  12d   chroma_mean (root-shifted, L2-normed)
  540d  ctx ll_mat   (9 × 5 × 12, key-unified rolling)   [existing]
  108d  root intervals (9 positions × 12-dim one-hot)      [NEW]
   12d  beat_seq root probs (from MIDI piano-roll)          [NEW]

Output (dual heads off 128-dim hidden):
  5d    family logits  (major/minor/dim/aug/sus)
  12d   root logits    (pitch classes 0-11)

Usage:
    .venv/bin/python scripts/train_ctx_model_v2.py
    .venv/bin/python scripts/train_ctx_model_v2.py --steps 3000 --init-songs 60

Saves: harmonia/models/ctx_v2.npz
Log:   /tmp/harmonia_ctx_v2.log
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
import warnings
from collections import deque
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from analyze_accomp_emission import parse_chord, song_chord_spans
from build_accomp_audio_hard import SOUNDFONTS, pink, render_to_array, stem_midi
from build_audio_chord_features import BUCKET_FAMILY
from experiment_ctx_model import (
    CTX_K,
    DIST_CACHE,
    FAMILIES,
    _ctx_tensor,
    _fit_entropy_gate,
    _ltas_frames,
    compute_softmax_att_ll,
)
from harmonia.data.midi_renderer import MIDIRenderer
from harmonia.models.chord_pipeline_v1 import MODELS, _BeatSeqModel, _chroma88
from harmonia.models.local_key_data import parse_global_key
from harmonia.theory.local_key import (
    consolidate_dominant_chains,
    continuity_scale_track_v2,
)

DB = REPO / "data" / "accomp_db" / "db.jsonl"

# ── key-relative local-key context feature (issue #20/#23, volet 2) ────────────
# NEW input block: for each of the 9 window positions, the *scale degree* of that
# chord's root relative to the local key at that position — (root_j - tonic_j) %
# 12, one-hot(12) — plus a 1-bit mode flag (0 major / 1 minor).  KEY-AGNOSTIC BY
# CONSTRUCTION: it encodes a degree (relation of root to local tonic), never an
# absolute tonic, so it is invariant under transposition of the whole song
# (a I-V-vi-IV in C and in F produce bit-identical features).  The local key per
# chord is the rule-based teacher (theory.local_key.continuity_scale_track_v2 for
# LOCAL_KEY_MODE="v2", + consolidate_dominant_chains for "v3"); "off" disables
# the block entirely (identical to the pre-volet-2 684d model — the baseline).
LOCAL_KEY_MODE = "off"   # {"off","v2","v3"} — set by --local-key at runtime
LK_DEG_DIM = 12          # one-hot scale degree
LK_MODE_DIM = 1          # major/minor bit
LK_POS_DIM = LK_DEG_DIM + LK_MODE_DIM   # 13 per window position


def _song_local_key_labels(span_tokens: list[str], span_roots_orig: list[int],
                           home_tonic: int, home_mode: str,
                           mode: str) -> list[tuple[int, int]]:
    """Per-span (scale_degree, mode_bit) from the rule-based local-key teacher.

    ``span_tokens``/``span_roots_orig`` are index-aligned over ALL chord spans of
    a song (in the *original*, unshifted key — the degree is transpose-invariant,
    so the audio pitch-shift never enters here).  ``mode`` selects the teacher:
    "v2" = raw :func:`continuity_scale_track_v2`; "v3" = that, post-processed by
    :func:`consolidate_dominant_chains` (secondary-dominant chains read as one
    key).  Returns ``(degree 0..11, mode_bit 0/1)`` per span; falls back to
    ``(root, 0)`` on any teacher failure (inert-ish, still degree-relative).
    """
    n = len(span_tokens)
    try:
        track = continuity_scale_track_v2(span_tokens, home_tonic=home_tonic,
                                          home_mode=home_mode)
        if mode == "v3":
            track = consolidate_dominant_chains(track, span_tokens,
                                                home_tonic=home_tonic,
                                                home_mode=home_mode)
    except Exception:
        return [(int(r) % 12, 0) for r in span_roots_orig]
    out: list[tuple[int, int]] = []
    for i in range(n):
        sc = track[i]
        deg = (span_roots_orig[i] - sc["tonic"]) % 12
        out.append((int(deg), 0 if sc["mode"] == "major" else 1))
    return out

MAJMIN_MAP = {
    "major": "major", "minor": "minor",
    "diminished": "other", "augmented": "other", "suspended": "other",
}

# ── easy audio scenarios ──────────────────────────────────────────────────────
EASY_SCENARIOS = [
    {"name": "clean",       "chords": 1.0,  "bass": 0.9, "drums": 0.7,  "snr_db": (28, 35), "reverb": False},
    {"name": "mild_reverb", "chords": 0.9,  "bass": 0.8, "drums": 0.6,  "snr_db": (25, 32), "reverb": True},
    {"name": "mild_noise",  "chords": 0.85, "bass": 0.8, "drums": 0.65, "snr_db": (20, 28), "reverb": False},
    {"name": "bass_up",     "chords": 0.7,  "bass": 1.1, "drums": 0.6,  "snr_db": (24, 32), "reverb": False},
    {"name": "drums_mild",  "chords": 0.8,  "bass": 0.8, "drums": 1.0,  "snr_db": (22, 30), "reverb": False},
]

_beat_seq_model: _BeatSeqModel | None = None

def _get_beat_seq() -> _BeatSeqModel | None:
    global _beat_seq_model
    if _beat_seq_model is not None:
        return _beat_seq_model
    p = MODELS / "beat_seq_model.npz"
    if p.exists():
        _beat_seq_model = _BeatSeqModel(p)
    return _beat_seq_model


# ── MIDI → per-beat piano roll ────────────────────────────────────────────────

def _midi_beat_rolls(pm, tempo: float, shift: int = 0
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-beat 88-dim onset + note piano rolls from a PrettyMIDI object.

    shift: semitone pitch shift already applied to pm (for bookkeeping only).
    Returns (onset_b, note_b) each (n_beats, 88) float32.
    """
    spb = 60.0 / max(tempo, 1)
    duration = pm.get_end_time()
    n_beats = max(1, int(np.ceil(duration / spb)) + 2)
    onset_b = np.zeros((n_beats, 88), np.float32)
    note_b  = np.zeros((n_beats, 88), np.float32)
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            pitch_idx = note.pitch - 21  # MIDI 21=A0 → index 0
            if not (0 <= pitch_idx < 88):
                continue
            b_on  = min(int(note.start / spb), n_beats - 1)
            b_off = min(int(note.end   / spb), n_beats - 1)
            onset_b[b_on, pitch_idx] += note.velocity / 127.0
            for b in range(b_on, b_off + 1):
                note_b[b, pitch_idx] += 1.0
    return onset_b, note_b


def _beat_seq_probs_song(pm, tempo: float) -> np.ndarray:
    """Run beat_seq_model on a PrettyMIDI, return (n_beats, 12) root probs."""
    bsm = _get_beat_seq()
    if bsm is None:
        n_beats = max(1, int(pm.get_end_time() / (60.0 / max(tempo, 1))) + 2)
        return np.full((n_beats, 12), 1.0 / 12, dtype=np.float32)
    onset_b, note_b = _midi_beat_rolls(pm, tempo)
    return bsm.predict_proba(onset_b, note_b)  # (n_beats, 12)


# ── audio rendering ───────────────────────────────────────────────────────────

def _render_easy(midi_path: Path, rng: np.random.Generator):
    """Returns (audio_f32, sr, pretty_midi, tempo, shift)."""
    import pretty_midi, copy
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    pm   = pretty_midi.PrettyMIDI(str(midi_path))
    scen = EASY_SCENARIOS[int(rng.integers(0, len(EASY_SCENARIOS)))]
    sf   = SOUNDFONTS[int(rng.integers(0, len(SOUNDFONTS)))]
    tempo = pm.estimate_tempo()

    shift = int(rng.integers(-5, 6))
    if shift != 0:
        pm = copy.deepcopy(pm)
        for inst in pm.instruments:
            if not inst.is_drum:
                for n in inst.notes:
                    n.pitch = int(np.clip(n.pitch + shift, 21, 108))

    stems = {
        "chords": stem_midi(pm, lambda i: (not i.is_drum) and "bass" not in i.name.lower()),
        "bass":   stem_midi(pm, lambda i: "bass" in i.name.lower()),
        "drums":  stem_midi(pm, lambda i: i.is_drum),
    }
    waves, sr = {}, 44100
    for name, s in stems.items():
        if s and s.instruments:
            w, sr = render_to_array(renderer, s, sf, reverb=scen["reverb"])
            waves[name] = w

    if not waves:
        raise RuntimeError("No stems rendered")

    L   = max(len(w) for w in waves.values())
    mix = np.zeros(L, np.float32)
    for name, w in waves.items():
        mix[:len(w)] += scen.get(name, 0.7) * w

    snr_db = float(rng.uniform(*scen["snr_db"]))
    p = float(np.mean(mix**2)) + 1e-9
    mix += (pink(L, rng) * float(np.sqrt(p / (10**(snr_db / 10))))).astype(np.float32)
    peak = float(np.abs(mix).max())
    if peak > 0.99:
        mix *= 0.99 / peak
    return mix, sr, pm, tempo, shift


# ── per-song feature extraction ───────────────────────────────────────────────

def _song_to_records(sid: str, recs: dict, dist: dict,
                     rng: np.random.Generator) -> list[dict]:
    rec = recs[sid]
    midi_path_str = rec.get("midi_path", "")
    if not midi_path_str:
        return []
    midi_path = REPO / midi_path_str
    if not midi_path.exists():
        return []

    try:
        audio, sr, pm_shifted, tempo, shift = _render_easy(midi_path, rng)
    except Exception:
        return []

    chroma, ct = _ltas_frames(audio, sr)
    # Beat-seq root probs from the shifted MIDI (oracle quality, no Basic Pitch)
    bsp = _beat_seq_probs_song(pm_shifted, tempo)   # (n_beats, 12)

    bpb = rec["beats_per_bar"]
    spb = 60.0 / max(rec["tempo"], 1)
    chord_at = {(e["bar"] - 1) * bpb + e["beat"]: e for e in rec.get("chord_timeline", [])}

    # ── local-key teacher labels (volet 2) ────────────────────────────────────
    # Run the rule-based tracker ONCE over the whole song's ordered iReal token
    # stream (in the original key — the degree feature is transpose-invariant, so
    # the audio pitch-shift is irrelevant here), then attach a (degree, mode) to
    # each kept record by span index.  Uses the GT quality tokens the tracker
    # needs; during training the context quality is known, unlike at inference
    # (the circularity the two-pass prod scheme handles — see report).
    spans = list(song_chord_spans(rec))
    lk_labels: list[tuple[int, int]] = []
    if LOCAL_KEY_MODE != "off":
        span_toks: list[str] = []
        span_roots: list[int] = []
        for t0, _t1, root_gt_orig, _q in spans:
            b0 = int(round(t0 / spb))
            entry = chord_at.get(b0, {})
            span_toks.append(entry.get("ireal") or entry.get("mma") or "C")
            span_roots.append(int(root_gt_orig) % 12)
        gk = parse_global_key(rec.get("key", "")) or (0, "major")
        lk_labels = _song_local_key_labels(span_toks, span_roots, gk[0], gk[1],
                                           LOCAL_KEY_MODE)

    records = []
    for span_idx, (t0, t1, root_gt_orig, _) in enumerate(spans):
        b0  = int(round(t0 / spb))
        mma = chord_at.get(b0, {}).get("mma")
        p   = parse_chord(mma) if mma else None
        if p is None or p[1] not in BUCKET_FAMILY:
            continue
        fam = BUCKET_FAMILY[p[1]]
        if fam not in FAMILIES:
            continue

        # GT root adjusted for pitch shift applied to audio
        root = int((root_gt_orig + shift) % 12)

        i0 = int(np.searchsorted(ct, t0))
        i1 = int(np.searchsorted(ct, t1))
        if i1 <= i0:
            i1 = i0 + 1
        frames_abs     = chroma[:, i0:i1]
        frames_shifted = np.roll(frames_abs, -root, axis=0)

        mean_s = frames_shifted.mean(axis=1)
        nn     = float(np.linalg.norm(mean_s))
        chroma_mean = (mean_s / nn).astype(np.float32) if nn > 1e-9 else np.zeros(12, np.float32)
        ll_mat = compute_softmax_att_ll(frames_shifted, dist)   # (5, 12)

        # Beat-seq root probs for this segment (average over beats in segment)
        beat_start = int(t0 / spb)
        beat_end   = max(beat_start + 1, int(t1 / spb))
        beat_start = min(beat_start, len(bsp) - 1)
        beat_end   = min(beat_end,   len(bsp))
        seg_root_probs_abs = bsp[beat_start:beat_end].mean(axis=0).astype(np.float32)  # (12,) absolute
        # Rolled version: GT root at index 0 (context-relative, for family features)
        seg_root_probs_rel = np.roll(seg_root_probs_abs, -root).astype(np.float32)

        rec_out = {
            "y":                FAMILIES.index(fam),
            "root_pc":          root,
            "chroma_mean":      chroma_mean,
            "ll_mat":           ll_mat,
            "root_probs_rel":   seg_root_probs_rel,   # rolled: root at 0 — helps family
            "root_probs_abs":   seg_root_probs_abs,   # absolute pitch class — needed for root head
            "song_id":          sid,
        }
        if lk_labels:
            deg, mbit = lk_labels[span_idx]
            rec_out["lk_degree"] = deg      # (root - local_tonic) % 12, degree-relative
            rec_out["lk_mode"]   = mbit     # 0 major / 1 minor
        records.append(rec_out)
    return records


# ── feature assembly ──────────────────────────────────────────────────────────

def _root_interval_onehots(records: list[dict], k: int = CTX_K) -> np.ndarray:
    """For each record i, 9 × 12-dim one-hot of (root_j - root_i) % 12.
    Returns (N, 9*12) = (N, 108)."""
    N = len(records)
    W = 2 * k + 1
    out = np.zeros((N, W * 12), dtype=np.float32)
    for i in range(N):
        root_i = records[i]["root_pc"]
        for j_idx, offset in enumerate(range(-k, k + 1)):
            ni = i + offset
            if 0 <= ni < N:
                delta = int((records[ni]["root_pc"] - root_i) % 12)
                out[i, j_idx * 12 + delta] = 1.0
    return out


def _localkey_ctx_onehots(records: list[dict], k: int = CTX_K) -> np.ndarray:
    """For each record i, 9 window positions × (12-dim degree one-hot + 1 mode bit).

    Position j carries the scale degree ``lk_degree`` (root vs local tonic) and
    ``lk_mode`` bit of chord ``i+offset`` — already degree-relative, so the block
    is transpose-invariant.  Returns ``(N, 9*13) = (N, 117)``.  All-zero if the
    records carry no ``lk_degree`` (LOCAL_KEY_MODE == "off")."""
    N = len(records)
    W = 2 * k + 1
    out = np.zeros((N, W * LK_POS_DIM), dtype=np.float32)
    if N == 0 or "lk_degree" not in records[0]:
        return out
    for i in range(N):
        for j_idx, offset in enumerate(range(-k, k + 1)):
            ni = i + offset
            if 0 <= ni < N:
                base = j_idx * LK_POS_DIM
                out[i, base + int(records[ni]["lk_degree"]) % 12] = 1.0
                out[i, base + LK_DEG_DIM] = float(records[ni]["lk_mode"])
    return out


def _build_features(records: list[dict]):
    """Returns X_logreg (N,17), X_ctx (N,684 or 801), y_family (N,), y_root (N,).

    X_ctx is 684d without the local-key block (LOCAL_KEY_MODE=="off") or 801d
    with it (684 + 9*13 key-relative local-key context features, volet 2)."""
    y_fam  = np.array([r["y"]       for r in records])
    y_root = np.array([r["root_pc"] for r in records])

    X_chroma    = np.stack([r["chroma_mean"] for r in records])          # (N,12)
    X_base_ll   = np.stack([r["ll_mat"].max(axis=1) for r in records])  # (N,5)
    X_logreg    = np.concatenate([X_chroma, X_base_ll], axis=1)         # (N,17)

    ctx         = _ctx_tensor(records, CTX_K)                           # (N,9,5,12)
    ctx_flat    = ctx.reshape(len(records), -1)                         # (N,540)
    X_root_inv  = _root_interval_onehots(records, CTX_K)               # (N,108)
    X_bsm_root  = np.stack([r["root_probs_rel"] for r in records])     # (N,12) root-relative

    X_bsm_abs  = np.stack([r["root_probs_abs"] for r in records])        # (N,12) absolute
    blocks = [X_chroma, ctx_flat, X_root_inv, X_bsm_root, X_bsm_abs]     # (N,684)
    if records and "lk_degree" in records[0]:
        blocks.append(_localkey_ctx_onehots(records, CTX_K))            # (N,117)
    X_ctx = np.concatenate(blocks, axis=1)
    return X_logreg, X_ctx, y_fam, y_root


# ── dual-head MLP ─────────────────────────────────────────────────────────────

class CtxMLPv2(nn.Module):
    """Shared trunk → family head (5) + root head (12)."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128),    nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
        )
        self.family_head = nn.Linear(128, 5)
        self.root_head   = nn.Linear(128, 12)

    def forward(self, x):
        h = self.shared(x)
        return self.family_head(h), self.root_head(h)


# ── background render worker ──────────────────────────────────────────────────

class RenderWorker:
    def __init__(self, recs: dict, dist: dict, seed: int = 0) -> None:
        self._recs    = recs
        self._dist    = dist
        self._req_q   : deque[str]              = deque()
        self._ready_q : deque[tuple[str, list]] = deque()
        self._lock    = threading.Lock()
        self._stop    = threading.Event()
        self._rng     = np.random.default_rng(seed)
        self._t_gen   : deque[float] = deque(maxlen=20)
        threading.Thread(target=self._run, daemon=True).start()

    def request(self, song_ids: list[str]) -> None:
        with self._lock:
            self._req_q.extend(song_ids)

    def drain(self) -> list[tuple[str, list]]:
        with self._lock:
            out = list(self._ready_q)
            self._ready_q.clear()
        return out

    def n_pending(self) -> int:
        with self._lock:
            return len(self._req_q)

    def mean_gen_time(self) -> float:
        with self._lock:
            return float(np.mean(self._t_gen)) if self._t_gen else 3.0

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                sid = self._req_q.popleft() if self._req_q else None
            if sid is None:
                time.sleep(0.05)
                continue
            t0  = time.time()
            out = _song_to_records(sid, self._recs, self._dist, self._rng)
            with self._lock:
                self._t_gen.append(time.time() - t0)
                if out:
                    self._ready_q.append((sid, out))


# ── curriculum pool ───────────────────────────────────────────────────────────

class SongPool:
    def __init__(self) -> None:
        self._songs     : dict[str, list[dict]] = {}
        self._difficulty: dict[str, float]      = {}

    def add(self, sid: str, records: list[dict]) -> None:
        self._songs[sid]      = records
        self._difficulty[sid] = 1.0

    def all_records(self) -> list[dict]:
        out: list[dict] = []
        for r in self._songs.values():
            out.extend(r)
        return out

    def update_difficulty(self, song_losses: dict[str, list[float]]) -> None:
        for sid, losses in song_losses.items():
            if losses:
                self._difficulty[sid] = float(np.mean(losses))

    def replace_easy(self, n_replace: int,
                     new_songs: list[tuple[str, list]]) -> list[str]:
        valid_new = [(s, r) for s, r in new_songs if r]
        n_swap    = min(n_replace, len(valid_new))
        if n_swap <= 0:
            return []
        ranked  = sorted(self._difficulty.items(), key=lambda x: x[1])
        evicted = []
        for sid, _ in ranked[:n_swap]:
            self._songs.pop(sid, None)
            self._difficulty.pop(sid, None)
            evicted.append(sid)
        for sid, records in valid_new[:n_swap]:
            self.add(sid, records)
        return evicted

    def size(self) -> int:
        return len(self._songs)

    def n_segments(self) -> int:
        return sum(len(v) for v in self._songs.values())


# ── training helpers ──────────────────────────────────────────────────────────

def _train_step(model: CtxMLPv2, Xt: torch.Tensor,
                yt_fam: torch.Tensor, yt_root: torch.Tensor,
                opt, loss_fam_fn, loss_root_fn,
                grad_epochs: int, batch: int = 128) -> float:
    model.train()
    ds = TensorDataset(Xt, yt_fam, yt_root)
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    total, n_seen = 0.0, 0
    for _ in range(grad_epochs):
        for xb, yf, yr in dl:
            opt.zero_grad()
            lf, lr = model(xb)
            loss = 0.6 * loss_fam_fn(lf, yf) + 0.4 * loss_root_fn(lr, yr)
            loss.backward()
            opt.step()
            total  += loss.item() * len(xb)
            n_seen += len(xb)
    return total / max(n_seen, 1)


def _eval(model: CtxMLPv2, Xv: torch.Tensor,
          yv_fam: np.ndarray, yv_root: np.ndarray,
          loss_fam_fn, loss_root_fn,
          device: str) -> dict:
    model.eval()
    with torch.no_grad():
        lf_v, lr_v = model(Xv)
        val_loss = (0.6 * loss_fam_fn(lf_v, torch.tensor(yv_fam, dtype=torch.long).to(device))
                  + 0.4 * loss_root_fn(lr_v, torch.tensor(yv_root, dtype=torch.long).to(device))).item()
        pred_fam  = lf_v.cpu().argmax(1).numpy()
        pred_root = lr_v.cpu().argmax(1).numpy()

    fam_acc  = float((pred_fam  == yv_fam ).mean())
    root_acc = float((pred_root == yv_root).mean())

    # MIREX proxy: root correct + family correct
    mirex_root   = float((pred_root == yv_root).mean())
    # MIREX majmin: root correct AND majmin correct
    mm_gt   = np.array([list(MAJMIN_MAP.values())[list(MAJMIN_MAP.keys()).index(FAMILIES[y])] for y in yv_fam])
    mm_pred = np.array([list(MAJMIN_MAP.values())[list(MAJMIN_MAP.keys()).index(FAMILIES[p])] for p in pred_fam])
    mirex_mm = float(((pred_root == yv_root) & (mm_pred == mm_gt)).mean())

    cm      = confusion_matrix(yv_fam, pred_fam, labels=list(range(5)))
    per_cls = cm.diagonal() / (cm.sum(axis=1) + 1e-9)
    return {
        "val_loss": val_loss,
        "fam_acc":  fam_acc,
        "root_acc": root_acc,
        "mirex_root": mirex_root,
        "mirex_mm": mirex_mm,
        "maj_acc":  float(per_cls[0]),
        "min_acc":  float(per_cls[1]),
    }


def _disk_free_gb() -> float:
    return shutil.disk_usage("/").free / 1e9


def _check_disk(warn_gb: float = 0.8) -> None:
    free = _disk_free_gb()
    if free < warn_gb:
        print(f"\n⚠  DISK LOW: {free:.2f} GB free — pausing 60s for manual cleanup",
              flush=True)
        time.sleep(60)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps",           type=int,   default=3000)
    ap.add_argument("--keep-frac",       type=float, default=0.5)
    ap.add_argument("--init-songs",      type=int,   default=60)
    ap.add_argument("--val-songs",       type=int,   default=20)
    ap.add_argument("--gradient-epochs", type=int,   default=2)
    ap.add_argument("--seed",            type=int,   default=42)
    ap.add_argument("--local-key",       choices=["off", "v2", "v3"], default="off",
                    help="add the key-relative local-key context block (volet 2); "
                         "v2=raw heuristic teacher, v3=+dominant-chain consolidation")
    ap.add_argument("--out",             type=Path,
                    default=REPO / "harmonia" / "models" / "ctx_v2.npz")
    ap.add_argument("--log",             type=Path,
                    default=Path("/tmp/harmonia_ctx_v2.log"))
    ap.add_argument("--status-every",    type=int,   default=10)
    args = ap.parse_args()

    global LOCAL_KEY_MODE
    LOCAL_KEY_MODE = args.local_key

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}  |  Disk free: {_disk_free_gb():.1f} GB  |  "
          f"local-key block: {LOCAL_KEY_MODE}", flush=True)
    print(f"Beat-seq model: {'loaded' if _get_beat_seq() else 'MISSING — using uniform priors'}",
          flush=True)

    if not DIST_CACHE.exists():
        sys.exit(f"Missing {DIST_CACHE}")

    dist_data = np.load(DIST_CACHE)
    dist      = {k: dist_data[k] for k in dist_data.files}

    recs_all = {json.loads(l)["song_id"]: json.loads(l) for l in open(DB)}
    avail    = [sid for sid, r in recs_all.items()
                if r.get("beats_per_bar") == 4 and r.get("midi_path")]
    rng   = np.random.default_rng(args.seed)
    avail = list(rng.permutation(avail))

    val_sids   = avail[:args.val_songs]
    train_sids = avail[args.val_songs:]
    print(f"{len(train_sids)} train  |  {len(val_sids)} val  |  {len(avail)} total", flush=True)

    # ── validation set ────────────────────────────────────────────────────────
    print(f"Rendering {args.val_songs} val songs...", flush=True)
    val_rng     = np.random.default_rng(args.seed + 999)
    val_records: list[dict] = []
    for i, sid in enumerate(val_sids):
        print(f"\r  [{i+1}/{args.val_songs}] {recs_all[sid].get('title','')[:40]:40s}",
              end="", flush=True)
        val_records.extend(_song_to_records(sid, recs_all, dist, val_rng))
    print(f"\n  {len(val_records)} val segments", flush=True)
    if not val_records:
        sys.exit("No val records")

    _, X_ctx_v0, yv_fam, yv_root = _build_features(val_records)

    # ── initial training pool ─────────────────────────────────────────────────
    print(f"Rendering {args.init_songs} initial train songs...", flush=True)
    _check_disk()
    init_rng  = np.random.default_rng(args.seed + 1)
    remaining = deque(train_sids[args.init_songs:])

    t0_gen = time.time()
    pool   = SongPool()
    for i, sid in enumerate(train_sids[:args.init_songs]):
        print(f"\r  [{i+1}/{args.init_songs}] {recs_all[sid].get('title','')[:40]:40s}",
              end="", flush=True)
        out = _song_to_records(sid, recs_all, dist, init_rng)
        if out:
            pool.add(sid, out)
        if (i + 1) % 10 == 0:
            _check_disk()
    print(flush=True)
    t_gen_per = (time.time() - t0_gen) / max(pool.size(), 1)
    print(f"Pool: {pool.size()} songs / {pool.n_segments()} segs  ({t_gen_per:.1f}s/song)",
          flush=True)
    if pool.n_segments() < 10:
        sys.exit("Too few segments — check MIDI files")

    # ── model ─────────────────────────────────────────────────────────────────
    _, X_ctx_tr0, y_tr_fam0, y_tr_root0 = _build_features(pool.all_records())
    sc_ctx   = StandardScaler().fit(X_ctx_tr0)
    flat_dim = X_ctx_tr0.shape[1]   # should be 672
    print(f"Feature dim: {flat_dim}", flush=True)

    model   = CtxMLPv2(flat_dim).to(device)
    counts  = np.bincount(y_tr_fam0, minlength=5).astype(float)
    wts_fam = torch.tensor(1.0 / (counts + 1e-9), dtype=torch.float32).to(device)
    wts_fam = wts_fam / wts_fam.sum() * 5
    loss_fam_fn  = nn.CrossEntropyLoss(weight=wts_fam)
    loss_root_fn = nn.CrossEntropyLoss()   # roots are more balanced
    opt   = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    # ── background worker ─────────────────────────────────────────────────────
    worker = RenderWorker(recs_all, dist, seed=args.seed + 7)
    prefill = [remaining.popleft() for _ in range(min(20, len(remaining)))]
    worker.request(prefill)

    # ── training loop ─────────────────────────────────────────────────────────
    log_f  = open(args.log, "w", buffering=1)
    header = ("step,train_loss,val_loss,fam_acc,root_acc,mirex_root,mirex_mm,"
              "maj_acc,min_acc,pool_songs,pool_segs,t_epoch,t_gen,disk_gb")
    log_f.write(header + "\n")
    print(header, flush=True)

    best_mirex_mm = 0.0
    best_state    : dict | None = None
    t_hist        : deque[float] = deque(maxlen=10)
    t_total       = time.time()

    for step in range(1, args.steps + 1):
        t0 = time.time()

        # disk guard every 50 steps
        if step % 50 == 0:
            _check_disk()

        flat_records = pool.all_records()
        if not flat_records:
            continue
        _, X_ctx_tr, y_tr_fam, y_tr_root = _build_features(flat_records)

        if step % 50 == 1:
            sc_ctx.fit(X_ctx_tr)

        X_sc = sc_ctx.transform(X_ctx_tr).astype(np.float32)
        Xt   = torch.tensor(X_sc).to(device)
        yt_f = torch.tensor(y_tr_fam,  dtype=torch.long).to(device)
        yt_r = torch.tensor(y_tr_root, dtype=torch.long).to(device)

        train_loss = _train_step(model, Xt, yt_f, yt_r, opt,
                                 loss_fam_fn, loss_root_fn, args.gradient_epochs)
        sched.step()
        t_epoch = time.time() - t0
        t_hist.append(t_epoch)

        # per-song difficulty from family CE only (what curriculum tracks)
        model.eval()
        with torch.no_grad():
            lf, _ = model(Xt)
            seg_losses = nn.CrossEntropyLoss(reduction="none")(lf, yt_f).cpu().numpy()
        song_losses: dict[str, list[float]] = {}
        for r, l in zip(flat_records, seg_losses):
            song_losses.setdefault(r["song_id"], []).append(float(l))
        pool.update_difficulty(song_losses)

        # val
        _, X_ctx_v, yv_fam_cur, yv_root_cur = _build_features(val_records)
        Xv  = torch.tensor(sc_ctx.transform(X_ctx_v).astype(np.float32)).to(device)
        metrics = _eval(model, Xv, yv_fam_cur, yv_root_cur,
                        loss_fam_fn, loss_root_fn, device)

        if metrics["mirex_mm"] > best_mirex_mm:
            best_mirex_mm = metrics["mirex_mm"]
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # curriculum
        n_replace = max(1, int(pool.size() * (1.0 - args.keep_frac)))
        evicted   = pool.replace_easy(n_replace, worker.drain())
        for sid in evicted:
            insert_at = int(rng.integers(0, max(len(remaining), 1)))
            remaining.insert(insert_at, sid)
        n_req = n_replace + max(0, 5 - worker.n_pending())
        batch: list[str] = []
        for _ in range(n_req):
            if not remaining:
                remaining.extend(rng.permutation(train_sids).tolist())
            batch.append(remaining.popleft())
        worker.request(batch)

        t_gen  = worker.mean_gen_time()
        disk   = _disk_free_gb()
        row = (f"{step},{train_loss:.4f},{metrics['val_loss']:.4f},"
               f"{metrics['fam_acc']:.4f},{metrics['root_acc']:.4f},"
               f"{metrics['mirex_root']:.4f},{metrics['mirex_mm']:.4f},"
               f"{metrics['maj_acc']:.4f},{metrics['min_acc']:.4f},"
               f"{pool.size()},{pool.n_segments()},{t_epoch:.1f},{t_gen:.1f},{disk:.1f}")
        log_f.write(row + "\n")

        if step % args.status_every == 0 or step <= 5:
            eta_s   = (args.steps - step) * float(np.mean(t_hist))
            eta_str = f"{int(eta_s//3600)}h{int((eta_s%3600)//60):02d}m"
            print(
                f"[{step:4d}/{args.steps}] "
                f"loss={train_loss:.3f}/{metrics['val_loss']:.3f}  "
                f"fam={metrics['fam_acc']:.1%}  root={metrics['root_acc']:.1%}  "
                f"mirex_root={metrics['mirex_root']:.1%}  mirex_mm={metrics['mirex_mm']:.1%}  "
                f"maj={metrics['maj_acc']:.1%}  min={metrics['min_acc']:.1%}  "
                f"pool={pool.size()}s/{pool.n_segments()}seg  "
                f"disk={disk:.1f}GB  best_mm={best_mirex_mm:.1%}  ETA={eta_str}",
                flush=True,
            )

    worker.stop()
    log_f.close()

    # ── save ─────────────────────────────────────────────────────────────────
    print(f"\nBest MIREX majmin proxy: {best_mirex_mm:.1%}  Saving → {args.out}", flush=True)
    if best_state is None:
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    # Fit entropy gate (family logits only) on val set
    flat_records = pool.all_records()
    X_lg_f, X_ctx_f, y_f_fam, y_f_root = _build_features(flat_records)
    sc_base2 = StandardScaler().fit(X_lg_f)
    clf_base2 = LogisticRegression(max_iter=2000, solver="lbfgs",
                                    class_weight="balanced", C=1.0)
    clf_base2.fit(sc_base2.transform(X_lg_f), y_f_fam)
    sc_ctx2 = StandardScaler().fit(X_ctx_f)

    _, X_ctx_v_final, yv_f2, _ = _build_features(val_records)
    oof_base_raw = clf_base2.predict_log_proba(
        sc_base2.transform(_build_features(val_records)[0])
    )
    # Pad to the full 5-family layout: predict_log_proba only emits columns for
    # the families the base logreg actually SAW in the (curriculum) train pool,
    # so a rare family absent there (aug/sus on a small pool) yields < 5 columns
    # and breaks the entropy-gate blend.  Fill any missing family with a large
    # negative log-prob.  (Full-budget runs see all 5 and never hit this.)
    oof_base = np.full((oof_base_raw.shape[0], 5), -20.0, dtype=np.float64)
    oof_base[:, clf_base2.classes_] = oof_base_raw
    model.eval()
    with torch.no_grad():
        oof_fam_logits, _ = model(
            torch.tensor(sc_ctx2.transform(X_ctx_v_final).astype(np.float32)).to(device)
        )
    oof_mlp = oof_fam_logits.cpu().numpy()
    w, b = _fit_entropy_gate(oof_base, oof_mlp, yv_f2)

    mlp_state_np = {k: v.numpy() for k, v in best_state.items()}
    np.savez(
        args.out,
        gate_w         = np.array(w),
        gate_b         = np.array(b),
        flat_dim       = np.array(flat_dim),
        sc_mean        = sc_ctx2.mean_.astype(np.float32),
        sc_std         = sc_ctx2.scale_.astype(np.float32),
        mlp_state      = np.array(mlp_state_np, dtype=object),
        best_mirex_mm  = np.array(best_mirex_mm),
        n_train        = np.array(len(y_f_fam)),
        feature_version= np.array(2),   # v2: includes root_intervals + bsm_root
        local_key_mode = np.array(LOCAL_KEY_MODE),  # off/v2/v3 (volet 2 block)
        **{f"dist_{k}": dist[k] for k in dist},
    )
    print(f"Saved. Best MIREX majmin proxy: {best_mirex_mm:.1%}", flush=True)
    print(f"Disk free: {_disk_free_gb():.1f} GB", flush=True)


if __name__ == "__main__":
    main()
