"""Online curriculum training for the ctx family classifier.

Key ideas:
  1. Background thread renders songs while GPU trains (no idle time).
  2. Pool of N songs; each epoch computes per-song CE loss, keeps the
     hardest (keep_frac) and requests replacements for the easy ones.
  3. Pool size N is auto-tuned so generation time ≈ train time per epoch
     (generation never bottlenecks training).
  4. Easy audio: balanced mix, SNR 20-35 dB, no phone filter, no extreme
     masking — gentler than the hard audio used in the baseline experiments.
  5. MPS (Apple Silicon) by default.

Usage:
    .venv/bin/python scripts/train_online_ctx_model.py
    .venv/bin/python scripts/train_online_ctx_model.py --steps 2000 --pool 80
    .venv/bin/python scripts/train_online_ctx_model.py --out harmonia/models/ctx_online.npz

Writes: harmonia/models/ctx_online.npz  (same format as ctx_family_model.npz)
Log:    /tmp/harmonia_online_train.log
"""
from __future__ import annotations

import argparse
import json
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
    _make_mlp,
    compute_softmax_att_ll,
    _ltas_frames,
)
from harmonia.data.midi_renderer import MIDIRenderer

DB = REPO / "data" / "accomp_db" / "db.jsonl"

# ── easy audio scenarios ───────────────────────────────────────────────────────
# Gentler than _render_hard: SNR 20-35 dB, no phone filter, no melody masking.
EASY_SCENARIOS = [
    {"name": "clean",       "chords": 1.0,  "bass": 0.9, "drums": 0.7,  "snr_db": (28, 35), "reverb": False},
    {"name": "mild_reverb", "chords": 0.9,  "bass": 0.8, "drums": 0.6,  "snr_db": (25, 32), "reverb": True},
    {"name": "mild_noise",  "chords": 0.85, "bass": 0.8, "drums": 0.65, "snr_db": (20, 28), "reverb": False},
    {"name": "bass_up",     "chords": 0.7,  "bass": 1.1, "drums": 0.6,  "snr_db": (24, 32), "reverb": False},
    {"name": "drums_mild",  "chords": 0.8,  "bass": 0.8, "drums": 1.0,  "snr_db": (22, 30), "reverb": False},
]


def _render_easy(midi_path: Path, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Render MIDI to easy-mode audio array."""
    import pretty_midi, copy
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    pm   = pretty_midi.PrettyMIDI(str(midi_path))
    scen = EASY_SCENARIOS[int(rng.integers(0, len(EASY_SCENARIOS)))]
    sf   = SOUNDFONTS[int(rng.integers(0, len(SOUNDFONTS)))]

    # Random pitch shift ±5 semitones for key diversity
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

    # Additive pink noise at target SNR (no phone filter, no time-varying gain)
    lo, hi = scen["snr_db"]
    snr_db = float(rng.uniform(lo, hi))
    p = float(np.mean(mix**2)) + 1e-9
    noise = pink(L, rng) * float(np.sqrt(p / (10**(snr_db / 10))))
    mix   = (mix + noise.astype(np.float32))

    peak = float(np.abs(mix).max())
    if peak > 0.99:
        mix *= 0.99 / peak
    return mix, sr


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
        audio, sr = _render_easy(midi_path, rng)
    except Exception:
        return []

    chroma, ct = _ltas_frames(audio, sr)
    bpb = rec["beats_per_bar"]
    spb = 60.0 / max(rec["tempo"], 1)
    chord_at = {(e["bar"] - 1) * bpb + e["beat"]: e for e in rec.get("chord_timeline", [])}

    records = []
    for t0, t1, root_gt, _ in song_chord_spans(rec):
        b0  = int(round(t0 / spb))
        mma = chord_at.get(b0, {}).get("mma")
        p   = parse_chord(mma) if mma else None
        if p is None or p[1] not in BUCKET_FAMILY:
            continue
        fam = BUCKET_FAMILY[p[1]]
        if fam not in FAMILIES:
            continue
        root = int(root_gt % 12)

        i0 = int(np.searchsorted(ct, t0))
        i1 = int(np.searchsorted(ct, t1))
        if i1 <= i0:
            i1 = i0 + 1
        frames_abs     = chroma[:, i0:i1]
        frames_shifted = np.roll(frames_abs, -root, axis=0)

        mean_s = frames_shifted.mean(axis=1)
        nn     = float(np.linalg.norm(mean_s))
        chroma_mean = (mean_s / nn).astype(np.float32) if nn > 1e-9 else np.zeros(12, np.float32)
        ll_mat = compute_softmax_att_ll(frames_shifted, dist)

        records.append({
            "y":           FAMILIES.index(fam),
            "chroma_mean": chroma_mean,
            "ll_mat":      ll_mat,
            "root_pc":     root,
            "song_id":     sid,
        })
    return records


# ── background render worker ──────────────────────────────────────────────────

class RenderWorker:
    """Continuously renders songs from a request queue into a ready queue."""

    def __init__(self, recs: dict, dist: dict, seed: int = 0) -> None:
        self._recs    = recs
        self._dist    = dist
        self._req_q   : deque[str]              = deque()
        self._ready_q : deque[tuple[str, list]] = deque()
        self._lock    = threading.Lock()
        self._stop    = threading.Event()
        self._rng     = np.random.default_rng(seed)
        self._t_gen   : deque[float] = deque(maxlen=20)
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

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
            dt  = time.time() - t0
            with self._lock:
                self._t_gen.append(dt)
                if out:
                    self._ready_q.append((sid, out))


# ── curriculum song pool ──────────────────────────────────────────────────────

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
        """Swap easiest songs for new ones. Returns list of evicted sids."""
        # Only evict as many as we can replace so pool size stays stable
        valid_new = [(sid, r) for sid, r in new_songs if r]
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


# ── feature building ──────────────────────────────────────────────────────────

def _build_features(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y         = np.array([r["y"] for r in records])
    X_chroma  = np.stack([r["chroma_mean"] for r in records])
    X_base_ll = np.stack([r["ll_mat"].max(axis=1) for r in records])
    X_logreg  = np.concatenate([X_chroma, X_base_ll], axis=1)
    ctx       = _ctx_tensor(records, CTX_K)       # (N, 9, 5, 12)
    ctx_flat  = ctx.reshape(len(records), -1)
    X_ctx     = np.concatenate([X_chroma, ctx_flat], axis=1)
    return X_logreg, X_ctx, y


# ── train / eval helpers ──────────────────────────────────────────────────────

def _train_epoch(model: nn.Module, Xt: torch.Tensor, yt: torch.Tensor,
                 opt, loss_fn, batch: int = 128) -> float:
    model.train()
    dl     = DataLoader(TensorDataset(Xt, yt), batch_size=batch, shuffle=True)
    total  = 0.0
    n_seen = 0
    for xb, yb in dl:
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        total  += loss.item() * len(yb)
        n_seen += len(yb)
    return total / max(n_seen, 1)


def _per_record_loss(model: nn.Module, Xt: torch.Tensor,
                     yt: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return nn.CrossEntropyLoss(reduction="none")(model(Xt), yt).cpu().numpy()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps",           type=int,   default=3000)
    ap.add_argument("--keep-frac",       type=float, default=0.5)
    ap.add_argument("--init-songs",      type=int,   default=150)
    ap.add_argument("--val-songs",       type=int,   default=20)
    ap.add_argument("--gradient-epochs", type=int,   default=2)
    ap.add_argument("--seed",            type=int,   default=42)
    ap.add_argument("--out",             type=Path,
                    default=REPO / "harmonia" / "models" / "ctx_online.npz")
    ap.add_argument("--log",             type=Path,
                    default=Path("/tmp/harmonia_online_train.log"))
    ap.add_argument("--status-every",    type=int,   default=10)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    if not DIST_CACHE.exists():
        sys.exit(f"Missing {DIST_CACHE} — run plot_family_likelihood.py --rebuild-cache")

    dist_data = np.load(DIST_CACHE)
    dist      = {k: dist_data[k] for k in dist_data.files}

    recs_all = {json.loads(l)["song_id"]: json.loads(l) for l in open(DB)}
    avail    = [sid for sid, r in recs_all.items()
                if r.get("beats_per_bar") == 4 and r.get("midi_path")]
    rng   = np.random.default_rng(args.seed)
    avail = list(rng.permutation(avail))

    val_sids   = avail[:args.val_songs]
    train_sids = avail[args.val_songs:]
    print(f"{len(train_sids)} train songs, {len(val_sids)} val, {len(avail)} total",
          flush=True)

    # ── validation set (rendered once) ───────────────────────────────────────
    print(f"Rendering {args.val_songs} validation songs...", flush=True)
    val_rng     = np.random.default_rng(args.seed + 999)
    val_records: list[dict] = []
    for i, sid in enumerate(val_sids):
        print(f"\r  [{i+1}/{args.val_songs}] {recs_all[sid].get('title','')[:40]:40s}",
              end="", flush=True)
        val_records.extend(_song_to_records(sid, recs_all, dist, val_rng))
    print(f"\n  {len(val_records)} val segments", flush=True)
    if not val_records:
        sys.exit("No val records — check DB midi_path fields")

    _, X_ctx_v_base, y_v = _build_features(val_records)

    # ── initial training pool ─────────────────────────────────────────────────
    print(f"Rendering {args.init_songs} initial training songs...", flush=True)
    init_rng  = np.random.default_rng(args.seed + 1)
    remaining = deque(train_sids[args.init_songs:])

    t0_gen = time.time()
    pool   = SongPool()
    for i, sid in enumerate(train_sids[:args.init_songs]):
        print(f"\r  [{i+1}/{args.init_songs}] {recs_all[sid].get('title','')[:40]:40s}",
              end="", flush=True)
        recs_out = _song_to_records(sid, recs_all, dist, init_rng)
        if recs_out:
            pool.add(sid, recs_out)
    print(flush=True)
    t_gen_per_song = (time.time() - t0_gen) / max(pool.size(), 1)
    print(f"Pool: {pool.size()} songs / {pool.n_segments()} segs  "
          f"({t_gen_per_song:.1f}s/song)", flush=True)
    if pool.n_segments() < 10:
        sys.exit("Too few segments — check MIDI files")

    # ── model init ────────────────────────────────────────────────────────────
    _, X_ctx_tr0, y_tr0 = _build_features(pool.all_records())
    sc_ctx   = StandardScaler().fit(X_ctx_tr0)
    flat_dim = X_ctx_tr0.shape[1]
    model    = _make_mlp(flat_dim).to(device)

    counts  = np.bincount(y_tr0, minlength=5).astype(float)
    wts     = torch.tensor(1.0 / (counts + 1e-9), dtype=torch.float32).to(device)
    wts     = wts / wts.sum() * 5
    opt     = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    loss_fn = nn.CrossEntropyLoss(weight=wts)

    # ── background worker ─────────────────────────────────────────────────────
    worker = RenderWorker(recs_all, dist, seed=args.seed + 7)
    prefill = [remaining.popleft() for _ in range(min(30, len(remaining)))]
    worker.request(prefill)

    # ── training loop ─────────────────────────────────────────────────────────
    log_f  = open(args.log, "w", buffering=1)
    header = "step,train_loss,val_loss,val_acc,maj_acc,min_acc,pool_songs,pool_segs,t_epoch,t_gen"
    log_f.write(header + "\n")
    print(header, flush=True)

    best_val_acc = 0.0
    best_state   : dict | None = None
    t_epoch_hist : deque[float] = deque(maxlen=10)
    t_total_start = time.time()
    X_ctx_v_cur   = X_ctx_v_base  # updated each step

    for step in range(1, args.steps + 1):
        t0 = time.time()

        flat_records = pool.all_records()
        if not flat_records:
            continue
        _, X_ctx_tr, y_tr = _build_features(flat_records)

        if step % 50 == 1:
            sc_ctx.fit(X_ctx_tr)

        X_sc = sc_ctx.transform(X_ctx_tr).astype(np.float32)
        Xt   = torch.tensor(X_sc).to(device)
        yt   = torch.tensor(y_tr, dtype=torch.long).to(device)

        train_loss = 0.0
        for _ in range(args.gradient_epochs):
            train_loss = _train_epoch(model, Xt, yt, opt, loss_fn)
        sched.step()

        t_epoch = time.time() - t0
        t_epoch_hist.append(t_epoch)

        # per-song difficulty update
        seg_losses = _per_record_loss(model, Xt, yt)
        song_losses: dict[str, list[float]] = {}
        for r, l in zip(flat_records, seg_losses):
            song_losses.setdefault(r["song_id"], []).append(float(l))
        pool.update_difficulty(song_losses)

        # validate
        _, X_ctx_v_cur, y_v_cur = _build_features(val_records)
        X_sc_v   = sc_ctx.transform(X_ctx_v_cur).astype(np.float32)
        Xv       = torch.tensor(X_sc_v).to(device)
        yv_t     = torch.tensor(y_v_cur, dtype=torch.long).to(device)
        model.eval()
        with torch.no_grad():
            logits_v = model(Xv)
            val_loss = nn.CrossEntropyLoss()(logits_v, yv_t).item()
            preds_v  = logits_v.cpu().argmax(1).numpy()
        val_acc = float((preds_v == y_v_cur).mean())
        cm      = confusion_matrix(y_v_cur, preds_v, labels=list(range(5)))
        per_cls = cm.diagonal() / (cm.sum(axis=1) + 1e-9)
        maj_acc = float(per_cls[0])
        min_acc = float(per_cls[1])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # curriculum swap
        n_replace = max(1, int(pool.size() * (1.0 - args.keep_frac)))
        evicted   = pool.replace_easy(n_replace, worker.drain())

        # Put evicted songs back into the rotation at a random future position
        # so they can return with difficulty=1.0 (keeps a low-freq easy signal).
        for sid in evicted:
            insert_at = int(rng.integers(0, max(len(remaining), 1)))
            remaining.insert(insert_at, sid)

        n_request = n_replace + max(0, 5 - worker.n_pending())
        batch: list[str] = []
        for _ in range(n_request):
            if not remaining:
                remaining.extend(rng.permutation(train_sids).tolist())
            batch.append(remaining.popleft())
        worker.request(batch)

        t_gen = worker.mean_gen_time()
        row = (f"{step},{train_loss:.4f},{val_loss:.4f},{val_acc:.4f},"
               f"{maj_acc:.4f},{min_acc:.4f},{pool.size()},{pool.n_segments()},"
               f"{t_epoch:.1f},{t_gen:.1f}")
        log_f.write(row + "\n")

        if step % args.status_every == 0 or step <= 5:
            steps_left = args.steps - step
            eta_s   = steps_left * float(np.mean(t_epoch_hist))
            eta_str = f"{int(eta_s//3600)}h{int((eta_s%3600)//60):02d}m"
            print(
                f"[{step:4d}/{args.steps}] "
                f"train={train_loss:.3f}  val={val_loss:.3f}  acc={val_acc:.1%}  "
                f"maj={maj_acc:.1%}  min={min_acc:.1%}  "
                f"pool={pool.size()}s/{pool.n_segments()}seg  "
                f"t_ep={t_epoch:.1f}s  t_gen={t_gen:.1f}s/song  "
                f"best={best_val_acc:.1%}  ETA={eta_str}",
                flush=True,
            )

    worker.stop()
    log_f.close()

    # ── save ─────────────────────────────────────────────────────────────────
    print(f"\nBest val acc: {best_val_acc:.1%}  Saving → {args.out}", flush=True)
    if best_state is None:
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    flat_records = pool.all_records()
    X_lg_f, X_ctx_f, y_f = _build_features(flat_records)
    sc_base2 = StandardScaler().fit(X_lg_f)
    clf_base2 = LogisticRegression(max_iter=2000, solver="lbfgs",
                                    class_weight="balanced", C=1.0)
    clf_base2.fit(sc_base2.transform(X_lg_f), y_f)
    sc_ctx2 = StandardScaler().fit(X_ctx_f)

    _, X_ctx_v_final, y_v_final = _build_features(val_records)
    oof_base = clf_base2.predict_log_proba(sc_base2.transform(X_lg_f[:len(val_records)]))
    model.eval()
    with torch.no_grad():
        oof_mlp = (model(torch.tensor(
                   sc_ctx2.transform(X_ctx_v_final).astype(np.float32)).to(device))
                   .cpu().numpy())
    w, b = _fit_entropy_gate(oof_base, oof_mlp, y_v_final)

    np.savez(
        args.out,
        gate_w      = np.array(w),
        gate_b      = np.array(b),
        flat_dim    = np.array(flat_dim),
        hidden1     = np.array(256),
        hidden2     = np.array(128),
        sc_mean     = sc_ctx2.mean_.astype(np.float32),
        sc_std      = sc_ctx2.scale_.astype(np.float32),
        mlp_state   = np.array({k: v.numpy() for k, v in best_state.items()}, dtype=object),
        best_val_acc= np.array(best_val_acc),
        n_train     = np.array(len(y_f)),
        **{f"dist_{k}": dist[k] for k in dist},
    )
    print(f"Saved. Best val acc: {best_val_acc:.1%}", flush=True)


if __name__ == "__main__":
    main()
