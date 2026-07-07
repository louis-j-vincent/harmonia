"""Diagnostic: CQT chroma for bars 1-4 with beat grid and segment cut markers.

Usage:
    .venv/bin/python scripts/debug_segmentation.py --song "Anthropology"
    .venv/bin/python scripts/debug_segmentation.py --song "Anthropology" --bars 8
"""
from __future__ import annotations
import argparse
import json
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pretty_midi
import soundfile as sf

from build_accomp_audio_hard import (
    SCENARIOS, SOUNDFONTS, make_melody, LEAD_PROGRAMS,
    render_to_array, stem_midi, time_varying_degrade,
)
from harmonia.data.midi_renderer import MIDIRenderer
from harmonia.models.stage1_pitch import PitchExtractor
from analyze_accomp_emission import parse_chord, song_chord_spans
from build_audio_chord_features import BUCKET_FAMILY

DB       = REPO / "data" / "accomp_db" / "db.jsonl"
MANIFEST = REPO / "data" / "accomp_db" / "audio" / "manifest.jsonl"
NOTE     = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

FAM_SUFFIX = {"major": "^7", "minor": "-7", "diminished": "o7",
              "augmented": "+", "suspended": "sus"}

# 60 chord templates (12 roots × 5 families) for best-template family detection
_FAM_TONES = {"major": [0,4,7], "minor": [0,3,7], "diminished": [0,3,6],
              "augmented": [0,4,8], "suspended": [0,5,7]}
_FAMILIES  = list(_FAM_TONES.keys())
_TEMPLATES = []
for _r in range(12):
    for _fn, _tones in _FAM_TONES.items():
        _t = np.zeros(12)
        for _off in _tones: _t[(_r + _off) % 12] = 1.0
        _TEMPLATES.append((_r, _fn, _t / np.linalg.norm(_t)))


def _reg(v88, lo, hi):
    c = np.zeros(12)
    for k in range(88):
        if lo <= 21 + k < hi:
            c[(21 + k) % 12] += v88[k]
    return c

def _unit(v):
    n = np.linalg.norm(v); return v / n if n > 1e-9 else v

def pool_to_beats(frame_times, probs, beat_times):
    n = len(beat_times)
    out = np.zeros((n, probs.shape[1]), dtype=np.float32)
    idx = np.searchsorted(beat_times, frame_times)
    for b, p in zip(idx, probs):
        if 0 <= b < n: out[b] += p
    return out


def run(song_q: str, n_bars: int, seed: int, out_path: Path):
    recs = {json.loads(l)["song_id"]: json.loads(l) for l in open(DB)}
    man: dict = {}
    for mm in map(json.loads, open(MANIFEST)):
        if mm["song_id"] not in man or mm.get("transpose", 0) == 0:
            man[mm["song_id"]] = mm

    sid = next((s for s in recs if song_q.lower() in recs[s]["title"].lower()), None)
    if sid is None:
        print(f"Song '{song_q}' not found"); sys.exit(1)
    rec = recs[sid]
    m   = man[sid]
    print(f"Song: {rec['title']}  tempo={m['tempo']}  bpb={m['beats_per_bar']}")

    # ── render hard audio ────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    pm = pretty_midi.PrettyMIDI(str(REPO / m["midi_path"]))
    scen = str(rng.choice(list(SCENARIOS)))
    gains = {k: v * float(rng.uniform(0.8, 1.2)) for k, v in SCENARIOS[scen].items()}
    sf_name = SOUNDFONTS[int(rng.integers(0, len(SOUNDFONTS)))]
    stems = {
        "chords": stem_midi(pm, lambda i: (not i.is_drum) and "bass" not in i.name.lower()),
        "bass":   stem_midi(pm, lambda i: "bass" in i.name.lower()),
        "drums":  stem_midi(pm, lambda i: i.is_drum),
    }
    waves, sr = {}, 44100
    for name, s in stems.items():
        if s and s.instruments:
            w, sr2 = render_to_array(renderer, s, sf_name, False)
            waves[name] = w; sr = sr2
    L = max(len(w) for w in waves.values())
    mix = np.zeros(L, np.float32)
    for name, w in waves.items(): mix[:len(w)] += gains.get(name, 0.5) * w
    mix = time_varying_degrade(mix, sr, rng)
    peak = np.abs(mix).max()
    if peak > 0.99: mix *= 0.99 / peak

    # ── beat tracking — perfect grid anchored on first detected beat ──────────
    bpb = m["beats_per_bar"]
    spb = 60.0 / m["tempo"]
    _, bf = librosa.beat.beat_track(y=mix.astype(float), sr=sr,
                                    bpm=float(m["tempo"]), units="frames")
    raw_bt = librosa.frames_to_time(bf, sr=sr)
    t0_phase = float(raw_bt[0])
    n_beats_total = int((len(mix) / sr - t0_phase) / spb) + 1
    bt = np.array([t0_phase + i * spb for i in range(n_beats_total)])

    # time range to plot: first n_bars bars
    t_end_plot = t0_phase + n_bars * bpb * spb
    print(f"Plotting 0 – {t_end_plot:.2f}s  ({n_bars} bars, {bpb} beats/bar)")

    # ── CQT chroma — LTAS-normalised ─────────────────────────────────────────
    hop = 512
    y_f = mix.astype(float)
    chroma_raw = librosa.feature.chroma_cqt(y=y_f, sr=sr, bins_per_octave=36, hop_length=hop)
    ltas = chroma_raw.mean(axis=1, keepdims=True)
    ltas = np.where(ltas < 1e-9, 1.0, ltas)
    chroma = chroma_raw / ltas   # LTAS-normalised: each PC has mean ≈ 1
    ct = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)

    # ── Basic Pitch activations ───────────────────────────────────────────────
    ex = PitchExtractor(cache_dir=REPO / "data" / "cache" / "debug_seg")
    tmp = Path(tempfile.mktemp(suffix=".wav"))
    try:
        sf.write(tmp, mix, sr)
        acts = ex.extract(tmp, use_cache=False)
    finally:
        tmp.unlink(missing_ok=True)

    onset_b = pool_to_beats(acts.frame_times, acts.onset_probs, bt)
    note_b  = pool_to_beats(acts.frame_times, acts.note_probs,  bt)

    # ── segmentation: perfect beat grid, window-pooled, dual trigger ─────────
    grid_step       = max(1, bpb // 2)
    bass_nov_thresh = 0.12
    cqt_nov_thresh  = 0.15

    n_beats = len(onset_b)

    def _pool_bass(b0, b1):
        chunk = onset_b[max(0,b0):min(b1,n_beats)].sum(0)
        return _unit(_reg(chunk, 0, 52))

    def _pool_full(b0, b1):
        chunk = onset_b[max(0,b0):min(b1,n_beats)].sum(0)
        return _unit(_reg(chunk, 0, 200))

    def _pool_treb(b0, b1):
        chunk = onset_b[max(0,b0):min(b1,n_beats)].sum(0)
        return _unit(_reg(chunk, 60, 200))

    # Raw L2-normalised CQT (no LTAS) — for A/B comparison
    chroma_l2 = chroma_raw / (np.linalg.norm(chroma_raw, axis=0, keepdims=True) + 1e-9)

    def _pool_cqt(b0, b1, arr):
        t0c = float(bt[max(0, b0)])
        t1c = float(bt[min(b1, len(bt)-1)])
        i0 = int(np.searchsorted(ct, t0c))
        i1 = int(np.searchsorted(ct, t1c))
        if i1 <= i0: i1 = i0 + 1
        c = arr[:, i0:i1].mean(axis=1)
        n = np.linalg.norm(c); return c / n if n > 1e-9 else c

    # novelty_full: one entry per grid position (for plotting)
    novelty_full = []
    for b in range(grid_step, n_beats, grid_step):
        prev_bass = _pool_bass(b - grid_step, b)
        curr_bass = _pool_bass(b, b + grid_step)
        prev_full = _pool_full(b - grid_step, b)
        curr_full = _pool_full(b, b + grid_step)
        prev_treb = _pool_treb(b - grid_step, b)
        curr_treb = _pool_treb(b, b + grid_step)
        prev_cqt_ltas = _pool_cqt(b - grid_step, b, chroma)
        curr_cqt_ltas = _pool_cqt(b, b + grid_step, chroma)
        prev_cqt_l2   = _pool_cqt(b - grid_step, b, chroma_l2)
        curr_cqt_l2   = _pool_cqt(b, b + grid_step, chroma_l2)
        bass_nov     = 1 - float(prev_bass @ curr_bass)
        full_nov     = 1 - float(prev_full @ curr_full)
        treb_nov     = 1 - float(prev_treb @ curr_treb)
        cqt_ltas_nov = 1 - float(prev_cqt_ltas @ curr_cqt_ltas)
        cqt_l2_nov   = 1 - float(prev_cqt_l2   @ curr_cqt_l2)
        novelty_full.append((b, full_nov, bass_nov, treb_nov, cqt_ltas_nov, cqt_l2_nov))

    boundaries = [0]
    for b, full_nov, bass_nov, treb_nov, cqt_ltas_nov, cqt_l2_nov in novelty_full:
        if bass_nov > bass_nov_thresh or cqt_l2_nov > cqt_nov_thresh:
            boundaries.append(b)
    boundaries.append(n_beats)

    # convert boundaries to time
    cut_times = [float(bt[min(b, len(bt) - 1)]) for b in boundaries]
    cut_times_plot = [t for t in cut_times if t <= t_end_plot + spb]

    # ── GT chord boundaries ───────────────────────────────────────────────────
    chord_at = {(e["bar"] - 1) * bpb + e["beat"]: e for e in rec["chord_timeline"]}
    gt_cuts = []
    for t0, t1, root, _q in song_chord_spans(rec):
        if t0 <= t_end_plot + spb:
            b0 = int(round(t0 / spb))
            mma = chord_at.get(b0, {}).get("mma")
            p = parse_chord(mma) if mma else None
            fam = BUCKET_FAMILY.get(p[1]) if p else None
            suf = FAM_SUFFIX.get(fam, "")
            gt_cuts.append((t0, NOTE[root % 12] + suf))

    # ── novelty curve (beat times within plot window) ─────────────────────────
    nov_times        = [float(bt[min(b, len(bt)-1)]) for b, *_ in novelty_full]
    nov_vals         = [v[1] for v in novelty_full]
    bass_nov_vals    = [v[2] for v in novelty_full]
    treb_nov_vals    = [v[3] for v in novelty_full]
    cqt_ltas_vals    = [v[4] for v in novelty_full]
    cqt_l2_vals      = [v[5] for v in novelty_full]

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(16, 11),
                             gridspec_kw={"height_ratios": [3, 3, 1.4, 1.2]},
                             facecolor="#0d1520")
    fig.suptitle(f"{rec['title']} — bars 1–{n_bars}  (seed {seed}, {scen})",
                 color="#e2e8f0", fontsize=13, y=0.995)

    mask = ct <= t_end_plot
    ct_plot = ct[mask]

    # pre-compute inferred segment spans + root labels (shared across both heatmaps)
    seg_cuts_in_plot = [cut_times_plot[0]] + [t for t in cut_times_plot[1:-1]
                                               if t <= t_end_plot] + [t_end_plot]
    shade_colors = ["#ffffff18", "#58d4ff18"]
    seg_roots = []
    for si in range(len(seg_cuts_in_plot) - 1):
        ta, tb = seg_cuts_in_plot[si], seg_cuts_in_plot[si + 1]
        # LTAS-CQT mean → best 60-template chord token
        i0_s = int(np.searchsorted(ct, ta))
        i1_s = int(np.searchsorted(ct, tb))
        if i1_s <= i0_s: i1_s = i0_s + 1
        seg_chroma = chroma[:, i0_s:i1_s].mean(axis=1)
        cn = np.linalg.norm(seg_chroma)
        if cn > 1e-9:
            seg_chroma = seg_chroma / cn
            best_r, best_fn, _ = max(_TEMPLATES, key=lambda x: float(seg_chroma @ x[2]))
            inf_label = NOTE[best_r] + FAM_SUFFIX[best_fn]
        else:
            inf_label = "?"
        seg_roots.append((ta, tb, inf_label, shade_colors[si % 2]))

    def _decorate_heatmap(ax, chroma_arr, title, cmap="YlOrRd"):
        """Draw a 12×T heatmap with GT cuts, inferred cuts, beat grid, segment labels."""
        cp = chroma_arr[:, mask]
        ax.set_facecolor("#0d1520")
        ax.imshow(cp, aspect="auto", origin="lower",
                  extent=[ct_plot[0], ct_plot[-1], -0.5, 11.5],
                  cmap=cmap, vmin=0, vmax=cp.max() * 0.85)
        for bt_i in bt:
            if bt_i > t_end_plot: break
            ax.axvline(bt_i, color="#ffffff22", lw=0.5)
        for bar in range(n_bars + 1):
            t_bar = bar * bpb * spb
            ax.axvline(t_bar, color="#ffffff55", lw=1.2)
            if bar < n_bars:
                ax.text(t_bar + 0.02, 11.6, f"bar {bar+1}", color="#88aacc",
                        fontsize=7.5, va="bottom")
        for t0, root_name in gt_cuts:
            ax.axvline(t0, color="#1baf7a", lw=1.2, linestyle="--", alpha=0.8)
            ax.text(t0 + 0.02, -0.4, root_name, color="#1baf7a", fontsize=7, va="top")
        for ta, tb, inf_label, shade in seg_roots:
            ax.axvspan(ta, tb, ymin=0, ymax=1, color=shade, zorder=0)
            if inf_label and inf_label != "?" and (tb - ta) > 0.15:
                ax.text((ta + tb) / 2, 11.0, inf_label, color="#ffffff",
                        fontsize=7.5, ha="center", va="top", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", fc="#1a2535cc", ec="none"))
        for t in cut_times_plot[1:-1]:
            ax.axvline(t, color="#e34948", lw=1.5, alpha=0.9)
        ax.set_xlim(0, t_end_plot); ax.set_ylim(-0.5, 11.5)
        ax.set_yticks(range(12)); ax.set_yticklabels(NOTE, fontsize=8, color="#88aacc")
        ax.tick_params(axis="x", colors="#5a6a7e", labelsize=8)
        ax.set_title(title, color="#88aacc", fontsize=8, pad=3)

    # ── panel 1: LTAS-normalised CQT ─────────────────────────────────────────
    _decorate_heatmap(axes[0], chroma,
                      "CQT chroma — LTAS-normalised (÷ row mean: equalises long-term energy per PC)"
                      "   ── green = GT   ─ red = inferred cut", cmap="YlOrRd")
    axes[0].set_xlabel("")

    # ── panel 2: raw L2-normalised CQT ───────────────────────────────────────
    _decorate_heatmap(axes[1], chroma_l2,
                      "CQT chroma — raw L2-normalised (column vectors unit-norm: dominated by bass energy)",
                      cmap="magma")
    axes[1].set_xlabel("time (s)", color="#5a6a7e", fontsize=8)

    # ── panel 3: bass + treble novelty ───────────────────────────────────────
    ax2 = axes[2]
    ax2.set_facecolor("#0d1520")
    nt_b = [(t, v) for t, v in zip(nov_times, bass_nov_vals) if t <= t_end_plot]
    nt_t = [(t, v) for t, v in zip(nov_times, treb_nov_vals) if t <= t_end_plot]
    if nt_b:
        ts, vs = zip(*nt_b)
        ax2.plot(ts, vs, "o-", color="#a65fd4", lw=1.2, ms=5, label="bass (<52)")
    if nt_t:
        ts, vs = zip(*nt_t)
        ax2.plot(ts, vs, "s-", color="#e07a3b", lw=1.2, ms=4, label="treble BP (>60)", alpha=0.85)
    nt_cl = [(t, v) for t, v in zip(nov_times, cqt_ltas_vals) if t <= t_end_plot]
    nt_c2 = [(t, v) for t, v in zip(nov_times, cqt_l2_vals)   if t <= t_end_plot]
    if nt_cl:
        ts, vs = zip(*nt_cl)
        ax2.plot(ts, vs, "^-", color="#58d4ff", lw=1.2, ms=4, label="CQT LTAS", alpha=0.85)
    if nt_c2:
        ts, vs = zip(*nt_c2)
        ax2.plot(ts, vs, "v-", color="#1baf7a", lw=1.2, ms=4, label="CQT L2", alpha=0.85)
    ax2.axhline(bass_nov_thresh, color="#a65fd4", lw=0.8, linestyle=":", alpha=0.6)
    ax2.axhline(cqt_nov_thresh,  color="#1baf7a", lw=0.8, linestyle=":", alpha=0.6)
    for t in cut_times_plot[1:-1]:
        ax2.axvline(t, color="#e34948", lw=1.0, alpha=0.7)
    for t_bar in [bar * bpb * spb for bar in range(n_bars + 1)]:
        ax2.axvline(t_bar, color="#ffffff22", lw=0.8)
    ax2.set_xlim(0, t_end_plot); ax2.set_ylim(0, 1)
    ax2.set_ylabel("novelty", color="#5a6a7e", fontsize=8)
    ax2.tick_params(colors="#5a6a7e", labelsize=7)
    ax2.legend(fontsize=7, facecolor="#111a26", edgecolor="#253447", labelcolor="#c8d0dc", loc="upper right")
    ax2.set_title("Bass + treble novelty (beat b vs beat b−1)  — triggers cut when either > threshold",
                  color="#88aacc", fontsize=8, pad=2)

    # ── panel 4: full-register novelty (reference) ───────────────────────────
    ax3 = axes[3]
    ax3.set_facecolor("#0d1520")
    nt_f = [(t, v) for t, v in zip(nov_times, nov_vals) if t <= t_end_plot]
    if nt_f:
        ts, vs = zip(*nt_f)
        ax3.plot(ts, vs, "o-", color="#3b8de0", lw=1.2, ms=5, label="full (0-200)")
    ax3.axhline(0.22, color="#3b8de0", lw=0.8, linestyle=":", alpha=0.6)
    for t in cut_times_plot[1:-1]:
        ax3.axvline(t, color="#e34948", lw=1.0, alpha=0.7)
    for t_bar in [bar * bpb * spb for bar in range(n_bars + 1)]:
        ax3.axvline(t_bar, color="#ffffff22", lw=0.8)
    ax3.set_xlim(0, t_end_plot); ax3.set_ylim(0, 1)
    ax3.set_ylabel("novelty", color="#5a6a7e", fontsize=8)
    ax3.set_xlabel("time (s)", color="#5a6a7e", fontsize=8)
    ax3.tick_params(colors="#5a6a7e", labelsize=7)
    ax3.set_title("Full-register novelty (reference — washed out by multi-stem noise)",
                  color="#88aacc", fontsize=8, pad=2)

    legend_items = [
        mpatches.Patch(color="#1baf7a", label="GT chord boundary"),
        mpatches.Patch(color="#e34948", label="Inferred cut"),
        mpatches.Patch(color="#ffffff33", label="Beat / barline"),
    ]
    fig.legend(handles=legend_items, loc="lower right", fontsize=8,
               facecolor="#111a26", edgecolor="#253447", labelcolor="#c8d0dc",
               ncol=3, bbox_to_anchor=(0.98, 0.01))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"→ {out_path}")
    print(f"  GT cuts in window : {len([t for t,_ in gt_cuts])}")
    print(f"  Inferred cuts     : {len(cut_times_plot) - 2}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--song",  default="Anthropology")
    ap.add_argument("--bars",  type=int, default=4)
    ap.add_argument("--seed",  type=int, default=42)
    ap.add_argument("--out",   default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else \
          REPO / "docs" / "plots" / f"seg_debug_{args.song.replace(' ','_').lower()}.png"
    run(args.song, args.bars, args.seed, out)
