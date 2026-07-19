"""Visual verdict: stock vs bestfit grid vs madmom beats — Commodores 'Easy'.

No listening required: if the stock grid's bar lines drift off madmom's
independently-detected beats by the end of the song while bestfit stays on
them, the fix is confirmed visually.

Run: .venv/bin/python scratchpad/beatgrid_visual_verdict.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harmonia.models.chord_pipeline_v1 import _bestfit_beat_period  # noqa: E402

AUDIO = REPO / "docs" / "audio" / "the_commodores_easy_1977.m4a"
OUT = REPO / "docs" / "plots" / "beatgrid_verdict_commodores.png"

# identity colors (fixed): reference=near-black ticks, stock=orange, bestfit=blue
C_REF, C_STOCK, C_BEST, C_WAVE = "#1a1a2e", "#efb118", "#4269d0", "#c8c8d0"


def main():
    import librosa

    from harmonia.models.rhythm import _ensure_madmom_compat

    y, sr = librosa.load(str(AUDIO), sr=22050, mono=True)
    dur = len(y) / sr
    tempo_arr, frames = librosa.beat.beat_track(y=y, sr=sr)
    p_stock = 60.0 / float(np.atleast_1d(tempo_arr)[0])
    lb = librosa.frames_to_time(frames, sr=sr)
    p_best = _bestfit_beat_period(lb, p_stock)

    # shared phase (exactly the pipeline's circular-mean, per mode)
    def grid(period):
        ang = 2 * np.pi * (lb % period) / period
        phase = (np.angle(np.mean(np.exp(1j * ang))) % (2 * np.pi)) * period / (2 * np.pi)
        return np.arange(phase, dur, period)

    g_stock, g_best = grid(p_stock), grid(p_best)

    _ensure_madmom_compat()
    from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor

    act = RNNBeatProcessor()(str(AUDIO))
    mb = np.asarray(DBNBeatTrackingProcessor(fps=100)(act))
    # madmom tracks the half-tempo octave here — its beats align with every
    # 2nd grid beat; that's fine, we compare *alignment*, not tempo octave.

    windows = [(20, 28, "start of song (t=20 s)"), (150, 158, "middle (t=150 s)"),
               (dur - 30, dur - 22, "end of song")]
    fig, axes = plt.subplots(len(windows), 1, figsize=(11, 7.2), dpi=150)
    fig.patch.set_facecolor("white")

    for ax, (t0, t1, title) in zip(axes, windows):
        s0, s1 = int(t0 * sr), int(t1 * sr)
        tt = np.linspace(t0, t1, s1 - s0)
        env = y[s0:s1]
        ax.plot(tt, env, color=C_WAVE, lw=0.4, zorder=1)
        for t in mb[(mb >= t0) & (mb <= t1)]:
            ax.axvline(t, color=C_REF, lw=1.6, ymin=0.82, ymax=1.0, zorder=4)
        for t in g_stock[(g_stock >= t0) & (g_stock <= t1)]:
            ax.axvline(t, color=C_STOCK, lw=1.4, ymin=0.0, ymax=0.38, zorder=3)
        for t in g_best[(g_best >= t0) & (g_best <= t1)]:
            ax.axvline(t, color=C_BEST, lw=1.4, ymin=0.41, ymax=0.79, zorder=3)
        ax.set_title(title, fontsize=10, loc="left", color="#333")
        ax.set_yticks([])
        ax.set_xlim(t0, t1)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#ddd")
        ax.tick_params(colors="#888", labelsize=8)

    from matplotlib.lines import Line2D

    fig.legend(handles=[
        Line2D([], [], color=C_REF, lw=2, label="madmom beats (independent reference)"),
        Line2D([], [], color=C_BEST, lw=2, label=f"bestfit grid ({60/p_best:.1f} bpm)"),
        Line2D([], [], color=C_STOCK, lw=2, label=f"stock grid ({60/p_stock:.1f} bpm)"),
    ], loc="upper right", fontsize=9, frameon=False)
    fig.suptitle("Commodores 'Easy' — beat-grid drift: stock vs bestfit vs madmom",
                 fontsize=12, x=0.01, ha="left")
    fig.text(0.01, 0.945, "aligned at the start; by the end the stock (orange) lines "
             "have slipped off the reference — bestfit (blue) stays on it",
             fontsize=9, color="#555")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT, f"p_stock={p_stock*1e3:.2f}ms p_best={p_best*1e3:.2f}ms")


if __name__ == "__main__":
    main()
