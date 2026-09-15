#!/usr/bin/env python3
"""Fit a perfect constant-tempo beat grid from hand-corrected chord onsets.

Given a small set of *trusted* (hand-aligned) chord onsets, fit a single rigid
metric grid  t = anchor + slope * m , where `m` is the chord's position in
*bars* (bar index + sub-bar offset) and `slope` is the seconds-per-bar duration.
A robust (Huber-IRLS) fit downweights outliers so a couple of shaky hand
corrections don't tilt the whole line. The fitted grid is then evaluated for
*every* chord in the chart, producing `t0_perfect` for all bars.

Why a rigid grid: the DTW starting alignment drifts. The user hand-corrects a
handful of bars they can hear clearly; those anchor a constant tempo, and the
rest of the chart is snapped onto that tempo. The play-along page then lets the
user hear whether the constant-tempo assumption actually holds for the whole
song (see `--diag-html`).

This is intentionally song-agnostic: pass any gt-align annotation JSON and a
`--fit-max-bar` (how many leading bars were hand-corrected).

CLI:
    python scripts/fit_beat_grid.py \
        --annot docs/plots/annotations/irealb_autumn_leaves.html.json \
        --bpm-prior 140 --fit-max-bar 7 \
        --out docs/plots/annotations/irealb_autumn_leaves_perfectgrid.json \
        --diag-html docs/plots/beat_grid_fit_autumn_leaves.html
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Core algorithm
# ─────────────────────────────────────────────────────────────────────────────
def metric_positions(chords, beats_per_bar=4):
    """Position (in bars) of each chord: bar index + a sub-bar offset.

    The gt-align `beat` field is an *ordinal* within the bar (0,1,2…), not a
    beat number. When a bar holds k chords we spread them evenly across the bar
    (chord j at fraction j/k) rather than trusting the raw ordinal — for a
    two-chord ii-V bar that puts the second chord at the half-bar, which is the
    musically correct place. Single-chord bars sit on the downbeat (offset 0).
    """
    per_bar = Counter(c["bar"] for c in chords)
    # ordinal of each chord within its bar, in list order
    seen: dict[int, int] = {}
    m = []
    for c in chords:
        bar = c["bar"]
        j = seen.get(bar, 0)
        seen[bar] = j + 1
        k = per_bar[bar]
        m.append(bar + j / k)
    return np.asarray(m, dtype=float)


def robust_linfit(x, y, delta=None, iters=25, tol=1e-9):
    """Huber iteratively-reweighted least squares:  y ≈ a + b·x.

    Returns (a, b, residuals, weights). `delta` (the Huber transition point) is
    auto-set to 1.345·(robust MAD scale) each iteration if not given, so points
    within ~1σ get full weight and outliers are downweighted ∝ 1/|residual|.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    A = np.vstack([np.ones_like(x), x]).T
    w = np.ones_like(x)
    coef = np.array([y.mean() if len(y) else 0.0, 0.0])
    for _ in range(iters):
        sw = np.sqrt(w)
        coef_new, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
        r = y - A @ coef_new
        mad = np.median(np.abs(r - np.median(r)))
        s = 1.4826 * mad + 1e-9
        d = delta if delta else 1.345 * s
        a = np.abs(r)
        w = np.where(a <= d, 1.0, d / np.maximum(a, 1e-9))
        if np.allclose(coef_new, coef, atol=tol):
            coef = coef_new
            break
        coef = coef_new
    a, b = float(coef[0]), float(coef[1])
    return a, b, (y - A @ coef), w


def fit_beat_grid(chords, bpm_prior, beats_per_bar=4, fit_max_bar=None,
                  fit_bars=None):
    """Fit a constant-tempo grid from the trusted (hand-corrected) chords.

    Parameters
    ----------
    chords : list of dicts with keys bar, beat, label, t0 (seconds)
    bpm_prior : starting/nominal BPM (used only for validation reporting)
    fit_max_bar : use chords with bar <= this as the trusted fit set
    fit_bars : explicit set/list of bar indices to fit on (overrides fit_max_bar)

    Returns a dict with anchor, slope, bpm_fit, per-chord perfect times, the
    beat/downbeat grid, and diagnostics (residuals, fit mask).
    """
    m_all = metric_positions(chords, beats_per_bar)

    if fit_bars is not None:
        fit_set = set(fit_bars)
        mask = np.array([c["bar"] in fit_set for c in chords])
    elif fit_max_bar is not None:
        mask = np.array([c["bar"] <= fit_max_bar for c in chords])
    else:
        mask = np.ones(len(chords), bool)
    # only downbeat chords anchor the fit (sub-bar offset 0) — cleaner
    mask = mask & (np.abs(m_all - np.round(m_all)) < 1e-9)

    xf, yf = m_all[mask], np.array([c["t0"] for c in chords])[mask]
    if len(xf) < 2:
        raise ValueError(f"need >=2 downbeat fit points, got {len(xf)}")

    anchor, slope, resid_fit, weights = robust_linfit(xf, yf)
    beat_dur = slope / beats_per_bar
    bpm_fit = 60.0 / beat_dur if beat_dur > 0 else float("nan")

    # evaluate the perfect grid for every chord
    t0_perfect = anchor + slope * m_all
    order = np.argsort(m_all)
    t1_perfect = np.empty_like(t0_perfect)
    for rank, idx in enumerate(order):
        nxt = order[rank + 1] if rank + 1 < len(order) else None
        t1_perfect[idx] = t0_perfect[nxt] if nxt is not None else t0_perfect[idx] + slope

    out_chords = []
    for i, c in enumerate(chords):
        out_chords.append({
            **c,
            "m": round(float(m_all[i]), 4),
            "t0_orig": c["t0"],
            "t1_orig": c.get("t1", c["t0"]),
            "t0_perfect": round(float(t0_perfect[i]), 4),
            "t1_perfect": round(float(t1_perfect[i]), 4),
            "residual": round(float(c["t0"] - t0_perfect[i]), 4),
            "in_fit": bool(mask[i]),
        })

    # full beat/downbeat grid across the chart span
    max_bar = int(max(c["bar"] for c in chords))
    beats, downbeats = [], []
    for bar in range(0, max_bar + 2):
        for b in range(beats_per_bar):
            t = anchor + slope * (bar + b / beats_per_bar)
            if t < 0:
                continue
            beats.append(round(float(t), 4))
            if b == 0:
                downbeats.append(round(float(t), 4))

    # ── validation ───────────────────────────────────────────────────────
    resid_all = np.array([c["t0"] - t for c, t in zip(chords, t0_perfect)])
    bpm_err_pct = 100.0 * (bpm_fit - bpm_prior) / bpm_prior if bpm_prior else float("nan")
    strictly_monotonic = bool(np.all(np.diff(np.sort(t0_perfect)) > 0))

    return {
        "anchor": round(anchor, 5),
        "slope_s_per_bar": round(slope, 5),
        "beat_dur_s": round(beat_dur, 5),
        "bpm_fit": round(bpm_fit, 3),
        "bpm_prior": bpm_prior,
        "bpm_err_pct": round(bpm_err_pct, 2),
        "beats_per_bar": beats_per_bar,
        "chords": out_chords,
        "beats": beats,
        "downbeats": downbeats,
        "validation": {
            "n_fit_points": int(mask.sum()),
            "fit_resid_rms_s": round(float(np.sqrt(np.mean(resid_fit ** 2))), 4),
            "fit_resid_max_s": round(float(np.max(np.abs(resid_fit))), 4),
            "all_resid_rms_s": round(float(np.sqrt(np.mean(resid_all ** 2))), 4),
            "all_resid_max_s": round(float(np.max(np.abs(resid_all))), 4),
            "strictly_monotonic": strictly_monotonic,
            "weights_fit": [round(float(w), 3) for w in weights],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic HTML
# ─────────────────────────────────────────────────────────────────────────────
def diagnostic_html(result, title):
    v = result["validation"]
    ch = result["chords"]
    xs = [c["m"] for c in ch]
    y_orig = [c["t0_orig"] for c in ch]
    y_perf = [c["t0_perfect"] for c in ch]
    resid = [c["residual"] for c in ch]
    infit = [c["in_fit"] for c in ch]
    labels = [c["label"] for c in ch]

    payload = json.dumps({
        "xs": xs, "y_orig": y_orig, "y_perf": y_perf, "resid": resid,
        "infit": infit, "labels": labels,
        "anchor": result["anchor"], "slope": result["slope_s_per_bar"],
    })

    rows = "".join(
        f"<tr class='{'fit' if c['in_fit'] else ''}'>"
        f"<td>{c['bar']}.{c['beat']}</td><td>{c['label']}</td>"
        f"<td>{c['t0_orig']:.3f}</td><td>{c['t0_perfect']:.3f}</td>"
        f"<td class='{'big' if abs(c['residual'])>0.5 else ''}'>{c['residual']:+.3f}</td></tr>"
        for c in ch
    )

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Beat-grid fit — {title}</title>
<style>
 body{{margin:0;font:14px system-ui;background:#0e1116;color:#e8edf4;padding:24px}}
 h1{{font-size:20px}} h2{{font-size:15px;color:#9fb3c8;margin-top:28px}}
 .kpis{{display:flex;gap:16px;flex-wrap:wrap;margin:16px 0}}
 .kpi{{background:#171c24;border:1px solid #2a3340;border-radius:8px;padding:12px 16px;min-width:120px}}
 .kpi b{{display:block;font-size:22px;color:#00c9a7}} .kpi span{{font-size:12px;color:#8b97a8}}
 .warn b{{color:#ffb454}}
 canvas{{background:#171c24;border:1px solid #2a3340;border-radius:8px;max-width:100%}}
 table{{border-collapse:collapse;margin-top:12px;font-size:12px}}
 td,th{{padding:3px 10px;border-bottom:1px solid #222b36;text-align:right}}
 tr.fit{{background:rgba(0,201,167,0.08)}} td.big{{color:#ff6b6b;font-weight:700}}
 .note{{background:#1e2530;border-left:3px solid #ffb454;padding:10px 14px;border-radius:4px;color:#cdd8e6;max-width:760px}}
</style></head><body>
<h1>🎚️ Perfect beat-grid fit — {title}</h1>
<div class="kpis">
  <div class="kpi"><b>{result['bpm_fit']}</b><span>fitted BPM</span></div>
  <div class="kpi"><b>{result['bpm_prior']}</b><span>prior BPM</span></div>
  <div class="kpi warn"><b>{result['bpm_err_pct']:+.1f}%</b><span>vs prior</span></div>
  <div class="kpi"><b>{result['slope_s_per_bar']}s</b><span>per bar</span></div>
  <div class="kpi"><b>{v['fit_resid_rms_s']}s</b><span>fit RMS resid</span></div>
  <div class="kpi warn"><b>{v['all_resid_rms_s']}s</b><span>all-chords RMS resid</span></div>
</div>
<div class="note">
 The grid is fit on <b>{v['n_fit_points']} hand-corrected downbeats</b> (RMS residual
 {v['fit_resid_rms_s']}s — the line fits them tightly). The <b>all-chords RMS residual
 of {v['all_resid_rms_s']}s</b> is how far the DTW times for the <i>uncorrected</i> bars
 sit from this constant tempo. A large value means the constant-tempo-from-head
 assumption does not extend to the whole song — press play in the play-along to
 hear where it drifts, then hand-correct more bars and refit.
 Strictly monotonic: <b>{v['strictly_monotonic']}</b>.
</div>

<h2>Onset time vs metric position (bars)</h2>
<canvas id="scatter" width="900" height="360"></canvas>
<h2>Residual (DTW − perfect grid) per chord</h2>
<canvas id="resid" width="900" height="240"></canvas>

<h2>Per-chord table (teal = in fit set)</h2>
<table><tr><th>bar.beat</th><th>label</th><th>t0 orig</th><th>t0 perfect</th><th>resid</th></tr>
{rows}</table>

<script>
const D = {payload};
function ax(ctx,w,h,pad){{ctx.strokeStyle='#3a4658';ctx.lineWidth=1;
 ctx.beginPath();ctx.moveTo(pad,h-pad);ctx.lineTo(w-8,h-pad);ctx.moveTo(pad,h-pad);ctx.lineTo(pad,8);ctx.stroke();}}
// scatter
(function(){{const c=document.getElementById('scatter'),ctx=c.getContext('2d'),w=c.width,h=c.height,pad=44;
 const xmax=Math.max(...D.xs)+1, ymax=Math.max(...D.y_orig,...D.y_perf)*1.05;
 const X=x=>pad+x/xmax*(w-pad-12), Y=y=>h-pad-y/ymax*(h-pad-12);
 ax(ctx,w,h,pad);
 // perfect line
 ctx.strokeStyle='#6ea8ff';ctx.lineWidth=2;ctx.beginPath();
 ctx.moveTo(X(0),Y(D.anchor));ctx.lineTo(X(xmax),Y(D.anchor+D.slope*xmax));ctx.stroke();
 // orig points
 for(let i=0;i<D.xs.length;i++){{ctx.fillStyle=D.infit[i]?'#00c9a7':'#ff8c42';
   ctx.beginPath();ctx.arc(X(D.xs[i]),Y(D.y_orig[i]),D.infit[i]?5:3,0,7);ctx.fill();}}
 ctx.fillStyle='#8b97a8';ctx.font='11px system-ui';
 ctx.fillText('bar (metric position) →',w/2-60,h-14);ctx.save();ctx.translate(14,h/2);ctx.rotate(-Math.PI/2);ctx.fillText('t0 (s) — orig=dots, line=perfect',0,0);ctx.restore();
 ctx.fillStyle='#00c9a7';ctx.fillText('● in-fit (hand corrected)',pad+8,20);
 ctx.fillStyle='#ff8c42';ctx.fillText('● DTW (uncorrected)',pad+8,36);
 ctx.fillStyle='#6ea8ff';ctx.fillText('— perfect grid',pad+8,52);
}})();
// residual
(function(){{const c=document.getElementById('resid'),ctx=c.getContext('2d'),w=c.width,h=c.height,pad=44;
 const xmax=Math.max(...D.xs)+1, rmax=Math.max(...D.resid.map(Math.abs),0.5)*1.1;
 const X=x=>pad+x/xmax*(w-pad-12), Y=r=>h/2 - r/rmax*(h/2-16);
 ctx.strokeStyle='#3a4658';ctx.beginPath();ctx.moveTo(pad,h/2);ctx.lineTo(w-8,h/2);ctx.stroke();
 for(let i=0;i<D.xs.length;i++){{const x=X(D.xs[i]);ctx.strokeStyle=D.infit[i]?'#00c9a7':'#ff8c42';ctx.lineWidth=2;
   ctx.beginPath();ctx.moveTo(x,h/2);ctx.lineTo(x,Y(D.resid[i]));ctx.stroke();
   ctx.fillStyle=D.infit[i]?'#00c9a7':'#ff8c42';ctx.beginPath();ctx.arc(x,Y(D.resid[i]),3,0,7);ctx.fill();}}
 ctx.fillStyle='#8b97a8';ctx.font='11px system-ui';ctx.fillText('bar →',w/2-16,h-10);
 ctx.fillText('+'+rmax.toFixed(1)+'s',6,18);ctx.fillText('-'+rmax.toFixed(1)+'s',6,h-8);
}})();
</script>
</body></html>"""


# ─────────────────────────────────────────────────────────────────────────────
def load_chords(annot_path):
    d = json.loads(Path(annot_path).read_text())
    return d.get("chords", []), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annot", required=True)
    ap.add_argument("--bpm-prior", type=float, default=140.0)
    ap.add_argument("--fit-max-bar", type=int, default=7)
    ap.add_argument("--beats-per-bar", type=int, default=4)
    ap.add_argument("--out")
    ap.add_argument("--diag-html")
    args = ap.parse_args()

    chords, meta = load_chords(args.annot)
    result = fit_beat_grid(chords, args.bpm_prior,
                           beats_per_bar=args.beats_per_bar,
                           fit_max_bar=args.fit_max_bar)

    v = result["validation"]
    print(f"fitted BPM      : {result['bpm_fit']}  (prior {result['bpm_prior']}, "
          f"{result['bpm_err_pct']:+.1f}%)")
    print(f"slope / anchor  : {result['slope_s_per_bar']} s/bar @ {result['anchor']} s")
    print(f"fit points      : {v['n_fit_points']}  RMS={v['fit_resid_rms_s']}s  "
          f"max={v['fit_resid_max_s']}s")
    print(f"all-chords resid: RMS={v['all_resid_rms_s']}s  max={v['all_resid_max_s']}s")
    print(f"monotonic grid  : {v['strictly_monotonic']}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "source_annot": str(args.annot),
            "bpm_prior": args.bpm_prior,
            "fit_max_bar": args.fit_max_bar,
            **{k: result[k] for k in
               ("anchor", "slope_s_per_bar", "beat_dur_s", "bpm_fit",
                "bpm_err_pct", "beats_per_bar", "chords", "beats",
                "downbeats", "validation")},
        }, indent=2))
        print(f"wrote {args.out}")

    if args.diag_html:
        title = Path(args.annot).stem.replace("irealb_", "").replace(".html", "")
        Path(args.diag_html).write_text(diagnostic_html(result, title))
        print(f"wrote {args.diag_html}")


if __name__ == "__main__":
    main()
