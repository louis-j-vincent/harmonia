"""make_leakage_robust_artifact.py — render the figure + HTML writeup for the
leakage-robust chord-identity screen (2026-07-24). Reads the two saved npz runs
(no re-computation) and writes:
  docs/research_sessions/leakage_robust_identity.png
  docs/research_sessions/leakage_robust_identity.html   (self-contained)

Run scripts/exp_leakage_robust_identity.py and
scripts/exp_leakage_robust_realaudio.py first.
"""
from __future__ import annotations

import base64
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "docs" / "research_sessions"

# dataviz categorical slots (validated trio, light surface #fcfcfb):
C_V0 = "#2a78d6"   # blue   — V0 mean-pool (the baseline / current regime)
C_V1 = "#eb6834"   # orange — V1 clip-pool (repo already has this)
C_V2 = "#4a3aa7"   # violet — V2 root-normalised directional delta (the new idea)
INK = "#0b0b0b"; MUTED = "#898781"; GRID = "#e1e0d9"; SURF = "#fcfcfb"


def make_figure(midi, real):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = midi["strengths"]; W = midi["windows"]
    wi06 = int(np.argmin(np.abs(W - 0.06)))
    wi_long = int(np.argmax(W))
    I_v0 = midi["I_v0"]; I_v1 = midi["I_v1"]; I_v2 = midi["I_v2"]

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 11, "axes.edgecolor": "#c3c2b7",
        "axes.linewidth": 0.8, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF,
        "axes.facecolor": SURF,
    })
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.5))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.83, bottom=0.15, wspace=0.28)

    def line(a, x, y, c, lab, ls="-"):
        a.plot(x, y, ls, color=c, lw=2.2, marker="o", ms=6.5, mfc=c, mec=SURF,
               mew=1.2, label=lab, zorder=3)

    # ---- Panel A: MIDI, the confound (micro-window 0.06s) ----
    a = ax[0]
    line(a, S, np.nanmean(I_v0[:, wi06], 1), C_V0, "V0 mean-pool")
    line(a, S, np.nanmean(I_v1[:, wi06], 1), C_V1, "V1 clip-pool")
    line(a, S, np.nanmean(I_v2, 1), C_V2, "V2 delta")
    a.set_title("A · MIDI, perfect boundaries\nmicro-window 0.06 s", fontsize=11,
                color=INK, loc="left", fontweight="bold")
    a.set_xlabel("contamination strength  s"); a.set_ylabel("next-chord interval accuracy")
    a.set_ylim(0.55, 0.87)
    a.annotate("V0 collapses  -7pp", (S[-1], np.nanmean(I_v0[:, wi06], 1)[-1]),
               color=C_V0, fontsize=9, xytext=(-4, -14), textcoords="offset points", ha="right")
    a.annotate("V2 flat (robust)", (S[-1], np.nanmean(I_v2, 1)[-1]),
               color=C_V2, fontsize=9, xytext=(-4, 8), textcoords="offset points", ha="right")

    # ---- Panel B: MIDI, window dependence at heavy contamination ----
    b = ax[1]
    line(b, W, np.nanmean(I_v0[-1], 1), C_V0, "V0 mean-pool")
    line(b, W, np.nanmean(I_v1[-1], 1), C_V1, "V1 clip-pool")
    v2h = float(np.nanmean(I_v2[-1]))
    b.axhline(v2h, color=C_V2, lw=2.2, ls="--", label="V2 delta (window-free)")
    b.set_title("B · MIDI, heavy contamination s=1.0\nconfound is a SHORT-window effect",
                fontsize=11, color=INK, loc="left", fontweight="bold")
    b.set_xlabel("V0/V1 pooling window  W  (s)"); b.set_ylabel("next-chord interval accuracy")
    b.set_xscale("log"); b.set_xticks(W); b.set_xticklabels([f"{w:g}" for w in W])
    b.set_ylim(0.60, 0.87)
    b.annotate("mean-pool fine\nat long W", (W[-1], np.nanmean(I_v0[-1], 1)[-1]),
               color=C_V0, fontsize=9, xytext=(-6, -22), textcoords="offset points", ha="right")

    # ---- Panel C: real-audio transfer ----
    c = ax[2]
    vw = real["v2_wins"]; v2b = real["I_v2_bywin"].mean(1)
    rv0 = real["I_v0"].mean(1); rv1 = real["I_v1"].mean(1)
    band_lo = min(rv0.min(), rv1.min()); band_hi = max(rv0.max(), rv1.max())
    c.axhspan(band_lo, band_hi, color=C_V0, alpha=0.13, zorder=0)
    c.axhline((band_lo + band_hi) / 2, color=C_V0, lw=1.6, ls="-", alpha=0.8)
    line(c, vw, v2b, C_V2, "V2 delta")
    c.set_title("C · REAL audio (NNLS bothchroma)\nV2 never beats mean-pool",
                fontsize=11, color=INK, loc="left", fontweight="bold")
    c.set_xlabel("V2 delta-window  (s)"); c.set_ylabel("next-chord interval accuracy")
    c.set_ylim(0.35, 0.95)
    c.annotate("mean-pool V0/V1 band", (vw[0], (band_lo + band_hi) / 2),
               color=C_V0, fontsize=9, xytext=(2, 8), textcoords="offset points", ha="left")
    c.annotate("delta only reaches this\nby pooling ~1s (not a delta)",
               (vw[-1], v2b[-1]), color=C_V2, fontsize=9, xytext=(-6, 6),
               textcoords="offset points", ha="right")

    for a_ in ax:
        a_.grid(True, color=GRID, lw=0.7, zorder=0)
        a_.set_axisbelow(True)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        a_.legend(frameon=False, fontsize=9, loc="lower left")

    fig.suptitle("Does a leakage-robust feature recover chord IDENTITY where mean-pool fails?",
                 fontsize=13, fontweight="bold", color=INK, x=0.055, ha="left", y=0.965)
    png = OUT / "leakage_robust_identity.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    return png


def b64(png):
    return base64.b64encode(png.read_bytes()).decode()


def fmt_table(S, W, arr, name):
    head = "".join(f"<th>{w:g}s</th>" for w in W)
    rows = ""
    for si, s in enumerate(S):
        cells = "".join(f"<td>{np.nanmean(arr[si, wi]):.3f}</td>" for wi in range(len(W)))
        rows += f"<tr><td class='rl'>s={s:.1f}</td>{cells}</tr>"
    return (f"<table><caption>{name}</caption><thead><tr><th>strength \\ W</th>{head}</tr>"
            f"</thead><tbody>{rows}</tbody></table>")


def main():
    midi = np.load(OUT / "leakage_robust_identity_curves.npz")
    real = np.load(OUT / "leakage_robust_realaudio.npz")
    png = make_figure(midi, real)
    img = b64(png)

    S = midi["strengths"]; W = midi["windows"]
    nT = int(midi["n_transitions"][0]); nS = int(midi["n_songs"][0])
    rT = int(real["n_transitions"][0]); rS = int(real["n_songs"][0])
    I_v0, I_v1, I_v2 = midi["I_v0"], midi["I_v1"], midi["I_v2"]
    wi06 = int(np.argmin(np.abs(W - 0.06)))
    v0_micro_drop = np.nanmean(I_v0[0, wi06]) - np.nanmean(I_v0[-1, wi06])
    v2v0_micro_hi = np.nanmean(I_v2[-1]) - np.nanmean(I_v0[-1, wi06])
    v2v1_micro_hi = np.nanmean(I_v2[-1]) - np.nanmean(I_v1[-1, wi06])
    rv2 = float(real["I_v2"].mean()); rv0 = float(real["I_v0"].mean(1).max())

    midi_tbl = fmt_table(S, W, I_v0, "V0 mean-pool — MIDI interval accuracy")
    real_v0 = "".join(f"<td>{v:.3f}</td>" for v in real["I_v0"].mean(1))
    real_v1 = "".join(f"<td>{v:.3f}</td>" for v in real["I_v1"].mean(1))
    real_hdr = "".join(f"<th>{w:g}s</th>" for w in real["windows"])
    real_tbl = (f"<table><caption>REAL audio interval accuracy — V2 delta (window-free) "
                f"= <b>{rv2:.3f}</b></caption><thead><tr><th>window</th>{real_hdr}</tr></thead>"
                f"<tbody><tr><td class='rl'>V0 mean-pool</td>{real_v0}</tr>"
                f"<tr><td class='rl'>V1 clip-pool</td>{real_v1}</tr></tbody></table>")

    html = f"""<title>Leakage-robust chord identity — screen verdict</title>
<style>
:root{{--bg:#f9f9f7;--card:#fff;--ink:#0b0b0b;--sec:#52514e;--mut:#898781;
--line:#e1e0d9;--v0:{C_V0};--v1:{C_V1};--v2:{C_V2};--red:#d03b3b;--good:#0ca30c;}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme=light])){{
--bg:#0d0d0d;--card:#1a1a19;--ink:#fff;--sec:#c3c2b7;--mut:#898781;--line:#2c2c2a;}}}}
:root[data-theme=dark]{{--bg:#0d0d0d;--card:#1a1a19;--ink:#fff;--sec:#c3c2b7;--line:#2c2c2a;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.55;}}
.wrap{{max-width:1080px;margin:0 auto;padding:32px 22px 80px;}}
h1{{font-size:26px;margin:0 0 4px;letter-spacing:-.01em;}}
.sub{{color:var(--sec);font-size:15px;margin:0 0 24px;}}
.verdict{{background:var(--card);border:1px solid var(--line);border-left:5px solid var(--red);
border-radius:12px;padding:20px 22px;margin:22px 0;}}
.verdict h2{{margin:0 0 8px;font-size:17px;}}
.verdict .k{{color:var(--red);font-weight:700;}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:22px 0;}}
.tile{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;}}
.tile .n{{font-size:26px;font-weight:700;letter-spacing:-.02em;}}
.tile .l{{font-size:12.5px;color:var(--sec);margin-top:3px;}}
.fig{{background:#fcfcfb;border:1px solid var(--line);border-radius:12px;padding:14px;margin:22px 0;}}
.fig img{{width:100%;height:auto;display:block;}}
.fig figcaption{{color:#52514e;font-size:12.5px;margin-top:10px;}}
h2.sec{{font-size:19px;margin:34px 0 10px;padding-top:8px;border-top:1px solid var(--line);}}
p{{color:var(--sec);}}
p strong,li strong{{color:var(--ink);}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0;
font-variant-numeric:tabular-nums;}}
caption{{text-align:left;color:var(--sec);font-size:12.5px;margin-bottom:6px;}}
th,td{{padding:5px 9px;text-align:right;border-bottom:1px solid var(--line);}}
th:first-child,td.rl{{text-align:left;color:var(--sec);}}
thead th{{color:var(--mut);font-weight:600;}}
.tblwrap{{overflow-x:auto;}}
code{{background:var(--card);border:1px solid var(--line);border-radius:5px;padding:1px 5px;
font-size:12.5px;}}
.legend b{{padding:1px 7px;border-radius:5px;color:#fff;font-size:12px;}}
ul{{color:var(--sec);}} li{{margin:5px 0;}}
.files{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:8px 20px;}}
</style>
<div class="wrap">
<h1>Leakage-robust chord identity: does a directional delta beat mean-pool?</h1>
<p class="sub">Research session 2026-07-24 · POP909 · screen-before-build · every number from a real run</p>

<p class="legend">
<b style="background:var(--v0)">V0 mean-pool</b> &nbsp;
<b style="background:var(--v1)">V1 clip-pool</b> &nbsp;
<b style="background:var(--v2)">V2 root-normalised directional delta</b>
</p>

<div class="verdict">
<h2>Verdict — <span class="k">NULL for V2 (delta); the surviving lever is V1 clip-pool</span></h2>
<p>The root-normalised directional delta <b>is not a real complementary identity signal
worth building into <code>FusionChordDecoder</code></b>. It wins only in the idealised
regime its cancellation premise assumes — <b>perfect GT boundaries + a clean linear
common-mode contamination</b> — where on synthetic MIDI it is the only feature immune to
leakage (Panels A/B). On <b>real</b> audio-derived NNLS chroma, where boundaries jitter and
leakage is nonlinear (harmonics, attack smear), it <b>collapses to {rv2:.2f} vs mean-pool's
~{rv0:.2f}</b> (Panel C) — it degrades exactly like the 5 dead adjudicators. The
diagnosis still holds (mean-pool's residual errors are 39% at &plusmn;P5 on real audio too),
but the delta is the wrong cure. Per the brief's own gate, the real-audio transfer screen
falsified the premise, so <b>no head was trained</b> — that would only fit a non-transferable
signal.</p>
</div>

<div class="tiles">
<div class="tile"><div class="n" style="color:var(--v0)">-{v0_micro_drop*100:.0f}pp</div>
<div class="l">V0 mean-pool drop under contamination, micro-window (MIDI) — the confound is real</div></div>
<div class="tile"><div class="n" style="color:var(--v2)">+{v2v0_micro_hi*100:.0f}pp</div>
<div class="l">V2 &minus; V0 at heavy contamination, micro-window (MIDI) — gap WIDENS from &minus;14pp</div></div>
<div class="tile"><div class="n" style="color:var(--red)">{rv2:.2f}</div>
<div class="l">V2 real-audio interval acc vs mean-pool ~{rv0:.2f} — does NOT transfer</div></div>
<div class="tile"><div class="n">{nT:,}</div>
<div class="l">MIDI chord transitions ({nS} songs) · {rT:,} real-audio ({rS} songs)</div></div>
</div>

<div class="fig">
<img src="data:image/png;base64,{img}" alt="three-panel result figure"/>
<figcaption><b>A</b> — on MIDI with perfect boundaries, a short (micro-segment) pooling
window makes mean-pool (V0) collapse as synthetic leakage rises, while the delta (V2) stays
flat. <b>B</b> — but this is purely a short-window effect: at pooling windows &ge;0.4&nbsp;s
mean-pool is immune and beats the delta. <b>C</b> — on real NNLS chroma the delta only
approaches the mean-pool band by widening its window to ~1&nbsp;s, at which point it is no
longer a boundary-local delta; at the short windows that are its whole reason to exist it is
far below mean-pool.</figcaption>
</div>

<h2 class="sec">The screen, in one sentence</h2>
<p>Under contamination a leakage-robust feature does beat V0 mean-pool on <b>root interval</b>
in the <b>micro-segment regime</b>, and the gap widens with contamination
(V2&minus;V0 goes from &minus;14pp clean to +{v2v0_micro_hi*100:.0f}pp at s=1.0; V2 even beats
clip-pool V1 by +{v2v1_micro_hi*100:.0f}pp there). But two facts kill it as a build target:
<b>(1)</b> the whole effect vanishes once the pooling window is &ge;0.4&nbsp;s — the real
pipeline's per-beat decode already pools far more than a micro-segment; <b>(2)</b> it does not
survive contact with real audio. V2 also never helps <b>quality</b> (real-audio quality
{float(real['Q_v2'].mean()):.2f} vs V0/V1 {float(real['Q_v0'].mean(1).max()):.2f}).</p>

<h2 class="sec">Method (reuse, don't reinvent)</h2>
<ul>
<li><b>Data</b> — POP909 MIDI-derived per-frame chroma (pretty_midi piano-roll &rarr;
bass/treble 12-pc halves @50&nbsp;Hz), so GT boundaries/labels are TRULY perfect and
contamination is the only knob. Real-audio arm reuses the {rS} cached NNLS
<code>bothchroma</code> features already on disk (<code>data/cache/nnls_infer/</code>) — no
render, no disk cost. Boundaries/labels from <code>harmonia.data.pop909_parser</code>
(functional root; <code>/bass</code> discarded &rarr; root+quality only).</li>
<li><b>V0</b> mean-pool anchored at the boundary + pre-boundary bleed (beat-snap register).
<b>V1</b> clip-pool strictly <code>[t0,t0+W)</code> (reuses the <code>_clip_pool</code>
framing from <code>yt_chord_corpus.py</code>). <b>V2</b> L1-normalised next-window minus
prev-window, root-normalised to prev-root, conditioned on prev quality — window-independent.</li>
<li><b>Contamination</b> (frame-level, shared by all variants): exp-decaying prev-chord
sustain into early frames + beat-snap neighbour bleed + per-frame dropout/noise. It
reproduces the documented confound: V0's residual root errors concentrate at
<b>&plusmn;P5 (~28&ndash;39%)</b>, matching the brief's "P4/P5 = 37% of root loss".</li>
<li><b>Screen</b> — cosine kNN (k=25), song-disjoint splits, multi-seed; predict
target = (interval=(next&minus;prev root) mod 12, reduced quality). No training.</li>
</ul>

<h2 class="sec">Honesty caveats</h2>
<ul>
<li>On clean chroma the delta is trivial (&Delta;=N&minus;P recovers N given P), so only the
<b>contamination / transfer</b> curves are informative — reported in full, including where V2
loses.</li>
<li>The synthetic contamination is a <b>linear</b> mix of the prev-chord profile — exactly the
common-mode a linear delta cancels. That is why V2 looks robust on MIDI and is precisely why
the <b>real-audio arm is the load-bearing test</b>: real leakage is not linear common-mode.</li>
<li>V2's clean-MIDI number is depressed by sparse MIDI boundary windows (kNN, not a trained
head); this does not change the transfer conclusion (a head cannot recover information that
boundary jitter destroys).</li>
</ul>

<h2 class="sec">If you build anything: the surviving finding</h2>
<p>The one leakage-robust lever that survives real audio is <b>clip-pool</b> (V1) — strictly
post-boundary framing, which the repo already ships (<code>seg_feature_clipped</code> /
<code>_clip_pool</code>). It edges V0 by ~5&ndash;8pp only in the short-window regime and is
essentially tied at production window lengths. <b>FusionChordDecoder emission hook</b> (for the
record, not implemented): <code>harmonia/align/inference.py::viterbi_decode</code> exposes a
swappable per-beat EMISSION factor; the acoustically-grounded transition factor the brief
wanted would go there. V2 does <b>not</b> earn that slot. Note the repo's ready transition
priors (<code>build_transition</code> fifth-motion; the <code>chord_progression_model.npz</code>
trigram) make fifth-motion cheap and would <b>reinforce</b> the P4/P5 error — so the P5
confusion is best attacked in the emission/feature, not the transition.</p>

<h2 class="sec">Data tables</h2>
<div class="tblwrap">{midi_tbl}</div>
<p style="font-size:12.5px">V0 mean-pool degrades monotonically with contamination at short W
and is flat at long W. (V1/V2 full arrays in the npz.)</p>
<div class="tblwrap">{real_tbl}</div>

<h2 class="sec">Files (new; nothing staged, nothing wired)</h2>
<div class="files"><ul>
<li><code>scripts/exp_leakage_robust_identity.py</code> — MIDI screen (2D window&times;contamination sweep)</li>
<li><code>scripts/exp_leakage_robust_realaudio.py</code> — real-audio transfer on cached NNLS bothchroma</li>
<li><code>scripts/make_leakage_robust_artifact.py</code> — this figure + page</li>
<li><code>docs/research_sessions/leakage_robust_identity{{.html,.png,_curves.npz}}</code>,
<code>leakage_robust_realaudio.npz</code></li>
</ul></div>
</div>
"""
    out_html = OUT / "leakage_robust_identity.html"
    out_html.write_text(html)
    print(f"Wrote {out_html}  ({len(html)//1024} KB)")
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
