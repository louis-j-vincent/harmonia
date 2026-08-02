"""Render the contested chords so they can be judged by ear. (Louis, 2026-08-02)

For every half-bar where the LM disagrees with the chart and both gates open,
this writes three clips:

    A  the real audio around the contested half-bar, untouched
    B  the same audio with the CHART's chord played over the contested half-bar
    C  the same audio with the LM's chord played over it

If a chord belongs, it locks into the record; if it does not, it beats against
it. That is the arbitration — numbers cannot settle which of two plausible
chords a record is actually playing, and the project has no verified ground
truth for these five songs.

The synthesised chord is deliberately plain (a few harmonics, soft attack, mid
register, a fifth below the record's level) so it colours the moment without
masking it. A soft click marks the start of the contested half-bar.

    .venv/bin/python scripts/chord_lm_audition.py
    -> docs/chord_lm_audition.html  (self-contained, audio embedded)
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import re
import sys
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harmonia_min.chord_lm import vocab                      # noqa: E402
from harmonia_min.chord_lm.intervene import propose          # noqa: E402

SR = 22050
LEAD_BARS = 2.0        # how much context before the contested half-bar
TAIL_BARS = 1.0
AUDIO_DIR = Path("docs/audio")

FAMILY_PCS = {
    "maj": (0, 4, 7), "min": (0, 3, 7), "dom": (0, 4, 7, 10),
    "dim": (0, 3, 6), "hdim": (0, 3, 6, 10), "sus": (0, 5, 7),
    "aug": (0, 4, 8),
}


def synth_chord(tok: int | None, dur: float, sr: int = SR) -> np.ndarray:
    """A plain sustained chord: root in the bass, the rest around C4."""
    n = int(dur * sr)
    out = np.zeros(n, dtype=np.float32)
    if tok is None or not vocab.is_chord(tok):
        return out
    root, fam = vocab.split_chord(tok)
    intervals = FAMILY_PCS[fam]
    t = np.arange(n) / sr
    midis = [36 + root] + [60 + (root + iv) % 12 for iv in intervals]
    for m in midis:
        f = 440.0 * 2 ** ((m - 69) / 12)
        for h, amp in ((1, 1.0), (2, 0.35), (3, 0.15)):
            out += amp * np.sin(2 * np.pi * f * h * t).astype(np.float32)
    # soft attack/release so it does not click at the seams
    env = np.ones(n, dtype=np.float32)
    a = min(int(0.04 * sr), n // 3)
    r = min(int(0.12 * sr), n // 3)
    env[:a] = np.linspace(0, 1, a)
    env[-r:] = np.linspace(1, 0, r)
    out *= env
    peak = np.abs(out).max()
    return out / peak if peak > 0 else out


def click(sr: int = SR) -> np.ndarray:
    n = int(0.03 * sr)
    t = np.arange(n) / sr
    return (np.sin(2 * np.pi * 1800 * t) *
            np.exp(-t * 90)).astype(np.float32) * 0.25


def write_clip(x: np.ndarray, path: Path, sr: int = SR) -> str:
    """Write a clip to disk and return the URL the app will serve it from.

    NOT a data: URI any more. iOS will not play media without HTTP Range
    (206 Partial Content), and `python -m http.server` answers 200 to a Range
    request — the page loaded on the iPhone and every player stayed silent.
    The app's own /audio route already answers 206 and carries the iOS
    container-type fix the 2026-08-02 stall triage landed, so clips go through
    it instead of being inlined.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = np.abs(x).max()
    if peak > 1.0:
        x = x / peak
    sf.write(path, x, sr, format="WAV", subtype="PCM_16")
    return f"/audio/{path.parent.name}/{path.name}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm-min", type=float, default=0.80)
    ap.add_argument("--pipe-max", type=float, default=0.60)
    ap.add_argument("--out",
                    default="harmonia_min/state/reports/chord_lm_audition.html",
                    help="served by the app at /reports/<name>")
    ap.add_argument("--clip-dir", default="docs/audio/lm_audition",
                    help="served by the app at /audio/<subdir>/<name>")
    args = ap.parse_args()

    device = "mps"
    try:
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    except Exception:
        device = "cpu"

    clip_dir = Path(args.clip_dir)
    for old in clip_dir.glob("*.wav"):
        old.unlink()                      # stale clips from a previous render
    cards = []
    for p in sorted(glob.glob("harmonia_min/state/charts/*.json")):
        chart = json.loads(Path(p).read_text())
        title = chart.get("title", "?")
        rel = (chart.get("audio_url") or "").lstrip("/")
        wav_path = AUDIO_DIR / Path(rel).name
        if not wav_path.exists():
            print(f"  {title}: audio missing ({wav_path})")
            continue
        sugg = propose(chart, device=device, lm_min=args.lm_min,
                       pipe_max=args.pipe_max)
        if not sugg:
            print(f"  {title}: 0 proposals")
            continue
        # librosa, not soundfile: the library songs are .m4a, which libsndfile
        # cannot open (it falls back to audioread + ffmpeg here).
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, _ = librosa.load(wav_path, sr=SR, mono=True)
        grid = chart.get("barGrid") or []
        bw = float(np.median(np.diff(grid))) if len(grid) > 2 else 2.0

        for s in sugg:
            a0 = max(0.0, s.t0 - LEAD_BARS * bw)
            a1 = min(len(y) / SR, s.t1 + TAIL_BARS * bw)
            i0, i1 = int(a0 * SR), int(a1 * SR)
            base = y[i0:i1].copy()
            if base.size < SR // 2:
                continue
            rms = float(np.sqrt((base ** 2).mean())) or 0.05
            off = int((s.t0 - a0) * SR)
            dur = max(s.t1 - s.t0, 0.2)

            def overlay(tok):
                mix = base.copy()
                ch = synth_chord(tok, dur) * rms * 0.9      # sit under the record
                end = min(off + len(ch), len(mix))
                mix[off:end] += ch[:end - off]
                ck = click()
                e2 = min(off + len(ck), len(mix))
                mix[off:e2] += ck[:e2 - off]
                return mix

            marked = base.copy()
            ck = click()
            e2 = min(off + len(ck), len(marked))
            marked[off:e2] += ck[:e2 - off]

            slug = re.sub(r"[^a-z0-9]+", "_",
                          f"{title}_{s.bar+1}_{s.slot+1}".lower()).strip("_")
            cards.append({
                "title": title, "bar": s.bar + 1, "slot": s.slot + 1,
                "t0": s.t0, "shown": s.shown, "proposed": s.proposed,
                "lm": s.lm_conf, "pipe": s.pipe_conf,
                "alts": s.alternatives,
                "a": write_clip(marked, clip_dir / f"{slug}_a.wav"),
                "b": write_clip(overlay(s.shown_tok), clip_dir / f"{slug}_b.wav"),
                "c": write_clip(overlay(s.proposed_tok), clip_dir / f"{slug}_c.wav"),
                "shown_silent": s.shown_tok is None,
                "prop_silent": s.proposed_tok is None,
                "context": s.context,
            })
        print(f"  {title}: {len(sugg)} proposals rendered")

    # chord-vs-chord disagreements first: those are the ones an ear can settle.
    # A card where either side is N.C. is a "does anything sound here" question,
    # which is a different (and easier) judgement.
    cards.sort(key=lambda c: (c["shown_silent"] or c["prop_silent"], c["title"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(render_html(cards, args))
    n_clips = len(list(Path(args.clip_dir).glob("*.wav")))
    print(f"\nwrote {args.out}  ({len(cards)} contested chords, "
          f"{Path(args.out).stat().st_size/1e3:.0f} kB page + {n_clips} clips)")
    print("open: http://<host>:7772/reports/" + Path(args.out).name)


def context_strip(ctx, shown: str, proposed: str) -> str:
    """The chart around the contested half-bar, grouped into bars.

    A chord is only right or wrong relative to its neighbours, so the target
    cell shows BOTH candidates stacked while everything around it shows what
    the chart already says.
    """
    if not ctx:
        return ""
    bars: list[tuple[int, list[dict]]] = []
    for cell in ctx:
        if not bars or bars[-1][0] != cell["bar"]:
            bars.append((cell["bar"], []))
        bars[-1][1].append(cell)
    out = []
    for bar_no, cells in bars:
        inner = []
        for cell in cells:
            if cell["target"]:
                inner.append(
                    f'<span class="cell tgt">'
                    f'<span class="was">{html.escape(shown)}</span>'
                    f'<span class="now">{html.escape(proposed)}</span></span>')
            else:
                inner.append(f'<span class="cell">'
                             f'{html.escape(cell["name"])}</span>')
        out.append(f'<span class="bar"><span class="barno">{bar_no}</span>'
                   f'{"".join(inner)}</span>')
    return f'<div class="strip">{"".join(out)}</div>'


def render_html(cards, args) -> str:
    rows = []
    for i, c in enumerate(cards):
        alts = " · ".join(f"{html.escape(n)} <b>{p:.2f}</b>" for n, p in c["alts"])
        b_lab = "silence (N.C.)" if c["shown_silent"] else html.escape(c["shown"])
        c_lab = "silence (N.C.)" if c["prop_silent"] else html.escape(c["proposed"])
        strip = context_strip(c.get("context", []), c["shown"], c["proposed"])
        rows.append(f"""
<div class="card">
  <div class="hd">
    <span class="song">{html.escape(c['title'])}</span>
    <span class="loc">mesure {c['bar']}, demi-mesure {c['slot']} · {c['t0']:.1f}s</span>
  </div>
  {strip}
  <div class="verdict">
    <span class="chip chart">chart : {html.escape(c['shown'])}</span>
    <span class="arrow">vs</span>
    <span class="chip lm">LM : {html.escape(c['proposed'])}</span>
    <span class="conf">LM {c['lm']:.2f} · pipeline {c['pipe']:.2f}</span>
  </div>
  <div class="players">
    <div class="p"><label>A — audio seul <em>(le clic marque l'endroit)</em></label>
      <audio controls preload="none" src="{c['a']}"></audio></div>
    <div class="p"><label>B — avec l'accord du <b>chart</b> : {b_lab}</label>
      <audio controls preload="none" src="{c['b']}"></audio></div>
    <div class="p"><label>C — avec l'accord du <b>LM</b> : {c_lab}{
        " <em>(donc identique à A : le LM dit qu'aucun accord ne sonne)</em>"
        if c['prop_silent'] else ""}</label>
      <audio controls preload="none" src="{c['c']}"></audio></div>
  </div>
  <div class="alts">top-3 du LM : {alts}</div>
</div>""")
    return f"""<meta charset="utf-8"><title>Accords contestés par le chord LM</title>
<style>
 :root {{ color-scheme: light dark; }}
 body {{ font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        max-width: 940px; margin: 2rem auto; padding: 0 1rem; }}
 h1 {{ font-size: 1.4rem; margin-bottom: .2rem; }}
 .intro {{ opacity:.8; margin-bottom: 1.6rem; }}
 .intro code {{ background: rgba(128,128,128,.18); padding: .1em .35em; border-radius:4px; }}
 .card {{ border: 1px solid rgba(128,128,128,.35); border-radius: 12px;
          padding: 1rem 1.1rem; margin-bottom: 1.1rem; }}
 .hd {{ display:flex; justify-content:space-between; align-items:baseline;
        gap:1rem; flex-wrap:wrap; }}
 .song {{ font-weight: 650; font-size: 1.05rem; }}
 .loc {{ opacity:.65; font-size:.87rem; font-variant-numeric: tabular-nums; }}
 .verdict {{ margin:.7rem 0 .9rem; display:flex; align-items:center;
             gap:.6rem; flex-wrap:wrap; }}
 .chip {{ padding:.2em .6em; border-radius:999px; font-weight:600; }}
 .chart {{ background: rgba(120,120,120,.22); }}
 .lm {{ background: rgba(56,142,255,.22); }}
 .arrow {{ opacity:.5; }}
 .conf {{ opacity:.6; font-size:.85rem; font-variant-numeric: tabular-nums; }}
 .players {{ display:grid; gap:.55rem; }}
 .p label {{ display:block; font-size:.85rem; opacity:.8; margin-bottom:.15rem; }}
 .p em {{ opacity:.6; font-style: normal; }}
 audio {{ width:100%; height:34px; }}
 .alts {{ margin-top:.7rem; font-size:.84rem; opacity:.75; }}
 /* context strip: the chart around the contested half-bar */
 .strip {{ display:flex; gap:.35rem; overflow-x:auto; padding:.55rem 0 .3rem;
           font-variant-numeric: tabular-nums; -webkit-overflow-scrolling:touch; }}
 .bar {{ display:flex; position:relative; border:1px solid rgba(128,128,128,.35);
         border-radius:7px; overflow:hidden; flex:0 0 auto; }}
 .barno {{ position:absolute; top:1px; left:4px; font-size:.6rem; opacity:.45; }}
 .cell {{ min-width:74px; padding:.85rem .5rem .45rem; text-align:center;
          font-size:.82rem; white-space:nowrap; }}
 .cell + .cell {{ border-left:1px dashed rgba(128,128,128,.3); }}
 .cell.tgt {{ display:flex; flex-direction:column; gap:.1rem; padding-top:.8rem;
              background:rgba(56,142,255,.14);
              box-shadow: inset 0 0 0 2px rgba(56,142,255,.55); }}
 .was {{ text-decoration: line-through; opacity:.55; font-size:.78rem; }}
 .now {{ color:#2f7bff; font-weight:650; font-size:.82rem; }}
 @media (prefers-color-scheme: dark) {{ .now {{ color:#7db0ff; }} }}
</style>
<h1>Accords contestés par le chord LM</h1>
<div class="intro">
  {len(cards)} demi-mesures où le LM est sûr (≥ {args.lm_min:.2f}) et le pipeline
  ne l'est pas (≤ {args.pipe_max:.2f}). Écoute <b>B</b> puis <b>C</b> : si l'accord
  appartient au disque il s'y encastre, sinon il bat contre. Le clic marque le
  début de la demi-mesure contestée ; ce qui précède est le contexte.
  Rien n'a été modifié dans les charts — <code>HARMONIA_CHORD_LM_SUGGEST</code>
  est désactivé par défaut.
</div>
{''.join(rows)}
"""


if __name__ == "__main__":
    main()
