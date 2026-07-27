#!/usr/bin/env python3
"""Tinder-style disagreement triage: PREDICTOR vs frozen GT, now with a
CORRECT action that opens the project's cylinder chord-picker (Root / Family /
7th / Ext / Bass, iOS scroll-snap — lifted from
harmonia/output/chart_interactive.py) so Louis can enter the TRUE chord.

For every verified brick0 GT song, run the SHIPPED pipeline, walk the GT chord
timeline, and emit one card per span where the prediction DISAGREES with the GT
(root and/or quality-family mismatch, incl. chord-vs-no-chord). Consecutive GT
chords carrying the same (gt_label, pred_label) pair are merged into one card.

Three actions per card:
  ◀ WRONG   the predictor is wrong (we don't learn the truth)
  RIGHT ▶   the predictor is right  ⇒ the GT is the thing that's wrong
  🎯 CORRECT open the cylinder picker, dial the TRUE chord (root+quality+bass)

Verdicts + corrections are POSTed to the companion server
(scripts/tinder_server.py) which appends them to a durable ledger in the repo
(docs/research_sessions/tinder_ledger.json); if that endpoint is absent (page
opened as a bare file) it falls back to localStorage. Analyse the ledger with
scripts/analyze_tinder_ledger.py.

Read-only on the repo except for the single output HTML.

Usage:
    .venv/bin/python scripts/build_pred_tinder.py
    .venv/bin/python scripts/build_pred_tinder.py --songs stand_by_me,bein_green
    .venv/bin/python scripts/build_pred_tinder.py --level root
"""
from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.eval.accuracy_score import (          # noqa: E402
    Chord, chord_family, load_frozen_gt, run_prediction, _canon_quality,
)

GT_DIR = REPO / "golden" / "brick0"
OUT_HTML = REPO / "docs" / "research_sessions" / "pred_tinder_2026-07-27.html"

LEAD = 1.2          # seconds of audio before the region
MAXLEN = 12.0       # cap on a single clip
BITRATE = "40k"
FLOOR_GIB = 1.5

PC = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# vocab-quality -> (family_id, seventh_token, ext_token) picker coordinates, so
# a card can pre-seed the cylinder columns from GT / pred. Best-effort for
# tokens the picker has no exact leaf for (min6, 11) — the truth is one scroll
# away regardless.
PICKER_FROM_VOCAB = {
    "maj": ("maj", "", ""), "6": ("maj", "6", "6"), "maj6": ("maj", "6", "6"),
    "maj7": ("maj", "^7", "^7"), "7": ("maj", "7", "7"), "9": ("maj", "7", "9"),
    "13": ("maj", "7", "13"), "maj9": ("maj", "^7", "^9"),
    "min": ("min", "-", "-"), "min7": ("min", "-7", "-7"),
    "min9": ("min", "-7", "-9"), "min6": ("min", "-", "-"),
    "minmaj7": ("min", "-^7", "-^7"), "11": ("min", "-7", "-11"),
    "dim": ("dim", "o", "o"), "dim7": ("dim", "o7", "o7"),
    "hdim7": ("dim", "h7", "h7"), "aug": ("aug", "+", "+"),
    "sus2": ("sus", "sus2", "sus2"), "sus4": ("sus", "sus4", "sus4"),
}


def free_gib() -> float:
    return shutil.disk_usage(REPO).free / 2**30


def root_match(a: Chord, b: Chord) -> bool:
    if a.is_nc or b.is_nc:
        return a.is_nc and b.is_nc
    return a.root_pc == b.root_pc


def family_match(a: Chord, b: Chord) -> bool:
    if a.is_nc or b.is_nc:
        return a.is_nc and b.is_nc
    return chord_family(a.quality) == chord_family(b.quality)


def seed_of(c: Chord) -> dict | None:
    """Cylinder-picker coordinates {root,fam,sev,ext,bass} for a chord, or None
    for no-chord (nothing to seed)."""
    if c.is_nc:
        return None
    fam, sev, ext = PICKER_FROM_VOCAB.get(_canon_quality(c.quality), ("maj", "", ""))
    bass = c.bass_pc if c.bass_pc is not None else c.root_pc
    return {"root": c.root_pc, "fam": fam, "sev": sev, "ext": ext, "bass": bass}


def dominant_pred(pred: list[Chord], t0: float, t1: float) -> Chord:
    best, best_ov = None, -1.0
    for c in pred:
        ov = min(t1, c.t1) - max(t0, c.t0)
        if ov > best_ov:
            best, best_ov = c, ov
    return best if best is not None else Chord(t0, t1, None, "N", None, "N")


def encode_clip(src: Path, out: Path, ss: float, dur: float) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{ss:.3f}", "-i", str(src),
         "-t", f"{dur:.3f}", "-ac", "1", "-ar", "22050", "-c:a", "libmp3lame",
         "-b:a", BITRATE, "-f", "mp3", str(out)],
        check=True,
    )


def build_cards(gt_path: Path, level: str) -> list[dict]:
    gt = load_frozen_gt(gt_path)
    pred = run_prediction(gt.resolved_audio_path)

    rows = []
    for g in gt.gt_chords:
        p = dominant_pred(pred, g.t0, g.t1)
        rm = root_match(g, p)
        fm = family_match(g, p)
        disagree = (not rm) if level == "root" else (not (rm and fm))
        rows.append((g, p, disagree, rm, fm))

    cards, i = [], 0
    while i < len(rows):
        g, p, dis, rm, fm = rows[i]
        if not dis:
            i += 1
            continue
        t0, t1 = g.t0, g.t1
        j = i + 1
        while (j < len(rows) and rows[j][2]
               and rows[j][0].label == g.label and rows[j][1].label == p.label):
            t1 = rows[j][0].t1
            j += 1
        cards.append({
            "song": gt.song_id, "title": gt.title,
            "t0": round(t0, 2), "t1": round(t1, 2),
            "gt": g.label, "pred": p.label,
            "root_ok": rm, "family_ok": fm,
            "gt_seed": seed_of(g), "pred_seed": seed_of(p),
            "audio_path": str(gt.resolved_audio_path),
        })
        i = j
    return cards


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", default="", help="comma slugs; default = all verified")
    ap.add_argument("--level", choices=("root", "family"), default="family")
    args = ap.parse_args(argv)

    want = set(s.strip() for s in args.songs.split(",") if s.strip())
    paths = []
    for f in sorted(GT_DIR.glob("*.gt.json")):
        if f.stem.startswith("_fixture"):
            continue
        d = json.loads(f.read_text())
        if not d.get("verified"):
            continue
        if want and d["song_id"] not in want:
            continue
        paths.append(f)

    all_cards = []
    for f in paths:
        print(f"[{f.stem}] predicting…", flush=True)
        cards = build_cards(f, args.level)
        print(f"    {len(cards)} disagreement cards", flush=True)
        all_cards.extend(cards)

    tmp = Path(tempfile.mkdtemp(prefix="tinder_"))
    print(f"encoding {len(all_cards)} clips; disk free {free_gib():.2f} GiB", flush=True)
    for k, c in enumerate(all_cards):
        if free_gib() < FLOOR_GIB:
            print(f"ABORT at clip {k}: disk floor", flush=True)
            break
        a = max(0.0, c["t0"] - LEAD)
        b = min(c["t1"] + 0.4, a + MAXLEN)
        mp3 = tmp / f"c{k}.mp3"
        encode_clip(Path(c["audio_path"]), mp3, a, b - a)
        c["b64"] = base64.b64encode(mp3.read_bytes()).decode("ascii")
        c["region_a"], c["region_b"] = round(c["t0"] - a, 2), round(c["t1"] - a, 2)
        c["id"] = f"{c['song']}:{c['t0']}"
        mp3.unlink()
    shutil.rmtree(tmp, ignore_errors=True)
    for c in all_cards:
        c.pop("audio_path", None)

    html = _TEMPLATE.replace("__DATA__", json.dumps(all_cards, separators=(",", ":"))) \
                    .replace("__N__", str(len(all_cards))) \
                    .replace("__LEVEL__", args.level) \
                    .replace("__APIBASE__", "/api") \
                    .replace("__STOREKEY__", "harmonia_pred_tinder_v2")
    OUT_HTML.write_text(html)
    print(f"\nwrote {OUT_HTML.relative_to(REPO)}  ({len(html)/1e6:.1f} MB, "
          f"{len(all_cards)} cards)")
    print(f"disk free {free_gib():.2f} GiB")
    return 0


_TEMPLATE = r"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Harmonia — predictor vs GT swipe</title>
<style>
:root{--bg:#0e1116;--card:#1b2027;--fg:#e8edf2;--muted:#8a97a6;
--wrong:#ff5c6c;--right:#37d67a;--fix:#c792ea;--seg:#ffa24d;--gt:#ffd166;--pred:#5aa9ff;--line:#2a323d}
*{box-sizing:border-box}
body{margin:0;font:15px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
background:var(--bg);color:var(--fg);overflow:hidden;height:100dvh;
display:flex;flex-direction:column;align-items:center;padding-top:env(safe-area-inset-top)}
header{width:100%;padding:12px 16px;border-bottom:1px solid var(--line);
display:flex;align-items:center;gap:12px;flex-wrap:wrap}
header h1{font-size:14px;margin:0;font-weight:600}
header .sub{color:var(--muted);font-size:11px}
.bar{flex:1;height:6px;background:#20262f;border-radius:3px;overflow:hidden;min-width:120px;max-width:340px}
.bar>i{display:block;height:100%;background:var(--pred);width:0}
.counts{font-variant-numeric:tabular-nums;font-size:11px;color:var(--muted)}
.counts b.w{color:var(--wrong)}.counts b.r{color:var(--right)}.counts b.f{color:var(--fix)}
#stage{position:relative;flex:1;width:100%;max-width:520px;display:flex;
align-items:center;justify-content:center}
.card{position:absolute;width:90%;max-width:440px;background:var(--card);
border:1px solid var(--line);border-radius:20px;padding:20px 20px 16px;
box-shadow:0 18px 50px rgba(0,0,0,.5);touch-action:none;user-select:none}
.card .meta{display:flex;justify-content:space-between;color:var(--muted);
font-size:12px;margin-bottom:14px}
.chords{display:flex;align-items:stretch;gap:10px;margin:4px 0 12px}
.chip{flex:1;border-radius:14px;padding:12px 8px;text-align:center;background:#141922;
border:1px solid var(--line)}
.chip .lbl{font-size:10px;letter-spacing:.13em;text-transform:uppercase;color:var(--muted)}
.chip .val{font-size:27px;font-weight:700;margin-top:5px;font-variant-numeric:tabular-nums;
word-break:break-word}
.chip.gt .val{color:var(--gt)}.chip.pred .val{color:var(--pred)}
.arrow{align-self:center;color:var(--muted);font-size:18px}
audio{width:100%;margin:4px 0;filter:invert(.9) hue-rotate(180deg)}
.hint{display:flex;justify-content:space-between;color:var(--muted);font-size:11px;margin-top:4px}
.hint .l{color:var(--wrong)}.hint .r{color:var(--right)}
.tags{display:flex;gap:6px;margin-top:8px;flex-wrap:wrap}
.tag{font-size:11px;padding:3px 8px;border-radius:20px;background:#141922;
border:1px solid var(--line);color:var(--muted)}
.tag.fixed{color:var(--fix);border-color:var(--fix)}
.tag.segbtn{cursor:pointer}
.tag.segbtn.on{color:#141922;background:var(--seg);border-color:var(--seg);font-weight:700}
.stamp{position:absolute;top:22px;font-size:30px;font-weight:800;letter-spacing:2px;
padding:4px 12px;border:4px solid;border-radius:12px;opacity:0;transform:rotate(-14deg)}
.stamp.w{left:20px;color:var(--wrong);border-color:var(--wrong)}
.stamp.r{right:20px;color:var(--right);border-color:var(--right)}
footer{width:100%;max-width:520px;padding:10px 14px calc(16px + env(safe-area-inset-bottom));
display:flex;gap:8px;align-items:center;justify-content:center;flex-wrap:wrap}
button{font:inherit;border:none;border-radius:14px;padding:12px 16px;cursor:pointer;
font-weight:600;color:#fff}
.bwrong{background:var(--wrong)}.bright{background:var(--right)}.bfix{background:var(--fix)}
.bskip{background:#39424f}.bseg{background:#39424f;color:var(--seg);border:1px solid var(--seg)}
.bundo{background:transparent;color:var(--muted);border:1px solid var(--line)}
#done{display:none;text-align:center;padding:34px 20px}
#done h2{font-size:22px}#exp{max-width:440px;margin:14px auto;white-space:pre-wrap;
background:#141922;border:1px solid var(--line);border-radius:12px;padding:12px;
font:12px/1.4 ui-monospace,monospace;text-align:left;max-height:38vh;overflow:auto}
#net{position:fixed;top:6px;right:10px;font-size:10px;color:var(--muted);opacity:.7}
/* ── cylinder picker modal (lifted from chart_interactive.py) ── */
.pk-back{position:fixed;inset:0;background:rgba(0,0,0,.55);display:none;z-index:50;
align-items:flex-end;justify-content:center}
.pk-back.open{display:flex}
.pk-panel{width:100%;max-width:460px;background:#f7f3e9;color:#242018;
border-radius:18px 18px 0 0;padding:16px 16px calc(16px + env(safe-area-inset-bottom));
box-shadow:0 -10px 40px rgba(0,0,0,.5)}
.pk-handle{width:40px;height:4px;background:#cfc7ae;border-radius:2px;margin:0 auto 12px}
.pk-head{display:flex;align-items:center;gap:10px;margin-bottom:8px}
.pk-cur{font:700 22px system-ui;flex:1}
.pk-play{background:#efe9d9;color:#8a2b2b;border:1px solid #e2dac4;border-radius:8px;
padding:8px 12px;font-weight:700}
.pk-seed{display:flex;gap:6px;margin-bottom:8px}
.pk-seed button{flex:1;background:#efe9d9;color:#8a2b2b;border:1px solid #e2dac4;
border-radius:8px;padding:7px;font:600 12px system-ui}
.pk-seed button.nc{color:#7a2b8a}
.ce-cyl-labels{display:flex;width:100%;padding:0 1px}
.ce-cyl-labels span{flex:1;text-align:center;font:700 10px system-ui;text-transform:uppercase;
letter-spacing:.04em;color:#8a8371}
.ce-cyl-picker{position:relative;display:flex;width:100%;height:180px;background:#efe9d9;
border-radius:10px;border:1px solid #e2dac4;overflow:hidden}
.ce-cyl-highlight{position:absolute;left:0;right:0;top:50%;height:36px;margin-top:-18px;
border-top:1px solid #cfc7ae;border-bottom:1px solid #cfc7ae;background:#8a2b2b14;
pointer-events:none;z-index:2}
.ce-cyl-col{flex:1 1 0;min-width:0;height:180px;overflow-y:scroll;scroll-snap-type:y mandatory;
-webkit-overflow-scrolling:touch;scrollbar-width:none;padding:72px 0;position:relative;
border-right:1px solid #e2dac4}
.ce-cyl-col:last-child{border-right:none}
.ce-cyl-col::-webkit-scrollbar{display:none;width:0;height:0}
.ce-cyl-item{height:36px;display:flex;align-items:center;justify-content:center;
scroll-snap-align:center;font:600 13.5px system-ui;color:#4a463699;white-space:nowrap;padding:0 2px}
.ce-cyl-item.sel{color:#242018;font-weight:700;font-size:15px}
.pk-actions{display:flex;gap:8px;margin-top:12px}
.pk-actions button{flex:1;border-radius:10px;padding:12px}
.pk-save{background:#8a2b2b;color:#fff}.pk-cancel{background:#efe9d9;color:#8a2b2b;border:1px solid #e2dac4}
</style></head><body>
<div id="net"></div>
<header>
  <h1>Predictor vs GT <span class="sub">(__LEVEL__-level)</span></h1>
  <div class="bar"><i id="prog"></i></div>
  <div class="counts"><span id="pos">0</span>/__N__ · <b class="w" id="cw">0</b> wrong · <b class="r" id="cr">0</b> right · <b class="f" id="cf">0</b> fix · <span id="ci">0</span> skip · <b id="cs" style="color:var(--seg)">0</b> seg</div>
</header>
<div id="stage"></div>
<div id="done">
  <h2><span id="dn"></span> judged</h2>
  <p style="color:var(--muted)">Corrections are saved to the server ledger automatically. You can also export:</p>
  <div><button class="bright" onclick="copyJSON()">Copy JSON</button>
       <button class="bskip" onclick="downloadJSON()">Download .json</button>
       <button class="bundo" onclick="resetAll()">Reset all</button></div>
  <pre id="exp"></pre>
</div>
<footer id="ctrls">
  <button class="bundo" onclick="undo()">Undo</button>
  <button class="bwrong" onclick="swipe(-1)">◀ Wrong</button>
  <button class="bskip" onclick="swipe(0)">Skip</button>
  <button class="bfix" onclick="openPicker()">🎯 Correct</button>
  <button class="bright" onclick="swipe(1)">Right ▶</button>
  <button class="bseg" onclick="toggleSeg()" title="Mark the segmentation/boundary here as wrong (key: s)">⌇ Seg</button>
</footer>

<!-- cylinder picker -->
<div class="pk-back" id="pkBack">
 <div class="pk-panel">
  <div class="pk-handle"></div>
  <div class="pk-head"><span class="pk-cur" id="pkCur">C</span>
    <button class="pk-play" id="pkPlay">▶ hear</button></div>
  <div class="pk-seed">
    <button id="pkSeedGt">= GT</button>
    <button id="pkSeedPred">= Pred</button>
    <button class="nc" id="pkNC">No chord</button>
  </div>
  <div class="ce-cyl-labels"><span>Root</span><span>Family</span><span>7th</span><span>Ext</span><span>Bass</span></div>
  <div class="ce-cyl-picker">
    <div class="ce-cyl-highlight"></div>
    <div class="ce-cyl-col" id="pkRoot"></div>
    <div class="ce-cyl-col" id="pkFam"></div>
    <div class="ce-cyl-col" id="pkSev"></div>
    <div class="ce-cyl-col" id="pkExt"></div>
    <div class="ce-cyl-col" id="pkBass"></div>
  </div>
  <div class="pk-actions">
    <button class="pk-cancel" id="pkCancel">Cancel</button>
    <button class="pk-save" id="pkSave">Save correct chord</button>
  </div>
 </div>
</div>

<script>
const CARDS=__DATA__;
const KEY="__STOREKEY__";const APIBASE="__APIBASE__";
const PC=["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"];
// picker token tables (lifted from chart_interactive.py)
const FAMILIES=[{id:'maj',label:'Maj'},{id:'min',label:'Min'},{id:'dim',label:'Dim'},
 {id:'aug',label:'Aug'},{id:'sus',label:'Sus'}];
const SEV_BY_FAM={
 maj:[{q:'',label:'Triad'},{q:'6',label:'6'},{q:'^7',label:'Maj7'},{q:'7',label:'Dom7'}],
 min:[{q:'-',label:'Triad'},{q:'-7',label:'Min7'},{q:'-^7',label:'MinMaj7'}],
 dim:[{q:'o',label:'Triad'},{q:'h7',label:'m7b5'},{q:'o7',label:'Dim7'}],
 aug:[{q:'+',label:'Triad'},{q:'+7',label:'Aug7'},{q:'+^7',label:'AugMaj7'}],
 sus:[{q:'sus4',label:'Sus4'},{q:'sus2',label:'Sus2'},{q:'7sus',label:'7Sus4'}]};
const EXT_BY_SEV={
 '':[{q:'',label:'—'}],'6':[{q:'6',label:'—'}],
 '^7':[{q:'^7',label:'7'},{q:'^9',label:'9'},{q:'^9#11',label:'9#11'},{q:'^13',label:'13'}],
 '7':[{q:'7',label:'7'},{q:'9',label:'9'},{q:'7b9',label:'b9'},{q:'7#9',label:'#9'},{q:'13',label:'13'}],
 '-':[{q:'-',label:'—'}],
 '-7':[{q:'-7',label:'7'},{q:'-9',label:'9'},{q:'-11',label:'11'},{q:'-13',label:'13'}],
 '-^7':[{q:'-^7',label:'—'}],'o':[{q:'o',label:'—'}],'h7':[{q:'h7',label:'—'}],'o7':[{q:'o7',label:'—'}],
 '+':[{q:'+',label:'—'}],'+7':[{q:'+7',label:'—'}],'+^7':[{q:'+^7',label:'—'}],
 'sus4':[{q:'sus4',label:'Triad'},{q:'9sus',label:'9'}],'sus2':[{q:'sus2',label:'—'}],
 '7sus':[{q:'7sus',label:'7'},{q:'9sus',label:'9'}]};
const extFor=s=>EXT_BY_SEV[s]||[{q:s,label:'—'}];
// picker ext token -> scorer vocab quality + human suffix
const VOCAB_FROM_PK={'':'maj','6':'6','^7':'maj7','7':'7','9':'9','7b9':'7','7#9':'7','13':'13',
 '^9':'maj9','^9#11':'maj9','^13':'13','-':'min','-7':'min7','-9':'min9','-11':'11','-13':'13',
 '-^7':'minmaj7','o':'dim','h7':'hdim7','o7':'dim7','+':'aug','+7':'aug','+^7':'aug',
 'sus4':'sus4','sus2':'sus2','7sus':'sus4','9sus':'sus4'};
const Q_SUFFIX={'':'','6':'6','^7':'maj7','7':'7','9':'9','7b9':'7b9','7#9':'7#9','13':'13',
 '^9':'maj9','^9#11':'maj9#11','^13':'maj13','-':'m','-7':'m7','-9':'m9','-11':'m11','-13':'m13',
 '-^7':'mMaj7','o':'dim','h7':'m7b5','o7':'dim7','+':'aug','+7':'aug7','+^7':'augMaj7',
 'sus4':'sus4','sus2':'sus2','7sus':'7sus4','9sus':'9sus4'};
const CE_IV={'':[0,4,7],'-':[0,3,7],'7':[0,4,7,10],'^7':[0,4,7,11],'-7':[0,3,7,10],
 'h7':[0,3,6,10],'o7':[0,3,6,9],'o':[0,3,6],'6':[0,4,7,9],'sus4':[0,5,7],'sus2':[0,2,7],'+':[0,4,8]};

let state=JSON.parse(localStorage.getItem(KEY)||"{}");   // id -> {verdict, correction}
let idx=0, cur=null;
const stage=document.getElementById('stage');
const netEl=document.getElementById('net');

function judged(id){return !!(state[id]&&state[id].verdict);}
function firstUnjudged(){let i=CARDS.findIndex(c=>!judged(c.id));return i<0?CARDS.length:i;}

// ── server ledger (durable). Falls back to localStorage-only if absent. ──
async function loadServer(){
  try{const r=await fetch(APIBASE+'/state',{cache:'no-store'});if(!r.ok)throw 0;
    const srv=await r.json();Object.assign(state,srv);localStorage.setItem(KEY,JSON.stringify(state));
    netEl.textContent='● ledger';}catch(e){netEl.textContent='○ local only';}
}
function save(id){localStorage.setItem(KEY,JSON.stringify(state));
  fetch(APIBASE+'/verdict',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(recordFor(id))}).catch(()=>{});}
function recordFor(id){const c=CARDS.find(x=>x.id===id),v=state[id]||{};
  return {id,song:c.song,from:c.t0,to:c.t1,gt:c.gt,pred:c.pred,
    verdict:v.verdict,correction:v.correction||null,seg_issue:!!v.seg_issue,
    ts:new Date().toISOString()};}

function fmt(t){const m=Math.floor(t/60),s=Math.floor(t%60);return m+":"+String(s).padStart(2,'0');}
function b64toURL(b){const bin=atob(b),a=new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++)a[i]=bin.charCodeAt(i);
  return URL.createObjectURL(new Blob([a],{type:'audio/mpeg'}));}

function counts(){let w=0,r=0,f=0,s=0,sg=0,done=0;for(const k in state){const st=state[k];
  if(st.seg_issue)sg++;const v=st.verdict;if(!v)continue;done++;
  if(v==='wrong')w++;else if(v==='right')r++;else if(v==='correct')f++;else if(v==='skip')s++;}
  cw.textContent=w;cr.textContent=r;cf.textContent=f;ci.textContent=s;cs.textContent=sg;
  pos.textContent=Math.min(done,CARDS.length);prog.style.width=(100*done/CARDS.length)+'%';}

function render(){
  if(cur){cur.remove();cur=null;}
  if(idx>=CARDS.length){finish();return;}
  const c=CARDS[idx];const mine=state[c.id];
  const el=document.createElement('div');el.className='card';
  el.innerHTML=`
   <div class="stamp w">WRONG</div><div class="stamp r">RIGHT</div>
   <div class="meta"><span>${c.title}</span><span>${fmt(c.t0)}–${fmt(c.t1)}</span></div>
   <div class="chords">
     <div class="chip gt"><div class="lbl">ground truth</div><div class="val">${c.gt}</div></div>
     <div class="arrow">vs</div>
     <div class="chip pred"><div class="lbl">predicted</div><div class="val">${c.pred}</div></div>
   </div>
   <audio id="au" controls autoplay></audio>
   <div class="tags">
     <span class="tag">root ${c.root_ok?'✓':'✗'}</span>
     <span class="tag">family ${c.family_ok?'✓':'✗'}</span>
     <span class="tag">${(c.t1-c.t0).toFixed(1)}s</span>
     ${mine&&mine.correction?`<span class="tag fixed">✎ ${mine.correction.label}</span>`:''}
     <span class="tag segbtn${mine&&mine.seg_issue?' on':''}" id="segchip">⌇ boundary wrong</span>
   </div>
   <div class="hint"><span class="l">◀ predictor wrong</span><span class="r">predictor right ▶</span></div>`;
  stage.appendChild(el);cur=el;
  el.querySelector('#segchip').addEventListener('click',ev=>{ev.stopPropagation();toggleSeg();});
  const au=el.querySelector('#au');au.src=b64toURL(c.b64);
  const ra=c.region_a,rb=c.region_b;
  au.addEventListener('loadedmetadata',()=>{try{au.currentTime=Math.max(0,ra-0.15);}catch(e){}});
  au.addEventListener('timeupdate',()=>{if(au.currentTime>=rb+0.25)au.currentTime=Math.max(0,ra-0.1);});
  attachDrag(el);counts();
}

function attachDrag(el){let sx=0,sy=0,dx=0,active=false;
  const down=e=>{if(e.target.closest('audio'))return;active=true;
    const p=e.touches?e.touches[0]:e;sx=p.clientX;sy=p.clientY;el.style.transition='none';};
  const move=e=>{if(!active)return;const p=e.touches?e.touches[0]:e;dx=p.clientX-sx;
    el.style.transform=`translate(${dx}px,${(p.clientY-sy)*0.25}px) rotate(${dx/18}deg)`;
    el.querySelector('.stamp.w').style.opacity=dx<0?Math.min(1,-dx/90):0;
    el.querySelector('.stamp.r').style.opacity=dx>0?Math.min(1,dx/90):0;};
  const up=()=>{if(!active)return;active=false;el.style.transition='transform .25s';
    if(dx>90)fly(1);else if(dx<-90)fly(-1);
    else{el.style.transform='';el.querySelector('.stamp.w').style.opacity=0;
      el.querySelector('.stamp.r').style.opacity=0;}dx=0;};
  el.addEventListener('mousedown',down);window.addEventListener('mousemove',move);
  window.addEventListener('mouseup',up);el.addEventListener('touchstart',down,{passive:true});
  el.addEventListener('touchmove',move,{passive:true});el.addEventListener('touchend',up);}

function fly(dir){const el=cur;if(!el)return;
  el.style.transform=`translate(${dir*600}px,60px) rotate(${dir*30}deg)`;el.style.opacity=0;
  const c=CARDS[idx];const prev=state[c.id]||{};
  state[c.id]={verdict:dir>0?'right':'wrong',correction:prev.correction||null,seg_issue:!!prev.seg_issue};
  save(c.id);idx++;setTimeout(render,180);}

function swipe(dir){const c=CARDS[idx];if(!c)return;
  const prev=state[c.id]||{};
  state[c.id]={verdict:dir>0?'right':dir<0?'wrong':'skip',correction:prev.correction||null,seg_issue:!!prev.seg_issue};
  save(c.id);if(dir===0){idx++;render();}else fly(dir);}

function toggleSeg(){const c=CARDS[idx];if(!c)return;const prev=state[c.id]||{};
  state[c.id]={verdict:prev.verdict,correction:prev.correction||null,seg_issue:!prev.seg_issue};
  save(c.id);const chip=document.getElementById('segchip');
  if(chip)chip.classList.toggle('on',state[c.id].seg_issue);counts();}

function undo(){if(idx<=0)return;idx--;delete state[CARDS[idx].id];save(CARDS[idx].id);
  document.getElementById('done').style.display='none';ctrls.style.display='flex';render();}

function finish(){ctrls.style.display='none';const d=document.getElementById('done');
  d.style.display='block';dn.textContent=Object.keys(state).length;exp.textContent=exportStr();counts();}

function exportObj(){return {tool:"pred-tinder",level:"__LEVEL__",date:new Date().toISOString(),
  verdicts:CARDS.filter(c=>c.id in state).map(c=>recordFor(c.id))};}
function exportStr(){return JSON.stringify(exportObj(),null,1);}
function copyJSON(){navigator.clipboard.writeText(exportStr());}
function downloadJSON(){const b=new Blob([exportStr()],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(b);
  a.download='pred_tinder_verdicts.json';a.click();}
function resetAll(){if(!confirm('Clear all verdicts?'))return;state={};localStorage.setItem(KEY,'{}');
  fetch(APIBASE+'/reset',{method:'POST'}).catch(()=>{});idx=0;
  document.getElementById('done').style.display='none';ctrls.style.display='flex';render();}

// ── cylinder picker ──
const pkBack=document.getElementById('pkBack');
let pk={root:0,fam:'maj',sev:'',ext:'',bass:0};
const H=36;
function buildCol(col,items,sel){col._items=items;col.innerHTML='';
  items.forEach((it,i)=>{const d=document.createElement('div');
    d.className='ce-cyl-item'+(i===sel?' sel':'');d.textContent=it.label;col.appendChild(d);});
  col.scrollTop=sel*H;}
function mark(col,i){[...col.children].forEach((d,j)=>d.classList.toggle('sel',j===i));}
function snapIdx(col){const n=(col._items||[]).length;if(!n)return 0;
  return Math.max(0,Math.min(n-1,Math.round(col.scrollTop/H)));}
const timers=new WeakMap();
function onScroll(col,cb){col.addEventListener('scroll',()=>{clearTimeout(timers.get(col));
  timers.set(col,setTimeout(()=>{const i=snapIdx(col);mark(col,i);
    if(navigator.vibrate)navigator.vibrate(4);cb(col._items[i]);},90));},{passive:true});}
const colRoot=pkRootEl(),colFam=document.getElementById('pkFam'),colSev=document.getElementById('pkSev'),
 colExt=document.getElementById('pkExt'),colBass=document.getElementById('pkBass');
function pkRootEl(){return document.getElementById('pkRoot');}
function renderRoot(){buildCol(colRoot,PC.map((n,pc)=>({pc,label:n})),pk.root);}
function renderFam(){const i=Math.max(0,FAMILIES.findIndex(f=>f.id===pk.fam));
  buildCol(colFam,FAMILIES.map(f=>({id:f.id,label:f.label})),i);}
function renderSev(){const o=SEV_BY_FAM[pk.fam]||[];const i=Math.max(0,o.findIndex(x=>x.q===pk.sev));
  buildCol(colSev,o.map(x=>({q:x.q,label:x.label})),i);}
function renderExt(){const o=extFor(pk.sev);const i=Math.max(0,o.findIndex(x=>x.q===pk.ext));
  buildCol(colExt,o.map(x=>({q:x.q,label:x.label})),i);}
function renderBass(){const items=[{pc:pk.root,label:'root'}].concat(PC.map((n,pc)=>({pc,label:n})));
  const i=Math.max(0,items.findIndex((x,k)=>k>0&&x.pc===pk.bass&&pk.bass!==pk.root));
  buildCol(colBass,items,pk.bass===pk.root?0:i);}
onScroll(colRoot,it=>{pk.root=it.pc;renderBass();pkPreview();});
onScroll(colFam,it=>{pk.fam=it.id;pk.sev=(SEV_BY_FAM[pk.fam]||[{q:''}])[0].q;renderSev();
  pk.ext=extFor(pk.sev)[0].q;renderExt();pkPreview();});
onScroll(colSev,it=>{pk.sev=it.q;pk.ext=extFor(pk.sev)[0].q;renderExt();pkPreview();});
onScroll(colExt,it=>{pk.ext=it.q;pkPreview();});
onScroll(colBass,it=>{pk.bass=it.pc;pkPreview();});
function pkLabel(){return PC[pk.root]+':'+VOCAB_FROM_PK[pk.ext]+(pk.bass!==pk.root?'/'+PC[pk.bass]:'');}
function pkDisplay(){return PC[pk.root]+(Q_SUFFIX[pk.ext]??'')+(pk.bass!==pk.root?'/'+PC[pk.bass]:'');}
function pkPreview(){document.getElementById('pkCur').textContent=pkDisplay();}
function pkSeed(s){if(!s){pk={root:0,fam:'maj',sev:'',ext:'',bass:0};}
  else{pk={root:s.root,fam:s.fam,sev:s.sev,ext:s.ext,bass:s.bass};}
  renderRoot();renderFam();renderSev();renderExt();renderBass();pkPreview();}
function openPicker(){const c=CARDS[idx];if(!c)return;
  const mine=state[c.id];pkSeed((mine&&mine.correction&&mine.correction.seed)||c.pred_seed||c.gt_seed);
  pkBack.classList.add('open');}
function closePicker(){pkBack.classList.remove('open');}
document.getElementById('pkCancel').addEventListener('click',closePicker);
pkBack.addEventListener('click',e=>{if(e.target===pkBack)closePicker();});
document.getElementById('pkSeedGt').addEventListener('click',()=>pkSeed(CARDS[idx].gt_seed));
document.getElementById('pkSeedPred').addEventListener('click',()=>pkSeed(CARDS[idx].pred_seed));
document.getElementById('pkNC').addEventListener('click',()=>{
  const c=CARDS[idx];const prev=state[c.id]||{};
  state[c.id]={verdict:'correct',correction:{label:'N',root:null,quality:'N',bass:null,seed:null},
    seg_issue:!!prev.seg_issue};
  save(c.id);closePicker();advanceAfterFix();});
document.getElementById('pkSave').addEventListener('click',()=>{
  const c=CARDS[idx];const prev=state[c.id]||{};
  const corr={label:pkLabel(),root:pk.root,quality:VOCAB_FROM_PK[pk.ext],
    bass:(pk.bass!==pk.root?pk.bass:pk.root),seed:{...pk}};
  state[c.id]={verdict:'correct',correction:corr,seg_issue:!!prev.seg_issue};
  save(c.id);closePicker();advanceAfterFix();});
document.getElementById('pkPlay').addEventListener('click',()=>playArp(pk.root,pk.sev||pk.ext));
function advanceAfterFix(){if(cur){cur.style.transition='transform .2s';
  cur.style.transform='translateY(-40px)';cur.style.opacity=0;}idx++;setTimeout(render,180);}
let actx=null;
function playArp(root,q){actx=actx||new (window.AudioContext||window.webkitAudioContext)();
  if(actx.state==='suspended')actx.resume();
  const iv=CE_IV[q]||CE_IV[''];const base=220*Math.pow(2,root/12);
  iv.forEach((s,i)=>{const f=base*Math.pow(2,s/12),t=actx.currentTime+i*0.09;
    const o=actx.createOscillator(),g=actx.createGain();o.type='triangle';o.frequency.value=f;
    g.gain.setValueAtTime(0,t);g.gain.linearRampToValueAtTime(0.18,t+0.015);
    g.gain.exponentialRampToValueAtTime(0.0001,t+0.42);o.connect(g);g.connect(actx.destination);
    o.start(t);o.stop(t+0.44);});}

window.addEventListener('keydown',e=>{
  if(pkBack.classList.contains('open')){if(e.key==='Escape')closePicker();return;}
  if(e.key==='ArrowLeft')swipe(-1);
  else if(e.key==='ArrowRight')swipe(1);
  else if(e.key==='ArrowDown'||e.key===' '){e.preventDefault();swipe(0);}
  else if(e.key==='c'||e.key==='ArrowUp'){e.preventDefault();openPicker();}
  else if(e.key==='s'){e.preventDefault();toggleSeg();}
  else if(e.key==='Backspace'){e.preventDefault();undo();}
  else if(e.key==='r'||e.key==='p'){const au=document.getElementById('au');
    if(au){au.currentTime=Math.max(0,(CARDS[idx]?.region_a||0)-0.1);au.play();}}
});

(async()=>{await loadServer();idx=firstUnjudged();render();})();
</script></body></html>"""


if __name__ == "__main__":
    sys.exit(main())
