"""Build the phone-first ground-truth triage page."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
SCRATCH = Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-Code-harmonia/"
               "22c6b747-cea4-410f-85ae-f6ed58a9d2eb/scratchpad")
OUT = REPO / "docs" / "research_sessions" / "gt_triage_2026-07-27.html"

TITLES = {
    "bein_green": "Bein' Green",
    "blue_bossa": "Blue Bossa (live recording)",
    "blue_bossa_backing": "Blue Bossa (backing track)",
    "close_to_you": "Close to You",
    "every_breath_you_take": "Every Breath You Take",
    "georgia_on_my_mind": "Georgia on My Mind",
    "stand_by_me": "Stand by Me",
}
BUCKETS = ["needs-ear", "drift", "extension-only", "auto-ok"]
BNAME = {"needs-ear": "Needs your ear", "drift": "Timing off",
         "extension-only": "Colour tone only", "auto-ok": "Confirmed"}


def mmss(t):
    return f"{int(t)//60}:{int(t) % 60:02d}"


def build():
    songs = json.loads((SCRATCH / "gt_triage.json").read_text())
    clips = json.loads((SCRATCH / "triage_clips.json").read_text())
    colour = json.loads((SCRATCH / "colour.json").read_text())

    data = {"songs": {}, "queue": [], "colour": [], "clips": clips}
    tot = {b: 0.0 for b in BUCKETS}
    for sid, S in songs.items():
        agg = {b: 0.0 for b in BUCKETS}
        ribbon = []
        for r in S["regions"]:
            d = r["t1"] - r["t0"]
            agg[r["bucket"]] += d
            tot[r["bucket"]] += d
            ribbon.append([round(r["t0"], 1), round(r["t1"], 1),
                           BUCKETS.index(r["bucket"]), r["region_idx"]])
        t = S["timing"]
        data["songs"][sid] = {
            "title": TITLES[sid], "dur": round(S["duration_s"], 1),
            "span": [round(S["gt_span"][0], 1), round(S["gt_span"][1], 1)],
            "n_chords": S["n_gt_chords"], "n_regions": len(S["regions"]),
            "agg": {b: round(agg[b], 1) for b in BUCKETS},
            "ribbon": ribbon,
            "timing": {k: (round(v, 3) if isinstance(v, float) else v)
                       for k, v in t.items()},
        }
        _set_spelling([c["label"] for r in S["regions"] for c in r["chords"]])
        for r in S["regions"]:
            if r["bucket"] not in ("needs-ear", "drift"):
                continue
            key = f"{sid}:{r['region_idx']}"
            data["queue"].append({
                "id": key, "song": sid, "idx": r["region_idx"],
                "t0": round(r["t0"], 2), "t1": round(r["t1"], 2),
                "bucket": r["bucket"], "doubt": round(r["doubt"], 2),
                "reasons": r["reasons"],
                "chords": [{"t0": round(c["t0"], 2), "t1": round(c["t1"], 2),
                            "l": c["label"],
                            "alt": (c.get("best") and
                                    _lab(c["best"])) or None,
                            "musx": (f"{_pc(c['musx_top_root'])}:{c['musx_top_triad']}"
                                     if c.get("musx_top_root") is not None else None)}
                           for c in r["chords"]],
                "ev": {
                    "root": _r(r["musx_root_agree"]),
                    "triad": _r(r["musx_triad_agree"]),
                    "bass": _r(r["musx_bass_agree"]),
                    "fit": _r(r["gt_fit"]),
                    "sib": _r(r["sib_gap"]), "sibn": r["sib_n"],
                    "off": _r(r["off_med"]), "offn": r["off_n"],
                    "coff": _r(r["coff_med"]),
                    "nc": _r(r["musx_nc"]), "rms": _r(r["rms_db"]),
                    "conflict": r["timing_conflict"], "noev": r["no_evidence"],
                },
                "offs": r["off_detail"],
            })
    data["queue"].sort(key=lambda x: -x["doubt"])
    for x in colour:
        x = dict(x)
        x["song_title"] = TITLES[x["song"]]
        x["at"] = mmss(x["t0"])
        data["colour"].append(x)
    data["total"] = {b: round(tot[b], 1) for b in BUCKETS}
    data["grand"] = round(sum(tot.values()), 1)

    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"wrote {OUT}  ({OUT.stat().st_size/1e6:.2f} MB)")
    print("queue:", len(data["queue"]), "colour flags:", len(data["colour"]))


_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
_SPELL = _FLAT          # set per song from the chart's own accidentals


def _set_spelling(gt_labels):
    """Spell OUR labels the way this song's chart spells its own."""
    global _SPELL
    sharps = sum(l.count("#") for l in gt_labels if l)
    flats = sum(l.count("b") for l in gt_labels if l and ":" in l
                and "b" in l.split(":")[0])
    _SPELL = _SHARP if sharps >= flats else _FLAT


def _pc(i):
    return _SPELL[i]


def _lab(b):
    return f"{_SPELL[b[0]]}:{b[1]}"


def _r(v):
    return None if v is None else round(v, 3)


TEMPLATE = r"""<title>Ground truth, region by region — what still needs your ear</title>
<style>
.gt{
  --surface-1:#fcfcfb; --plane:#f9f9f7;
  --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,0.10);
  --good:#0ca30c; --warning:#fab219; --serious:#ec835a; --critical:#d03b3b;
  --chip-good:rgba(12,163,12,.12); --chip-warn:rgba(250,178,25,.18);
  --chip-ser:rgba(236,131,90,.18); --chip-crit:rgba(208,59,59,.14);
  color-scheme:light;
  background:var(--plane); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
  font-size:16px; line-height:1.55; -webkit-text-size-adjust:100%;
  padding:0 0 6rem; margin:0; overflow-x:hidden;
}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .gt{
  --surface-1:#1a1a19; --plane:#0d0d0d; --ink:#fff; --ink-2:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,0.10);
  --chip-good:rgba(12,163,12,.20); --chip-warn:rgba(250,178,25,.18);
  --chip-ser:rgba(236,131,90,.18); --chip-crit:rgba(208,59,59,.22);
  color-scheme:dark;}}
:root[data-theme="dark"] .gt{
  --surface-1:#1a1a19; --plane:#0d0d0d; --ink:#fff; --ink-2:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,0.10);
  --chip-good:rgba(12,163,12,.20); --chip-warn:rgba(250,178,25,.18);
  --chip-ser:rgba(236,131,90,.18); --chip-crit:rgba(208,59,59,.22);
  color-scheme:dark;}
.gt *{box-sizing:border-box;}
.wrap{max-width:44rem;margin:0 auto;padding:1.25rem 1rem 0;}
h1{font-size:1.5rem;line-height:1.2;margin:.4rem 0 .5rem;letter-spacing:-.01em;}
h2{font-size:1.06rem;margin:2rem 0 .5rem;letter-spacing:-.005em;}
h3{font-size:.95rem;margin:1.2rem 0 .4rem;}
p{margin:.55rem 0;color:var(--ink-2);}
.lede{font-size:1.02rem;color:var(--ink-2);}
.small{font-size:.83rem;color:var(--muted);}
.card{background:var(--surface-1);border:1px solid var(--ring);border-radius:14px;
  padding:1rem;margin:.85rem 0;}
.chip{display:inline-flex;align-items:center;gap:.35rem;font-size:.76rem;
  font-weight:600;padding:.2rem .55rem;border-radius:999px;white-space:nowrap;}
.c-needs{background:var(--chip-crit);color:var(--critical);}
.c-drift{background:var(--chip-ser);color:var(--serious);}
.c-ext{background:var(--chip-warn);color:#7a5400;}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .gt .c-ext{color:var(--warning);}}
:root[data-theme="dark"] .gt .c-ext{color:var(--warning);}
.c-ok{background:var(--chip-good);color:var(--good);}
.dot{width:.6rem;height:.6rem;border-radius:50%;flex:0 0 auto;}
/* tabs */
.tabs{display:flex;gap:.4rem;overflow-x:auto;padding:.2rem 0 .6rem;
  position:sticky;top:0;background:var(--plane);z-index:5;
  scrollbar-width:none;}
.tabs::-webkit-scrollbar{display:none;}
.tab{flex:0 0 auto;min-height:44px;padding:.55rem .85rem;border-radius:11px;
  border:1px solid var(--ring);background:var(--surface-1);color:var(--ink-2);
  font-size:.86rem;font-weight:600;cursor:pointer;font-family:inherit;}
.tab[aria-selected="true"]{background:var(--ink);color:var(--plane);
  border-color:var(--ink);}
.pane[hidden]{display:none;}
/* stacked bars */
.bar{display:flex;height:14px;border-radius:7px;overflow:hidden;
  background:var(--grid);margin:.3rem 0 .15rem;}
.bar>span{display:block;height:100%;}
.b-needs{background:var(--critical);} .b-drift{background:var(--serious);}
.b-ext{background:var(--warning);} .b-ok{background:var(--good);}
.legend{display:flex;flex-wrap:wrap;gap:.5rem .9rem;margin:.5rem 0 .2rem;}
.legend span{display:inline-flex;align-items:center;gap:.35rem;font-size:.79rem;
  color:var(--ink-2);}
.rowlab{display:flex;justify-content:space-between;gap:.6rem;font-size:.86rem;
  align-items:baseline;}
.rowlab b{font-weight:600;color:var(--ink);}
.rowlab i{font-style:normal;color:var(--muted);font-size:.79rem;
  font-variant-numeric:tabular-nums;}
/* queue card */
.q-head{display:flex;justify-content:space-between;align-items:flex-start;
  gap:.6rem;flex-wrap:wrap;}
.q-title{font-weight:650;font-size:1rem;color:var(--ink);}
.q-time{font-variant-numeric:tabular-nums;color:var(--muted);font-size:.85rem;}
audio{width:100%;margin:.7rem 0 .2rem;height:44px;}
.claim{margin:.6rem 0 0;font-size:.9rem;}
.chords{display:flex;flex-wrap:wrap;gap:.3rem;margin:.35rem 0 .1rem;}
.ch{font-size:.82rem;padding:.2rem .5rem;border-radius:7px;background:var(--grid);
  color:var(--ink);font-variant-numeric:tabular-nums;}
.why{margin:.55rem 0 0;padding-left:1.05rem;}
.why li{margin:.25rem 0;font-size:.89rem;color:var(--ink-2);}
.meters{margin:.7rem 0 0;}
.meter{display:grid;grid-template-columns:8.5rem 1fr 3.2rem;gap:.5rem;
  align-items:center;margin:.28rem 0;font-size:.8rem;color:var(--ink-2);}
.meter .track{height:8px;border-radius:4px;background:var(--grid);position:relative;}
.meter .fill{height:100%;border-radius:4px;background:var(--ink-2);}
.meter .val{text-align:right;font-variant-numeric:tabular-nums;color:var(--ink);}
.ribbon{width:100%;height:26px;display:block;margin:.6rem 0 .1rem;}
.verdict{display:flex;gap:.4rem;margin:.8rem 0 0;flex-wrap:wrap;}
.vb{flex:1 1 6.5rem;min-height:46px;border-radius:11px;border:1px solid var(--ring);
  background:var(--surface-1);color:var(--ink);font-size:.87rem;font-weight:600;
  cursor:pointer;font-family:inherit;padding:.5rem;}
.vb[aria-pressed="true"][data-v="ok"]{background:var(--good);border-color:var(--good);color:#fff;}
.vb[aria-pressed="true"][data-v="bad"]{background:var(--critical);border-color:var(--critical);color:#fff;}
.vb[aria-pressed="true"][data-v="idk"]{background:var(--muted);border-color:var(--muted);color:#fff;}
.sticky{position:fixed;left:0;right:0;bottom:0;background:var(--surface-1);
  border-top:1px solid var(--ring);padding:.6rem 1rem;z-index:20;}
.sticky .inner{max-width:44rem;margin:0 auto;display:flex;gap:.5rem;
  align-items:center;justify-content:space-between;}
.copy{min-height:46px;padding:.6rem 1rem;border-radius:11px;border:none;
  background:var(--ink);color:var(--plane);font-weight:650;font-size:.9rem;
  cursor:pointer;font-family:inherit;}
table{width:100%;border-collapse:collapse;font-size:.83rem;}
th,td{text-align:left;padding:.4rem .5rem;border-bottom:1px solid var(--grid);
  vertical-align:top;}
th{color:var(--muted);font-weight:600;font-size:.76rem;text-transform:uppercase;
  letter-spacing:.03em;}
td.num{font-variant-numeric:tabular-nums;text-align:right;}
.scrollx{overflow-x:auto;-webkit-overflow-scrolling:touch;}
.note{border-left:3px solid var(--axis);padding:.1rem 0 .1rem .8rem;margin:.9rem 0;}
.kv{display:flex;flex-wrap:wrap;gap:.15rem .9rem;font-size:.82rem;color:var(--ink-2);}
.big{font-size:2rem;font-weight:650;letter-spacing:-.02em;color:var(--ink);
  line-height:1.1;}
details{margin:.6rem 0;} summary{cursor:pointer;font-size:.87rem;font-weight:600;
  color:var(--ink-2);min-height:30px;}
</style>

<div class="gt"><div class="wrap">

<h1>Ground truth, region by region</h1>
<p class="lede">You approved seven songs, but nobody sat through all 28 minutes of
them. So this treats the reference as <b>provisional in places</b> rather than
right everywhere: every passage was re-checked against evidence that never sees
our own chord output, and only the passages that came out genuinely doubtful are
asked of you.</p>

<div class="card" id="summary"></div>

<div class="tabs" role="tablist" id="tabs"></div>

<section class="pane" id="pane-queue">
  <p class="small" id="queue-intro"></p>
  <div id="queue"></div>
</section>

<section class="pane" id="pane-drift" hidden>
  <div id="drift"></div>
</section>

<section class="pane" id="pane-colour" hidden>
  <div id="colour"></div>
</section>

<section class="pane" id="pane-ok" hidden>
  <div id="okpane"></div>
</section>

<section class="pane" id="pane-how" hidden>
  <div id="how"></div>
</section>

</div>

<div class="sticky"><div class="inner">
  <span class="small" id="progress"></span>
  <button class="copy" id="copybtn">Copy my verdicts</button>
</div></div>
</div>

<script>
const D = __DATA__;
const BK = ["needs-ear","drift","extension-only","auto-ok"];
const BCLS = {"needs-ear":"needs","drift":"drift","extension-only":"ext","auto-ok":"ok"};
const BLAB = {"needs-ear":"Needs your ear","drift":"Timing off",
              "extension-only":"Colour tone only","auto-ok":"Confirmed"};
const BCOL = {"needs-ear":"var(--critical)","drift":"var(--serious)",
              "extension-only":"var(--warning)","auto-ok":"var(--good)"};
const ICON = {"needs-ear":"◆","drift":"◑","extension-only":"△","auto-ok":"✓"};
const KEY = "harmonia-gt-triage-verdicts";
let V = {};
try { V = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { V = {}; }

const mmss = t => Math.floor(t/60) + ":" + String(Math.floor(t%60)).padStart(2,"0");
const mins = s => (s/60).toFixed(1);
const esc = s => String(s).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));

/* ---------- summary ---------- */
function stackBar(agg, total){
  return '<div class="bar">' + BK.map(b => {
    const w = 100*(agg[b]||0)/total;
    return w > 0 ? `<span class="b-${BCLS[b]}" style="width:${w}%"></span>` : "";
  }).join("") + '</div>';
}
function renderSummary(){
  const G = D.grand;
  let h = `<div class="big">${mins(D.total["auto-ok"])} min confirmed &middot; ${mins(D.total["needs-ear"])} min for you</div>`;
  h += `<p class="small">Out of ${mins(G)} minutes of reference across all seven songs.
        Percentages are share of playing time, not share of chords.</p>`;
  h += '<div class="legend">' + BK.map(b =>
      `<span><i class="dot" style="background:${BCOL[b]}"></i>${BLAB[b]} &mdash; ${mins(D.total[b])} min (${Math.round(100*D.total[b]/G)}%)</span>`
    ).join("") + '</div>';
  h += stackBar(D.total, G);
  h += '<details style="margin-top:.8rem"><summary>Song by song</summary><div>';
  for (const sid of Object.keys(D.songs)){
    const S = D.songs[sid], t = Object.values(S.agg).reduce((a,b)=>a+b,0);
    h += `<div class="rowlab" style="margin-top:.7rem"><b>${esc(S.title)}</b><i>${mins(t)} min &middot; ${S.n_chords} chords</i></div>`;
    h += stackBar(S.agg, t);
    h += `<div class="small">${BK.filter(b=>S.agg[b]>0).map(b=>`${Math.round(100*S.agg[b]/t)}% ${BLAB[b].toLowerCase()}`).join(" &middot; ")}</div>`;
  }
  h += '</div></details>';
  document.getElementById("summary").innerHTML = h;
}

/* ---------- ribbon ---------- */
function ribbon(sid, hi){
  const S = D.songs[sid], W = 340, H = 26, pad = 1;
  const t0 = S.span[0], t1 = S.span[1], sp = Math.max(t1-t0, 1);
  let r = `<svg class="ribbon" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="where this passage sits in the song">`;
  r += `<rect x="0" y="8" width="${W}" height="10" rx="5" fill="var(--grid)"></rect>`;
  for (const [a,b,bi,idx] of S.ribbon){
    const x = pad + (W-2*pad)*(a-t0)/sp, w = Math.max((W-2*pad)*(b-a)/sp - 1, 1.2);
    r += `<rect x="${x.toFixed(1)}" y="8" width="${w.toFixed(1)}" height="10" fill="${BCOL[BK[bi]]}" opacity="${idx===hi?1:.42}"></rect>`;
    if (idx === hi){
      r += `<rect x="${x.toFixed(1)}" y="4" width="${Math.max(w,2.5).toFixed(1)}" height="18" fill="none" stroke="var(--ink)" stroke-width="1.5" rx="2"></rect>`;
    }
  }
  r += `</svg>`;
  return r;
}

/* ---------- meters ---------- */
function meter(label, val, txt, good){
  const pct = Math.max(0, Math.min(1, val));
  return `<div class="meter"><span>${label}</span>
    <span class="track"><span class="fill" style="width:${(pct*100).toFixed(0)}%;background:${good}"></span></span>
    <span class="val">${txt}</span></div>`;
}
function evidenceBlock(q){
  const e = q.ev; let h = '<div class="meters">';
  if (e.noev){
    const why = (e.nc !== null && e.nc > 0.4)
      ? `the outside model returns &ldquo;no chord&rdquo; for ${Math.round(e.nc*100)}% of this passage`
      : `this passage sits ${Math.abs(e.rms).toFixed(1)} dB below the song&rsquo;s own level`;
    h += `<div class="note" style="margin:.2rem 0 .6rem"><span class="small">
      <b>Not measured.</b> ${why}, so the agreement figures below are an absence of
      evidence, not evidence against the chart. Only your ear settles this one.</span></div>`;
  }
  if (e.root !== null)
    h += meter("Same root", e.root, Math.round(e.root*100)+"%",
               e.root>=.7?"var(--good)":e.root>=.5?"var(--warning)":"var(--critical)");
  if (e.triad !== null && e.root)
    h += meter("Major / minor", e.triad/Math.max(e.root,1e-6),
               Math.round(100*e.triad/Math.max(e.root,1e-6))+"%",
               (e.triad/Math.max(e.root,1e-6))>=.7?"var(--good)":"var(--warning)");
  if (e.bass !== null)
    h += meter("Same bass note", e.bass, Math.round(e.bass*100)+"%",
               e.bass>=.7?"var(--good)":e.bass>=.5?"var(--warning)":"var(--critical)");
  if (e.off !== null)
    h += meter("Bar lines", 1-Math.min(Math.abs(e.off)/1.0,1),
               (e.off>0?"+":"") + e.off.toFixed(2) + " s",
               Math.abs(e.off)<.3?"var(--good)":Math.abs(e.off)<.5?"var(--warning)":"var(--serious)");
  if (e.sib !== null && e.sibn > 0)
    h += meter("Vs its repeats", 1-Math.min(Math.max(e.sib,0)/.4,1),
               (e.sib>0?"+":"") + e.sib.toFixed(2),
               e.sib<.15?"var(--good)":"var(--warning)");
  h += '</div>';
  return h;
}

/* ---------- cards ---------- */
function card(q, rank){
  const S = D.songs[q.song];
  const cl = BCLS[q.bucket];
  let h = `<div class="card" id="card-${q.id.replace(":","-")}">`;
  h += `<div class="q-head"><div><div class="q-title">${esc(S.title)}</div>
        <div class="q-time">${mmss(q.t0)} &ndash; ${mmss(q.t1)} &middot; ${(q.t1-q.t0).toFixed(1)} s${rank?` &middot; #${rank} most doubtful`:""}</div></div>
        <span class="chip c-${cl}">${ICON[q.bucket]} ${BLAB[q.bucket]}</span></div>`;
  const c = D.clips[q.id];
  if (c){
    h += `<audio controls playsinline preload="metadata" src="data:audio/mpeg;base64,${c.b64}"></audio>`;
    h += `<div class="small">Starts ${(q.t0-c.a).toFixed(1)} s early so you hear the run-in.</div>`;
  } else {
    h += `<div class="small">No excerpt available for this passage.</div>`;
  }
  h += `<div class="claim"><b>The reference says</b></div><div class="chords">` +
       q.chords.map(c => `<span class="ch">${esc(c.l)}</span>`).join("") + `</div>`;
  const alts = q.chords.filter(c => c.musx && c.musx.split(":")[0] !== c.l.split(":")[0].split("/")[0]);
  if (alts.length){
    h += `<div class="claim"><b>The outside model hears instead</b></div><div class="chords">` +
      alts.map(c => `<span class="ch">${esc(c.l)} → ${esc(c.musx)}</span>`).join("") + `</div>`;
  }
  h += `<div class="claim"><b>Why it is here</b></div><ul class="why">` +
       q.reasons.map(r => `<li>${esc(r)}</li>`).join("") + `</ul>`;
  h += evidenceBlock(q);
  if (q.bucket === "drift" && q.offs && q.offs.length){
    h += `<details><summary>Where each chord change really lands</summary><div class="scrollx"><table>
      <tr><th>Written change</th><th>Heard at</th><th class="num">Off by</th></tr>` +
      q.offs.map(o => `<tr><td>${esc(o.from)} → ${esc(o.to)}<br><span class="small">${mmss(o.t)}</span></td>
        <td>${mmss(o.t + o.off)}</td><td class="num">${o.off>0?"+":""}${o.off.toFixed(2)} s</td></tr>`).join("") +
      `</table></div></details>`;
  }
  h += ribbon(q.song, q.idx);
  h += `<div class="small">Where this sits in the song. Every band is one passage,
        coloured by its verdict; the outlined one is this passage.</div>`;
  h += `<div class="verdict" data-id="${q.id}">
    <button class="vb" data-v="ok"  aria-pressed="false">Reference is right</button>
    <button class="vb" data-v="bad" aria-pressed="false">Reference is wrong</button>
    <button class="vb" data-v="idk" aria-pressed="false">Can&rsquo;t tell</button></div>`;
  h += `</div>`;
  return h;
}

/* ---------- panes ---------- */
function renderQueue(){
  const q = D.queue.filter(x => x.bucket === "needs-ear");
  document.getElementById("queue-intro").innerHTML =
    `${q.length} passages, worst first. Everything else was either confirmed by
     the evidence or is only a colour-tone question &mdash; those are in the other tabs.
     Judging by ear beats every number on this page; that is the point of the queue.`;
  document.getElementById("queue").innerHTML = q.map((x,i) => card(x, i+1)).join("");
}
function renderDrift(){
  const q = D.queue.filter(x => x.bucket === "drift");
  let h = `<p class="small">Here the chord <em>names</em> hold up but the bar lines do
    not sit where the reference puts them. Each song's overall slip is measured
    below; the passages after it are the ones where the local slip is at least
    0.30 s. Playable, same as the queue.</p>`;
  h += '<div class="card"><div class="scrollx"><table><tr><th>Song</th><th class="num">Typical slip</th>' +
       '<th class="num">Change across the song</th><th>Shape</th></tr>';
  for (const sid of Object.keys(D.songs)){
    const t = D.songs[sid].timing;
    const drifty = Math.abs(t.slope_s_per_min) >= 0.15;
    const shape = drifty
      ? `creeps from ${t.start>0?"+":""}${t.start.toFixed(2)} s to ${t.end>0?"+":""}${t.end.toFixed(2)} s`
      : (Math.abs(t.median) >= 0.25 ? "constant offset, no creep" : "steady, no real slip");
    h += `<tr><td>${esc(D.songs[sid].title)}</td>
      <td class="num">${t.median>0?"+":""}${t.median.toFixed(2)} s</td>
      <td class="num">${t.slope_s_per_min>0?"+":""}${t.slope_s_per_min.toFixed(2)} s/min</td>
      <td>${shape}</td></tr>`;
  }
  const nb = Object.values(D.songs).reduce((a,s)=>a+s.timing.n,0);
  h += `</table></div><p class="small" style="margin-top:.6rem">Positive means the recording changes chord <em>after</em> the reference says it does. Measured at ${nb} chord changes across the seven songs, and corrected for the 0.11 s lag the outside model is known to carry.</p></div>`;
  h += q.map(x => card(x, 0)).join("");
  document.getElementById("drift").innerHTML = h;
}
function renderColour(){
  let h = `<p class="small">These are <b>not</b> errors to review. The root and the
    major/minor quality are both independently confirmed; only the seventh or the
    added colour tone is in question &mdash; the kind of thing that is genuinely hard
    to catch by ear and that costs almost nothing when it is wrong. Listed per
    chord, not per passage, because a colour tone belongs to one chord.</p>`;
  h += `<div class="card"><div class="scrollx"><table>
    <tr><th>Song</th><th>At</th><th>Written</th><th>Better fit</th><th class="num">Gain</th><th>Outside model</th></tr>`;
  for (const x of D.colour){
    h += `<tr><td>${esc(x.song_title)}</td><td class="num">${mmss(x.t0)}</td>
      <td>${esc(x.gt)}</td><td>${esc(x.suggest||"—")}</td>
      <td class="num">+${x.gap.toFixed(2)}</td><td>${esc(x.musx||"—")}</td></tr>`;
  }
  h += `</table></div></div>`;
  document.getElementById("colour").innerHTML = h;
}
function renderOK(){
  let h = `<p class="small">Auto-confirmed: the outside model backs the written root
    for at least 70% of the passage, it backs the major/minor quality on at least
    70% of those frames, the bar lines land within 0.30 s, and the same bars
    elsewhere in the song do not fit noticeably better. Nothing here is asked of you.</p>`;
  h += `<div class="card"><div class="scrollx"><table><tr><th>Song</th>
    <th class="num">Confirmed</th><th class="num">Passages</th><th class="num">Of its time</th></tr>`;
  for (const sid of Object.keys(D.songs)){
    const S = D.songs[sid], t = Object.values(S.agg).reduce((a,b)=>a+b,0);
    h += `<tr><td>${esc(S.title)}</td><td class="num">${mins(S.agg["auto-ok"])} min</td>
      <td class="num">${S.n_regions}</td><td class="num">${Math.round(100*S.agg["auto-ok"]/t)}%</td></tr>`;
  }
  h += `</table></div></div>`;
  document.getElementById("okpane").innerHTML = h;
}

/* ---------- tabs ---------- */
const PANES = [
  ["queue", () => "Your queue (" + D.queue.filter(x=>x.bucket==="needs-ear").length + ")"],
  ["drift", () => "Timing (" + D.queue.filter(x=>x.bucket==="drift").length + ")"],
  ["colour", () => "Colour tone (" + D.colour.length + ")"],
  ["ok", () => "Confirmed"],
  ["how", () => "How this was decided"],
];
function renderTabs(){
  const t = document.getElementById("tabs");
  t.innerHTML = PANES.map(([id,lab],i) =>
    `<button class="tab" role="tab" data-p="${id}" aria-selected="${i===0}">${lab()}</button>`).join("");
  t.addEventListener("click", e => {
    const b = e.target.closest(".tab"); if (!b) return;
    for (const x of t.querySelectorAll(".tab")) x.setAttribute("aria-selected", x===b);
    for (const [id] of PANES) document.getElementById("pane-"+id).hidden = (id !== b.dataset.p);
    window.scrollTo({top:0, behavior:"instant"});
  });
}

/* ---------- verdicts ---------- */
function paint(){
  for (const el of document.querySelectorAll(".verdict")){
    const v = V[el.dataset.id];
    for (const b of el.querySelectorAll(".vb"))
      b.setAttribute("aria-pressed", String(!!v && v === b.dataset.v));
  }
  const n = Object.keys(V).length, tot = D.queue.length;
  document.getElementById("progress").textContent = `${n} of ${tot} judged`;
}
document.addEventListener("click", e => {
  const b = e.target.closest(".vb"); if (!b) return;
  const id = b.closest(".verdict").dataset.id;
  V[id] = (V[id] === b.dataset.v) ? undefined : b.dataset.v;
  if (!V[id]) delete V[id];
  try { localStorage.setItem(KEY, JSON.stringify(V)); } catch(err) {}
  paint();
});
document.getElementById("copybtn").addEventListener("click", async () => {
  const out = {tool:"gt-triage", date:"2026-07-27", verdicts:
    Object.keys(V).map(id => {
      const q = D.queue.find(x => x.id === id) || {};
      return {id, song:q.song, from:q.t0, to:q.t1, bucket:q.bucket, verdict:V[id]};
    })};
  const txt = JSON.stringify(out, null, 1);
  const btn = document.getElementById("copybtn");
  try { await navigator.clipboard.writeText(txt); btn.textContent = "Copied ✓"; }
  catch(err){
    const ta = document.createElement("textarea");
    ta.value = txt; document.body.appendChild(ta); ta.select();
    try { document.execCommand("copy"); btn.textContent = "Copied ✓"; }
    catch(e2){ btn.textContent = "Press &#8984;C"; }
    ta.remove();
  }
  setTimeout(() => { btn.textContent = "Copy my verdicts"; }, 1800);
});

/* ---------- how ---------- */
function renderHow(){
  document.getElementById("how").innerHTML = `
  <div class="card">
  <h3>The rule the whole page obeys</h3>
  <p>Nothing here is judged by our own chord output. A model cannot be the judge of
  the ground truth it is scored against, so every number on this page comes from
  one of four things that never see it:</p>
  <ul class="why">
    <li><b>The raw spectrum.</b> A constant-Q chroma of the recording, compared
      against the note-set of the written chord and against every other chord in a
      180-chord vocabulary. This says which chord the sound actually resembles.</li>
    <li><b>An outside chord-recognition model.</b> A published third-party network
      (music-x-lab's ISMIR-2019 large-vocabulary net, which ships in this repo as a
      vendored clone), read one frame at a time for its root, its major/minor
      quality and its bass note. It has never been trained on our targets and
      knows nothing about our chart.</li>
    <li><b>The song repeating itself.</b> Whenever the same written chord sequence
      occurs somewhere else in the same song, the two occurrences are compared. If
      one place fits the recording much worse than its own twins, that place is
      suspect, not the chart.</li>
    <li><b>Level.</b> How loud the passage is against the song's own median, so a
      near-silent intro is reported as <em>unmeasured</em> rather than guessed.</li>
  </ul>

  <h3>What a passage is</h3>
  <p>Consecutive written chords grouped into roughly eight-second runs, always
  cut on chord boundaries, so each one is a phrase you can actually listen to.
  194 of them across the seven songs, 27.6 minutes in total.</p>

  <h3>How the four verdicts are decided</h3>
  <p>Three questions in order, each answered by the instrument best suited to it.</p>
  <ol class="why">
    <li><b>Is the root right?</b> Confirmed if the outside model backs it for at
      least 70% of the passage &mdash; or for at least 50% with the raw spectrum
      showing no better chord on another note. A reading the outside model spells
      on a different note is <em>not</em> counted against the chart when its notes
      fit inside the written chord plus one colour tone: an Em7 voiced without its
      root simply is a G major triad, and calling that a root error would be wrong.</li>
    <li><b>Is major-versus-minor right?</b> Of the frames that back the root, at
      least 70% must also back the written quality.</li>
    <li><b>Do the bar lines land?</b> Every chord change is re-found by sliding a
      matched filter over the outside model's frame probabilities, and the result
      is corrected for the 0.11 s lag that model is known to carry on this
      benchmark. A passage is <em>Timing off</em> when its changes sit at least
      0.30 s away &mdash; half a beat at these tempi.</li>
  </ol>
  <p>Pass all three and the passage is <b>Confirmed</b>. Pass all three but with the
  spectrum clearly preferring a different seventh on the same root and quality, and
  it is <b>Colour tone only</b>. Fail the timing question alone and it is
  <b>Timing off</b>. Fail a naming question, contradict its own repeats, sit in
  silence, or have the two timing instruments disagree by over a second &mdash; then
  it is <b>yours</b>.</p>

  <h3>What this page will not claim</h3>
  <p>It never says the reference is wrong. The chart from iReal Pro outranks a
  model, so a model disagreeing is a reason to <em>listen</em>, not a verdict. Seven
  songs is seven songs: the thresholds above were set against the spread of these
  194 passages and would need re-checking on anything else. And where a signal
  could not be measured &mdash; a passage too quiet, or one the outside model refuses to
  call &mdash; it is shown as missing rather than filled in.</p>

  <h3>Two things worth knowing before you start</h3>
  <p><b>Georgia's E minor is fine.</b> The bars written <span class="ch">E:min</span>
  then <span class="ch">E:min/D</span> come out as E minor throughout on every
  instrument here; what actually moves is the seventh, Em &rarr; Em(maj7) &rarr;
  Em7/D, the descending inner line. The outside model calls that last chord G major
  &mdash; which is exactly Em7 without its root. It is filed under colour tone, not
  as an error.</p>
  <p><b>Stand by Me is the one place the instruments fight.</b> From 0:52 onward the
  outside model puts every chord change within 0.15 s of the reference, and its
  spread over the whole song (0.05 s) is as tight as the metronomic backing track
  &mdash; by that instrument this is the best-aligned song of the seven, not the
  worst. But the raw spectrum insists, over and over, that the A&rarr;F#m and
  F#m&rarr;D changes come about 1.2 s earlier. That instrument is the weaker
  witness here &mdash; A, F#m, D and E share two notes out of three, so its peak is
  barely identifiable (0.03&ndash;0.11 against 0.2&ndash;0.35 on the other songs)
  &mdash; yet a 1.2 s disagreement that repeats every chorus is too systematic to
  wave away. On top of that, the reference does not start until 0:16 and its first
  two passages (0:16&ndash;0:36) have no usable independent evidence at all: the
  outside model returns &ldquo;no chord&rdquo; for three quarters of that stretch,
  because the intro is bass and percussion with no harmony instrument. Those
  passages sit at the top of your queue, and only your ear can settle them.</p>
  </div>`;
}

renderSummary(); renderTabs(); renderQueue(); renderDrift(); renderColour();
renderOK(); renderHow(); paint();
</script>
"""

if __name__ == "__main__":
    build()
