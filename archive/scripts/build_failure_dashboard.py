#!/usr/bin/env python3
"""Build the failure-mode dashboard (docs/error_analysis_dashboard_v2.html) and
report (docs/failure_mode_analysis.md) from the analyze_failure_modes.py JSON.

Run:  python scripts/analyze_failure_modes.py --out /tmp/fm.json
      python scripts/build_failure_dashboard.py /tmp/fm.json
"""
from __future__ import annotations
import json, sys, re
from pathlib import Path
from collections import Counter

REPO = Path(__file__).resolve().parent.parent
PLOTS = REPO / "docs" / "plots"
sys.path.insert(0, str(REPO / "scripts"))
from analyze_failure_modes import (  # noqa: E402
    load_inferred, quality_family, inferred_quality, fifth_competition, ROOT_NAMES,
)

QUAL_DISPLAY = {"maj": "maj", "min": "min", "dom": "dom", "hdim": "ø", "dim": "dim",
                "sus": "sus", "other": "?"}


def interval_groups(root_confus):
    """Collapse the 12x12 competitor-mass matrix into interval families by the
    signed offset (competitor - chosen) mod 12."""
    off = Counter()
    for i in range(12):
        for j in range(12):
            off[(j - i) % 12] += root_confus[i][j]
    groups = {
        "Perfect 4th / 5th (+5/+7)": off[5] + off[7],
        "Semitone (+1/+11)": off[1] + off[11],
        "Whole tone (+2/+10)": off[2] + off[10],
        "Thirds (+3/+4/+8/+9)": off[3] + off[4] + off[8] + off[9],
        "Tritone (+6)": off[6],
    }
    return groups, dict(off)


def collect_fifth_examples(limit=14):
    """Concrete fifth-confusion cases: chords whose model posterior puts a
    perfect-fifth competitor near the chosen root. Grounded, no GT needed."""
    rows = []
    for f in sorted(PLOTS.glob("inferred_*.html")):
        slug = f.stem[len("inferred_"):]
        inf = load_inferred(slug)
        if not inf:
            continue
        for c in inf.get("chords", []):
            if not fifth_competition(c):
                continue
            root = c.get("root", -1)
            fam = quality_family(inferred_quality(c))
            sug = c.get("sug") or []
            chosen_c = next((s.get("c", 0.0) for s in sug if s.get("root") == root), None)
            comp = None
            for s in sug:
                sr = s.get("root", -1)
                if sr is None or sr < 0 or sr == root:
                    continue
                if (sr - root) % 12 in (5, 7):
                    comp = s
                    break
            if comp is None or chosen_c is None:
                continue
            # genuine near-tie: the chosen root must itself carry real posterior
            # support (>=0.15) — otherwise it's a decode-overrode-acoustics case,
            # not a fifth ambiguity — and the two must be close (|margin|<=0.35).
            if chosen_c < 0.15 or abs(chosen_c - comp.get("c", 0.0)) > 0.35:
                continue
            rows.append({
                "song": slug, "bar": c.get("bar"), "t0": round(c.get("t0", 0.0), 1),
                "chosen": f"{ROOT_NAMES[root]} {QUAL_DISPLAY[fam]}",
                "chosen_c": round(chosen_c, 2),
                "competitor": f"{ROOT_NAMES[comp['root']]} {QUAL_DISPLAY[quality_family(comp.get('q',''))]}",
                "competitor_c": round(comp.get("c", 0.0), 2),
                "interval": "P5" if (comp["root"] - root) % 12 == 7 else "P4",
                "margin": round(chosen_c - comp.get("c", 0.0), 2),
            })
    # closest margins first = most genuinely ambiguous; keep a diverse-ish set
    rows.sort(key=lambda r: abs(r["margin"]))
    seen, out = set(), []
    for r in rows:
        if r["song"] in seen and len([x for x in out if x["song"] == r["song"]]) >= 3:
            continue
        seen.add(r["song"]); out.append(r)
        if len(out) >= limit:
            break
    return out


def main():
    fm_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not fm_path:
        print("usage: build_failure_dashboard.py <fm.json>", file=sys.stderr)
        sys.exit(1)
    d = json.loads(Path(fm_path).read_text())
    ss = d["self_signal"]
    groups, off = interval_groups(ss["root_confusability"])
    examples = collect_fifth_examples()
    total_mass = sum(groups.values()) or 1.0
    group_pct = {k: round(100 * v / total_mass, 1) for k, v in groups.items()}

    payload = {
        "roots": d["roots"], "families": d["families"],
        "familyDisplay": [QUAL_DISPLAY[f] for f in d["families"]],
        "rootConfus": ss["root_confusability"],
        "qualWaver": ss["quality_waver"],
        "famDist": ss["family_distribution"],
        "confBins": ss["conf_bins"], "confEdges": ss["conf_bin_edges"],
        "totalChords": ss["total_chords"], "nSongs": ss["n_songs"],
        "fifthPct": ss["fifth_competition_pct"], "fifthN": ss["total_fifth_competition"],
        "fifthDir": ss["fifth_direction"],
        "lowConfPct": ss["low_conf_pct"], "lowConfN": ss["low_conf_total"],
        "groups": group_pct, "songs": ss["songs"],
        "gt": d["gt_integrity"], "examples": examples,
    }

    html = build_html(payload)
    out_html = REPO / "docs" / "error_analysis_dashboard_v2.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"wrote {out_html}")

    md = build_report(payload)
    out_md = REPO / "docs" / "failure_mode_analysis.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"wrote {out_md}")


# ---------------------------------------------------------------------------
def build_report(p) -> str:
    g = p["groups"]
    songs = p["songs"]
    lowconf_songs = sorted([s for s in songs if s["low_conf_pct"] > 80], key=lambda s: -s["low_conf_pct"])
    highconf_songs = [s for s in songs if s["low_conf_pct"] <= 80]
    fifth_songs = sorted(songs, key=lambda s: -s["fifth_pct"])[:5]
    fam = p["famDist"]
    famtot = sum(fam.values()) or 1
    ex = p["examples"]

    lines = []
    lines.append("# Harmonia — Failure-Mode Analysis\n")
    lines.append("_Generated by `scripts/analyze_failure_modes.py` + "
                 "`scripts/build_failure_dashboard.py`. "
                 "Dashboard: `docs/error_analysis_dashboard_v2.html`._\n")
    lines.append("## Scope & method\n")
    lines.append(f"- **{p['nSongs']} inferred charts, {p['totalChords']} chords** "
                 "(`docs/plots/inferred_*.html`, model posterior read from each chart's "
                 "`const P` payload).\n")
    lines.append("- **Two evidence tiers, trust order per CLAUDE.md (iReal GT > model self-signal):**\n"
                 "  1. _GT-anchored_ (irealb reference charts) — **BLOCKED**, see Finding 0.\n"
                 "  2. _Model self-signal_ — every inferred chord carries `sug` (its own ranked "
                 "root/quality alternatives) and `lv.exact.c` (chosen-chord confidence). This is "
                 "fully grounded, needs no labels, and is the basis for every number below.\n")

    lines.append("\n## Finding 0 (data integrity) — no valid GT eval from current artifacts\n")
    lines.append("The `irealb_<slug>.html` reference charts and the inferred charts are **on "
                 "different timelines**, so a GT-anchored root/quality accuracy cannot be computed "
                 "from these files. Evidence (two mutually-contradictory alignment strategies):\n")
    lines.append("| pilot | GT span | inferred span | ratio | time-overlap root-acc | free-seq-align root-acc |")
    lines.append("|---|--:|--:|--:|--:|--:|")
    for c in p["gt"]["pilots"]:
        lines.append(f"| {c['slug'][:34]} | {c['gt_dur']}s | {c['inf_dur']}s | "
                     f"{c['dur_ratio']}× | {c['time_overlap_acc']} | {c['seq_align_acc']} |")
    lines.append("\nTime-overlap sampling puts them at ~chance (~0.11); free sequence alignment "
                 "cherry-picks a matching segment from the ~10× longer inferred list and reports "
                 "1.0. Both are artifacts of the timeline mismatch, not real accuracy. **Fix: export "
                 "GT and inference on a shared audio timeline** (same DTW pass / same clock) before "
                 "attempting any accuracy panel. Until then, root/quality _accuracy_ is unmeasured; "
                 "what follows measures where the model is _internally uncertain_, which is where "
                 "accuracy is most at risk.\n")

    lines.append("## Executive summary — top 3 failure modes\n")
    lines.append(f"1. **Perfect-4th/5th root ambiguity — {g['Perfect 4th / 5th (+5/+7)']}% of all "
                 "root competitor mass** (vs a ~17% uniform baseline for two of twelve intervals). "
                 "When the model wavers on a root, a fifth away is by far the most likely rival — "
                 f"the known 5th-apart acoustic confusion, now quantified. Directionality is "
                 f"balanced ({p['fifthDir']['fifth_up_+7']} up-a-fifth vs "
                 f"{p['fifthDir']['fourth_up_+5']} up-a-fourth). "
                 f"{p['fifthN']} chords ({p['fifthPct']}%) have a near-tie fifth competitor.\n")
    lines.append(f"2. **A low-confidence song cluster — {p['lowConfPct']}% of all chords sit below "
                 f"0.4 confidence** ({p['lowConfN']} chords), but it is not spread evenly: "
                 f"{len(lowconf_songs)} songs are ~globally uncertain (mean conf ≈ 0.17–0.22, "
                 "≥80% of chords low-conf) while the rest sit at mean conf ≈ 0.75–0.9. This is a "
                 "**two-domain split**, not gradual degradation — consistent with the real-audio "
                 "confidence-calibration work (recent Mission 4 / `#19/#26`).\n")
    lines.append("3. **Quality collapses toward maj/dom.** Family mix is "
                 f"{round(100*fam['maj']/famtot)}% maj / {round(100*fam['dom']/famtot)}% dom / "
                 f"{round(100*fam['min']/famtot)}% min / {round(100*fam['hdim']/famtot)}% ø / "
                 f"{round(100*fam['dim']/famtot)}% dim. Half-diminished and diminished — the "
                 "chords that most disambiguate a minor ii–V — are ~6% combined, so a ø/dim that "
                 "reads as its relative maj/dom is a systematic quality error the GT eval would "
                 "otherwise catch.\n")

    lines.append("## Root confusion — interval breakdown (model competitor mass)\n")
    lines.append("| interval family | share of competitor mass |")
    lines.append("|---|--:|")
    for k, v in sorted(g.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v}% |")
    lines.append("\n_Hypothesis:_ perfect-fifth dominance is a **template-geometry / octave-"
                 "generalisation** effect — a maj/min triad shares two of three chroma bins with "
                 "the triad a fifth away, so a chroma-template emission scores them near-equally "
                 "(cf. `known_issues.md #5`, cosine-vs-dot emission). _Suggested fix:_ a bass-aware "
                 "or transition-context tiebreak (prefer the root that a functional ii–V–I or a "
                 "detected bass note supports) applied specifically when the top-2 roots are a "
                 "fifth apart.\n")

    lines.append("## Confidence distribution\n")
    lines.append("| confidence bin | chords |")
    lines.append("|---|--:|")
    for edge, n in zip(p["confEdges"], p["confBins"]):
        lines.append(f"| {edge} | {n} |")
    lines.append("")

    lines.append("## Per-song self-signal\n")
    lines.append("**Globally-uncertain cluster** (candidate calibration/domain failures — inspect "
                 "these first):\n")
    lines.append("| song | chords | mean conf | low-conf % | fifth-comp % |")
    lines.append("|---|--:|--:|--:|--:|")
    for s in lowconf_songs:
        lines.append(f"| {s['slug'][:38]} | {s['n_chords']} | {s['mean_conf']} | "
                     f"{s['low_conf_pct']} | {s['fifth_pct']} |")
    lines.append("\n**Confident cluster** (top-5 by fifth-competition — where the 5th ambiguity bites "
                 "hardest):\n")
    lines.append("| song | chords | mean conf | fifth-comp % |")
    lines.append("|---|--:|--:|--:|")
    for s in fifth_songs:
        lines.append(f"| {s['slug'][:38]} | {s['n_chords']} | {s['mean_conf']} | {s['fifth_pct']} |")

    lines.append("\n## Example fifth-confusion cases (model vs. its own top rival)\n")
    lines.append("Concrete near-ties where a perfect-fifth competitor rivals the chosen root "
                 "(smallest confidence margin first). These are the cases a bass/transition tiebreak "
                 "would target.\n")
    lines.append("| song | bar | t0 | chosen (conf) | fifth rival (conf) | int | margin |")
    lines.append("|---|--:|--:|---|---|:--:|--:|")
    for e in ex:
        lines.append(f"| {e['song'][:26]} | {e['bar']} | {e['t0']}s | {e['chosen']} ({e['chosen_c']}) | "
                     f"{e['competitor']} ({e['competitor_c']}) | {e['interval']} | {e['margin']} |")

    lines.append("\n## Suggested fixes (ranked)\n")
    lines.append("1. **Fifth-apart tiebreak** — when the top-2 candidate roots are ±5/±7 semitones "
                 "and within a small confidence margin, break the tie with (a) detected bass note "
                 "and (b) key/functional context (ii–V–I likelihood). Screen the premise first "
                 "(`known_issues.md #5` cosine-vs-dot sweep) before implementing — highest expected "
                 "payoff, directly targets the dominant failure mode.\n")
    lines.append("   _Secondary signal:_ in some chords the decoded `root` differs from the argmax "
                 "of its own `sug` list by a fifth (chosen-root posterior near zero while a fifth-away "
                 "rival dominates) — the HMM prior/transition overriding weak acoustic evidence. Worth "
                 "a separate look; excluded from the near-tie examples above.\n")
    lines.append("2. **Diagnose the low-confidence cluster as a domain, not per-song noise** — the "
                 "clean ~0.18 vs ~0.8 split says one calibration map is misfiring on a subset "
                 "(likely real-audio vs backing-track/MMA). Verify which domain each low-conf song "
                 "is in and whether the Mission-4 two-domain calibrator is being applied to them.\n")
    lines.append("3. **Half-dim / dim recovery** — add ø/dim-specific evidence (the b5/dim5 chroma "
                 "signature) so minor ii–V ø chords stop collapsing to maj/dom.\n")
    lines.append("4. **Unblock GT eval** — emit inference and iReal GT on a shared timeline so the "
                 "real root/quality confusion matrices can replace this posterior-based proxy.\n")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
def build_html(p) -> str:
    data_json = json.dumps(p)
    return TEMPLATE.replace("/*__DATA__*/null", data_json)


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Harmonia — Failure-Mode Dashboard v2</title>
<style>
:root{
  --surface:#171c24; --plane:#0e1116; --panel:#1e2530; --line:#2a3340;
  --ink:#e8edf4; --sec:#aab6c6; --muted:#8b97a8; --grid:#242c38;
  --blue100:#cde2fb;--blue250:#86b6ef;--blue400:#3987e5;--blue550:#1c5cab;--blue700:#0d366b;
  --aqua:#199e70; --yellow:#c98500; --red:#e66767; --violet:#9085e9; --orange:#d95926;
  --good:#0ca30c; --warn:#fab219; --crit:#d03b3b;
}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.5;
  padding:0 0 60px;}
.wrap{max-width:1120px;margin:0 auto;padding:0 20px;}
header{padding:34px 0 10px;border-bottom:1px solid var(--line);margin-bottom:22px;}
h1{font-size:24px;margin:0 0 4px;letter-spacing:.2px;}
.subtitle{color:var(--sec);font-size:13.5px;}
h2{font-size:15px;margin:0 0 3px;letter-spacing:.3px;}
.card{background:var(--surface);border:1px solid var(--line);border-radius:14px;
  padding:18px 20px;margin:16px 0;overflow:hidden;}
.card .desc{color:var(--muted);font-size:12.5px;margin:2px 0 14px;max-width:80ch;}
.hero{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:16px 0;}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:16px 18px;}
.tile .n{font-size:30px;font-weight:700;letter-spacing:-.5px;}
.tile .l{color:var(--muted);font-size:12px;margin-top:2px;}
.tile.warn-tile{border-color:#5a3a12;}
.tile.warn-tile .n{color:var(--warn);}
.tile.crit-tile .n{color:var(--red);}
.gtcard{background:#241a12;border:1px solid #5a3a12;}
.gtcard h2{color:var(--warn);}
table{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums;}
th,td{text-align:left;padding:5px 9px;border-bottom:1px solid var(--grid);}
th{color:var(--muted);font-weight:600;}
td.num,th.num{text-align:right;}
.scrollx{overflow-x:auto;}
svg{display:block;max-width:100%;}
.axlab{fill:var(--muted);font-size:10px;font-variant-numeric:tabular-nums;}
.axttl{fill:var(--sec);font-size:11px;font-weight:600;}
.tip{position:fixed;pointer-events:none;background:#0b0e13;border:1px solid var(--line);
  border-radius:8px;padding:6px 9px;font-size:11.5px;color:var(--ink);opacity:0;
  transition:opacity .08s;z-index:50;box-shadow:0 6px 20px #0009;white-space:nowrap;}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:11.5px;color:var(--sec);margin-top:10px;}
.legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;vertical-align:-1px;}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
@media(max-width:820px){.grid2{grid-template-columns:1fr;}}
.pill{display:inline-block;padding:1px 7px;border-radius:20px;font-size:11px;font-weight:600;}
.pill.fifth{background:#3a1414;color:#ff9a9a;border:1px solid #6a2a2a;}
.foot{color:var(--muted);font-size:11.5px;margin-top:24px;text-align:center;}
code{background:#0b0e13;border:1px solid var(--line);border-radius:5px;padding:1px 5px;font-size:11.5px;}
</style></head><body>
<div class="wrap">
<header>
  <h1>Harmonia — Failure-Mode Dashboard <span style="color:var(--muted);font-weight:400">v2</span></h1>
  <div class="subtitle" id="sub"></div>
</header>

<div class="hero" id="hero"></div>

<div class="card gtcard">
  <h2>⚠ GT-anchored accuracy is BLOCKED — timeline mismatch</h2>
  <div class="desc">The iReal reference charts and the inferred charts are on different
    timelines, so a true root/quality accuracy cannot be computed from these artifacts.
    Two alignment strategies contradict each other, proving the mismatch. Numbers below
    instead measure the model's <b>own posterior uncertainty</b> — where accuracy is most at risk.</div>
  <div class="scrollx"><table id="gttable"></table></div>
</div>

<div class="card">
  <h2>Panel 1 · Root confusability matrix</h2>
  <div class="desc">Where the model wavers on the <b>root</b>. Cell [row = chosen root,
    col = competing root] = total confidence mass the model's own suggestion list assigns
    to that rival across all 3,384 chords. Darker = more competition. Cells a
    <span class="pill fifth">perfect 4th/5th</span> from the chosen root are ringed — they
    dominate every row.</div>
  <div class="scrollx"><div id="rootmx"></div></div>
</div>

<div class="grid2">
  <div class="card">
    <h2>Panel 2 · Quality-waver matrix</h2>
    <div class="desc">Chosen quality family (row) vs the family of competing suggestions
      (col). Off-diagonal mass shows which qualities the model conflates.</div>
    <div class="scrollx"><div id="qualmx"></div></div>
  </div>
  <div class="card">
    <h2>Panel 3 · Failure distribution</h2>
    <div class="desc">Root competitor mass by interval family. Perfect 4th/5th is one bar
      but two of twelve intervals — a flat model would give it ~17%.</div>
    <div id="groupbars"></div>
    <h2 style="margin-top:18px">Quality mix</h2>
    <div id="qualbars"></div>
  </div>
</div>

<div class="card">
  <h2>Panel 4 · Per-song self-signal</h2>
  <div class="desc">Each song's mean confidence (bar) and fifth-competition rate (dot).
    Note the two clusters: a low-confidence group (mean ≈ 0.18, left) and a confident
    group (mean ≈ 0.8) where the fifth ambiguity concentrates.</div>
  <div class="scrollx"><div id="songchart"></div></div>
  <div class="legend">
    <span><i style="background:var(--blue400)"></i>mean confidence</span>
    <span><i style="background:var(--red);border-radius:50%"></i>fifth-competition %</span>
    <span><i style="background:var(--warn)"></i>≥80% chords low-conf (flag)</span>
  </div>
</div>

<div class="card">
  <h2>Example fifth-confusion cases</h2>
  <div class="desc">Concrete near-ties: chords where a perfect-fifth rival rivals the
    chosen root in the model's own posterior (smallest margin first). The target set for a
    bass/transition tiebreak.</div>
  <div class="scrollx"><table id="extable"></table></div>
</div>

<div class="foot">Model self-signal from <code>docs/plots/inferred_*.html</code> ·
  generated by <code>scripts/analyze_failure_modes.py</code> +
  <code>scripts/build_failure_dashboard.py</code></div>
</div>

<div class="tip" id="tip"></div>
<script>
const D = /*__DATA__*/null;
const tip=document.getElementById('tip');
function showTip(html,e){tip.innerHTML=html;tip.style.opacity=1;moveTip(e);}
function moveTip(e){tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY+14)+'px';}
function hideTip(){tip.style.opacity=0;}
const SVGNS='http://www.w3.org/2000/svg';
function el(tag,attrs){const n=document.createElementNS(SVGNS,tag);
  for(const k in attrs)n.setAttribute(k,attrs[k]);return n;}

// blue sequential ramp, t in 0..1
function seq(t){
  const stops=[[0.00,'#0e1116'],[0.12,'#0d366b'],[0.35,'#1c5cab'],[0.6,'#3987e5'],
    [0.82,'#86b6ef'],[1,'#cde2fb']];
  t=Math.max(0,Math.min(1,t));
  for(let i=1;i<stops.length;i++){if(t<=stops[i][0]){
    const a=stops[i-1],b=stops[i];const f=(t-a[0])/(b[0]-a[0]||1);
    return lerp(a[1],b[1],f);}}
  return stops[stops.length-1][1];
}
function lerp(c1,c2,f){const a=hx(c1),b=hx(c2);
  return `rgb(${Math.round(a[0]+(b[0]-a[0])*f)},${Math.round(a[1]+(b[1]-a[1])*f)},${Math.round(a[2]+(b[2]-a[2])*f)})`;}
function hx(c){return [parseInt(c.slice(1,3),16),parseInt(c.slice(3,5),16),parseInt(c.slice(5,7),16)];}

// ---------- header + hero ----------
document.getElementById('sub').textContent =
  `${D.nSongs} inferred charts · ${D.totalChords.toLocaleString()} chords · model-posterior self-signal (no ground truth required)`;
const fifthShare = D.groups['Perfect 4th / 5th (+5/+7)'];
document.getElementById('hero').innerHTML = [
  ['crit-tile', fifthShare+'%', 'root competitor mass is a Perfect 4th/5th'],
  ['', D.fifthPct+'%', `chords (${D.fifthN}) with a near-tie fifth rival`],
  ['warn-tile', D.lowConfPct+'%', `chords (${D.lowConfN}) below 0.4 confidence`],
  ['', D.totalChords.toLocaleString(), `chords across ${D.nSongs} charts`],
].map(([c,n,l])=>`<div class="tile ${c}"><div class="n">${n}</div><div class="l">${l}</div></div>`).join('');

// ---------- GT integrity table ----------
{
  let h='<tr><th>pilot</th><th class="num">GT span</th><th class="num">inferred span</th>'
      +'<th class="num">ratio</th><th class="num">time-overlap acc</th><th class="num">seq-align acc</th></tr>';
  D.gt.pilots.forEach(c=>{ h+=`<tr><td>${c.slug.slice(0,34)}</td><td class="num">${c.gt_dur}s</td>`
    +`<td class="num">${c.inf_dur}s</td><td class="num">${c.dur_ratio}×</td>`
    +`<td class="num">${c.time_overlap_acc}</td><td class="num">${c.seq_align_acc}</td></tr>`; });
  document.getElementById('gttable').innerHTML=h;
}

// ---------- heatmap ----------
function heatmap(mount,mx,rowLab,colLab,opt){
  opt=opt||{};
  const n=mx.length,m=mx[0].length;
  const cell=Math.max(20,Math.min(34,Math.floor(760/m)));
  const padL=44,padT=26,padR=8,padB=10;
  const W=padL+m*cell+padR,Hh=padT+n*cell+padB;
  let max=0; for(const r of mx)for(const v of r)if(v>max)max=v;
  const svg=el('svg',{width:W,height:Hh,viewBox:`0 0 ${W} ${Hh}`});
  // col labels
  colLab.forEach((c,j)=>{const t=el('text',{x:padL+j*cell+cell/2,y:padT-8,'text-anchor':'middle','class':'axlab'});t.textContent=c;svg.appendChild(t);});
  rowLab.forEach((r,i)=>{const t=el('text',{x:padL-8,y:padT+i*cell+cell/2+3,'text-anchor':'end','class':'axlab'});t.textContent=r;svg.appendChild(t);});
  for(let i=0;i<n;i++)for(let j=0;j<m;j++){
    const v=mx[i][j];const t=max?v/max:0;
    const g=el('rect',{x:padL+j*cell,y:padT+i*cell,width:cell-2,height:cell-2,rx:3,
      fill:i===j?'#12161d':seq(Math.pow(t,0.6))});
    if(opt.highlight&&opt.highlight(i,j)&&v>0){g.setAttribute('stroke','var(--red)');g.setAttribute('stroke-width','2');}
    g.addEventListener('pointerenter',e=>showTip(
      `<b>${rowLab[i]} → ${colLab[j]}</b><br>competitor mass ${v.toFixed(1)}`
      +(opt.highlight&&opt.highlight(i,j)?'<br><span style="color:var(--red)">perfect 4th/5th</span>':''),e));
    g.addEventListener('pointermove',moveTip);g.addEventListener('pointerleave',hideTip);
    svg.appendChild(g);
  }
  mount.appendChild(svg);
}
heatmap(document.getElementById('rootmx'),D.rootConfus,D.roots,D.roots,
  {highlight:(i,j)=>((j-i)%12===7||(j-i)%12===5)});
heatmap(document.getElementById('qualmx'),D.qualWaver,D.familyDisplay,D.familyDisplay,{});

// ---------- horizontal bars ----------
function hbars(mount,items,opt){
  opt=opt||{};
  const rowH=26,padL=opt.padL||160,padR=54,W=opt.W||480;
  const H=items.length*rowH+8;
  const max=opt.max||Math.max(...items.map(i=>i.value));
  const svg=el('svg',{width:'100%',height:H,viewBox:`0 0 ${W} ${H}`,preserveAspectRatio:'xMinYMin meet'});
  items.forEach((it,i)=>{
    const y=i*rowH+4;
    const t=el('text',{x:padL-8,y:y+rowH/2+3,'text-anchor':'end','class':'axlab'});t.textContent=it.label;svg.appendChild(t);
    const bw=(W-padL-padR)*(it.value/max);
    const r=el('rect',{x:padL,y:y+3,width:Math.max(2,bw),height:rowH-10,rx:4,fill:it.color||'var(--blue400)'});
    r.addEventListener('pointerenter',e=>showTip(`<b>${it.label}</b><br>${it.value}${opt.unit||''}`,e));
    r.addEventListener('pointermove',moveTip);r.addEventListener('pointerleave',hideTip);
    svg.appendChild(r);
    const vt=el('text',{x:padL+Math.max(2,bw)+6,y:y+rowH/2+3,'class':'axlab'});vt.textContent=it.value+(opt.unit||'');svg.appendChild(vt);
  });
  mount.appendChild(svg);
}
{
  const gs=Object.entries(D.groups).sort((a,b)=>b[1]-a[1]).map(([k,v])=>({
    label:k,value:v,unit:'%',color:k.indexOf('4th / 5th')>=0?'var(--red)':'var(--blue550)'}));
  hbars(document.getElementById('groupbars'),gs,{max:Math.max(...gs.map(g=>g.value)),unit:'%',padL:190});
}
{
  const fd=D.famDist;const tot=Object.values(fd).reduce((a,b)=>a+b,0);
  const order=D.families.filter(f=>fd[f]>0);
  const items=order.map((f,i)=>({label:D.familyDisplay[D.families.indexOf(f)]+' ('+f+')',
    value:Math.round(1000*fd[f]/tot)/10,unit:'%',color:'var(--blue400)'}));
  hbars(document.getElementById('qualbars'),items,{max:Math.max(...items.map(i=>i.value)),unit:'%',padL:120});
}

// ---------- per-song chart ----------
{
  const songs=D.songs.slice().sort((a,b)=>a.mean_conf-b.mean_conf);
  const rowH=22,padL=250,padR=60,W=900,H=songs.length*rowH+30;
  const svg=el('svg',{width:'100%',height:H,viewBox:`0 0 ${W} ${H}`,preserveAspectRatio:'xMinYMin meet'});
  // axis ticks 0..1
  [0,.25,.5,.75,1].forEach(t=>{const x=padL+(W-padL-padR)*t;
    svg.appendChild(el('line',{x1:x,y1:20,x2:x,y2:H-6,stroke:'var(--grid)','stroke-width':1}));
    const lb=el('text',{x:x,y:14,'text-anchor':'middle','class':'axlab'});lb.textContent=t;svg.appendChild(lb);});
  songs.forEach((s,i)=>{
    const y=22+i*rowH;
    const flag=s.low_conf_pct>=80;
    const nm=el('text',{x:padL-8,y:y+rowH/2+2,'text-anchor':'end','class':'axlab'});
    nm.textContent=(flag?'⚑ ':'')+s.slug.slice(0,32);
    if(flag)nm.setAttribute('fill','var(--warn)');
    svg.appendChild(nm);
    const bw=(W-padL-padR)*s.mean_conf;
    const r=el('rect',{x:padL,y:y+3,width:Math.max(2,bw),height:rowH-9,rx:3,
      fill:flag?'var(--warn)':'var(--blue400)'});
    r.addEventListener('pointerenter',e=>showTip(`<b>${s.slug}</b><br>mean conf ${s.mean_conf}<br>`
      +`low-conf ${s.low_conf_pct}% · fifth-comp ${s.fifth_pct}% · ${s.n_chords} chords`,e));
    r.addEventListener('pointermove',moveTip);r.addEventListener('pointerleave',hideTip);
    svg.appendChild(r);
    // fifth-competition dot on the same 0..1 axis (pct/100 * ... use pct/40 scale? keep pct/100)
    const fx=padL+(W-padL-padR)*(s.fifth_pct/100);
    const dot=el('circle',{cx:fx,cy:y+rowH/2,r:4.5,fill:'var(--red)',stroke:'#0e1116','stroke-width':1});
    dot.addEventListener('pointerenter',e=>showTip(`<b>${s.slug}</b><br>fifth-competition ${s.fifth_pct}%`,e));
    dot.addEventListener('pointermove',moveTip);dot.addEventListener('pointerleave',hideTip);
    svg.appendChild(dot);
  });
  document.getElementById('songchart').appendChild(svg);
}

// ---------- examples table ----------
{
  let h='<tr><th>song</th><th class="num">bar</th><th class="num">t0</th><th>chosen (conf)</th>'
      +'<th>fifth rival (conf)</th><th>int</th><th class="num">margin</th></tr>';
  D.examples.forEach(e=>{ h+=`<tr><td>${e.song.slice(0,24)}</td><td class="num">${e.bar}</td>`
    +`<td class="num">${e.t0}s</td><td>${e.chosen} <span style="color:var(--muted)">(${e.chosen_c})</span></td>`
    +`<td>${e.competitor} <span style="color:var(--muted)">(${e.competitor_c})</span></td>`
    +`<td><span class="pill fifth">${e.interval}</span></td><td class="num">${e.margin}</td></tr>`; });
  document.getElementById('extable').innerHTML=h;
}
</script>
</body></html>"""


if __name__ == "__main__":
    main()
