"""harmonia_min/minimal_view.py — la représentation MINIMALISTE (Louis,
2026-08-02, proposition 1 + twist).

Règle d'or : aucune info écrite deux fois. Chaque lettre écrit UN bloc :
  * la cellule du repli, complétée à ≥ 4 barres (le twist : longueur
    minimale d'un bloc = 4 barres, quand c'est possible) ;
  * SI les queues des passes divergent, le bloc couvre TOUT le passage
    représentatif jusqu'à la queue qui varie (This Love B = 8 barres :
    3 cellules + la cadence Cm F7 | Ab G) — « bien représenter toute la
    partie ».
La timeline chronologique fait foi pour le déroulement : chip = lettre ×
(longueur de la passe / longueur du bloc) — d'où « A×4 B A×3 … » : B sans
compteur quand son bloc EST la passe entière. Un chip se tape → l'audio
saute au passage.
"""
from __future__ import annotations

import json

NOTE = "C D♭ D E♭ E F G♭ G A♭ A B♭ B".split()
_SUP = {"": "", "-": "m", "7": "7", "^7": "Δ7", "-7": "m7", "h7": "ø7",
        "o": "°", "o7": "°7", "9": "9", "-9": "m9", "^9": "Δ9", "13": "13",
        "+": "+", "sus2": "sus2", "sus4": "sus4", "7sus4": "7sus4"}


def _chord(c) -> str:
    if c["nc"]:
        return "<span class='nc'>N.C.</span>"
    s = f"<b>{NOTE[c['root']]}</b><sup>{_SUP.get(c['q'], c['q'])}</sup>"
    if c["bass"] >= 0 and c["bass"] != c["root"]:
        s += f"<span class='bass'>/{NOTE[c['bass']]}</span>"
    return s


def _bar(bar) -> str:
    show = [c for c in bar if not (c.get("carry") and len(bar) > 1)]
    if bar and all(c.get("carry") for c in bar):
        show = bar[:1]
    inner = "".join(
        f"<span class='q q{c['beat']}{' carry' if c.get('carry') else ''}'>"
        f"{_chord(c)}</span>" for c in show)
    return f"<div class='bar'>{inner}</div>"


def _sig(bar):
    return tuple((c["root"], c["q"], c["nc"]) for c in bar)


def render_minimal(model: dict) -> str:
    secs = model["sections"]
    fold = model.get("fold") or {}
    grid = model["barGrid"]

    # all bars, indexable globally (representative passes carry the content)
    allbars = {}
    for s in secs:
        b0 = s["barRanges"][0][0]
        for k, bar in enumerate(s["bars"]):
            allbars[b0 + k] = bar

    # one BLOCK per letter
    blocks: dict[str, list] = {}
    for s in secs:
        L = s["label"]
        if L in blocks:
            continue
        b0, b1 = s["barRanges"][0]
        pass_len = b1 - b0 + 1
        P = (fold.get(L) or {}).get("period")
        cell = s["bars"][:P] if P else list(s["bars"])
        # do the passes of this letter diverge in their TAILS? (Louis's
        # twist targets the QUEUE — compare the last 2 bars of each pass,
        # aligned from the END; mid-pass noise like an N.C. gap must not
        # inflate the block to a full identical-rows pass)
        ranges = [tuple(r) for x in secs if x["label"] == L
                  for r in x["barRanges"]]
        tails_vary = False
        if len(ranges) > 1:
            ref = ranges[0]
            ref_tail = [_sig(allbars.get(ref[1] - k, [])) for k in (1, 0)]
            for c0, c1 in ranges[1:]:
                tail = [_sig(allbars.get(c1 - k, [])) for k in (1, 0)]
                if tail != ref_tail:
                    tails_vary = True
                    break
        pass_lens = {c1 - c0 + 1 for c0, c1 in ranges}
        is_pass = False
        if P and tails_vary and len(pass_lens) == 1:
            # fixed-length passage whose tail varies: cover the WHOLE pass
            # up to the varying tail (This Love's B = 3 cells + the Ab G
            # cadence, 8 bars written once)
            block = [allbars.get(b, []) for b in range(b0, b1 + 1)]
            is_pass = True
        elif P and tails_vary:
            # unequal passes, tail varies (This Love's A ends Dø7 vs D°):
            # cell + the DIVERGENT tail cell — 8 bars, not the 16-bar pass
            # (Louis: « 8 suffisent largement »)
            div = next(((c0, c1) for c0, c1 in ranges[1:]
                        if [_sig(allbars.get(c1 - k, [])) for k in (1, 0)]
                        != [_sig(allbars.get(ranges[0][1] - k, []))
                            for k in (1, 0)]), None)
            tailbars = [allbars.get(b, [])
                        for b in range(div[1] - len(cell) + 1, div[1] + 1)]                 if div else []
            block = list(cell) + ([tb for tb in tailbars] if all(tailbars)
                                  else [])
        elif P:
            block = list(cell)
            while len(block) < min(4, pass_len):   # twist: blocs >= 4 barres
                block += cell
            block = block[:max(len(cell), min(4, pass_len))]
        else:
            block = list(cell)
        blocks[L] = {"bars": block, "cell": max(1, len(cell)),
                     "is_pass": is_pass}

    # chronological timeline: chips count in CELL units (A×4 = the 4-bar
    # cell played 4 times); a letter whose block IS the pass counts in
    # pass units (B plain, final B ×3)
    chips = []
    for s in secs:
        L = s["label"]
        unit = len(blocks[L]["bars"]) if blocks[L]["is_pass"]             else blocks[L]["cell"]
        for b0, b1 in s["barRanges"]:
            n = max(1, round((b1 - b0 + 1) / max(1, unit)))
            chips.append((b0, L, n, grid[b0], grid[min(len(grid) - 1, b1 + 1)]))
    chips.sort()
    chips_html = "".join(
        f"<button class='chip' onclick=\"play({t0:.2f},{t1:.2f})\">"
        f"{L}{'×' + str(n) if n > 1 else ''}</button>"
        for _, L, n, t0, t1 in chips)

    body = ""
    for L, blk in blocks.items():
        block = blk["bars"]
        rows = ""
        for i in range(0, len(block), 4):
            rows += ("<div class='row'>" +
                     "".join(_bar(b) for b in block[i:i + 4]) + "</div>")
        body += (f"<div class='sec'><span class='letter'>{L}</span>"
                 f"<div class='barsbox'>{rows}</div></div>")

    return f"""<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{model['title']} — minimal</title><style>
body{{font-family:-apple-system,Georgia,serif;background:#e7e0d0;color:#1c1c1c;margin:0;padding:18px;line-height:1.5}}
main{{max-width:820px;margin:0 auto}}
h1{{font:italic 700 24px Georgia;margin:6px 0 2px}} .sub{{color:#8a8371;font-size:13px;margin-bottom:12px}}
.chip{{background:#fffdf6;border:1.5px solid #8a2b2b;color:#8a2b2b;font:700 14px -apple-system;border-radius:16px;padding:4px 12px;margin:2px 3px;cursor:pointer}}
.sec{{display:flex;align-items:flex-start;margin-top:16px}}
.letter{{font:800 15px -apple-system;color:#8a2b2b;border:1.5px solid #8a2b2b;border-radius:5px;padding:1px 8px;background:#fffdf6;margin-right:10px;margin-top:8px}}
.barsbox{{flex:1}}
.row{{display:grid;grid-template-columns:repeat(4,1fr);border-left:2px solid #1c1c1c}}
.bar{{border-right:1px solid #b9b09a;border-bottom:1px solid #e5dcc6;padding:12px 8px;min-height:36px;display:grid;grid-template-columns:repeat(4,1fr);align-items:center;background:#fffdf6}}
.q{{grid-row:1}} .q0{{grid-column:1}} .q1{{grid-column:2}} .q2{{grid-column:3}} .q3{{grid-column:4}}
.bar b{{font:italic 700 21px Georgia}} sup{{font-size:12px;font-weight:700}}
.bass{{opacity:.65;font-size:13px}} .nc{{color:#8a8371;font-style:italic}} .carry{{opacity:.6}}
audio{{width:100%;margin:10px 0}}</style></head><body><main>
<h1>{model['title']}</h1>
<div class="sub">{model['keyName']} · {model['bpb']}/4 · <a href="/">chart complet</a></div>
<audio id="aud" controls src="{model['audio_url']}"></audio>
<p>{chips_html}</p>
{body}
<script>let _st=null;
function play(t0,t1){{const a=document.getElementById('aud');a.currentTime=t0;a.play();
if(_st)clearTimeout(_st);_st=setTimeout(()=>a.pause(),(t1-t0)*1000);}}</script>
</main></body></html>"""
