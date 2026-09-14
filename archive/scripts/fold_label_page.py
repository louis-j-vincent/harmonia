"""Render the ear-labelling page from the deck, data inlined.

Served by the EXISTING `/reports/<name>` route — no server.py change, because a
concurrent session owns that file. The deck JSON is inlined rather than fetched
so the page never depends on that route serving .json.

    http://localhost:7772/reports/fold_label_deck.html
    (or http://louiss-macbook-air:7772/... from the phone, via Tailscale)

Verdicts live in localStorage and are exported by copy-to-clipboard — no write
endpoint exists on :7772 and inventing one is another session's file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DECK = HERE / "harmonia_min/state/reports/fold_label_deck.json"
OUT = HERE / "harmonia_min/state/reports/fold_label_deck.html"

PAGE = """<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Un bloc ou deux ?</title>
<style>
 *{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
 body{margin:0;background:#e7e0d0;font:16px/1.45 -apple-system,BlinkMacSystemFont,system-ui,sans-serif;color:#1c1c1c}
 .wrap{max-width:640px;margin:0 auto;padding:14px 14px calc(20px + env(safe-area-inset-bottom))}
 .top{display:flex;align-items:baseline;gap:10px;margin-bottom:10px}
 .prog{font:700 13px system-ui;color:#8a8371}
 .tier{font:700 10px system-ui;letter-spacing:.06em;text-transform:uppercase;
       border:1.5px solid #8a2b2b;color:#8a2b2b;border-radius:5px;padding:1px 6px}
 .card{background:#fffdf6;border:1px solid #e5dcc6;border-radius:16px;padding:16px;
       box-shadow:0 10px 26px -18px rgba(50,35,20,.5)}
 h1{font:italic 600 19px Georgia,serif;margin:0 0 3px}
 .sub{font:500 12px system-ui;color:#8a8371;margin-bottom:14px}
 .pass{display:flex;gap:9px;margin-bottom:9px}
 .pass button{flex:1;min-width:0;border:1.5px solid #b9b09a;background:#f7f3e9;border-radius:12px;
   padding:13px 8px;font:600 14px system-ui;color:#1c1c1c;cursor:pointer;text-align:left}
 .pass button.on{border-color:#8a2b2b;background:#8a2b2b;color:#fff}
 .pass small{display:block;font:500 11px system-ui;opacity:.72;margin-top:3px}
 .chain{width:100%;border:1.5px dashed #b9b09a;background:transparent;border-radius:12px;
   padding:11px;font:600 13px system-ui;color:#8a8371;cursor:pointer;margin-bottom:16px}
 .ask{font:italic 15px Georgia,serif;margin:0 0 10px}
 .verdicts{display:flex;flex-direction:column;gap:8px}
 .verdicts button{border:none;border-radius:13px;padding:16px;font:700 15px system-ui;cursor:pointer}
 .one{background:#1f8a5b;color:#fff} .two{background:#8a2b2b;color:#fff}
 .dunno{background:#eee7d6;color:#8a8371}
 .reveal{margin-top:14px;font:500 12px system-ui;color:#8a8371;
   background:#f7f3e9;border:1px solid #e5dcc6;border-radius:10px;padding:10px;display:none}
 .foot{margin-top:18px;display:flex;gap:8px;flex-wrap:wrap}
 .foot button{flex:1;border:1px solid #b9b09a;background:#fffdf6;border-radius:10px;
   padding:11px;font:600 12px system-ui;color:#8a8371;cursor:pointer}
 textarea{width:100%;height:110px;margin-top:10px;font:11px ui-monospace,monospace;
   border:1px solid #e5dcc6;border-radius:8px;padding:8px;display:none;background:#fffdf6}
 .done{text-align:center;padding:40px 10px}
 .done h2{font:italic 600 22px Georgia,serif}
</style></head><body><div class="wrap" id="app"></div>
<audio id="au" preload="auto" playsinline></audio>
<script>
const DECK = __DECK__;
const KEY = "foldLabels_v1";
let V = {}; try { V = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e){}
const au = document.getElementById("au");
let stopAt = null, chainNext = null;

au.addEventListener("timeupdate", () => {
  if (stopAt != null && au.currentTime >= stopAt) {
    au.pause(); stopAt = null;
    document.querySelectorAll(".pass button").forEach(b => b.classList.remove("on"));
    if (chainNext) { const f = chainNext; chainNext = null; setTimeout(f, 260); }
  }
});
function play(src, t0, t1, btn, then) {
  if (au.getAttribute("src") !== src) { au.setAttribute("src", src); au.load(); }
  document.querySelectorAll(".pass button").forEach(b => b.classList.remove("on"));
  if (btn) btn.classList.add("on");
  chainNext = then || null;
  const go = () => { try { au.currentTime = t0; } catch(e){} stopAt = t1; au.play().catch(()=>{}); };
  if (au.readyState >= 1) go(); else au.addEventListener("loadedmetadata", go, {once:true});
}
const next = () => DECK.findIndex(c => !V[c.id]);
const fmt = s => (s < 60 ? Math.round(s) + " s" : Math.floor(s/60) + " min " + Math.round(s%60) + " s");

function render() {
  const i = next(), app = document.getElementById("app");
  const done = Object.keys(V).length;
  if (i < 0) {
    app.innerHTML = `<div class="card done"><h2>Fini — ${done} verdicts</h2>
      <p style="font:500 13px system-ui;color:#8a8371">Copie-les et envoie-les moi.</p></div>
      <div class="foot"><button onclick="copyOut()">Copier les verdicts</button>
      <button onclick="reset()">Tout effacer</button></div><textarea id="out"></textarea>`;
    return;
  }
  const c = DECK[i];
  app.innerHTML = `
   <div class="top"><span class="prog">${done} faits · ${DECK.length - done} restants</span>
     <span class="tier">${c.tier === "A" ? "cas clé" : c.tier === "B" ? "limite" :
                          c.tier === "D" ? "contrôle" : "net"}</span></div>
   <div class="card">
     <h1>${c.title}</h1>
     <div class="sub">deux passages du même morceau</div>
     <div class="pass">
       <button id="pa">▶ Passage 1<small>mes. ${c.a.b0+1}–${c.a.b1+1} · ${c.a.n} mes. · ${fmt(c.a.t1-c.a.t0)}</small></button>
       <button id="pb">▶ Passage 2<small>mes. ${c.b.b0+1}–${c.b.b1+1} · ${c.b.n} mes. · ${fmt(c.b.t1-c.b.t0)}</small></button>
     </div>
     <button class="chain" id="pc">▶ enchaîner 1 puis 2</button>
     <p class="ask">Faut-il les écrire comme <b>un seul bloc</b>, ou comme <b>deux blocs différents</b> ?</p>
     <div class="verdicts">
       <button class="one" onclick="vote('one')">Un seul bloc</button>
       <button class="two" onclick="vote('two')">Deux blocs</button>
       <button class="dunno" onclick="vote('skip')">Je ne sais pas / pas net</button>
     </div>
     <div class="reveal" id="rev"></div>
   </div>
   <div class="foot"><button onclick="copyOut()">Copier les verdicts</button>
     <button onclick="undo()">Annuler le dernier</button></div>
   <textarea id="out"></textarea>`;
  document.getElementById("pa").onclick = e =>
    play(c.audio, c.a.t0, c.a.t1, e.currentTarget);
  document.getElementById("pb").onclick = e =>
    play(c.audio, c.b.t0, c.b.t1, e.currentTarget);
  document.getElementById("pc").onclick = () =>
    play(c.audio, c.a.t0, c.a.t1, document.getElementById("pa"),
         () => play(c.audio, c.b.t0, c.b.t1, document.getElementById("pb")));
}
let lastId = null;
function vote(v) {
  const c = DECK[next()];
  V[c.id] = {v, t: Date.now(), align: c.align, len: c.len, tier: c.tier,
             song: c.song, a: [c.a.b0, c.a.b1], b: [c.b.b0, c.b.b1],
             same_letter: c.same_letter, shift: c.shift_bars};
  lastId = c.id;
  localStorage.setItem(KEY, JSON.stringify(V));
  au.pause(); stopAt = null; chainNext = null;
  render();
  const r = document.getElementById("rev");
  if (r) { r.style.display = "block";
    r.textContent = `précédent : ${c.title} — la mesure disait align ${c.align}` +
      (c.shift_bars ? ` (après décalage de ${c.shift_bars} mes.)` : "") +
      `, longueurs ${c.a.n} vs ${c.b.n}. Ton verdict : ` +
      (v === "one" ? "un bloc" : v === "two" ? "deux blocs" : "pas net"); }
}
function undo(){ if(lastId){ delete V[lastId]; lastId=null;
  localStorage.setItem(KEY, JSON.stringify(V)); render(); } }
function reset(){ if(confirm("Effacer tous les verdicts ?")){ V={};
  localStorage.removeItem(KEY); render(); } }
function copyOut(){
  const txt = JSON.stringify(V);
  const ta = document.getElementById("out");
  ta.style.display = "block"; ta.value = txt; ta.select();
  navigator.clipboard && navigator.clipboard.writeText(txt).catch(()=>{});
}
render();
</script></body></html>
"""


def main():
    deck = json.load(open(DECK))
    OUT.write_text(PAGE.replace("__DECK__", json.dumps(deck, ensure_ascii=False)))
    from collections import Counter
    print(f"{len(deck)} cards, {len({c['song'] for c in deck})} songs "
          f"{Counter(c['tier'] for c in deck)}")
    print(f"-> {OUT}  ({OUT.stat().st_size // 1024} KB)")
    print("   http://localhost:7772/reports/fold_label_deck.html")


if __name__ == "__main__":
    main()
