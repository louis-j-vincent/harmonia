"""One-off migration: patch the static wheel CSS/JS in already-generated chart
HTML files to the rotary-dial version, without re-running inference.

The blocks below are copy-pasted verbatim from the old vs new
harmonia/output/chart_interactive.py template — they're song-data-independent
boilerplate, so a straight string swap upgrades any previously rendered file.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLOTS_DIR = REPO / "docs" / "plots"

OLD_CSS = """  .wheel { position:relative; width:112px; height:112px; border-radius:50%;
           flex:0 0 auto; }
  .wheel button { position:absolute; left:50%; top:50%; width:31px; height:31px;
                  margin:-15.5px 0 0 -15.5px; border-radius:50%;
                  border:1px solid #0003; color:#242018; font:700 11px system-ui,sans-serif;
                  cursor:pointer; box-shadow:0 1px 2px #0002; }
  .wheel button[aria-pressed=true] { outline:3px solid var(--accent); outline-offset:1px;
                                     color:#111; }
  .wheel .hub { position:absolute; left:50%; top:50%; transform:translate(-50%,-50%);
                width:39px; height:39px; border-radius:50%; background:#f7f3e9e8;
                border:1px solid #cfc7ae; display:flex; align-items:center;
                justify-content:center; font:700 11px system-ui,sans-serif; color:#4a4636; }"""

NEW_CSS = """  .wheel { position:relative; width:112px; height:112px; border-radius:50%;
           flex:0 0 auto; touch-action:none; -webkit-user-select:none; user-select:none;
           background:radial-gradient(circle at 50% 42%,#f2ead4 0%,#ddd0ac 72%,#c3b48c 100%);
           box-shadow:inset 0 1px 4px #0003, 0 1px 2px #0002; }
  .wheel-ring { position:absolute; inset:0; transform-origin:50% 50%; will-change:transform; }
  .wheel-ring.snap { transition:transform .45s cubic-bezier(.32,1.5,.55,1); }
  .wheel-ring.snap .lbl { transition:transform .45s cubic-bezier(.32,1.5,.55,1); }
  .wheel button { position:absolute; left:50%; top:50%; width:31px; height:31px;
                  margin:-15.5px 0 0 -15.5px; border-radius:50%; padding:0;
                  border:1px solid #0003; color:#242018; font:700 11px system-ui,sans-serif;
                  cursor:grab; box-shadow:0 1px 2px #0002; -webkit-tap-highlight-color:transparent; }
  .wheel button .lbl { display:block; pointer-events:none; }
  .wheel button[aria-pressed=true] { outline:3px solid var(--accent); outline-offset:1px;
                                     color:#111; }
  .wheel-pointer { position:absolute; top:-6px; left:50%; transform:translateX(-50%);
                   width:0; height:0; border-left:6px solid transparent;
                   border-right:6px solid transparent; border-bottom:9px solid var(--accent);
                   pointer-events:none; filter:drop-shadow(0 1px 1px #0003); }
  .wheel .hub { position:absolute; left:50%; top:50%; transform:translate(-50%,-50%);
                width:39px; height:39px; border-radius:50%; background:#f7f3e9e8;
                border:1px solid #cfc7ae; display:flex; align-items:center;
                justify-content:center; font:700 11px system-ui,sans-serif; color:#4a4636;
                pointer-events:none; box-shadow:inset 0 1px 2px #0002; }"""

OLD_JS = """// ── build transpose wheel (chromatic order, CoF-derived colours) ──
(function(){
  const wheel=document.getElementById("transposeWheel");
  const field=document.getElementById("transpose");
  for(let off=0;off<12;off++){
    const ang=(-90+off*30)*Math.PI/180;  // chromatic: C at top, then C#, D, ...
    const b=document.createElement("button");
    b.type="button"; b.value=off;
    b.textContent=keyLabel(P.home.tonic,P.home.mode,off).replace(/ major| minor/,"");
    b.title=keyLabel(P.home.tonic,P.home.mode,off)+(off===0?" (original)":"");
    b.style.background=colOf({tonic:mod(P.home.tonic+off,12),kind:P.home.mode});
    b.style.transform=`translate(${Math.cos(ang)*42}px, ${Math.sin(ang)*42}px)`;
    b.addEventListener("click",()=>{field.value=off;render();});
    wheel.appendChild(b);
  }
  const hub=document.createElement("span");
  hub.className="hub"; hub.textContent="";
  wheel.appendChild(hub);
})();
function updateTransposeWheel(offset){
  document.querySelectorAll("#transposeWheel button").forEach(b=>{
    b.setAttribute("aria-pressed",parseInt(b.value)===offset?"true":"false");
  });
  document.getElementById("transposeLabel").textContent=keyLabel(P.home.tonic,P.home.mode,offset);
}"""

NEW_JS = """// ── build rotary transpose dial ─ drag/swipe to spin like an old phone dial,
//    or tap a key directly; snaps to the nearest of 12 chromatic stops ──
(function(){
  const wheel=document.getElementById("transposeWheel");
  const field=document.getElementById("transpose");
  const ring=document.createElement("div");
  ring.className="wheel-ring"; ring.id="wheelRing";
  wheel.appendChild(ring);
  const pointer=document.createElement("div");
  pointer.className="wheel-pointer";
  wheel.appendChild(pointer);

  const btns=[];
  for(let off=0;off<12;off++){
    const ang=(-90+off*30)*Math.PI/180;  // chromatic: C at top, then C#, D, ...
    const b=document.createElement("button");
    b.type="button"; b.value=off; b.dataset.ang=ang;
    const lbl=document.createElement("span");
    lbl.className="lbl";
    lbl.textContent=keyLabel(P.home.tonic,P.home.mode,off).replace(/ major| minor/,"");
    b.appendChild(lbl);
    b.title=keyLabel(P.home.tonic,P.home.mode,off)+(off===0?" (original)":"");
    b.style.background=colOf({tonic:mod(P.home.tonic+off,12),kind:P.home.mode});
    ring.appendChild(b);
    btns.push(b);
  }
  const hub=document.createElement("span");
  hub.className="hub"; hub.id="wheelHub"; hub.textContent="";
  wheel.appendChild(hub);

  function positionButtons(){
    const size=wheel.getBoundingClientRect().width||112;
    const radius=size*0.375;   // matches the original 42/112 ratio
    btns.forEach(b=>{
      const ang=parseFloat(b.dataset.ang);
      b.style.transform=`translate(${Math.cos(ang)*radius}px, ${Math.sin(ang)*radius}px)`;
    });
  }
  positionButtons();
  window.addEventListener("resize",positionButtons);

  let rot=0, dragging=false, startAngle=0, startRot=0, moved=0;

  function setLabelsRotation(deg){
    btns.forEach(b=>{ b.querySelector(".lbl").style.transform=`rotate(${-deg}deg)`; });
  }
  function applyRot(deg,animate){
    ring.classList.toggle("snap",!!animate);
    ring.style.transform=`rotate(${deg}deg)`;
    setLabelsRotation(deg);
  }
  function offsetFromRot(deg){
    return ((Math.round(-deg/30)%12)+12)%12;
  }
  applyRot(0,false);

  function angleAt(clientX,clientY){
    const r=wheel.getBoundingClientRect();
    return Math.atan2(clientY-(r.top+r.height/2),clientX-(r.left+r.width/2))*180/Math.PI;
  }
  wheel.addEventListener("pointerdown",e=>{
    dragging=true; moved=0;
    wheel.setPointerCapture(e.pointerId);
    startAngle=angleAt(e.clientX,e.clientY);
    startRot=rot;
    ring.classList.remove("snap");
  });
  wheel.addEventListener("pointermove",e=>{
    if(!dragging) return;
    let delta=angleAt(e.clientX,e.clientY)-startAngle;
    while(delta>180) delta-=360;
    while(delta<-180) delta+=360;
    moved=Math.max(moved,Math.abs(delta));
    rot=startRot+delta;
    applyRot(rot,false);
  });
  function endDrag(e){
    if(!dragging) return;
    dragging=false;
    const tapped = moved<8 ? e.target.closest("button") : null;
    let off;
    if(tapped){
      off=parseInt(tapped.value);
      let target=-off*30;
      while(target-rot>180) target-=360;
      while(target-rot<-180) target+=360;
      rot=target;
    } else {
      off=offsetFromRot(rot);
      rot=Math.round(rot/30)*30;
    }
    applyRot(rot,true);
    field.value=off;
    render();
  }
  wheel.addEventListener("pointerup",endDrag);
  wheel.addEventListener("pointercancel",endDrag);
})();
function updateTransposeWheel(offset){
  document.querySelectorAll("#transposeWheel button").forEach(b=>{
    b.setAttribute("aria-pressed",parseInt(b.value)===offset?"true":"false");
  });
  const label=keyLabel(P.home.tonic,P.home.mode,offset);
  document.getElementById("transposeLabel").textContent=label;
  const hub=document.getElementById("wheelHub");
  if(hub) hub.textContent=label.replace(/ major| minor/,"");
}"""


def main() -> None:
    files = sorted(PLOTS_DIR.glob("inferred_*.html"))
    patched, skipped = 0, 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        if OLD_CSS not in text or OLD_JS not in text:
            skipped += 1
            continue
        text = text.replace(OLD_CSS, NEW_CSS, 1).replace(OLD_JS, NEW_JS, 1)
        f.write_text(text, encoding="utf-8")
        patched += 1
        print(f"patched {f.name}")
    print(f"\n{patched} patched, {skipped} already up to date / not matching")


if __name__ == "__main__":
    sys.exit(main())
