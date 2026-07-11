"""One-off migration #2: add click sound + detent resistance to the rotary
dial already patched into docs/plots/inferred_*.html by migrate_wheel_to_rotor.py.
Song-data-independent boilerplate, straight string swap, no re-inference.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLOTS_DIR = REPO / "docs" / "plots"

OLD_CSS = """  .wheel-pointer { position:absolute; top:-6px; left:50%; transform:translateX(-50%);
                   width:0; height:0; border-left:6px solid transparent;
                   border-right:6px solid transparent; border-bottom:9px solid var(--accent);
                   pointer-events:none; filter:drop-shadow(0 1px 1px #0003); }"""

NEW_CSS = """  .wheel-pointer { position:absolute; top:-6px; left:50%; transform:translateX(-50%);
                   width:0; height:0; border-left:6px solid transparent;
                   border-right:6px solid transparent; border-bottom:9px solid var(--accent);
                   pointer-events:none; filter:drop-shadow(0 1px 1px #0003); }
  @keyframes wheelTick { 0%{transform:translateX(-50%) scale(1);}
                         45%{transform:translateX(-50%) scale(1.5);}
                         100%{transform:translateX(-50%) scale(1);} }
  .wheel-pointer.tick { animation:wheelTick .16s ease-out; }"""

OLD_JS = """  let rot=0, dragging=false, startAngle=0, startRot=0, moved=0;

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
    field.value=off;"""

NEW_JS = """  let rot=0, dragging=false, startAngle=0, startRot=0, moved=0, lastNotch=0;

  // ── detent feel: each 30° step gets a little "give" (sticky near the
  // centre of a stop, springs through past the boundary) plus a click ──
  let actx=null;
  function ensureAudio(){
    if(!actx){
      const AC=window.AudioContext||window.webkitAudioContext;
      if(!AC) return;
      actx=new AC();
    }
    if(actx.state==="suspended") actx.resume();
  }
  function playTick(){
    if(!actx) return;
    const t=actx.currentTime;
    const osc=actx.createOscillator(), gain=actx.createGain();
    osc.type="square";
    osc.frequency.setValueAtTime(1500,t);
    osc.frequency.exponentialRampToValueAtTime(650,t+0.018);
    gain.gain.setValueAtTime(0.05,t);
    gain.gain.exponentialRampToValueAtTime(0.0001,t+0.03);
    osc.connect(gain).connect(actx.destination);
    osc.start(t); osc.stop(t+0.04);
  }
  function pulsePointer(){
    pointer.classList.remove("tick");
    void pointer.offsetWidth;  // restart the CSS animation
    pointer.classList.add("tick");
  }
  function detent(raw){
    const notch=Math.round(raw/30)*30;
    const diff=raw-notch;                 // -15..15, distance from the stop
    const t=Math.max(-1,Math.min(1,diff/15));
    const shaped=Math.sign(t)*Math.pow(Math.abs(t),1.7)*15;  // resist near centre, give at edge
    return notch+shaped;
  }

  function setLabelsRotation(deg){
    btns.forEach(b=>{ b.querySelector(".lbl").style.transform=`rotate(${-deg}deg)`; });
  }
  function applyRot(deg,animate){
    ring.classList.toggle("snap",!!animate);
    const shown=animate?deg:detent(deg);
    ring.style.transform=`rotate(${shown}deg)`;
    setLabelsRotation(shown);
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
    ensureAudio();
    dragging=true; moved=0;
    wheel.setPointerCapture(e.pointerId);
    startAngle=angleAt(e.clientX,e.clientY);
    startRot=rot;
    lastNotch=Math.round(rot/30);
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
    const notch=Math.round(rot/30);
    if(notch!==lastNotch){ lastNotch=notch; playTick(); pulsePointer(); }
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
    playTick(); pulsePointer();
    field.value=off;"""


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
