/* ─────────────────────────────────────────────────────────────────────────
   prompter_keys.js — the play-along keyboards (2a).

   PASTE inside window.APP's IIFE, after renderVoicing(). Then swap the
   prompter's kb block as shown in HANDOFF.md §4.

   The shell already has renderKeys()/renderVoicing() drawing ONE keyboard for
   the whole voicing. The refresh asks for the SPLIT default to be two stacked
   keyboards, one per hand, each zoomed on its own octave — right hand on top
   (it is what you read while playing), left below.

   Hand split rule: the lowest note is the left hand; anything within 14
   semitones of it that is not part of the upper structure joins it. Everything
   else is the right hand. This matches how the voicing engine builds shell and
   rootless forms (voShell / voRootlessA), where the bass is always the outlier.
   ───────────────────────────────────────────────────────────────────────── */

  function splitHands(midis){
    const s=midis.slice().sort((a,b)=>a-b);
    if(s.length<=2) return {left:s.slice(0,1), right:s.slice(1)};
    const lo=s[0];
    const left=s.filter(m=>m-lo<=14 && s.indexOf(m)<=1);
    const right=s.filter(m=>left.indexOf(m)<0);
    return {left: left.length?left:[lo], right: right.length?right:s.slice(1)};
  }

  /* One octave-zoomed keyboard for one hand. `role` is "bass" (maroon) or
     "chord" (blue) and sets the fill; a note the hand may drop (an optional
     5th) passes role "opt" and renders as a tint with ink text. */
  function renderHand(host, midis, h, role){
    clear(host);
    if(!midis || !midis.length){ host.style.cssText=`position:relative;height:${h}px;`; return; }
    const lo = 12*Math.floor(Math.min(...midis)/12);
    const octaves = Math.max(1, Math.ceil((Math.max(...midis)-lo+1)/12));
    const on = new Set(midis);
    host.style.cssText=`position:relative;height:${h}px;`;

    const whitePc=[0,2,4,5,7,9,11], wOrder={0:0,2:1,4:2,5:3,7:4,9:5,11:6}, bLeft={1:0,3:1,6:3,8:4,10:5};
    const nW = octaves*7, wW = 100/nW;
    const fillFor = () => role==="bass" ? T.accent : role==="opt" ? "rgba(138,43,43,.3)" : "#2a6fb0";
    const inkFor  = () => role==="opt" ? T.accent : "#fff";

    for(let o=0;o<octaves;o++) for(const pc of whitePc){
      const midi=lo+o*12+pc, slot=o*7+wOrder[pc], lit=on.has(midi);
      const k=el("div",
        `position:absolute;top:0;bottom:0;left:${slot*wW}%;width:${wW}%;`+
        `border:1px solid ${T.rule};border-radius:0 0 5px 5px;`+
        `background:${lit?fillFor():T.card};`);
      if(lit) k.appendChild(el("span",
        `position:absolute;bottom:6px;left:0;right:0;text-align:center;`+
        `font:700 12px ${UI};color:${inkFor()};`, note(midi%12)));
      host.appendChild(k);
    }
    for(let o=0;o<octaves;o++) for(const pc of [1,3,6,8,10]){
      const midi=lo+o*12+pc, slot=o*7+bLeft[pc], lit=on.has(midi);
      const k=el("div",
        `position:absolute;top:0;height:62%;left:${(slot+1)*wW - wW*0.32}%;width:${wW*0.64}%;`+
        `border-radius:0 0 4px 4px;background:${lit?fillFor():"#2b2f36"};z-index:2;`);
      if(lit) k.appendChild(el("span",
        `position:absolute;bottom:4px;left:0;right:0;text-align:center;`+
        `font:700 11px ${UI};color:${inkFor()};`, note(midi%12)));
      host.appendChild(k);
    }
  }

  /* The card: header (note names + Two hands / One), then the keyboards.
     `getMidis()` returns the current voicing; call repaint() on chord change
     only (the prompter already tracks that with _lastCoachIdx). */
  function buildHandsCard(getMidis, onPlay){
    let split=(()=>{ try{ return localStorage.getItem("harmPrompterSplit")!=="0"; }catch(e){ return true; } })();

    const card=el("div",
      `width:min(100%,460px);background:${T.card};border:1px solid ${T.line};`+
      `border-radius:${SZ.radiusLg}px;padding:12px 12px 10px;margin-top:18px;cursor:pointer;`);
    const head=el("div","display:flex;align-items:center;gap:8px;margin-bottom:10px;");
    const names=el("div",`flex:1;min-width:0;font:600 13px ${UI};color:${T.faint};`);
    head.appendChild(names);
    const tog=kitSegmented([["split","Two hands"],["joined","One"]],
      ()=>split?"split":"joined",
      v=>{ split=(v==="split");
           try{ localStorage.setItem("harmPrompterSplit", split?"1":"0"); }catch(e){}
           repaint(); },
      {inline:true, height:34});
    tog.style.flex="0 0 auto";
    head.appendChild(tog);
    card.appendChild(head);

    const body=el("div","display:flex;flex-direction:column;gap:9px;");
    card.appendChild(body);
    card.onclick=e=>{ if(!tog.contains(e.target) && onPlay) onPlay(getMidis()); };

    function row(tag, host){
      const r=el("div","display:flex;align-items:center;gap:9px;");
      r.appendChild(el("div",`flex:0 0 auto;width:26px;font:700 11px ${UI};letter-spacing:.08em;color:${T.faint};`,tag));
      host.style.flex="1"; r.appendChild(host);
      return r;
    }

    function repaint(){
      clear(body);
      const midis=getMidis()||[];
      if(!midis.length){ names.textContent=""; return; }
      if(split){
        const {left,right}=splitHands(midis);
        names.innerHTML = `<span style="color:${T.accent}">L `+left.map(m=>note(m%12)).join(" ")+
                          `</span> · <span style="color:#2a6fb0">R `+right.map(m=>note(m%12)).join(" ")+`</span>`;
        const rh=el("div",""), lh=el("div","");
        body.appendChild(row("R", rh));
        body.appendChild(row("L", lh));
        renderHand(rh, right, 64, "chord");
        renderHand(lh, left, 64, "bass");
      } else {
        names.innerHTML = `<span style="color:${T.faint}">`+midis.map(m=>note(m%12)).join(" ")+`</span>`;
        const kb=el("div","");
        body.appendChild(row("", kb));
        renderVoicing(kb, midis, 88, true);
      }
    }
    repaint();
    card._repaint=repaint;
    return card;
  }
