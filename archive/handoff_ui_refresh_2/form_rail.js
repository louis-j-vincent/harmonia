/* ─────────────────────────────────────────────────────────────────────────
   form_rail.js — the whole song's form on ONE line, no horizontal scroll.

   PASTE inside window.APP's IIFE and use it INSTEAD of buildFormStrip().
   Depends on ui_kit.js (SZ) and on formRuns(), which is unchanged.

   Louis, 2026-08-07: "les A B devraient apparaitre en petit afin que toute la
   chanson puisse fitter horizontalement sous forme de A B C .." — so the rail
   never scrolls. Every run is an equal flex column; the LETTER shrinks as the
   song gets longer, the ROW does not (44px stays the tap target, which is what
   the finger needs — a 34×44 target is still a legal one, a 26px chip is not).

   Below ~14 runs the letters stay readable. Past that the rail switches to a
   two-line wrap rather than shrinking into illegibility — a 40-run song is not
   a strip you read, it is a strip you scrub, and that is the timeline's job.
   ───────────────────────────────────────────────────────────────────────── */

  function buildFormRail(){
    const runs=formRuns();
    if(runs.length<2) return null;

    // letter size vs. how many runs must fit across the screen
    const n=runs.length;
    const size = n<=6 ? 16 : n<=9 ? 15 : n<=12 ? 14 : 13;
    const wrap = n>14;

    const rail=el("div",
      `flex:0 0 auto;display:flex;gap:3px;padding:0 14px 12px;background:${T.paper};`+
      (wrap?"flex-wrap:wrap;row-gap:6px;":""));
    S._formChips=[];

    runs.forEach(r=>{
      const b=el("button",
        `flex:${wrap?"0 0 auto":"1"};min-width:${wrap?"46px":"0"};height:${SZ.control}px;`+
        `border-radius:10px;cursor:pointer;display:flex;align-items:${r.n>1?"baseline":"center"};`+
        `justify-content:center;gap:1px;font:700 ${size}px ${UI};color:${T.ink};`+
        `border:1px solid ${T.line};background:${T.card};`+
        `-webkit-tap-highlight-color:transparent;transition:background .14s,border-color .14s;`+
        (r.n>1?`padding-top:${Math.round(SZ.control/2-size/2)}px;`:""));
      b.appendChild(el("span","",r.label));
      if(r.n>1) b.appendChild(el("span",`font:600 9px ${UI};color:${T.faint};`,"×"+r.n));
      b.setAttribute("aria-label", r.label+(r.n>1?(" ×"+r.n):"")+", bars "+(r.b0+1)+"–"+(r.b1+1));
      b.title="bars "+(r.b0+1)+"–"+(r.b1+1);
      b.onclick=()=>{
        haptic();
        const dur=totalDur()||1;
        if(r.t0!=null && S.audio){ seekFrac(r.t0/dur); }
        else { setPlayhead(r.rb0>=0?r.rb0:0, 0, r.t0); }
        const cell=(S._cells||[]).find(c=>c.bar===r.rb0);
        if(cell) cell.el.scrollIntoView({block:"center",behavior:"smooth"});
      };
      S._formChips.push({el:b, t0:r.t0, t1:r.t1});
      rail.appendChild(b);
    });

    paintFormChip(S.playTime);   // unchanged — still keyed on TIME, not bar index
    return rail;
  }

  /* paintFormChip() keeps working as written; it only touches .style.background
     and .style.borderColor. If you want the active run to read at a glance in a
     dense rail, make the ON state solid rather than tinted: */
  function paintFormRail(t){
    paintFormChip(t);
    (S._formChips||[]).forEach(c=>{
      const on = c.el.style.borderColor && c.el.style.borderColor!==T.line;
      c.el.style.background = on ? "rgba(138,43,43,.12)" : T.card;
      c.el.style.fontWeight = on ? "800" : "700";
    });
  }
