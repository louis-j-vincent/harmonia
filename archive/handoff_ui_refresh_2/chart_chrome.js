/* ─────────────────────────────────────────────────────────────────────────
   chart_chrome.js — replaces the four-row header of renderChart() with one
   toolbar + one bottom dock, and fixes the duplicated FORM strip.

   PASTE inside window.APP's IIFE, after buildFormStrip(). Then edit
   renderChart() as shown in HANDOFF.md §2. Depends on ui_kit.js.
   ───────────────────────────────────────────────────────────────────────── */

  /* ONE toolbar. Key, mode and form all used to claim their own row; key and
     form become the title's subtitle line (tap = rotor), and mode moves to the
     dock. Everything here is >= 44. */
  function chartToolbar(){
    const m=S.model;
    const bar=el("div",
      `flex:0 0 auto;display:flex;align-items:center;gap:12px;`+
      `padding:calc(6px + env(safe-area-inset-top)) 14px 12px;background:${T.paper};`);
    bar.appendChild(kitIcon("‹", ()=>go("library"), "Back"));

    const tw=el("div","flex:1;min-width:0;");
    tw.appendChild(el("div",
      `font:italic 600 19px ${SERIF};color:${T.ink};`+
      `overflow:hidden;text-overflow:ellipsis;white-space:nowrap;`, m?m.title:""));
    const meta=el("button",
      `display:inline-flex;align-items:center;gap:6px;margin-top:2px;border:none;`+
      `background:none;padding:6px 0;font:500 13px ${UI};color:${T.faint};cursor:pointer;`);
    meta.appendChild(el("span",`color:${T.ink};font-weight:600;`, note(S.key)+" "+(S.keyMode||"major")));
    if(m && m.form){ meta.appendChild(el("span","","·")); meta.appendChild(el("span","",m.form)); }
    meta.appendChild(el("span",`color:${T.accent};`,"↻"));
    meta.onclick=openRotor;
    tw.appendChild(meta);
    bar.appendChild(tw);

    bar.appendChild(kitIcon("Aa", openPrefsSheet, "Visualisation preferences", {serif:true}));
    return bar;
  }

  /* The dock: transport + the Read/Analyse/Annotate segmented, pinned in thumb
     reach. This REPLACES both the old modeBar and the separate transport strip
     — syncTransport() should target this element (HANDOFF.md §3). */
  function chartDock(){
    const dock=el("div",
      `flex:0 0 auto;background:${T.card};border-top:1px solid ${T.line};`+
      `padding:12px 14px calc(10px + env(safe-area-inset-bottom));`+
      `box-shadow:0 -12px 30px -22px rgba(50,38,20,.5);`);

    const row=el("div","display:flex;align-items:center;gap:14px;");
    const play=el("button",
      `flex:0 0 auto;width:${SZ.primary}px;height:${SZ.primary}px;border-radius:50%;border:none;`+
      `background:${T.accent};color:#fff;font:400 22px ${UI};cursor:pointer;`+
      `display:flex;align-items:center;justify-content:center;`, S.playing?"❚❚":"▶");
    play.setAttribute("aria-label", S.playing?"Pause":"Play");
    play.onclick=togglePlay;
    row.appendChild(play);

    const mid=el("div","flex:1;min-width:0;");
    const track=el("div",`height:6px;border-radius:3px;background:${T.line};overflow:hidden;`);
    const fill=el("div",`width:0%;height:100%;background:${T.accent};border-radius:3px;`);
    track.appendChild(fill);
    const labels=el("div",
      `display:flex;justify-content:space-between;margin-top:6px;font:500 12px ${UI};color:${T.faint};`);
    const tNow=el("span","","0:00"), tWhere=el("span","",""), tEnd=el("span","","0:00");
    labels.appendChild(tNow); labels.appendChild(tWhere); labels.appendChild(tEnd);
    mid.appendChild(track); mid.appendChild(labels);
    row.appendChild(mid);
    S._dockFill=fill; S._dockNow=tNow; S._dockWhere=tWhere; S._dockEnd=tEnd;

    const loop=el("button",
      `flex:0 0 auto;width:${SZ.primary}px;height:${SZ.primary}px;border-radius:${SZ.radiusLg}px;`+
      `cursor:pointer;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px;`+
      ctlSkin(S._loopEngine?"on":"off"));
    loop.appendChild(el("div",`font:700 15px ${UI};`,"A–B"));
    loop.appendChild(el("div",`font:500 10px ${UI};opacity:.7;`,"loop"));
    loop.onclick=()=>{ loopEngineToggle(); go("chart",{push:false}); };
    row.appendChild(loop);
    dock.appendChild(row);

    const modes=kitSegmented(
      [["read","Read"],["analyse","Analyse"],["annotate","Annotate"]],
      ()=>S.mode,
      v=>{ S.mode=v; go("chart",{push:false}); });
    modes.style.marginTop="12px";
    dock.appendChild(modes);
    return dock;
  }

  /* The Analyse lens row + its ONE-LINE legend. Replaces the three outlined
     pills and the full-width legendFor() sentence block. */
  function analyseLens(){
    const wrap=el("div",`flex:0 0 auto;padding:0 14px 10px;background:${T.paper};`);
    wrap.appendChild(kitSegmented(
      [["function","Role"],["key","Local keys"],["global","Home key"]],
      ()=>S.colorMode,
      v=>{ S.colorMode=v; go("chart",{push:false}); },
      {height:SZ.control}));
    const L={
      function:[T.green, "colour marks what each chord does — home, setting up, pulling"],
      key:[lensBand(210,false), "colour marks where the tune borrows another key"],
      global:[lensBand(keyHue(collOf(S.key,S.keyMode)), S.keyMode==="minor"), "one wash — the whole song in its home key"]
    }[S.colorMode] || [null,""];
    wrap.appendChild(kitLegend(L[0], L[1]));
    return wrap;
  }
