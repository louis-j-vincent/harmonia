/* ─────────────────────────────────────────────────────────────────────────
   loading_chart.js — the two-phase loading screen (design 2b).

   PASTE inside window.APP's IIFE and use it INSTEAD of renderAnalysing() /
   paintAnalysing(). Depends on ui_kit.js and on the job fields defined in
   progressive_job.py (phase, raw_model, n_bars, n_chords, sections_found).

   Read progressive_job.py's header first. The short version: musx returns the
   whole decode at once, so bars do NOT trickle in. There is a genuine wait,
   then the RAW CHART appears complete (one section, no letters), then the
   letters fold in over the same grid.

   The trick that makes the hand-off invisible: phase "raw" hands the shell a
   real ChartModel, so we call the SAME loadModel() + buildIReal() the final
   chart uses. When the real model arrives we reload and re-render — the grid
   is identical, so nothing jumps.

   renderChordPreview(), drawCircleOfFifths() and the STAGES list are retired
   by this; keep STAGES only for naming a failed stage on the error path.
   ───────────────────────────────────────────────────────────────────────── */

  const PHASE_COPY={
    listening:["listening to the track",      "finding the beat"],
    decoding: ["reading the harmony",         "this is the long part — a minute or so"],
    raw:      ["the chart, bar by bar",       "no letters yet — you can already play along"],
    sections: ["finding the sections",        "grouping the repeats into A, B, C…"],
    done:     ["ready",                       ""]
  };

  function renderLoading(){
    const w=screenWrap();

    const bar=el("div",
      `flex:0 0 auto;display:flex;align-items:center;gap:12px;`+
      `padding:calc(6px + env(safe-area-inset-top)) 14px 12px;background:${T.paper};`);
    bar.appendChild(kitIcon("‹", ()=>{ S.job=null; go("library"); }, "Back"));
    const tw=el("div","flex:1;min-width:0;");
    const title=el("div",
      `font:italic 600 19px ${SERIF};color:${T.ink};`+
      `overflow:hidden;text-overflow:ellipsis;white-space:nowrap;`);
    const sub=el("div",`font:500 13px ${UI};color:${T.faint};margin-top:2px;`);
    tw.appendChild(title); tw.appendChild(sub); bar.appendChild(tw); w.appendChild(bar);

    /* Two segments, not one bar: the wait and the sectioning are different
       kinds of work, and a single bar that jumps from 40% to 100% when the
       chart appears reads as a lie. */
    const prog=el("div",`flex:0 0 auto;padding:0 14px 14px;background:${T.paper};`);
    const segs=el("div","display:flex;gap:5px;");
    const s1=el("div",`flex:1;height:6px;border-radius:3px;background:${T.line};overflow:hidden;`);
    const s1f=el("div",`width:0%;height:100%;background:${T.amber};transition:width .5s ease;`);
    const s2=el("div",`flex:1;height:6px;border-radius:3px;background:${T.line};overflow:hidden;`);
    const s2f=el("div",`width:0%;height:100%;background:${T.accent};transition:width .5s ease;`);
    s1.appendChild(s1f); s2.appendChild(s2f); segs.appendChild(s1); segs.appendChild(s2);
    prog.appendChild(segs);
    const meta=el("div",
      `display:flex;justify-content:space-between;margin-top:8px;font:500 13px ${UI};color:${T.faint};`);
    prog.appendChild(meta); w.appendChild(prog);

    const body=el("div",`flex:1;min-height:0;overflow-y:auto;padding:0 10px;background:${T.paper};`);
    body.className="ap-scroll"; w.appendChild(body);

    const foot=el("div",
      `flex:0 0 auto;padding:14px 14px calc(14px + env(safe-area-inset-bottom));background:${T.paper};`);
    w.appendChild(foot);

    S._load={title, sub, s1f, s2f, meta, body, foot, shown:null};
    paintLoading();
    return w;
  }

  function paintLoading(){
    const L=S._load, j=S.job;
    if(!L || !j) return;
    const phase=j.phase||"listening";

    L.title.textContent=j.title||"your song";

    if(j.error){
      L.sub.textContent="something went wrong";
      clear(L.body);
      const card=el("div",
        `margin:18px 4px;background:${T.card};border:1px solid ${T.line};`+
        `border-radius:${SZ.radiusLg}px;padding:20px;`);
      card.appendChild(el("div",`font:600 15px ${UI};color:${T.accent};margin-bottom:6px;`,"That didn't work"));
      card.appendChild(el("div",`font:500 13px/1.5 ${UI};color:${T.faint};`, j.error));
      L.body.appendChild(card);
      clear(L.foot);
      L.foot.appendChild(kitButton("Back to library", ()=>{ S.job=null; go("library"); },
        {height:SZ.primary, grow:true, state:"on"}));
      return;
    }

    const copy=PHASE_COPY[phase]||PHASE_COPY.listening;
    L.sub.textContent=copy[0];

    const done = phase==="done";
    const hasRaw = phase==="raw" || phase==="sections" || done;
    L.s1f.style.width = hasRaw ? "100%" : (phase==="decoding" ? "62%" : "18%");
    if(hasRaw) L.s1f.style.background=T.green;
    L.s2f.style.width = done ? "100%" : (phase==="sections" ? "45%" : "0%");
    if(done) L.s2f.style.background=T.green;

    clear(L.meta);
    if(hasRaw && j.n_bars){
      const a=el("div","");
      a.appendChild(el("span",`color:${done?T.green:T.ink};font-weight:700;`, j.n_bars+" bars"));
      a.appendChild(el("span","", done ? " · "+(j.sections_found||1)+" sections" : " decoded"));
      L.meta.appendChild(a);
    } else {
      L.meta.appendChild(el("span","", copy[1]));
    }
    if(hasRaw && !done) L.meta.appendChild(el("span","","finding the sections…"));

    /* ── the body ──────────────────────────────────────────────────────────
       Before the raw chart: a quiet waiting state, NOT a fake grid. After it:
       the real chart, rendered by the real code path. */
    if(!hasRaw){
      if(L.shown!=="wait"){
        clear(L.body);
        const wrap=el("div",
          `margin:34px 4px;display:flex;flex-direction:column;align-items:center;gap:14px;text-align:center;`);
        wrap.appendChild(el("div",
          `width:34px;height:34px;border-radius:50%;border:2px solid ${T.line};`+
          `border-top-color:${T.accent};animation:ap-spin .8s linear infinite;`));
        wrap.appendChild(el("div",`font:600 15px ${UI};color:${T.ink};`, copy[0]));
        wrap.appendChild(el("div",
          `font:italic 13.5px/1.6 ${SERIF};color:${T.faint};max-width:280px;`, copy[1]));
        L.body.appendChild(wrap);
        L.shown="wait";
      }
      clear(L.foot);
      L.foot.appendChild(el("div",`height:${SZ.primary}px;`));   // hold the space
      return;
    }

    if(L.shown!=="chart" && j.raw_model){
      clear(L.body);
      loadModel(j.raw_model);          // the real path — same cells as the final chart
      const head=el("div","display:flex;align-items:center;gap:8px;margin:0 4px 8px;");
      head.appendChild(el("div",
        `font:600 11px ${UI};letter-spacing:.08em;text-transform:uppercase;color:${T.faint};`,
        "raw chart · no letters yet"));
      head.appendChild(el("div",`flex:1;height:1px;background:${T.line};`));
      L.body.appendChild(head);
      L.body.appendChild(buildIReal());
      L.body.appendChild(kitLegend(T.green, "every bar written out — you can already play along"));
      L.shown="chart";
    }

    clear(L.foot);
    if(done){
      L.foot.appendChild(kitButton("Open the chart", ()=>openChart(j.file),
        {height:SZ.primary, grow:true, state:"on"}));
    } else {
      const busy=el("div",
        `height:${SZ.primary}px;border-radius:${SZ.radiusLg}px;border:1px solid ${T.line};`+
        `background:${T.card};display:flex;align-items:center;justify-content:center;gap:10px;`+
        `font:600 15px ${UI};color:${T.faint};`);
      busy.appendChild(el("div",
        `width:14px;height:14px;border-radius:50%;border:2px solid ${T.line};`+
        `border-top-color:${T.accent};animation:ap-spin .7s linear infinite;`));
      busy.appendChild(el("span","","Finding the sections"));
      L.foot.appendChild(busy);
    }
  }
