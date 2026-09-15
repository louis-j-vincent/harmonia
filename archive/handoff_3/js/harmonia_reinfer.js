/* ============================================================================
 * Harmonia — Re-infer with my corrections (collaborative loop + section merge)
 * Drop-in, dependency-free. Plain vanilla JS + DOM (no framework, no build).
 *
 * USE:
 *   <script src="harmonia_reinfer.js"></script>
 *   <div data-ri="desktop"></div>   (or data-ri="phone")
 *   (auto-initialises on DOM ready; styles are injected automatically.)
 *
 * WHAT IT DOES:
 *   - Renders the chart with per-chord confidence % always visible; shaky chords
 *     render at reduced level-of-detail (family only) + a "?".
 *   - Tap a chord -> confirm sheet (candidates) -> hard-clamps it (c=1).
 *   - "Re-infer with N fixes" -> POST {confirms:[{t0,t1,root,q}], merges:[]} ->
 *     applies the response diff (Harte "root:quality" labels) by time overlap,
 *     then shows which nearby chords the fix sharpened.
 *   - Form ribbon (A¹ A² B C): two-tap two sections -> merge sheet -> its OWN
 *     re-infer: POST {confirms:[], merges:[[idA,idB]]} -> pooled reading.
 *
 * ⇩ SWAP FOR LIVE (one line, in reinfer()): uncomment the fetch, delete the mock.
 *   The mock returns the EXACT /api/reinfer contract shape:
 *     {key, tempo_bpm, n_changed, diff:[{index,start_s,end_s,old_label,new_label,
 *      old_confidence,new_confidence}], chords:[]}
 *
 * DATA (swap for real Harmonia data — shapes are already what Harmonia produces):
 *   CH  = chords: {root:0..11, q:iReal token, c:certainty 0..1, t0,t1:seconds}
 *   SECTIONS = form: {id, label, tag, bars}
 *   REINFER_DIFF / MERGE_DIFF = scripted demo responses; live server replaces these.
 * ========================================================================== */
(function(){var s=document.createElement('style');s.textContent="\n\n  * { box-sizing: border-box; }\n  body { margin: 0; background: #e7e0d0; }\n  a { color: #8a2b2b; } a:hover { color: #6f2020; }\n  @keyframes ri-in { from { opacity:0; transform:translateY(8px);} to { opacity:1; transform:translateY(0);} }\n  @keyframes ri-spin { to { transform: rotate(360deg); } }\n  @keyframes ri-halo { 0% { box-shadow:0 0 0 0 rgba(31,138,91,.0);} 30% { box-shadow:0 0 0 5px rgba(31,138,91,.35);} 100% { box-shadow:0 0 0 0 rgba(31,138,91,0);} }\n  @keyframes ri-flip { 0% { opacity:0; transform:translateY(-6px);} 100% { opacity:1; transform:translateY(0);} }\n  @keyframes ri-sheet { from { transform:translateY(100%);} to { transform:translateY(0);} }\n  @keyframes ri-toast { 0%{opacity:0;transform:translate(-50%,8px);} 12%,82%{opacity:1;transform:translate(-50%,0);} 100%{opacity:0;transform:translate(-50%,-6px);} }\n  .ri-scroll::-webkit-scrollbar{width:6px;} .ri-scroll::-webkit-scrollbar-thumb{background:#cdc4ad;border-radius:3px;}\n\n";document.head.appendChild(s);})();
window.RI = (function(){
  "use strict";
  const UI="-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif";
  const SERIF="Georgia,'Times New Roman',serif";
  const T={ paper:"#f7f3e9", card:"#fffdf6", ink:"#1c1c1c", rule:"#b9b09a", faint:"#8a8371", accent:"#8a2b2b", line:"#e5dcc6", deep:"#2a2622", green:"#1f8a5b", amber:"#c58a2e" };
  const FLAT=["C","D\u266d","D","E\u266d","E","F","G\u266d","G","A\u266d","A","B\u266d","B"];
  const mod=(n,m)=>((n%m)+m)%m;
  const note=pc=>FLAT[mod(pc,12)];
  const el=(t,c,x)=>{const e=document.createElement(t); if(c)e.style.cssText=c; if(x!=null)e.textContent=x; return e;};
  const API={};

  // quality token -> display tail (exact), base seventh, family suffix
  const TOK={"":"","6":"6","^7":"maj7","7":"7","-":"m","-7":"m7","-^7":"mMaj7","o":"dim","-7b5":"m7\u266d5","o7":"dim7","9":"9","7b9":"7\u266d9"};
  function seventhOf(q){ if(q==="6"||q==="^7")return "^7"; if(q==="-7b5")return "-7b5"; if(q==="o7"||q==="o")return q; if(q.indexOf("-")===0)return "-7"; if(q==="")return ""; return "7"; }
  function famSuffix(q){ if(q.indexOf("-7b5")===0||q.indexOf("o")===0)return "\u00b0"; if(q.indexOf("-")===0)return "m"; if(q.indexOf("+")===0)return "+"; if(q.indexOf("sus")===0)return "sus"; return ""; }
  // calibrated-confidence colour ramp (sure = ink, shaky = amber -> red)
  function confColor(c){ if(c>=.82)return T.ink; if(c>=.66)return "#5c4a30"; if(c>=.5)return "#9a6a1e"; if(c>=.38)return "#bd6a22"; return "#a8281f"; }
  function depthOf(c){ return c>=.66?"exact":(c>=.42?"seventh":"family"); }

  // typeset a chord glyph at a given depth (level-of-detail)
  function glyph(root, q, size, depth, color){
    const w=el("span", `font-family:${SERIF};font-style:italic;line-height:1;white-space:nowrap;color:${color};`);
    w.appendChild(el("span", `font-size:${size}px;font-weight:600;`, note(root)));
    let suf="";
    if(depth==="family") suf=famSuffix(q);
    else if(depth==="seventh") suf=TOK[seventhOf(q)]||"";
    else suf=TOK[q]!=null?TOK[q]:q;
    if(suf) w.appendChild(el("sup", `font-size:${Math.round(size*0.44)}px;font-weight:600;`, suf));
    return w;
  }

  // ── Autumn Leaves excerpt (key G# major — matches the mission's real inference).
  // 8 bars, one chord/bar. confidence = calibrated P(correct). A couple are left
  // genuinely shaky (low c) — those are the invitations to interact.
  // t0/t1 in seconds (what the /api/reinfer request needs).
  const CH=[
    {root:10,q:"-7",  c:.88, t0:0.00, t1:2.14},   // A#m7  (ii)
    {root:3, q:"7",   c:.79, t0:2.14, t1:4.28},   // D#7   (V)
    {root:8, q:"^7",  c:.94, t0:4.28, t1:6.04},   // G#maj7 (I)
    {root:1, q:"^7",  c:.46, t0:6.04, t1:6.68},   // C#maj7 (IV) — wobbly
    {root:7, q:"-7b5",c:.33, t0:6.68, t1:8.64},   // G(hdim) — SHAKY, the one to fix -> D7
    {root:7, q:"-7b5",c:.33, t0:8.64, t1:10.8},   // G(hdim) — SHAKY neighbour -> G7
    {root:6, q:"^7",  c:.40, t0:10.8, t1:12.9},   // F#maj7 — wobbly
    {root:10,q:"-7",  c:.90, t0:12.9, t1:15.0},   // A#m7
  ];
  // scripted /api/reinfer response for the demo: user confirms CH[4] as D7 ->
  // the joint decoder propagates, sharpening 3 nearby chords (labels + confidence).
  const REINFER_DIFF=[
    {i:5, oldRoot:7, oldQ:"-7b5", oldC:.33, root:7, q:"7",  c:.71},  // G(hdim) -> G7
    {i:6, oldRoot:6, oldQ:"^7",   oldC:.40, root:6, q:"7",  c:.77},  // F#maj7 -> F#7
    {i:3, oldRoot:1, oldQ:"^7",   oldC:.46, root:1, q:"7",  c:.68},  // C#maj7 -> C#7
  ];
  // scripted /api/reinfer response for a SECTION MERGE: pooling the two A
  // sections doubles the evidence on the shaky bars, resolving the G(hdim)
  // guesses into the real V-of-the-key and sharpening a neighbour.
  const MERGE_DIFF=[
    {i:4, oldRoot:7, oldQ:"-7b5", oldC:.33, root:2, q:"7", c:.76},  // G(hdim) -> D7  (V)
    {i:5, oldRoot:7, oldQ:"-7b5", oldC:.33, root:7, q:"7", c:.69},  // G(hdim) -> G7
    {i:6, oldRoot:6, oldQ:"^7",   oldC:.40, root:6, q:"7", c:.73},  // F#maj7 -> F#7
  ];
  // song form (this excerpt is the first A). The two A sections are the
  // natural merge — same letter, same changes, split by the section detector.
  const SECTIONS=[
    {id:"A1", label:"A", tag:"\u00b9", bars:"1\u20138"},
    {id:"A2", label:"A", tag:"\u00b2", bars:"9\u201316"},
    {id:"B",  label:"B", tag:"",       bars:"17\u201324"},
    {id:"C",  label:"C", tag:"",       bars:"25\u201332"},
  ];

  // ── Harte label parsing (the /api/reinfer response uses "root:quality") ──
  const NOTE_PC={C:0,"C#":1,Db:1,D:2,"D#":3,Eb:3,E:4,F:5,"F#":6,Gb:6,G:7,"G#":8,Ab:8,A:9,"A#":10,Bb:10,B:11};
  const HARTE_Q={maj:"",maj7:"^7",min:"-",min7:"-7","7":"7",hdim7:"-7b5",dim:"o",dim7:"o7","6":"6"};
  const NOTE_NAME=["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"];
  function parseLabel(lbl){ const p=String(lbl).split(":"); return { root: NOTE_PC[p[0]]!=null?NOTE_PC[p[0]]:0, q: HARTE_Q[p[1]]!=null?HARTE_Q[p[1]]:(p[1]||"") }; }
  function qToHarte(q){ for(const k in HARTE_Q){ if(HARTE_Q[k]===q) return k; } return "maj"; }
  function labelOf(root,q){ return NOTE_NAME[mod(root,12)]+":"+qToHarte(q); }
  function overlaps(a0,a1,b0,b1){ return Math.min(a1,b1)-Math.max(a0,b0) > 1e-6; }

  // Build the POST body from the user's confirmed chords (spans in SECONDS).
  function buildRequest(chords, pending, merges){
    return { confirms: pending.map(i=>{ const c=chords[i]; return { t0:c.t0, t1:c.t1, root:c.root, q:c.q }; }), merges: merges||[] };
  }
  // ⇩ SWAP FOR LIVE: uncomment the fetch line, delete the mock line.
  const FILENAME="inferred_autumn_leaves.html";
  async function reinfer(body){
    // return (await fetch("/api/reinfer/"+FILENAME,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)})).json();
    return mockReinfer(body);
  }
  // Mock returning the EXACT contract response shape (key/tempo/n_changed/chords/diff).
  function mockReinfer(body){
    const src = (body.merges && body.merges.length) ? MERGE_DIFF : REINFER_DIFF;
    const diff = src.map(d=>({ index:d.i, start_s:CH[d.i].t0, end_s:CH[d.i].t1,
      old_label:labelOf(d.oldRoot,d.oldQ), new_label:labelOf(d.root,d.q),
      old_confidence:d.oldC, new_confidence:d.c }));
    return { key:"G# major", tempo_bpm:112.3, n_changed:diff.length, diff, chords:[] };
  }

  API.build=function(host, device){
    host.style.position="relative"; host.innerHTML="";
    // per-instance working copy
    const chords=CH.map(c=>Object.assign({},c));
    let pending=[];            // indices confirmed this round
    let reinfering=false;
    const big = device!=="phone";

    const stack=el("div","display:flex;flex-direction:column;height:100%;");
    // header
    const head=el("div",`display:flex;align-items:center;justify-content:space-between;padding:${big?"4px 2px 12px":"2px 2px 10px"};flex:0 0 auto;`);
    const hl=el("div","display:flex;align-items:baseline;gap:10px;");
    const wm=el("div",`font:italic 600 20px ${SERIF};color:${T.ink};`); wm.innerHTML='harmon<span style="color:'+T.accent+'">ia</span>';
    hl.appendChild(wm);
    hl.appendChild(el("div",`font:500 12.5px ${UI};color:${T.faint};`,"Autumn Leaves \u00b7 G\u266f major"));
    head.appendChild(hl);
    stack.appendChild(head);

    // legend
    const legend=el("div",`display:flex;align-items:center;gap:12px;flex-wrap:wrap;padding:9px 12px;background:${T.card};border:1px solid ${T.line};border-radius:10px;margin-bottom:12px;flex:0 0 auto;`);
    legend.appendChild(el("span",`font:600 9.5px ${UI};letter-spacing:.07em;text-transform:uppercase;color:${T.faint};`,"How sure the model is"));
    const ramp=el("span",`width:96px;height:11px;border-radius:6px;background:linear-gradient(90deg,#a8281f,#bd6a22,#9a6a1e,${T.ink});`); legend.appendChild(ramp);
    legend.appendChild(el("span",`font:500 10.5px ${UI};color:${T.faint};`,"shaky chords show just the family \u2014 tap one you know to lock it in"));
    stack.appendChild(legend);

    // chart grid (iReal-ish, 4 per row)
    const bodyWrap=el("div",`flex:1;overflow-y:auto;`); bodyWrap.className="ri-scroll";
    const frame=el("div",`position:relative;background:${T.paper};border:1px solid ${T.line};border-left:3px double ${T.ink};border-radius:0 12px 12px 0;`);
    const grid=el("div","display:grid;grid-template-columns:repeat(4,1fr);");
    bodyWrap.appendChild(frame); frame.appendChild(grid);
    stack.appendChild(bodyWrap);

    // section letter box
    const secBox=el("div",`position:absolute;left:5px;top:8px;font:800 12px ${UI};color:${T.accent};border:1.5px solid ${T.accent};border-radius:4px;padding:1px 6px;background:${T.paper};pointer-events:none;`,"A");
    frame.appendChild(secBox);

    const cells=[];
    function renderCell(i){
      const c=chords[i];
      const col=i%4;
      const cell=el("button",`position:relative;display:flex;align-items:center;justify-content:center;min-height:${big?78:62}px;padding:0 8px 0 ${col===0?22:8}px;background:transparent;cursor:pointer;border:none;${col>0?`border-left:1px solid ${T.rule};`:""}${i>=4?`border-top:1px solid ${T.rule};`:""}-webkit-tap-highlight-color:transparent;transition:background .2s;`);
      const depth=c.confirmed?"exact":depthOf(c.c);
      const color=c.confirmed?T.ink:confColor(c.c);
      const g=glyph(c.root,c.q,big?30:24,depth,color);
      cell.appendChild(g);
      // always-on sureness read-out (percentage) under the chord
      if(!c.confirmed){
        cell.appendChild(el("span",`position:absolute;bottom:${big?7:5}px;left:0;right:0;text-align:center;font:600 ${big?11:10}px ${UI};color:${confColor(c.c)};opacity:.75;letter-spacing:.02em;`, Math.round(c.c*100)+"%"));
      }
      // low-confidence invitation: a soft "?" until confirmed
      if(!c.confirmed && c.c<.42){
        cell.appendChild(el("span",`position:absolute;top:6px;right:8px;font:700 12px ${UI};color:${confColor(c.c)};opacity:.8;`,"?"));
      }
      // confirmed ✓ pin (committed by fiat — never hedged)
      if(c.confirmed){
        cell.appendChild(el("span",`position:absolute;top:5px;right:7px;width:16px;height:16px;border-radius:50%;background:${T.accent};color:${T.paper};font:700 10px ${UI};display:flex;align-items:center;justify-content:center;`,"\u2713"));
      }
      cell.onclick=()=>openConfirm(i, cell);
      return cell;
    }
    function paintGrid(){
      grid.innerHTML=""; cells.length=0;
      for(let i=0;i<chords.length;i++){ const cl=renderCell(i); cells.push(cl); grid.appendChild(cl); }
    }
    paintGrid();

    // ── confirm / edit popover ──
    let pop=null;
    function closePop(){ if(pop){ pop.remove(); pop=null; } }
    function openConfirm(i, cell){
      if(reinfering) return;
      closePop();
      const c=chords[i];
      const back=el("div",`position:absolute;inset:0;z-index:30;background:rgba(28,24,20,.28);display:flex;align-items:flex-end;`);
      const sheet=el("div",`width:100%;background:${T.card};border-top-left-radius:20px;border-top-right-radius:20px;border-top:1px solid ${T.line};padding:16px 18px 20px;box-shadow:0 -12px 30px -14px rgba(50,35,20,.4);animation:ri-sheet .3s cubic-bezier(.32,.9,.35,1);`);
      sheet.appendChild(el("div",`width:36px;height:4px;border-radius:2px;background:${T.rule};margin:0 auto 12px;`));
      const cur=el("div","display:flex;align-items:center;gap:12px;margin-bottom:4px;");
      cur.appendChild(glyph(c.root,c.q,34,c.confirmed?"exact":depthOf(c.c),c.confirmed?T.ink:confColor(c.c)));
      const meta=el("div","");
      meta.appendChild(el("div",`font:600 12px ${UI};color:${c.confirmed?T.green:confColor(c.c)};`, c.confirmed?"confirmed by you":Math.round(c.c*100)+"% sure"));
      meta.appendChild(el("div",`font:500 11.5px ${UI};color:${T.faint};margin-top:2px;`, c.c<.42?"the model is guessing here \u2014 it only shows the family":"tap a candidate to lock it in"));
      cur.appendChild(meta); sheet.appendChild(cur);
      // candidates (the current exact + a couple alternates)
      const cands=candidatesFor(i);
      const list=el("div","display:flex;flex-direction:column;gap:7px;margin-top:12px;");
      cands.forEach(cd=>{
        const row=el("button",`display:flex;align-items:center;justify-content:space-between;gap:10px;background:${T.paper};border:1.5px solid ${T.line};border-radius:11px;padding:11px 13px;cursor:pointer;text-align:left;-webkit-tap-highlight-color:transparent;`);
        const lg=el("span","display:flex;align-items:baseline;gap:10px;");
        lg.appendChild(glyph(cd.root,cd.q,22,"exact",T.ink));
        lg.appendChild(el("span",`font:500 11.5px ${UI};color:${T.faint};`, cd.why));
        row.appendChild(lg);
        row.appendChild(el("span",`font:600 12px ${UI};color:${T.paper};background:${T.accent};border-radius:8px;padding:7px 12px;`,"Confirm"));
        row.onclick=()=>{ confirmChord(i, cd.root, cd.q); closePop(); };
        list.appendChild(row);
      });
      sheet.appendChild(list);
      const cancel=el("button",`width:100%;margin-top:12px;background:transparent;border:none;color:${T.faint};font:600 13px ${UI};padding:8px;cursor:pointer;`,"Cancel");
      cancel.onclick=closePop; sheet.appendChild(cancel);
      back.appendChild(sheet);
      back.onclick=(e)=>{ if(e.target===back) closePop(); };
      host.appendChild(back); pop=back;
    }
    function candidatesFor(i){
      const c=chords[i];
      // demo: the shaky G-hdim offers D7 (the correct fix) first
      if(i===4) return [ {root:2,q:"7",why:"V\u2077 \u2014 sets up the key"}, {root:7,q:"7",why:"secondary dominant"}, {root:c.root,q:c.q,why:"keep the model\u2019s guess"} ];
      const alt = c.q==="^7"?{root:c.root,q:"7",why:"dominant reading"}:{root:mod(c.root+2,12),q:"-7",why:"a fifth away"};
      return [ {root:c.root,q:c.q,why:"the model\u2019s pick"}, alt ];
    }
    function confirmChord(i, root, q){
      const c=chords[i]; c.root=root; c.q=q; c.confirmed=true; c.c=1;
      if(pending.indexOf(i)<0) pending.push(i);
      const old=cells[i]; const nw=renderCell(i); grid.replaceChild(nw, old); cells[i]=nw;
      nw.style.animation="ri-halo 1s ease-out";
      updateActionBar();
    }

    // ── form ribbon: two-tap to merge two sections ──
    let selectedSecs=[];
    const ribbon=el("div",`flex:0 0 auto;display:flex;align-items:center;gap:10px;margin-top:12px;`);
    ribbon.appendChild(el("span",`font:600 9.5px ${UI};letter-spacing:.07em;text-transform:uppercase;color:${T.faint};flex:0 0 auto;`,"Form"));
    const chips=el("div","display:flex;gap:7px;flex-wrap:wrap;flex:1;");
    ribbon.appendChild(chips);
    const mergeHint=el("span",`font:italic 11px ${SERIF};color:${T.faint};flex:0 0 auto;`,"tap two to merge");
    ribbon.appendChild(mergeHint);
    stack.appendChild(ribbon);
    function renderChips(){
      chips.innerHTML="";
      SECTIONS.forEach(s=>{
        const on=selectedSecs.indexOf(s.id)>=0;
        const chip=el("button",`display:flex;align-items:baseline;gap:1px;border:1.5px solid ${on?T.accent:T.line};background:${on?T.accent:T.card};color:${on?T.paper:T.ink};border-radius:9px;padding:7px 13px;font:700 13px ${UI};cursor:pointer;-webkit-tap-highlight-color:transparent;transition:all .15s;`);
        chip.appendChild(el("span","",s.label));
        if(s.tag){ const sup=el("sup",`font-size:9px;font-weight:700;opacity:.85;`,s.tag); chip.appendChild(sup); }
        chip.onclick=()=>toggleSec(s.id);
        chips.appendChild(chip);
      });
      const n=selectedSecs.length;
      mergeHint.textContent = n===0?"tap two to merge":(n===1?"pick one more to merge":"\u00a0");
    }
    function toggleSec(id){
      if(reinfering) return;
      const k=selectedSecs.indexOf(id);
      if(k>=0){ selectedSecs.splice(k,1); }
      else { selectedSecs.push(id); if(selectedSecs.length>2) selectedSecs.shift(); }
      renderChips();
      if(selectedSecs.length===2) openMerge(); else closePop();
    }
    renderChips();

    // ── merge confirm sheet ──
    function openMerge(){
      closePop();
      const secs=selectedSecs.map(id=>SECTIONS.find(s=>s.id===id));
      const sameLetter=secs[0].label===secs[1].label;
      const back=el("div",`position:absolute;inset:0;z-index:30;background:rgba(28,24,20,.28);display:flex;align-items:flex-end;`);
      const sheet=el("div",`width:100%;background:${T.card};border-top-left-radius:20px;border-top-right-radius:20px;border-top:1px solid ${T.line};padding:16px 18px 20px;box-shadow:0 -12px 30px -14px rgba(50,35,20,.4);animation:ri-sheet .3s cubic-bezier(.32,.9,.35,1);`);
      sheet.appendChild(el("div",`width:36px;height:4px;border-radius:2px;background:${T.rule};margin:0 auto 14px;`));
      // two section badges with a join glyph between
      const pair=el("div","display:flex;align-items:center;justify-content:center;gap:14px;margin-bottom:12px;");
      secs.forEach((s,idx)=>{
        if(idx===1) pair.appendChild(el("span",`font:600 22px ${UI};color:${T.accent};`,"\u22c8"));
        const b=el("div",`display:flex;align-items:baseline;gap:1px;border:2px solid ${T.accent};border-radius:9px;padding:8px 15px;font:800 20px ${UI};color:${T.accent};background:${T.paper};`);
        b.appendChild(el("span","",s.label));
        if(s.tag) b.appendChild(el("sup","font-size:12px;",s.tag));
        pair.appendChild(b);
      });
      sheet.appendChild(pair);
      sheet.appendChild(el("div",`text-align:center;font:700 15px ${UI};color:${T.ink};margin-bottom:4px;`, sameLetter?("Merge the two "+secs[0].label+" sections"):("Merge "+secs[0].label+" and "+secs[1].label)));
      sheet.appendChild(el("div",`text-align:center;font:italic 12.5px ${SERIF};color:${T.faint};line-height:1.5;margin-bottom:16px;padding:0 6px;`, sameLetter?"They\u2019re the same music \u2014 the detector split them. Pooling their audio gives the decoder twice the evidence, so the shaky bars resolve.":"Their audio will be pooled into one shared reading and re-decoded together."));
      const go=el("button",`width:100%;border:none;background:${T.accent};color:${T.paper};border-radius:12px;padding:14px;font:600 14px ${UI};cursor:pointer;display:flex;align-items:center;justify-content:center;gap:9px;`);
      go.appendChild(el("span",`font:600 15px ${UI};`,"\u22c8"));
      go.appendChild(el("span","","Merge & re-infer"));
      go.onclick=runMerge;
      sheet.appendChild(go);
      const cancel=el("button",`width:100%;margin-top:10px;background:transparent;border:none;color:${T.faint};font:600 13px ${UI};padding:8px;cursor:pointer;`,"Cancel");
      cancel.onclick=()=>{ selectedSecs=[]; renderChips(); closePop(); };
      sheet.appendChild(cancel);
      back.appendChild(sheet);
      back.onclick=(e)=>{ if(e.target===back){ selectedSecs=[]; renderChips(); closePop(); } };
      host.appendChild(back); pop=back;
    }
    // ── merge: its OWN re-infer flow (pooled evidence) ──
    function runMerge(){
      if(reinfering) return;
      reinfering=true; closePop();
      const ov=el("div",`position:absolute;inset:0;z-index:40;background:rgba(247,243,233,.86);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;border-radius:inherit;animation:ri-in .2s both;`);
      ov.appendChild(el("div",`width:34px;height:34px;border:3px solid ${T.line};border-top-color:${T.accent};border-radius:50%;animation:ri-spin .8s linear infinite;`));
      ov.appendChild(el("div",`font:600 14px ${UI};color:${T.ink};`,"Pooling both sections\u2026"));
      ov.appendChild(el("div",`font:italic 12px ${SERIF};color:${T.faint};`,"one shared reading, re-decoded \u00b7 ~2s"));
      const req=buildRequest(chords, [], [selectedSecs.slice()]);
      host.appendChild(ov);
      setTimeout(async ()=>{ let resp; try{ resp=await reinfer(req); }catch(e){ resp={diff:[]}; } ov.remove(); applyResp(resp, "merge"); selectedSecs=[]; renderChips(); reinfering=false; }, 1900);
    }

    // ── re-infer action bar ──
    const action=el("div",`flex:0 0 auto;display:flex;align-items:center;justify-content:space-between;gap:12px;margin-top:12px;padding:12px 14px;background:${T.deep};border-radius:13px;color:${T.paper};`);
    const actionTx=el("div","");
    actionTx.appendChild(el("div",`font:600 13px ${UI};`,"Re-infer"));
    const actionSub=el("div",`font:500 11px ${UI};color:rgba(247,243,233,.65);margin-top:2px;`,"confirm a chord, then re-run with it locked");
    actionTx.appendChild(actionSub);
    action.appendChild(actionTx);
    const goBtn=el("button",`flex:0 0 auto;border:none;border-radius:11px;padding:12px 18px;font:600 14px ${UI};cursor:pointer;background:${T.rule};color:${T.deep};opacity:.6;pointer-events:none;display:flex;align-items:center;gap:8px;transition:opacity .2s;`,"Re-infer");
    action.appendChild(goBtn);
    stack.appendChild(action);
    host.appendChild(stack);
    function updateActionBar(){
      const n=pending.length;
      if(n>0){ goBtn.style.opacity="1"; goBtn.style.pointerEvents="auto"; goBtn.style.background=T.accent; goBtn.style.color=T.paper; goBtn.textContent="Re-infer with "+n+" fix"+(n>1?"es":""); actionSub.textContent="your "+n+" locked chord"+(n>1?"s":"")+" will re-decode the whole song"; }
      else { goBtn.style.opacity=".6"; goBtn.style.pointerEvents="none"; goBtn.style.background=T.rule; goBtn.style.color=T.deep; goBtn.textContent="Re-infer"; actionSub.textContent="confirm a chord, then re-run with it locked"; }
    }
    goBtn.onclick=runReinfer;

    // ── re-infer: spinner -> apply diff -> propagation payoff ──
    function runReinfer(){
      if(reinfering || !pending.length) return;
      reinfering=true; closePop();
      const ov=el("div",`position:absolute;inset:0;z-index:40;background:rgba(247,243,233,.86);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;border-radius:inherit;animation:ri-in .2s both;`);
      ov.appendChild(el("div",`width:34px;height:34px;border:3px solid ${T.line};border-top-color:${T.accent};border-radius:50%;animation:ri-spin .8s linear infinite;`));
      ov.appendChild(el("div",`font:600 14px ${UI};color:${T.ink};`,"Re-analysing with your fix\u2026"));
      ov.appendChild(el("div",`font:italic 12px ${SERIF};color:${T.faint};`,"re-decoding on cached audio \u00b7 ~2s"));
      const req=buildRequest(chords, pending, []);
      host.appendChild(ov);
      setTimeout(async ()=>{ let resp; try{ resp=await reinfer(req); }catch(e){ resp={diff:[]}; } ov.remove(); applyResp(resp); reinfering=false; }, 1900);
    }
    function applyResp(resp, mode){
      const diff=(resp&&resp.diff)||[];
      const changed=[];
      diff.forEach(d=>{
        const pl=parseLabel(d.new_label);
        let j=-1; for(let k=0;k<chords.length;k++){ if(!chords[k].confirmed && overlaps(chords[k].t0,chords[k].t1,d.start_s,d.end_s)){ j=k; break; } }
        if(j<0) return;
        chords[j].root=pl.root; chords[j].q=pl.q; chords[j].c=(d.new_confidence!=null?d.new_confidence:chords[j].c);
        changed.push({j, d});
      });
      paintGrid();
      changed.forEach(ch=>{ const cl=cells[ch.j]; if(cl) cl.style.animation="ri-halo 1.1s ease-out"; });
      showPropagation(changed, mode);
      pending=[]; updateActionBar();
    }
    function showPropagation(changed, mode){
      const n=changed.length; if(!n) return;
      const isMerge = mode==="merge";
      const banner=el("div",`position:absolute;left:12px;right:12px;bottom:12px;z-index:45;background:${T.card};border:1.5px solid ${T.green};border-radius:14px;padding:14px 16px;box-shadow:0 16px 40px -16px rgba(31,138,91,.5);animation:ri-in .35s both;`);
      const top=el("div","display:flex;align-items:center;gap:10px;margin-bottom:10px;");
      top.innerHTML=`<span style="width:26px;height:26px;border-radius:50%;background:${T.green};color:#fff;display:flex;align-items:center;justify-content:center;font:700 14px ${UI};">\u2713</span>`;
      top.appendChild(el("div",`font:700 14px ${UI};color:${T.ink};`, isMerge ? "Merged \u2014 both A sections now share one reading" : ("Your fix sharpened "+n+" nearby chord"+(n>1?"s":""))));
      banner.appendChild(top);
      changed.forEach(ch=>{
        const d=ch.d; const o=parseLabel(d.old_label), nw=parseLabel(d.new_label);
        const row=el("div","display:flex;align-items:center;gap:9px;padding:5px 0;");
        const from=glyph(o.root,o.q,17,"exact",confColor(d.old_confidence!=null?d.old_confidence:.3)); from.style.opacity=".7";
        row.appendChild(from);
        row.appendChild(el("span",`color:${T.faint};font:600 13px ${UI};`,"\u2192"));
        row.appendChild(glyph(nw.root,nw.q,19,"exact",T.ink));
        row.appendChild(el("span",`margin-left:auto;font:600 11px ${UI};color:${T.green};`, Math.round((d.old_confidence||0)*100)+"% \u2192 "+Math.round((d.new_confidence||0)*100)+"%"));
        banner.appendChild(row);
      });
      const done=el("button",`width:100%;margin-top:10px;border:none;background:${T.deep};color:${T.paper};border-radius:10px;padding:11px;font:600 13px ${UI};cursor:pointer;`,"Nice \u2014 keep going");
      done.onclick=()=>{ banner.remove(); };
      banner.appendChild(done);
      host.appendChild(banner);
    }

    updateActionBar();
  };
  return API;
})();

// auto-initialise on DOM ready
(function(){
  function mount(){ document.querySelectorAll("[data-ri]").forEach(function(h){ window.RI.build(h, h.getAttribute("data-ri")); }); }
  if(document.readyState==="loading") document.addEventListener("DOMContentLoaded", mount); else mount();
})();
