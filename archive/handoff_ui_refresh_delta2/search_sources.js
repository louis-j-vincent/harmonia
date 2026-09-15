/* ─────────────────────────────────────────────────────────────────────────
   search_sources.js — l'écran de recherche à trois sources (design 6a).

   À COLLER dans l'IIFE de window.APP. Nouvel écran "search" ; retire la
   barre de recherche et le toggle YouTube/iReal de l'ancien renderLibrary().
   Dépend de ui_kit.js et de library_folders.js (chartName / nameEl).

   Une source à la fois, choisie en haut, YouTube par défaut. La ligne sous le
   sélecteur dit ce que la source IMPLIQUE, parce que les trois ne font pas du
   tout la même chose : YouTube écoute l'audio et déduit la grille (une minute
   d'attente, un écran de chargement) ; iReal et les tablatures importent une
   grille déjà écrite par un humain et s'ouvrent directement.
   ───────────────────────────────────────────────────────────────────────── */

  const SOURCES=[
    {id:"youtube", label:"YouTube",     swatch:()=>T.accent,
     note:"Harmonia écoute l'audio et déduit la grille"},
    {id:"ireal",   label:"iReal Pro",   swatch:()=>T.blue,
     note:"grille transcrite par un musicien · pas d'audio, pas d'analyse"},
    {id:"tabs",    label:"Tablatures",  swatch:()=>T.green,
     note:"accords relevés à la main sur les sites de tablatures"}
  ];

  function renderSearch(){
    NOTES=FLAT;
    const w=screenWrap();
    const src=S.searchMode||"youtube";

    const bar=el("div",
      `flex:0 0 auto;display:flex;align-items:center;gap:12px;`+
      `padding:calc(6px + env(safe-area-inset-top)) 14px 12px;`);
    bar.appendChild(kitIcon("‹", ()=>go("library"), "Retour"));

    const field=el("div",
      `flex:1;min-width:0;height:48px;border-radius:14px;background:${T.card};`+
      `border:1.5px solid ${T.accent};display:flex;align-items:center;gap:9px;padding:0 14px;`);
    field.appendChild(el("span",`font:400 16px ${UI};color:${T.faint};`,"⌕"));
    const inp=el("input",
      `flex:1;min-width:0;border:none;background:none;outline:none;`+
      `font:500 15px ${UI};color:${T.ink};`);
    inp.placeholder="Un titre, un artiste";
    inp.value=S.searchQuery||"";
    inp.setAttribute("enterkeyhint","search");
    inp.onkeydown=e=>{ if(e.key==="Enter") runSearch(inp.value); };
    field.appendChild(inp);
    if(S.searchQuery || (S.results||[]).length){
      const clr=kitIcon("×", ()=>{ S.searchQuery=""; S.results=[]; go("search",{push:false}); }, "Effacer");
      clr.style.width="32px"; clr.style.height="32px"; clr.style.border="none";
      clr.style.background="none"; clr.style.color=T.faint; clr.style.font="400 18px "+UI;
      /* exception assumée au 44 : le × vit DANS un champ de 48 px dont toute
         la surface est déjà tapable pour placer le curseur. Lui donner 44
         mangerait la moitié du champ. Cf. HANDOFF §acceptance. */
      field.appendChild(clr);
    }
    bar.appendChild(field);
    w.appendChild(bar);

    /* le sélecteur de source + ce qu'il implique */
    const pick=el("div",`flex:0 0 auto;padding:0 14px 12px;`);
    pick.appendChild(kitSegmented(
      SOURCES.map(s=>[s.id, s.label]),
      ()=>S.searchMode||"youtube",
      v=>{ S.searchMode=v; S.results=[];
           try{ localStorage.setItem("harmSearchMode", v); }catch(e){}
           go("search",{push:false});
           if(S.searchQuery) runSearch(S.searchQuery); },
      {height:SZ.control}));
    const S_=SOURCES.find(s=>s.id===src)||SOURCES[0];
    pick.appendChild(kitLegend(S_.swatch(), S_.note));
    w.appendChild(pick);

    const body=el("div",
      `flex:1;min-height:0;overflow-y:auto;padding:0 14px calc(20px + env(safe-area-inset-bottom));`);
    body.className="ap-scroll"; w.appendChild(body);
    S._searchBody=body;
    paintResults();
    setTimeout(()=>{ if(!S.searchQuery) inp.focus(); },120);
    return w;
  }

  async function runSearch(q){
    q=(q||"").trim(); if(!q) return;
    S.searchQuery=q; S.searching=true; S.searchPage=0; paintResults();
    const src=S.searchMode||"youtube";
    try{
      if(src==="ireal"){
        const r=await api.post("/api/irealb-search",{title:q});
        S.results=(r.results||[]).map(x=>({...x,_src:"ireal"}));
        S.searchMore=false;
      } else if(src==="tabs"){
        const r=await api.post("/api/tab-search",{q});
        S.results=(r.results||[]).map(x=>({...x,_src:"tabs"}));
        S.searchMore=false;
      } else {
        const r=await api.post("/api/yt-search",{q,page:0});
        S.results=(r.results||[]).map(x=>({...x,_src:"youtube"}));
        S.searchMore=!!r.hasMore;
      }
    }catch(e){ S.results=[]; }
    S.searching=false; paintResults();
  }

  function paintResults(){
    const body=S._searchBody; if(!body) return;
    clear(body);

    if(S.searching){
      const wrap=el("div","display:flex;align-items:center;justify-content:center;gap:10px;padding:40px 0;");
      wrap.appendChild(el("div",
        `width:18px;height:18px;border-radius:50%;border:2px solid ${T.line};`+
        `border-top-color:${T.accent};animation:ap-spin .8s linear infinite;`));
      wrap.appendChild(el("div",`font:500 14px ${UI};color:${T.faint};`,"…"));
      body.appendChild(wrap);
      return;
    }
    if(!(S.results||[]).length){
      if(S.searchQuery) body.appendChild(el("div",
        `font:italic 13.5px/1.7 ${SERIF};color:${T.faint};padding:28px 6px;text-align:center;`,
        "rien trouvé — essaie le nom de l'artiste"));
      return;
    }

    body.appendChild(el("div",
      `font:700 11px ${UI};letter-spacing:.08em;text-transform:uppercase;color:${T.faint};margin:6px 4px 10px;`,
      "Résultats"));
    const list=el("div","display:flex;flex-direction:column;gap:8px;");

    S.results.forEach(r=>{
      /* LA CARTE ENTIÈRE est la cible (≈362×67). La pastille colorée est une
         ÉTIQUETTE, pas un bouton : ne lui câble pas de onclick, elle fait 38 px
         et violerait la règle des 44. Même construction que renderLibrary(). */
      const card=el("div",
        `border-radius:14px;background:${T.card};border:1px solid ${T.line};`+
        `padding:12px 14px;display:flex;align-items:center;gap:12px;cursor:pointer;`);
      const tx=el("div","flex:1;min-width:0;");
      tx.appendChild(el("div",
        `font:600 15.5px ${UI};color:${T.ink};white-space:nowrap;overflow:hidden;text-overflow:ellipsis;`,
        r.title||""));
      tx.appendChild(el("div",`font:500 12.5px ${UI};color:${T.faint};margin-top:3px;`,
        r._src==="ireal" ? [r.composer,r.style,r.time_sig].filter(Boolean).join(" · ")
        : r._src==="tabs" ? [r.artist,r.rating?("★ "+r.rating):null,r.kind].filter(Boolean).join(" · ")
        : [r.channel, r.duration].filter(Boolean).join(" · ")));
      card.appendChild(tx);

      const here=!!r.local;
      const pill=el("div",
        `flex:0 0 auto;height:38px;padding:0 14px;border-radius:12px;`+
        `display:flex;align-items:center;font:600 13px ${UI};`+
        (here ? `border:1.5px solid ${T.green};color:${T.green};`
              : `background:${T.accent};color:#fff;`),
        here ? "Déjà là" : (r._src==="youtube" ? "Analyser" : "Importer"));
      pill.setAttribute("aria-hidden","true");
      card.appendChild(pill);

      card.onclick=()=>{
        if(r._src==="ireal")      importIrealChart(r);
        else if(r._src==="tabs")  importTabChart(r);
        else startAnalysis(r.local ? ("local:"+String(r.id).replace(/^local:/,""))
                                   : ("https://www.youtube.com/watch?v="+r.id), r.title);
      };
      list.appendChild(card);
    });
    body.appendChild(list);
  }

  /* ── note backend ────────────────────────────────────────────────────────
     La troisième source est nouvelle côté UI. Elle a besoin de :

         POST /api/tab-search   {q}      → {results:[{id,title,artist,kind,rating,url}]}
         POST /api/tab-import   {url}    → {url:"/chart/<file>"} | {error}

     `harmonia/tab_fetcher.py` + `tab_parser.py` + `tab_aligner.py` font déjà le
     travail : ces deux routes ne sont qu'une façade HTTP dessus. Tant qu'elles
     n'existent pas, laisser l'onglet Tablatures VISIBLE mais renvoyer une
     liste vide — l'état « rien trouvé » est déjà écrit, et masquer l'onglet
     coûterait un deuxième chemin de rendu à retirer plus tard.

     Un import (iReal ou tablature) n'a PAS d'écran de chargement : la grille
     arrive complète, on ouvre le chart. Seul YouTube passe par renderLoading().
   ───────────────────────────────────────────────────────────────────────── */
