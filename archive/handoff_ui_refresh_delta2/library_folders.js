/* ─────────────────────────────────────────────────────────────────────────
   library_folders.js — l'accueil à deux univers + Mes charts en dossiers
   (design 5a).

   À COLLER dans l'IIFE de window.APP. Remplace renderLibrary() ; ajoute
   renderCharts(). Dépend de ui_kit.js.

   Louis, 2026-08-08 : « l'écran d'accueil devrait être + simpliste — on vire
   le training mode et section cleanup, et je veux qu'il y ait 2 univers :
   chercher une nouvelle chanson et consulter mes charts ; pour mes charts je
   peux créer des folders pour les trier ; les chansons sont nommées
   artiste / chanson. »

   Donc l'accueil ne liste plus rien. Il pose deux portes et le bouton
   Reprendre. La liste vit derrière la deuxième porte.
   ───────────────────────────────────────────────────────────────────────── */

  /* ── nommage artiste / chanson ───────────────────────────────────────────
     Le titre YouTube est sale (« (Official Video) », « HD », « feat. »). On
     le nettoie à l'affichage seulement — jamais dans le fichier, qui reste la
     clé. `artist` vient du serveur quand il l'a (voir la note backend en bas) ;
     sinon on tente la séparation sur le premier tiret. */
  const _NOISE=/\s*[\(\[](official|lyric|audio|video|hd|4k|remaster\w*|live|mv)[^\)\]]*[\)\]]/ig;
  function chartName(c){
    if(c.artist) return {artist:c.artist, title:c.title||""};
    const t=(c.title||"").replace(_NOISE,"").trim();
    const m=t.match(/^(.{2,40}?)\s+[-–—]\s+(.+)$/);
    return m ? {artist:m[1].trim(), title:m[2].trim()} : {artist:"", title:t};
  }
  /* « Artiste / Chanson » : artiste en gris, titre en encre. Une seule ligne,
     ellipse à droite — jamais deux lignes, la liste doit se scanner. */
  function nameEl(c, size){
    const n=chartName(c);
    const w=el("div",
      `font:600 ${size||16}px ${UI};color:${T.ink};`+
      `white-space:nowrap;overflow:hidden;text-overflow:ellipsis;`);
    if(n.artist){
      w.appendChild(el("span",`color:${T.faint};font-weight:500;`, n.artist));
      w.appendChild(el("span",`color:${T.faint};font-weight:500;`, " / "));
    }
    w.appendChild(el("span","", n.title));
    return w;
  }

  /* ── dossiers ────────────────────────────────────────────────────────────
     Un chart appartient à AU PLUS un dossier (c'est un classeur, pas des
     tags) : « ranger » doit vider « Hors dossier », et une chanson dans deux
     dossiers ne le vide jamais. Si tu veux des tags un jour, c'est une autre
     fonctionnalité, pas une extension de celle-ci.

     Stockage : sidecar serveur si l'endpoint existe, localStorage sinon — la
     liste des charts reste la source de vérité, les dossiers ne sont qu'un
     classement par-dessus. Un chart supprimé disparaît des dossiers tout seul
     parce qu'on filtre toujours sur S.library. */
  const FOLDERS_KEY="harmFolders";
  function loadFolders(){
    try{ return JSON.parse(localStorage.getItem(FOLDERS_KEY))||{order:[],of:{}}; }
    catch(e){ return {order:[],of:{}}; }
  }
  function saveFolders(f){
    try{ localStorage.setItem(FOLDERS_KEY, JSON.stringify(f)); }catch(e){}
    api.post("/api/folders", f).catch(()=>{});   // no-op tant que la route n'existe pas
  }
  function folderCount(name){
    const f=S.folders||loadFolders();
    return S.library.filter(c=>f.of[c.file]===name).length;
  }

  /* ── 1. ACCUEIL ─────────────────────────────────────────────────────────*/
  function renderLibrary(){
    NOTES=FLAT;
    S.folders=loadFolders();
    const w=screenWrap();

    const head=el("div",`flex:0 0 auto;padding:calc(14px + env(safe-area-inset-top)) 20px 22px;`);
    const logo=el("div",`font:italic 600 30px ${SERIF};color:${T.ink};`);
    logo.innerHTML='harmon<span style="color:'+T.accent+'">ia</span>';
    head.appendChild(logo);
    head.appendChild(el("div",`font:italic 14px ${SERIF};color:${T.faint};margin-top:3px;`,
      "l'harmonie de n'importe quelle chanson"));
    w.appendChild(head);

    const body=el("div",
      `flex:1;min-height:0;overflow-y:auto;padding:0 20px;display:flex;flex-direction:column;gap:14px;`);
    body.className="ap-scroll";

    /* porte 1 — chercher. Le champ est DANS la carte : on ne fait pas taper
       l'utilisateur sur un écran pour l'emmener sur un autre où il retape. */
    const seek=el("div",
      `background:${T.card};border:1.5px solid ${T.accent};border-radius:20px;padding:18px 18px 20px;cursor:pointer;`);
    seek.appendChild(el("div",
      `font:700 11px ${UI};letter-spacing:.08em;text-transform:uppercase;color:${T.accent};`,"Chercher"));
    seek.appendChild(el("div",`font:600 22px ${UI};margin-top:6px;`,"Une nouvelle chanson"));
    seek.appendChild(el("div",`font:italic 13.5px/1.5 ${SERIF};color:${T.faint};margin-top:5px;`,
      "un titre, un artiste"));
    const field=el("div",
      `margin-top:14px;height:52px;border-radius:14px;background:${T.paper};`+
      `border:1px solid ${T.line};display:flex;align-items:center;gap:10px;padding:0 16px;`);
    field.appendChild(el("span",`font:400 17px ${UI};color:${T.rule};`,"⌕"));
    field.appendChild(el("span",`font:500 15px ${UI};color:${T.rule};`,"Autumn Leaves"));
    seek.appendChild(field);
    seek.onclick=()=>go("search");
    body.appendChild(seek);

    /* porte 2 — mes charts */
    const mine=el("div",
      `background:${T.card};border:1px solid ${T.line};border-radius:20px;padding:18px;`+
      `display:flex;align-items:center;gap:14px;cursor:pointer;`);
    const mt=el("div","flex:1;min-width:0;");
    mt.appendChild(el("div",
      `font:700 11px ${UI};letter-spacing:.08em;text-transform:uppercase;color:${T.faint};`,"Consulter"));
    mt.appendChild(el("div",`font:600 22px ${UI};margin-top:6px;`,"Mes charts"));
    const nf=(S.folders.order||[]).length;
    mt.appendChild(el("div",`font:italic 13.5px/1.5 ${SERIF};color:${T.faint};margin-top:5px;`,
      S.library.length+" chanson"+(S.library.length>1?"s":"")+(nf?(" · "+nf+" dossier"+(nf>1?"s":"")):"")));
    mine.appendChild(mt);
    mine.appendChild(el("span",`flex:0 0 auto;font:400 26px ${UI};color:${T.rule};`,"›"));
    mine.onclick=()=>go("charts");
    body.appendChild(mine);

    /* reprendre — statistiquement le premier geste */
    const last=S.library[0];
    if(last){
      const cont=el("div",
        `background:${T.card};border:1px solid ${T.line};border-radius:18px;padding:14px 16px;`+
        `display:flex;align-items:center;gap:14px;cursor:pointer;`);
      const ct=el("div","flex:1;min-width:0;");
      ct.appendChild(el("div",
        `font:700 11px ${UI};letter-spacing:.08em;text-transform:uppercase;color:${T.accent};`,"Reprendre"));
      const nm=nameEl(last,16); nm.style.marginTop="3px"; ct.appendChild(nm);
      cont.appendChild(ct);
      cont.appendChild(el("span",`flex:0 0 auto;font:400 24px ${UI};color:${T.rule};`,"›"));
      cont.onclick=()=>openChart(last.file);
      body.appendChild(cont);
    }
    w.appendChild(body);

    const foot=el("div",
      `flex:0 0 auto;padding:16px 20px calc(18px + env(safe-area-inset-bottom));display:flex;gap:10px;`);
    foot.appendChild(kitButton("Enregistrer", ()=>go("record"), {height:52, grow:true}));
    foot.appendChild(kitButton("Jam",         ()=>go("jam"),    {height:52, grow:true}));
    w.appendChild(foot);
    return w;
  }

  /* ── 2. MES CHARTS ──────────────────────────────────────────────────────*/
  function renderCharts(){
    NOTES=FLAT;
    S.folders=S.folders||loadFolders();
    const w=screenWrap();
    const open=S.openFolder||null;          // null = racine

    const bar=el("div",
      `flex:0 0 auto;display:flex;align-items:center;gap:12px;`+
      `padding:calc(6px + env(safe-area-inset-top)) 14px 10px;`);
    bar.appendChild(kitIcon("‹", ()=>{ if(open){ S.openFolder=null; go("charts",{push:false}); } else go("library"); }, "Retour"));
    bar.appendChild(el("div",`flex:1;min-width:0;font:italic 600 20px ${SERIF};color:${T.ink};`,
      open||"Mes charts"));
    bar.appendChild(kitButton(S.libraryEdit?"OK":"Modifier",
      ()=>{ S.libraryEdit=!S.libraryEdit; go("charts",{push:false}); },
      {state:S.libraryEdit?"on":"off"}));
    w.appendChild(bar);

    const filt=el("div",`flex:0 0 auto;padding:0 14px 14px;`);
    const fb=el("div",
      `height:48px;border-radius:14px;background:${T.card};border:1px solid ${T.line};`+
      `display:flex;align-items:center;gap:10px;padding:0 14px;`);
    fb.appendChild(el("span",`font:400 16px ${UI};color:${T.rule};`,"⌕"));
    const fi=el("input",
      `flex:1;min-width:0;border:none;background:none;outline:none;font:500 14px ${UI};color:${T.ink};`);
    fi.placeholder="Filtrer mes charts"; fi.value=S.chartFilter||"";
    fi.oninput=()=>{ S.chartFilter=fi.value; paintChartList(); };
    fb.appendChild(fi); filt.appendChild(fb); w.appendChild(filt);

    const body=el("div",`flex:1;min-height:0;overflow-y:auto;padding:0 14px calc(20px + env(safe-area-inset-bottom));`);
    body.className="ap-scroll"; w.appendChild(body);
    S._chartBody=body;
    paintChartList();
    return w;
  }

  function paintChartList(){
    const body=S._chartBody; if(!body) return;
    clear(body);
    const F=S.folders, open=S.openFolder||null;
    const q=(S.chartFilter||"").trim().toLowerCase();
    const match=c=>{ if(!q) return true;
      const n=chartName(c); return (n.artist+" "+n.title).toLowerCase().indexOf(q)>=0; };

    function sectionHead(label, action, onAction){
      const h=el("div","display:flex;align-items:center;justify-content:space-between;margin:2px 4px 9px;");
      h.appendChild(el("span",
        `font:700 11px ${UI};letter-spacing:.08em;text-transform:uppercase;color:${T.faint};`,label));
      if(action){
        const b=el("button",
          `border:none;background:none;padding:12px 4px;margin:-12px -4px;`+
          `font:600 13px ${UI};color:${T.accent};cursor:pointer;`, action);
        b.onclick=onAction; h.appendChild(b);
      }
      return h;
    }

    if(!open){
      body.appendChild(sectionHead("Dossiers","Nouveau",()=>promptNewFolder()));
      const list=el("div","display:flex;flex-direction:column;gap:8px;");
      (F.order||[]).forEach(name=>{
        const row=el("div",
          `height:60px;border-radius:14px;background:${T.card};border:1px solid ${T.line};`+
          `display:flex;align-items:center;gap:13px;padding:0 14px;cursor:pointer;`);
        row.appendChild(folderMark(name===(F.order||[])[0]));
        const tx=el("div","flex:1;min-width:0;");
        tx.appendChild(el("div",`font:600 16px ${UI};color:${T.ink};`,name));
        const n=folderCount(name);
        tx.appendChild(el("div",`font:500 12.5px ${UI};color:${T.faint};margin-top:2px;`,
          n+" chanson"+(n>1?"s":"")));
        row.appendChild(tx);
        row.appendChild(el("span",`flex:0 0 auto;font:400 22px ${UI};color:${T.rule};`,"›"));
        row.onclick=()=>{ S.openFolder=name; go("charts",{push:false}); };
        list.appendChild(row);
      });
      if(!(F.order||[]).length) list.appendChild(el("div",
        `font:italic 13px/1.6 ${SERIF};color:${T.faint};padding:0 4px 4px;`,
        "aucun dossier — « Nouveau » pour ranger tes charts"));
      body.appendChild(list);
      body.appendChild(sectionHead("Hors dossier"));
      body.appendChild(chartRows(S.library.filter(c=>!F.of[c.file] && match(c))));
    } else {
      body.appendChild(chartRows(S.library.filter(c=>F.of[c.file]===open && match(c))));
    }
  }

  /* la petite forme de dossier — un rectangle et son onglet, pas une icône
     importée : deux divs, ça tient dans le système */
  function folderMark(accent){
    const col=accent?T.accent:T.rule;
    const d=el("div",
      `flex:0 0 auto;width:34px;height:28px;border:1.5px solid ${col};`+
      `border-radius:4px 6px 6px 6px;position:relative;`+
      `background:${accent?"rgba(138,43,43,.07)":T.paper};`);
    d.appendChild(el("div",
      `position:absolute;top:-4.5px;left:-1.5px;width:15px;height:5px;`+
      `background:${col};border-radius:2px 2px 0 0;`));
    return d;
  }

  function chartRows(items){
    const list=el("div","display:flex;flex-direction:column;gap:8px;");
    items.forEach(c=>{
      const row=el("div",
        `min-height:64px;border-radius:14px;background:${T.card};border:1px solid ${T.line};`+
        `display:flex;align-items:center;gap:12px;padding:10px 14px;cursor:pointer;`);
      const tx=el("div","flex:1;min-width:0;");
      tx.appendChild(nameEl(c,16));
      tx.appendChild(el("div",`font:500 12.5px ${UI};color:${T.faint};margin-top:3px;`,
        [c.key, STATUS_LABEL[c.status]||""].filter(Boolean).join(" · ")));
      row.appendChild(tx);
      if(S.libraryEdit){
        row.appendChild(kitButton("Ranger", e=>{ e.stopPropagation(); openMoveSheet(c); }, {small:true}));
      } else {
        row.appendChild(el("span",`flex:0 0 auto;font:400 22px ${UI};color:${T.rule};`,"›"));
      }
      row.onclick=()=>openChart(c.file);
      list.appendChild(row);
    });
    return list;
  }

  /* « Ranger » : une feuille avec les dossiers + Nouveau + Retirer. Un seul
     dossier possible, donc c'est un choix, pas des cases à cocher. */
  function openMoveSheet(c){
    const back=overlay(), sheet=sheetBox(); sheet.appendChild(handle());
    sheet.appendChild(el("div",
      `font:600 11px ${UI};letter-spacing:.06em;text-transform:uppercase;color:${T.faint};margin-bottom:6px;`,
      "Ranger dans"));
    sheet.appendChild(nameEl(c,17));
    const list=el("div","display:flex;flex-direction:column;gap:8px;margin-top:16px;");
    (S.folders.order||[]).forEach(name=>{
      const on=S.folders.of[c.file]===name;
      list.appendChild(kitButton(name, ()=>{
        S.folders.of[c.file]=name; saveFolders(S.folders);
        closeOverlay(back); paintChartList();
      }, {height:52, grow:true, state:on?"on":"off"}));
    });
    list.appendChild(kitButton("Nouveau dossier…", ()=>{
      closeOverlay(back); promptNewFolder(c);
    }, {height:52, grow:true}));
    if(S.folders.of[c.file]) list.appendChild(kitButton("Retirer du dossier", ()=>{
      delete S.folders.of[c.file]; saveFolders(S.folders);
      closeOverlay(back); paintChartList();
    }, {height:52, grow:true}));
    sheet.appendChild(list);
    back.appendChild(sheet); root.appendChild(back);
  }

  /* Saisie du nom : champ dans une feuille, pas un prompt() — le prompt de
     l'OS casse l'identité papier/bordeaux, même raison qu'au confirm() de
     suppression (voir le commentaire de 2026-07-20 dans app_shell.html). */
  function promptNewFolder(alsoMove){
    const back=overlay(), sheet=sheetBox(); sheet.appendChild(handle());
    sheet.appendChild(el("div",`font:600 17px ${UI};color:${T.ink};margin-bottom:12px;`,"Nouveau dossier"));
    const inp=el("input",
      `width:100%;height:52px;border-radius:14px;border:1px solid ${T.line};background:${T.paper};`+
      `padding:0 14px;font:500 16px ${UI};color:${T.ink};outline:none;box-sizing:border-box;`);
    inp.placeholder="Standards";
    sheet.appendChild(inp);
    const row=el("div","display:flex;gap:10px;margin-top:14px;");
    row.appendChild(kitButton("Annuler", ()=>closeOverlay(back), {height:52, grow:true}));
    row.appendChild(kitButton("Créer", ()=>{
      const name=(inp.value||"").trim(); if(!name) return;
      S.folders.order=S.folders.order||[];
      if(S.folders.order.indexOf(name)<0) S.folders.order.push(name);
      if(alsoMove) S.folders.of[alsoMove.file]=name;
      saveFolders(S.folders); closeOverlay(back); paintChartList();
    }, {height:52, grow:true, state:"on"}));
    sheet.appendChild(row);
    back.appendChild(sheet); root.appendChild(back);
    setTimeout(()=>inp.focus(),80);
  }

  /* go() gagne deux écrans :
       else if(screen==="charts") elS=renderCharts();
       else if(screen==="search") elS=renderSearch();      // search_sources.js
     et `S.openFolder` se remet à null en quittant "charts". */

  /* ── note backend (petite) ───────────────────────────────────────────────
     1. /api/library gagne un champ `artist` par chart, quand il est connu.
        Aujourd'hui chartName() le devine sur le tiret du titre YouTube, ce qui
        marche pour « Artiste - Titre » et échoue sur le reste. Le champ doit
        être ÉDITABLE côté client (le titre YouTube ment souvent) — prévoir
        POST /api/chart-meta/<file> {artist, title}.
     2. Les dossiers vivent en localStorage tant que POST /api/folders n'existe
        pas ; l'appel est déjà là et échoue en silence. Forme du document :
            {"order": ["Standards","Blues"], "of": {"<file>": "Standards"}}
        Un seul dossier par chart — c'est un classeur, pas des tags. */
