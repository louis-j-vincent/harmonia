/* ─────────────────────────────────────────────────────────────────────────
   immersive_play.js — la lecture sans bandeaux (design 3a).

   À COLLER dans l'IIFE de window.APP, après chartDock(). Dépend de ui_kit.js
   et de chart_chrome.js.

   Louis, 2026-08-08 : « quand on joue le chart en mode normal, les options ne
   devraient pas apparaitre […] les bannières haut et bas disparaissent pour
   laisser + de place aux accords, mais toucher une section montre quand même
   les accords piano. »

   Donc : jouer = lire. Le chrome s'efface pendant la lecture, la grille prend
   tout, et toucher un accord fait monter une carte de voicing PAR-DESSUS la
   grille — sans arrêter la musique et sans ouvrir la grande feuille modale.

   Le shell a déjà `S.chartChrome` (masqué à l'ouverture, poignée + swipe pour
   le rappeler). Ce module en fait un état piloté par la LECTURE, et ajoute le
   seul morceau qui manque : la carte flottante.
   ───────────────────────────────────────────────────────────────────────── */

  const IMMERSIVE_IDLE_MS = 2600;   // délai avant que le chrome se retire

  /* Préférence : accords au piano pendant la lecture (Louis, 2026-08-08).
     Trois valeurs, rangée dans la feuille Aa avec les autres préférences de
     lecture, persistée comme elles :

       "never"  — rien ne monte, la grille reste seule
       "tap"    — la carte apparaît sur l'accord touché (DÉFAUT)
       "follow" — le clavier reste en bas et suit le playhead

     "follow" est exactement l'ancien S.coach : on le remplace par ce réglage
     plutôt que d'avoir deux commandes qui disent la même chose. Migration :
     un `harmCoach==="1"` existant devient "follow", sinon "tap". */
  const PIANO_MODE=(()=>{
    try{
      const v=localStorage.getItem("harmPiano");
      if(v==='never'||v==='tap'||v==='follow') return v;
      return localStorage.getItem("harmCoach")==="1" ? "follow" : "tap";
    }catch(e){ return "tap"; }
  })();
  S.piano=PIANO_MODE;

  /* Dans openPrefsSheet(), une ligne de plus, avec les autres :

       body.appendChild(prefRow("Accords au piano", kitSegmented(
         [["never","Jamais"],["tap","Au toucher"],["follow","Suivre"]],
         ()=>S.piano,
         v=>{ S.piano=v;
              try{ localStorage.setItem("harmPiano", v); }catch(e){}
              if(v!=="tap") closeVoicingCard();
              go("chart",{push:false}); })));

     Et le coach du bas d'écran ne se rend QUE si S.piano==="follow". */

  /* Le chrome ne se retire QUE si on est en lecture, en mode read, et qu'on ne
     touche à rien. En annotate on ne cache jamais : l'utilisateur y travaille,
     les outils doivent rester sous la main. */
  function immersiveEligible(){
    return S.screen==="chart" && S.playing && S.mode==="read" && !S._voicingCard;
  }
  function immersiveArm(){
    clearTimeout(S._immTimer);
    if(!immersiveEligible()) return;
    S._immTimer=setTimeout(()=>{
      if(!immersiveEligible()) return;
      if(!S.chartChrome) return;
      S.chartChrome=false;
      go("chart",{push:false});
    }, IMMERSIVE_IDLE_MS);
  }
  /* Toute intervention rappelle le chrome et relance le compte à rebours. */
  function immersiveWake(){
    if(S.screen!=="chart") return;
    if(!S.chartChrome){ S.chartChrome=true; go("chart",{push:false}); }
    immersiveArm();
  }

  /* À câbler :
       togglePlay()            → immersiveArm() après avoir mis S.playing
       stopPlayback()          → clearTimeout(S._immTimer); S.chartChrome=true
       attachChromeSwipe()     → immersiveWake() sur le swipe vers le bas
       buildChromeHandle()     → immersiveWake() au tap
     et dans renderChart(), le corps de la grille :
       body.addEventListener("pointerdown", e=>{
         if(!e.target.closest("[data-chord-cell]")) immersiveWake();
       });
     Un tap sur un ACCORD ne rappelle pas le chrome — il ouvre la carte. */

  /* La barre de progression fine qui remplace le dock en immersif : 3px en
     haut de l'écran, aucune interaction, juste « où j'en suis ». */
  function immersiveProgress(){
    const t=el("div",`flex:0 0 auto;height:3px;background:${T.line};`);
    const f=el("div",`width:0%;height:100%;background:${T.accent};`);
    t.appendChild(f); S._immFill=f;
    return t;
  }
  function paintImmersiveProgress(frac){
    if(S._immFill) S._immFill.style.width=Math.round((frac||0)*100)+"%";
  }

  /* ── la carte de voicing flottante ───────────────────────────────────────
     Remplace openChordSheet() PENDANT LA LECTURE seulement (à l'arrêt, la
     grande feuille garde tout son sens : c'est là qu'on édite). Elle flotte
     au-dessus de la grille, ne prend pas le focus, ne coupe pas l'audio, et
     se referme au prochain tap ailleurs. */
  function openVoicingCard(ch, anchorEl){
    if(S.piano==="never") return;      // réglage Aa : le piano ne monte jamais
    closeVoicingCard();
    const lv=chordDepth(ch);
    const midis=voicingFor(ch.root, ch.q, S.coachStyle||"close", ch.bass);

    const card=el("div",
      `position:absolute;left:12px;right:12px;bottom:calc(14px + env(safe-area-inset-bottom));`+
      `z-index:40;background:${T.card};border:1px solid ${T.line};border-radius:18px;`+
      `padding:14px 14px 12px;box-shadow:0 20px 44px -18px rgba(50,38,20,.55);`+
      `animation:ap-in .22s ease both;`);

    const head=el("div","display:flex;align-items:center;gap:10px;margin-bottom:12px;");
    head.appendChild(glyph(ch.root, ch.q, 30, lv, T.ink, ch.bass));
    const names=el("div",`flex:1;min-width:0;font:600 13px ${UI};color:${T.faint};`);
    head.appendChild(names);
    head.appendChild(kitIcon("×", closeVoicingCard, "Fermer"));
    card.appendChild(head);

    const body=el("div","display:flex;flex-direction:column;gap:8px;");
    card.appendChild(body);

    /* mêmes claviers que le prompter — une seule implémentation pour les deux
       écrans, voir prompter_keys.js (splitHands / renderHand) */
    const {left,right}=splitHands(midis);
    names.innerHTML=`<span style="color:${T.accent}">L `+left.map(m=>note(m%12)).join(" ")+
                    `</span> · <span style="color:#2a6fb0">R `+right.map(m=>note(m%12)).join(" ")+`</span>`;
    [["R",right,"chord"],["L",left,"bass"]].forEach(([tag,mids,role])=>{
      const r=el("div","display:flex;align-items:center;gap:9px;");
      r.appendChild(el("div",
        `flex:0 0 auto;width:22px;font:700 11px ${UI};letter-spacing:.08em;color:${T.faint};`,tag));
      const host=el("div","flex:1;");
      r.appendChild(host); body.appendChild(r);
      renderHand(host, mids, 58, role);
    });

    card.appendChild(el("div",
      `margin-top:10px;font:italic 12.5px ${SERIF};color:${T.faint};`,
      "la musique continue · touche ailleurs pour refermer"));

    /* tap sur la carte = entendre le voicing ; ne se referme pas */
    card.onclick=e=>{ if(e.target.tagName!=="BUTTON"){ e.stopPropagation(); playMidis(midis); } };

    S._screenEl.appendChild(card);
    S._voicingCard=card;
    if(anchorEl) anchorEl.setAttribute("data-voicing-open","1");
    S._voicingAnchor=anchorEl||null;

    /* un tap n'importe où ailleurs referme — capture, pour passer avant les
       handlers des cellules */
    S._voicingDismiss=e=>{ if(!card.contains(e.target)) closeVoicingCard(); };
    setTimeout(()=>document.addEventListener("pointerdown", S._voicingDismiss, true), 0);
    haptic();
  }

  function closeVoicingCard(){
    if(S._voicingDismiss){
      document.removeEventListener("pointerdown", S._voicingDismiss, true);
      S._voicingDismiss=null;
    }
    if(S._voicingAnchor){ S._voicingAnchor.removeAttribute("data-voicing-open"); S._voicingAnchor=null; }
    const c=S._voicingCard; S._voicingCard=null;
    if(!c) return;
    c.style.animation="ap-in .16s reverse both";
    setTimeout(()=>{ if(c.parentNode) c.parentNode.removeChild(c); }, 150);
    immersiveArm();
  }

  /* Dans buildIReal(), le handler de cellule devient :

       cell.el.onclick=()=>{
         if(S.mode==="read" && S.playing && S.piano==="tap") openVoicingCard(ch, cell.el);
         else openChordSheet(ch);            // inchangé
       };

     En "follow", le clavier vit déjà en bas et se met à jour tout seul : un tap
     sur un accord ouvre alors la grande feuille, comme à l'arrêt.

     La cellule ouverte se marque `data-voicing-open` — style-la comme la
     cellule du playhead mais en bleu (l'accent est déjà pris par la lecture) :

       box-shadow: inset 0 0 0 2px #2a6fb0; background: rgba(42,111,176,.10);
   */

  /* ── ce que renderChart() rend en immersif ───────────────────────────────
     Remplacer le bloc `if(S.chartChrome){ … }` par :

       if(S.chartChrome){
         w.appendChild(chartToolbar());
         if(S.mode==="analyse") w.appendChild(analyseLens());
         w.appendChild(buildFormRail() || el("div",""));
       } else {
         w.appendChild(immersiveProgress());
         w.appendChild(el("div","flex:0 0 auto;height:38px;"));   // marge haute
       }

     …et le dock à la fin :

       if(S.chartChrome) w.appendChild(chartDock());
       else w.appendChild(immersiveFade());

     La grille garde exactement la même largeur dans les deux états : seules
     les hauteurs changent, donc rien ne se recompose latéralement quand le
     chrome part ou revient. */
  function immersiveFade(){
    /* un dégradé de 64px en bas : les dernières mesures s'estompent au lieu de
       se couper net, et ça dit « ça continue » sans occuper de place */
    return el("div",
      `position:absolute;left:0;right:0;bottom:0;height:64px;pointer-events:none;`+
      `background:linear-gradient(to top, ${T.paper}f5, ${T.paper}00);`);
  }

  /* En immersif la grille peut respirer : mêmes cellules, plus hautes.
     Dans buildIReal(), la hauteur de cellule devient :
         const cellH = S.chartChrome ? 74 : 88;
     C'est le gain réel de l'opération — pas juste du vide en plus. */
