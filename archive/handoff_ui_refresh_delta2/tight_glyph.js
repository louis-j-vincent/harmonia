/* ─────────────────────────────────────────────────────────────────────────
   tight_glyph.js — la typographie serrée des barres à plusieurs accords
   (design 4a).

   À COLLER dans l'IIFE de window.APP, à la place de glyphCompact(). Dépend de
   ui_kit.js pour rien — mais de note(), famSuffix(), seventhOf(), TOK, CTOK,
   SERIF, el(), mod(), qui sont déjà là.

   Louis, 2026-08-08 : « même en écriture compacte, 2 accords par barre ça
   prend vite toute la place et 2 accords vont se chevaucher — iRealB solve ça
   avec une écriture très compacte horizontalement. »

   Ce que fait iReal (relevé sur la capture de « 9.20 Special », barre
   `C7 B7 B♭7 A7` dans une barre de ~90 px) :

     1. les accords sont EMPILÉS À GAUCHE avec un écart fixe — ils ne sont PAS
        répartis sur les temps. C'est de loin le plus gros gain de place, et
        c'est la seule raison pour laquelle quatre accords tiennent.
     2. la qualité est un INDICE tucké sous l'épaule droite de la lettre, pas
        un exposant posé à côté ; l'altération se range au-dessus, dans la même
        colonne étroite.
     3. la taille descend avec le nombre d'accords, mais reste grande — pas de
        micro-typo illisible.

   La règle 1 est un changement de LAYOUT (buildIReal), pas de glyphe. Les deux
   vont ensemble : le glyphe serré tout seul ne suffit pas.
   ───────────────────────────────────────────────────────────────────────── */

  /* Taille du glyphe selon le nombre d'accords dans la barre.
     19 px est un plancher dur : en dessous, on ne lit plus le chart à un mètre
     avec les mains occupées, et c'est la barre qu'il faut couper en deux
     (l'app le fait déjà — un accord par temps, quatre temps maximum). */
  const TIGHT_SIZE={1:30, 2:25, 3:22, 4:19};
  const TIGHT_GAP ={1:0,  2:9,  3:6,  4:4};
  function tightSize(n){ return TIGHT_SIZE[Math.min(4, Math.max(1, n))]; }
  function tightGap(n){ return TIGHT_GAP[Math.min(4, Math.max(1, n))]; }

  /* Le glyphe serré. Même signature que glyph()/glyphCompact().

     La colonne étroite fait toute la hauteur de la lettre, altération collée
     en haut, qualité posée sur la ligne de base (padding-bottom 19% : Georgia
     assied ses capitales à ~80% de la boîte, donc sans ce padding l'indice
     tombe SOUS la ligne de base et l'accord a l'air décroché).

     `margin-left:-1px` : l'italique de Georgia dépasse à droite. Ce dépassement
     est de la place gratuite — on s'en sert au lieu de le subir. C'est le même
     raisonnement que noteEl() applique déjà à ses altérations. */
  function glyphTight(root, q, size, depth, color, bass){
    const w=el("span",
      `display:inline-flex;align-items:stretch;height:${size}px;`+
      `font:italic 600 ${size}px ${SERIF};line-height:1;white-space:nowrap;color:${color};`);

    const nm=note(root), m=nm.match(/^([A-G])([♭♯]?)$/);
    w.appendChild(el("span","", m?m[1]:nm));
    const acc=m?m[2]:"";

    let suf;
    if(depth==="family"){ const f=famSuffix(q); suf=(f==="m")?"−":f; }
    else if(depth==="seventh"){ const s7=seventhOf(q); suf=CTOK[s7]!=null?CTOK[s7]:(TOK[s7]||""); }
    else suf=CTOK[q]!=null?CTOK[q]:(TOK[q]!=null?TOK[q]:q);

    const sz=Math.round(size*0.52);
    const col=el("span",
      `display:inline-flex;flex-direction:column;justify-content:space-between;`+
      `align-items:flex-start;margin-left:-1px;height:100%;box-sizing:border-box;`+
      `padding:3% 0 19%;line-height:.92;`);
    // le slot du haut est TOUJOURS réservé (♭ masqué quand la note est
    // naturelle) : sans ça la qualité remonte et deux accords voisins n'ont
    // plus la même assise
    col.appendChild(el("span",
      `font-size:${sz}px;font-weight:600;${acc?"":"visibility:hidden;"}`, acc||"♭"));
    if(suf) col.appendChild(el("span",`font-size:${sz}px;font-weight:700;`, suf));
    w.appendChild(col);

    if(bass!=null && bass>=0 && mod(bass,12)!==mod(root,12)){
      const bs=Math.round(size*0.46);
      const bw=el("span",
        `display:inline-flex;align-items:baseline;font-weight:600;opacity:.7;margin-left:1px;`);
      bw.appendChild(el("span",`font-size:${bs}px;`,"/"));
      bw.appendChild(noteEl(bass, bs));
      w.appendChild(bw);
    }
    return w;
  }

  /* glyph() route vers la version serrée quand la préférence « compact » est
     active. `n` = nombre d'accords dans la barre, passé par buildIReal ; sans
     lui, 1. */
  function glyph(root,q,size,depth,color,bass){
    if(PREF.notation==="compact") return glyphTight(root,q,size,depth,color,bass);
    /* … version normale inchangée … */
  }

  /* ── LE CHANGEMENT DE LAYOUT (buildIReal) ────────────────────────────────
     C'est ici que se gagne la place. Aujourd'hui la cellule est une grille de
     `bpb` colonnes et chaque accord se pose sur sa colonne de temps
     (`.q0….q3`), donc quatre accords se partagent la largeur ET gardent leurs
     gouttières — ils se chevauchent avant d'avoir la place de respirer.

     Remplacer par : une seule rangée flex, empilée à gauche, écart fixe.

         const n = bar.chords.length;
         const cell = el("div",
           `height:${cellH}px;display:flex;align-items:center;`+
           `gap:${tightGap(n)}px;padding-left:${n>=4?7:n>=2?9:12}px;`+
           `border-right:…;border-bottom:…;`);
         bar.chords.forEach(ch=>{
           cell.appendChild(glyphTight(ch.root, ch.q, tightSize(n),
                                       chordDepth(ch), chordColor(ch), ch.bass));
         });

     Ce qu'on perd, et pourquoi c'est acceptable : l'abscisse ne dit plus le
     temps. Sur un chart, personne ne lit le temps à la position horizontale —
     on le lit au nombre d'accords dans la barre (2 accords = 2 temps chacun,
     4 = un par temps). iReal fonctionne exactement comme ça depuis toujours,
     et c'est ce que Louis a montré en référence.

     L'EXCEPTION à garder : un accord qui n'entre pas sur un temps régulier
     (syncope, `beat` non entier) garde sa position proportionnelle — sinon le
     chart ment. Dans ce cas, poser l'accord avec `margin-left:auto` calculé
     sur `ch.beat / bpb` plutôt que dans le flux serré.

     Enfin : le mode « normal » (non compact) garde la grille de temps. Le
     serrage est le contrat de l'écriture compacte, pas un changement global.
   ───────────────────────────────────────────────────────────────────────── */
