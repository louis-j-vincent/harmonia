# Harmonia — cleanup spec (2026-08-19)

Cible : `harmonia_min/app_shell.html` @ `f00a123` (seul fichier UI ; aucun changement backend).
Maquette : `Harmonia Cleanup.dc.html` (frames 1→5). Harnais de test : `mock/mock_live.html`
(= app_shell + un prélude qui mocke `/api/*` sur Autumn Leaves — le régénérer après chaque pull).

## Comment donner ceci à un agent

Une tâche = une section ci-dessous = un commit. **Ne pas** enchaîner les tâches dans une seule
passe : chaque section a ses critères d'acceptation et ils se vérifient à l'écran.
Le prompt à coller est littéralement :

> Applique la tâche **T<n>** de `handoff_cleanup/CLEANUP_SPEC.md` dans
> `harmonia_min/app_shell.html`. Ne touche à aucune autre fonction que celles nommées dans la
> tâche. Quand tu as fini : ouvre `mock/mock_live.html`, va sur l'écran concerné, et vérifie un par
> un les critères d'acceptation en les recopiant avec ✓/✗. Si un critère est ✗, corrige avant de
> répondre. Si une contrainte de la tâche est impossible sans toucher autre chose, ARRÊTE-TOI et
> dis-le au lieu d'improviser.

Invariants du projet, valables pour toutes les tâches :

- Tokens existants uniquement : `T.paper #f7f3e9`, `T.card #fffdf6`, `T.ink #1c1c1c`,
  `T.rule #b9b09a`, `T.faint #8a8371`, `T.accent #8a2b2b`, `T.line #e5dcc6`, `T.deep #2a2622`,
  vert `#1f8a5b`. Aucune nouvelle couleur, aucun dégradé.
- Glyphes d'accord = Georgia italique (`SERIF`), UI = `UI`. Pas d'emoji, pas d'icône dessinée.
- Toute cible tactile ≥ 44 px sur les deux axes. Quand le dessin fait moins de 44 px, on gagne la
  cible avec du padding + marges négatives (motif déjà utilisé : `immersiveHandle`, `meta` dans
  `chartToolbar`) — **jamais** en grossissant le dessin.
- Réutiliser `kitIcon` / `kitButton` / `kitSegmented` / `overlay()` / `sheetCol()`. Pas de
  `el("button", …)` ad hoc.
- Ne rien casser côté playhead : `setPlayhead`, `paintFormChip`, `paintDockTime` gardent leurs
  signatures et restent appelés aux mêmes endroits.

---

## T1 — La bande de forme devient une bande de 22 px (frame 1)

**Pourquoi** : c'est un aperçu de l'architecture du morceau + un playhead, pas une rangée de
boutons. Aujourd'hui `buildFormRail()` fait 44 px de haut, passe à deux lignes dès ~10 runs, et
mange le tiers de l'espace au-dessus de la grille.

**Où** : `buildFormRail()` (~l. 3168) et `paintFormRail()` (~l. 3216). `formRuns()` ne change pas.

Cas de charge à tenir (Louis, 2026-08-19) : `A×3 B A×3 B A×3 B bridge B A×3` = **17 passages**.
À 17 segments égaux sur 362 px utiles, plus rien n'est lisible ni tappable — d'où le regroupement
ci-dessous, qui est la partie NON négociable de la tâche.

À faire :

- Rangée unique, `display:flex; gap:2px; height:22px`, `padding:0 14px 12px`, **jamais**
  `flex-wrap` : quel que soit le nombre de runs, une ligne.
- **GROUPER les passages consécutifs identiques** (même `secId`) en UN segment :
  `A A A B A A A …` → groupes `A×3`, `B`, `A×3`, … Le regroupement se fait sur la sortie de
  `formRuns()`, en aval, sans la modifier.
- Largeur de chaque groupe **proportionnelle à sa durée totale** : `flex: <somme des t1-t0>`
  (fallback `flex: <nombre de passages>` quand `t0/t1` sont nuls — cas import iReal).
  `min-width:14px`.
- Dans un groupe de N passages, **N−1 traits fins internes** : enfant absolu
  `top:3px;bottom:3px;width:1px;background:#e2d9c3`, à `i*100/N %`. Ils disent la répétition sans
  écrire `×N`. Chaque passage garde sa propre zone de tap (le groupe porte N zones, pas une).
- Contenu : **une seule** étiquette par groupe, centrée. Lettre `A…H` → `font:700 11px UI`,
  `color:T.ink`. Section **nommée** (bridge, intro, outro, solo…) → sa **première lettre en
  minuscule**, `font:italic 700 11px SERIF`, `color:T.accent` : `bridge` → *b*. Jamais le nom
  complet (il ne rentre pas), jamais tronqué avec des points. Le nom complet reste dans
  `aria-label`, dans `title`, et dans l'éditeur de sections (T3).
- Fond `T.card`, bord `1px solid T.line`, `border-radius:5px`.
- **Le playhead vit dans la bande** : dans chaque segment, un enfant absolu
  `left:0;top:0;bottom:0;background:rgba(138,43,43,.14)` dont `width` = fraction jouée du segment
  (0 % avant, 100 % après, `(t-t0)/(t1-t0)` dedans). C'est `paintFormRail(t)` qui écrit ces
  largeurs — il reçoit déjà `t` en secondes.
- Segment courant : `borderColor:T.accent`, lettre en `800`. Aucun autre changement d'état.
- Cible tactile : le bouton garde `height:22px` visuellement mais
  `padding:11px 0;margin:-11px 0` → 44 px réels. `onclick` inchangé (seek + scroll).
- La bande est visible en Read, Analyse et Annotate, jamais deux fois sur la même page.

Acceptation (mock, Autumn Leaves) :
1. La bande occupe 22 px de haut, mesurés (`getBoundingClientRect().height`).
2. Sur une forme forgée à la main `A×3 B A×3 B A×3 B bridge B A×3` : une seule ligne, 9 groupes,
   aucun groupe < 14 px, la lettre de chaque groupe lisible, `b` en italique bordeaux.
   (Test à écrire dans le mock : remplacer `sections` du MODEL, pas dans le shell.)
3. En lecture, la teinte progresse **à l'intérieur** du segment courant, pas par sauts de segment.
4. Un tap n'importe où sur la bande déclenche le seek (tester à 3 px du bord haut).
5. `paintFormChip()` n'a plus d'effet visible mort : soit il est fusionné dans `paintFormRail`,
   soit il n'écrit plus de `background` concurrent.

---

## T2 — Annotate : un seul bouton d'outils, et il n'existe qu'ici (frames 2 et 3)

**Pourquoi** : le ruban actuel (`buildRibbon`) empile une légende dégradée + deux blocs titrés
« THE GRID » / « THE SECTIONS » avec des boutons pleine largeur, au-dessus de la grille. Trois
lignes de chrome pour deux actions rares. Et aucun outil de section ne doit apparaître en Read.

**Où** : `buildRibbon()` (~l. 3724), `chartToolbar()` (~l. 2745), `renderChart()` (~l. 3040).

À faire :

- Supprimer `buildRibbon()` et son appel dans `renderChart()`.
- Dans `chartToolbar()`, **si et seulement si `S.mode==="annotate"`**, ajouter à droite, avant
  `Aa`, un `kitIcon("···", openAnnotateTools, "Outils")` avec `border:1px solid T.accent` et
  `color:T.accent`. Dans les autres modes la barre est inchangée (aucun trou : `Aa` reste en bout).
- Nouvelle `openAnnotateTools()` : `overlay()` + `sheetCol()` + `handle()`, titre « Outils »,
  sous-titre italique Georgia « on répare la grille, pas la musique », puis trois lignes
  (`min-height:60px`, titre `600 14.5px UI` + note `500 12px T.faint`, chevron `›` à droite) :
  1. **Redessiner les sections** — « les lettres et leurs frontières, au doigt » → T3
  2. **Caler la mesure 1** — « le vrai départ du morceau » → `openBar1Sheet()` (existant)
  3. **Vérifier les fusions** — « à l'oreille, deux passages à la fois » → lien existant
     `/debug/section-merge-game`, **uniquement si la route répond** ; sinon la ligne n'est pas
     rendue du tout (pas de ligne grisée).
  Puis « Fermer ».
- Ce qui reste au-dessus de la grille en Annotate : **une** ligne, `font:italic 12.5px SERIF`,
  `color:T.faint`, `padding:0 16px 10px` : « touche un accord dont tu es sûr — il se verrouille ».
  La légende dégradée disparaît.
- `mergeMode` / `suggMode` / `sectSuggMode` restent dormants et non atteignables (aucun bouton).
  Ne pas supprimer leur code.

Acceptation :
1. En Read et en Analyse : aucun bouton d'outil, aucune ligne d'aide, la grille commence
   directement sous la bande de forme.
2. En Annotate : exactement un « ··· » dans la barre du haut, exactement une ligne d'italique.
3. La feuille s'ouvre, se referme au tap sur le fond, et chaque ligne fait ≥ 60 px de haut.
4. Rien dans la feuille n'ouvre un écran qui répond « no suggestions yet ».
5. Le bandeau Re-infer (`renderReinferAction`) est inchangé et reste au-dessus du dock.

---

## T3 — L'éditeur de sections en plein écran (frame 4)

**Pourquoi** : « modifier les sections au doigt » posé sur le chart est un objet étranger. C'est une
tâche à part entière : on la fait, on valide, on revient.

**Où** : nouvel écran `renderSectionEditor()` + entrée dans `go()`, atteint depuis T2 seulement.

À faire :

- Même bande de forme que T1, **agrandie à 74 px** : même étiquette courte qu'en T1 (lettre, ou
  initiale italique bordeaux pour une section nommée), `×N` en `500 9.5px T.faint` dessous.
  Un groupe nommé a `min-width:34px`. Le nom complet est écrit sous la bande
  (« *b* = bridge — touche la lettre pour renommer »), jamais dans le segment.
- Les poignées vivent **dans le gap** : `gap:13px` entre les groupes, poignée centrée dans le gap
  (dessin `3×40px` `T.accent`, zone de saisie `13×44px`), **et aucune poignée après le dernier
  groupe** — rien ne dépasse du bord droit de la bande. Drag → la frontière **colle au début de mesure** (`m.barGrid`) ; jamais entre
  deux mesures.
- Sous la bande, la carte « sous le doigt » : « B commence mesure 17 » + timecode, puis
  « écouter la jointure » (joue 1 mesure avant → 1 mesure après) et `−1` / `+1` mesure.
- Tap sur une lettre = renommer (A…H + intro/bridge/outro, la liste que `labels` autorise déjà).
  Rapprocher deux frontières jusqu'au contact = fusionner les deux sections.
- Sortie : `Garder cette forme` (52 px, `T.accent`) → POST des nouvelles frontières, puis retour
  au chart. `Annuler` en texte. **Aucun état de cet écran ne survit dans le chart.**
- Contrat serveur : si la route « redéfinir les sections » n'existe pas encore, l'écran est
  construit mais l'entrée de T2 n'est pas rendue. Écrire la demande dans
  `docs/handoff_mission3_ui_contract.md`, ne pas inventer d'endpoint et **ne pas** livrer un bouton
  qui ne persiste rien.

Acceptation :
1. Une frontière traînée s'arrête toujours sur une mesure ; l'aperçu suit le doigt sans saccade.
2. « écouter la jointure » joue et s'arrête seule, sans casser la lecture du chart au retour.
3. Après `Annuler`, la forme affichée sur le chart est exactement celle d'avant.
4. Aucun contrôle de section n'est visible sur le chart après retour.

---

## T4 — Les « ? » entrent dans la cellule (frame 2)

**Pourquoi** : aujourd'hui `top:-4px;right:-2px` → le point d'interrogation sort de la cellule,
chevauche la barre de mesure et la ligne du dessus.

**Où** : `buildIReal()`, la ligne `if(ch.c<.42) item.appendChild(el("span", … "?"))` (~l. 3552).

À faire :

- Remplacer le `?` par un **point de 5 px** (`border-radius:50%`, `background:confColor(ch.c)`),
  posé dans le flux sous le glyphe : la cellule devient `flex-direction:column;gap:3px`, la
  rangée du dessous fait `height:5px` (réservée même sans point → aucun saut de ligne de base).
- Un accord verrouillé (`ch.confirmed`) affiche dans cette même rangée un tiret `14×2px` vert
  `#1f8a5b` au lieu du `✓` actuel s'il déborde ; sinon garder le `✓` existant s'il est déjà
  contenu dans la cellule.
- Le point n'apparaît **qu'en Annotate**. En Read/Analyse, la couleur du texte suffit.

Acceptation :
1. Aucun élément de cellule ne dépasse de son rectangle (`overflow` vérifié sur les 4 coins).
2. Deux accords dans la même mesure : les points ne se touchent pas.
3. En Read, aucun point, aucun `?`.

---

## T5 — Practise : trois zones (frame 5)

**Pourquoi** : `renderPrompter()` empile stage + rangée speed + label + reel 96 px + mini-timeline
26 px + boutons + ajusteur de seam ; sur un petit écran tout se marche dessus (bug déjà vu : seam
visible mais intappable).

**Où** : `renderPrompter()` (~l. 4153) et son bloc `reelWrap`.

À faire, dans cet ordre de haut en bas et rien d'autre :

1. Barre du haut : titre + sous-titre `mesure N · 1×` (la vitesse s'y lit), plus un `···`
   (`openPractiseTools`).
2. Stage : accord courant 74 px, rôle sur une ligne (italique Georgia), les deux suivants à
   `opacity:.42`, `26px`. Centré, `flex:1`.
3. **Le clavier** (Louis, 2026-08-19 : « il faut aussi afficher les accords au piano joliment si
   besoin ») — carte `T.card`, bord `T.line`, `border-radius:12px`, hauteur 74 px, `padding:6px` :
   - fenêtre FIXE pour tout le morceau (déjà calculée dans `renderPrompter` : l'union des mains),
     jamais de recadrage entre deux accords ;
   - touches blanches en flex `gap:1px`, noires en absolu à `(i+1)*100/N − larg/2`, largeur
     `0.62×` une blanche, hauteur `62 %` — c'est `renderHand()` qui dessine, **le réutiliser** ;
   - note allumée : la touche blanche porte une **bande basse pleine** (`bottom:0;height:34%`) en
     `T.accent` (main gauche) ou `#2a6fb0` (main droite), plus un léger lavis sur toute la touche
     — le haut de la touche reste blanc, là où les noires se serrent. Noire allumée : la touche
     entière prend la couleur de la main. Noires à `0.55×` une blanche (à `0.62×` elles se touchent
     et les blanches allumées se lisent comme des taches).
   - au-dessus, UNE ligne : `AU PIANO` à gauche (`600 10px`, majuscules, `T.faint`), les noms de
     notes à droite (`G.` bordeaux puis `D.` bleu). Pas de titre de carte, pas de sélecteur de
     voicing, pas de légende.
   - se met à jour au changement d'accord **sans reconstruire l'écran** (écrire les `background`
     des touches, comme `paintFormRail` écrit les largeurs).
   - Réglage `S.piano==="never"` → la carte n'est pas rendue et les 74 px reviennent au stage.
4. Reel 96 px, inchangé fonctionnellement — mais **sans** le label « next four bars » (le repère
   `now` dans le reel le dit déjà).
5. Transport centré : `‹‹4`, play 64 px, `4››`, `loop` (52 px chacun sauf play).
6. La bande de forme de T1, **16 px**, tout en bas — elle remplace la mini-timeline `tl`.
   Même composant, même comportement (tap = seek).

Partent dans `openPractiseTools()` (une feuille, mêmes lignes que T2) : la rangée `speed`
(0.5× / 0.75× / 1× en `kitSegmented`), le calage de seam (`earlier` / `later`), le métronome, le
clavier (`S.piano`), le style de voicing.

Acceptation :
1. Sur un viewport 375×667, tous les éléments sont visibles sans scroll et chaque bouton reçoit
   son tap (tester les 4 boutons de transport + la bande). Si ça ne rentre pas, c'est le reel qui
   rétrécit (96 → 80 px), jamais le clavier ni le stage.
2. Le clavier allume exactement les notes du voicing courant, main gauche et main droite
   distinguées, et ne se recadre jamais pendant la lecture d'un morceau entier.
2. Changer la vitesse depuis la feuille met à jour le sous-titre de la barre du haut.
3. Le seam adjuster est atteignable quand une boucle est armée depuis le chart (A–B) comme depuis
   Practise.
4. Aucun scroll vertical dans l'écran Practise, jamais.

---

## Ce qui est supprimé (et doit avoir disparu du DOM)

- `buildFormRail` version 44 px + `flex-wrap` + `×N`.
- `buildRibbon` entier : légende dégradée, blocs titrés « THE GRID » / « THE SECTIONS ».
- Les `?` en position absolue négative.
- La rangée `speed`, le label « next four bars », la mini-timeline `tl` du prompter.
- Tout contrôle de section hors Annotate.

## Ce qui ne doit PAS bouger

`buildIReal()` (grille, tailles de glyphes, barres de mesure), `chartDock()`, `analyseLens()`,
`openEditor()`, `renderReinferAction()`, l'audio, `loopEngine*`, la persistance
(`/api/annotations`), le libellé des accords.

## Rendre compte

Réponse attendue de l'agent, pour chaque tâche : la liste des critères recopiée avec ✓/✗, les
fonctions touchées, et une capture de l'écran concerné à 375 px de large. Pas de résumé en prose.
