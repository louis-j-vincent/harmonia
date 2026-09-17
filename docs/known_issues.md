# Harmonia — Known Issues

Journal court, à jour. Historique complet (2026-07 à 2026-09, avant le
refactor `harmonia_min` → `harmonia`) : `docs/archive/known_issues_2026-07_2026-09.md`.
Mécanique du projet (comment vérifier un changement, où sont les fichiers) :
`docs/STATE.md`.

## Ouvert

### La liste blanche de `chart.js` a mangé un troisième champ

`chart.js` reconstruit chaque accord dans `S.chords` en recopiant une LISTE
EN DUR de champs. Un champ qu'on n'y recopie pas n'arrive jamais dans
l'éditeur d'annotation, et rien ne se plaint. Trois fois maintenant : `sug`
(2026-08-17), `sugBass` (2026-09-17, cf. `tests/test_meta_accord_recollee.py`)
et `casc` (2026-09-17, le compas en cascade).

Le pendant PYTHON de ce piège a été renversé — `soudure.accords_par_mesure`
garde tout sauf ce qu'elle recalcule, et un test rougit pour n'importe quel
champ futur. Côté JS la liste est restée, parce que la renverser change un
chemin de rendu chaud sans test.

**Et elle cache déjà un défaut visible** : `chart.js:1388` lit `ch.carry`
pour afficher un accord tenu à `opacity:.72`. `carry` n'est PAS dans la liste
blanche, donc il vaut toujours `undefined` et cette règle de style ne
s'applique jamais. La corriger changerait l'aspect de tous les charts repliés
— c'est un arbitrage de Louis, pas une correction à faire en passant.

### Les queues altérées du compas en cascade n'ont pas de glyphe

`TOK` (`ui/kit.js`) traduit une queue d'accord en son écriture. Le compas en
cascade sait produire des queues altérées combinatoires (`-6b9`, `-6#9`,
`9#11`, `13b9`…) : elles ne sont pas dans `TOK`, donc `glyph` retombe sur la
chaîne brute et on lit `D-6b9` à côté d'un `Dm6/9`. Les queues SIMPLES ont été
ajoutées (2026-09-17) ; les combinaisons ne peuvent pas l'être une par une.
La sortie propre serait que `glyph` détache une altération finale et traduise
le tronc — une modification d'un chemin de rendu partagé, à faire seule et
vérifiée sur un chart entier, pas en marge d'une autre.


### RÉSOLU 2026-09-17 — le compas était illisible en thème sombre

Trouvé en portant la charte du compas sur une page de démo
(`tools/page_compas_da.py`), puis reproduit dans l'app.

`petalFill`, `petalEdge` et `keyTint` (`ui/kit.js`) n'ont **pas** de branche
sombre : leur clarté est calculée (`84 - 34c`, et 90 % pour le moyeu) et c'est
voulu — un pétale est un fond clair, c'est l'identité du compas. Mais ce qu'on
écrivait dessus prenait `T.ink`, qui bascule à `#f2ebde` en thème sombre. Donc
dès qu'on passait le thème, le compas de `Annotate` écrivait du crème sur du
pâle : contraste ~1,1:1, le symbole du moyeu et celui de chaque orbe
disparaissaient. Quatre endroits, tous dans `annotate.js` : le glyphe de
l'orbe, son pourcentage (`T.faint`), le glyphe du moyeu, et la pastille de
degré du Guide.

Le thème clair ne montrait rien, d'où un défaut qui a vécu longtemps.

**La loi maintenant** : `kit.js` exporte `INK_ON_PETAL` / `FAINT_ON_PETAL`,
figés sur l'encre du thème clair, et c'est ce qu'on écrit sur un pétale, quel
que soit le thème. Vérifié dans l'app à 390 px, thèmes clair et sombre
(`buildCompass` rendu avec les deux réglages, captures en session).


### Audit 2026-09-15 — ce qui reste ouvert

Détail et mesures : `docs/audit_2026-09-15_plan.md`.

- **RÉSOLU 2026-09-17 — un trait de section tracé à la main était arrondi sur
  une grille de bi-mesures, et un sur deux était jeté.** Louis : « relaxe les
  règles qu'on a mises sur les sections qui doivent commencer sur un début de
  4 barres. Du moment qu'un humain annote une section, il n'y a pas à le
  corriger, c'est LA vérité terrain, et c'est lui qui définit où commence la
  chanson. » `sections_inferer` calculait la grille de jetons SANS lui — des
  bi-mesures posées de deux en deux depuis la mesure 1 — puis y arrondissait
  ses traits, et jetait EN SILENCE tout trait dont le jeton de départ était
  déjà pris par le précédent. Mesuré sur ses 18 découpages annotés,
  191 traits : **19 débuts reculés d'une mesure, 23 traits purement perdus**.
  Sur Chain of Fools, 5 traits sur 10 disparaissaient et les 5 autres
  reculaient d'un cran. Les morceaux à phase paire n'en voyaient rien, d'où un
  bug qui paraissait capricieux : le décalage ne dépendait que de la parité de
  son trait. **C'est l'explication du symptôme déjà noté ici sans être
  attribué** — « un découpage parfait noté 0 %, décalé d'un cran » : la machine
  trouvait bien ses blocs de 8 mesures, c'est la quantification de SES traits
  qui les décalait.
  **La loi maintenant** : ses traits sont les BORNES de la grille
  (`soudure.jetons_sur_traits`) ; les bi-mesures se reposent à l'intérieur de
  chaque région, jamais à cheval sur une frontière qu'il a tracée. Le mot des
  jetons est recalculé sur cette grille (`soudure.mot_sur_traits`, mêmes deux
  sources qu'avant : ressemblance harmonique si l'audio est là, égalité des
  basses sinon). Après : 0 déplacé, 0 perdu sur les 191. Un trait qui ne peut
  vraiment pas être gardé (recouvrement réel) est RENDU dans `ecartes` et
  affiché à l'écran, plus jamais avalé.
  **Ce que ça ne résout pas** : (1) la recherche de coutures de
  `_meilleure_grille` — le jeton de 1 ou 3 mesures posé là où le morceau a une
  mesure en trop — ne tourne pas sur la grille des traits ; dans un TROU entre
  deux traits, une mesure surnuméraire décale encore la parité jusqu'au trait
  suivant, qui la rattrape. (2) La mesure 1 mal détectée (entrée ci-dessous)
  est un problème DISTINCT : elle déplace la grille de mesures elle-même, donc
  l'audio sous les traits. (3) L'outil Soudure (`/api/phrases4`) garde sa
  règle à lui — une soudure est faite pour être prolongée, c'est son objet ;
  seule l'annotation de sections est concernée ici. Ses fichiers
  d'annotation existants sont intacts (vérifié : brouillons et vérité
  identiques sur les 10 brouillons).

- **RÉSOLU (règle trouvée) 2026-09-17 — le vrai début d'un morceau, c'est sa
  PREMIÈRE NOTE DE BASSE, calée sur la grille.** Louis a tranché à l'oreille
  le début de 42 morceaux (`state/human/debuts.json`, vérité terrain tracée
  par git) : 18 où le traqueur avait déjà raison, 24 qu'il a déplacés. Mesuré
  contre ses 42 réponses, en comptant juste quand on désigne LA MÊME ligne de
  mesure que lui :

  | indice | exact | à ±1 mesure |
  |---|---|---|
  | le traqueur seul (mesure 1 = 1re ligne) | 28/42 | 37/42 |
  | la marche d'énergie à la ligne de mesure | 27/42 | 32/42 |
  | le 1er son audible du fichier | 31/42 | 37/42 |
  | le 1er accord non-N.C. de musx | 32/42 | 36/42 |
  | **la 1re note de basse** | **35/42** | **38/42** |

  Douze combinaisons essayées (consensus, médiane, garde-fous mutuels) :
  aucune ne dépasse la basse seule, les meilleures l'égalent. On garde la plus
  simple. La règle est `harmonia/debut.py` (`debut_du_morceau`), avec ses
  réglages mesurés et `tests/test_debut.py` ; `tools/debut_page.py` l'importe
  au lieu d'en garder une copie, et c'est elle que la page propose maintenant.
  **PAS ENCORE BRANCHÉE dans la pipeline** : ça déplacerait la mesure 1 sur
  toute la bibliothèque, donc ça passe par un rapport d'or que Louis arbitre.
  **Ce que ça ne résout pas**, et le mécanisme est musical : les 7 ratés se
  rangent en trois familles. (1) Le morceau ouvre sur la BATTERIE — Billie
  Jean, Be My Baby : la basse arrive une à deux mesures trop tard. C'est la
  piste que Louis a nommée en premier et qui reste à faire, un détecteur de
  transitoires aigus, indépendant de l'harmonie. (2) La basse joue déjà
  pendant l'intro (Urdlvw0SSEc, 9 mesures d'avance ; fd02pGJx0s0). (3) La
  grille elle-même est fausse (h_D3VFfhvs4, trous de 12 et 42 s dans
  `barGrid`) — caler sur la ligne la plus proche ne peut pas le rattraper.
  Enfin, 5 des 42 réponses de Louis ne tombent sur AUCUNE ligne (Stand By Me à
  deux tiers de mesure) : là c'est la phase du traqueur qui est en cause.

- **La page du vrai début jouait 1,5 s AVANT le curseur — les 12 clics libres
  de Louis du 2026-09-17 sont décalés d'autant (corrigé le jour même).** Il
  l'avait soupçonné : « j'espère que t'as pas mis de temps de latence quand on
  clique sur écouter ici, car sinon ça fausse tout ». C'était le cas : `jouer()`
  démarrait à `t0 - 1.5`. Mesuré sur ses 20 marques — 8 posées au pas de mesure
  (◀ ▶, donc pile sur une ligne) et 12 au clic libre : **les 12 tombent entre
  1,17 et 1,59 s après une ligne de mesure (médiane 1,41 s), et en retirant
  1,5 s, 10 sur 12 reviennent à moins de 0,25 s d'une ligne.** Ces 12 marques
  ont re-cuit 12 charts avec une mesure 1 EN PLEIN MILIEU d'une mesure ; les
  valeurs d'origine sont sauvegardées hors dépôt avant toute reprise.

- **Les trois pistes pour trouver le début automatiquement échouent, et on
  sait pourquoi (2026-09-17).** Pistes de Louis : le début de l'accompagnement,
  la première note de basse, et « regarder si musx ou Beat This chope déjà tout
  seul le bon début ». Mesuré sur ses 20 marques (vérité provisoire : le clic
  corrigé de 1,5 s, ramené sur la ligne la plus proche) :

  | indice | juste à la mesure près |
  |---|---|
  | le traqueur seul (mesure 1 = 1re ligne) | 9/20 |
  | marche d'énergie à la ligne de mesure | 8/20 |
  | 1re note de basse (tête basse de musx) | 9/20 |
  | 1er accord non-N.C. | 9/20 |
  | deux pistes d'accord entre elles | 9/18 |

  **Le mécanisme** : la basse et l'accord sortent tous deux de musx, sur le
  même audio, et le premier instant où musx cesse de dire « silence » est
  exactement ce sur quoi Beat This accroche sa première battue. Ils ne sont
  donc PAS indépendants du traqueur : là où le traqueur se trompe, ils se
  trompent pareil. Le seul indice vraiment indépendant testé — la marche
  d'énergie — est le plus mauvais, parce qu'il trouve le refrain le plus fort
  quand le morceau démarre tout de suite.
  **Ce que le mécanisme impose pour la suite** : il faut un indice indépendant
  de l'harmonie. Le candidat nommé par Louis et pas encore testé est la
  BATTERIE (« le grid donné par le temps qu'on chope quand la batterie se cale
  après ») : un détecteur de transitoires hautes fréquences ne partage rien
  avec musx. À faire sur une vérité propre — les 20 marques actuelles sont
  contaminées par les 1,5 s ci-dessus.

- **La mesure 1 tombe dans le fondu d'entrée quand le fichier contient
  l'intro du CLIP** (2026-09-16, Louis sur Sam Smith : « ça a mal détecté le
  début du morceau, et ça cause un décalage tout le long »). Mesuré sur
  `sam_smith_i_m_not_the_only_one_official_music_video` : le fichier dure
  4 min 40 et ses **40 premières secondes sont l'intro parlée du clip**, pas
  de la musique. Beat This! ne trouve sa première battue qu'à 37,1 s — il a
  raison — mais la grille démarre alors à 39,22 s, en plein fondu, alors que
  le groupe entre à 42,0 s (énergie lissée ×3) sur un downbeat du traqueur à
  42,14 s. La mesure 1 du chart est donc UNE mesure trop tôt, et tout le
  morceau est décalé derrière : l'outil de sections cale ses jetons sur la
  mesure 1 (`soudure._mesure1` → `grille_et_mot(depart=…)`), donc chaque trait
  au doigt tombe à côté. READY, qui a une marque « mesure 1 » posée à la main,
  ne souffre pas de ça — c'est toute la différence entre les deux.
  **Contournement, qui marche** : Outils → « Caler la mesure 1 » sur 42,14 s.

  **Candidate pour l'automatiser, mesurée le 2026-09-16, pas encore une
  règle** : `tools/debut_page.marche_energie`. La bonne question n'est pas
  « à quelle seconde » mais « à quelle LIGNE DE MESURE » — les lignes du
  traqueur sont justes, il ignore seulement laquelle est la première. Pour
  chaque ligne des 40 premières mesures, le rapport entre l'énergie des
  4 secondes qui suivent et des 4 secondes qui précèdent ; on retient le
  maximum. Sur les 45 morceaux de la bibliothèque : 33 gardent la mesure 1
  du traqueur, 12 la verraient bouger. Le rapport sépare nettement deux
  populations — Hot N Cold ×43, The Lazy Song ×27,8, Gbo7Qqlixt8 ×8,1,
  **Sam Smith ×4,0 à 42,14 s (exactement la valeur trouvée à la main)** d'un
  côté ; Stand By Me ×1,4, Autumn Leaves ×1,5, Chasing Pavements ×1,7,
  READY ×1,8 de l'autre, tous trouvés au milieu d'un refrain — c'est le
  groupe qu'une règle ne doit PAS suivre. Le seuil qui les sépare est
  précisément ce que la page va faire trancher à Louis.
  **Ce que ça ne résout pas** : une grille dont les lignes sont elles-mêmes
  mal placées (h_D3VFfhvs4 : trous de 12 et 42 s dans `barGrid`). Aucune
  ligne n'est alors bonne et le rapport ne le dit pas.

  **La page d'arbitrage** : `python -m tools.debut_page` →
  `/reports/debut_morceaux.html`. Énergie + lignes de mesure + la mesure 1
  actuelle + ma proposition ; Louis désigne une ligne du doigt ou au pas de
  mesure, écoute, et tranche (bonne / ailleurs / aucune ligne ne tombe juste
  / je ne sais pas). Les verdicts vivent dans `localStorage` et se copient en
  bloc ; **la page n'écrit jamais dans `state/human/marks/`**.
- **Le re-calage harmonique de la phase (`bars._phase_correction`) ne se
  déclenche sur aucun des 44** (seuils 0,55 / 0,15 jamais satisfaits
  ensemble). Sur 2 des 3 marques « Set bar 1 » de Louis, le vote des accords
  pointait sur sa phase à 45–50 %. Pas de correctif proposé : Let It Be
  (54 %) serait re-calé d'une demi-mesure — son oreille tranche.
- **Lost Without U (B, mes. 26) : la boucle repart avec une position de
  retard après une mesure de pause** (mes. 25, N.C. à 0,20). La grille est
  bonne ; la pile compte les positions depuis le début de la section, donc
  empile la 26 (D-7, vraie) avec les G7 des autres passes. Loi candidate :
  « une mesure de silence dans une boucle n'est pas une position » —
  contre-cas : un break qui fait partie de la boucle. Pas posée.
- **Les crochets de fin 1./2. notent du bruit dès que le tronc s'accorde** :
  les dernières mesures de chaque passe sont gardées hors pile exprès et
  décodées passe par passe ; `_ireal_endings` en fait autant de fins que
  de groupes (vu en lisant les fins sur la vue de façade : Cry Me A River
  5 fins / 8 passes ; et Lazy Song « 1. B G#m » remonte une mesure
  rejetée). Loi candidate « pas de crochets quand les queues se
  dispersent » — page dédiée à faire, pas posée.
- **Yesterday A : E- écrit, Eo entendu par Louis.** Consensus E- à 0,36
  contre Eo 0,14 — qualité d'accord du modèle, pas du repli. Annotation.
- `test_minimal_fold_separe_les_longueurs_dune_meme_lettre` échoue à HEAD
  depuis la cascade iReal (le B de 4 replié comme préfixe du B de 8).


- **RÈGLES D'OR de la basse — arbitrées à l'oreille, écrites, testées, PAS
  branchées** (2026-09-15). 33 arbitrages de Louis en deux tours d'écoute sur
  351 accords de 6 morceaux. Vérité terrain : `state/human/bass_verdicts.json`
  (ne pas régénérer). Code : `harmonia/bass_rules.py`. Garde-fou :
  `tests/test_bass_rules.py` rejoue les arbitrages — **une règle qui bouge doit
  faire rougir ce test**. Raisonnement complet, avec ce que chaque piste a
  coûté : `docs/bass_slash_rules.md`. En résumé : (1) la quinte est
  asymétrique — +7 décore, +5 porte et a la priorité ; (2) le retour de la
  basse de départ ne démonte pas une quarte (4 arbitrages sur 4) ; (3) le
  discriminant n'est aucune de nos mesures mais « le candidat est-il la
  fondamentale de l'accord » (16/17, contre 10/17 pour cinq curseurs) ; (4)
  ce qui jette une basse, c'est l'intervalle, jamais la confiance (un plancher
  de confiance se trompe 3 fois sur 4 ; une 9e à la basse est vraie, une b9
  jamais). Effet mesuré : 70 slashes → 28 (20 % → 8 % des accords), zéro
  intervalle impossible, zéro slash ajouté. **Ne résout pas** : 6 morceaux
  seulement ; rien dans `pipeline.py` ne l'appelle ; la basse mobile
  (16 accords sur 20 en ont une) reste écrite avec un seul symbole.
  **Révisions du soir même**, après arbitrage des deux cas restés ouverts :
  la règle (2) a perdu sa preuve propre (3 de ses 4 appuis sont déjà
  expliqués par la règle (3), le 4e était mal classé) — conservée mais
  marquée, et non utilisée par la procédure livrée. Et surtout, les deux cas
  ouverts n'étaient PAS des questions de basse : Ready mesure 3 est une
  **étiquette d'accord fausse** (Louis : `Ab^7`, pas `Eb`) — la règle de
  basse ne peut être juste que si l'accord l'est ; Ready mesure 49 est un
  **changement d'accord manqué** (Louis : `Bb-7` puis `Eb6b9`, un seul accord
  écrit sur 4,5 s à cheval sur deux mesures). Sur ce dernier, musx tient
  `Bb-` de 73 % à 95 % et ne voit jamais `Eb` (2,3 % max) : **c'est la
  lecture de basse qui pose la frontière que le détecteur d'accords rate**,
  exactement l'inverse de la règle (3). Piste suivante identifiée : se servir
  de la basse comme détecteur de frontière d'accord.
- **Basse détectée à l'attaque, pas en moyenne sur tout l'accord — 6/11 → 11/11
  sur un morceau, PAS ENCORE branché au pipeline live** (2026-09-15, piste
  ouverte par Louis : « dans le doute entre deux notes basses, si l'une est
  la quinte de l'autre, on prend la fondamentale »). Cette règle-là, testée
  telle quelle sur les 11 accords des mesures 1-8 de "Ready" (PJ Morton) :
  6/11 → 4/11, elle casse plus qu'elle ne répare — l'intervalle de quinte
  seul ne dit pas quel côté fait autorité (Ab^7 confirme l'intuition : Eb est
  bien un harmonique parasite d'Ab ; mais Bb-7, Eb7, F7 cassent pour la
  raison inverse, leur « quinte du dessus » est une vraie note jouée, une
  quarte ascendante). Le vrai signal : le **timing**, pas l'intervalle — ne
  lire que les 150 premières ms de chaque accord (l'attaque) au lieu de
  moyenner tout le span : 10/11, et le 11e (un "Db" lu "Eb") était en fait un
  Db/Eb — Eb était donc la bonne basse SONNANTE depuis le début, 11/11 réel.
  Mécanisme : la fondamentale domine à l'attaque ; sa propre quinte
  harmonique et les notes de passage voisines s'accumulent ensuite avec la
  résonance/pédale.
  Persisté : `harmonia.nnls_features.bass_pc_onset()` (+ `BASS_ONSET_S=0.15`),
  testé unitairement (`tests/test_bass_pc_onset.py`, données synthétiques).
  **Ce que ça NE résout PAS** : mesuré sur UN SEUL morceau (règle #5, CLAUDE.md)
  — pas de banc corpus encore. Pas câblé dans `infer` ni dans `c["bass"]` du
  chart live : `pool_beats()` (la version tout-le-span) reste inchangée et
  continue seule à nourrir les têtes entraînées (root/quality), qui ont été
  entraînées sur CE pooling précis — leur substituer l'attaque sans
  réentraînement dégraderait ces têtes en silence (règle #6). Prochaine étape
  avant tout branchement live : bench corpus (POP909 ou RWC, bass GT connue).
- **Corrections « Set bar 1 » / Annotate invisibles d'un checkout à l'autre**
  (2026-09-15, investigation demandée par Louis). Pas un bug du code de
  propagation lui-même — vérifié ligne par ligne (`api_bar1` → `jobs.start_job`
  → `_run_job` → `analyze_steps(bar1_time=...)`, et `tools/golden.py` pour le
  moteur `harmonia`) : tout est correctement câblé, et le chart servi
  (`state/cache/charts/min_191P7nIeECo.json`) porte bien la marque corrigée
  (1.355s) et la bonne grille. Le vrai problème : `state/human/marks/` et
  `state/human/annotations/` sont sous git mais PAS partagés entre checkouts —
  une correction faite sur le serveur d'un worktree (ici `:7773`,
  `.claude/worktrees/refactor`) reste invisible du serveur d'un AUTRE checkout
  (`:7772`, `feat/section-criteres`) tant qu'elle n'est pas commitée+mergée :
  vérifié, `state/human/marks/191P7nIeECo.json` et
  `state/human/annotations/min_191P7nIeECo.json` n'existent QUE dans le
  worktree refactor (`git status` les montre `??`, jamais ajoutés). Deux
  serveurs tournent en ce moment (`ps`: PID 44180 sur :7772, cwd = checkout
  principal ; PID 50047 sur :7773, cwd = worktree refactor, up depuis la
  veille ~21h — process de longue durée, code potentiellement pas rechargé
  depuis). Si Louis corrige un morceau sur un serveur puis regarde le résultat
  sur l'autre, ça ressemble exactement à « le recalage ne se transmet pas ».
  Pas de fix appliqué (délégué à l'agent dédié de Louis) — actions
  candidates : commit+push les fichiers `state/human/` non trackés avant de
  changer de checkout ; vérifier laquelle des deux instances (:7772/:7773)
  tourne du code à jour.
- **Test hérité rouge.** `tests/test_songformer_sections.py::test_minimal_fold_separe_les_longueurs_dune_meme_lettre`
  échoue depuis la fusion de la cascade iReal (commit `8146efd`, 2026-09-14) :
  la cascade a changé ce que ce test affirme sur le repli d'une même lettre à
  deux longueurs. Appartient à la piste qualité-des-sections (voir plus bas) ;
  pas un bug du refactor.
- **Recherche Ultimate Guitar : HTTP 500.** Le site renvoie un blocage
  anti-bot (même famille que le blocage YouTube ci-dessous) sur :7772 comme
  sur :7773 ; la route logge un WARNING et rend une liste vide. Fichier :
  `harmonia/integrations/tab_fetcher.py` (route : `harmonia/server/routes/irealb.py`).
- **Deux morceaux « froids » dans le rapport d'or.** `min_autumn_leaves` et
  `min_h_D3VFfhvs4` n'ont pas de cache SongFormer chaud — le rapport les
  saute plutôt que de relancer un modèle. Cache : `state/cache/songformer/`.
- **Deux clés toujours vides dans le rapport de repli** (`fold.demiton`,
  `fold.pos_skip`, `harmonia/folding.py`), gardées vides pour ne pas changer
  la forme du JSON du chart (sprint 7). À retirer avec une page avant/après
  quand le schéma du chart sera formalisé.
- **Trois scripts archivés lisent l'ancienne clé de cache musx.**
  `archive/scripts/page_concordance_barre.py`, `backfill_musx_sug.py`,
  `chart_retour.py` testent `musx_cache_path(audio).exists()` au lieu de
  `harmonia.cache.exists()` (seul à connaître le repli d'ancienne clé) — 20
  morceaux perdraient leurs sections en silence s'ils tournaient (sprint 15).
  Archivés, pas exécutés en prod ; à corriger seulement s'ils reviennent.
- **`data/cache/` garde ses anciennes clés.** Les caches beats/musx/nnls
  utilisent encore `<stem>` au lieu de `<stem>__<taille>` (`harmonia.cache`
  lit les deux, avec repli journalisé). Renommage en place prévu à la bascule
  prod (sprint 21 du refactor) — disque à 96 %, pas de copie possible avant.
- **SIGKILL SongFormer sur les longs morceaux, mitigé.** SongFormer peut
  saturer la mémoire sur un morceau long (mesuré : code retour 137 sur un
  morceau de 422 s, machine 16 Go). Depuis le sprint 9 du refactor, il tourne
  dans un sous-processus jetable (`harmonia/sections/songformer.py`) : sa
  mort fait échouer le job PROPREMENT (erreur remontée), au lieu de tuer le
  serveur. Vérifié dans le code (pas de `try/except` autour de l'appel
  enfant) ; pas re-testé sur un vrai morceau long depuis le portage.
- **`docs/harmonia_min_explainer.md` est désormais historique.** Il décrit
  `harmonia_min`, la référence du refactor, pas le paquet `harmonia/` actuel
  — utile pour le RAISONNEMENT (pourquoi SongFormer tourne en sous-processus,
  etc.), pas pour un chemin de fichier. Marqué en tête du fichier.
- **La recherche qualité-des-sections reprend après la bascule prod**
  (décision de Louis, `docs/refactor_2026-09/plan.md`). Le banc de recherche
  (`tools/sections_bench/`) n'importe PAS proprement — corrigé le 2026-09-16,
  cette ligne disait le contraire : une douzaine de ses scripts importent
  `harmonia_min.harmonic_sections`, `harmonia_min.voice_sections` et
  `harmonia_min.sections`, trois modules supprimés au sprint 9, bien avant la
  disparition de `harmonia_min` au sprint 22. Panne antérieure à ce sprint,
  jamais causée par lui. Et `bench.py --quick` ne
  tourne pas encore de bout en bout : `ssm_zoo.capture()` espionne
  `harmonia.sections.detect_sections`, dont la signature a changé au
  sprint 9 (elle ne reçoit plus les tableaux NNLS/musx pré-calculés que
  l'ancien dispatcher passait) — à réécrire pour lire ces features
  directement via `harmonia.nnls_features`/`harmonia.musx`.
- **`confirmed` protège l'AFFICHAGE, pas le calcul du repli** (trouvé
  2026-09-16 en vérifiant qu'un accord ajouté à la main survit à Souder).
  `library.chart_model` réapplique `annotations.overlay` à CHAQUE lecture, donc
  un accord verrouillé se réaffiche correctement même après un Souder qui a
  tout recalculé — vérifié : verrouillé un accord d'une mesure vide (l'outro
  de `min_Ju8Hr50Ckwk`), relancé Souder sur les mêmes sections (mesures
  réécrites), l'accord verrouillé était toujours là à la relecture. Mais rien
  n'alimente jamais `confirmed` dans les mesures BRUTES avant que
  `folding.fold_letter_groups`/`harmonia_min.folding` ne les empile — son
  garde-fou « `if any(c.get("confirmed") ...)` » (`harmonia/folding.py:587`,
  testé par `tests/test_fold_respecte_annotations.py`) ne se déclenche donc
  jamais en pratique : rien dans `pipeline.py`, `refold.py` ni
  `harmonia_min/soudure.py` ne lit le sidecar (`state/human/annotations/`)
  avant le repli. Conséquence concrète : une mesure verrouillée continue de
  fournir sa valeur D'AVANT verrouillage quand le repli MOYENNE les
  occurrences d'une même lettre — l'affichage de CETTE mesure reste juste
  (l'overlay la corrige après coup), mais elle peut fausser en silence la
  moyenne des mesures VOISINES de la même lettre. Pas de correctif proposé :
  il faudrait faire lire le sidecar à `accords_par_mesure`/aux étapes du
  pipeline avant le repli, une piste distincte. `harmonia/server/routes/
  annotate.py::fill_empty_bars` (2026-09-16) est un correctif plus étroit,
  sans rapport : il comble seulement le cas où `overlay` n'a même pas
  d'accord-hôte à modifier (une mesure entièrement tenue, `bars == []`).

## Trois trouvailles récupérées d'un arbre non commité, auditées le 2026-09-15

Trois tests non commités traînaient dans l'arbre principal (fichiers datés du
2026-07-21 au 2026-08-02, jamais commités ni jetés). Le code qu'ils testaient
a disparu avec le legacy (sprint 1) : les fichiers sont jetés, mais chaque
trouvaille est vérifiée contre le code vivant et notée ici.

- **Accord « add9 » mal lu comme dominant.** Un accord iReal comme « Aadd9 »
  (un accord majeur avec une neuvième ajoutée, sans septième) était compris
  comme un accord dominant, parce que le chiffre « 9 » de « add9 » suffisait
  à déclencher la règle. Vivait dans l'IMPORT iReal (`harmonia/irealb_fetcher.py`,
  supprimé) — l'import iReal n'existe plus dans l'app (seuls la recherche et
  l'export ont survécu, décision du refactor). **Rien à corriger aujourd'hui** ;
  si l'import revient un jour, ne pas chercher un chiffre dans le texte de
  l'accord, comparer le suffixe entier.
- **Un accord dominant non résolu devrait pencher vers le relatif mineur.**
  Sur une grille « Do maj7, Mi min7, Fa maj7, La7 », le La7 (qui contient un
  Do dièse) n'a de sens que comme le 5e degré de Ré mineur — mais sans accord
  de Ré qui suit pour le confirmer, l'ancien traceur de tonalité restait sur
  Do majeur quand même. Vivait dans `harmonia/theory/local_key.py` (supprimé,
  lisait le CHART, pas l'audio). **Pas transposable tel quel** : le traceur de
  tonalité vivant (`harmonia/harmonic_key.py`) lit le son réel (le chroma),
  pas les symboles d'accords, et accumule la preuve accord par accord
  (CUSUM) plutôt que de comparer des gammes. L'ancien bug ne peut plus
  arriver puisque son code est parti, mais personne n'a écouté si la même
  situation musicale (une dominante ambiguë, jamais résolue) trompe le
  nouveau traceur À SA manière. À vérifier à l'oreille quand la piste
  qualité-des-sections reprendra.
- **Un accord de substitution casse le comptage des répétitions.** Sur This
  Love, trois refrains identiques suivis d'un quatrième avec UN accord
  différent (une substitution dominante, très courante en vrai) étaient vus
  comme un seul bloc de 79 mesures au lieu de 3 répétitions + 1 variante,
  parce que la règle de groupage exigeait que CHAQUE accord corresponde
  exactement. Vivait dans `harmonia/output/chart_model.py` (supprimé, le
  rendu de l'ancienne app :7771). **Le groupage a changé de nature** : le
  repli vivant (`harmonia/folding.py`) groupe par l'ÉTIQUETTE que le
  détecteur de sections (SongFormer, qui écoute le son) a déjà posée, pas en
  comparant les accords — ce bug précis ne peut plus arriver là. Reste une
  question ouverte, différente : quand plusieurs passages de la même lettre
  sont empilés et moyennés, un passage avec une substitution est-il bien
  écarté comme exception (le contrôle de cohérence existant), ou fausse-t-il
  la moyenne en silence ? Pas vérifié ; même piste de recherche.

## Sprint 22 — fusionné (2026-09-16)

Fusionné dans `feat/section-criteres` (`3715466`). Vérifié indépendamment
après fusion, à part du merge lui-même : `harmonia_min/pipeline.py` avait
disparu, donc le rapport d'or a été relancé avec `--engine harmonia` contre
l'ancienne baseline, comparaison champ par champ (pas seulement « mesures
changées ») — **0 différence en dehors de `meta.engine`** sur les 44
morceaux communs. La baseline est re-gelée depuis le run PROPRE de l'agent
(`state/cache/golden/baseline_sprint22_2026-09-16/`, fait dans son worktree
isolé, donc jamais mélangé au travail de basse en cours dans l'arbre
partagé) — jamais depuis un run local, qui aurait mélangé les deux
chantiers. `pytest` : un test sentinelle ajouté le jour même par une autre
session (`test_jam_redecode.py::test_les_trois_appelants_existent_toujours`)
listait encore `harmonia_min/jam.py` en dur ; mis à jour vers
`harmonia/jam.py`, où le fichier vit maintenant — c'est exactement ce que ce
test est censé attraper. Suite complète : 303 verts, la seule rouge connue
inchangée.

## Ouvert — à arbitrer

- **Élargir le vocabulaire de musx ne servirait à rien — MESURÉ, impasse**
  (2026-09-17, Louis : « récupères TOUS les accords que musx sait faire, et
  élargis notre propre classification »). Fait dans l'ordre le moins cher :
  décoder UN morceau avec `chord_dict="full"` (382 entrées) au lieu de
  `submission` (26). Résultat : **exactement le même décodage** — mêmes 54
  segments, mêmes 6 types. Le vocabulaire n'est pas ce qui bride.
  Ce qui bride, ce sont les têtes d'extension du modèle. Sur **1255 accords
  décodés de 14 morceaux** : la 9e gagne **3 fois**, la 11e et la 13e
  **jamais**. Mécanisme : chaque extension multiplie le score par la
  probabilité de sa classe, et « aucune » domine (88-98 % en moyenne) ; un
  `maj9` est donc structurellement onze fois moins probable qu'un `maj7` et ne
  peut pas gagner le Viterbi. Les parts maximales atteintes existent pourtant
  (9e jusqu'à 54 %, 13e jusqu'à 43 %) : l'information est là, elle ne gagne
  jamais l'argmax.
  Donc : `full` ne changerait rien, et le modèle — à qui Louis fait 100 %
  confiance — répond « pas d'extension ». Piste si on veut ces couleurs un
  jour : ne pas les faire CONCOURIR mais les AFFICHER, comme l'accord optionnel
  « en petit au-dessus » déjà noté plus haut. Ce serait un affichage, pas un
  décodage, et ça n'engagerait rien.

## Résolu récemment

- **Audit : rien ne se perd entre musx et nos charts** (2026-09-17, demandé par
  Louis). Les 25 types du vocabulaire `submission` passent tous dans
  `labels._QUAL` — aucun absent. Une seule COLLISION : `sus4(b7)` et `11`
  deviennent tous deux `7sus4` (mapping assumé, « nearest iReal tail »), donc
  deux accords musx distincts deviennent indistinguables.
  Toutes les qualités que le décodeur émet réellement se retrouvent écrites
  dans la bibliothèque (`+`, `h7`, `sus4`, `o`, `9` compris). Les cinq que
  notre table sait produire sans qu'aucun chart ne les porte (`-9`, `13`,
  `^9`, `o7`, `sus2`) ne sont jamais émises par le décodeur sur le corpus —
  elles ne sont pas perdues, elles ne sont pas produites.
  Les RENVERSEMENTS que le décodeur écrit lui-même (`maj/5`, `min/5`, `maj/3`,
  `min/b3`, `maj/2`) sont écrasés par la règle de basse depuis le 2026-09-16 —
  mais les deux sont **d'accord 15 fois sur 15** sur un échantillon de six
  morceaux, donc rien n'est perdu en pratique. C'est aussi une validation
  indépendante de la tête basse : elle retombe sur la décision JOINTE du
  décodeur (triade+basse ensemble, lissée par Viterbi) à chaque fois.



- **Le compas ne pouvait proposer ni maj7 ni min7** (2026-09-17, Louis :
  « typiquement l'endroit ou j'ai marqué un Emaj7, ca ne le proposait jamais,
  c'est pas detecté par musx les major 7th ? »). Mesuré avant de répondre :
  `^7` apparaissait 177 fois dans les suggestions de la bibliothèque, et les
  177 fois l'accord ÉCRIT était déjà ce maj7 ; comme ALTERNATIVE, zéro. Idem
  `-7`, pourtant écrit 348 fois.
  Cause : l'espace de candidats est 12 racines × CINQ familles
  (`span_rescore.QUAL5`), et `_TRIAD_SEV_TO_QUAL5` y replie ("maj","maj7") sur
  "maj" et ("min","b7") sur "min" ; `Q5_TAIL` écrivait ensuite `""` et `"-"`.
  Seul l'accord déjà écrit gardait sa queue — d'où les 177/177.
  Correctif : `span_rescore.queue_du_candidat` raffine le NOM du candidat avec
  la tête de septième de musx (`probs[2]`, quatre colonnes), que
  `acoustic_logp_musx` lisait déjà pour SCORER. Le classement ne bouge pas —
  même espace, mêmes probabilités, même ordre. Conservateur et sans seuil
  nouveau : « maj » ne devient `^7` que si la septième la plus probable est la
  maj7, « min » ne devient `-7` que si c'est la b7 ; une « maj » avec une b7
  garde sa triade (cette combinaison a déjà sa famille, `dom`).
  Effet sur la bibliothèque : `-7` proposé 0 -> **434** fois, `^7` 0 -> **137**.
  Zéro accord écrit ne change (vérifié par diff).
  **Ce que ça ne règle pas** : `h7`, `7sus4`, `sus4`, `9`, `+` restent
  improposables — ils ne sont dans aucune des cinq familles, et musx n'a pas
  de tête qui les distingue comme il le fait pour la septième.



- **Verrou d'octave du tracker de battues : bouton ÷2/×2 dans l'écran Outils**
  (2026-09-17, Louis sur *Can't Take My Eyes Off You* de Frankie Valli : « le
  bpm est 2x too quick »). Mesuré : Beat This! lisait 125 BPM, avec des
  downbeats tous les 4 de ces temps (bpb=4, `direct`≈1.0, couverture≈1.0) — une
  grille **parfaitement saine aux yeux de `check_grid`**, juste deux fois trop
  rapide. C'est le piège documenté en tête de `beats.py` pour `librosa`
  (« doubled to ~129 BPM ») mais qu'aucun garde-fou algorithmique ne peut voir
  ici : la cohérence de grille est invariante au tempo absolu, il faut une
  référence externe (l'oreille de Louis) pour la détecter.
  **La correction n'est pas un simple `beats[::2]`** : sur ce morceau, ne
  garder qu'un temps sur deux en laissant les downbeats intacts fait tomber
  l'écart downbeat-à-downbeat de 4 à 2 temps dans la grille amincie — c'est
  très exactement la signature du verrou demi-tempo que `check_grid` refuse
  (test rouge ajouté : `test_halving_only_beats_would_have_read_as_a_half_tempo_lock`).
  `harmonia.beats.apply_tempo_octave` amincit **`beats` ET `downbeats`
  séparément** ; vérifié sur le vrai cache de ce morceau, la grille corrigée
  repasse `check_grid` avec metre=4, couverture 1.0, et le chart passe de 102
  à 51 mesures pour ~61 BPM (au lieu de 125).
  Persistance : `state/human/marks/<stem>.json` porte maintenant `tempo_factor`
  à côté de `bar1` (même sidecar, jamais un fichier par correction — les deux
  routes fusionnent au lieu d'écraser). La correction **s'accumule** : ÷2 posé
  deux fois vaut ÷4, et ÷2 puis ×2 annule proprement (`POST /api/tempo/<file>`).
  **Piège trouvé en vérifiant, pas en lisant le code** : `tools/golden.py`
  lisait déjà `bar1` du fichier de marque avant de recuire un chart, mais pas
  `tempo_factor` — un rebake/publish futur aurait donc **effacé la correction
  en silence**, exactement l'incident qui avait fait déplacer `bar1` vers ce
  fichier de marque au sprint 15. Corrigé dans le même commit
  (`tools/golden.py::cuire`), vérifié par comparaison avant/après isolée (git
  stash du code) : sans le correctif à `golden.py`, le rapport d'or recuisait
  *Can't Take My Eyes Off You* à 102 mesures malgré la marque posée ; avec, à
  51. Chart appliqué en direct sur `min_J36z7AnhvOM` : rapport
  `/reports/avant_apres.html`.
  **Ce que ça ne résout pas** : (1) la PHASE au moment d'amincir les
  downbeats (quelle moitié est la vraie mesure 1) est une ambiguïté musicale
  réelle — on ancre sur le premier downbeat existant, faute de mieux, même
  remainder non résolu que `score_periods` (voir règle #4 de CLAUDE.md) ; (2)
  ça ne DÉTECTE rien — c'est un correctif manuel, pas un nouveau garde-fou, et
  aucune tentative n'a été faite pour repérer ce cas automatiquement.

- **« Enregistrer comme vérité » repliait bien le chart, mais Read montrait le
  déplié** (2026-09-17, Louis : « quand je vais dans read j'ai le chart brut
  avec les sections colorées, pas bon du tout »).
  Le disque avait raison depuis le début : `/api/soudure/valider/<file>`
  re-empile les accords (`refold`) ET replie l'affichage
  (`soudure.sections_pour_chart`) — vérifié sur Ready, 8 occurrences à la main
  deviennent 3 sections avec reps 5/2/1. Le bug était à l'écran :
  `writeSectionsToChart` charge EXPRÈS le modèle DÉPLIÉ pour que l'outil de
  sections reste utilisable, et range le replié dans `S._foldedModel` ; seul
  `closeSectionTool()` le restituait. Passer en « Read » ne faisait que changer
  `S.mode` sans recharger — on relisait donc le déplié. Le geste naturel après
  avoir enregistré, c'est d'aller voir le résultat : le sélecteur de mode ferme
  maintenant l'outil, ce qui rend la vue repliée.
  Vérifié dans l'app vivante, rouge puis vert : sans le correctif 20 -> 60 ->
  60 cellules, avec 20 -> 60 -> 20.
  **Ce que ça ne règle pas** : il existe DEUX implémentations du repli
  d'affichage — `folding.minimal_fold` (pipeline) et
  `soudure.sections_pour_chart` (cette route). Elles peuvent diverger sans que
  rien ne le dise.



- **« Valider les sections » complète le reste avec les BRIQUES de Louis
  (2026-09-16).** Sa demande : « une fois qu'on a acté les premières sections
  au doigt et cliqué sur valider, les sections suivantes devraient
  automatiquement être complétées en cherchant le même pattern plusieurs fois
  dans la chanson via les matrices ssm », puis « on lui ajoute l'info de
  quelles sont les vraies briques des sections ».
  **Ce qui existait déjà, et ce qui manquait.** La validation appelait déjà
  `/api/sections/inferer` (algo des quatre mots) qui découpe TOUT le morceau et
  propage ses lettres — mais seulement sur une égalité EXACTE du mot : un
  refrain dont un seul jeton diffère repartait sous une lettre de machine. La
  SSM entre en SECOND RECOURS : un bloc non expliqué est comparé aux plages
  que Louis a tracées (`sections.similarity._diag`, la SSM chord-tone de la
  détection) et prend SA lettre si la ressemblance dépasse `RESSEMBLE_MIN`
  (0,75, choisi sur la courbe précision/couverture — voir la constante).
  Nouvelle source `ressemble`, distincte de `propage` : l'écran doit pouvoir
  dire d'où vient un nom.
  **La règle de Louis, vérifiée.** « Déjà une intro ne se rejoue pas plus
  tard » : intro et outro ne sont jamais proposées comme gabarit
  (`JAMAIS_REJOUEES`). Vérifié sur ses 20 découpages — « intro » est unique
  dans les 15 morceaux qui en ont une, « outro » dans les 8, zéro répétition.
  Sans cette règle, son intro d'UNE mesure sur Chain of Fools ressemblait à
  tout et raflait les dix blocs du morceau.
  **Mesuré** en simulant son geste (il marque la 1re occurrence de chaque
  lettre, puis valide), sur ses 16 découpages validés exploitables :

  | | accord lettre-par-mesure | paires groupées comme lui |
  |---|---|---|
  | quatre mots seuls | 37 % | 70 % |
  | + ressemblance SSM | **44 %** | **73 %** |

  **Deux choses apprises en mesurant, qui valent plus que les chiffres :**
  (a) un décalage d'UNE mesure met 0 % à un découpage parfait — sur Chain of
  Fools la machine trouve exactement ses blocs de 8 mesures, décalés d'un
  cran, et le premier mètre annonçait 0 % ; d'où la seconde colonne,
  invariante au nom et au décalage, qui dit la vraie qualité (73 %) ;
  (b) l'harmonie ne peut pas séparer deux sections qui vampent sur la même
  boucle — sur Chain of Fools la structure est juste à 92 % mais toutes les
  lettres sont fausses (ses A sont nommés B), et aucun réglage n'y changera
  rien : la voie mélodie, qui le pourrait, a été supprimée au refactor
  (`section_tool.substrates`).
  **Essayés et retirés** le même jour, mesure à l'appui : chercher les
  reprises du bloc sélectionné (`find_repeats`, ce que la demande décrivait
  littéralement) — 19 %, car au niveau de cellule qu'il choisit le même motif
  de 4 mesures se retrouve dans TOUTES les sections ; pénaliser les gabarits
  courts (répare Chain of Fools mais coûte Grenade 72 → 46) ; départager les
  ressemblances proches par la longueur (72 % contre 73 %).
  **Ce que ça ne résout pas** : (a) la vérité terrain est son propre
  découpage, qu'il dit imparfait — ces chiffres mesurent l'accord avec lui,
  pas la justesse musicale ; (b) deux de ses sections au contenu harmonique
  IDENTIQUE se confondent dans `par_contenu` (dict clé=contenu) : sur This
  Love son « intro » ressort en « A », bug préexistant à cette entrée ;
  (c) Bora Bora, The Walk et Sunny restent sous 60 % de paires groupées.
  Page à regarder : `python -m tools.sections_completion_page` →
  `/reports/sections_completion.html` (5 morceaux, chaque bloc écoutable).
  Code : `harmonia/server/routes/sections.py`.

- **La basse sonnante vient de la tête basse de musx, et c'est la règle partout**
  (2026-09-16, Louis : « ok tete musx meilleure partout », puis « republie,
  persiste, et switch pour que cette nouvelle basse musx soit la regle PARTOUT,
  et plus de bass rules »).
  **La règle entière** : la basse la plus probable selon la tête basse de musx,
  moyennée sur la durée de l'accord, écrite dès qu'elle diffère de la
  fondamentale. C'est tout.
  Trois choses ont été retirées en deux commits, chacune mesurée inutile ou
  nuisible AVANT de la retirer, jamais par goût de la simplicité :
  * la **fenêtre d'attaque** de 150 ms — décisive pour la chroma, nuisible pour
    musx (8/12 contre 11/12 sur tout l'accord). La chroma est de l'énergie
    brute où l'attaque isole la fondamentale avant la résonance ; la tête de
    musx est une sortie de modèle déjà lissée que 150 ms rendent bruitée ;
  * le **plancher de confiance** — la part de la fondamentale sépare les classes
    sans recouvrement (5,8-42,1 % avec slash, 80,5-97,2 % sans), mais dans les
    sept cas sans slash l'argmax est déjà la fondamentale : un plancher entre 40
    et 80 % donne exactement le même résultat que pas de plancher ;
  * le **filtre d'intervalle** — les quatre intervalles écartés à l'oreille
    l'avaient été sur des erreurs de LECTURE de la chroma, et aux quatre mêmes
    endroits musx donne la fondamentale à 80-97 % sans écrire de slash. Le
    filtre refusait au passage tous les 3es renversements de septième
    (`Am7/G`, `C7/Bb`, `Fmaj7/E`).
  `harmonia/bass_rules.py` et `tests/test_bass_rules.py` sont **supprimés**.
  La mémoire des 33 arbitrages reste dans `state/human/bass_verdicts.json`
  (suivi par git, ne pas régénérer) et `docs/bass_slash_rules.md` : c'est là
  qu'il faudra revenir pour écrire la règle de litige.
  Bibliothèque **republiée** (45 charts), sauvegarde dans
  `state/cache/charts.bak_20260916_2251_avant_basse_musx`, référence d'or
  re-gelée et vérifiée (45/45 identiques, 0 mesure changée), paires AVANT/APRÈS
  nettoyées. Garde-fou : `tests/test_basse_musx.py` (9 tests) fige la lecture
  des colonnes de la tête basse — la 0 est « pas de basse », les 1 à 12 sont
  les hauteurs à partir de DO, et se tromper d'origine décale d'un triton.
  **Ce que ça ne règle pas** : `Eb-/Gb` (Ready mes. 4), le seul cas que musx
  rate et que la chroma trouvait ; les 4 `Bb7/B` de Virtual Insanity (voir
  ci-dessus) ; et un accord dont la basse bouge reste écrit avec un seul
  symbole.



- **Ajouter un accord sur un temps TENU remplaçait l'accord existant
  (2026-09-16, Louis : « quand je clique pour créer un nouvel accord et que je
  valide, ça remplace l'ancien accord existant, donc pas sûr que ça fasse bien
  le delta prior au bon endroit »).** Les deux moitiés de sa remarque étaient
  justes, et la seconde découlait de la première.
  La vue « barre en grand » décidait avec `cidx != null` — « un accord SONNE
  sur ce temps » — au lieu de « un accord COMMENCE ici ». Sur une mesure tenue
  (un seul accord au temps 1, le cas le plus courant), les temps 2, 3 et 4
  passaient donc pour occupés et renvoyaient tous vers l'accord EXISTANT :
  verrouiller le remplaçait. Et comme aucun nouveau créneau n'était créé, le
  classement par delta (2026-09-16) ne tournait jamais — d'où son doute, fondé.
  Le test porte maintenant sur l'ATTAQUE (`onset`) : un temps que l'accord
  précédent se contente de tenir est « un endroit vide de la barre », les mots
  mêmes de la demande initiale. Point plein = un accord commence ici (on
  l'édite) ; anneau = rien ne commence (on en ajoute un), et le petit glyphe
  grisé montre ce qui tient, pour savoir sur quoi on ajoute.
  Le chemin serveur, lui, était déjà bon : `annotations.overlay` insère une
  correction dont la clé (mesure, temps) est neuve en clonant un accord-hôte
  de la même mesure — le mécanisme du « split », éprouvé depuis juillet.
  **Vérifié de bout en bout** sur Lost Without U mes. 18 (`G7` seul au temps
  1) : taper le temps 3 ouvre « new chord » avec ses candidats par delta
  (G 15 %, D-7 35 %), verrouiller G donne côté serveur `G7` temps 1 **et** `G`
  temps 3 coché — l'ancien intact.
  `harmonia/static/screens/annotate.js`.

- **Le compas : orbes plus grands et vraiment proportionnels, basses
  SÉLECTIONNABLES (2026-09-16).** Louis : « je veux que les cercles soient
  plus grands, ils devraient être proportionnels à leur proba de suggestion,
  et les basses sont sélectionnées à part, en cercles pareil ».
  **La taille.** La loi était déjà l'aire ∝ probabilité ; le problème était la
  BANDE — de 27 à 36 px de rayon, un candidat à 2 % faisait 75 % de la taille
  d'un candidat à 90 %, donc la proportion ne se voyait pas. Bande élargie
  (22 px, le seuil tactile, à Sz×0,15) et roue agrandie (plafond 286 → 318 px :
  il restait 92 px de marge inutile sur un téléphone de 390).
  **Ce que la géométrie impose, et l'arbitrage qui en découle.** Deux orbes à
  30° l'un de l'autre dans un anneau de 127 px ne peuvent pas être gros tous
  les deux sans se chevaucher — et ne jamais se chevaucher est la règle de
  Louis du 2026-08-08. La boucle de rétrécissement gardait les cinq candidats
  en les écrasant TOUS (jusqu'à 8 px de rayon, sous le seuil tactile).
  Désormais : on rétrécit un peu, et si le plancher tactile est franchi on
  RETIRE le candidat le moins probable et on recommence. Une ligne au-dessus
  de la roue dit combien sont sortis et renvoie au Guide, qui les montre tous.
  Mesuré sur Lost Without U (G7, 4 candidats sur des quintes consécutives) :
  2 orbes affichés, D-7 à 47 % visiblement plus gros que G à 11 %.
  **La basse.** Les anneaux se touchent maintenant, et ce que Louis touche
  devient SA basse : `Lock G7/D`. Re-taper annule ; taper la fondamentale
  elle-même annule aussi (ce n'est pas un slash, c'est la position
  fondamentale). Le serveur savait déjà recevoir une basse affirmée
  (`annotations.overlay` : « un bass explicite est une assertion ») — c'était
  le client qui envoyait `-1` en dur faute d'interface. Le drapeau
  `bassPicked` distingue « Louis l'affirme » de « le modèle l'a devinée » :
  sans lui, verrouiller un accord promouvrait en affirmation humaine la basse
  que le modèle avait inférée.
  **Ça ne contredit pas le banc corpus** qu'exige l'entrée « Basse détectée à
  l'attaque » : celui-ci porte sur la DÉCISION AUTOMATIQUE d'écrire un slash
  (`bass_rules`), pas sur le droit de Louis de poser la sienne à l'oreille.
  **Ce que ça ne résout pas** : (a) on ne peut pas RETIRER un slash que le
  modèle a écrit — il faudrait une troisième valeur côté serveur (« j'affirme
  la position fondamentale »), que la convention actuelle de `-1` (« pas
  d'avis ») ne distingue pas ; (b) la légende du bas (taille/couleur/angle)
  demande maintenant de faire défiler, la roue ayant grandi.
  `harmonia/static/screens/annotate.js`.

- **Les candidats d'un accord AJOUTÉ sont classés par le delta, pas par la
  probabilité brute (2026-09-16).** Louis : « dans l'option editing quand on
  clique pour créer un nouvel accord, dans les suggestions on prend celles du
  delta prior qu'on avait testé avant […] au lieu de les classer par leur
  probabilité sur ce créneau, on les classe par ce qu'ils ont gagné depuis le
  créneau précédent. L'idée étant que la résonance et la pédale font que
  l'accord d'avant continue de bien scorer — le delta l'annule, puisqu'il
  était déjà haut. »
  Un créneau qu'on vient de créer n'a pas de `sug` : il n'existait pas à la
  cuisson, personne n'a classé ses candidats, et l'éditeur s'ouvrait donc sur
  « By hand », sans rien à proposer. `span_rescore.delta_candidates` les
  calcule à la demande sur les postérieures musx en cache — MÊME pooling que
  `musx_suggestions` (`pool_span_musx` + `acoustic_logp_musx`, mêmes 60
  cases), SEUL le classement change. Route `POST /api/chord-candidates/<file>`
  (span courant + span précédent, que le client seul connaît puisqu'il vient
  de poser le brouillon) ; `openNewChord` la lit APRÈS avoir ouvert la
  feuille, qui se redessine et passe au compas quand la réponse arrive.
  L'en-tête dit quel classement on regarde (« what comes IN here — ranked by
  what each gains »), et `c` reste la probabilité DU CRÉNEAU, jamais le gain.
  **Reproduit les deux mesures de Louis au dixième près**, sur Ready (PJ
  Morton), avec la convention span de l'accord ÉCRIT (pas la mesure) :
  Ab^7 mes. 3, #3 à 6,7 % → **#2** ; Gb maj mes. 4, #6 à 6,9 % → **#4**.
  Effet concret sur le second : à 6e il était HORS du top 5 affiché, à 4e il
  devient visible dans le compas.
  **Un écart assumé par rapport à la description**, à confirmer par Louis : le
  plancher (`SUG_FLOOR`, 2 %) s'applique AVANT le classement. Un candidat qui
  passe de 0,1 % à 2 % a « gagné » plus qu'un vrai accord qui passe de 30 % à
  31 % et sortirait devant lui sur le delta seul. Sur les deux cas mesurés le
  plancher ne change rien (6,7 % et 6,9 %, très au-dessus).
  **CE QUE ÇA NE RÉSOUT PAS, et Louis le dit lui-même** : c'est une
  HYPOTHÈSE, pas un résultat. 2 accords, UN morceau (règle #5). **Aucun des
  deux n'atteint #1** — `Eb` reste en tête des deux classements, la pédale
  n'est pas complètement annulée. Ce qui manque pour trancher, c'est de la
  vérité terrain sur l'IDENTITÉ des accords : 13 accords annotés en tout sur
  5 morceaux, les GT de brick0 condamnées à l'oreille, pas de GuitarSet sur le
  disque. La piste ouverte est le banc par les TABLATURES (16 en cache, très
  notées, transposables) — il faut les aligner sur nos créneaux ; ce banc
  appartient à la session concurrente, qui l'a proposé le même jour.
  N'est branché QUE sur le chemin « nouvel accord », là où il n'y avait aucun
  classement du tout : il ne peut donc pas dégrader un classement existant.
  Code : `harmonia/span_rescore.py` (`delta_candidates`),
  `harmonia/server/routes/annotate.py`, `harmonia/static/screens/annotate.js`.
  Tests : `tests/test_delta_candidates.py`.

- **`DEFAULT_PENALTY = 40` est INERTE sur le corps d'un morceau** (2026-09-16,
  trouvé en cherchant à illustrer ce que chaque étage change). De 5 à 200,
  Ready rend exactement les mêmes 86 accords. Cause structurelle : le décodeur
  vendu n'applique `diff_trans_penalty` que là où `beat_arr[t] == 1`
  (`xhmm_ismir.py:120`), et `musx.make_beat_arr` ne laisse la valeur 1 que sur
  les images HORS de la grille de temps — 49 images sur 6671 pour Ready, avant
  le premier temps et après le dernier. Le vrai curseur de densité d'accords
  est `beat_trans_penalty` : 5/15/30 donne 92 accords, 15/45/100 en donne 86,
  40/80/200 en donne 78. La docstring d'origine annonçait « pooled optimum 40,
  LOSO-stable sur 7 morceaux » ; aucune trace de cette étude dans `docs/` ni
  `archive/`, et elle aurait de toute façon mesuré un curseur mort. Valeur
  gardée telle quelle (la changer ne peut rien casser), docstring corrigée,
  garde-fou `tests/test_penalite_inerte.py` : si `make_beat_arr` repose un jour
  des 1 au milieu d'un morceau, le test rougit au lieu de laisser un curseur
  ressusciter en silence. **Ce que ça ne règle pas** : les trois nombres
  15/45/100 restent posés — leur FORME est arbitrée (This Love, 2026-07-31),
  pas leurs valeurs.



- **`harmonia_min` supprimé (sprint 22, 2026-09-16).** Le paquet n'était plus
  qu'un décor : sur ses 20 modules, 12 étaient des PONTS de 2 à 7 lignes
  (`from harmonia.X import *`) qui réexportaient `harmonia` — la dépendance
  était inversée depuis les sprints 2-10. Seuls 7 fichiers portaient du code
  que `harmonia/` n'avait pas ; ils sont déplacés tels quels (`git mv`, 97 à
  100 % de similarité détectée par git) à plat dans `harmonia/` : `soudure.py`,
  `ssm_page.py`, `section_tool.py`, `jam.py`, `phrases4.py`, `titles.py`,
  `annotations.py`. Les 22 sites d'import de `harmonia/` visent maintenant le
  module réel ; aucun `import *` n'a été ajouté (c'est ce que faisaient les
  ponts, et ils sautaient les noms à underscore — un ajout concurrent,
  `_ireal_cascade`, avait déjà cassé là-dessus le 2026-09-14).
  Vérifié champ par champ, pas seulement « 0 mesure changée » : le run frais
  contre la baseline gelée donne **un seul chemin JSON différent,
  `.meta.engine`, sur 44 morceaux sur 44**. Écrans Soudure, /ssm et éditeur de
  sections ouverts à 390×667 sur un serveur jetable (:7775) — zéro erreur
  console, contenu réel affiché. Tests : 267 passés, 1 rouge (le rouge hérité
  ci-dessus, seul et inchangé).
  **Ce que ça ne résout PAS** : (a) `harmonia_min/` existe toujours, réduit à
  son `__init__.py` + `chord_lm/` — recherche en cours d'une autre session,
  intouchée ; (b) `HARMONIA_MIN_PORT` reste lu par `settings.py`,
  volontairement : c'est le nom qu'emploient les serveurs de worktree pour ne
  PAS se poser sur :7772, le retirer les ferait tomber sur l'app vivante ;
  (c) la baseline n'est pas re-gelée (voir juste au-dessus).

- **Compass et Guide montrent jusqu'à cinq accords et la basse entendue
  (2026-09-16, demande de Louis).** « Je veux qu'il apparaisse les autres
  suggestions que juste le top 2, plus les suggestions sur la ligne de basse
  en faisant des petits cercles autour des lettres du cercle […] toujours la
  taille du cercle proportionnelle à la proba d'être détectée, top 3 basses,
  top 5 accords si relevant. »
  Les deux vues partagent déjà leur source (`candList`, un seul endroit), donc
  la liste d'accords s'allonge une fois pour les deux. Le « si relevant » est
  un PLANCHER, jamais un remplissage, et c'est la même loi pour les deux
  listes : **un candidat ne s'affiche que s'il bat le pur hasard sur son
  propre jeu de candidats**. musx répartit sa masse sur 60 cases (12 racines ×
  5 familles), l'uniforme vaut 1,67 %, plancher **2 %** (`SUG_FLOOR`) ; la
  basse répartit la sienne sur 12 classes de hauteur, l'uniforme vaut 8,3 %,
  plancher **12,5 %** (`BASS_SUG_FLOOR`, une fois et demie l'uniforme). Le
  premier candidat passe toujours : un accord très sûr ouvre l'éditeur sur lui
  seul, comme avant, jamais sur du vide.
  Le plancher a été choisi AVANT d'écrire le rendu, en mesurant l'échelle des
  postérieures sur les deux morceaux les plus ambigus du banc (Yesterday,
  19 accords ; Lost Without U, 74) : au rang 5 la médiane vaut 0,017–0,025,
  donc 2 % tombe pile où le 5e candidat cesse de dire quelque chose. Effet
  mesuré : **3,7 candidats par accord** en moyenne contre 3 figés avant, 12/19
  et 47/74 accords en montrent ≥ 4. À 5 % on retomberait à 2,4–2,9, c'est-à-dire
  l'ancien top-3 : rien n'aurait bougé.
  La basse est lue à l'ATTAQUE de chaque accord (`bass_pc_onset`, 150 ms — la
  mesure du 2026-09-15, 6/11 → 11/11 sur « Ready »), sur le `bothchroma` que
  `pipeline.py` a déjà en mémoire à cette étape : aucune extraction en plus.
  Rendu : un anneau bleu autour de la LETTRE de la roue (une lettre de la roue
  est une classe de hauteur — exactement ce qu'une basse nomme), aire ∝
  probabilité comme les orbes d'accord depuis le 2026-08-08. Bleu et pas la
  teinte de quintes de l'orbe : la basse est une autre VOIX, pas un autre
  accord.
  Code : `harmonia/span_rescore.py` (`SUG_FLOOR`, `BASS_SUG_FLOOR`,
  `bass_suggestions`), `harmonia/pipeline.py`, `harmonia/static/screens/
  annotate.js` (`candList`, `bassList`, `byArea`, `bassStrip`),
  `chart.js` + `prefs.js` (la liste blanche des champs, et la transposition).
  Tests : `tests/test_suggestions_floor.py`.
  **Régression trouvée et corrigée en route — le plancher vide la roue sur les
  accords les PLUS sûrs.** Le compas retire l'accord écrit avant de placer ses
  orbes (2026-08-08 : le redessiner sur la jante ne disait rien et volait un
  rayon). Avec un top-3 figé, en retirer un en laissait toujours deux. Avec le
  plancher, un accord que musx tient à 95 % ne garde que SA propre case : la
  roue se vidait, et `noCandBox` annonçait « written without the model's
  ranking » — l'exact contraire de la vérité, précisément là où le modèle est
  le plus sûr (2 accords sur 23 dans Yesterday, 2 sur 86 dans Lost Without U).
  `buildCompass` distingue maintenant deux silences : aucun classement du tout
  (→ `noCandBox`, inchangé) versus le modèle est sûr (→ la roue reste, avec
  son moyeu, ses anneaux de basse, et l'en-tête « the model is sure — 95 % on
  this chord »). Gelé côté Python par
  `test_un_accord_tres_sur_ne_garde_que_sa_propre_case`.
  **CE QUE ÇA N'ÉCRIT PAS, et c'est ce qui permet de le brancher aujourd'hui** :
  aucun slash n'apparaît dans le chart. Le champ `bass` de l'accord n'est pas
  touché, `bass_rules.decide_bass` n'est pas appelé. Le banc corpus qu'exige
  l'entrée « Basse détectée à l'attaque » ci-dessus porte sur la DÉCISION
  d'écrire une basse, pas sur le fait de MONTRER la lecture. Les deux
  planchers de basse qui cohabitent désormais (`bass_rules.FLOOR` = 30 %, en
  pourcents, décide ; `BASS_SUG_FLOOR` = 0,125, en part, affiche) sont gardés
  distincts par un test exprès — les confondre serait l'erreur de calibration
  silencieuse de CLAUDE.md #1.
  **Ce que ça ne résout pas** : (a) le plancher est calibré sur la FORME de
  l'échelle (où le 5e rang décroche), il n'a pas été arbitré à l'oreille, et
  sur deux morceaux seulement (règle #5) — à rouvrir si la liste paraît trop
  longue ou trop courte ; (b) ~~les anneaux de basse ne sont pas tapables~~
  **TRANCHÉ le 2026-09-16 : ils le sont** (Louis : « les basses sont
  sélectionnées à part, en cercles pareil ») — voir l'entrée suivante ; (c) la basse MOBILE (une note par
  temps) reste résumée par une seule lecture à l'attaque de l'accord ; (d) les
  charts d'avant le 2026-09-16 n'ont ni 5 candidats ni `sugBass` tant qu'ils
  ne sont pas recuits — l'affichage marche sans, il montre juste ce qu'il a.

- **L'accord optionnel « en petit au-dessus », comme iReal (2026-09-16).**
  Louis, sur Easy On Me section B : « c'est un F puis la basse descend sur D
  (donc ça donne un D-7) avant d'atterrir sur le C … typiquement le genre de
  cas où j'aimerais avoir juste F affiché et le D-7 en optionnel en petit
  au-dessus en suggestion, comme iReal fait. » `harmonia.folding._bar_variant`
  lit, à chaque position d'un gabarit replié, le décodage de première passe
  de chaque passe GARDÉE avant que le gabarit ne les réécrive toutes — la
  seule fenêtre où cette information existe encore — et retient l'accord
  minoritaire (confiance ≥ `VAR_MIN_CONF = 0.5`) le plus soutenu, groupé par
  fondamentale + famille (`harmonia.roles.family`, pour que « D- » et « D-7 »
  comptent comme la même lecture). Posé comme champ `var` sur l'accord écrit
  — un champ que le rendu (`harmonia/static/screens/chart.js`) attendait déjà
  depuis le 2026-08-17 (« l'autre lecture de cette case, sur un chart replié
  ») sans qu'aucun code serveur ne l'alimente ; distinct de `sug` (le
  classement du modèle sur le span de l'accord ÉCRIT, une question d'
  incertitude intra-mesure — `var` est un accord RÉELLEMENT joué par une
  minorité d'AUTRES passes, une question de désaccord inter-passes).
  Mesuré sur Easy On Me (`min_X-yIEMduRXk`, section B position 0) : F écrit,
  3 des 6 passes empilées jouent F puis D- au 4e temps (0,749/0,698/0,722 de
  confiance) — `var: {root: D, q: "-", c: 0.749, n: 3}`. Rapport d'or sur 45
  morceaux cuits : 6 changent (`identique: false`, dont Easy On Me), tous à
  **0 mesure changée** — vérifié en retirant `var` de chaque chart, byte-
  identique à la baseline dans les 6 cas : aucune régression racine/qualité/
  basse, uniquement l'apparition du nouveau champ.
  **Ce que ça ne résout pas** (à confirmer par Louis) : (1) l'orthographe
  affichée est celle de la passe la PLUS confiante du groupe (ici « D- »,
  pas « D-7 » bien que 1 des 3 passes ait décodé la septième) — pas un
  mélange inventé, mais peut-être pas ce que Louis préfère voir ; (2) une
  seule variante par position (si deux temps différents portaient chacun
  leur propre lecture minoritaire, seule la plus soutenue survivrait) ;
  (3) seuil `VAR_MIN_CONF = 0.5` mesuré sur UN SEUL morceau (règle #5,
  CLAUDE.md) — à vérifier corpus entier ; (4) ne couvre que le chemin
  `fold_letter_groups` (sections auto-détectées) — le chemin
  `harmonia.soudure.sections_pour_chart` (découpage à la main, déplacé de
  `harmonia_min` au sprint 22, fusionné le même jour que cette entrée) n'a
  pas son propre calcul de variante.
  `harmonia/folding.py` (`_bar_variant`, `_attach_variant`, `VAR_MIN_CONF`),
  `harmonia/pipeline.py` (docstring ChartModel), `harmonia/static/screens/
  chart.js`, `tests/test_folding_variant.py`.

- **Un temps en trop à 0,28 fois le temps décale toutes les barres après lui
  (2026-09-15, Louis sur Ready de PJ Morton).** Son diagnostic : « entre la
  mesure 8 et la mesure 9 il y a un petit temps de pause qui devrait être
  détecté […] on décale tous les accords d'un quart de barre ». La pause est
  réelle et mesurée (le temps à 20,90 s est à −15 dB sous le niveau courant,
  musx y met 12–18 % sur « aucun accord »), mais ce n'est pas elle qui décale :
  Beat This! pose **deux marques à 180 ms** à 13,9 s, soit 0,281 fois le temps
  médian, et `drop_duplicate_beats` coupait à 0,25. La paire passait, la liste
  gardait un temps de trop, et toute la grille de mesures construite par
  arithmétique d'indices tombait un temps trop tôt à partir de là — mesure 9 à
  20,900 au lieu de 21,520.
  Le seuil ne pouvait pas monter : remesuré sur les 199 morceaux en cache, les
  intervalles courts ne font pas les trois paquets nets qu'annonçait la
  docstring — la distribution est continue de 0,10 à 0,40 et le paquet des
  triolets (0,34–0,36, 130 cas) est trop proche. Ce qui sépare proprement,
  c'est ce que la paire COUVRE du temps d'avant au temps d'après : **1,0
  période** (142 cas) ou **2,0 périodes** (459 cas), creux franc entre les
  deux. Règle ajoutée, additive, sans toucher au seuil existant.
  Vérification par un juge extérieur : les barres retombent exactement sur les
  temps forts de Beat This! (18,960 / 21,520 / 24,100 / 34,380) et les trois
  accords que Louis avait déplacés à la main retombent à 10 ms près tout
  seuls. Portée : 9 morceaux / 15 paires sur 197 en cache ; rapport d'or 16
  morceaux, 82 mesures (page d'effets de bord à part).
  `harmonia/beats.py::drop_duplicate_beats`, `tests/test_beats_jumeaux_larges.py`,
  pages `/plots/ready_jumeau.html` et `/reports/jumeau_large/avant_apres.html`.
  **Ce que ça ne règle pas** : (a) la phase des barres reste un seul nombre
  pour tout le morceau (`bars.py`, `Counter(...).most_common(1)`) — un morceau
  qui gagne vraiment un temps au milieu reste inécrivable ; (b) l'accord posé
  sur le temps où il « prend l'avantage » plutôt que sur celui où il est joué
  — hypothèse mesurée et **refusée comme loi** : sur 2132 changements / 46
  morceaux, l'accord écrit est déjà l'argmax dans 91,0 % des cas et le décaler
  d'un temps ne monte qu'à 91,4 % ; 4 % de cas stricts, dont 29 % sur le seul
  *Man I Need* (le même Gb, 25 fois, en section repliée). Voir
  `scratchpad/early_switch_screen.md`.

- **La façade d'un bloc ×N montre le consensus, jamais une mesure rejetée
  (2026-09-15, verdicts de Louis sur la page d'arbitrage).** Lazy Song A
  affichait « B Ab- » ×6, Every Breath « Ab » ×7, Yesterday « E- D-/A » ×4 :
  le consensus était juste, la passe affichée portait une mesure rejetée
  par la pile qui gardait son propre décodage. `folding.facade_view`, une
  vue d'affichage lue par les deux chemins de rendu pour le BLOC écrit ;
  les fins iReal et la cascade restent sur le brut (lues sur la vue,
  elles exposaient le bruit des dernières mesures — Louis : « ça a
  empiré tous les morceaux », retiré le soir même) ; `bars` intact ;
  jamais sur une passe seule ni à deux. 4 mesures changent, les trois
  verdicts.

- **La tenue qui ouvre une boucle repliée n'était jamais écrite
  (2026-09-15, Louis sur Let It Be : « le premier accord (C) n'est jamais
  propagé sur la première barre »).** `folding._decode_template` écrivait
  la tenue au temps 1 pour toutes les positions sauf la 0 ; le segment qui
  chevauche le début de la copie du milieu du gabarit ×3 devient cette
  tenue (jamais un N.C.). Let It Be A : C~ G | Am F | C G | F C.

- **Recherche de latence musx supprimée (2026-09-15, audit, accepté par
  Louis).** Mesuré sur 46 morceaux : les changements du modèle tombent
  23–46 ms AVANT la battue Beat This!, jamais après ; la recherche ne testait
  que 0…+280 ms. Relancée par le décodage du gabarit du repli, elle aliasait
  d'une période de battue : 18 morceaux (239 mesures) avec des accords un
  temps trop tôt, invisibles dans `meta.musx_latency_ms` (champ disparu).
  Une loi : pas de latence. Baseline re-gelée, bibliothèque republiée.
- **Passe affichée = la plus riche, sur les deux chemins (2026-09-15).**
  `sections_pour_chart` (découpage à la main) écrivait `occ[0]` ; Stand By
  Me affichait son intro basse-voix (5 N.C. sur 8, ×6). Même loi que
  `minimal_fold` (`folding.pass_evidence`). 9 morceaux changent de passe.

- Le paquet legacy `harmonia/` (pré-refactor), `scripts/harmonia_server.py`,
  et 137+ fichiers associés sont supprimés (tag `pre-refactor-2026-09-14`
  pour les retrouver). Voir `docs/refactor_2026-09/sprints.md`.
