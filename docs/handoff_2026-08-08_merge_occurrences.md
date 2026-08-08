# Handoff — se servir vraiment des N occurrences d'une section

**Pour** : une session Claude dédiée, plein temps, sur une branche à elle.
**Écrit le** 2026-08-08 par la session `fix/annotation-musx-chords`.
**Arbitre** : Louis. Il tranchera **à l'oreille, sur tes pages**, pas sur un chiffre.

---

## 0. La mission en une phrase

Quand une section (A, B, refrain…) est jouée N fois, on a N observations du
**même** enchaînement d'accords. Empiler ces N observations devrait diviser le
bruit par √N. Aujourd'hui l'empilement existe (`fold_letter_groups`) mais il est
naïf sur cinq points. Ta mission : **traiter les cinq leviers dans l'ordre, et
produire pour chacun une page HTML cliquable et écoutable** qui permette à Louis
de trancher lui-même.

Le principe est bon. C'est la mise en œuvre qui est le problème.

---

## 1. Règles de travail — non négociables

Plusieurs sessions Claude travaillent en parallèle sur ce dépôt. Ces règles ont
chacune été payées par un incident réel.

- **Travaille dans ton propre worktree.** La copie de travail principale
  (`/Users/vincente/Documents/Projets Perso/Code/harmonia`) est partagée en
  direct avec d'autres sessions.
  ```
  git worktree add /private/tmp/.../scratchpad/wt-merge -b feat/occurrence-merge
  ```
  **Ne commit JAMAIS sur `fix/annotation-musx-chords`** ni sur `main`.
- **Jamais** `git checkout` / `switch` / `reset` / `stash` / `rebase` dans l'arbre
  principal. Jamais `git add -A`, `git add .`, `git commit -a`, `--no-verify` :
  tu stages des chemins explicites, un par un.
- **Ne touche pas au port 7771** ni à `scripts/harmonia_server.py` (ancienne prod,
  une autre session en est propriétaire). L'app vivante est `harmonia_min` sur
  **:7772**.
- Le serveur `harmonia_min` tourne **sans reloader**. Si ton code semble ne pas
  s'appliquer en live, la première question est « le serveur a-t-il été
  redémarré ? ». Un process vieux de 4 jours a déjà servi du code périmé pendant
  toute une session de debug.
- `harmonia_min/state/` est **gitignoré** : c'est là que vont tes pages HTML, et
  elles ne seront pas commitées — c'est normal et voulu. Commit les **scripts**
  qui les génèrent, dans `scripts/`.
- **Français** dans les docs, les pages et les commentaires destinés à Louis.
  **Anglais** dans les messages de commit.
- **Ne conclus jamais sur un seul morceau.** Toute règle validée sur une chanson
  est une hypothèse tant qu'elle n'a pas tenu sur le corpus.
- **Vérifie le rendu réel** (Playwright, largeur 390 px) avant de dire qu'une page
  marche. Louis a dû le demander plusieurs fois.

Lis avant de commencer : `CLAUDE.md`, `docs/known_issues.md`, et
`git log --oneline -30`. Les six « patterns d'erreur » de `CLAUDE.md` s'appliquent
mot pour mot à ce chantier.

---

## 2. Ce qui existe déjà — à lire dans le code, pas dans cette prose

### 2.1 musx : ce qu'il mange et ce qu'il rend

`harmonia_min/musx.py`.

- **Entrée** : un **CQT** (`extractors.cqt.CQTV2`, SR 22050, hop 512) — **pas** un
  chroma. C'est important pour le levier 1 : la représentation n'est pas invariante
  par transposition, une rotation de chroma n'a pas de sens ici.
- **Sortie** : `frame_posteriors(audio_path)` → **6 flux de postérieures** à
  **43,07 fps** (`FRAME_DT = 0.02322 s`) :

  | index | flux   | dim | contenu |
  |------|--------|-----|---------|
  | 0 | triad  | 73 | N + 12 fondamentales × 6 types |
  | 1 | bass   | 13 | N + 12 classes de hauteur |
  | 2 | s7     | 4  | |
  | 3 | s9     | 4  | |
  | 4 | s11    | 3  | |
  | 5 | s13    | 3  | |

- **Cache** : `data/cache/musx_probs/<stem>.npz`, clé = le stem du fichier audio.
  Déjà peuplé pour tout le corpus de l'app → **coût de calcul quasi nul**, tu peux
  itérer vite.
- **Décodeur** : `musx.redecode(beat_times, probs, downbeat_times=…)` — c'est le
  décodage Viterbi sur grille de battues. `musx.label_confidence(...)` donne une
  confiance par étiquette.
- Modèle : ensemble 5-fold `ChordNet` (ISMIR 2019).

### 2.2 Le merge d'aujourd'hui

`harmonia_min/folding.py:114`, `fold_letter_groups()`. Appelé depuis
`harmonia_min/pipeline.py:580`. Le déroulé exact :

1. Groupe les sections par lettre.
2. Choisit une **période consensus** `P` (en mesures) par vote majoritaire
   (`section_period`, :81).
3. **Empile** : la mesure `b` d'une section commençant en `b0` va dans la case
   `k = (b - b0) % P`. **C'est un modulo rigide** — pas de recherche de décalage,
   pas de tolérance ±1 mesure. C'est le levier 1.
4. **Filtre les intrus** par distance au centroïde, seuil médiane+MAD
   (`OUTLIER_Z`, :169-186). Ce filtre travaille sur les features demi-mesure de la
   détection de sections, **pas** sur les postérieures musx.
5. Vérifie la cohérence du paquet (`STACK_COHERENCE`, :190-201).
6. Vérifie un **CV** demi-mesure par mesure (:206-219) : une position trop variable
   n'est pas écrasée.
7. `_template_chords` (:251) : **moyenne arithmétique** des postérieures des
   membres de chaque position (après ré-échantillonnage temporel `_resample`),
   puis **un seul décodage** du template de P mesures, pavé ×3 contre les effets de
   bord de Viterbi.
8. Réécrit le résultat dans **toutes** les mesures acceptées (:235-240).

### 2.3 Un bug réel, déjà localisé, non corrigé

`cv_skip` (le refus d'écraser une position trop variable) est calculé en :206,
utilisé en :236… et **n'est jamais mis dans `report[letter]`** en :241. Or c'est
`report` que `minimal_fold` (:468) consulte pour décider d'afficher « ×N ».
Conséquence : **l'affichage replie ×N ce que le code a explicitement refusé de
fusionner**. Constaté sur **5 morceaux sur 10**.

C'est petit et sûr. **Ne le corrige pas en douce** : fais-en une page avant/après
comme les autres et laisse Louis arbitrer. `display_fold` (:345) est du code mort,
ne perds pas de temps dessus.

### 2.4 Où viennent les sections

`harmonia_min/voice_sections.py` (mode `voice`, le défaut depuis aujourd'hui).
Score corpus **0,782** sur la métrique de distance d'annotation
(`scripts/section_metric.py`, vérifiée par 19 tests dans
`tests/test_section_metric.py`). Banc : `scripts/section_bench.py`.

Tu **consommes** ces sections, tu ne les changes pas. Si tu penses qu'une section
est fausse et que ça pollue ton merge, dis-le, ne la corrige pas.

---

## 3. Le piège, à intégrer dès la première heure

**Il n'existe aucune vérité terrain d'accords utilisable sur ce projet.** Louis a
condamné explicitement les GT de brick0 (vérifiées à l'oreille ×4, clairement
fausses) — voir la mémoire `feedback_brick0_gt_condemned`.

Donc :

- **Interdit** : « +2,3 pp d'accuracy » comme argument. Il n'y a rien contre quoi
  mesurer.
- **Obligatoire** : chaque idée se juge sur une **page d'écoute avant/après**.
- Ordre de confiance quand des sources s'affrontent : **iReal Pro > tablatures
  Ultimate Guitar ≥ 4,7★ > sortie du modèle**. Utilise-les comme repère
  approximatif — et affiche-les **en tant que tel**, étiquetées « approximatif ».
- Tu peux (et dois) construire des **proxys internes** : stabilité du décodage,
  vraisemblance `musx.path_loglik`, désaccord entre occurrences. Ce sont des
  indicateurs pour **toi**, pour prioriser. Ils ne remplacent pas l'oreille de
  Louis pour la décision finale.

---

## 4. Les cinq leviers, dans l'ordre imposé

L'ordre n'est pas négociable : le levier 1 conditionne le gain de tous les autres.

### Levier 1 — Aligner avant d'empiler  *(≈71 % des dégâts)*

**Hypothèse** : le modulo rigide `(b - b0) % P` empile des mesures qui ne se
correspondent pas. Une occurrence qui démarre une mesure plus tôt, ou une reprise
dont le premier tour a une intro de 2 mesures, contamine **toutes** les positions.

**À faire** : tester chaque occurrence contre le centroïde du paquet en balayant
les décalages (`0…P-1` positions, et **±1 mesure**), retenir le meilleur, et
**refuser** l'occurrence qui ne s'aligne avec rien, au lieu de l'empiler à
l'aveugle.

**Attention** : le mot « rotation » dans la formulation initiale désignait la
rotation **de phase dans la boucle**, pas une transposition de hauteur. Une
rotation de classes de hauteur n'a **pas** de sens sur un CQT (§2.1). Si tu veux
tester une invariance par transposition, fais-le sur le flux `triad` réindexé par
fondamentale, et dis explicitement que c'est ce que tu fais.

**Ce qui réfute** : si le meilleur décalage est déjà 0 partout, l'hypothèse tombe
et tu passes au levier suivant en le disant. Vérifie ça **en premier**, c'est le
test le moins cher (pattern d'erreur n°2 : falsifier avant d'implémenter).

### Levier 2 — Moyenner robuste, pas linéaire

`np.mean` en `_template_chords` : dans une pile épaisse, une occurrence
contaminée (un solo, une modulation, une prise de son différente) passe inaperçue.
Une **médiane** ou une **moyenne tronquée** résiste pour le même coût. Le filtre
`OUTLIER_Z` existant travaille sur d'autres features et ne protège pas ce calcul.

### Levier 3 — Choisir la loi de combinaison, et l'assumer

Deux lois, deux hypothèses différentes sur le monde :

| loi | hypothèse | comportement |
|-----|-----------|--------------|
| moyenne arithmétique | « ce sont N tirages différents » | indulgent, lisse |
| **produit** des postérieures = **somme des log-probs** | « c'est le même accord vu N fois » | tranchant ; un dissident confiant peut opposer son veto |

Le gain théorique du merge — bruit / √N, donc ×1,7 sur 3 occurrences — **n'est
atteint que si l'hypothèse est vraie**. D'où le levier 1 avant celui-ci.
Montre les deux côte à côte sur la même page, avec les cas où elles divergent.

### Levier 4 — Pondérer par la confiance

Une occurrence noyée sous un solo doit peser moins qu'une occurrence propre.
**L'entropie des frames de cette occurrence donne ce poids gratuitement.**
Voir aussi `musx.label_confidence` et `musx.path_loglik`.

### Levier 5 — Sortir la basse de la moyenne

Deux reprises changent souvent d'inversion **exprès** — c'est de la musique, pas
du bruit. L'ancien code excluait le flux basse de la moyenne, test à l'appui :
retrouve ce test dans l'historique (`git log -S` sur `folding.py` et sur l'ancien
`harmonia/`) avant de réimplémenter. La cible du projet est la **basse sonnante**
(`harmonia.data.corpus_schema.sounding_bass_pc`), pas la fondamentale
fonctionnelle — donc écraser la basse par une moyenne détruit exactement ce qu'on
cherche à prédire.

---

## 5. Les deux expériences que Louis a demandées nommément

Elles portent sur les **bi-mesures** détectées (les blocs de 2 mesures de l'analyse
de sections). Il existe déjà `scripts/bibar_prep.py`, `bibar_report.py`,
`bibar_voice.py`, `bibar_sweep.py` — **regarde-les avant d'écrire quoi que ce
soit**.

**Expérience A — superposer l'audio.** Partir des bi-mesures détectées, **empiler
les signaux audio** (les moyenner, alignés), puis donner ce signal à musx.
Question : **musx détecte-t-il mieux les accords sur l'audio superposé ?**
Piège attendu : le désalignement de phase détruit les transitoires et peut donner
une bouillie ; c'est un résultat en soi, et il faut le montrer, pas le cacher.
Fournis l'audio superposé **écoutable** sur la page — c'est la moitié de l'intérêt.

**Expérience B — fusionner les postérieures.** Sur **exactement les mêmes**
bi-mesures, montrer ce que donne la fusion des postérieures de frames.
A et B se comparent directement : même matériau, deux endroits différents dans la
chaîne où on fusionne (signal vs postérieures). C'est la comparaison qui intéresse
Louis.

---

## 6. Livrable — le cahier des charges des pages

**Une page HTML par idée.** C'est le livrable principal, pas un sous-produit.
Louis arbitrera lui-même : ta page doit lui donner de quoi le faire **sans te
poser de question**.

Chaque page doit contenir, sans exception :

1. **Les prédictions d'accords individuelles AVANT merge**, occurrence par
   occurrence — pas seulement le résultat fusionné.
2. **Les probabilités affichées** — la confiance de chaque prédiction, visible.
3. **La vérité terrain si on l'a, même approximative**, étiquetée comme telle.
4. **De l'audio écoutable et cliquable** : cliquer une mesure joue cette mesure.
   Précédents à copier : `scripts/beats_player_page.py`,
   `scripts/dictionary_audio.py`, `harmonia_min/state/reports/chord_lm_audition.html`.
5. **L'avant / après** du changement proposé, côte à côte.
6. Une **phrase en tête de page** disant ce qu'il faut regarder et quelle question
   trancher.

**Où** : `harmonia_min/state/reports/<nom>.html`.
**Lien à donner à Louis** : `http://100.89.209.63:7772/reports/<nom>.html`
(les `file://` sont morts pour lui, il lit sur iPhone).
**Format** : lisible à **390 px de large**. Vérifie au Playwright avant d'annoncer.

Le script générateur va dans `scripts/`, il est commité, et il est
**re-exécutable** (`--clean` pour défaire ce qu'il pose dans l'app, si tu poses
quoi que ce soit dans l'app).

---

## 7. Rythme et critères d'arrêt

- **Point d'étape toutes les ~30 min** : ce que tu as testé, ce que ça a donné, ce
  que tu fais ensuite. Inclus un **contrôle d'espace disque** (un incident de
  disque plein a déjà eu lieu).
- **Log immédiat** : tout résultat non trivial va dans `docs/known_issues.md` ou
  `docs/blog/` **avant** de passer à la suite, pas à la fin de la session.
- **Un levier est fini** quand sa page existe, est vérifiée au rendu, et que le
  lien est donné. Pas quand le code marche.
- **Un levier est mort** quand tu as identifié le **mécanisme** de son échec, pas
  quand un chiffre n'a pas bougé. « Ça n'a pas marché » n'est pas un résultat ;
  « ça n'a pas marché parce que X, et voici la mesure qui le montre » en est un.
- Si les cinq leviers sont traités, la question suivante est **d'où vient la
  source de fondamentale en amont** — c'est le prochain palier identifié
  (mémoire `project_chord_bricks_plateau`).

---

## 8. Pièges spécifiques à ce chantier

- **Ne mesure jamais un merge sur un seul morceau.** Le corpus de l'app fait ~29
  entrées ; Louis en a annoté 18 en sections.
- **Après un swap de composant, diffe TOUS les intermédiaires** (nombre de mesures,
  nombre d'accords, formes de tableaux), pas seulement la métrique visée. Un
  changement de soundfont a déjà doublé silencieusement un tempo.
- **Vérifie ce qu'un interrupteur FAIT avant d'expliquer pourquoi il marche.** Une
  histoire causale entièrement fausse a déjà été construite sur la lecture d'une
  prose au lieu du code.
- **Cache** : `data/cache/musx_probs/` est clé par stem d'audio et ne couvre **pas**
  les constantes de module. Si tu changes une constante en amont, vide le cache.
- Ne re-génère pas la bibliothèque de charts de l'app sans sauvegarde préalable :
  une session concurrente peut être en train d'y travailler.
