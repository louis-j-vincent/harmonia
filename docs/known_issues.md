# Harmonia — Known Issues

Journal court, à jour. Historique complet (2026-07 à 2026-09, avant le
refactor `harmonia_min` → `harmonia`) : `docs/archive/known_issues_2026-07_2026-09.md`.
Mécanique du projet (comment vérifier un changement, où sont les fichiers) :
`docs/STATE.md`.

## UX top-5 de l'audit du 2026-10-01 : ce que chaque changement NE résout PAS

Page avant/après : `docs/plots/ux_5_changements_2026-10-01.html`. Tests Playwright
(rien n'est écrit dans `state/`) : `tests/ui/check_tap_zones.py`,
`check_chord_listen.py`, `check_cut_bar.py`.

1. **Zone à toucher = part de la mesure.** Une mesure sans accord à elle (« % »)
   n'a qu'une zone, qui ajoute un accord au temps 1 ; en écriture serrée les
   frontières sont mesurées sur les glyphes (après rendu), pas sur les temps.
   Pendant l'outil de sections, plus d'éditeur au tap (avant : le tap ouvrait
   quand même l'éditeur sous l'outil). Non fait : numéros de mesure dans la grille.
2. **▶ Original.** Joue `ch.t0→ch.t1` (la 1re passe seulement) : pour un accord
   replié « joué 13 fois », pas de choix du passage. Précision de l'arrêt = un
   `setTimeout` sur l'horloge média, ~±30 ms. Non vérifié sur iPhone réel
   (seulement Chromium headless) : à écouter une fois sur le téléphone.
3. **Couper la mesure / Recoller.** Temps supposés égaux (`bar.t0..bar.t1 / bpb`).
   Le premier accord (hôte) rétrécit en mémoire mais, s'il n'était pas verrouillé,
   son `t1` n'est PAS écrit : au rechargement il redevient pleine mesure (le
   chart se lit par les temps de début, donc sans effet visible ; seul ▶ Original
   de l'hôte jouerait toute la mesure). Recoller ramène la mesure à son premier
   accord (pas de « recoller deux voisins sur trois »). Recoller un accord du
   MODÈLE passe par un nouveau champ `removed` du sidecar
   (`annotations.overlay`) : **il faut redémarrer le serveur** pour que la route
   `/api/annotations` le garde. Une coupe ne s'applique qu'à une mesure qui a au
   moins un accord ; les mesures tenues (« % ») ajoutent seulement au temps 1.
   Trouvé en route : les tests `test_annotations.py` écrivaient dans le registre
   tracké `state/human/corrections/` (corrigé : le fixture le redirige).
4. **Tout en français.** Dock : Lire / Corriger / S'entraîner ; éditeur : Roue /
   Extensions / Propositions / À la main, « Garder Gm » actif dès l'ouverture
   (grisé sur un accord NEUF : il est amorcé par le précédent, on ne le garde pas
   sans choix). Laissés tels quels exprès : les noms de voicings Shell / Rootless /
   Drop-2 (jargon jazz courant), « tab » et « chart ». **Pas fait** : replier Roue
   et Extensions sous « Plus… » (le P2 de l'audit veut 2 onglets seulement) ;
   les mots internes du serveur (messages d'erreur d'API) ne sont pas traduits ;
   les pages `/soudure`, `/ssm` (hors de ces écrans) non revues.
5. **Filet de sécurité.** « Annuler » (6 s) après garder un accord, couper,
   recoller : remet une copie des accords/mesures d'AVANT et réécrit le sidecar.
   Après ÷2/×2 et « Caler la mesure 1 » : proposé 12 s une fois le chart revenu
   (l'attente dure ~1 min) ; ÷2 s'annule par ×2 (déjà exact côté serveur),
   `/api/bar1` rend désormais `previous_bar1` et accepte `{"clear": true}`.
   Une sauvegarde qui échoue affiche « ⚠ Pas enregistré · Réessayer » (avant :
   `catch(e){}`, le toast « Gardé » mentait). **Ne défait pas** : la ligne déjà
   ajoutée au registre des corrections (`state/human/corrections/`) ; annuler
   un tempo/mesure 1 relance une analyse complète (~1 min) ; l'annulation se
   perd si on quitte l'écran avant qu'elle expire. Les routes `removed` et
   `previous_bar1` demandent un **redémarrage du serveur**. Trouvé en route :
   l'écran de chargement (`analyse.js`) est encore en anglais (« listening to the
   track », « Open the chart ») — hors des cinq écrans demandés, pas touché.

## Transposer puis verrouiller un accord réécrivait les annotations (2026-10-01, RÉSOLU)

Trouvé par l'audit UX (`docs/plots/ux_audit_2026-10-01.html`). `transposeTo`
décalait `S.chords` sur place et `saveAnnotations` envoyait ces racines
décalées : après une transposition, le moindre verrouillage réécrivait TOUS
les accords verrouillés du morceau dans la mauvaise tonalité (Gm/B♭ enregistré
Cm/B♭). La basse n'était même pas transposée à l'écran. Correctif : la basse
suit la transposition, et la sauvegarde ramène tout dans la tonalité
d'origine. Test rouge→vert : `tests/ui/check_transpose_save.py` (Playwright,
POST intercepté). **Non résolu** : (1) les annotations déjà écrites après
une transposition ne sont pas détectables automatiquement — à réécouter si
tu as transposé puis corrigé ; (2) les suggestions d'un accord ajouté
(`/api/chord-candidates`) arrivent dans la tonalité d'origine et ne sont pas
décalées dans une vue transposée.

## TODO — la fin alternative de B en 2 mini-cellules, pas un bracket (2026-10-01)

Louis, après le repli de San Francisco Street (B a une vraie fin
alternative, `Em7` au lieu de `A7`) : « on verra après pour ce double
mini-cellules ». Idée : au lieu du bracket 1st/2nd-ending compressé sur
UNE colonne (`_ireal_endings`/`endingBlocks` dans `chart.js`), dessiner le
passage à fin alternative en DEUX mini-cellules — sa moitié commune, puis
sa moitié divergente avec son vrai contenu — comme dans
`tools/repli_propose.py --folded` (qui le fait déjà, page de diagnostic
seulement, jamais branché sur l'app). Volontairement PAS fait le
2026-09-25 : réécrire le rendu des 1st/2nd-endings dans `chart.js` touche
des centaines de lignes de logique de rangée/padding déjà réglées à
l'oreille sur plusieurs morceaux — un chantier à part, avec son propre
avant/après, pas un ajout de fin de session.

## Lost Without U : « juste un A et un B » — A′/A″/B′ inventés par la cascade (2026-10-01, RÉSOLU en partie — 631c071)

Page : `docs/plots/lost_without_u.html`. Le découpage validé n'a que A, B,
outro ; `folding._ireal_cascade` imprimait A′ (mes. 34–37, raccord refusé
pour un Gm7 de relance absent), A″ (mes. 58, un N.C.) et B′ (l'hôte de B
était la mes. 25, un N.C. seul, et le vrai B plus long ne pouvait s'y
fondre). Règles générales : une plage muette n'est jamais hôte et rejoint
sa lettre sans allumer de case ; `meme_moment` = même harmonie ou un accord
de passage près ; un fondu muet final ne compte pas contre un raccord.
Effet ailleurs : San Francisco Street seulement (mes. 71 rejoint A).
**Reste :** (1) l'« outro » 83–89 est l'étiquette de Louis ; c'est le
dernier A avec une vraie cadence (G7sus4 → G7 → C tenu) + fondu. Renommée A
elle sortirait A′ (2/5 mesures diffèrent) : Louis arbitre. (2) une plage
muette absorbée s'affiche comme sa lettre dans la bande, pas comme un trou.
(3) le A reste écrit en 8 mesures sans ×N (règle de boucle de 0031e9a :
G7sus4 contre Gm7 d'un tour à l'autre).

## Boucle minimale : au défaut seulement si l'harmonie est la même, et le curseur (2026-10-01, FAIT)

Louis : « ne la fais pas quand il y a vraiment un changement important ».
`folding.meme_harmonie` (temps par temps, fondamentale + famille ; couleurs,
Bb/C ≡ C7sus4 et renversements ignorés). Curseur « Section merging » (Aa,
par morceau) : 0 rien, 1 défaut (boucle si tous les tours ont la même
harmonie, sinon section écrite en entier), 2/3 différences proches en petite
lecture, 4 toujours. Au défaut : This Love intro/A/B/bridge redeviennent 8
écrites ; en boucle restent Let It Be A, The Walk intro/A, She Will Be Loved
A, Lazy Song D, Every Breath C, X-yIEMduRXk B, San Francisco Street
intro/A′, Be My Baby A, Julia C. Captures (crans 0/1/4) :
`docs/plots/boucle_curseur_2026-10-01/`. **Reste ouvert** : une différence
cantonnée à la fin d'une boucle n'est pas écrite en crochet 1./2. de tour —
la section s'écrit en entier (les crochets indexent des passages).

## Sections écrites par leur boucle minimale + reprises ||: :|| ×N (2026-09-30, remplacé au défaut par l'entrée du dessus)

Louis : « This Love […] la section A ne devrait faire que 4 barres ».
`folding.boucle_minimale_ecrite` (pipeline + sections à la main), reprises
dessinées par chart.js (captures `docs/plots/reprises_2026-09-30/`). Rapport
d'or : 100 mesures, 16 morceaux ; avant/après `/reports/avant_apres.html`.
Raccourcies : This Love intro/A/bridge 8→4, B 8→4 ; Let It Be A 8→4 ; The
Walk intro/A/bridge 8→4, B 12→4 ; She Will Be Loved A/B 8→4, D 10→4 ;
Don't Know Why B 8→4 ; Sunny A 8→4 ; Be My Baby A 8→4 ; Lazy Song D 8→4 ;
Every Breath C 10→4 ; Claustrophobic B 8→4 ; San Francisco Street intro
8→4, A 24→8 (×3/2), B 9→4 ; X-yIEMduRXk A 12→4, B 8→4 ; Fjufjv4rH0s A 8→4 ;
0DdCoNbbRvQ A 8→4 ; Julia (à la main) A 8→4. **Reste ouvert** : (a) les
lectures d'un tour qui diffèrent sont en `var` (petite lecture), pas en
crochet 1./2. — Julia A (Eb9 Eb7 du 2e tour), SF A (turnaround du dernier
tour, 4 mesures), This Love B (Ab G) ; les crochets indexent des passages,
pas des tours ; (b) à 390px les points « :|| » mordent un 2e accord large
dans la dernière mesure (This Love intro « Fm/A♭ ») ; (c) grille transposée
au 2e tour (Fjufjv4rH0s B, Sunny C) : refusée par le garde 25 %.

## Julia : « C = B avec une queue », « D = petite boucle sur 3 lignes » (2026-09-30, RÉSOLU en partie)

Page : `docs/plots/julia_sections.html`. (1) Le « C » (12 mes.) était B
(10) + 2 mesures : les quatre mots donnaient `cdada` / `cdadaa`, et une
lettre n'était partagée qu'à mot IDENTIQUE (prime seulement à longueur
égale). **Fait** : `simulation._prefixe_et_queue` — un bloc libre qui
commence par le mot entier d'un autre bloc (≥ 4 bi-mesures, ≥ 2 lettres
distinctes, queue plus courte) devient cette lettre + « queue ». 21
découpages validés × 2 protocoles : 0 bloc changé. (2) `soudure.
sections_pour_chart` n'écrivait jamais la boucle interne d'un passage.
**Fait** : `soudure.boucle_ecrite` — si tous les passages d'une lettre
suivent une cellule de 2 ou 4 mesures (famille d'accord), fondu N.C. de fin
et dernier tour incomplet tolérés, sur ≥ 3 rangées → une rangée de 4,
`barSpans` allume chaque tour, champ `boucle` {rangee, cellule, tours}.
Rapport d'or : 5 mesures changées, San Francisco Street A′ seul (13 mes. →
1 rangée, A7/A7sus4 confondus en A7sus4). **Reste ouvert** : pas de marque
×N dessinée (question à Louis, maquette sur la page) ; seuil 2 rangées
(8 mesures = 4 × 2) non branché, il toucherait 8 morceaux ; préfixe décalé
ou tête ajoutée avant B non traités ; Julia doit être re-validée (bouton)
après redémarrage de :7772 pour voir le correctif.

## Fins alternatives : Bb/C ≡ C7sus4, et le curseur « Section merging » (2026-09-30, RÉSOLU en partie)

Louis, sur Claustrophobic (mCwI0YA4mIY) : un A avec Bb/C et un A avec
C7sus4 faisaient deux fins. Cause : `folding._ireal_endings` groupait les
queues par égalité exacte (fondamentale, qualité), basse ignorée — Bb/C
comptait pour « Bb ». **Fait** : (1) un accord sur basse étrangère avec b7
et sans tierce au-dessus de la basse se lit AUSSI 7sus4 sur la basse
(`accord_sonnant`, `lectures`, comparaison `_compat` dans les fins et dans
le raccord `_merge_coupe`) ; la lecture écrite reste, donc rien ne se
dé-fusionne (une 1re version l'a fait sur 9zEl-FQLI4A : `G/A` contre
`G^7`). Rapport d'or : 0 mesure changée / 63 ; seule décision changée :
Claustrophobic, fins [1] + [2,3] fusionnées. (2) Curseur 0–4 par morceau
dans Aa (défaut 1 = avant) : 0 garde les fins de couleur, 2/3 fusionnent
les fins qui ressemblent à ≥ 0,90 / 0,70 (cosinus chord-tone) à la plus
jouée, 4 = un bloc ×N. Rendus précalculés (`chart.sectionsFusion`) à
« Valider » et au rapport d'or, réglage dans `chart_meta.json[stem].fusion`.
Captures : `docs/plots/fusion_fins_2026-09-30/`.
**Reste ouvert** : (a) les charts déjà en cache n'ont pas les rendus → le
curseur n'apparaît qu'après un « Valider les sections » ou une republication
(`tools.golden --publish`), et après redémarrage du serveur ; (b) le curseur
ne touche pas les blocs de longueurs différentes (A′) ni le chemin pipeline
(morceaux non annotés, une seule section) ; (c) Eb7/C (= C-7b9) n'est PAS
lu comme C-7 — autre règle, à arbitrer ; (d) Claustrophobic au défaut garde
4 fins (Bb-7 au lieu d'Eb7, C9 F7, Bb7 F^7) ; l'app n'en dessine que 3
(« A », « alt 1 », « alt 2 ») — la variante C9 F7 manque au rendu, non
investigué.

## « Il manque une section avant un A : c'est une intro, pas un C » (2026-09-30, RÉSOLU en partie)

« Valider les sections » nommait la zone libre de tête d'une lettre neuve
(« C »). Désormais `simulation.remplir_par_algo` en fait UN bloc « intro »
(la tête seulement ; trous du milieu inchangés ; pas d'outro). Sur 21
découpages validés, Louis supposé ne pas tracer son intro : 9 intros rendues
aux mesures exactes, 2 sur des mesures qu'il a laissées sans section (l'app
les appelle déjà « intro »), 1 trop courte (Don't Know Why 1-2 au lieu de
1-4), 0 fausse. **Reste ouvert** : 4 intros ratées (San Francisco Street,
She Will Be Loved, This Love, The Walk) parce qu'une REPRISE d'une brique
prend déjà la mesure 1 — l'intro joue la matière d'une section, et il n'y a
plus de zone libre. Détail : `docs/sections_etat_2026-09-30.md` § 13. Actif
au prochain redémarrage du serveur.

## Retrouver les occurrences d'une section donnée : les règles 1–5b de Louis branchées + candidats en suspens + option (c) (2026-09-30, RÉSOLU en partie — reste les jumeaux)

Louis sur la page du matin : « le détecteur avec pics est bon », verdicts
(Stand By Me = une seule section ; The Walk, B peut être plus court) et trois
règles — pas de pic dans la zone donnée ; chercher la BOUCLE MINIMALE de la
brique ; juger un pic par rapport à son fond local. Implémentées dans
`simulation.cherche_sur_matrices` (chemin live de `cherche_brique`, actif au
prochain redémarrage du serveur). Détail et justification :
`docs/sections_etat_2026-09-30.md` § 8 ; page terme par terme :
`docs/plots/occurrences_brique_v2_2026-09-30.html` (rejouable :
`python -m tools.occurrences_brique_v2 --page`).

Sur 20 découpages validés (mesures) : justes 1270 → 1402, ratées 260 → 128,
en trop 278 → 408. Corrigés : San Francisco Street A (24 → 40/41, la reprise
de 16 est retrouvée), Stand By Me A (32 → 80/80), Let It Be A (8 → 32/32),
The Walk A (12 → 40/40). Rapport d'or : 0 mesure changée (60 cuits).

RÈGLE 4 (2026-09-30 soir, après la page v2 : « beaucoup beaucoup beaucoup
mieux » sauf Let It Be) : la meilleure reprise hors zone (R) calibre le
morceau ; on coupe sous le plus grand trou entre hauteurs de pics s'il dépasse
2 × max(1 − R, écart du groupe du haut, 0,01) ET si le groupe du bas est plus
nombreux (`simulation.reference_du_morceau`, eb72b62). Page :
`docs/plots/occurrences_brique_v3_2026-09-30.html`. Ne coupe que 3 lettres :
Let It Be A 24 en trop → 0 (trou 0,066 contre tolérance 0,02, 15 pics dessous
contre 6) ; Stand By Me intact (tolérance 0,114, trou 0,020) ; **The Walk B**
20/16/0 au lieu de 24/12/4 (décalage supprimé mais 4 mesures justes perdues) et
**X-yIEMduRXk A** 20/5/8 au lieu de 24/1/8 (un jumeau dans le C aussi haut que
la vraie reprise gonfle le groupe du haut, 31–34 coupé) — à arbitrer par
Louis. Total : justes 1402 → 1394, ratées 128 → 136, en trop 408 → 380.
Rapport d'or : 0 mesure changée. **Arbitré par Louis le 2026-09-30 : « cette
règle est top, on garde » — EN PROD (:7772 redémarré), pertes The Walk B /
X-yIEMduRXk A acceptées.** Nouveau signalement le même jour : sur The Walk,
après les ~3 premiers petits A, les reprises semblent décalées (~2 mesures).
**Diagnostiqué** (`docs/plots/the_walk_decalage.html`,
`tools/the_walk_decalage.py`) : grille (premiers temps beatthis à ±20 ms,
tempo fixe), accords (A/Bm, jamais de changement de parité), sections de Louis
et affichage (intro avec accords, pas masquée) sont justes. Le décalage vient
de l'outil d'annotation : A donné en 9–16 → boucle de 4 (A–Bm–A–Bm) posée à
27, 31, 35… au lieu de 29, 33, 37 (mes. 27 = 75,0 s), de nouveau juste à 57,
décalée encore à 83. Cause : le choix final « le plus de boucles sans
recouvrement » — depuis 27 on en case 7 dans 25–56, depuis 29 seulement 6 ;
les deux boucles de bord (27, 51) commencent sur le C♯m du B et ne valent que
0,91 contre 0,996, mais passent plancher, règle 3 (fond ~0,2 aux décalages
impairs) et règle 4 (7 pics bas contre 29 hauts : pas de coupe). C'est le
point 2 de CE QUI RESTE, en plus général. Options, à arbitrer par Louis :
(a) un départ n'est gardé que s'il est au moins aussi haut que ses voisins à
±sous-période (27 perd contre 29, 51 contre 49) ; (b) règle 4 sans la
condition « plus de pics en bas qu'en haut » quand la boucle a une
sous-période ; (c) au lieu du nombre, maximiser la somme des scores au-dessus
du fond. **Deuxième défaut trouvé en chemin, corrigé** : une reprise
pouvait recouvrir un trait de Louis posé plus loin (les traits sont traités
dans l'ordre des mesures) et le trait disparaissait — The Walk B 25–36 mangé
par les A décalés. Sur les découpages validés : 11 morceaux concernés, mesures
bien étiquetées en hausse partout sauf Let It Be (égal), aucune en baisse ;
rapport d'or 0 mesure changée (seules les propositions de l'outil bougent).
Serveur PAS redémarré.

**RÈGLE 5 de Louis, branchée (2026-09-30 nuit, commit 63c3df7)** — l'option
(a) généralisée : « si on hésite entre un départ et celui de ses deux voisins,
on prend celui qui est plus grand que ses voisins ». `simulation.maxima_locaux` :
parmi les départs qui passent la règle 3, seul le plus haut à ±(boucle − 1)
mesures reste candidat ; égalité à 0,01 près (les deux restent, sinon un
plateau de sous-période serait réduit à un départ) ; la zone donnée n'est
jamais voisine ; posée après la règle 4 (inchangée), avant le choix « le plus
de boucles ». **The Walk A réglé** : 29, 33, …, 49 au lieu de 27, …, 51 ; 85,
89, 93 au lieu de 83…95 (justes 40 = 40, en trop 6 → 4). Page
`docs/plots/occurrences_brique_v4_2026-09-30.html`
(`python -m tools.occurrences_brique_v4 --page`). **Pertes, à arbitrer par
Louis** : Lazy Song A −8 mesures justes, She Will Be Loved A −4 — deux accords
qui alternent, et le bon départ y est un peu PLUS BAS que son voisin décalé de
2 (0,963 contre 0,977–0,991) : la règle tranche sur du bruit. She Will Be
Loved D : −8 en trop. Total justes 1394 → 1382, en trop 380 → 370.
**Variante 5b, NON branchée** : égalité jusqu'à la tolérance du morceau de la
règle 4 (0,04–0,07) au lieu de 0,01 — The Walk réglé pareil, aucune perte
(justes 1394, en trop 378). Serveur pas redémarré.

**5b arbitrée par Louis et branchée, + CANDIDATS EN SUSPENS (2026-09-30,
commits 14cb9ad, 0ffa940)** — « Oui pour la 5b très bien, mais alors il faut
trancher par rapport à la cohérence globale du morceau, en cherchant d'autres
sections et en gardant ces 2 candidats potentiels sans en acter un ».
`maxima_locaux(tol=)` reçoit la tolérance de la règle 4 ; deux calages rivaux
qui casent AUTANT de boucles ne sont plus actés : `occurrences_de_tous` les
retient, pose toutes les briques de Louis et leurs reprises sûres, puis
`resoudre_suspens` tranche : mangée par une autre section > phase (collé aux
autres occurrences de la brique) > bouts orphelins < 4 mesures > score. Le
choix sort « ressemble » (pointillés, pas renvoyé comme contrainte, donc
retranché à chaque « Valider »). Pas de changement d'UI. Page
`docs/plots/occurrences_brique_v5_2026-09-30.html` (`python -m
tools.occurrences_brique_v5 --page`) — elle rejoue le VRAI flux (1re
occurrence de chaque lettre tracée, toutes d'un coup) : justes/ratées/en trop
v4 1232/218/81 → v5 1244/206/81. Lazy Song A 40 → 48 (sans suspens : 58 + 62
casent 2 boucles contre 1 pour 60) ; She Will Be Loved A+B+C justes 56 → 64,
en trop 32 → 24 ; The Walk inchangé (phase gardée). 14 suspens sur le corpus,
tous tranchés par « mangée » : dans ce flux, les critères phase / orphelins /
score ne décident rien (ils décident quand la brique est seule). **Coût, à
arbitrer par Louis : Sunny A −8** — son A 18–25 en suspens (18 à 0,950 contre
22 à 0,874 : tolérance du morceau large, 0,10) est pris pendant l'attente par
la boucle de C (mêmes accords, 0,984) ; avant, A gardait la zone seulement
parce qu'il était tracé avant C. Rapport d'or : 60/60 identiques, 0 mesure
changée. Serveur pas redémarré.

**Louis a choisi (c), 2026-09-30 — branché.** Une autre section ne mange la
zone d'un calage en suspens que si son score y dépasse celui du calage de PLUS
que la tolérance du morceau (celle qui définit l'égalité en 5b, jointe à chaque
groupe : pas de nouvelle constante). Sinon sa reprise est retenue, et ne revient
que si le suspens est tranché ailleurs. Traits de Louis toujours gagnants. Les
zones en suspens sont toutes connues avant qu'aucune reprise sûre ne soit posée
→ indépendant de l'ordre des briques (testé en échangeant les places de A et C).
Colonne « (c) » sur la page v5. v5 1244/206/81 → (c) 1248/202/79 : **Sunny A
8 → 16 justes** (et C perd ses 8 en trop). **Nouveau coût, à arbitrer :
X-yIEMduRXk A 20 → 16 justes, C +9 en trop** — un jumeau de C (27–35, 0,947)
est en suspens dans le 2e A de Louis ; la reprise de A (27–30, 0,926) est plus
BASSE, donc ne le mange plus. Reste non résolu : entre deux reprises SÛRES de
briques différentes, le premier trait dans les mesures garde la priorité
(inchangé depuis v1).

CE QUI RESTE :
1. **Jumeaux harmoniques au niveau des vraies reprises** : She Will Be Loved A
   (+28) et D (+60, même grille presque tout du long), This Love A (l'intro),
   X-yIEMduRXk B. La règle 4 ne les voit pas (aucun trou) ; Louis tranche
   (comme Stand By Me ?). Si le meilleur pic hors zone est un jumeau, la règle 4
   se calibre dessus (X-yIEMduRXk A).
2. **Phase d'une boucle à deux accords** : The Walk B — v3 garde la bonne phase
   mais rate toujours les 2e et 3e passages de chaque B. La règle 5 règle The
   Walk A ; la 5b + les suspens (ci-dessus) rendent Lazy Song A et She Will
   Be Loved A. Reste : quand deux calages casent autant de boucles et que rien
   ne les mange, la phase est décidée par le contact avec les autres
   occurrences, puis par le score — l'harmonie seule ne sait pas la phase
   d'A–Bm–A–Bm. Un suspens n'est plus mangé par un jumeau à peine plus haut
   (option (c), Sunny réglé) — mais un jumeau EN SUSPENS garde la zone contre
   une vraie reprise plus basse (X-yIEMduRXk C).
3. **Vamp** (Chain of Fools, 0,99 partout) : abstention, comme avant.
4. Stand By Me C : 15–22 refusé de justesse (proéminence 0,57).

DIAGNOSTIC DU MATIN (conservé) :

Louis : « une fois que l'annotateur a donné une première section, il faudrait
pouvoir BIEN détecter les occurrences de cette section ». Simulé sur ses 20
découpages validés (première occurrence de chaque lettre = la brique),
`simulation.cherche_brique` tel qu'en prod. Page, terme par terme :
`docs/plots/occurrences_brique_2026-09-30.html` (rejouable :
`python -m tools.occurrences_brique --page`). État des lieux :
`docs/sections_etat_2026-09-30.md`.

CE QUI RATE, par ordre d'importance :

1. **Le seuil bute au plafond 0,999.** Quand médiane + 1,5 σ dépasse 1 (la
   brique revient souvent), le seuil devient 0,999 et recale des reprises à
   0,989–0,998. Let It Be A (0/3 trouvées), The Walk A et B, Chain of Fools :
   15 occurrences perdues sur 4 morceaux, toutes au sommet de leur courbe.
2. **La longueur est figée à celle de la brique.** San Francisco Street A
   (24 mes.) contre sa reprise de 16 : moyenne 0,84 < 0,85, ratée alors que ses
   16 mesures collent à 0,97–1,0 (calées sur la 2e boucle de la brique). Même
   défaut sur 8 autres occurrences courtes/longues et 2 fins alternatives
   (She Will Be Loved C).
3. **Jumeaux harmoniques** (presque toutes les fausses trouvailles) : Stand By
   Me A/C, Yesterday B dans A, She Will Be Loved C dans B, l'intro de This Love
   et d'Every Breath, Sunny C = A transposé. Aucun seuil ne les séparera — il
   faut un autre canal que les accords (déjà prouvé sur Stand By Me le 09-20).

CONTRE LE CADRAGE DE DÉPART : les décalages d'UNE mesure n'apparaissent pas
(zéro en prod), et les reprises transposées sont déjà trouvées (Sunny,
Fjufjv4rH0s, T64BgKEL-Sw). Deux décalages de 4 mesures (boucle interne : Lazy
Song, The Walk).

CANDIDATS (dans l'outil, RIEN n'est branché) :
  * **candidat 1 — plafond 0,97** (brique ≥ 8 mesures) : +10 occurrences
    justes (Let It Be ×3, The Walk ×5, Chain ×2), +5 décalées de 4 (Chain,
    boucle), +3 en trop (The Walk : l'intro et la 2e moitié des B jouent la
    boucle). Plateau de 0,96 à 0,985. Sous 8 mesures, 0,97 inonde (Lazy Song C
    +5, une brique d'1 mesure +6) — d'où la borne.
  * **candidat 2 — passages plus courts + fin alternative** : la prémisse
    tient (le contenu colle mesure à mesure à ≥ 0,96 sur les occurrences de
    mauvaise longueur) mais cette version rattrape peu (SF : 12 mesures sur 16)
    et ajoute des fragments jumeaux. Il lui manque de chercher la reprise sur
    TOUTES les phases de la brique, pas seulement son début.

QUESTION OUVERTE pour Louis : les 3 derniers « A » d'Every Breath You Take
(mes. 89–111) n'ont que 4 mesures sur 8 en commun avec le premier (la boucle
de fin) — A, ou A′ (sous-replier) ?

## Le chart commençait par « N.C. » ou « % » (2026-09-30, RÉSOLU en partie — e2d26cb)

Louis : « on commence toujours une mesure trop tôt j'ai l'impression, à
chaque fois le chart commence par un % ou un N.C. ». Mesuré (46 charts
chauds, rendu Read émulé) : **13/46 imprimaient « N.C. » ou « % » en
premier ; 6/46 après.** Page : `/plots/premiere_mesure.html`.
**Arbitré par Louis le 2026-09-30 : « clairement mieux le après » — EN PROD
(:7772 redémarré le même jour).** Consigne liée : virer l'ancienne façon
(`_force_bar1_sections` et la branche SongFormer qui la portait).

**Cause (7 des 13).** La grille part de la 1re battue du traqueur ; la
musique ou la marque « mesure 1 » arrive 1 à 7 mesures plus tard. Ces
mesures muettes allaient dans une « intro » (`_force_bar1_sections`) que
chart.js cache en Read — jusqu'à bbb0ab0 (2026-09-19, SongFormer retiré) :
le chart est devenu UNE section « A » qui les contenait. `pipeline._one_section`
rétablit l'intro (tête sans accord → intro cachée ; avant la marque avec
accords → intro imprimée ; la marque ouvre A). Rien ne sort de la grille.
Effet visible en plus : les rangées de 4 retombent sur les phrases (Billie
Jean : les Bm reviennent en début de ligne).

**1787a03 annulé.** Il retirait le N de tête de `prompter.chords` : aucun
effet sur le chart auto, mais sur les morceaux découpés à la main
(`accords_par_mesure` lit cette liste) le N.C. vrai d'une intro devenait une
mesure vide, donc un « % ». Son histoire de « dédoublonnage désynchronisé »
ne correspondait à aucun chart affiché.

**Ce qui reste (6), aucun n'est la grille :**
- marque de Louis suivie de mesures où musx n'entend aucun accord — Hot N
  Cold (4), Uw5OLnN7UvM (3), Sam Smith (1) : la marque reste souveraine,
  ces mesures vides s'impriment « % » (chart.js imprime « % » pour une mesure
  vide même après un N.C. ou en tout début — contraire à la docstring de
  `bars.py` « a bar left empty renders blank » ; arbitrage cosmétique à
  demander à Louis : vide, ou « N.C. » ?) ;
- Grenade : ses sections à la main ouvrent A à la mesure 0, une mesure AVANT
  sa marque (4,86 s) — tout le découpage est peut-être décalé d'une mesure ;
- Be My Baby : intro batterie, « N.C. N.C. » est la vraie notation ;
- Stand By Me : A commence où musx n'entend pas encore la basse (connu).

**Fragilité vue au passage** : la métadonnée du chart auto (`var`, `sug`…)
est recollée par `accords_par_mesure` dans le repli des sections à la main ;
changer les sections AUTO a changé 6 mesures de Stand By Me (A pos. 3 :
N.C. → A F♯m — plus juste, mais par un canal qui ne devrait pas exister).

**Incident** : la session de 1787a03 avait effacé `state/cache/charts/min_*.json`
(bibliothèque vide). Recuite le 2026-09-30 avec le code corrigé (60 charts ;
191P7nIeECo, PWgvGjAhvIw, h_D3VFfhvs4 recopiés de sauvegardes faute
d'audio/grille ; 8oCqn3mN2C8 refusé par le traqueur (2 temps/mesure) et
rec_1787242791735 sans audio : perdus). Sauvegarde d'avant :
`state/cache/charts.bak_20260930_premiere_mesure/`. `tools.golden` exige
encore un cache SongFormer que la pipeline n'utilise plus (16 morceaux
« froids » à tort) ; la baseline gelée date du 17/09, d'avant la sortie de
SongFormer — elle diffère de 554 mesures pour cette seule raison.
**Ancienne façon supprimée (2026-09-30)** : branche SongFormer, `_force_bar1_sections`, `SETTINGS.sections` et `sections/songformer.py` retirés ; `tools.golden` n'exige plus de cache SongFormer ; suppression = 0 mesure changée (60/60 identiques), baseline re-gelée sur la prod (ancienne : `state/cache/golden/baseline_20260917`).

## Un accord fantôme peut apparaître sur la 2e occurrence d'une lettre repliée (2026-09-24, OUVERT — non confirmé en prod)

San Francisco Street, pont (lettre F) : les DEUX occurrences (mes.74-81 et
mes.82-89, après correction — voir l'entrée du dessous) affichaient un
Am7 en tête. Décodé indépendamment mesure par mesure, à même les
postérieures musx brutes, sur la fenêtre audio RÉELLE de chaque mesure
(hors gabarit de chart) : mes.74 (1er passage) n'a jamais eu d'Am7 —
c'est juste l'accord de tournant du beta voisin (confiance 0,26, comme
les 14 AUTRES tournants du morceau) ; mes.82 (2e passage) est du Dmaj7
confiant à 0,64, pas de l'Am du tout. Page qui montre le calcul :
`docs/plots/san_francisco_street_repli_propose_2026-09-24.html`.

LE MÉCANISME PROBABLE (pas encore vérifié sur un chart servi en direct,
seulement sur `state/cache/charts/min_9zEl-FQLI4A.json`) :
`soudure.sections_pour_chart` (`harmonia/soudure.py:885`) choisit UN SEUL
passage représentatif par lettre repliée (`pass_rank`, « la passe qui
porte le plus d'attaques réelles ») et écrit SES accords pour TOUTES les
occurrences (`"bars": vue[b0:b1+1]`, réutilisé pour chaque `barRange`). Le
repli interne par période (`folding.py`, qui moyennerait les postérieures
des deux passages) avait justement été refusé pour cette lettre
(`fold["F"] = {"period": 4, "reason": "stack incoherent (min median
pairwise 0.82)"}`) — donc rien ne vérifie que le passage NON écrit
ressemble vraiment au passage écrit. Si l'audio du 2e passage diverge, le
chart peut montrer un accord qui n'existe dans AUCUNE des deux occurrences
réelles.

Pas encore fait : vérifier si ce morceau, ouvert dans l'app en direct,
montre bien ce même Am7 fantôme (le cache peut être différent du rendu
live) ; si oui, chercher d'autres lettres du corpus dans le même cas
(refus de repli interne + occurrences qui divergent).

## Un brique gelée ne reconnaît plus ses propres reprises ailleurs dans le morceau (2026-09-24, OUVERT)

San Francisco Street (Sun Rai, `9zEl-FQLI4A`) : le vamp `Dmaj7 Em7` (mesures
41–48, section C) est EXACTEMENT le motif intérieur de la section A (mesures
8–31, qui n'est que ce même vamp joué 6 fois). Louis : « on devrait clairement
avoir le pattern Dmaj Em qui se répète qui devrait compter dans un
repliement ». Il a raison — et l'algo du retour, tourné À FROID sur ce
morceau (sans aucune brique humaine gelée), le voit très bien : le mot
harmonique donne `abab` pour A(8–15,16–23), B(32–38, en partie), C(41–48),
et 5 AUTRES blocs plus loin (57–64, 65–72, 89–96, 97–104, 105–112) — tous
`type: abab`, donc tous nommés **la même lettre** par `nommer()`
(`harmonia/phrases4.py:96-124`, comparaison stricte de la chaîne `type`). Sur
9 occurrences repliées ensemble, `folding.py` aurait largement de quoi
empiler (voir l'entrée voisine).

LA CASSE ARRIVE DÈS QU'UNE OCCURRENCE EST GELÉE. `sections_draft` de ce
morceau vient d'une passe où A (8–31) a déjà été fixée comme brique unique de
24 mesures — vérifié en rejouant `remplir_par_algo(chart, occ=[{8,31,"A"}])`
(`harmonia/sections/simulation.py:249`) : les AUTRES occurrences du même
`abab` (0–7, 32–39, 42–49…) ressortent bien groupées ENTRE ELLES sous une
lettre commune (`B`), mais JAMAIS sous `A` — alors que c'est visiblement le
même motif. La cause : une brique gelée entre dans `nommer()` comme UN SEUL
jeton, non décomposé (`geles` dans `merges4`, `simulation.py:262-267`) —
son `type` est la concaténation de TOUTES ses bi-mesures d'un coup
(`"cbcbcbcbcbcb"`, 12 caractères), jamais `"abab"` (4 caractères). La
comparaison de `nommer()` (`u == t`, ou même longueur pour le prime) ne peut
donc PLUS jamais matcher un bloc libre de 4 caractères contre une brique
gelée de 12 — même s'ils sonnent identiques. Et `remplir_par_algo` (:284-297)
aggrave : il ne compare jamais le `type` d'un bloc « algo » à celui d'une
brique gelée, il lui donne juste la prochaine lettre libre de l'alphabet.
**Une lettre humaine gelée devient un trou noir pour la reconnaissance de son
propre motif ailleurs dans le morceau.** Résultat mesuré : la même idée
musicale ressort sous 4 lettres différentes (A, B, D, F/G selon la zone) au
lieu d'une seule, chacune avec trop peu d'occurrences pour replier avec
confiance — c'est la vraie raison pour laquelle C se retrouve seule et
tombe dans le veto « too few gated members » décrit ci-dessous.

Pas encore fixé : il faudrait comparer le `type` d'un bloc libre à la
DÉCOMPOSITION interne de chaque brique gelée (pas à son blob entier), ou
laisser les bri­ques gelées entrer dans `nommer()` sous leur propre période
plutôt qu'en un seul jeton. Les deux touchent l'algo du retour — état de
référence de Louis (2026-08-16, « c'est vraiment top »/`retour.py`) — donc
arbitrage avant d'y toucher.

**Prototype de la lecture cible : `tools.repli_propose <chart>` (2026-09-24).**
Reproduit tout San Francisco Street automatiquement — 4 cellules trouvées
(alpha ×8, beta ×15, gamma ×2, pont ×4), sans frontière de lettre, avec les
2 fins alternatives repérées à l'oreille par Louis. N'utilise PAS
`find_repeats` (sa recherche transposée, calibrée pour un humain qui a déjà
désigné une occurrence réelle, fusionnait tout le morceau en une seule
famille à l'amorçage aveugle) — fenêtres glissantes + vecteurs chord-tone
concaténés (pas moyennés) + seuil 0,90. Diagnostic seul, ne touche à rien ;
sert de base si `nommer()`/`remplir_par_algo` sont un jour corrigés.

**Limite confirmée (2026-09-24, This Love) :** l'outil essaie les cellules
de 4 mesures avant celles de 2, et ne revérifie jamais si une cellule de 4
qu'il vient de trouver est en fait une cellule de 2 déguisée. Sur This
Love B, il rend deux cellules de 4 mesures là où `fold["B"]` (period=2,
cohérence 0,93+, en production) dit une pompe de 2 — exactement ce que le
commentaire de `folding.py` décrit. Vérifié aussi avec `--brut` (le chart
tel qu'affiché + décode indépendant en regard, même commit) : la confiance
brute est uniformément faible (0,35–0,55) sur TOUTE la section B, contre
0,6–0,9+ ailleurs dans le même morceau — un vrai point faible localisé de
musx sur ce passage (harmonie rapide, 2 accords par mesure), pas du bruit
partout. Pas encore fixé : il faudrait retester en 2 après un succès en 4
(`section_tool.choose_granularity` sait déjà faire ça sur UNE sélection —
pas encore branché ici).

## Une section qui n'apparaît qu'une fois, avec une boucle interne jouée EXACTEMENT 2×, ne replie jamais (2026-09-24, OUVERT — symptôme du dessus, chacune indépendante)

San Francisco Street (Sun Rai, `9zEl-FQLI4A`), section C (mesures 41–48,
occurrence unique une fois qu'elle a perdu son identité avec A — voir
l'entrée au-dessus) : `D^7 D^7 E-7 E-7 | D^7 D^7 E-7 E-7` — les 4 dernières
mesures répètent texto les 4 premières. `fold["C"]` dit
`{"period": 4, "reason": "too few gated members"}` : la boucle EST détectée,
mais le repli est refusé quand même. Ce défaut resterait même si l'entrée
au-dessus était corrigée : n'importe quelle lettre à occurrence unique avec
boucle interne 2× le rencontrerait.

LA MÉCANIQUE (`harmonia/folding.py:596-641`). Pour une section à occurrence
unique dont la période interne P est plus courte que la section
(`sous_phrase = P < longueur`), la dernière mesure de l'occurrence est retirée
de la pile par `FIN_SECTION_HORS_PILE` (`:614-624`) — un garde-fou pensé pour
protéger la cadence de fin de CHAQUE passage quand plusieurs occurrences
s'empilent (This Love B, 4 passages). Avec une seule occurrence il n'y a rien
d'autre à empiler contre cette cadence : la retirer prive juste la dernière
position de pile d'une observation. Pour une boucle à 2 répétitions
(longueur == 2×P, le cas le plus simple et le plus fréquent), c'est
mathématiquement garanti d'échouer : sum(gated) = 2×(P−1) + 1 = 2P−1, et le
veto `:640` exige `>= 2×P`. Toute section à occurrence unique, boucle interne
de période P, jouée EXACTEMENT 2 fois, échoue TOUJOURS ce veto — pas un cas
limite, une garantie arithmétique. Ça ne mord que le cas 2 occurrences : à 3+
occurrences internes la dernière position retombe à 2 membres (suffisant), et
à occurrence multiple (plusieurs passages du label dans le morceau) il y a
assez d'autres occurrences pour compenser la perte d'une mesure.

Pas encore diagnostiqué : combien de lettres du corpus tombent dans ce cas
précis (occurrence unique × boucle interne 2×) — un seul exemple confirmé
pour l'instant (San Francisco Street C). Pas encore fixé — c'est un seuil
calibré (`docs/refactor_2026-09/` + l'en-tête de `folding.py` : « toute
refonte doit se comparer » aux valeurs mesurées), Louis arbitre avant qu'on y
touche.

## Déplacer le début de la section 1 n'atteint pas le chart (2026-09-20, RÉSOLU — trois voisins restent ouverts)

Louis : « quand dans l'annotation on redéfinit le début de la section 1, c'est
mal fait : ça n'est pas reflété dans le chart, dans la tête de lecture ».

REPRODUIT dans le navigateur (Playwright, 390 px, `?open=min_X-yIEMduRXk`,
Easy On Me) en appelant les mêmes fonctions que le doigt : on pose « A » sur
les mesures 5–12 alors qu'elle commençait mesure 1, puis « Enregistrer comme
vérité ». Note rendue à l'écran, telle quelle :

    enregistré comme vérité : 8 section(s) · ⚠ chart inchangé :
    Error: {"error":"4 mesures ne sont dans aucune secti…

Le fichier de vérité EST écrit, le chart NE L'EST PAS, et l'avertissement est
tronqué à 70 caractères, en JSON brut, derrière un début de phrase qui dit
« enregistré ». C'est pour ça que ça se lit comme un silence.

LA MÉCANIQUE. Déplacer le début de la première section laisse les mesures
avant elle sans propriétaire. `sectionRuns()` les nomme « ? » ; les deux
chemins d'écriture les JETTENT (`confirmSectionTruth`
`sections_editor.js:680-683`, `writeSectionsToChart` `:606`) ; et
`/api/soudure/valider` refuse — à juste titre — un découpage qui ne couvre
pas le morceau (`routes/sections.py:203-212`). Vérifié en direct :

    POST /api/soudure/valider/min_X-yIEMduRXk  {"sections":[A 5–12]}
    → 400  "55 mesures ne sont dans aucune section (la première est la 1)"

« Valider les sections » ne tombe PAS dedans : `/api/sections/inferer` rend un
découpage complet, le trou est rebouché, le chart est réécrit (vérifié :
A passe à `[[4,11],[22,29],[43,50]]`). Mais `remplir_par_algo` ne sait produire
que des LETTRES — la tête du morceau revient donc en « B », jamais en « intro ».
Le geste survit, mais le chart montre toujours une section qui commence
mesure 1 : ça se lit comme « mon édition a été ignorée ».

LA TÊTE DE LECTURE NE POUVAIT PAS BOUGER, par construction. Deux notions de
« début » cohabitent et ne se parlent pas :

    state/human/marks/<stem>.json   {"bar1": 11.27}   SECONDES  → phase de la
        grille, relue par jobs.bar1_for → bars.layout_bars. SEUL « Caler la
        mesure 1 » y touche, et il relance une analyse complète.
    state/human/sections/<stem>.json  b0/b1            MESURES   → lettres et
        frontières. Lu par le seul outil qui les a écrites, la page /ssm et
        l'export — JAMAIS par le pipeline.

Le seul pont est le navigateur appelant `/api/soudure/valider`, qui réécrit
`chart["sections"]` et jamais `barGrid`. Déplacer une frontière de section ne
peut donc pas déplacer la tête de lecture ni la numérotation des mesures.

CE QUE ÇA N'EXPLIQUE PAS ENCORE, et qui est du même nid :

  * **une ré-analyse efface le découpage validé.** `/api/bar1` et `/api/tempo`
    réécrivent le chart depuis zéro et, avec `SETTINGS.sections = "aucune"`,
    rendent UNE section « A » (`pipeline.py:749-765`). Rien dans `jobs.py` ne
    relit `SETTINGS.sections_dir`. Le docstring de `pipeline.py:758-760` dit
    que « `harmonia.refold` le refait depuis les sections de Louis dès qu'il
    les écrit » : ce n'est PAS câblé — `refold` n'est appelé que depuis
    `routes/sections.py:221`, dans la validation pilotée par le navigateur.
  * **les indices de mesure sont stockés sans leur origine.**
    `routes/sections.py:718-721` écrit `b0/b1` et `n`, jamais le `bar1` en
    secondes contre lequel ils ont été tracés. Un « Caler la mesure 1 »
    postérieur re-phase la grille, et `_sections_known` (`:568-613`) ressert
    les anciens indices sur la nouvelle grille sans décalage ni avertissement.
  * **`GET /api/chart-model` ne pose aucun `Cache-Control`** (contrairement à
    `sections.py:94` et `:134`) et `static/api.js:8` n'ajoute pas de
    casse-cache : le re-fetch de `writeSectionsToChart` (`:613`) peut être
    servi par le cache heuristique du navigateur. Secondaire, mais réel.

CE QUE LOUIS A TRANCHÉ (2026-09-20) : « trou au milieu — queue, trou en tête —
intro », et « déplacer le début de la section 1 ne recale PAS la mesure 1 du
morceau ». Les deux notions de « début » restent donc séparées, et c'est voulu :
« Caler la mesure 1 » est le seul geste qui déplace la grille.

CE QUI A ÉTÉ FAIT. `comblerTrous()` dans `static/screens/sections_editor.js` :
les mesures que personne ne réclame reçoivent un nom avant l'envoi — « intro »
si le trou commence mesure 1, « queue » sinon. `confirmSectionTruth` ne jette
plus les runs « ? », et `writeSectionsToChart` est la seule porte, donc les deux
boutons en profitent.

LE SUFFIXE, ET POURQUOI IL EXISTE. `soudure.sections_pour_chart` groupe sur
`(label, longueur)` : deux trous de même longueur tous deux nommés « queue »
deviendraient UNE section écrite une fois et jouée N fois. Mesuré sur Easy On Me
avec trois trous de 2 mesures — accords réels A♭ A♭ / F D-7 / A♭ F — le chart
écrivait « F D-7 ×3 », deux passages sur trois faux. C'est « under-fold, never
over-fold » : un repli que Louis n'a pas demandé est une affirmation que la
machine n'a pas le droit de faire. Les trous comblés qui entreraient en collision
prennent donc « queue 2 », « queue 3 ». Un seul trou lit « queue », sans suffixe.
Quand Louis écrit « queue » deux fois de sa main (This Love, Don't Know Why),
c'est SON affirmation et elle est respectée telle quelle.

Le message d'échec est aussi devenu lisible : `raisonLisible()` sort la phrase du
serveur au lieu d'afficher `Error: {"error":"4 mesures ne sont dans aucune secti…`.

VÉRIFIÉ DANS LE NAVIGATEUR (Playwright, 390 px), même scénario que le repro :

    avant   enregistré comme vérité : 8 section(s) · ⚠ chart inchangé : Error:…
            chart : A [[0,11]] …                       (le geste perdu)
    après   enregistré comme vérité : 8 section(s) · chart réécrit (9 sections)
            chart : intro [[0,3]] · A [[4,11]] · queue [[12,13]] · B …
            tête de lecture à 20 s → mesure 6, dans A   (avant : dans l'ancienne A)

L'état réel d'Easy On Me a été restauré après le test (chart, vérité, brouillon,
registre des corrections).

CE QUI RESTE OUVERT, et qui est du même nid — les trois points ci-dessus (une
ré-analyse efface le découpage validé ; les indices de mesure sont stockés sans
leur origine ; `GET /api/chart-model` sans `Cache-Control`) ne sont PAS corrigés.

ET UNE QUESTION LAISSÉE À LOUIS : par « Valider les sections », le trou de tête
n'est pas un trou — `remplir_par_algo` le remplit d'une LETTRE (mesuré : « B »
sur les 4 premières mesures d'Easy On Me, une lettre qui ne se rejoue nulle
part). Sa règle « trou en tête = intro » ne s'y applique donc pas encore.
**Résolu le 2026-09-30** : la tête libre sort « intro » (voir l'entrée du
2026-09-30 « Il manque une section avant un A » en haut).

## Sections sans modèle : la grille de 4 mesures, et un échange à arbitrer (2026-09-20)

Après le retrait de SongFormer, les sections viennent de la brique de Louis
plus un remplissage des trous. Trois briques neuves ont été mesurées :
transposition des reprises, nouveauté de Foote repêchée de git (`8d5137b`), et
calage des propositions sur une grille de 4 mesures.

LE FAIT LE PLUS SOLIDE, et il est corpus-level : **180 des 238 frontières de
section que Louis a écrites à la main tombent sur une grille de 4 mesures
(76 %), sur 22 morceaux annotés.** Neuf morceaux sont à 10/10. La PÉRIODE n'est
donc jamais en doute ; c'est la PHASE qui varie, et la fin de son intro la
donne juste sur 13 morceaux sur 16.

LA NOUVEAUTÉ SEULE NE SUFFIT PAS. Sur trois morceaux : 10 frontières justes
sur 27 (±1 mesure), 5 inventées. Ce n'est PAS un problème de seuil — vérifié,
les fausses ne sont pas les plus faibles, elles sont mélangées aux bonnes
(rangs 3, 5 et 7 sur This Love). Sur Stand By Me la nouveauté aux frontières de
Louis vaut 0,9× la médiane du morceau, c'est-à-dire SOUS la moyenne : impasse
prouvée pour le canal harmonique, ce morceau est une boucle de quatre accords
du début à la fin et ses sections diffèrent par le chant et l'arrangement. Même
mur qu'en 2026-08-08 (détecteur voix 0,769 contre harmonique 0,599).

L'ÉCHANGE, mesuré dans le vrai régime d'usage (Louis pose UNE brique, la
machine déduit le reste), 18 morceaux, 186 frontières :

    AUJOURD'HUI   128 justes (69 %)   51 inventées
    AVEC          120 justes (65 %)   24 inventées

**Moitié moins de frontières inventées, contre 8 trouvées en moins.** Ce n'est
pas un gain net, c'est un échange — il attend l'arbitrage de Louis sur
`docs/plots/sections_nouveautes_2026-09-20.html`. Rejouable :
`.venv/bin/python -m tools.sections_nouveautes SORTIE.json`.

HYPOTHÈSE TESTÉE ET RÉFUTÉE, écrite pour qu'elle ne soit pas retentée : les
morceaux qui perdent ne sont PAS ceux qui collent mal à la grille. C'est
l'inverse — ajustement médian à la grille de 93 % chez les 7 qui perdent contre
69 % chez les 11 qui ne perdent rien. Aucun des indicateurs mesurés (ajustement
à la grille, nombre de reprises trouvées, longueur du morceau) n'explique le
signe du delta, qui reste petit partout (−3 à +2). Ne pas inventer de
mécanisme : il n'y en a pas de mesuré.

CE QUI N'EST BRANCHÉ NULLE PART : `harmonia/sections/nouveaute.py` est mergé et
testé (26 tests) mais aucun appelant de production ne l'importe, par décision —
il attend la grille, qui attend une origine de phase fiable. La transposition,
elle, EST en production (`simulation.cherche_brique`) : elle n'a tiré que sur
2 morceaux sur 18 (Sunny, 3 reprises transposées ; T64BgKEL-Sw, 1).

DEUX PIÈGES HÉRITÉS DU CODE REPÊCHÉ, trouvés en le relisant : `KERNEL_HB = 16`
était en DEMI-mesures (l'ancienne SSM était au demi-mesure), donc le reprendre
tel quel sur une SSM par mesure doublait silencieusement la fenêtre — motif
d'erreur nº 1 du CLAUDE.md, évité. Et la couche de pavage (`tuilages`) NE
TRANSFÈRE PAS : 0 à 1 run selon le morceau, `tuilage=True` et `False` rendent
la même chose ; son gain mesuré de +11,7pp l'avait été sur le substrat NNLS.

## SongFormer retiré : plus aucune détection automatique de sections (2026-09-19)

Louis : « quand on prep une nouvelle chanson, la détection de sections pédale
dans la semoule complet, et limite elle fait crasher mon Mac ». Puis, la cause
mesurée : « on vire s'il prend autant de temps ».

CE N'ÉTAIT PAS UNE BOUCLE INFINIE — c'était la machine qui swappait. SongFormer
reçoit le morceau ENTIER d'un coup (sa config met `win_size = hop_size = 420 s`,
voir le `config.json` du dépôt HF), donc sa mémoire grimpe avec le CARRÉ de la
durée. Mesuré sur la machine de Louis (16 Go, 10 cœurs), processus enfant seul :

    141 s de musique    60 s de calcul   pic 3,7 Go            passe
    326 s de musique    JAMAIS FINI (tué à 3 min)
                        swap 0 → 9,5 Go, CPU tombé à 3 %, état « uninterruptible »

Le swap est retombé de 9,5 à 3,9 Go DÈS que l'enfant a été tué — c'est bien lui.
Le CPU à 3 % explique le symptôme : il ne calcule plus, il attend le disque. Et
`DELAI = 1800` lui laissait une demi-heure pour le faire.

LA DISPROPORTION, sur le MÊME fichier de 326 s, caches froids :

    beatthis      5,1 s   pic 0,8 Go    poids   77 Mo   les battues
    musx          5,5 s   pic 1,6 Go    poids   25 Mo   les accords
    SongFormer    ∞       > 16 Go       poids  2,7 Go   SEULEMENT les sections

Il enchâsse deux modèles de fondation audio (MuQ + MusicFM). Tout le coût de
l'analyse tenait dans le seul étage qui nomme les sections.

CE QUI A ÉTÉ FAIT. `SETTINGS.sections = "aucune"` (constante, pas variable
d'environnement). `pipeline.analyze_steps` rend UNE section « A » sur tout le
morceau — la forme du chart brut ; `routes/sections.sections_inferer` remplit
les trous avec `simulation.remplir_par_algo` (l'agglomération des quatre mots,
sans modèle) au lieu de `remplir_par_songformer`, et n'importe plus
`detect_sections`. Vérifié : analyse complète du morceau de 326 s en **5,1 s /
1,43 Go** (caches musx+beats chauds), et la route d'annotation répond en moins
d'une seconde sur un vrai morceau de 71 mesures. 551 tests verts.

CE QUE ÇA COÛTE, et il faut le dire :

  * **plus d'« intro » ni d'« outro » automatiques.** L'algo des quatre mots ne
    sait produire que des lettres — c'est écrit dans `remplir_par_algo`. Seul
    Louis peut écrire ces deux noms-là. C'était la seule chose que SongFormer
    savait faire et que personne d'autre ne sait faire ici.
  * **l'empilement des accords ne se fait plus à l'analyse.** Une seule lettre,
    donc `fold_letter_groups` n'a rien à empiler. Ce n'est PAS perdu :
    `harmonia.refold` le refait depuis les sections de Louis dès qu'il les
    écrit, et c'était déjà la loi (« la vérité terrain doit toujours être le
    chart brut », 2026-08-17). Mais un chart jamais annoté n'a plus ce gain —
    mesuré à 42 accords sur 797, soit 5,3 %, dans `refold`.
  * **le banc `/api/sections/simuler/` est devenu creux.** Il compare encore
    « algo » et « songformer », mais son `auto` vient de `analyze()`, qui ne
    rend plus qu'une section. À retirer ou à recâbler.
  * **les 46 caches de `state/cache/songformer/` sont gardés**, et le module
    `harmonia/sections/songformer.py` n'est pas supprimé : `SETTINGS.sections =
    "songformer"` le rebranche tel quel.

MESURÉ AUSSI, pour qui voudrait le rebrancher un jour : plafonner la fenêtre du
modèle à 150 s fait passer le morceau de 326 s en **112 s / 7,6 Go**, sans
thrashing, et les FRONTIÈRES sont identiques au dixième de seconde près (testé
sur Billie Jean contre son cache 420 s). Ce qui se dégrade, ce sont les NOMS du
dernier tiers : les deux « outro » deviennent « refrain », le « pont » devient
« instrumental » — c'est la vue globale qui permet de dire « là, c'est la fin ».
La demi-précision (bfloat16) est une impasse : MuQ casse en interne,
`Input type (c10::Half) and bias type (c10::BFloat16) should be the same`.

BUG LATENT REPÉRÉ CHEZ EUX, pas la cause d'ici : `modeling_songformer.py`
ligne 152 fait `continue` sans incrémenter `i`, donc boucle à l'infini pour un
morceau dont la durée tombe dans (420,000 ; 420,043] s. Assez rare pour ne
jamais s'être produit, assez réel pour être écrit.

## Tabs branchés : source de chart alternative + marque de doute (2026-09-18)

Louis : « branche-nous ça comme façon alternative de choper des charts, et tu
vas t'en servir pour flagger si on a fait des détections d'accords douteux ».

L'import de tablature avait été abandonné au refactor pour une raison écrite
dans `routes/irealb.py` : « une tablature n'a ni mesures ni temps, en faire un
chart demande l'alignement audio, pas construit ». L'alignement existe
(`tab_align.poser_tout`), donc la raison est tombée.

TROIS ROUTES, un seul calcul derrière (`harmonia/integrations/tab_chart.py`) :

    POST /api/tab-doutes   va chercher un tab, marque les accords contestés,
                           met en cache dans state/cache/tab_doutes/
    GET  /api/tab-doutes   relit le cache, ne recalcule jamais
    POST /api/tab-import   écrit un NOUVEAU chart `tab_<stem>` — jamais
                           d'écrasement, les annotations sont indexées par stem

UI : deux lignes dans la feuille Outils d'Annotate, et en mode annotate un
ANNEAU sous l'accord contesté + la mention « tab X ». L'anneau passe avant le
point de faible confiance (un désaccord entre deux sources en dit plus qu'un
score bas d'une seule) et jamais avant le tiret vert d'un accord confirmé.

MESURÉ en vrai sur :7772 : Grenade 6 accords douteux sur 68 écrits, tab 4,86★,
requête « Bruno Mars Grenade » construite toute seule depuis chart_meta + le
titre nettoyé de « official music video ». Chart importé : 100 mesures,
8 sections, 44 mesures à écrire contre 68 au nôtre, forme
`A B C C D E E F E B C C D G B C C D E E H`.

CE QUI EST REFUSÉ, et c'est le cœur : un accord CONFIRMÉ à la main n'est jamais
marqué ; un N.C. non plus ; un passage que le tab ne couvre pas n'est compté ni
pour ni contre ; et il faut que le tab conteste au moins la MOITIÉ des passages
d'un accord écrit — contester un passage sur quatre est du bruit d'alignement.
Deux gravités : « fondamentale » (fondamentale ou basse différente — la cible
du projet est la basse qui SONNE) et « couleur » (même fondamentale et basse,
autre famille : le `D°` du tab contre notre `Dø`).

CE QUE ÇA NE DIT PAS : qui a raison. Nos charts battent les tabs UG
(2026-08-05) ; la marque dit où ÉCOUTER, pas quoi écrire.

## Tabs : la basse était dans le tab, et la forme au rasoir d'Occam (2026-09-18)

Louis : « alors le tab a bien le slash chord, juste on le choppe pas, et bon
avec cette vérité-là en tête et la règle qu'un pattern en général se répète,
tu peux déduire la structure finale de la grille → rasoir d'Occam ».

LA BASSE. Vrai, et vérifié : le tab de This Love écrit `E7/G#` huit fois, soit
`G7/B` transposé de +3 — exactement la basse que notre chart écrit sous son
`G/B`. `lire_accord` la lisait déjà et `compresser` la gardait ; c'est
l'affichage et la comparaison qui la jetaient, et la transposition ne
l'appliquait pas. Corrigé. Effet sur la comparaison de This Love : les
19 temps `G/B → G7` deviennent `G/B → G7/B`, une différence de COULEUR et plus
de basse. Nouveau chiffre : 85,5 % de fondamentale ET basse identiques. Et
6 temps où le TAB a une basse que nous n'avons pas (`G` chez nous, `G7/B` chez
lui).

LA FORME. `harmonia/integrations/tab_structure.py` : la suite de mesures
posée est SYMBOLIQUE et exacte, donc la question redevient celle d'un
compresseur — quelle est la plus courte écriture qui l'explique. Résultat :

    Grenade    A B C C D E E F E B C C D G B C C D E E H
               44 mesures à écrire, 8 sections, 21 posées
               `B C C D` est le refrain, joué trois fois ; `E E` le couplet ;
               `G` le pont ; `E` et `F` ne diffèrent que d'UNE mesure
    This Love  A A A B C A D A E C F E C E C E C
               36 mesures à écrire, 6 sections, 17 posées
               `A` = G7/B | Cm | Fm | D°, la boucle de 4 mesures, 5 fois

TROIS ERREURS MESURÉES EN ROUTE, consignées pour ne pas les refaire.

1. **Compter d'abord les sections DIFFÉRENTES ne marche pas.** Ça semble être
   ce que demande « le moins de sections différentes », mais la façon la moins
   chère d'avoir peu de lettres est d'en faire de très longues : This Love
   sortait « A B C C » en blocs de 32 mesures. C'est la LONGUEUR ÉCRITE qui
   fait payer la répétition ; le peu de lettres en découle.
2. **Une mesure vaut ce qui y SONNE, pas ce qui y commence.** Le premier jet
   prenait les accords démarrant dans la mesure : la même boucle sortait
   `· Cm Fm D°` puis `G7/B Cm Fm D°` selon qu'un accord démarrait une fraction
   de temps avant la barre ou après, et la répétition était perdue. 72 → 64
   mesures écrites en corrigeant.
3. **Le choix de LZ est trop permissif.** Rendre une reprise gratuite dès que
   le bloc est APPARU plus tôt (et pas seulement s'il a été POSÉ) tombe à
   32 mesures écrites — mais sort dix lettres pour quatre payées, parce qu'un
   bloc devient gratuit en citant un passage à cheval sur deux sections. Aucun
   musicien n'écrit ça. On compte le CHART qu'on écrirait, donc une reprise
   n'est gratuite que si la même section a déjà été posée — au prix d'un coût
   qui dépend du chemin, traité par un faisceau de 24.

CE QUE ÇA NE RÉSOUT PAS : l'égalité est exacte, donc un couplet dont la
dernière mesure change compte pour une section neuve. `variantes()` signale
les lettres qui ne diffèrent que par leur fin (Grenade : `E` et `F`, une
mesure) sans les fondre — « under-fold, never over-fold ».

## Tabs : la comparaison qui compte, contre NOTRE chart (2026-09-18)

Louis : « alors pas bon, compare maintenant au vrai chart qu'on ferait pour
cette chanson, et compare les alignements ».

Il a raison que l'accord-avec-musx ne prouvait rien : musx est la matière
première des deux côtés. `tools/tabs_vs_chart.py` compare au chart servi,
déplié (le chart est REPLIÉ — 54 mesures écrites pour 100 jouées sur Grenade,
via `barSpans`), les deux vocabulaires ramenés aux cinq familles.

    This Love   même accord 72,3 %   même fondamentale 88,9 %
                100 des 113 changements au MÊME temps
    Grenade     même accord 93,8 %   même fondamentale 94,4 %
                100 des 130 changements au même temps

LECTURE. L'ALIGNEMENT est bon : 100 changements sur 113 tombent au même temps
sur This Love. Ce qui diffère, ce sont les ACCORDS, et 17 des 28 points
d'écart ne sont que de l'orthographe du tab :

    Dø → D°    20 temps   le ii de do mineur, demi-diminué chez nous
    G/B → G7   19 temps   même fonction, la basse en plus chez nous
    Cm → Fm     8 temps   vrai désaccord
    Bb → Eb     5 temps   vrai désaccord

Les deux premières familles disent la même chose : **notre chart est plus
riche que le tab**, pas l'inverse — ce qui va dans le sens de la règle du
2026-08-05 (« nos charts battent les tabs UG »). Le tab reste utile pour
l'ORDRE et la STRUCTURE, pas comme vérité d'accord.

PIÈGE DE MESURE corrigé en route : apparier un accord du tab au PREMIER accord
du chart qui le chevauche donnait 25 « fondamentales différentes » ; apparier
à celui qui le chevauche le PLUS en donne 13. La moitié des erreurs était dans
la comparaison, pas dans le placement.

RESTE OUVERT : sur This Love le tab tient un accord là où nous changeons
(mesures 13, 16, 17, 30) et double là où nous tenons (mesure 29). C'est le
défaut « une boucle écrite une fois, jouée quatre » — l'alignement ne sait pas
rejouer un bloc.

## Tabs : le modèle de placement, corrigé par Louis (2026-09-18, RÉSOLU)

Louis, en lisant la page d'explication : « le tab n'a aucune mesure ?? on est
d'accord on matche bien toute la longueur des accords consécutifs à toute la
longueur du chart, et ensuite la seule question qui permet de maximiser la
log-proba totale c'est comment je pose mes accords du tab à l'intérieur, sans
jamais en changer l'ordre (un accord toujours après un autre) ».

Deux erreurs de ma part, les deux réelles.

1. **J'ai parlé de « la mesure 18 du tab ».** Un tab n'a pas de mesures. Le
   numéro venait de MON alignement. Ce que le tab dit est ordinal : quel
   accord ouvre la section.
2. **`aligner` ancrait le début et laissait la fin flotter**, et jetait les
   accords qu'il n'arrivait pas à caser (39 sur 115 sur This Love) parce
   qu'une mesure ne porte qu'un accord. Le chiffre était affiché sur la page
   depuis le début — « cases utilisées 76/115 » — et je n'en ai pas tiré la
   conclusion.

`poser_tout` applique son modèle : unité = le TEMPS, les deux bouts ancrés,
aucun accord jeté, ordre préservé. Mesuré sur la même grille de temps, contre
le top-1 de musx :

    This Love   61,1 % (76/115 accords posés)  →  80,7 % (115/115)
    Grenade     81,7 % (73/105)                →  96,7 % (105/105)

Deux priors, tous deux tirés du tab seul : la DURÉE annoncée (un accord écrit
sur trois lignes dure trois fois plus) et la PLACE (un accord change sur un
temps fort). Le second à 0,4 nat coûte 0,3 point d'accord et fait passer le
placement sur temps fort de 93 % à 97 %.

FAUSSE ALERTE, consignée. Le prior de temps fort semblait d'abord inutilisable :
5 % des changements « sur le temps fort » sur This Love contre 63 % sur
Grenade. C'était ma MESURE qui était fausse — This Love a une levée d'un temps,
donc ses premiers temps de mesure sont les indices 1, 5, 9…, pas 0, 4, 8. Avec
la bonne origine les deux morceaux se comportent pareil (62 % / 63 % sur le
premier temps, 93 % / 92 % sur un temps fort). `_cout_depart` prend maintenant
une `origine`.

## Tabs : frontières de section décalées d'une mesure (2026-09-18, OUVERT — arbitrage)

Louis, sur la page POC : « le découpage de sections est quasi bon mais pas bon,
il est bon à une mesure ou deux près ».

MESURÉ. Sur Grenade, six frontières sur neuf sont en retard d'exactement +1
mesure par rapport à notre chart. Le signe est constant, donc ce n'est pas du
bruit. Les cinq frontières concernées tombent toutes sur le même accord :

    m15 Dm   m16 Bb   m17 A  ← notre chart ouvre ici   m18 Dm  ← le tab ouvre ici

Le tab termine chaque section sur le A (la dominante) et ouvre la suivante sur
le Dm (la tonique) ; notre chart ouvre la section sur le A. **Ce n'est pas un
bug d'alignement : c'est un désaccord de lecture**, et il attend l'oreille de
Louis. Page : /reports/tabs_chemin.html, étape 6, chaque frontière s'écoute.

Sur This Love l'écart est plus gros (−4 puis −3) et la cause est encore
ailleurs : le tab appelle « Intro » le 1er passage de la boucle de 4 accords et
« Verse 1 » les deux suivants, là où notre chart appelle « intro » les deux
premiers passages. L'alignement, lui, est juste — les mesures 1 à 12 tombent
une pour une sur les cases 0 à 11.

HYPOTHÈSE TESTÉE ET REJETÉE. 25 % des mesures de Grenade voient le chemin
global contredire le meilleur coup local ; sur cinq d'entre elles on écrit Bb
là où musx dit Dm à 0,50, et Dm est la case qui précède Bb. D'où l'idée :
quand une mesure avale deux accords du tab, écrire le PREMIER (l'accord du
premier temps) plutôt que le dernier. Testé deux fois —

  * en réétiquetant après coup : 95 % → 73 % (Grenade), 79 % → 43 % (This Love).
    Test invalide : la PD avait optimisé la case d'arrivée, je réétiquetais un
    accord jamais évalué ;
  * en notant la mesure sur le premier accord avalé DANS la PD, ce qui est le
    vrai test : 95 % → 94 %, 79 % → 74 %.

Donc **le modèle actuel est meilleur**, et ces cinq mesures sont la PD qui
échange une petite perte locale contre un meilleur enchaînement. Consigné pour
que l'idée ne soit pas retentée dans un mois.

CE QUI RESTE OUVERT : un tab écrit une boucle une fois là où le disque la joue
deux ou quatre fois. L'alignement ne sait pas « rejouer un bloc », seulement
tenir une case plus longtemps — il écrit `Am Am F F` au lieu de `Am F Am F`.
C'est la même loi que l'algo du retour (une section = une mini-boucle jouée
≥ 2 fois) et c'est le prochain vrai levier.

## Ouvert

### Fusion de lettres : la règle est prête et mesurée, elle reste ÉTEINTE

Le remplacement du comparateur est fait et testé (`folding.merge_similar_letters`
compare désormais les ACCORDS DÉCODÉS, 21 tests) ; `SETTINGS.merge_letters`
reste à `False`. Pourquoi, et ce qu'il faut pour l'allumer :

LA RÈGLE EST BONNE DANS LE RÉGIME OÙ ELLE A ÉTÉ MESURÉE. Sur les lettres que
portent les charts EN CACHE, les six morceaux arbitrés sortent tous justes
(Fallin' 6→1, Stand By Me A+C, Sunny A+B et C+D, Every Breath A+A′+A″,
Yesterday et Let It Be intacts), Autumn Leaves et Built For Love d'iReal sont
intacts, et le corpus passe de 435 à 279 lettres.

ELLE N'EST PAS VALIDÉE DANS LE RÉGIME OÙ ELLE SERVIRAIT. La fusion vit à
l'ANALYSE. Une ré-analyse fraîche (caches chauds, ~1 s par morceau) donne un
découpage amont très différent de ce que portent les charts en cache — le
détecteur a bougé le 2026-09-18 même. Contrôle merge ON/OFF sur les six, à
code égal :

    morceau         en cache            ré-analyse SANS     ré-analyse AVEC
    Fallin'         A B A′ B′ C B″      A B C D C′ C″       A A′
    Stand By Me     A C B′              A′ A″ B A C B′      A′ A″ B A B′
    Sunny           A B C D             A B C D C′          A B C D C′   (aucun effet)
    Yesterday       A B                 A B B′              A B B′       (intact ✔)
    Let It Be       A B B′ C bridge     A B B′ C C′         A B B′ A′    (RÉGRESSION)

Sur Let It Be ré-analysé, `A` absorbe `C` à 1.00 — c'est exactement le morceau
qui doit rester séparé, et c'est lui qui avait déjà tué la version chroma.

CE QU'IL FAUT POUR ALLUMER, dans l'ordre :
 1. refaire l'arbitrage sur des charts RÉ-ANALYSÉS, pas sur le cache : c'est le
    piège n°6 de CLAUDE.md (un composant a bougé, tout l'aval a bougé avec) ;
 2. comprendre pourquoi le détecteur segmente aujourd'hui bien plus finement
    qu'au moment où le cache a été cuit (Yesterday gagne un `B′`, Stand By Me
    passe de 3 lettres à 6) — c'est peut-être ÇA le vrai problème, et la
    fusion n'en serait que le pansement ;
 3. une page avant/après sur les charts ré-analysés, que Louis arbitre.

DEUX CHANGEMENTS REPRIS EN ARRIÈRE le même jour, pour la même raison (mesurés
dans le régime du cache, pas dans celui de la ré-analyse) : `_merge_coupe`
acceptant un bloc PLUS LONG qui n'est que de la répétition (sans lui, Fallin'
garde trois blocs pour une boucle de 2 mesures), et le même `_merge_coupe`
comparant à la granularité (fondamentale, famille) plutôt qu'à l'accord exact
(sans lui, un seul `B-7` au lieu d'un `B-` suffit à déclarer deux passages
différents). `folding._barsig_famille` est écrit et documenté, prêt à servir.


### `merge_letters` mesuré : il répare 5 morceaux sur 6 et aucun seuil ne sauve le 6e

Six morceaux analysés à la main le 2026-09-18 (Fallin', Stand By Me, Sunny,
Every Breath You Take, Yesterday, Let It Be), page
`docs/plots/sections_6morceaux.html` — chaque lettre y est écoutable.

Le levier existant (`SETTINGS.merge_letters` + `MERGE_LETTERS_COS = 0.93`,
centroïde des vecteurs de mesure) fait ce qu'il faut sur Fallin', Stand By Me,
Sunny et Every Breath, et **casse Let It Be** : couplet ↔ refrain sortent à
0.970, donc ils fusionneraient. Et aucun seuil ne répare les six —

    fusionner Fallin'          exige  ≤ 0.961   (sa paire la plus basse)
    garder Let It Be intact    exige  > 0.967   (couplet ↔ refrain)

DIAGNOSTIC, pas simple constat. Deux défauts du substrat, pas du seuil :
un centroïde est un SAC de notes (le couplet `C G | A- F | C G | F C` et le
refrain `A- G | F C | C G | F C` ont le même sac), et la chroma NNLS porte la
PRODUCTION en plus de l'harmonie (voix, ad-libs, mix), donc deux passages aux
mêmes accords mais à l'arrangement différent s'éloignent.

RÉFUTÉ EN CHEMIN : rendre la mesure sensible à l'ordre en comparant les
SÉQUENCES de chroma au lieu de leur moyenne. Mesuré, échoue aussi — Fallin'
tombe à 0.839 quand Let It Be reste à 0.864, l'ordre est encore inversé. Ce
n'est donc pas la moyenne le problème, c'est la chroma.

LA PISTE QUI SÉPARE, mesurée sur les six : poser la même question aux ACCORDS
DÉCODÉS (une mesure = ses fondamentales, concordance cyclique sur la plus
courte des deux). Fallin' descend à 0.75 au pire, Let It Be couplet↔refrain
tombe à 0.50 — **un écart de 0.25 au lieu de −0.009**. Yesterday reste à 0.25,
Every Breath sépare son pont à 0.00 et fusionne A/A′ à 0.88. Aucun audio n'est
nécessaire : tout est déjà dans le chart. C'est la « brique chaîne d'accords »
de la mémoire du projet.

CE QUE ÇA NE RÉGLERAIT PAS ENCORE : une lettre trouée de N.C. (Stand By Me
A↔C tombe à 0.67, sous le seuil, parce que les mesures muettes raccourcissent
la séquence) et deux lettres qui ne partagent qu'un PRÉFIXE (Sunny A/B à 0.38,
Every Breath A↔A″ à 0.57) — ces deux-là sont des 1re/2e fins, pas des
fusions, et relèvent de `_ireal_endings`, qui ne regarde jamais à travers deux
lettres.

LES AUTRES BOUTONS repérés en chemin, chacun avec le morceau qui en a besoin :
le groupage `(lettre, longueur)` de `minimal_fold` (Fallin' en 2/4/6/8/9
mesures, Every Breath en 8/10/7) ; `_merge_coupe` qui refuse un bloc PLUS LONG
que l'hôte (le couplet de 10 mesures d'Every Breath ne peut jamais rejoindre
celui de 8) ; la finesse de `_barsig`, pour qui `B-`, `B-7` et `E-/B` sont
trois musiques ; l'absence de longueur minimale (Let It Be écrit trois
« sections » de 2 mesures) ; l'absence de transposition (Sunny C/D = A/B un
demi-ton plus haut).


### Une partie muette n'est plus une section — ce qu'il reste à voir

Louis, 2026-09-18 : « une partie avec que des NC ne peut jamais être une
section, c'est toujours la queue d'une section ». La règle est appliquée à
l'AFFICHAGE : la feuille de lecture ne l'écrit pas, la bande de forme la garde
en trou à sa vraie largeur, Annotate et Analyse gardent ses mesures (c'est là
qu'on corrige un N.C. faux).

Ce qui reste ouvert est EN AMONT : le détecteur continue de lui donner une
lettre, un `id` et une entrée dans `sections`. Si « c'est toujours la queue
d'une section », la bonne place de ces mesures est la section précédente — pas
une section à part qu'on masque ensuite. Tant que ce n'est pas fait, une
lettre peut encore être « consommée » par une plage muette (`B′` sur Stand By
Me), et le renommage A/B/C compte une section qui ne s'écrit jamais.


### RÉSOLU 2026-09-18 — la tête de lecture ne suivait qu'un passage sur N

Louis : « la case highlighted ne suit plus à un moment ». Mesuré : sur
*Don't Know Why*, **92 s sur 183** sans aucune mesure allumée — et **27 charts
sur 85** touchés, tous réécrits dans le même lot.

`soudure.sections_pour_chart` construisait `barSpans` depuis la seule passe
représentative (`for b in range(b0, b1+1)` → une fenêtre par rangée), alors
que `reps`, `spans` et `barRanges` listaient bien les N occurrences. La case
suivait donc le premier passage puis s'éteignait pour tous les suivants.
`tools.golden --publish` passe par cette fonction : republier la bibliothèque
DÉTRUISAIT la carte de lecture de chaque chart replié.

Ce n'était pas visible depuis le code : `folding.minimal_fold` (le chemin
pipeline) fait la bonne chose depuis toujours, et les deux fonctions écrivent
le même chart sans que rien ne les compare. `tests/test_carte_de_lecture.py`
ajoute cette comparaison — c'est le test qui manquait, plus que le correctif.

Correction : `soudure._carte_de_lecture()`, même loi proportionnelle que
`minimal_fold`. Les 27 charts en cache ont été réparés en place (sauvegarde
dans `state/cache/charts_avant_carte/`, accords vérifiés identiques). Zéro
trou > 1 s sur les 85 après coup.

**Ce que ça ne résout pas** : un passage plus court que le bloc écrit répartit
ses mesures (rangées 0, 2, 4… pour 4 mesures sur 8) — la case saute alors des
rangées pendant ce passage. C'est la loi arbitrée de `minimal_fold`, gardée
telle quelle pour que les deux chemins s'accordent ; personne n'a arbitré si
c'est la bonne lecture.


### La liste blanche a mangé un QUATRIÈME champ — dans `buildIReal` cette fois

Le piège déjà décrit plus bas (`chart.js` recopie les accords par une liste de
champs EN DUR) a un second exemplaire, dans la même fonction de rendu :
`buildIReal` remappe chaque MESURE avec sa propre liste en dur
(`sec, secId, secFirst, reps, ending, endingFirst, …`). Le champ `endingOwn`
ajouté le 2026-09-18 y a été oublié : la fin alternée s'écrivait en colonne 1
au lieu de s'aligner sous la classique, et sa rangée gardait son quadrillage.
Aucune erreur, aucun test rouge — juste un chart faux, trouvé en sondant le
DOM. Quatrième occurrence (`sug`, `sugBass`, `casc`, `endingOwn`).

Les deux listes se renversent de la même façon (copier l'objet, écraser ce
qu'on recalcule) et le pendant Python l'a déjà fait. Ce qui manque des deux
côtés JS est un test : c'est un chemin de rendu chaud qui n'en a aucun.


### Deux accords larges dans une mesure débordent de leur cellule

En notation NORMALE, une mesure à deux accords aux étiquettes longues sort de
sa cellule à 390 px : `A♭maj7 B♭maj7` dépasse de 35 px, `B♭maj7 Cmaj7` de 21,
`D♭9 G♭maj7` de 14, `B♭7sus4 A♭/B♭` de 10. Mesuré le 2026-09-17 sur 4 des 10
morceaux d'un tirage au hasard, et identique AVANT le lot « treillis » — c'est
un défaut ancien, pas une régression. L'écriture COMPACTE (`glyphTight`, qui
tucke la qualité sous l'épaule de la lettre) ne l'a pas : c'est exactement le
problème qu'elle a été écrite pour résoudre, mais elle n'est pas le défaut.

La sortie n'est pas d'agrandir la cellule (elle fait un quart de rangée par
construction) ni de rapetisser le glyphe (SZ2 a déjà été arbitré contre
iRealB). Piste la plus probable : faire retomber une mesure à deux accords
larges sur le calage serré, sans changer la taille.

### `test_minimal_fold_separe_les_longueurs_dune_meme_lettre` est rouge

`tests/test_songformer_sections.py` — rouge au 2026-09-17, indépendamment du
frontend (vérifié en remisant les changements JS). 375 autres tests passent.


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


### RÉSOLU 2026-09-18 (3) — la demi-barre de 2/4 était effacée

Louis, sur *Virtual Insanity* : « il y a un passage à la fin qui ne fait que
2 temps, qui est noté comme du 2/4 ; en fait c'est une demi-barre et puis ça
reprend, très jamiroquai, et il faut le noter ».

iReal change de chiffrage EN COURS de morceau. On ne lisait que le premier :
toutes les mesures duraient donc pareil, et **701 mesures dans 101 des 2207
standards** étaient écrites à la mauvaise longueur. Sur ce morceau, trois
demi-barres — et le repli fusionnait la fin de 2 temps avec la fin de 4,
c'est-à-dire qu'il effaçait exactement le trait qui fait le morceau.

Ce qui a changé :
* chaque mesure porte ses PULSATIONS (`_Bar.meter` → `beats`), et la grille se
  construit en les ACCUMULANT — jamais en multipliant un index par une durée
  fixe. `barGrid` est désormais la seule source des temps, ici comme en aval ;
  `beatTimes` n'est plus `nBars × bpb` (une demi-barre n'apporte que 2 temps).
* la signature de repliement inclut les pulsations : une mesure de 2 temps
  n'est plus « la même » qu'une de 4, même avec les mêmes accords.
* une queue qui ne change QUE de durée compte comme une vraie 2e fin —
  l'exigence de `folding._ireal_endings` (« changer de fondamentale ») est
  élargie à la durée, sinon le cas de Louis passait à travers.
* le modèle porte `sections[].meters`, dans le même ordre d'émission que
  `barSpans` (tronc puis queue de chaque variant), et `chart.js` écrit le
  chiffrage DANS la mesure qui change, à gauche, en poussant l'accord — comme
  iReal. Une grille en 4/4 du début à la fin ne porte aucune marque.

Vérifié sur les 2207 standards : `somme(reps × longueur) == nBars`, grille
strictement croissante, accords contenus dans leur propre mesure, et
`len(meters) == len(barSpans)` partout.

### RÉSOLU 2026-09-18 (2) — un chart iReal importé ne se repliait pas

Louis, sur *Virtual Insanity* : « il n'y a pas un repliement comme je
l'attendrais des sections ». Trois défauts empilés, trouvés en comparant
notre lecture à la chaîne brute d'iReal.

1. **Aucun repliement.** iReal écrit un chart LINÉAIRE : son refrain `*B` est
   réécrit en entier à chacune de ses quatre reprises. On les recopiait telles
   quelles — quatre blocs « B » l'un sous l'autre. Harmonia écrit la musique
   une fois et compte les passages : `_fold_blocks` fusionne les blocs
   identiques (`B ×6` sur ce morceau) et `_lay_out_time` pose leurs passages
   là où ils se jouent. 391 blocs du corpus se replient désormais.
2. **Les blocs sans repère volaient les lettres d'iReal.** Ils prenaient A, B,
   C dans l'ordre et tombaient pile sur les `*A`, `*B`, `*C` du
   transcripteur : le chart affichait deux « A » qui n'avaient pas les mêmes
   accords, et un « B » de 8 mesures à côté du vrai `*B` de 4. On réserve
   maintenant les lettres d'iReal d'abord ; deux blocs sous un même repère
   mais avec une musique différente prennent un prime (A, A′), comme
   `folding._ireal_cascade`. Vérifié : **0 étiquette en double sur les 2207
   standards.**
3. **`W/G♭` était lu comme un accord de sol♭ majeur.** C'est en fait « le même
   accord, une autre basse » — la basse qui marche sous un accord tenu
   (`E♭m7 · /G♭ · A♭m11 · /B♭`). **877 occurrences dans 154 des 2207
   standards**, *Round Midnight* compris : on écrivait donc des accords que
   personne n'avait écrits. Le modèle porte l'accord complet, le rendu n'écrit
   que `/G♭` (`ch.pedal` → `kit.pedalGlyph`) — comme iReal, et c'est aussi ce
   qui fait tenir quatre cases dans une mesure.

Au passage : le chiffrage du chart est désormais le PREMIER rencontré, pas le
dernier (*Virtual Insanity* est en 4/4 avec des mesures de 2/4 de passage — la
grille entière pouvait basculer sur une mesure de transition).

**`ch.pedal` est le QUATRIÈME champ à passer par la liste blanche de
`loadModel`** (après `sug`, `sugBass`, `casc`). Voir la section dédiée
ci-dessus : la dette est toujours là, elle ne se plaint jamais.

### Ouvert — l'intro à accords est cachée sur 44 charts (arbitrage)

En lecture, `chart.js` réduit TOUTE section nommée « intro » à une bande
« intro … » d'une ligne. L'intention écrite dans le code (Louis, 2026-08-08)
dit pourtant « the CHORD-LESS leading intro » : le test, lui, ne regarde que
le NOM. **44 des charts de la bibliothèque ont un intro qui porte de vrais
accords et qui n'est donc jamais imprimé.**

L'import iReal contourne : un `*i` qui porte des accords prend une lettre au
lieu du nom « intro ». Corriger la règle elle-même (n'escamoter que les
intros sans accord) changerait ces 44 charts d'un coup — c'est un arbitrage
de Louis, pas une correction à faire en marge.

### TRANCHÉ 2026-09-18 — les lettres suivent l'ordre de jeu

Louis : « l'ordre des sections c'est TOUJOURS ABCDE… ». On n'hérite donc plus
des repères d'iReal — ils ne sortent pas dans l'ordre (sur *Virtual Insanity*
son `*A` arrive en 3e position, son `*B` en 5e, entre deux passages qu'il ne
nomme pas). Les lettres sont réattribuées A, B, C… à la première apparition,
et une section qui revient reprend la sienne : la forme se lit
`A B C×2 D E×2 F×2 E×2 G×2 H×2 I×2 E×2 J`.

Conséquence assumée : le repliement ne regarde plus QUE la musique. 131 des
2207 standards nomment `*A` et `*C` deux passages rigoureusement identiques ;
les garder séparés aurait affiché deux fois la même musique sous deux lettres.

Corrigé dans la foulée : `formTag` (`chart.js`) ne reconnaissait que `A-H`
comme une lettre — au-delà, le bandeau de forme écrivait la lettre en bas de
casse, italique et bordeaux, à côté des vraies. Exactement le défaut que son
propre commentaire décrit sur *Bora Bora*. Deux charts de la bibliothèque
étaient concernés.

### Ouvert — lisibilité d'un chart iReal dense (arbitrage)

*Virtual Insanity* tient sur une page en écriture COMPACTE et déborde en
NORMALE — c'est le défaut « deux accords larges » ci-dessus, pas une
régression de l'import. Deux leviers de repliement supplémentaires ont été
mesurés sur les 2207 standards, sans être implémentés :

| levier | sections écrites | charts touchés |
| --- | --- | --- |
| aujourd'hui | 7315 | — |
| tolérer la 7e/l'extension pour DÉCIDER que c'est la même section | 7102 (−213) | 204 |
| + absorber un passage coupé (début exact d'un plus long) | 6611 (−491) | 380 |

Sur *Virtual Insanity* : 10 sections → 8. Le premier levier a un coût réel —
il faut choisir UNE écriture pour un accord que le transcripteur écrit de deux
façons (`Cø` / `Cø7`, `B△` / `B△7`, `E♭m7` / `E♭m9` sur ce morceau), donc le
chart n'écrit plus exactement ce qu'iReal écrit. À arbitrer par Louis.

Ce qui reste VRAIMENT illisible sur ce morceau, c'est la mesure à quatre cases
du refrain (`E♭m7 · /G♭ · A♭m11 · /B♭`). iReal la résout en suspendant la
basse SOUS l'accord précédent, sans lui donner de largeur — ce que
`glyphTight` sait déjà faire pour une basse slash, et que `pedalGlyph` ne fait
pas encore.

### RÉSOLU 2026-09-18 — la source « iReal Pro » ne pouvait ouvrir aucun morceau

L'onglet iReal de la recherche cherchait bien (`/api/irealb-search`, en place
depuis les sprints 11-14) mais n'importait rien : `search.js` appelait un
`importIrealChart` **qui n'a jamais existé**, et la carte d'un résultat iReal
retombait dans la branche YouTube — elle partait analyser
`watch?v=undefined`. La moitié « import » du portage avait été laissée de côté
(`irealb_fetcher.py` le dit dans sa propre en-tête : « sans sa seconde
moitié : la conversion irealb:// → ChordChart »).

Ajouté : `harmonia/integrations/irealb_import.py` (lecture de la grille) +
`POST /api/irealb-import` + le geste dans l'écran de recherche. Vérifié sur
les **2207 standards** du corpus iReal : 0 échec de lecture, médiane 34
mesures. Tests : `tests/test_irealb_import.py` (la grille),
`tests/test_irealb_import_route.py` (la chaîne serveur, bout en bout).

**Pourquoi un parseur maison plutôt que `pyRealParser.measures_as_strings`** :
celui-ci APLATIT — il efface les repères `*A`, déroule les `{…}` et fusionne
les 1re/2e fins, c'est-à-dire exactement ce qu'on veut garder. On ne lui
emprunte que le désembrouillage et les métadonnées.

**Ce que l'import NE fait PAS** (à savoir avant de s'y fier) :
* **Pas de D.C. / D.S. al Coda.** Les repères `S` et `Q` sont ignorés : la
  grille écrit ce qui est écrit, sans dérouler les sauts. Sur un morceau à
  coda, la section finale peut être découpée d'une façon qui n'est pas celle
  d'iReal.
* **Pas d'audio, donc pas de tempo mesuré.** Les temps sont une grille
  NOMINALE à 120 BPM (`irealb_import.NOMINAL_BPM`), là seulement pour que les
  empans soient monotones. Le transport et l'onglet Practise sont masqués sur
  un chart sans audio ; les recoller à un enregistrement est un autre chantier
  (c'est le problème d'alignement chart↔audio, déjà ouvert).
* **Le placement dans la mesure est déduit, pas lu.** Les barres obliques `p`
  d'iReal sont respectées (`C7 p p F7` → temps 1 et 4) ; sans elles, les
  accords d'une mesure sont répartis également. iReal encode la position réelle
  dans un compte de cellules que ce parseur n'exploite pas.
* **L'EXPORT, lui, écrit tout en dièses.** `irealb_export.NOTE_SHARP` ne connaît
  pas les bémols : un B♭ de *Autumn Leaves* repart en A♯ dans iReal Pro. Même
  hauteur, mauvaise orthographe. Défaut ancien de l'exportateur, pas de
  l'import — épinglé dans
  `tests/test_irealb_import_route.py::test_l_aller_retour_avec_l_export_garde_les_accords`.

Au passage, 31 queues d'accord iReal ont été ajoutées à `TOK` (`ui/kit.js`) —
3,9 % des accords du corpus les portent, et sans clé elles s'écrivaient avec un
`b` et un `#` d'ASCII au milieu d'une grille qui écrit ♭ et ♯ partout ailleurs.
Mesuré : ces ajouts ne changent AUCUN débordement (le défaut « deux accords
larges » ci-dessus est identique avec et sans).

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
