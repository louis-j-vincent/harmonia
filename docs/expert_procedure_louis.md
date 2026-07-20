# La procédure d'oreille de Louis — spec de référence du modèle

*Interview 2026-07-20 (en cours). Chaque réponse est mappée au module qu'elle
définit. Ce doc est LA référence pour l'architecture d'inférence : quand un
choix de design se pose, on encode ce que fait l'oreille experte.*

## A. Ordre des opérations
**A1.** Tonalité + tempo en parallèle d'abord ; puis basse + forme en parallèle.
**A2.** Plusieurs écoutes ; les écoutes suivantes corrigent la première avec la
structure connue. Correction typique : Gm7 → B♭maj/G (même contenu de notes,
réécriture fonctionnelle).
→ *Modules* : ordre du pipeline = structure avant accords définitifs
(re-décodage poolé — SHIPPÉ 2026-07-20, Jaccard Let It Be 0.80→1.00) ;
**canonicaliseur de label** : à pc-set égal (Am7=C6, Gm7=B♭6…), choisir
l'écriture par basse réelle + fonction (À CONSTRUIRE).

## B. Identifier un accord
**B3.** La basse est TOUJOURS entendue en premier ; la couleur se déduit
ensuite. Renversements « ressentis » avec l'expérience. Suit le voice-leading
(« où se baladent les notes suivantes », pitch + timbre).
→ *Modules* : émission root pilotée par la basse (le biais C-sur-Am vient de
là : la basse joue A, la tête root préfère C — la basse doit avoir un droit de
veto pondéré) ; continuité de voicing entre accords voisins (piste, jamais
testée comme contrainte de continuité).

**B4.** Hésitation entre deux accords → il JOUE les deux et compare à ce qu'il
entend : « le classifieur te donne deux accords, tu compares le vrai
histogramme de chroma des deux ». Plus : partiels spécifiques à l'instrument
(un C au piano a des harmoniques qu'un F n'a pas) ; et le contexte/répétition
(l'alternance avec la basse, les choses qui se répètent).
→ *Module* : **arbitre top-2 par analyse-par-synthèse** — pour les barres à
faible marge, comparer le chroma OBSERVÉ aux chromas ATTENDUS des deux
candidats (templates voicing/partiels-aware), poolé sur les répétitions.
NB : ce n'est PAS le template-scoring 60-way de Gen-1 (qui a échoué) — c'est
un référé BINAIRE sur un segment disputé, problème beaucoup plus facile.

**B5.** Perception d'abord, attente ensuite : une fois la chanson « comprise »,
il reconnaît les patterns d'oreille (ii-V-I…) — « une fois les deux premiers
accords trouvés, les 3ᵉ et 4ᵉ découlent logiquement, par blocs ».
→ *Module* : **complétion de pattern gated** — petite bibliothèque de patterns
fonctionnels (ii-V-I, I-vi-IV-V, blues 12…) en espace root-relatif ; quand ≥2
slots matchent avec confiance, prior de complétion sur les slots restants via
l'arbitrage de Bayes (jamais un prior de corpus qui écrase — la leçon des
priors morts : ils ne marchent qu'arbitrés contre une évidence calibrée).

## C. Granularité du vocabulaire
**C6-C7.** La 7ᵉ s'ÉCRIT seulement si elle s'ENTEND — pas de règle générale ni
de défaut stylistique. La dominante 7 est la plus importante à entendre
vraiment ; le maj7 a une consonance reconnaissable ; le 6 est le plus dur
(noyé dans la pentatonique) → déprioriser. « Dans le doute on ne met rien —
c'est pour ça que l'arbre de famille est logique ; quand on est sûr, on
descend. »
→ *Modules* : valide l'affichage par profondeur d'arbre piloté par la
confiance calibrée (déjà en place : famille → 7ᵉ → exact). Raffinement à
tester : la descente vers la 7ᵉ devrait exiger l'évidence du degré lui-même
(masse chroma sur la ♭7/maj7 du candidat), pas seulement la confiance
globale de l'accord. Pas de prior de style pour les 7ᵉ. Les 6tes : ne pas
investir pour l'instant.

**C7.** Sur un chart : TOUJOURS sous-écrire plutôt que sur-écrire.

## D. Forme et sections
**D8.** À l'INTÉRIEUR des répétitions d'un A : pas d'indice fort — elles
s'entendent comme une continuité. Le signal vit à la TRANSITION vers du
matériau différent (A→B) : changement rythmique + harmonique, souvent un
silence, ou le DERNIER accord de la section qui change pour amener la
suivante (pivot/turnaround). « Il y a toujours un petit quelque chose. »
Regarder la discontinuité à toutes les échelles — MAIS le tri harmonique
suffit peut-être : « le A se répète jusqu'à ce qu'il ne se répète plus. »
→ *Modules* : valide largest-unit (répétition-jusqu'à-rupture) comme
détecteur primaire ; les indices acoustiques (silence, pivot de dernier
accord, rupture rythmique) sont des CONFIRMATEURS de frontière, pas des
détecteurs — cohérent avec l'échec du modèle multi-facteurs (la position de
phrase domine). Le pivot-de-dernier-accord est une feature ciblée jamais
testée (regarder la DERNIÈRE barre de chaque bloc candidat).

**D8-bis — BUG IDENTIFIÉ PAR L'OREILLE : le drift fabriquait de faux B.**
Un A répété 3× dérivait de phase à la 3ᵉ répétition → le système voyait
« les mêmes accords mais décalés » → cluster séparé → étiqueté B à tort.
→ *Action* : la similarité de sections doit pooler sur la grille SNAPPÉE
AUX BEATS RÉELS (comme l'affichage désormais) ou matcher avec tolérance de
phase (±1 barre) ; re-vérifier les splits A/B existants une fois fait —
certains B actuels sont peut-être des A dérivés.

**D9.** A vs B = la STRUCTURE HARMONIQUE, comparée au niveau FAMILLE.
Équivalence pour la distance entre sections : A ≡ Amaj7 (réharmonisation,
même famille, même gamme) ; les mineurs : pas de sous-qualité distinctive.
LA seule 7ᵉ qui distingue : la DOMINANTE (maj vs dom7 = différent ;
maj vs maj7 = pareil ; min vs min7 = pareil). Le vrai marqueur de
changement : le 2ᵉ accord du bloc qui diffère. Prior : un A fait 8 barres
(« le classique »).
→ *Modules* : la métrique de similarité de sections doit projeter les
qualités sur {maj≡maj7≡6, min≡min7, DOM distinct, dim/aug/sus...} avant
comparaison ; position-2 du bloc = discriminateur fort ; le prior 8 barres
est déjà dans la machinerie largest-unit.

## E. Temps
**E11.** Le « 1 » (et les débuts de section) : en général là où LA VOIX
commence ; sinon « un petit changement clair » — même signal que les fins de
section. Consigne explicite : prendre TOUS les ingrédients que lui
utiliserait — changement harmonique, changement mélodique, LA VOIX qui entre
ou sort, la batterie — et les donner à un CLASSIFIEUR qui apprend le
mélange ; « il n'y a pas de règle simple ».
→ *Modules* : re-tenter le classifieur de frontières avec les BONS
ingrédients (le premier essai avait des features frustes + 6 chansons) —
la feature nouvelle et probablement décisive : **détection d'activité
vocale** (la voix entre au « 1 » des sections, sort aux ponts/instrus).
Jamais extraite dans ce projet. + changement de ligne mélodique, changement
de pattern batterie. Il faudra plus de paires GT pour l'entraîner.

## G. Vérification du chart fini
**G15.** C'est le RASOIR D'OCKHAM à nouveau, appliqué à la vérification :
« je veux la structure minimale qui explique au mieux la chanson ». Boucle :
pour chaque paire de sections candidates, se demander « est-ce la même chose
répétée différemment ? » et merger si oui — MAIS ne jamais merger deux
sections si leur ENTROPIE (distance) est trop grande. Contrainte de bon sens
en butée haute : si on se retrouve avec ~8 sections distinctes, c'est le
signal qu'on doit merger davantage — ce n'est pas plausible musicalement.
→ *Modules* : la vérification finale = une passe de clustering hiérarchique
sous contrainte double (seuil d'entropie/distance MIN pour merger + budget
de labels MAX, ex. k≤5 déjà en place) — c'est le PENDANT en sens inverse de
l'anti-crush (qui empêche de sur-simplifier) : ici on empêche de
sous-simplifier. Boucle de vérification = ré-appliquer cette passe après
tout fix, pas seulement au décodage initial.

## F. Arbitrage pattern/exception
**F13.** Exemple concret : un vamp lu [E, E, C#m7], où C#m7 ≈ E (accords
proches, même famille au sens D9). Décision : si TOUTES les autres passes du
cycle ne montrent que E → « je me suis trompé, c'est E comme les autres »
(bruit de décodage absorbé — la LIGNE DE BASSE tranche systématiquement).
Si c'est ~50/50 (une fois sur deux du E, une fois sur deux du C#m7/7) →
« c'est un vrai 4-temps qui boucle » — le pattern se redéfinit pour
l'inclure, ce n'est plus une exception. Le critère : la RÉGULARITÉ de la
récurrence à travers les passes, arbitrée par la basse.
→ *Module* : le vote « exception vraie » ne doit pas être un seuil de marge
par barre isolée (ce qu'on a) mais une évidence AGRÉGÉE par position de
cycle à travers TOUTES les passes — exactement le pooling déjà en cours de
build (Let It Be), généralisé comme critère d'exception plutôt que seulement
comme correcteur de root.

**F14.** Deux critères séparés, à ne pas confondre :
1. **`%` (tenu)** : aucune DIMENSION harmonique ne change vs l'accord
précédent — même si on n'entend qu'une bribe, si rien ne bouge
harmoniquement, c'est un tenu, pas un nouvel accord.
2. **N.C. (vraiment rien)** : nécessite un jugement séparé, à l'oreille,
« harmonie mélodique vs bruit aléatoire » — et Louis reconnaît que **le
modèle a besoin d'être mieux calibré sur les non-accords** pour trancher
ça avec certitude ; ce n'est pas une simple absence de changement, c'est
une discrimination positive signal-harmonique / bruit.
→ *Modules* : (1) déjà couvert par le mécanisme `%`/tenu existant — aucun
changement de dimension harmonique détecté = hold, indépendant de la
confiance du modèle. (2) N.C. actuellement dérivé de musx + un garde
d'énergie/flatness NNLS (seuil dur) — Louis demande un vrai calibrage
signal-vs-bruit, pas juste un seuil d'énergie. Piste : un classifieur
harmonicité (présence de structure de pics chroma stable vs bruit large-
bande/percussif), calibré sur des segments N.C. connus (les ponts a
cappella / silences déjà localisés au sein du set GT apparié) —
généralise l'infra de calibration de confiance déjà construite (isotonic,
ECE) à une tâche binaire « y a-t-il de l'harmonie ici ? ».

## Questions en attente — INTERVIEW COMPLÈTE (A-G répondues)
Reste ouvert : E12 (choix d'octave de tempo — recherche littérature en
cours, réponse séparée).
C6-C7 (quand écrire la 7ᵉ ; sous- vs sur-écrire), D8 (indices de fin de
section, le fill précisément), D9 (couplet vs refrain à accords égaux),
D10 (réharmonisation = même A ?), E11-E12 (trouver le « 1 » ; octave de
tempo), F13-F14 (vraie exception ; N.C.), G15 (vérification par frottements).

## H. Grammaire de segmentation apprise du CORPUS (étude 2026-07-20)
*Étude symbolique GT-propre sur iReal `pop400` (345 morceaux parsés) + `jazz1460`
(1460), via `sectionized_measures` (les LABELS de section A/B/C sont la vérité-
terrain du compositeur). Généralise D8 (taille/nombre de blocs) et D9 (A vs B) avec
des nombres réels. Artefact `docs/plots/segmentation_grammar_corpus_2026_07_20.png`.
Script `scratchpad/grammar_study.py`. Méthodo = celle de l'étude Occam (distributions
corpus d'abord, algorithme ensuite).*

**H1 — "est-ce que c'est toujours du 8 ?" → 8 est le mode DANS LES DEUX genres, mais
le pop est IRRÉGULIER, le jazz est RÉGULIER.**
| | 8-bar | 16-bar | 4-bar | puissance-de-2 | médiane |
|---|---|---|---|---|---|
| pop400 (2144 instances) | 32% | 13% | 10% | **59%** | 8 |
| jazz1460 (4177) | **52%** | 25% | 1% | **81%** | 8 |
→ Un prior 8-barres est justifié mais doit rester SOUPLE : au niveau instance le pop
le viole ~2 fois sur 3 (12/9/10/6/7/14 barres existent réellement). Le jazz est carré.

**H2 — la 1ʳᵉ section (le "début") : pop 8 (34%) ou 4 (30%) ; jazz 16 (49%) ou 8 (22%).**
Le prior "un A = 8 barres" tient pour ~1/3 du pop seulement ; il faut autoriser 4.

**H3 — l'hypothèse "une phrase répétée" (largest-unit) est de forme JAZZ, pas POP.**
Morceaux où une seule taille de bloc couvre ≥70% des sections : **pop 9.9%**, jazz 40%.
Morceaux mono-taille (toutes sections identiques en longueur) : pop 1.4%, **jazz 35%**.
→ Le pop MÉLANGE les tailles (couplet 8 / pré-refrain 4 / refrain 8 / pont 16) : le
détecteur doit accepter des blocs de longueur VARIABLE dans un même morceau (pop),
là où le jazz supporte un bloc unique.

**H4 — nombre de sections distinctes / morceau : k≤5 VALIDÉ ; "un A un B" trop agressif
pour le pop.** pop médiane **4** (3:33%, 4:39%, 5:21%, **0% > 5**) ; jazz médiane **2**
(1:16%, 2:55%, 3:24%). → Le budget k≤5 est empiriquement correct (0% le dépassent).
Mais le pop a vraiment ~4 types (intro/couplet/pré-refrain/refrain/pont) — viser "un A
un B" SOUS-segmente le pop. Le jazz colle à A/B (AABA = 2 types).

**H5 — SUR-FUSION quantifiée : les accords SEULS ne distinguent pas les sections
1 fois sur 5 en pop.** Paires de labels DISTINCTS partageant le même vocabulaire de
fondamentales : **pop 21% de vocab IDENTIQUE** (Jac=1.0), 25% à Jac≥0.8 ; jazz 4% / 11%.
→ C'EST la preuve empirique du problème de sur-fusion (couplet↔refrain à mêmes accords,
ex. She Will Be Loved) : un regroupement par vocabulaire-d'accords SE TROMPE dans ~1
morceau pop sur 5. La distinction de section doit s'appuyer sur l'ORDRE / la POSITION /
le compte de répétitions / la mélodie — PAS le vocabulaire d'accords seul. Valide D9
(discriminateur "2ᵉ accord du bloc") et le besoin de blocs ancrés-grille plutôt que
d'un clustering de vocabulaire.

**H6 — l'échelle de récurrence est spécifique au genre : pop pique au lag 4, jazz au
lag 16.** P[root[b]==root[b−lag]] : pop {lag4:43%, lag8:41%, lag16:40%, lag2:33%} ;
jazz {lag16:40%, lag8:27%, lag4:23%}. Meilleur lag PAR MORCEAU : pop {4:39%, 2:26%,
8:19%, 16:17%}, jazz {**16:58%**, 2/4/8:~15%}. → Pas de lag fixe : scanner 4/8/16 et
choisir par morceau. Le pop = boucles 4-barres NICHÉES dans des phrases 8-barres (le
Flag 2 de l'user : "les petites boucles vivent DANS les sections").

**Synthèse pour la construction (ce qui EST une règle stable vs contextuel) :**
- STABLE : 8-barres est l'unité de phrase modale (les deux genres) ; k≤5 (jamais violé) ;
  le vocabulaire d'accords NE suffit PAS à séparer les sections (sur-fusion 21% pop).
- CONTEXTUEL (genre-dépendant, ne PAS coder en dur) : régularité (jazz carré, pop mixte) ;
  échelle de boucle (pop 4, jazz 16) ; nombre de sections (pop ~4, jazz ~2) ; longueur du
  1ᵉʳ bloc (pop 4-ou-8, jazz 16). → Un prior de longueur de bloc doit être SOUPLE et
  idéalement conditionné au genre/tempo, arbitré contre l'évidence (jamais un prior mort).
