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

## Questions en attente
C6-C7 (quand écrire la 7ᵉ ; sous- vs sur-écrire), D8 (indices de fin de
section, le fill précisément), D9 (couplet vs refrain à accords égaux),
D10 (réharmonisation = même A ?), E11-E12 (trouver le « 1 » ; octave de
tempo), F13-F14 (vraie exception ; N.C.), G15 (vérification par frottements).
