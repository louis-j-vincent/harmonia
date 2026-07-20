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

## Questions en attente
C6-C7 (quand écrire la 7ᵉ ; sous- vs sur-écrire), D8 (indices de fin de
section, le fill précisément), D9 (couplet vs refrain à accords égaux),
D10 (réharmonisation = même A ?), E11-E12 (trouver le « 1 » ; octave de
tempo), F13-F14 (vraie exception ; N.C.), G15 (vérification par frottements).
