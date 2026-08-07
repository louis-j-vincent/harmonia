# Quart de barre : la porte s'ouvre, mais l'évidence ne passe pas (2026-08-07)

Branche `feat/quarter-bar`. Question de Louis : « passer à granularité 1/4 de
barre — partout, ou garder la demi-barre puis un détecteur qui dit là où il y a
plus d'accords, et re-mesurer ces zones en 1/4 ? »

## Ce qui a été construit

Le décodeur vendored (XHMM) supportait depuis toujours un 4e grade de
transition « autre temps » (pénalité `beat_trans_penalty[2]` = 100) ; la
restriction demi-barre de harmonia_min le mettait à zéro en une ligne.
`make_beat_arr`/`redecode` ont maintenant un paramètre `quarter_beats` :

* `None` — demi-barre seule (comportement livré, inchangé, défaut) ;
* `"all"` — changement permis sur chaque temps, pénalité 100 ;
* liste d'indices de temps — mode CIBLÉ, l'entrée prévue pour un détecteur.

Flag pipeline : `HARMONIA_QUARTER_BAR=all`. Tests `tests/test_musx_quarter_bar.py`.
Harnais : `scratchpad/quarter_bar_{census,run,score}.py` (7 chansons gelées,
zéro décodage audio — beats + postérieurs musx en cache).

## Recensement : le benchmark n'a presque pas de vrai 1/4 de barre

Sur la grille de downbeats VÉRIFIÉE À LA MAIN des GT Brick-0 (pas celle du
tracker — première mesure : 30 % « quarter », artefact intégral de phase/dérive
du tracker sur blue_bossa et l'overlay ; pattern d'erreur n°1) :

| position de l'onset | n / 729 |
|---|---|
| temps 0 (barre) | 632 |
| temps 2 (demi-barre) | 87 |
| temps 1 ou 3 (**quart**) | **4** (tous bein_green) |
| hors grille | 6 |

Enjeu total du quart de barre : **3.2 s sur 1654 s de GT (0.2 %)** — le
walkdown de Bein Green (G#/D# → F#/C# → F:7/C, un accord par temps à 75 BPM).

## A/B : ouvrir partout ne change (presque) rien

Premiers scores Brick-0 de harmonia_min (jamais mesuré jusqu'ici), pooled
duration-weighted, contrôle = demi-barre :

| bras | root | partial | strict | frontières F1 ±0.25s | #pred/#GT |
|---|---|---|---|---|---|
| contrôle (demi-barre) | 0.7272 | 0.6489 | 0.4755 | 0.48 | 0.94 |
| quart partout, pén. 100 | 0.7307 | 0.6510 | 0.4743 | 0.50 | 0.95 |
| pén. 45 | 0.7301 | 0.6501 | 0.4741 | 0.50 | 0.95 |
| pén. 20 | 0.7298 | 0.6510 | 0.4718 | 0.50 | 0.97 |

Par chanson : le +0.35pp de root vient de georgia (+2.9pp root mais −2.4pp
strict ; sa « barre » détectée = 2 temps, octave métrique fausse) et
stand_by_me (+0.7pp) — c'est la rigidité demi-barre sur des BARRES IRRÉGULIÈRES
du tracker qui se détend, pas de la granularité harmonique. bein_green : zéro
changement à toutes les pénalités.

## Le verdict qui compte : l'évidence est le goulot, pas la grille

Zone 37.6–40.0 s de bein_green, GT `G#:maj/D# | F#:maj/C# | F:7/C` (0.8 s
chacun). Décodé — même à pénalité 20, porte grande ouverte :
`F:sus4(b7)` sur 3.2 s. Les postérieurs musx lissent le walkdown ; aucune
position de transition supplémentaire ne peut créer une frontière que
l'évidence ne voit pas.

Conséquence pour l'architecture « détecteur puis re-mesure » (l'instinct de
Louis) : elle reste la bonne — mais la re-mesure des zones signalées doit être
un AUTRE instrument (plan basse musx, chroma NNLS à pas fin, suivi de la ligne
de basse), pas le même décodage musx avec plus de positions permises. Détecter
ne suffit pas si l'instrument de mesure est aveugle à l'échelle visée.

## Ce que cette session NE règle pas (règle n°4)

* blue_bossa root 0.58 : phase downbeat du tracker en désaccord avec les
  downbeats GT vérifiés — problème de GRILLE, domine tout gain de granularité.
* georgia : octave métrique (bpb détecté 2 vs 4/4 réel), même famille.
* Le mode ciblé (`quarter_beats=[...]`) est câblé mais SANS détecteur ni
  benchmark : les 4 accords visés sur 7 chansons ne permettent ni d'étalonner
  ni d'évaluer. Il faudrait d'abord 2–3 GT riches en vrai 1/4 de barre
  (let_it_be — 3 accords/barre au walkdown, GT non vérifié ; gospel).
* La carte LM chante en tokens demi-barre (`slots_per_bar=2`) : une sortie
  quart de barre ne mappe plus 1:1 sur sa grille (`chord_lm/from_app.py`).

## Recommandation

Laisser le défaut demi-barre. Le flag et le mode ciblé restent prêts. Le
prochain pas UTILE vers le 1/4 de barre n'est pas le décodeur : c'est (1) un
petit jeu GT avec de vrais walkdowns, (2) un instrument de re-mesure fine des
zones signalées (basse), et — avant tout ça — (3) la phase/octave de la grille,
qui pèse 10× plus sur les mêmes chansons.

## Suivi (même jour) : let_it_be renverse la nuance — le quart marche quand la FONDAMENTALE bouge

Demande de Louis : passer le quart de barre dans le pipeline entier et montrer
let_it_be + bein_green. Résultat qui affine le verdict d'hier :

* **let_it_be (GT non vérifié, indicatif)** : le décodage quart récupère les
  trois `F:maj` d'un temps du « whisper words of wisdom » (109.6 s, 116.6 s,
  235.5 s) — exactement ce que le GT iReal écrit et que la demi-barre avale.
  127 → 130 segments, zéro accord parasite ajouté.
* **bein_green** : toujours rien — le walkdown est fait de RENVERSEMENTS
  (G♯/D♯, F♯/C♯ : le plan triade musx ne bouge presque pas), là où les F de
  let_it_be sont des changements de fondamentale que le plan triade voit.

Donc : « l'évidence est le goulot » vaut pour les walkdowns par renversement ;
pour les accords de passage à fondamentale franche, la porte quart suffit.

Conséquences aval vérifiées (pipeline entier, `HARMONIA_QUARTER_BAR=all`,
E2E OK sur les deux chansons) : assemblage des barres = slots par temps ✓ ;
sections = moyennes PAR BARRE, insensibles ✓ ; harmonic_key par accord ✓ ;
prompteur ✓ ; **repli corrigé** — son re-décodage du template disait
« quart cher (100) » en commentaire mais l'interdisait en réalité depuis la
restriction du 2026-08-01 (drift silencieux), il suit maintenant le flag ;
chord LM (opt-in, OFF) : tokens demi-barre, deux accords dans le même slot →
un seul survit — perte d'info, non cassant.

Pages d'écoute (audio réel + tête de lecture, docs/plots/) :
`diag_quarter_let_it_be.html`, `diag_quarter_bein_green.html`,
`diag_grid_blue_bossa.html` (phase downbeats tracker vs GT),
`diag_grid_georgia_on_my_mind.html` (octave métrique bpb 2 vs 4).
Générateur : `scratchpad/quarter_bar_pages.py`.

## Suivi 2 (même jour) : Louis tranche à l'écoute — restriction levée, GT blue_bossa contesté

Deux verdicts de Louis sur les pages :

1. **blue_bossa : le tracker a raison, le GT brick0 a tort** (downbeats — et
   donc la timeline d'accords issue de la même passe). Mon interprétation
   d'hier était inversée ; le score root 0.58 est un artefact GT probable,
   pas une défaillance de grille. GT à re-vérifier.
2. **« On ne met plus de restrictions sur la granularité »** : le défaut du
   pipeline ET du repli est maintenant `quarter_beats="all"` (quart de barre
   en 4/4, tiers en 3/4 — même mécanisme). Plancher = le temps ; entre deux
   temps reste interdit. Kill-switch `HARMONIA_QUARTER_BAR=off`. E2E : le
   prompteur de let_it_be affiche G→F→C, 3 accords dans la barre, la typo de
   l'app suivait déjà.

Chiffres frozen-7 du nouveau défaut = le bras « all » d'hier (root 0.7307,
partial 0.6510, strict 0.4743) — blue_bossa suspect dans les deux colonnes.
