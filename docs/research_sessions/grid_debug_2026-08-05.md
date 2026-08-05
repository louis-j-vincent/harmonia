# En amont de la méthode verrouillée : grille, dérive, modulation (2026-08-05)

Session ~90 min sur le brief de Louis : la méthode harmonique est verrouillée
(`scripts/harmonic_method.py`, importée, jamais ré-implémentée — les sorties
verrouillées This Love 7 pics / Norah 10 / Sunny 2 sont reproduites exactement
ici) ; le sujet est POURQUOI la matrice SSM est mauvaise sur certains morceaux.
Rapport visuel : `/reports/grid_debug.html`. Prototypes :
`scripts/grid_debug_core.py`, `scripts/grid_debug_report.py`.

## 1. La « dérive de −11,4 % » de Beat It est un artefact (erreur type n°1)

La définition qui reproduit les chiffres publiés (Beat It −11,4/−11,8, Sunny
−2,2, This Love −0,0, Norah −0,4, Close to You −1,5) : pente OLS des
intervalles entre beats sur le temps × durée / intervalle moyen. Or sur Beat
It, **les 8 premiers beats (21–30 s, l'intro au gong avant l'accrochage du
groove) valent 0,74–1,74 s au lieu de 0,44 s** ; placés en tête de régression
ils fabriquent toute la pente. Sans eux : **−0,27 %**. L'intervalle médian vaut
0,4400 s dans chacune des 20 tranches du morceau — tempo métronomique.
Le diagnostic « Louis's drift theory, confirmed here » de l'entrée verrouillée
de known_issues est donc à corriger : un chiffre plausible jamais confronté à
son tracé (le contre-exemple exact que la règle n°1 du projet vise).

## 2. Le vrai signal pour le garde-fou : l'instabilité de PHASE, pas la dérive

Mesuré sur les 83 fichiers du cache beats (60 GuitarSet + 23 vraies chansons,
agent corpus, ancres validées) :

- **Dérive naïve** (OLS brute) : inutilisable — dominée par les outliers
  (Nina Simone −174 %, Ray Charles −76 %).
- **Dérive robuste** (outliers ±50 ms exclus) : seules Let It Be (+7,4 %) et
  Kermit (+6,5 %) dépassent |3 %| — deux morceaux qui accélèrent VRAIMENT
  (interprétation humaine). Or une vraie accélération ne casse pas la grille :
  les mesures suivent les beats réels détectés, chaque mesure fait toujours 4
  vrais beats. **La dérive n'est pas le bon signal de refus.**
- **Instabilité de phase** = part des downbeats hors de la phase modale de la
  grille à 4 beats (calculée après le 8e downbeat) : **bimodale parfaite sur
  les vraies chansons** — saines ≤ 0,088, suspectes ≥ 0,296, personne entre.
  Elle attrape ce que la cohérence du garde-fou ne voit pas : **Chain of Fools
  0,486 avec cohérence 0,99** (UN glissement précoce qui déphase la moitié du
  morceau — chaque comparaison inter-moitiés de la SSM est alors décalée d'un
  nombre fixe de beats).

**Proposition (à valider par Louis)** : garder le garde-fou actuel ET refuser
au-dessus de 0,15 d'instabilité de phase. Prises nouvelles : Chain of Fools
(0,486), Blue Bossa 150 (0,317), Kermit (0,296) — les trois passent
aujourd'hui ; à vérifier à l'oreille avant de coder. Ce que ça ne résout pas :
un tracker faux en bloc (Close to You) reste l'affaire de la cohérence ; un
défaut local de 6 mesures (Beat It, 0,039) reste sous tout seuil raisonnable.

## 3. La grille de Beat It n'était pas cassée — le ré-ancrage AGGRAVE

Le seul litige : 6 mesures (50–60 s) où le tracker décale ses downbeats d'une
demi-mesure puis revient (151/160 downbeats en phase 0 ailleurs). Réparation
testée : frontières de mesures = les downbeats du tracker eux-mêmes (ré-ancrage
local), ± intro purgée. Verdict par l'image (zoom aligné EN TEMPS, 42–68 s) :
**le ré-ancrage fait apparaître un bloc uniforme clair — la signature du
mélange demi-mesure (chaque mesure contient moitié mi, moitié ré) — là où la
grille rigide garde son damier net.** Dans la zone litigieuse c'est le TRACKER
qui vacillait, pas la musique ; `off + b×4` avait raison de l'ignorer. Pics :
30 (rigide) → 2 (ré-ancré).

**Le vrai problème de Beat It n'est pas la grille** : un riff de 2 mesures
partout (L=2 détecté, le motif colle sur la moitié du morceau — 30 « pics »
tous vrais musicalement, aucun utile pour une forme) + un bloc uniforme
(~78–95, le breakdown). Morceau harmoniquement homogène : hors du domaine d'un
détecteur de répétition harmonique. Piste : substrat rythmique/timbral.

## 4. Sunny : max 12 rotations détruit, max ±1 demi-ton est le bon diagnostic

- **Max sur les 12 rotations de chroma** : moyenne hors-diagonale 0,37 → 0,86,
  matrices délavées, courbe plate (amplitude ~0,03), les contrôles cassent
  (This Love perd sa phase, 7 → 11 pics ; Norah 10 → 14). Rejeté par l'image.
- **Max sur {−1, 0, +1} demi-ton** (les modulations réelles sont à un
  demi-ton) : les contrôles gardent période, phase et leurs pics verrouillés
  (This Love : les 7 exacts + 2 parasites ; Norah : les 10 exacts + 1) ; Sunny
  passe de 2 pics muets à L=16 avec 6 pics, et la carte de gain S(±1) − S(0)
  montre des BLOCS cohérents (répétitions modulées) là où les contrôles n'ont
  que du bruit diffus.
- **Limites, dites** : inflation quand même (0,37 → 0,64) ; et Sunny module en
  CHAÎNE, donc les sections à ≥ 2 demi-tons cumulés restent invisibles à un
  max ±1 qui ne suit qu'une marche — les 6 pics vivent tous dans la première
  moitié. En l'état : un diagnostic de modulation solide, pas un substrat de
  remplacement. Aucun chiffre de placement produit (donc pas de repère
  « ne bouge jamais » à opposer).

## Données et reproductibilité

- Grilles live capturées via le spy de `harmonic_method.__main__` (pipeline
  réel, 4 morceaux) ; posteriors musx de Beat It calculés et mis en cache au
  passage.
- Corpus dérive : `drift_corpus.json` (scratchpad session, 83 fichiers, un
  agent ; définitions exactes dans le rapport).
- Le rapport est autonome (PNG en dur), servi par `/reports/grid_debug.html`.
