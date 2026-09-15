# Soudure — brief

Page autonome (`soudure.html`, un seul fichier, tout inline, aucune ressource
externe). Source éditable : `Soudure.dc.html`. Mobile d'abord, 390 px.

## Le principe

Transformer le découpage d'une chanson en un jeu de soudure. La chanson est une
bande de jetons ; l'utilisateur soude des paires voisines, et chaque soudure
s'applique **partout à la fois** dans la chanson. Le plaisir vient de la
cascade : un geste, beaucoup d'effet.

## Données

```js
window.SONG = {
  titre: "…",
  n_mesures: 84,
  temps_par_mesure: [1.9, …],   // secondes, longueur = n_mesures
  mot: "aaaaaabc…",             // une lettre par jeton
  audio_url: null
}
```

Lu au chargement s'il existe, sinon jeu d'exemple en dur :
`"aaaaaabc"×5 + "aa"` = 42 jetons, 84 mesures, 1,9 s par mesure.
`mesures_par_jeton = n_mesures / mot.length` (2 dans l'exemple).

Pas d'audio externe : les mesures sont sonorisées par synthèse (une couleur de
son par lettre), donc `audio_url` est optionnel.

## Modèle

Une **entité** = `{ contenu: "aa", jeton_debut, nb_jetons }`. Au départ, une
entité par jeton (`contenu` = sa lettre). Deux entités de même `contenu` sont le
même endroit de la chanson : même couleur, même lettre majuscule.

**Souder (x, y)** — parcours de gauche à droite, sans chevauchement : partout où
une entité de contenu `x` est suivie d'une de contenu `y`, les deux fusionnent en
une entité `x+y`. Une seule opération, N fusions. C'est du BPE.

Chaque soudure empile un état dans l'historique. Les entités grossissent :
2 mesures → 4 → 8 → 16.

## Écran (de haut en bas)

1. **Titre** — Georgia italique, sous-titre `84 mesures · 42 entités`.
2. **Sélecteur de mode** — segmenté 44 px : `Écouter` | `Souder`, avec une ligne
   d'aide en dessous qui dit le geste du mode courant.
3. **La bande** — les entités, plaques crème bordées de leur couleur, qui
   s'enroulent sur plusieurs lignes. Largeur ∝ nombre de jetons, hauteur et
   corps de la lettre croissent avec la taille (46/54/62 px). Lettre minuscule
   tant que l'entité est brute, majuscule dès qu'elle est soudée.
4. **Le compagnon** — bandeau permanent : la paire la plus fréquente restante,
   `a + a · 16 fois`, et un bouton `Souder` de 48 px. Quand il n'y a plus de
   paire répétée : « Plus aucune répétition ».
5. **Transport** — lecture/pause 56 px, annuler 48 px, réglette d'historique
   (rejoue tout l'historique en avant et en arrière), compteur `3/7`.
6. **Résultat** — la bande des entités finales, une lettre majuscule chacune,
   largeur proportionnelle, + `Exporter JSON`.

## Gestes

**Mode Écouter** (par défaut)
- 1 tap : sélectionne l'entité et la joue en boucle.
- 2e tap sur une autre : les deux sont sélectionnées (petit chiffre 1 / 2) et
  jouées l'une après l'autre, en boucle, pour comparer.
- Re-tap sur la sélection unique : arrêt.

**Mode Souder**
- 1 tap : sélectionne et joue en boucle.
- 2e tap sur une **voisine** : soudure, appliquée partout.
- 2e tap ailleurs : la sélection se déplace.

Le bouton `Souder` du bandeau marche dans les deux modes.

## La cascade — le moment à réussir

À la soudure : chaque nouvelle entité s'anime avec un décalage de 45 ms selon sa
position, ce qui fait courir une vague de gauche à droite ; un éclat traverse la
jointure ; un badge `13 soudures` monte au centre de la bande et s'efface ;
une série de blips montants, un par fusion, à 45 ms d'intervalle.

## DA — Harmonia

Papier `#f7f3e9`, cartes `#fffdf6`, filets `#e5dcc6`, encre `#1c1c1c`,
secondaire `#8a8371`, accent `#8a2b2b`. Couleurs de section : accent, `#2a6fb0`,
`#6b7a3a`, `#b07d2a`, `#6d4670`, `#3d7a72`, `#a4552a`, `#4a5c86` — assignées par
hachage du contenu, donc stables d'un état à l'autre.
Georgia italique pour tout ce qui est musical (lettres, titre) ; -apple-system
pour l'UI. Trois rayons : 9 / 14 / 20. Cibles 56 / 48 / 44 px.

## Export

Bouton → téléchargement JSON :

```json
[{ "label": "A", "mesure_debut": 1, "mesure_fin": 12 }]
```

Mesures 1-indexées, `mesure_fin` incluse.

## À ne pas faire

Pas de tableau de bord, pas de graphique, pas de statistiques. L'écran ne montre
qu'une bande, une suggestion, un lecteur.
