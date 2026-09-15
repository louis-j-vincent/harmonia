# Prompt de direction artistique — Harmonia

À coller en tête de session, avant `HANDOFF.md`. Le handoff dit *quoi* changer ;
ce texte dit *comment ça doit se sentir*. Quand les deux semblent se contredire,
c'est ce texte qui tranche.

---

Tu travailles sur Harmonia, une app qui donne à lire l'harmonie d'une chanson.
L'objet de référence n'est pas une app : c'est un **real book posé sur un pupitre**.
Papier crème, encre noire, une seule couleur d'accent — le bordeaux du tampon
d'éditeur. Tout ce que tu ajoutes doit avoir l'air d'avoir été imprimé là, pas
d'avoir été collé par-dessus.

**Ce qui compte, dans l'ordre :** les accords. Puis la position dans le morceau.
Puis tout le reste. Un musicien lit ça à un mètre, en jouant, avec les mains
occupées. S'il doit chercher, on a perdu. S'il doit viser, on a perdu.

## Les règles fermes

**Trois tailles, trois rayons, trois états.** 56 primaire, 48 onglet, 44 tout le
reste ; rayon 12 / 16 / plein ; sélectionné (accent plein) / disponible (carte +
filet) / éteint (atténué). Rien en dessous de 44 px, jamais, sur aucun axe. Si un
contrôle ne rentre dans aucune de ces cases, c'est une question de design : pose-la,
n'invente pas une quatrième hauteur.

**Une seule famille de contrôles.** Pas de pilules contournées d'accent à côté de
boutons fantômes à côté de segments. Si tu as besoin d'un choix entre 2 et 4
options, c'est un `kitSegmented`. Une action, c'est un `kitButton`. Rien d'autre.

**Pas d'emoji.** Jamais, nulle part, pas même « juste pour cette ligne de debug ».
Un symbole devant un label (`◎`, `⧉`, `⇧`) est un emoji qui se cache : écris le mot.

**Pas de bordures en pointillés.** Ça dit « pas fini ». Un filet plein, comme tout
le reste.

**Deux couleurs de fond maximum** sur un écran. La couleur porte du sens ici — clé
locale, rôle harmonique, confiance du modèle — donc elle ne peut jamais être
décorative. Si tu colores quelque chose, tu dois pouvoir dire ce que la couleur
signifie, et cette signification doit être écrite quelque part à l'écran.

## Le ton

**Écris comme quelqu'un qui sait, à quelqu'un qui apprend.** Minuscules, phrases
courtes, pas de majuscules de titre, pas de point d'exclamation. « sets up the V
into C », pas « This chord creates tension! ». On explique la musique, on ne vend
rien.

**Une ligne suffit.** Les légendes de l'app font aujourd'hui une phrase entière, en
permanence, sur chaque chanson. Une ligne, sous le contrôle qu'elle explique, et
elle disparaît quand elle n'est plus utile.

**Ne remplis pas.** Une zone vide n'est pas un problème à résoudre avec une carte de
plus. Le vide autour de la grille, c'est ce qui rend la grille lisible.

## Le geste

**Le pouce d'abord.** Ce qu'on touche en jouant vit en bas de l'écran. Ce qu'on lit
vit en haut. Le chrome ne vole jamais plus de deux rangées au-dessus de la musique.

**Rien ne saute.** Une grille qui se redessine, un accord qui se décale d'une case,
une barre de progression qui bondit de 40 % à 100 % — chacun de ces gestes coûte la
confiance du lecteur, et elle ne revient pas. Quand un état remplace un autre, les
mêmes cellules restent aux mêmes pixels.

**Ne mens pas sur l'attente.** Si le calcul ne renvoie rien pendant une minute, dis
que ça prend une minute. Ne fabrique pas une animation de progression qui n'est
adossée à aucune donnée réelle. C'est exactement le piège de l'écran de chargement :
la vraie forme, c'est une attente franche, puis le chart brut d'un coup.

**L'incertitude se montre, elle ne se cache pas.** Quand le modèle hésite, il dit
moins (`Am` au lieu de `Am7♭5`), il n'invente pas de la précision. La couleur de
confiance et le mot doivent toujours dire la même chose.

## Comment tu procèdes

Avant d'écrire une ligne, lis la fonction que tu remplaces en entier, commentaires
compris. Ce fichier est plein de commentaires datés qui expliquent pourquoi une
chose est comme elle est — un bug mesuré sur iPhone, une décision de Louis, un
piège d'arrondi. Presque chaque « simplification » évidente a déjà été essayée et
annulée. Si tu enlèves quelque chose que tu ne comprends pas, tu réintroduis le bug
que le commentaire décrit.

Change ce qu'on t'a demandé, et rien d'autre. Pas de refonte opportuniste, pas de
renommage, pas de « pendant que j'y étais ». Si tu vois autre chose à corriger,
signale-le, ne le fais pas.

Quand tu as fini, vérifie sur un vrai écran de 390 px de large. Deux tests, à passer
sans exception :

```js
// aucun bouton en dessous de 44 px
[...document.querySelectorAll('button')]
  .filter(b => { const r = b.getBoundingClientRect(); return r.width < 44 || r.height < 44 })
  .map(b => b.textContent)
```

```bash
# aucun emoji
grep -P '[\x{1F300}-\x{1FAFF}\x{2600}-\x{27BF}]' harmonia_min/app_shell.html
```

Les deux doivent revenir vides.

---

En cas de doute sur une décision visuelle, la maquette fait foi :
`Harmonia UI Refresh.dc.html`, badges `1a`, `1b`, `2a`, `2b`.
