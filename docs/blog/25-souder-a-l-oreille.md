# 25 — Souder à l'oreille

*2026-08-13.*

Le brief tient en cent lignes (`.claude/handoff-others/SOUDURE_BRIEF.md`) et il
part d'une observation simple : `scripts/bpe_lab.py` **déroule** l'agglomération
des bi-mesures et la donne à lire, ligne par ligne, sur un graphique. On voit
l'ordre dans lequel le morceau s'agglomère. Ce qu'on ne peut pas faire, c'est
l'arrêter au bon endroit — parce que le critère d'arrêt n'existe pas.

La page Soudure retourne le problème : elle ne déroule plus rien toute seule.
Louis soude, une paire à la fois, en écoutant. La machine ne fait qu'une chose,
proposer la paire la plus fréquente restante. Le jugement reste à l'oreille.

## Ce qu'on a construit

`docs/plots/soudure.html` — un seul fichier, tout en ligne, aucune ressource
externe. La chanson est une bande de jetons ; deux jetons de même contenu sont
le même endroit du morceau, donc même couleur et même lettre. On soude une paire
voisine et **la soudure s'applique partout à la fois** : un geste, N fusions.
C'est BPE, exactement l'algorithme de `bpe_lab`, mais conduit à la main.

Deux modes. *Écouter* : un tap joue une entité en boucle, un deuxième tap en
sélectionne une autre et les enchaîne — c'est comme ça qu'on entend si deux
endroits sont vraiment le même. *Souder* : un tap, puis sa voisine, et la
cascade part.

`scripts/soudure_pages.py` écrit une copie de la page par morceau, avec le
`window.SONG` du morceau devant : le mot vient de `vote_fill.fill`, la même
matière que `bpe_lab`, et le vrai disque est jouable plage par plage. Douze
morceaux, index sur `/plots/soudure_lab.html`.

## Trois choses qui ont demandé une décision

**Le compte annoncé est le compte réel.** Le compagnon dit « a + a · 16 fois ».
Ce n'est pas le nombre de paires adjacentes (26 sur l'exemple) mais le nombre de
fusions qui vont *vraiment* avoir lieu, de gauche à droite et sans
chevauchement — sinon `aaa` se mangerait lui-même. Le badge de la cascade
annonce le même chiffre, donc l'annonce et le résultat ne peuvent pas diverger.
Le brief donnait « 16 » comme exemple sur Blue Lights : c'est bien ce compte-là.

**Les couleurs ne peuvent pas se ressembler.** La palette du brief contient deux
bleus séparés de 71 unités perçues (médiane de la palette : 167). Deux endroits
différents du morceau tombaient côte à côte en deux bleus voisins, et toute la
lecture « même couleur = même endroit » s'effondrait. L'attribution reste le
hachage du contenu — donc stable d'un état à l'autre — mais on s'en écarte quand
la couleur serait confondue avec une déjà posée.

**Les jetons ne font pas tous deux mesures.** Sur She Will Be Loved, un jeton en
fait trois : le morceau porte une mesure impaire au milieu. Supposer un pas
constant décalait tout l'audio après cette mesure. La page accepte donc des
bornes de jetons explicites (`SONG.jetons`) et retombe sur le pas constant quand
on ne lui en donne pas.

## Le son ne sortait pas — et c'était un bug déjà payé

Première réaction de Louis : « ça ne play pas les morceaux, je n'entends rien ».

La page pilotait la tête de lecture avec une boucle `requestAnimationFrame` qui
écrit dans le DOM soixante fois par seconde. C'est exactement le motif que
`harmonia_min/app_shell.html` documente depuis le 2026-07-20 : sur iPhone, cette
boucle empêche le moteur audio de WebKit de démarrer — `currentTime` reste gelé
à 0 tant que la page est au premier plan, et le son ne revient qu'une fois
l'onglet mis en arrière-plan. L'app a été corrigée en passant le tic sur
l'événement `timeupdate`, c'est-à-dire sur l'horloge du média elle-même.

La page fait maintenant pareil : plus une seule boucle d'animation. La fin de
plage est armée par un `setTimeout` calculé sur l'horloge du média et
re-corrigée à chaque `timeupdate`. Vérifié sur le rendu servi : la tête avance
de 1,61 s en 1,6 s de temps réel, `readyState` 4, et la boucle repart bien de
21,0 s à 11,6 s en fin d'entité.

Deux garde-fous ajoutés au passage. Un chien de garde : si au bout d'une
seconde et demie la tête n'a pas bougé, on réessaie avec le blob, et si ça ne
part toujours pas **la page le dit** — une page muette ne doit pas faire
semblant de jouer (le `.catch(() => {})` qui avalait le refus était exactement
le silence qu'on s'interdit ici). Et les liens de l'index portent l'empreinte de
la page : Safari garde les pages en cache avec entêtement, et une correction
invisible parce qu'on relit l'ancienne version coûte un aller-retour pour rien.

**La leçon, pour la prochaine fois :** avant d'écrire une boucle de rendu dans
une page qui joue du son, lire ce que l'app a déjà appris. Le commentaire de
`app_shell.html` était là, daté, avec la cause et la correction.

## Ce que ça ne résout pas

**Rien du critère d'arrêt.** La page déplace la décision vers l'oreille au lieu
de l'automatiser — c'est l'intention, mais il faut le dire : après cette séance
on ne sait toujours pas écrire la règle qui dit « arrête-toi ici ». Ce qu'on
gagne, ce sont des découpages faits à l'oreille, exportables en JSON, sur
lesquels une règle pourra être testée plus tard.

**L'agglomération s'arrête tôt.** Le compagnon ne propose que des paires qui se
répètent au moins deux fois, comme `bpe_lab`. Sur This Love ça s'arrête à 15
entités (`A A A A B C A A A B C d e f g f h B C B C B C`), pas à 5 : les derniers
tours de `bpe_lab` soudent des entités qui ne se répètent qu'une fois. La page ne
les propose pas ; rien n'empêche de les souder à la main.

**Le mot est celui de la basse**, avec tout ce que ça implique — la page hérite
de la qualité du mot, elle ne la répare pas.

## Où c'est

* `http://100.89.209.63:7772/plots/soudure_lab.html` — les douze morceaux
* `http://100.89.209.63:7772/plots/soudure.html` — l'exemple du brief (c'est
  Blue Lights, 84 mesures)
