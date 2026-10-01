# 33 — Une brique et ses reprises

*2026-09-30 → 2026-10-01*

Deux jours sur une seule question : une fois que Louis a tracé UNE section,
où le morceau la rejoue-t-il, et comment l'écrire ? Elle a fini par une
mesure de ressemblance entre accords, branchée sur un curseur. Ce billet
raconte surtout COMMENT on y est arrivé, parce que le chemin a été meilleur
que chacune de mes propositions prises une à une.

## Un faux départ, d'abord

La séance commence par une erreur. Louis : « le chart commence toujours par
un % ou un N.C. ». La première session (Haiku) a trouvé un N.C. en tête de
la liste à plat, l'a supprimé et a raconté une histoire de « dédoublonnage
désynchronisé ». Personne n'avait ouvert un chart rendu. Le commit (1787a03)
ne corrigeait rien, il cassait même les vraies intros N.C. des morceaux
découpés à la main. Pire : la même session a vidé `state/cache/charts/min_*`.

La vraie cause, trouvée par un agent qui a d'abord **caractérisé** au lieu de
corriger : la grille de mesures part du premier temps entendu par le traqueur,
et la musique arrive 1 à 7 mesures plus tard. Avant la sortie de SongFormer
(19/09), ces mesures muettes allaient dans une intro cachée. Depuis, elles
s'affichaient en tête de A. Correctif : `pipeline._one_section` rétablit
l'intro (e2d26cb), 1787a03 annulé, bibliothèque recuite. Louis : « clairement
mieux le après ». Puis « vire l'ancienne façon » : le chemin SongFormer entier
supprimé (717dbf0, −775 lignes, 0 mesure changée), baseline du rapport d'or
regelée.

Leçon (règle 1 et « verify what a thing DOES » du CLAUDE.md) : un fix qui ne
regarde pas le rendu n'est pas un fix.

## La recherche de reprises : Louis a donné les règles, les pages les ont affinées

Le diagnostic de départ (`occurrences_brique_2026-09-30.html`) montrait trois
défauts : un seuil plafonné à 0,999 qui rejetait des reprises à 0,99, une
brique cherchée seulement à sa longueur exacte, et des « jumeaux » harmoniques.
J'avais proposé de baisser le plafond à 0,97. Louis a refusé le réglage et
donné des **principes** à la place :

1. pas de pic dans la zone qu'il a tracée lui-même ;
2. une brique est peut-être plusieurs tours d'une boucle plus courte : c'est la
   **boucle minimale** qu'on cherche (San Francisco Street réparé) ;
3. un pic se juge **par rapport à ses voisins**, pas en absolu.

Chaque règle a été livrée avec une page terme par terme (v2, v3, v4, v5) sur
les mêmes morceaux, jamais un score global. C'est ce qui a permis à Louis
d'arbitrer vite. Puis, page après page :

4. la meilleure reprise hors zone **calibre le morceau** ; on coupe sous le
   plus grand trou entre hauteurs de pics, si le bas est majoritaire (Let It
   Be : refrain B refusé, Stand By Me intact) ;
5. entre deux départs voisins, garder celui qui domine ses voisins à ±(boucle−1)
   (The Walk dérivait de 2 mesures parce que l'outil maximisait le NOMBRE de
   boucles casées). La version stricte perdait Lazy Song et She Will Be
   Loved : **5b** (égalité = tolérance du morceau) et, idée de Louis, les
   ex æquo restent **en suspens** jusqu'à ce que les autres sections « aient
   mangé ce qu'elles ont à manger » ;
6. option (c) : un suspens n'est pris par une autre section que si elle est
   nettement plus haute, ce qui rend le résultat indépendant de l'ordre de
   tracé (Sunny réparé).

Plus : zone libre de tête = intro ; un bloc qui joue B puis continue = B +
queue (Julia) ; une mesure muette n'est jamais le modèle de sa lettre
(Lost Without U) ; dernière section qui rejoue A puis s'éteint = dernier tour
de A.

## L'écriture : boucle minimale, puis le curseur

Écrire chaque section par sa boucle minimale (This Love A : 8 → 4, ‖: :‖ ×2)
a d'abord été trop loin : les deuxièmes moitiés de This Love, harmoniquement
différentes, partaient en petits accords suggérés. Louis : « ne la fais pas
quand il y a vraiment un changement important, mets-moi le curseur ». C'est
devenu le curseur **Section merging** (0 jamais → 4 toujours), défaut = on
factorise seulement si l'harmonie est la même.

## La mesure de ressemblance

Pour Lost Without U (G7sus4 joué à la place de D-7), j'ai proposé une
équivalence de plus, écrite à la main. Louis : non, « il faut une mesure de
similarité reproductible et généralisable, avec le contexte ». Toutes les
comparaisons passent d'abord par une seule fonction (`roles.meme_lecture`).
Puis 7 candidates ont été comparées sur une liste de paires
(`similarite_accords.html`). Surprises honnêtes : la mesure apprise de musx ne
rapproche presque jamais deux fondamentales (D-7/G7sus4 à 0,03), et le
contexte de tonalité est trop grossier. La plus simple gagne : **les notes
jouées, sans 9/11/13**.

La v1 mettait G7 et Gm7 à 0,75 : faux pour un jazzman. Verdicts de Louis :
la tierce pèse (familles différentes), un sus4 est proche des deux, et la
tolérance au rythme des accords dépend du morceau, donc du curseur. Formule
finale : tierces opposées ×0,4, sus sans tierce neutre, deux fondamentales
tiercées ×0,8 sauf inclusion. G7/Gm7 0,30 · G7sus4/G7 1,00 · D-7/G7sus4
0,75 · C/F 0,27. Curseur : 0–1 (seuil 0,72, rythme exact) · 2 (0,63, ±1
temps) · 3–4 (0,38, ±2 temps). Au défaut, seuls This Love (intro en boucle,
validé) et She Will Be Loved (mes. 33) changent. Louis, sur les captures :
« c'est parfait, c'est ce que je voulais ».

## Ce qui a marché dans le raisonnement

- **Caractériser avant de corriger.** Les seuls vrais faux pas de ces deux
  jours (1787a03, le plafond à 0,97, l'équivalence codée en dur) étaient des
  corrections proposées avant d'avoir regardé.
- **Des principes plutôt que des seuils.** Chaque fois que j'ai proposé un
  nombre, Louis a répondu par une règle relative (voisins, morceau, contexte).
  Les règles ont tenu, les nombres non.
- **Une page par itération, sur les mêmes morceaux.** v2→v5, puis captures du
  vrai rendu : Louis arbitrait en une phrase.
- **Dire le coût.** Chaque règle est arrivée avec ses pertes (The Walk B,
  X-yIEMduRXk, Lazy Song, Sunny) : c'est ce qui a produit 5b, le suspens et
  l'option (c).
- **Déléguer, mais vérifier.** Les agents ont fait le travail ; mais un
  `--clean` a effacé 44 avant/après publiés, et `--publish --backup` ne
  sauvegardait rien (corrigé, 152f256). Ces deux-là, je ne les ai vus qu'en
  relisant leurs rapports et le dossier.

## Ce qui reste

- Lost Without U au cran 3 : les N.C. du fondu final disparaissent du chart.
- h_D3VFfhvs4 refusé par le traqueur (rythme trop libre pour une grille fixe).
- Les deux « outros » qui sont des A (Lost Without U, Be My Baby, plus douteux)
  sont à renommer à la main dans l'outil.
- Disque à 98 % (5 Go libres) : une quarantaine de vieilles cuissons dans
  `state/cache/golden/`.
