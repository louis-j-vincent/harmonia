# 28 — Un modèle qui reconnaît le refrain (au lieu de chercher la répétition)

*2026-08-18*

Louis, après avoir écouté la page de comparaison : « je suis d'accord avec lui
partout, on le prend en prod ».

## Ce qui a changé de nature

Depuis le début du projet, tous nos détecteurs de sections font la même chose :
ils cherchent une **répétition**. Le premier la cherchait dans le chroma brut,
le deuxième dans un dictionnaire de motifs d'accords, le troisième dans la ligne
de chant. Trois substrats, une seule idée — *ce passage ressemble à celui-là,
donc c'est le même*.

SongFormer ne fait pas ça. Il **reconnaît** un refrain à sa texture, comme un
auditeur qui n'a jamais entendu la chanson sait quand même qu'on vient d'entrer
dans le refrain. Il trouve donc des frontières qu'aucune répétition ne trahit,
et — ce qu'aucun des nôtres ne savait faire — il les **nomme**.

## Comment il marche

Lu dans son code, pas dans son README.

1. Le morceau est rééchantillonné à 24 kHz.
2. Il passe dans **deux encodeurs audio pré-entraînés**, MuQ et MusicFM, dont on
   prend la 10ᵉ couche cachée. Ce sont des modèles de fondation musicaux : ils
   ne sortent ni accords ni notes, mais un vecteur par tranche de temps qui
   encode « à quoi ça ressemble ».
3. Chacun voit le morceau **à deux échelles** : par fenêtres de 30 s (le détail
   local) et par fenêtres de 420 s (le morceau entier d'un coup). Quatre flux de
   vecteurs, concaténés.
4. Un **Transformer**, entraîné sur des morceaux annotés à la main, lit cette
   bande et sort deux courbes à 8,33 images par seconde :
   - *y a-t-il une frontière ici ?*
   - *quelle étiquette ?* (intro, couplet, refrain, pont, instrumental, outro,
     silence, pré-refrain)
5. Les frontières sont les **pics locaux** de la première courbe. L'étiquette
   d'un segment est la **moyenne** de la seconde entre deux frontières, argmax.
6. Quelques règles finales recollent les segments d'une seconde en tête et en
   queue.

Les deux échelles sont le vrai truc : la fenêtre de 30 s entend qu'il se passe
quelque chose ici, la fenêtre de 420 s sait que *ce quelque chose* est le même
qu'à la minute 2.

## L'arbitrage

Pas un chiffre. Une page : trois bandes par morceau sur le même axe de temps —
son annotation, SongFormer, notre détecteur — chaque bloc jouable d'un tap. Il a
écouté, il a tranché. C'est la forme de livrable qu'il réclame depuis le 9 août
(« je ne veux JAMAIS juste voir le résultat »), et c'est la deuxième fois
d'affilée qu'elle règle en une écoute une question qu'un tableau de scores
aurait laissée ouverte.

## Le prix d'entrée

Trois frictions, toutes invisibles depuis la documentation du modèle :

- `AutoModel.from_pretrained` ne le charge pas avec transformers 5.x — le modèle
  est construit sous le device « meta » et son MuQ interne y mélange des
  tenseurs cpu et meta. Il faut importer la classe et la construire soi-même.
- Les poids **ne sont pas** dans le fichier qui porte le nom du modèle :
  `SongFormer.safetensors` n'est que la copie EMA de la tête (62 tenseurs),
  `model.safetensors` porte les 1052.
- Les dépendances du dépôt veulent faire monter torch de 2.12 à 2.13, ce qui
  casserait beatthis, demucs et musx d'un coup. Installées `--no-deps`.

Ça tourne sur CPU (les noyaux de MuQ ne passent pas tous sur MPS) : une minute
de chargement, puis quelques dizaines de secondes par morceau, mis en cache.

## Ce qu'il a fallu réparer en aval

Donner la même lettre à toutes les occurrences d'un rôle n'était pas gratuit.
Le refrain d'*Another Day* est joué cinq fois et mesure 9, 7, 8, 8 puis 12
mesures — la frontière du modèle tombe au demi-temps près, la nôtre s'arrondit à
la barre. Or `minimal_fold` groupait les occurrences **par lettre seule** et
écrivait un seul bloc à la longueur du représentant : la tête de lecture se
décalait d'une barre à chaque reprise. C'est exactement le bug que Louis avait
signalé le matin même.

Le chemin Soudure, lui, groupait déjà par (lettre, **longueur**) — la règle
« under-fold, never over-fold ». Les deux chemins rendaient donc deux charts
différents pour le même découpage. `minimal_fold` fait pareil maintenant : deux
longueurs d'un même refrain restent deux blocs, tous deux nommés B, chacun écrit
à la longueur qu'il joue vraiment.

La tentation était de séparer les longueurs **dans le détecteur** — c'était plus
simple, et c'était faux : le chart aurait annoncé quatre sections là où l'oreille
entend « le refrain, cinq fois ». La règle appartient à l'endroit où elle a un
effet, pas à l'endroit où elle est facile à écrire.
