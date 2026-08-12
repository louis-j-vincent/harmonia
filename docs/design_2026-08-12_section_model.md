# Un modèle qui apprend le découpage en sections — proposition d'architecture

*2026-08-12. Louis : « tu peux prendre tous les ingrédients qu'on a et me faire un
modèle ML qui apprend de tout ça ? quelle architecture proposes-tu ? »*

---

## 1. La contrainte qui décide de tout : on a 18 chansons étiquetées

Avant l'architecture, le budget d'étiquettes, parce que c'est lui qui choisit le
modèle et pas l'inverse.

| source | morceaux | sections | audio ? | ce qu'on peut en tirer |
|---|---|---|---|---|
| tes annotations validées | **18** | 214 | oui | **les 7 matrices** |
| McGill Billboard (sur disque) | 890 | 8 753 | **non** | chroma NNLS seulement |
| `docs/audio` sans annotation | 77 | — | oui | pré-entraînement non supervisé |

Billboard est sur le disque (`~/mir_datasets/billboard`, 1 Go) avec ses chromas
Chordino, **mais sans audio**. Donc les trois substrats qui portent la grande
échelle — timbre, rythme, voix — n'y existent pas. Le grand corpus n'étiquette
que les canaux qu'on a mesurés comme les plus faibles pour la question qui nous
intéresse.

**Conséquence directe sur le design** : tout ce qui concerne la fusion
inter-substrats sera appris sur 18 morceaux. Le modèle doit donc être petit,
régularisé, et validé morceau par morceau (leave-one-song-out), jamais sur une
moyenne.

---

## 2. L'architecture proposée : deux têtes et un décodeur

```
        les 7 matrices SSM (demi-mesure)  +  voix muette  +  accords
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
   ÉTAGE 1                                     ÉTAGE 2
   score de frontière                          plongement de section
   (par mesure)                                (par segment)
   TCN dilaté + porte par substrat             MLP + perte contrastive
        │                                           │
        └─────────────────► DÉCODEUR ◄──────────────┘
                     semi-markovien (DP)
              durée + frontière + identité de lettre
```

### Étage 1 — le score de frontière, par mesure

**Entrée** (~110 nombres par mesure, tous déjà calculés par nos scripts) :

* nouveauté en damier de chacune des 7 matrices, à **4 largeurs de noyau**
  (2, 4, 8, 16 mesures) → 28. Les largeurs multiples remplacent le choix
  arbitraire de 8 mesures qu'on fait aujourd'hui ;
* similarités à distance fixe `S[b, b±L]`, `L ∈ {2,4,8,16}`, par matrice → 56.
  C'est ce qui dit « la mesure b se répète 8 plus loin », l'information de
  période ;
* voix : muette, début et fin de silence → 3 ;
* accords : changement sur cette mesure, nombre de changements, rythme de
  changement identique à la mesure précédente → 3 ;
* position : `b mod 2/4/8`, `b/n`, distance aux deux bouts → 6 ;
* **contexte du morceau**, diffusé sur toutes les mesures : le contraste de
  chaque matrice et son étalement hors diagonale → 14.

**Corps** : un TCN dilaté (ou un BiGRU) de champ réceptif ~32 mesures, largeur
64, ~40 k paramètres. Sortie : un logit de frontière par mesure.

**La porte par substrat, et c'est le cœur de la proposition.** Une petite tête
d'attention sur les 7 canaux, conditionnée par le vecteur de contexte du
morceau, produit 7 poids qui multiplient les canaux avant le corps. C'est la
version APPRISE de notre `contraste` : aujourd'hui on écrit à la main « sur ce
morceau, écoute le timbre à 36 % et les accords à 2 % », et cette règle a été
mesurée meilleure que la netteté — mais elle reste une heuristique à un seul
nombre. La porte apprend la même chose avec le droit d'être non linéaire et de
dépendre de l'endroit du morceau.

**Perte** : BCE avec lissage d'étiquette à ±1 mesure. Ce n'est pas un détail —
nos pics ne tombent exactement juste que dans 49 % des cas et à ±1 mesure dans
79 %, et une partie du résidu vient de la **levée** (le chanteur entre une mesure
avant la barre) et de la grille de mesures elle-même. Punir un décalage d'une
mesure comme une erreur franche apprendrait du bruit.

### Étage 2 — les lettres, par apprentissage de métrique

Un segment → un vecteur : postérieurs d'accords moyennés repliés sur 12
hauteurs, chroma basse et aiguë, MFCC (moyenne et écart-type), profil de chant,
durée. MLP → plongement 32-d, perte contrastive (deux occurrences de la même
lettre = paire positive).

**Augmentation par transposition** : on fait tourner les 12 hauteurs de k
demi-tons pour les paires positives. C'est la réparation directe du cas Sunny,
où le morceau monte d'un demi-ton à chaque reprise et où nos lettres cassent.
`voice_sections._rot_sim` fait déjà ça à la main sur un seul chemin.

Remplace le seuil `SAME = 0,94` réglé à l'œil.

### Le décodeur — semi-markovien

On ne prend pas l'argmax mesure par mesure. On cherche le découpage complet qui
maximise

```
Σ_segments [ frontière(début) + durée(longueur) + identité(segment, lettre) ]
```

par programmation dynamique en O(n × durée_max × lettres).
`harmonia/models/semi_markov_decode.py` existe déjà et fait cette DP.

**C'est là que tes règles deviennent des potentiels, pas des `if`** :

| ta règle | où elle vit dans le modèle |
|---|---|
| « une section ne traverse jamais un pic » | pénalité forte sur une frontière interne à un segment, apprise au lieu d'être un veto |
| « cœur pair + queue amovible » | le **prior de durée**, appris : il piquera à 4/8/12/16 avec une petite masse à 1–3 pour les queues |
| « un A de 16 s'écrit deux A de 8 » | le prior de durée décroît au-delà de 8–12, donc deux segments coûtent moins qu'un long |
| « la voix muette = solo / pont » | une lettre réservée, avec sa propre vraisemblance |

---

## 3. Le protocole, et il compte autant que l'architecture

1. **Pré-entraînement sur Billboard** (890 morceaux, 8 753 sections) avec les
   seuls canaux harmoniques disponibles, en **abandon de canal** (channel
   dropout) systématique : le modèle ne doit jamais supposer que les 7 canaux
   sont là, puisqu'à ce stade il n'en a que 4.
2. **Affinage sur tes 18**, tous canaux. C'est là qu'on apprend la porte.
3. **Validation un-contre-tous par morceau**, et le rapport est **par morceau**,
   avec `section_metric.compare` (ta métrique) plus précision/rappel de frontière
   à ±1 mesure. Jamais une moyenne seule.
4. **Référence à battre** : l'état actuel, `prod + pics durs + queue`, à 0,789 de
   médiane (prod seule 0,781). Un modèle qui ne dépasse pas ça n'a rien apporté.

---

## 4. Le premier pas, avant d'écrire une ligne du modèle

**Une régression logistique sur exactement ces features**, un-contre-tous sur les
18, contre le profil fusionné actuel. Deux heures de travail.

Si un modèle linéaire sur nos features ne bat pas la règle à la main, le TCN ne
le fera pas non plus : ça voudra dire que l'information n'est pas dans les
features, et il faudra retourner aux substrats plutôt qu'au modèle. C'est la
règle n°2 du projet — écrire le test le moins cher qui puisse tuer l'idée, et le
lancer en premier.

---

## 5. Ce qui peut rater, dit d'avance

* **18 morceaux.** Tout repose sur le fait que nos features sont déjà très
  travaillées ; la porte inter-substrats est la partie qui sur-apprendra.
* **Billboard n'a pas d'audio**, donc timbre / rythme / voix ne sont jamais
  pré-entraînés. C'est précisément l'inverse de ce qu'il faudrait : ce sont eux
  qui portent la grande échelle.
* **Les conventions diffèrent.** Un modèle entraîné sur Billboard apprend le
  découpage de Billboard, pas le tien — tes queues hors section, ta préférence
  pour les mesures paires, ta façon de couper les reprises adjacentes.
* **Le plafond de la grille.** 79 % à ±1 mesure vient en partie de la grille de
  mesures et de la levée. Aucun modèle de frontière ne répare une grille fausse.

**Et donc, l'action la plus rentable n'est peut-être pas le modèle.** Vingt
morceaux annotés de plus, avec audio, valent probablement plus que n'importe quel
changement d'architecture — la page d'annotation existe déjà
(`/reports/annotate.html`). C'est le seul chiffre du tableau du §1 qu'on peut
faire bouger nous-mêmes.
