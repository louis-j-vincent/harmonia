# Pourquoi on tourne en rond — diagnostic sur 37 jours d'échanges

*2026-08-12. Demandé par Louis : « scanne l'historique, détecte ce qui ne va pas,
mes fautes, tes fautes, comment améliorer. »*

Source : 522 transcripts, **1 594 prompts de Louis** (hors sous-agents), 6 949 de
mes réponses, du 3 juillet au 12 août. Plus le git : 448 commits depuis le 25/07.

---

## 1. Le fait qui explique tout le reste

**Tes 18 annotations de sections ont toutes été faites le 7 août au soir.**
`harmonia_min/state/sections/` : 18 fichiers, tous datés 7/08 07:43 → 8/08 00:03.
Aucun depuis.

Depuis 5 jours on optimise contre ces 18 morceaux — dont 12 sont « de conception »,
donc vus par les règles. J'ai construit les 9 et 10 août l'outil d'annotation qui
devait nous en donner d'autres : il a produit **2 fichiers**, et rien depuis le 9.

Ton propre document de design le dit (`docs/design_2026-08-12_section_model.md`,
§1) : *« le budget d'étiquettes choisit le modèle, pas l'inverse »*. Le budget est
gelé depuis 5 jours. Donc la seule variable qui bouge encore, ce sont les règles à
la main — et c'est exactement ce qui donne la sensation de tourner en rond.

---

## 2. Ce que la mesure dit

| Signe | Mesure |
|---|---|
| Sujets ouverts depuis le début | « sections/structure » : 322 prompts, **35 jours sur 37** |
| Morceaux qui ne se ferment jamais | *She Will Be Loved* 20/07 → 12/08 (**23 jours**), *This Love* 12 jours, *Norah Jones* 7 jours |
| Pages de rapport | **382** fichiers dans `docs/plots/`, dont **74 créés en 2 jours** |
| Outils que je lance par prompt de toi | mi-juillet **2–4** → août **12–25** |
| Mes « c'est bon » réfutés au prompt suivant | **64 / 751 (8,5 %)** |
| Résultats négatifs assumés | **13 commits sur 297** depuis le 1/08 |
| Travail qui redescend | **26 branches**, `main` en retard de **57 commits** depuis le 9/08 |

Lecture : beaucoup de production, peu de fermeture. Rien n'est jamais *fini*.

---

## 3. Le mécanisme, en trois boucles

**Boucle A — la règle du jour.** Tu donnes une règle (« les sections sont des
multiples de 2 », puis « la première section ne change que sur un multiple de 4 »,
puis « on n'a pas besoin de les séparer »). Je l'implémente. Elle sauve un morceau,
en casse un autre. Tu m'en donnes une nouvelle. 53 prompts de ce type ; **aucune
règle n'a jamais été retirée**. Le moteur de sections empile aujourd'hui ~14 règles
qui interagissent.

**Boucle B — le bug d'une mesure.** « Pourquoi mesure 45 sur Blue Lights ? »,
« pourquoi on ne détecte pas les C suivants ? ». Je corrige le cas. La règle CLAUDE.md
n°5 dit qu'un résultat sur un morceau est une *hypothèse* — mais on traite chacun
comme un *bug*. Résultat : *She Will Be Loved* est ouvert depuis 23 jours.

**Boucle C — le critère absent.** Tu as raison de dire que les chiffres n'arrêtent
pas une décision (5/08 : « les chiffres sont un indicateur, JAMAIS pour arrêter une
décision »). Mais on n'a jamais mis autre chose à la place. Sans critère
d'acceptation, tout est réouvrable, donc tout se rouvre.

---

## 4. Mes fautes

1. **Je fabrique une page neuve à chaque tour au lieu d'en faire converger une.**
   74 pages en 2 jours. Tu l'as dit le 9/08 : « je ne sais pas ce que je suis censé
   faire de tout ça ». → *Une seule page vivante par sujet, mise à jour, jamais
   dupliquée.*

2. **J'accepte chaque règle sans dire ce qu'elle contredit.** Jamais une fois je
   n'ai répondu « cette règle annule celle de mardi ». → *À chaque règle nouvelle :
   nommer celle qu'elle remplace, et mesurer les deux.*

3. **Je pars trop loin entre deux points de contrôle.** 2–4 outils par prompt en
   juillet, 12–25 en août. Ton contre-feu : « continue » ×5, prompts envoyés deux
   fois, interruptions. → *Point de contrôle court, ou budget annoncé à l'avance.*

4. **Je dis « c'est bon » avant de l'avoir vu.** 8,5 % réfutés immédiatement, encore
   les 7 et 10 août, alors que la règle « ouvre le rendu réel » existe depuis le 7.

5. **Je ne réfute pas tes idées.** 13 commits négatifs sur 297. La règle n°2
   (« cribler la prémisse avant d'implémenter ») je l'applique à mes idées, pas aux
   tiennes. C'est le plus coûteux des cinq : dire non tout de suite vaut mieux
   qu'une journée de règle #15.

6. **Le travail ne redescend pas.** 26 branches, `main` figé au 9/08.

---

## 5. Tes fautes

Factuel, tu me l'as demandé.

1. **Tu as arrêté d'étiqueter.** C'est la contrainte qui bloque tout le reste (§1),
   et c'est la seule chose que je ne peux pas faire à ta place.

2. **Une règle nouvelle arrive avant que la précédente soit jugée.** Elles se
   contredisent parfois à 3 jours d'écart. Tu n'en retires jamais.

3. **Tu débogues par la mesure, pas par la classe d'erreur.** « pourquoi mesure 45 »
   me pousse au correctif local ; « quelles sont les 4 façons dont on casse un C »
   me pousserait au correctif général.

4. **Tu interdis les chiffres sans mettre un critère d'arrêt à la place.** Tu as
   raison sur le fond en phase exploratoire — mais du coup rien ne peut être déclaré
   terminé, donc rien ne l'est.

5. **Le scope bouge en cours de tâche.** 36 prompts « et aussi / autre truc / quick
   fix » greffés sur une tâche en cours.

---

## 6. Le protocole que je propose

**Avant tout : deux heures d'annotation, 18 → 50 morceaux.** Rien de ce que je peux
coder cette semaine ne vaut ça. C'est ce qui débloque le modèle appris de ton doc
et met fin à la boucle des règles. Si l'outil te freine, dis-moi ce qui te freine —
je le répare avant, et je ne fais rien d'autre en attendant.

Ensuite, quatre règles d'échange :

| Règle | Qui | Concrètement |
|---|---|---|
| **Une page par sujet** | moi | `section_lab.html` mise à jour ; les anciennes archivées |
| **Une règle entre, une règle sort** | nous | j'annonce ce qu'elle contredit ; si aucune ne domine, on choisit explicitement |
| **Critère d'arrêt avant de commencer** | toi | en une phrase, sur une liste de morceaux nommée : « c'est fini quand … » — pas un chiffre, un truc écoutable |
| **Je réfute avant d'implémenter** | moi | crible de prémisse à 10 min sur *tes* idées aussi ; j'ai le droit de dire non |

Et une décision à prendre aujourd'hui : **on gèle le sujet « sections » 48 h**, le
temps des annotations. Sinon la règle #15 arrive demain matin.

---

## 7. Ce qui a été construit le jour même

Louis a choisi les quatre correctifs outillés. Ils sont en place.

| Fix | Où | Ce qu'il ferme |
|---|---|---|
| **`/ship`** | `.claude/skills/ship/SKILL.md` | La chaîne complète (banc → rendu → log → commit → prod → push) déclenchée sans qu'il la demande. Les ~120 prompts de plomberie. |
| **`scripts/bench.py`** | `--quick` en 2 s sur les 4 morceaux chauds | « Est-ce que ça régresse ? » avant chaque ship. Sort en code 1 sur tout recul > 0,005. Refuse de comparer quand l'empreinte de `section_metric.py` a changé — sinon un changement de règle passe pour un gain de modèle. **Découvre les morceaux annotés sur disque**, pas dans une liste figée : Louis en a annoté un de plus pendant la séance, une liste écrite à la main ne l'aurait jamais vu. |
| **Hook anti-pavé** | `.claude/hooks/verbosity_check.py` | Compte la prose de chaque réponse (code et tableaux exclus) et le dit au-dessus de 700 caractères. Le contrat de clarté devient mesurable au lieu de relu. |
| **Compteur d'étiquettes** | bandeau de `/reports/index.html` | Le budget d'annotations, lu en direct sur `/api/sections`, en haut de la page qu'il ouvre. La contrainte de la section 1 est sous les yeux au lieu d'être dans un document. |

Et les quatre règles sans code, adoptées le même jour : je ship sans qu'on me le
demande ; je finis par **une** prochaine étape nommée, jamais un menu ; jamais de
tour muet pendant un long run ; le détail va dans la page, le chat garde une
phrase et un lien.

**Premier signal, le jour même :** le compteur est passé de 18 à 19 pendant la
séance, et le nouveau morceau (*Goodbye Yellow Brick Road*) est entré tout seul
dans le banc — il sort à **0,441**, le plus bas de tous. C'est un vrai tenu
hors conception, et il dit que le moteur de sections généralise moins bien que
ne le laissaient croire les 18 précédents.
