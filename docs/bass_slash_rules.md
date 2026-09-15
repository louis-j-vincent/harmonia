# Quelle note écrire à la basse — règles arbitrées à l'oreille

**2026-09-15.** Chaque règle ci-dessous a été tranchée par Louis sur des cas
réels, en deux tours d'écoute. Ce ne sont pas des réglages : ce sont des
constats. Les 33 arbitrages sont conservés dans
`state/human/bass_verdicts.json` (vérité terrain, ne pas régénérer), le code
est `harmonia/bass_rules.py`, et `tests/test_bass_rules.py` rejoue les
arbitrages contre le code — **une règle qui bouge doit faire rougir ce test.**

Corpus : 351 accords de 6 morceaux (Ready · Stand By Me · This Love ·
Let It Be · Bora Bora · Sunny Afternoon), tous à caches chauds.

---

## D'où vient le problème

La basse d'un accord se lit par `nnls_features.bass_pc_onset` : la classe de
hauteur la plus forte dans le grave sur les **150 premières ms** de l'accord.
Lire l'attaque plutôt que moyenner tout l'accord était déjà un gain mesuré
(6/11 → 11/11 sur Ready, voir `docs/known_issues.md`), parce que la quinte
harmonique et les notes de passage s'accumulent avec la résonance.

Mais une sonde unique au temps 1 se trompe : **sur les 351 accords, 20 %
recevaient une basse différente de la fondamentale**, donc un slash — bien
trop pour de vrais charts, et Stand By Me en recevait 10 sur 19.

## Les trois branches

Dans l'ordre, la première qui répond gagne :

1. **La fondamentale de l'accord est retrouvée** ailleurs dans l'accord, à
   ≥ 30 % d'énergie grave → c'est elle la basse. Le temps 1 s'était trompé.
2. **Sinon la quarte au-dessus de la fondamentale** (famille *sus*) est
   retrouvée à ≥ 30 % → c'est elle. C'est le cas `Bb-7/Eb` : Bb est la quinte
   d'Eb, donc Eb porte.
3. **Sinon on garde la basse du temps 1**, mais seulement si l'intervalle
   qu'elle forme avec la fondamentale est jouable. Sinon : accord nu.

Résultat sur les 351 accords : **70 slashes → 28** (20 % → 8 % des accords),
**zéro intervalle impossible**, et zéro slash ajouté — la règle ne fait que
retirer ce que la sonde du temps 1 avait inventé.

| | avant | après |
|---|---|---|
| Ready | 18 | 8 |
| Stand By Me | 10 | **1** |
| This Love | 25 | 13 |
| Let It Be | 3 | 3 |
| Bora Bora | 12 | 3 |
| Sunny Afternoon | 2 | 0 |

Profil final des 28 : M3 ×8, 5te ×7, 9 ×4, 4te/11 ×4, 7M ×4, m3 ×1.

---

## Règle d'or 1 — la quinte est asymétrique

Entre deux lectures de basse d'un même accord :

* **+7 (quinte au-dessus)** — la seconde est la quinte de la première : la
  première est la fondamentale, la seconde la décore. **Veto.**
* **+5 (quarte au-dessus)** — une quarte au-dessus est une quinte en dessous :
  c'est la *première* qui est la quinte de la seconde. La seconde porte, et
  elle a la **priorité**.
* **0 (octave)** — décoration.

**Ce que ça a coûté d'apprendre :** appliquée *sans* la direction (« entre
deux notes à la quinte, garder la plus grave »), la règle fait **6/11 → 4/11**
— elle casse plus qu'elle ne répare. Elle a raison sur Ab^7 (Eb est un
harmonique parasite d'Ab) et tort sur Bb-7, Eb7, F7, où la « quinte du
dessus » est une vraie quarte ascendante jouée — un des mouvements de basse
les plus courants. L'intervalle seul ne distingue pas les deux ; la direction,
si.

## Règle d'or 2 — le retour ne démonte pas une quarte *(socle effondré)*

Si la basse de départ **revient** après une autre note (A‑B‑A), c'est
normalement le signe que B décore. **Sauf si B est à une quarte** : 4
arbitrages (c04, c05, c11, c12) disaient que c'est B qui porte, même quand A
retombe sur le temps fort.

> **Révision du 2026-09-15, au soir.** Cette règle n'a plus de preuve
> propre. Trois de ses quatre appuis (c04, c05, c12) ont le candidat **égal à
> la fondamentale de l'accord** : la règle d'or 3 les explique déjà, sans rien
> savoir du retour. Le quatrième, c11, s'est révélé être un **changement
> d'accord manqué** (voir « Les deux cas ouverts », plus bas). Rien ne
> contredit la règle 2 — mais plus rien ne la soutient seule, et la procédure
> livrée ne l'utilise pas (`is_decoration` n'est appelée nulle part).
> À re-arbitrer sur des cas neufs avant de s'en servir.

Ce qui reste solide, c'est l'asymétrie elle-même, mesurée dans l'autre sens
(règle d'or 1) : lever le veto de retour sur les quartes faisait passer le
1er tour de **10/17 à 14/17** — mais ce gain est entièrement repris par la
règle 3, qui fait 16/17 toute seule.

## Règle d'or 3 — la fondamentale de l'accord tranche mieux que toutes les mesures

C'est le résultat le plus net des deux tours. Le discriminant n'est ni la
durée, ni la part d'énergie, ni le poids métrique, ni le retour : c'est
**« le candidat est-il la fondamentale de l'accord ? »**

| règle | score sur le 1er tour |
|---|---|
| 5 mesures, 5 curseurs | 10/17 |
| + retour non appliqué à la quarte | 14/17 |
| **candidat == fondamentale + part ≥ 30 %** | **16/17** |

Le plancher de 30 % est au milieu d'un plateau large (les arbitrages tiennent
de 20 % à 32 %), donc ce n'est pas un réglage fragile.

**Ce que ça dit du mécanisme, et c'est le vrai enseignement :** dans ces
cas-là, le candidat n'est pas une deuxième basse — c'est **la bonne basse que
la sonde de 150 ms a ratée**. Sur Stand By Me, le temps 1 lit Eb sur un accord
de E, ou Db sur un accord de A. Le détecteur d'accords, lui, savait. La
question n'est donc pas « note de passage ou slash » mais « laquelle de mes
lectures est la bonne », et l'accord donne la réponse gratuitement.

Au 2e tour, cette branche est **6/6 juste**.

## Règle d'or 4 — c'est l'intervalle qui jette une basse, jamais la confiance

Quand ni la fondamentale ni la quarte ne sont retrouvées, il faut décider si
on garde la lecture du temps 1. **Un plancher de confiance ne marche pas** :
testé à 35 %, il se trompe **3 fois sur 4**, et le seul rejet justifié était
le **plus confiant** des quatre.

| cas | ligne | intervalle | temps 1 | verdict |
|---|---|---|---|---|
| r16 | `F/G` | +2 | 27,3 % | le slash était **vrai** |
| r17 | `Db/Eb` | +2 | 24,7 % | le slash était **vrai** |
| r19 | `E/Gb` | +2 | 27,9 % | le slash était **vrai** |
| r20 | `F-7/Gb` | +1 | 32,4 % | le slash était **faux** |

L'intervalle sépare parfaitement. `F/G`, `Db/Eb`, `E/Gb` sont des **sus** —
une 9e à la basse est un son courant. Une **b9** à la basse est un frottement,
et n'est en pratique jamais autre chose qu'une erreur de lecture.

**Intervalles jouables** (depuis la fondamentale) : 0, 2, 3, 4, 5, 7, 10, 11.
**Impossibles** : 1 (b9), 6 (b5), 8 (b13), 9 (6te).

> À noter : j'avais d'abord classé la 9e (+2) comme douteuse — c'était mon
> a priori, pas une mesure, et il jetait trois vraies basses. L'oreille de
> Louis l'a corrigé.

---

## Les deux cas ouverts, tranchés — et aucun n'était une question de basse

**Ready, mesure 3** (`o1`, ex-`r10`). Le chart écrit `Eb`, Louis tranche
**`Ab^7`**. L'étiquette d'accord était fausse, pas la basse : une fois
l'accord juste, aucun slash n'est nécessaire, et la règle avait raison de
retirer le `/Ab` sur l'accord qu'on lui donnait.
→ **La règle de basse ne peut être juste que si l'accord l'est.**

**Ready, mesure 49** (`o2`, ex-`c11`). Le chart écrit un seul `Bb-7` de
123,553 s à 128,012 s — 4,5 s, **à cheval sur deux mesures**. Louis entend
**deux accords : `Bb-7` puis `Eb6b9`**. Ce n'était donc pas un slash mais un
**changement d'accord manqué**. Ce que disent les mesures :

* musx tient `Bb-` de **73 % à 95 %** sur toute la durée et ne voit jamais
  `Eb` (2,3 % au mieux) — le détecteur d'accords est **aveugle** à ce
  changement ;
* la lecture de basse, elle, pose `Eb` sur le **temps 1 de la seconde mesure**
  (125,48 s, 31,7 %) — c'est le seul signal qui existe.

→ **L'inverse de la règle d'or 3.** Là, c'est la basse qui voit ce que
l'accord rate. Les deux sens existent, et le règlement ne couvre aujourd'hui
que le premier.

---

## Ce que ces règles ne règlent PAS

* **Le corpus est de 6 morceaux.** Les tables tiennent sur 351 accords et 33
  arbitrages, dont beaucoup choisis pour être *limites*. Avant tout passage en
  prod : rejouer sur le reste de la bibliothèque.
* **Rien n'est branché dans `pipeline.py`.** `bass_rules.decide_bass` existe et
  est testé, mais aucun chart ne l'utilise ; `c["bass"]` continue de venir du
  label décodé par musx.
* **`pool_beats` reste intact** et nourrit seul les têtes entraînées
  root/quality — leur changer leur pooling d'entrée sans réentraînement les
  dégraderait en silence (CLAUDE.md, règle #6).
* **La basse mobile n'est pas traitée, et c'est par là que ça casse.**
  16 accords sur 20 ont une basse qui bouge à l'intérieur de l'accord ; on
  écrit un seul symbole. Ready mesure 49 montre le cas limite : ce qu'on
  prenait pour une alternance de basse était deux accords. Découper l'accord
  là où la basse change franchement — et se servir de la basse comme
  détecteur de frontière quand le modèle d'accords n'en voit pas — est la
  suite directe de ce travail.

## Comment refaire la mesure

Les deux pages d'arbitrage (extraits audio, bandes de basse par temps,
réponses enregistrées) :

* 1er tour, candidats bruts — <https://claude.ai/artifact/PnLrDbrZBYoBWAEVTFf6pm>
* 2e tour, la ligne écrite — <https://claude.ai/artifact/Kc33QQ9zX9qdcaDkyKuAYR>
