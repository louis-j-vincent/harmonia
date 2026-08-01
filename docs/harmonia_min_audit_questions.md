# Audit harmonia_min vs règles — questions pour Louis (2026-08-01)

Audit adversarial mené par un agent indépendant (lecture seule, vérifié sur
les 5 charts + le code). Verdict global : règles 0/1/3/4 conformes ; règle 2
(snap demi-barre) NON tenue corpus-wide ; règles 5/6/7 conformes avec
divergences listées. Détails et pointeurs code dans le rapport ci-dessous.

## BUG RÉEL trouvé (hors questions — décision de fix à prendre)

**Le sélecteur de latence musx est structurellement biaisé vers 0 ms** :
dans `path_loglik`, les frames laissées hors des spans décalés gardent le
tag 0 = `C:min/b7` (première ligne du vocabulaire), pas du silence — donc
tout décalage L>0 paie une pénalité artificielle sur ses L premières frames.
15/16 chansons choisissent 0 ms là où l'étude d'origine mesurait +46…+289 ms.
⚠️ Le même bug existe dans l'original `harmonia/models/musx_redecode.py`
(toutes les branches). Le fix (comparer les latences sur le même ensemble de
frames) peut DÉPLACER des frontières d'accords validées → à faire en A/B
mesuré, pas en silence. **Go / no-go Louis.**

## Les 11 questions (classées par impact)

**Q1 — Snap demi-barre : contre-exemple.** Let It Be livre 3 accords dans
UNE barre (G F C, temps 0/1/2). Si on interdit les quarts structurellement,
lequel saute ? Ou la vraie règle est « quarts permis mais chers » (l'actuel) ?

**Q2 — Portée du test de variance.** Il ne tourne que sur le repli de boucle
(phase 1) ; le squash ×N des sections merge sur égalité exacte des labels,
sans test de variance. La règle 6 doit-elle gater aussi le squash de sections ?

**Q3 — Unités de la métrique variance.** var/moyenne n'est pas invariante au
volume (NNLS 2× plus fort ⇒ métrique 2×). Passer à une forme sans dimension
(var/moyenne², ou écart-type/moyenne) avant de figer ton seuil ?

**Q4 — Statut du N.C.** Deux charts ont N.C. + accord sur LE MÊME slot
(She Will Be Loved barre 0 ; Let It Be dernière barre). On merge, on jette le
N.C., ou on interdit la collision en amont ?

**Q5 — Plafond de variantes.** She Will Be Loved (avant la règle raffinée)
pliait avec 58 variantes sur 107 barres — le consensus venait de la minorité.
Au-delà de quelle proportion de variantes un empilement est-il illégitime ?

**Q6 — Granularité du veto variance.** Demi-barre ou barre entière ?
(Le cas Dø7 de This Love se comporte différemment selon la réponse.)

**Q7 — Profondeur d'une fin variante.** 1 barre, 2 barres, ou une cellule
entière (P barres) ? Le ≤2 du pli d'affichage est un choix du code.

**Q8 — Multiples de 2 en fin de morceau.** Dernière section de She Will Be
Loved = 33 barres (total impair). La règle porte sur les coupes (l'actuel)
ou sur chaque longueur — et qui absorbe la barre impaire ?

**Q9 — Rigidité de la grille.** La grille est strictement périodique — les
downbeats Beat This! ne votent que la phase. Une barre de 2/4 insérée décale
tout. « La rigidité de la barre » = périodicité stricte, ou suivre les
downbeats réels barre par barre ?

**Q10 — La demi-barre en 3/4 ?** Tout le vocabulaire des règles est binaire
(demi/quart) et les coûts de transition supposent 4 temps. En 3/4 ou 6/8, le
snap devient quoi ?

**Q11 — Lettre qui échoue au failsafe.** Stand By Me : 6 sections A de
longueurs 20/6/4/14/14 qui ne se recoupent pas — l'échec ne vit que dans les
logs. On éclate en lettres distinctes, ou on le marque visiblement dans la
chart ?

## Constats conformité corrigés dans la foulée (2026-08-01)

- make_beat_arr recevait toujours beats_per_bar=4 → bpb réel transmis.
- Stage-4 (clé KS sur chord-tones) était du code mort écrasé par l'étage
  harmonique, avec double report de clé → supprimé.
- Ma note « de facto 100% demi-barre » dans l'audit des règles était fausse
  corpus-wide (Let It Be a des onsets au temps 1) → corrigée.
- Position sans membre gated : jamais réécrite (vérifié) ; la variante ne
  sert que de contexte au décodage — risque accepté, commenté.
