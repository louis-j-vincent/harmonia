# T5-prep — leviers de vitesse (avant l'arrivée du téléphone)

## Résultat en 3 lignes

Aucun levier n'est gratuit : le parallélisme battues‖musx est **déjà dans le
code de prod** (`pipeline.py`, `ThreadPoolExecutor`) et ne coûte rien au
chart, mais `small0` réécrit la GRILLE DE BATTUES sur 2 morceaux sur 3 (Stand
By Me 345→354 battues, Autumn Leaves 1126→1203 — pas un arrondi, un
redécoupage) tandis que This Love ne bouge pas du tout ; réduire musx à 1 ou
3 folds ne touche JAMAIS la grille mais réécrit 1 à 52 % des mesures selon le
morceau (moyenne d'ensemble perdue). La quantification int8 exporte et
tourne (÷3,9 sur la taille, 191 s de lowering) mais insère ~15 % de battues
fantômes proches des vraies (This Love, mesuré en plus proche voisin après
avoir écarté un premier diagnostic trompeur — voir §Levier 4) ; pas encore
un verdict "non" mais pas encore un "oui" non plus. Page à écouter :
`docs/portage_mobile/plots/fastmode_3songs.html` (miroir servable tout de
suite : `docs/plots/fastmode_3songs_mobile_t5prep.html`).

## Ce qui a été vérifié (commandes, chiffres, chemins)

**Protocole.** `harmonia.pipeline.analyze()` rejoué tel quel (`.venv`, torch
réel, device MPS — c'est ce que sert :7772 aujourd'hui), sur les 3 morceaux
de référence (`maroon_5_this_love`, `ben_e_king_stand_by_me_audio`,
`autumn_leaves`, les mêmes fichiers que T1b/T1c). AUCUN fichier de
`harmonia/` modifié : le seul changement par levier est un monkeypatch en
mémoire dans mon propre script
(`mobile/tools/fastmode/run_lever.py`) — `beats_mod._f2b` pointé sur
`File2Beats(checkpoint_path="small0")`, ou `musx_mod.MODEL_NAMES` tronqué à
1/3 noms — exactement la même technique que T1c pour son harnais de parité.
Caches sandboxés (`HARMONIA_STATE_DIR` + `harmonia.cache.KINDS` redirigés
vers `/private/tmp/.../fastmode_state/<levier>/`, un dossier PAR levier) :
**aucune écriture dans `state/` ni dans `data/cache/` partagé** — la clé de
cache d'harmonia est `stem+taille`, pas le modèle ; écrire dans le cache
partagé avec un modèle différent l'aurait silencieusement corrompu pour le
serveur :7772 vivant (le piège exact que T1c documente déjà et que ce script
évite par construction).

Chaque mesure comparée AVANT/APRÈS via `barGrid`/`barSpans` (comme
`tools/avant_apres.py::song_bars`, repris sans importer ce module — il
republie dans la bibliothèque, pas voulu pour une comparaison hors-ligne).

### Tableau des temps (Mac 1 processus, torch/MPS — PAS le téléphone)

| combinaison | This Love (205 s) | Stand By Me (177 s) | Autumn Leaves (422 s) | total 3 morceaux | ×3 (extrapolation tél.) |
|---|---:|---:|---:|---:|---:|
| **baseline** (final0, 5 folds — :7772 aujourd'hui) | 15,5 s | 12,0 s | 34,0 s | 61,5 s | 184,6 s |
| small0 (beat_this) | 13,9 s | 10,3 s | 25,3 s | 49,5 s | 148,5 s |
| musx 1 fold | 14,4 s | 11,6 s | 23,6 s | 49,6 s | 148,8 s |
| musx 3 folds | 10,8 s | 7,9 s | 20,3 s | 39,0 s | 117,1 s |

`×3` = facteur générique du PLAN pour un téléphone 2-4× plus lent par cœur —
**PAS une mesure**, à remplacer dès que T5 tourne sur le vrai appareil
(règle CLAUDE.md 6 : un changement de composant se mesure sur toute la
chaîne, pas sur un seul chiffre). Le budget dur est 240 s/4 min de musique →
120 s de calcul ; aucune combinaison ci-dessus, extrapolée ×3, ne tient sur
Autumn Leaves seul (68-76 s) une fois sommée aux autres étages (NNLS, CQT,
bars) — mais rappel : ces temps sont **torch/MPS sur Mac**, pas le chemin
`.pte`/numpy qui tournera réellement sur téléphone (T1b/T1c mesurent CE
chemin séparément, voir "Ce qui reste ouvert").

### Levier 1 — parallélisme (ne change AUCUN chart)

Le pipeline de prod lance DÉJÀ battues et musx en parallèle
(`pipeline.py::analyze_steps`, `ThreadPoolExecutor(max_workers=1)`,
commentaire du code : « Sur un morceau neuf de 9 min : battues 11,8 s, CQT+5
réseaux 8,9 s. Enchaînées, c'est la somme ; lancées ensemble, c'est le max »)
— confirmé par les temps mesurés ci-dessus : sur This Love, `beats`=10,6 s et
`musx_features+musx_posteriors`=6,3 s tournent en même temps, total pipeline
15,5 s ≈ max(10,6; 6,3) + nnls(4,6) + petite marge, PAS la somme (16,9 s).
Ce levier est donc déjà acquis côté Mac.

Reste à vérifier côté MOBILE (numpy + `.pte`, le chemin qui tournera sur
Android) : est-ce que le GIL Python est bien relâché pendant XNNPACK ET
pendant OpenBLAS sur CE binding précis (pas en théorie) ? Mesuré
(`mobile/tools/fastmode/speed_threads.py`, venv `.venv-mobile`, This Love) :

| | 1 thread | 2 threads | 4 threads |
|---|---:|---:|---:|
| musx numpy (OpenBLAS, ensemble 5 folds) | 24,1 s | 62,0 s | 92,7 s |
| beat_this `.pte` (XNNPACK) | 51,4 s | 31,5 s | 22,2 s |

**Mesure prise SOUS CONTENTION** (le job de quantification int8 du levier 4
tournait en même temps sur la même machine, gros consommateur CPU) — une
tentative de re-mesure isolée a été lancée mais n'a pas fini dans la boîte
de temps de cette tâche. Ce qui reste EXPLOITABLE malgré la contention :

- **`.pte` XNNPACK profite clairement de plus de threads** (51→32→22 s,
  monotone, direction cohérente avec l'attente — les convolutions/attention
  du réseau se parallélisent bien).
- **musx numpy DEVIENT PLUS LENT avec plus de threads** (24→62→93 s) — ceci
  est presque certainement un ARTEFACT de la contention (le job int8
  concurrent). OpenBLAS demandant plus de threads pendant que l'autre job
  sature déjà le CPU cause du thrashing, pas une propriété du code numpy
  lui-même. **Ne PAS conclure "musx ne profite pas du multi-thread" de ce
  chiffre — à re-mesurer isolément avant toute décision.**
- Séquentiel (1 thread chacun) : musx 42,8 s + pte 28,9 s = 71,7 s (ces deux
  chiffres sont DIFFÉRENTS de la ligne "1 thread" ci-dessus — même sous
  contention, deux passes du même calcul à des moments différents de la
  fenêtre de contention donnent des temps différents, autre signe que ces
  valeurs absolues ne sont pas fiables). En parallèle (2 threads Python, 1
  thread lib chacun) : mur d'horloge 65,4 s → accélération ×1,10 seulement —
  BEAUCOUP moins que le ×2 espéré d'un vrai relâchement du GIL des deux
  côtés. Signal faible mais probablement réel (contrairement au point
  précédent, un ralentissement en parallèle n'est PAS ce que la contention
  produirait — la contention aurait plutôt caché un vrai gain) : sur CE
  binding (executorch runtime python + `threadpoolctl`), le parallélisme
  Python entre les deux jobs n'apporte quasiment rien, à vérifier isolément.

### Levier 2 — beat_this `small0` au lieu de `final0`

Rejoué le pipeline COMPLET (pas juste beat_this isolé) avec `small0` à la
place de `final0` :

| morceau | mesures avant→après | battues avant→après | mesures changées | temps gagné |
|---|---:|---:|---:|---:|
| This Love | 80→80 | 322→322 | **0** | 1,6 s |
| Stand By Me | 86→89 | 345→354 | **66** | 1,7 s |
| Autumn Leaves | 281→300 | 1126→1203 | **235** | 8,7 s |

This Love : rien ne change — grille et accords identiques. Stand By Me et
Autumn Leaves : `small0` **déplace la grille de battues elle-même** (pas
juste 1-2 battues de flottement comme T1c le mesurait sur `final0` vs torch
— là c'est `small0` vs `final0`, deux réseaux différents, sur un morceau
entier) ; en aval, le redécoupage en mesures + le re-décodage musx sur CETTE
grille réécrivent des dizaines de mesures. Exemples (Stand By Me, mesures 11
à 18, le passage se redécoupe entièrement autour d'un silence détecté
différemment) :

    mesure 11 : D/E  ->  ·
    mesure 12 : E    ->  ·
    mesure 13 : N    ->  D/E
    mesure 14 : ·    ->  E
    mesure 15 : ·    ->  N

Ce que ça dit : `small0` n'est pas un « `final0` en moins précis » — c'est un
réseau qui peut désaccorder la mesure PAR RAPPORT au morceau lui-même, pas
seulement par rapport à `final0`. Gain de temps modeste (10-25 %) pour un
risque qui dépend du morceau (2 sur 3 ici, échantillon trop petit pour
généraliser — règle CLAUDE.md 5, hypothèse à confirmer sur plus de morceaux
si Louis veut creuser ce levier).

### Levier 3 — musx à 1 ou 3 folds au lieu de 5

La grille de battues ne bouge JAMAIS (musx n'y touche pas) ; seuls les
accords changent, proportionnellement à combien de folds manquent :

| morceau | mesures totales | changées à 1 fold | changées à 3 folds | temps gagné (1 fold / 3 folds) |
|---|---:|---:|---:|---:|
| This Love | 80 | 19 (24 %) | 8 (10 %) | 1,0 s / 4,7 s |
| Stand By Me | 86 | 12 (14 %) | 1 (1 %) | 0,4 s / 4,1 s |
| Autumn Leaves | 281 | 147 (52 %) | 48 (17 %) | 10,5 s / 13,7 s |

Exemples (This Love, 1 fold) — le décodeur d'ensemble perd la 7e et
distingue moins bien maj/dim quand un seul avis compte :

    mesure 2  : C-        ->  C-7
    mesure 4  : Do F-/Ab  ->  Dh7 Ab-
    mesure 5  : G/B       ->  Bh7 G/B

À 3 folds les mêmes mesures se stabilisent (This Love : mesure 4 devient
`Do Abo`, plus proche de la version 5-folds `Do F-/Ab` que la version
1-fold). Autumn Leaves (jazz, harmonie dense) est systématiquement le plus
sensible : 52 % des mesures bougent à 1 fold contre 10-24 % pour les deux
morceaux pop — cohérent avec l'ensemble moyennant plus d'ambiguïté sur un
langage harmonique plus riche. **Le temps gagné par fold n'est PAS linéaire
sur Mac (MPS)** : passer de 5 à 1 fold ne réduit `musx_posteriors` que de
~40-50 % au lieu des 80 % attendus (overhead de dispatch MPS fixe qui domine
à ce volume) — un signal que ce levier vaudra probablement MIEUX sur
téléphone (CPU pur, pas de MPS) que ce que ces chiffres Mac suggèrent, à
vérifier à T5.

### Levier 4 — quantification int8 (XNNPACK quantizer, PT2E)

**Export réussi**, largement sous la boîte de temps de 40 min :
`torch.export` (8,5 s) → `prepare_pt2e` (3,4 s, `XNNPACKQuantizer` symétrique
per-channel) → calibration sur 2 morceaux (1 bloc de 1500 trames chacun) →
`convert_pt2e` (26,2 s) → ré-export + lowering XNNPACK (145 s) — **191 s au
total** (3,2 min). `.pte` = **20,9 Mo, contre 82,4 Mo en float32** (÷3,9).

**Premier diff (indice par indice) : FAUX ALARME caractérisée, pas un verdict.**
Comparer `beats_float[i]` à `beats_int8[i]` terme à terme donne des écarts
énormes (jusqu'à 69 s) dès que les DEUX modèles n'ont pas le même nombre de
battues — un seul décalage de compte au début du morceau désaligne tout le
reste de la comparaison, ce qui exagère massivement l'écart réel. Recalculé
en PLUS PROCHE VOISIN (comme un vrai score de battues, `mobile/tools/fastmode/
diag_int8.py`) sur This Love :

| | battues | temps forts |
|---|---:|---:|
| float32 | 322 | 81 |
| int8 | 371 (**+49, +15 %**) | 90 (**+9, +11 %**) |
| écart max\|Δt\| (plus proche voisin) | 640 ms | — |
| écart médian | **20 ms** | — |
| écart moyen | 33 ms | — |
| battues à >50 ms | 25/322 (8 %) | — |

**Mécanisme** (pas juste « ça casse ») : les battues existantes bougent PEU
(médiane 20 ms — cohérent avec un bruit de quantification ordinaire, pas une
catastrophe) ; le vrai défaut est que le modèle int8 **insère des battues
fantômes tout près d'une vraie** — ex. This Love, `float: [0.44, 1.08, …]`
vs `int8: [0.42, 0.50, 1.08, …]` : une battue à 0,42 s (proche de la vraie,
0,44 s) PUIS une seconde à 0,50 s, 80 ms plus tard — trop loin pour que le
post-traitement les fusionne (`_deduplicate_peaks`, fenêtre ≤1 frame=20 ms),
donc le bruit de quantification a créé un DEUXIÈME maximum local concurrent
là où il n'y en avait qu'un. Calibration = 2 fenêtres de 1500 trames (~30 s
d'audio au total, 2 morceaux) — probablement trop peu pour bien estimer la
plage dynamique des couches de sortie (tête beat/downbeat), qui est
justement celle dont dépend le test strict `logit == maxpool(logit)`.

**Fix impliqué par ce mécanisme, PAS implémenté (hors boîte de temps du
levier, 40 min déjà consommées par export+diagnostic)** : (1) calibrer sur
beaucoup plus de fenêtres/morceaux (le script le permet déjà, juste une
boucle à étendre) ; (2) garder la dernière couche (logits beat/downbeat) en
float32 — `XNNPACKQuantizer` sait exclure un module par nom — puisque c'est
CETTE couche que le post-traitement teste au bit près. Prochain pas concret
si Louis retient ce levier : réessayer avec (1)+(2), mesurer à nouveau en
plus-proche-voisin sur les 3 morceaux.

**Non re-vérifié en plus-proche-voisin** : Stand By Me et Autumn Leaves —
seuls leurs diffs indice-par-indice (trompeurs, voir ci-dessus) ont tourné ;
This Love ci-dessus est le seul cas caractérisé correctement (CLAUDE.md
règle 5 : un seul morceau, hypothèse à confirmer, pas une conclusion
générale).

**Vitesse** : This Love float 19,3 s / int8 15,8 s ; Stand By Me float 17,5 s
/ int8 31,9 s (PLUS LENT — contention avec un autre job tournait en même
temps, pas fiable) ; Autumn Leaves float 56,8 s / int8 47,1 s. Pas de gain de
vitesse net et cohérent démontré ici — à re-mesurer isolément si ce levier
est repris.

## Ce qui reste ouvert

- **Le tableau du levier 1 est mesuré SOUS CONTENTION** (le job int8 tournait
  en même temps) — une re-mesure isolée a été lancée mais tuée avant la fin
  (disque descendu à 3,8 Go pendant la session, proche du seuil d'arrêt de
  3 Go du PLAN ; rien perdu, juste pas eu le temps). Premier pas concret si
  ce levier est repris : relancer SEUL `mobile/tools/fastmode/
  speed_threads.py` (rien d'autre de lourd en même temps).
- **Ces temps sont Mac/torch/MPS, pas le chemin `.pte`/numpy qui tournera
  sur téléphone** — T1b/T1c ont déjà mesuré CE chemin séparément (musx numpy
  29-42 s/4 min, beat_this `.pte` ~35 s/4 min projeté, 1 thread Mac) ; ce
  document ajoute le CÔTÉ CHART (quels accords bougent) que T1b/T1c n'avaient
  pas regardé, plus le vrai coût pipeline complet (nnls, bars, fold, clé).
  Personne n'a encore sommé « pipeline complet, chemin mobile, 3 morceaux » —
  candidat naturel pour T5 dès que le téléphone arrive.
- **`small0` : échantillon de 3 morceaux, 2 affectés** — hypothèse à
  confirmer sur plus de morceaux avant d'en tirer une règle générale (CLAUDE.md
  règle 5, single-song/few-song findings).
- **Le fold1/fold3 est un raccourci arithmétique** (`MODEL_NAMES` tronqué) —
  identique à ce que ferait vraiment un déploiement mobile n'embarquant que
  1 ou 3 des 5 `.npz` de poids (économie de RAM/disque proportionnelle, pas
  mesurée ici — seulement le temps de calcul et le chart).
- **Pas de mesure de RAM/pic mémoire** pour aucun des 4 leviers — hors
  périmètre de cette tâche (budget 3h), pertinent si T5 échoue aussi sur la
  mémoire et pas seulement le temps.
- **La quantification int8, si elle a abouti, n'a été comparée qu'aux
  BATTUES** (pas au chart complet re-décodé) faute de temps dans la boîte de
  40 min — à compléter si Louis retient ce levier.

## Fichiers produits

- `mobile/tools/fastmode/run_lever.py` — rejoue `harmonia.pipeline.analyze()`
  avec un levier (baseline/small0/fold1/fold3), sandboxé, sur les 3 morceaux.
- `mobile/tools/fastmode/compare_levers.py` — diff mesure par mesure
  baseline vs chaque levier (repris de `tools/avant_apres.py::song_bars`).
- `mobile/tools/fastmode/speed_threads.py` — levier 1, venv mobile,
  `.pte` (XNNPACK, `_unsafe_reset_threadpool`) + musx numpy (OpenBLAS,
  `threadpoolctl`) à 1/2/4 threads + test parallèle réel (2 threads Python).
- `mobile/tools/fastmode/export_beat_this_int8.py` — levier 4, PT2E +
  `XNNPACKQuantizer`, boîte de temps 40 min intégrée au script (191 s
  réellement utilisés).
- `mobile/tools/fastmode/parity_int8.py` — diff indice-par-indice float32 vs
  int8, 3 morceaux (résultat trompeur, voir §Levier 4).
- `mobile/tools/fastmode/diag_int8.py` — diff CORRIGÉ (plus proche voisin),
  This Love, qui a renversé le premier verdict.
- `mobile/tools/fastmode/build_page.py` — assemble la page HTML depuis les
  JSON ci-dessus.
- `docs/portage_mobile/plots/fastmode_3songs.html` (chemin demandé) et
  `docs/plots/fastmode_3songs_mobile_t5prep.html` (miroir, réellement
  servi par la route `/plots/` de `harmonia/server/app.py` — voir note
  ci-dessous) — le livrable à écouter.
- Ce fichier : `docs/portage_mobile/t5prep_leviers_vitesse.md`.

**Note sur le chemin de la page.** `harmonia/server/app.py:156` sert
`/plots/<name>` UNIQUEMENT depuis `docs/plots/`, jamais depuis
`docs/portage_mobile/plots/` (lu dans le code, pas supposé) — et de toute
façon, cette page vit dans LE WORKTREE, pas dans l'arbre que sert le serveur
:7772 vivant (processus séparé, checkout séparé). J'ai donc écrit le fichier
à l'endroit demandé (`docs/portage_mobile/plots/fastmode_3songs.html`,
liens audio en `/audio/<stem>.m4a` absolu, pas `../audio/` relatif — ce
chemin-là suppose `docs/plots/`) ET un second exemplaire sous `docs/plots/`
pour que la page soit réellement ouvrable dès que ce contenu est mergé —
à l'orchestrateur de trancher lequel garder au moment du merge.
