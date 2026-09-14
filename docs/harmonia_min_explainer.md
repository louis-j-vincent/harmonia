# harmonia_min expliqué — pour le réimplémenter soi-même

**HISTORIQUE (sprint 20 du refactor, 2026-09-14).** Ce document décrit
`harmonia_min`, l'app qui a servi de référence pendant le refactor. Le nouveau
paquet `harmonia/` (voir `docs/STATE.md`) a un layout différent — sections en
sous-package, routes en blueprints, un seul point de lecture de
l'environnement (`settings.py`). Utile pour comprendre le RAISONNEMENT
derrière chaque étage (pourquoi songformer tourne en sous-processus, pourquoi
le fold n'a qu'une loi, etc.), pas pour trouver un chemin de fichier actuel.
`harmonia_min` disparaît à la fin du refactor (sprint 22).

2026-08-05, branche `feat/chord-lm` @ 3c58773. Compagnon du schéma
`docs/harmonia_min_schema.png` (régénérable :
`archive/scripts/render_harmonia_min_schema.py`, archivé au sprint 20 — il
dessine le pipeline `harmonia_min` spécifiquement, pas le nouveau). Objectif :
comprendre chaque étage assez pour le refaire de zéro, sans lire le code.

Vue en une phrase : **audio → beats (Beat This!) → posteriors d'accords
(musx) → re-décodage sur NOS beats → mesures → sections → folding →
tonalité/couleurs → un JSON de chart → servi par Flask sur :7772, avec le
lock qui re-décode en différentiel.**

## 0 · Audio

`POST /api/yt-search` cherche d'abord dans `docs/audio/*.m4a`, puis via
yt-dlp (vraie recherche YouTube). `POST /api/analyze {url}` lance un job en
thread ; le client sonde `GET /api/job/<id>` qui expose `stage` 0→5, tempo,
clé, accords bruts puis finaux.

## 1 · Beats — `beats.py` (99 lignes)

- **Quoi** : Beat This! (`File2Beats`, CPU, sans DBN), le seul tracker.
  librosa est banni : il verrouille une octave de tempo 2x (mesuré 65 % vs
  78 % sur POP909). Pas de repli : échec = erreur franche.
- **Entrée** : m4a (retranscodé wav par ffmpeg si le décodage direct rate —
  leçon payée : Beat This! sur m4a échouait silencieusement).
- **Sortie** : `{beats:[s], downbeats:[s], bpm}` ; cache
  `harmonia_min/state/beats/<stem>.json`.

## 2 · Accords bruts — `musx.py` (338 lignes)

- **Quoi** : le modèle music-x-lab ISMIR2019 (notre SOTA accords), ensemble
  5-fold vendored sous `harmonia/third_party/ISMIR2019-…/`, importé par un
  context-manager qui chdir dans le clone.
- **Sortie de `frame_posteriors()`** : `[triad(T,73), bass(T,13), s7(T,4),
  s9(T,4), s11(T,3), s13(T,3)]` sur une grille de 23,22 ms (43,07 fps).
  Colonne 0 du triad = « pas d'accord » ; ensuite index = racine + 12×type,
  type ∈ {maj,min,sus4,sus2,dim,aug}. Cache
  `data/cache/musx_probs/<stem>.npz` (3-9 Mo/chanson).
- **`redecode()`** : Viterbi qui n'autorise les changements d'accord QUE sur
  nos beats, avec des coûts gradués (downbeat < demi-mesure < autre beat ;
  granularité demi-mesure minimum depuis le 01/08). La latence du modèle
  (il entend les changements en retard) est compensée : on essaie
  0→280 ms par pas de 40 et on garde celle qui maximise la
  log-vraisemblance du chemin — aucun ground-truth requis.
- `label_confidence()` : LA définition partagée de « à quel point l'évidence
  soutient le label X sur [t0,t1) » — pipeline et folding doivent utiliser
  la même échelle (un désaccord a déjà coûté 2 points d'AUC).

## 3 · Mesures — dans `pipeline.py`

Chaque onset d'accord est snappé à l'**index** de beat le plus proche
(jamais par contenance temporelle — bug historique aux frontières de
mesure). **Vote de phase** : si ≥55 % des accords tombent sur le même beat
non-zéro de la mesure (et ≤15 % sur le beat 0), ce sont les accords qui ont
raison contre le downbeat du tracker — on décale la grille. Un accord tenu
sur plusieurs mesures est réécrit au beat 0 avec `carry:true` (pas de « % »).

**Coupure BRUT/FINAL (2026-08-07, non documentée ici avant).** `analyze_steps`
rend le chart deux fois : le **chart brut**, dès la fin de §3 — mesures,
accords, une seule section « A » couvrant tout le morceau, `pending:
["sections","key"]` — puis reprend EXACTEMENT là où il s'est arrêté pour
produire le **chart final** (§4 à §6 ci-dessous). Ce n'est pas deux pipelines :
le même `bars` est muté sur place. Raison mesurée : la détection de sections
pèse 96-98 % du temps total, donc Louis voit un chart jouable en <2 s pendant
que le reste tourne en fond (`HARMONIA_RAW_CHART=1` arrête tout ici et ne
rend jamais que le brut).

## 4 · Sections — `sections.py` (665 lignes) + `songformer.py` (318 lignes)

**MIS À JOUR 2026-09-13** — cette section datait du détecteur d'avant
2026-08-05 comme s'il était toujours le seul. Il reste dans le code
(`chroma` ci-dessous) mais n'est plus ce qui sert. `detect_sections()`
choisit entre QUATRE implémentations via `HARMONIA_SECTIONS` (défaut
`songformer`), avec repli en cascade — jamais silencieux, chaque saut
s'écrit en ERROR/WARNING dans les logs :

1. **`songformer`** (défaut depuis 2026-08-18). Un modèle pré-entraîné
   (SongFormer, ASLP-lab) qui n'écoute PAS les accords mais le SON — deux
   encodeurs audio (MuQ + MusicFM) — et RECONNAÎT le rôle de chaque passage
   (intro/couplet/refrain/pont/instrumental/outro) au lieu de chercher une
   répétition. C'est le premier détecteur du projet qui trouve ses
   frontières dans le temps réel plutôt que sur notre grille de mesures (on
   les tire ensuite sur la mesure la plus proche). Coûte cher en mémoire
   (SIGKILL mesuré sur un morceau de 422 s sur une machine 16 Go) donc
   tourne dans un sous-processus jetable ; si ce sous-processus meurt ou
   plante, repli LOGGÉ vers `voice`. Validé à l'oreille par Louis sur 19
   morceaux (« je suis d'accord avec lui partout, on le prend en prod »).
   **Ne règle PAS** le repli des répétitions, la longueur d'écriture d'une
   lettre, ni les queues divergentes — ça reste le travail de `folding.py`
   en aval (§5).
2. **`voice`** (défaut du 2026-08-08 au 2026-08-18) — la voix cherche où
   l'intro finit et le chant commence, un bloc de 8 mesures cherche ses
   propres répétitions, des blocs de 4 comblent les trous, les lettres qui
   nomment la même musique fusionnent. 0,769 contre 0,599 pour `harmonic`
   sur les 17 morceaux annotés de Louis (13/17 gains).
3. **`harmonic`** (défaut du 2026-08-05 au 2026-08-08) — dictionnaire de
   répétition sur les postérieures d'accords musx, projetées en 12 classes
   de hauteur.
4. **`chroma`** (l'algorithme d'origine, ce que décrivait cette section
   avant) — reste dans le fichier, atteignable par `HARMONIA_SECTIONS=chroma`.
   Substrat : chroma NNLS brut agrégé par **demi-mesure**, PAS les accords
   décodés (ils écrasent la texture qui distingue couplet/refrain). SSM =
   produit scalaire, sans flou (le flou déplace la position des pics ;
   l'enlever a gagné 27,6→30,1 % à lui seul). Frontières = UNION de deux
   sources : les bords des « runs de tuilage » (une cellule se répète à
   période 2/4/8 mesures) et les pics de nouveauté checkerboard — 41,8 %
   exact-bar sur 285 Billboard, contre 35,8 et 30,1 % séparément. Lettres :
   ratio de similarité cross-block entre segments, seuil 0,96 — pas
   invariant à la transposition (bug connu : une modulation du même
   refrain devient une nouvelle lettre).

## 5 · Folding — `folding.py` (1212 lignes)

Toujours en deux temps, mais le premier a deux réglages qui n'existaient
pas au 2026-08-05 (**mis à jour 2026-09-13**) :

**(a) Fold d'observation** (`fold_letter_groups`) : pour chaque groupe de
lettre, trouver la période qui se répète — soit à l'INTÉRIEUR d'une
occurrence (2/4/8 mesures, `loop=internal`), soit l'occurrence ENTIÈRE
quand elle revient N fois à longueur égale sans boucle interne
(`loop=occurrence`, le défaut depuis le 2026-08-08 — sans lui, 61 lettres
sur 162 refusaient le repli, le cas le plus fréquent du corpus). Puis :
empiler les occurrences position par position, jeter les outliers
(z-score 3 sur médiane+MAD), vérifier la cohérence du stack (cosinus
médian ≥0,85, veto par LETTRE entière par défaut — `HARMONIA_FOLD_GATE=bar`
le décide mesure par mesure), **moyenner les postérieures musx** du stack
(`HARMONIA_MERGE=mean`, le défaut depuis le 2026-08-19 — `=cqt` moyenne les
spectres AVANT le modèle à la place ; une seule loi, lue au même endroit
par `pipeline.py` et par `refold.py` depuis le 2026-08-20, voir
`loi_de_merge()`) et re-décoder le template (tuilé ×3 contre les effets de
bord du Viterbi) → les accords consensus sont réécrits sur toutes les
occurrences contribuantes ; une occurrence trop éloignée (z-score) reste
telle quelle (c'est comme ça que les mesures de transition survivent).
`HARMONIA_FOLD_TRANSPOSE=1` ramène un passage rejoué plus haut dans le ton
du premier avant l'empilement (Bora Bora : une montée d'un demi-ton).

**(b) Fold d'affichage** (`minimal_fold`) : une section rendue par
**(lettre, longueur)** — pas par lettre seule (« under-fold, never
over-fold » : deux refrains de 8 et 12 mesures restent deux blocs, tous
deux nommés B, chacun écrit à sa longueur réelle). `reps` = nombre de
passages ; le bloc écrit est le passage avec le PLUS d'accords réels (pas
forcément le premier — un fade-in initial vide n'écrase plus les passages
complets qui suivent) ; `barSpans` = pour chaque mesure affichée, les
fenêtres temporelles réelles de chaque passage (le contrat du playhead).
Un fold **refusé** (`reason` non vide dans le rapport) écrit son passage en
entier plutôt qu'une cellule raccourcie — le principe under-fold s'applique
aussi ici.

**Piège payé** : un fold refusé garde quand même son champ `period` — il
faut vérifier `reason` avant de s'en servir (un chart a été reconstruit
depuis un fold refusé le 02/08 ; corrigé, mais la convention reste).

**Illustration réelle** (Sunny Afternoon, Benny Sings, analysé le
2026-09-13) — 80 mesures brutes, une seule section « A » au chart brut →
7 blocs au chart final :

| lettre | mesures brutes couvertes | reps | repli accepté ? |
|---|---|---|---|
| intro | 0–7 | 1 | non (trop peu de membres) |
| A | 8–15, 56–63 | 2 | non (cohérence 0,76 < 0,85) |
| B | 16–23, 32–39 | 2 | **oui**, période 4 (cohérence 0,90–0,94) |
| C | 24–31, 40–47, 64–71 | 3 | **oui**, période 4 (cohérence 0,94–0,95) |
| D | 48–55 | 1 | non (occurrence unique) |
| C (variante courte) | 72–77 | 1 | — (6 mesures, occurrence isolée) |
| outro | 78–79 | 1 | non (longueurs inégales) |

B et C : le repli a vu le motif se répéter avec assez d'accord entre les
passages, donc leurs accords viennent de la MOYENNE des passages (moins de
bruit qu'un seul passage) et le chart les écrit une fois avec « ×2 »/« ×3 ».
A n'a pas passé le seuil de cohérence (ses deux occurrences diffèrent trop)
— le chart écrit alors le passage entier plutôt que de forcer une fausse
répétition (under-fold, never over-fold).

## 6 · Tonalité & couleurs — `harmonic_key.py` (v7c) + `key_profiles.py`

- `infer_key` : Krumhansl-Schmuckler bayésien (log-vraisemblance
  multinomiale sur le chroma brut sommé, pas un produit scalaire de
  distributions normalisées — c'était un bug).
- `analyze_harmony` : (1) chroma par accord (fenêtre Hann, aigus
  seulement) ; (2) piste de **tonique causale** — CUSUM sur la « masse
  interdite » (b2+#4 de la tonique courante), on tient jusqu'à preuve du
  contraire ; (3) audit du **mode** (part de tierce majeure sur les accords
  de tonique) ; (4) **couleurs** : HMM collant 4 états
  (naturel/harmonique/dorien/mélodique) sur l'évidence 6te/7e ; (5)
  inflections (emprunts momentanés) + challenges (accord non diatonique →
  suggestions). Le tout écrit `colour/inflect/flag/sug` sur chaque accord.

## 7 · ChartModel → servir

JSON dans `harmonia_min/state/charts/<file>.json` :
`{key, keyName, keySegments, bpb, nBars, barGrid, beatTimes, sections,
fold, bars[[{root,q,bass,c,t0,t1,carry,colour…}]], meta}`.
Serveur Flask :7772 (`server.py`, 519 lignes) : `GET /` = app_shell.html
(client autonome de 4 113 lignes, jamais modifié côté serveur),
`GET /api/chart-model/<f>` (avec overlay des annotations),
`/api/library` (+ `capabilities`), `/min/<f>` (vue minimale),
`/api/annotations` (écriture atomique), et un 404 honnête pour tout le
reste — l'UI dégrade proprement.

## Interaction : lock & propagation

Lock d'un accord → `POST /api/context_rescore/<f>` (alias `/api/reinfer`) →
re-décodage **différentiel** : une passe baseline sans lock, une passe avec
le lock clampé, mêmes paramètres — seuls les changements **causés** par le
lock sont appliqués. Évidence = musx_probs (cache uniquement, jamais de
run musx frais dans la requête ; repli NNLS-24 sinon) ; prior = trigramme
relatif-à-la-cible (voisins encodés en intervalle-depuis-le-candidat +
qualité), λ=2, δ=0,5, K=6. Limites actuelles : un lock ne déplace jamais
une frontière ; le re-décodage est en QUAL5 donc la 7e est perdue sur un
span déplacé ; un lock sur une mesure foldée s'applique à toutes ses
copies.

## Hors chemin par défaut : `chord_lm/`

Transformer 256d×4 couches (RoPE, cloze masqué), vocab 90 tokens (7
familles, grille demi-mesure), entraîné sur 2 401 grilles iReal.
`HARMONIA_CHORD_LM_SUGGEST=1` → suggestions seulement (gate deux côtés :
LM ≥0,80 ET pipeline ≤0,60), jamais de réécriture. Resté OFF : net 0 sur
GuitarSet. L'extension du vocabulaire (7e/6te/9e) est screenée viable mais
pas construite (`docs/handoff_2026-08-02_extended_chord_lm.md`).

## Pour couper le cordon avec l'ancien code

Imports Python résiduels : 2, tous côté entraînement (jamais live) —
`fine_to_q5` (ré-entraînement du prior) et `ireal_corpus` (corpus du LM).
Dépendances fichiers : poids musx sous `harmonia/third_party/`,
`nnls24_heads.npz` sous `harmonia/models/`, caches `data/cache/` partagés.
Déplacer ces trois chemins + copier deux loaders = harmonia_min autonome.
