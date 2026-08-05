# harmonia_min expliqué — pour le réimplémenter soi-même

2026-08-05, branche `feat/chord-lm` @ 3c58773. Compagnon du schéma
`docs/harmonia_min_schema.png` (régénérable :
`scripts/render_harmonia_min_schema.py`). Objectif : comprendre chaque étage
assez pour le refaire de zéro, sans lire le code.

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

## 4 · Sections — `sections.py` (490 lignes)

- **Substrat** : chroma NNLS brut (VAMP, cache `data/cache/nnls_infer/`),
  agrégé par **demi-mesure**, PAS les accords décodés (ils écrasent la
  texture qui distingue couplet/refrain). SSM = produit scalaire, **sans
  flou** (le flou déplace la position des pics ; l'enlever a gagné
  27,6→30,1 % à lui seul).
- **Frontières = UNION de deux sources** (le choix exclusif est un piège
  mesuré) : (a) les bords des « runs de tuilage » — plages où une cellule se
  répète à période 2/4/8 mesures (seuil 0,80) ; (b) les pics de nouveauté
  checkerboard. Union : 41,8 % exact-bar sur 285 Billboard (runs seuls
  35,8, pics seuls 30,1). Coupes snappées aux mesures paires, jamais au
  milieu d'une cellule récurrente.
- **Lettres** (A/B/C…) : ratio de similarité cross-block entre segments,
  seuil 0,96. **Bug connu** : pas invariant à la transposition — une
  modulation du même refrain devient une nouvelle lettre (Sunny).

## 5 · Folding — `folding.py` (555 lignes)

Deux temps. (a) **Fold d'observation** : pour chaque lettre, trouver la
période interne (2/4/8), empiler les occurrences position par position,
jeter les outliers (z-score 3 sur médiane+MAD), vérifier la cohérence du
stack (cosinus médian ≥0,85), **moyenner les posteriors musx** du stack et
re-décoder le template (tuilé ×3 contre les effets de bord du Viterbi) →
les accords consensus sont réécrits sur toutes les occurrences
contribuantes ; une occurrence trop éloignée reste telle quelle (c'est comme
ça que les mesures de transition survivent). (b) **Fold d'affichage**
(`minimal_fold`) : une section rendue par lettre, `reps` = nombre de
passages, `barSpans` = pour chaque mesure affichée, les fenêtres temporelles
réelles de chaque passage (le contrat du playhead).

**Piège payé** : un fold **refusé** garde quand même son champ `period` —
il faut vérifier `reason` avant de s'en servir (un chart a été reconstruit
depuis un fold refusé le 02/08 ; corrigé, mais la convention reste).

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
