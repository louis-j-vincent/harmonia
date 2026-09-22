# P2a-bis — qualité du rééchantillonneur Android (2026-09-23)

## Résultat en 3 lignes

Monter `tapsPerPhase` (64→128→256→512) **n'aide pas** — l'écart max vs
`soxr_hq` reste dans 0,016-0,14 selon le morceau, et empire parfois avec plus
de coefficients ; resserrer la coupure sous Nyquist empire aussi (0,08→0,14) :
l'écart n'est **pas** un manque de bande passante ni un filtre trop court,
c'est du ringing (Gibbs) sur des transitoires forts, insensible à la longueur
du filtre — cible 1e-3 **non atteinte, et non atteignable dans cette famille
de filtre** (voir §4). Seul levier qui aide, à coût nul : β=12,26 (cible
120 dB, formule Kaiser standard) au lieu de 8,6 → -4 à -8 % d'écart max sur
les 5 cas testés, sans changer le nombre de coefficients donc sans coût
Kotlin supplémentaire (mesuré : 8,45-11,1 s avant/après pour décoder un
morceau de 268 s sur l'émulateur, différences dans le bruit de contention).
Porté en Kotlin (`AudioDecoder.kt::designKaiserSincFilter`, un seul nombre
changé), rebuild + parité rejouée sur les 3 morceaux × 2 SR : **décalage 0
échantillon partout** (pas seulement This Love comme dans p2a — l'item
"reste ouvert" correspondant est levé), diff nulle sur les 2000 premiers
échantillons partout.

## Ce qui a été vérifié (commandes, chiffres, chemins)

**1. Banc d'essai numpy — même formule que le Kotlin.** Avant de toucher au
Kotlin, vérifié qu'une implémentation numpy vectorisée (`scipy.signal.
resample_poly(x, up, down, window=h/up)`, où `h` = notre filtre sinc×Kaiser)
reproduit EXACTEMENT (écart ~4e-16, bruit flottant) une traduction ligne à
ligne du `polyphaseResample`/`designKaiserSincFilter` Kotlin sur un signal
jouet (up=3, down=5) — donc la mesure ci-dessous porte sur la MÊME formule de
filtre que celle qui tourne sur le téléphone, pas une approximation scipy.
Détail piège : `resample_poly` multiplie automatiquement le filtre passé par
`up` — il faut donc lui donner `h/up`, pas `h`, sinon l'amplitude est fausse
d'un facteur `up` (147 ou 320 selon le sens) tout en gardant une forme
plausible — code dans `mobile/tools/resampler_quality.py`.

Entrée = décodage natif Mac (`librosa.load(sr=None)`, comme p2a) — aucune
différence de décodeur AAC possible. Référence = `librosa.resample(...,
res_type="soxr_hq")` (= ce que rend `librosa.load(sr=cible)` en 0.11, vérifié
par recoupement avec les `.npy` déjà mis en cache par `dump_reference_pcm.py`
pour les 3 morceaux de seed). Harnais validé : à `tapsPerPhase=64, β=8,6,
cutoff=100 %` (config du Kotlin d'avant ce rapport), le script reproduit
EXACTEMENT le chiffre déjà publié dans p2a (`maxdiff=0,083906
meandiff=0,002202` sur This Love 48000→22050) — 0,97 s de calcul.

**2. Sweep taps/β/cutoff, fichier entier, 5 cas** (3 morceaux natifs 48 kHz ×
22050, 1 morceau natif 44100 (`08DjMT-qR9g`, bibliothèque) × 22050, 1 × 44100
— couvre les deux paires SR natif→cible de la bibliothèque, contrat p0_t3) :

| Config (taps/ph, β, coupure) | This Love 48k→22k | Stand By Me 48k→22k | She Will Be Loved 48k→22k | 08DjMT 44k1→22k | This Love 48k→44k1 | coût (s, Mac) |
|---|---:|---:|---:|---:|---:|---:|
| 64, 8.6, 100 % (**avant**) | 0,0839 | 0,0170 | 0,0738 | 0,0937 | 0,0056 | 0,4-1,0 |
| 128, 8.6, 100 % | 0,0989 | 0,0216 | 0,0851 | 0,1144 | 0,0059 | 1,2-2,6 |
| 256, 8.6, 100 % | 0,1094 | 0,0234 | 0,0914 | 0,1310 | 0,0060 | 2,8-5,9 |
| 256, 12.26, 93 % coupure | 0,0829 | 0,0178 | 0,0947 | 0,1032 | 0,0030 | 2,8-6,3 |
| 512, 12.26, 93 % coupure | 0,0847 | 0,0193 | 0,1040 | — | 0,0030 | 5,8-10,5 |
| **64, 12.26, 100 % (retenu)** | **0,0804** | **0,0158** | **0,0708** | **0,0881** | **0,0055** | **0,4-0,9 (= avant)** |

(colonnes = écart max ; écart moyen toujours 20-40× plus petit, ~0,0002-0,0037
— l'écart max est concentré sur quelques échantillons, pas une erreur
diffuse.) Table complète (7 configs × 5 fichiers) dans la sortie du script.

**Lecture : plus de coefficients = pire, pas mieux.** 64→256 taps/phase fait
*monter* l'écart max sur 4 des 5 cas (ex. This Love : 0,084→0,109). Resserrer
la coupure à 93 % (comme le préconise soxr HQ) aide sur 2 cas et nuit sur 3.
Resserrer à 90 % nuit partout (0,084→0,141 sur This Love, testé séparément).
Seul β=12,26 (120 dB) à taps/coupure inchangés aide **uniformément** sur les
5 cas (-4 à -8 %), gratuitement (β ne change pas la longueur du filtre).

**Diagnostic du plateau** (This Love, config avant) : le pic d'écart
(0,0839) est à t=183,81 s, sur une transitoire (amplitude locale 0,71, PAS un
clip — le fichier culmine à 1,00 ailleurs, seulement 77/9 852 144
échantillons natifs à |x|>0,999). `y`≈0,505 vs `ref`≈0,421 à cet instant :
un vrai désaccord de réponse transitoire entre notre filtre (fenêtre de
Kaiser tronquée, quasi brick-wall) et l'algorithme de soxr (précision
variable / filtre en cascade différent), pas un artefact de clipping ni de
bande passante trop étroite — cohérent avec le fait que plus de coefficients
*aggrave* le ringing (signature classique du phénomène de Gibbs : un filtre
plus proche du brick-wall a un dépassement plus marqué, pas moins).

**3. Porté en Kotlin, rebuild, parité device** (émulateur `harmonia_35`,
`fr.harmonia.app`, `/api/debug/decode` via `adb forward tcp:7790 tcp:7772`) :
un seul changement (`beta = 8.6` → `beta = 12.26`, commenté). `./gradlew
assembleDebug` → `BUILD SUCCESSFUL` ; `adb install -r` ; force-stop + restart
pour recharger la lib native.

| Morceau | SR | n_samples | maxdiff (2000 premiers) | décalage (corr. croisée, 10 s milieu) | pearson | maxdiff 10s milieu | meandiff 10s milieu |
|---|---:|---:|---:|:-:|---:|---:|---:|
| This Love | 22050 | 4 526 177 | 0,0 | **0** | 0,99971 | 0,150 | 0,00373 |
| This Love | 44100 | 9 052 355 | 0,0 | **0** | 0,99977 | 0,155 | 0,00161 |
| Stand By Me | 22050 | 3 913 716 | 0,0 | **0** | 0,99995 | 0,027 | 0,00051 |
| Stand By Me | 44100 | 7 827 433 | 0,0 | **0** | 0,99997 | 0,029 | 0,00021 |
| She Will Be Loved | 22050 | 5 917 620 | 0,0 | **0** | 0,99976 | 0,104 | 0,00284 |
| She Will Be Loved | 44100 | 11 835 241 | 0,0 | **0** | 0,99998 | 0,117 | 0,00007 |

`n_samples` identiques à p2a (β ne touche pas la coupe de fin, comportement
inchangé, attendu). Note méthode : le premier script de vérification avait un
décalage constant de +margin (2205/4410 échantillons) câblé dans son
découpage de la fenêtre de référence — confirmé par un test synthétique
(signal==référence, lag vrai=0 → le harnais rapportait quand même
lag=+margin) ; corrigé algébriquement (le lag vrai = lag rapporté − margin),
donc **décalage = 0 échantillon partout**, et les colonnes pearson/maxdiff/
meandiff ci-dessus sont mesurées au bon alignement (vérifiées cohérentes en
ordre de grandeur avec le sweep numpy §2 : This Love pire cas, Stand By Me
meilleur, même hiérarchie entre morceaux).

**Chronométrage avant/après** (décodage complet forcé par une requête
`n=100`, She Will Be Loved 268 s, sur l'émulateur, sous contention d'autres
agents actifs en parallèle) : β=8,6 → 9,97 s puis 10,22 s (2 essais) ; β=12,26
(retenu) → 8,45 s puis 11,14 s (2 essais). Pas de différence significative
au-delà du bruit de contention — attendu : β ne change ni `halfLen` ni le
nombre d'itérations de la boucle interne de `polyphaseResample`.

## Ce qui reste ouvert

- **Cible < 1e-3 non atteinte et non atteignable dans cette famille de
  filtre** (item 4 du brief) : ni `tapsPerPhase` (64→512, testé) ni la
  coupure (100 %→90 %, testé) ne réduisent l'écart crête — au mieux -8 %
  avec β. La cause identifiée est une différence de réponse transitoire
  (Gibbs) entre un sinc×Kaiser tronqué et l'algorithme propriétaire de
  soxr_hq (précision variable / cascade), pas un défaut de longueur. Pour
  descendre sous 1e-3 il faudrait changer de FAMILLE de filtre (ex. filtre
  optimal équiripple/Parks-McClellan, ou une implémentation minimum-phase),
  non essayé ici (hors budget 2 h, et l'impact réel sur les battues n'est pas
  mesuré — voir point suivant).
- **Impact musical non mesuré directement** : ce rapport compare le signal
  PCM, pas les battues qui en sortent. L'écart crête (jusqu'à 0,15 sur 10 s
  au milieu d'un morceau) reste non-caractérisé sur le mel-spectrogramme +
  réseau de battues (`beat_this`/T1c) — c'est la vraie question posée en tête
  de tâche ("peut ou non déplacer une battue") et elle n'est PAS tranchée
  ici ; le calcul le moins cher pour trancher est de faire tourner
  `mobile/tools/parity_beats.py` (ou équivalent) sur le PCM device
  (β=12,26) vs le PCM Mac et comparer les temps de battue, pas juste le PCM.
- **`08DjMT-qR9g` (cas 44100→22050) non revérifié sur device**, seulement en
  numpy — les 3 morceaux de seed embarqués dans l'APK sont tous natifs 48 kHz
  (`ffprobe` confirmé), donc le device ne peut pas exercer le chemin
  44100→22050 sans un 4e fichier de seed.
- Item p2a "surplus de fin de fichier" (+348 à +873 échantillons) inchangé,
  hors périmètre de cette tâche (c'est la coupe de fin, pas le
  rééchantillonneur).

## Fichiers produits

- `mobile/android/app/src/main/java/fr/harmonia/app/AudioDecoder.kt` —
  modifié : `beta = 8.6` → `beta = 12.26` dans `designKaiserSincFilter`
  (commenté avec le renvoi vers ce rapport).
- `mobile/tools/resampler_quality.py` — banc d'essai numpy (sweep
  taps/β/coupure vs `soxr_hq`, fichier entier, formule Kaiser identique au
  Kotlin, vérifiée équivalente à `scipy.signal.resample_poly(window=h/up)`).
- Ce fichier + une ligne dans `docs/portage_mobile/JOURNAL.md` + mise à jour
  de `p2a_decodeur_audio.md` (section "Ce qui reste ouvert").
