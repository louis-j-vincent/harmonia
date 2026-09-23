# 33 — Analyser un morceau sans Mac

*2026-09-23*

Mandat de Louis, le 22 : une app Android qui analyse un morceau — pas de
serveur, pas de Mac, le calcul dans le téléphone. Ce billet raconte pourquoi
l'estimation de départ était fausse, ce qui a cassé en chemin, et où ça en
est un jour plus tard.

## L'estimation de départ était fausse par 3×

Trois audits séparés (chemin critique, import YouTube, faisabilité) ont
d'abord dit « 3 mois ». Le chiffre venait d'imaginer une réécriture. Or
l'appli a un plan : Android sait faire tourner du **CPython + numpy** via une
bibliothèque gratuite nommée **Chaquopy**, et cette bibliothèque peut aussi
faire tourner **Flask en local, dans le téléphone**. Le frontend JS existant
(10 000 lignes) et l'essentiel du pipeline Python partent donc *tels quels*,
affichés dans une simple fenêtre web embarquée. Il ne reste à porter que les
deux réseaux de neurones et un plugin audio en C++. D'où la vraie estimation :
**3 à 4 semaines** pour Android, 1 à 2 de plus pour iOS.

## Les deux fois où les audits se sont trompés

Avant de coder, deux affirmations des audits ont été vérifiées ligne à ligne
dans le code — et les deux étaient fausses, dans des sens opposés.

- **SongFormer** (un détecteur de sections, 2,7 Go de poids) était annoncé
  comme le pire problème de mémoire. Faux : Louis l'a désactivé le 19
  septembre (`settings.py`), il ne tourne plus jamais. Conséquence assumée :
  un morceau neuf, sur mobile, sort avec une seule section « A » — pas de
  découpage en couplet/refrain avant qu'un outil d'annotation existe côté
  téléphone.
- **NNLS-chroma** (un plugin C++ qui calcule la « couleur » harmonique, la
  basse comprise — c'est ce qui distingue un Eb d'un Eb/Bb) était annoncé
  optionnel. Faux : il est appelé deux fois, sans condition, à chaque
  analyse. C'est lui le vrai risque n°1, pas SongFormer.

Sans cette vérification, le projet aurait porté le mauvais morceau en
premier.

## Les cinq briques du pipeline, portées une par une

Chaque brique a son verdict de parité : l'écart mesuré contre ce que le Mac
calcule pour le même morceau.

| Brique | Ce qu'elle fait | Portage | Parité |
|---|---|---|---|
| Décodage audio | fichier → échantillons PCM | `MediaCodec` (décodeur Android natif) + un ré-échantillonneur maison | décalage temporel nul ; le filtrage lui-même reste à 0,02–0,15 d'écart avec la référence Mac, structurel (voir surprise 2) |
| NNLS-chroma | couleur harmonique + basse sonnante | réécrit en numpy pur | écart 1e-4, la basse trouvée est identique à 100 % |
| CQT de musx | transformée en fréquences pour le détecteur d'accords | réécrit en numpy pur | écart 2,6e-5, accords identiques à 99,99 % |
| Réseau musx (accords) | CNN + LSTM bidirectionnelle → accord par trame | réécrit en numpy pur (pas ExecuTorch, voir surprise 1) | écart 2e-6, accords identiques à 100 % |
| Réseau beat_this (battues) | trouve les temps et les mesures | exporté en **ExecuTorch** (format `.pte`, un graphe figé exécutable hors Python) | battues identiques à 0 ms sur 1802 sur 1803 (la 1803e diffère de 20 ms — une vraie quasi-égalité du réseau, pas une erreur de portage) |

Résultat sur l'émulateur, pipeline complet : 271 mesures identiques au Mac
sur 273 (99,27 %), sur 3 morceaux tests.

## Trois surprises

**1. musx refuse ExecuTorch, et c'est numpy qui gagne.** ExecuTorch exporte
un réseau torch vers un format qui tourne sans Python, avec un accélérateur
nommé **XNNPACK**. Pour le réseau à battues ça marche du premier coup. Pour
musx (qui contient une **LSTM bidirectionnelle** — une couche qui lit la
musique dans les deux sens à la fois), la conversion XNNPACK reste bloquée
plus de 27 minutes sans finir ni planter. Une variante sans accélérateur
finit, mais tourne 8 à 10× trop lentement. Le port numpy, lui, marche et va
plus vite (29-42 s pour 4 min de musique) — une fois un vrai bug corrigé
(un calcul de normalisation qui se trompait sur un tableau réordonné en
mémoire, invisible sur un petit extrait, énorme sur un morceau entier).

**2. Le Mac décode le même fichier de deux façons différentes**, et personne
ne le savait. Le pipeline Mac utilise `ffmpeg` pour trouver les battues et
`librosa`/CoreAudio pour les accords — deux décodeurs du même MP3, dont la
sortie diffère assez (écart crête 0,387) pour changer le compte de battues
d'un morceau (345 contre 348). Sur téléphone il n'y aura qu'un seul
décodeur : la divergence entre Mac et téléphone n'est donc pas un bug de
portage à corriger, c'est le Mac qui était incohérent avec lui-même.

**3. Les cookies YouTube copiés meurent en 10-20 minutes**, et pas par
rotation comme supposé d'abord. Google supprime purement et simplement les
13 cookies d'authentification (SID, HSID, APISID...) et n'en laisse qu'un
seul, inutile. La solution n'est donc pas de rafraîchir un fichier, c'est de
se connecter *depuis le téléphone* : un écran de connexion Google (une vraie
page web affichée dans l'app) a été construit et vérifié — il écrit les
cookies au bon endroit. Reste à le tester avec le compte réel de Louis.

## Ce qui n'est pas gratuit : les leviers de vitesse

Trois raccourcis pour aller plus vite ont été mesurés, mesure par mesure —
et aucun n'est gratuit. Le détail, à écouter plutôt qu'à lire un chiffre :
`docs/portage_mobile/plots/fastmode_3songs.html`.

| Levier | Gain visé | Ce qu'il casse réellement |
|---|---|---|
| checkpoint réduit du réseau à battues (`small0`) | plus léger | redessine la grille des mesures sur 2 morceaux sur 3 (pas un arrondi) |
| musx à 1 ou 3 modèles au lieu de 5 | plus rapide | réécrit 1 à 52 % des mesures selon le morceau |
| quantification int8 (réseau compressé) | 4× plus petit | insère ~15 % de fausses battues, proches des vraies |

## Où on en est, et ce qui reste

La porte dure n'est pas encore passée : le vrai test tourne sur téléphone,
pas sur émulateur — l'émulateur ne peut pas exécuter XNNPACK du tout (une
instruction lève une erreur processeur), donc aucun chiffre de vitesse pris
dessus ne compte. La règle d'arrêt : **4 minutes de musique analysées en
moins de 2 minutes de calcul, sinon on arrête et on revient à l'hybride**.
Ce test attend le téléphone de Louis, semaine du 28.

En attendant : l'APK signé est en cours (phase 4), la connexion YouTube
reste à valider avec le vrai compte, et iOS n'a pas commencé.

## Fichiers de référence

- `docs/portage_mobile/PLAN.md` — statut à jour, tenu par l'orchestrateur
- `docs/portage_mobile/JOURNAL.md` — chronologie complète, une ligne par fait
- `docs/portage_mobile/portage_mobile_2026-09-22_synthese.md` — la décision
  et ses raisons
- `docs/portage_mobile/p2c_integration.md`, `p2d_pipeline_device.md` — le
  pipeline complet, sur Mac puis sur l'émulateur
- `docs/portage_mobile/t1b_musx.md` — musx contre ExecuTorch
- `docs/portage_mobile/p3c_youtube_session.md` — la mort des cookies et
  l'écran de connexion
- `docs/portage_mobile/t5prep_leviers_vitesse.md` +
  `docs/portage_mobile/plots/fastmode_3songs.html` — les leviers de vitesse,
  morceau par morceau
