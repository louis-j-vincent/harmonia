# Harmonia sur téléphone, sans Mac — synthèse de décision (2026-09-22)

Trois audits parallèles (chemin critique, import YouTube, faisabilité
Android/iOS), consolidés, corrigés, et le plan qui en découle.

Détail du chemin critique ligne par ligne : `docs/audit_portage_mobile_2026-09-22.md`.

---

## Le verdict en cinq lignes

1. **Android, 3 à 4 semaines.** Pas 3 mois. Chaquopy fait tourner CPython +
   numpy sur Android, et peut faire tourner **Flask en local dans l'app** —
   donc le frontend JS (10 000 lignes) et l'essentiel du Python partent tels
   quels. Il n'y a pas de réécriture.
2. **iOS, +1 à 2 semaines après Android**, pas un projet séparé : le gros du
   travail (export des réseaux via ExecuTorch) se réutilise à ~80 %.
3. **L'import YouTube ne survit que sur Android, en sideload.** Sur iOS il est
   mort — précédent Musi, retirée de l'App Store en 2024.
4. **Le risque n°1 est `nnls-chroma`**, un plugin C++ sur le chemin critique
   de chaque analyse, que personne n'a jamais compilé pour mobile.
5. **La vraie inconnue est la vitesse.** Aucun benchmark public de `beat_this`
   sur mobile. C'est ça que la phase 0 doit mesurer.

---

## 1. Ce qu'il faut vraiment porter

Sur les 17 642 lignes du paquet `harmonia/` :

| Bucket | Lignes | % |
|---|---:|---:|
| **Moteur** — ce qu'`analyze_steps` exécute toujours | 7 050 | 40,0 % |
| **+ service minimal** (jobs, cache, servir le chart, YouTube, titre) | 1 715 | 9,7 % |
| **= minimum vital** | **8 765** | **49,7 %** |
| Accessoire (vivant, hors chemin critique d'un premier chart) | 6 877 | 39,0 % |
| Mort pour mobile (recherche, diagnostic, orphelin) | 2 000 | 11,3 % |

À quoi s'ajoutent **6 045 lignes** dans `third_party/musx_ismir2019/`, non
comptées ci-dessus et pourtant exécutées.

**Seules 2 routes serveur sur 7 sont indispensables** pour analyser et écouter
un chart : `library` et `analyze` (479 lignes). Les 5 autres (`annotate`,
`sections`, `jam`, `irealb`, `verdicts`) sont de vraies fonctionnalités, mais
elles peuvent rester sur le Mac pour une v1.

Répartition du minimum vital par **coût** de portage : 66,4 % de logique pure
(port mécanique), 20,2 % de modèle torch (à convertir), 8,0 % de DSP, 5,4 % de
binaire natif. Le piège : les 66 % faciles ne calculent rien sans les 34 %
difficiles en amont.

---

## 2. Deux corrections aux audits (vérifiées dans le code)

Les deux agents se sont trompés sur ce qui est, ou non, sur le chemin
critique — dans des directions opposées. Vérifié à la main :

**SongFormer n'est PAS un point dur — il est éteint.** L'audit le classe
« difficulté n°1 » (2,7 Go de poids, tue un Mac 16 Go). Mais
`SETTINGS.sections = "aucune"` (`settings.py:91`, décision de Louis du
2026-09-19) et l'appel songformer est dans la branche `else` de
`pipeline.py:750`. **Il ne s'exécute jamais.** 2,7 Go et le pire problème de
mémoire du projet disparaissent du portage.

*Conséquence à assumer :* sur mobile, un morceau neuf sort avec **une seule
section « A »**. Le découpage vient de l'outil d'annotation, qui n'est pas
dans la v1. Charts justes, mais sans forme.

**`nnls-chroma` EST sur le chemin critique.** L'audit Android le dit optionnel
(« le défaut est `bp48` »). Faux : `extract_bothchroma()` est appelé
**inconditionnellement deux fois** par analyse, `pipeline.py:744` et
`pipeline.py:845`, et alimente aussi `refold.py`, `span_rescore.py`,
`debut.py`. C'est un plugin VAMP C++ (GPL, Mauch & Cannam) sans implémentation
Python de référence. Il faut le compiler au NDK, ou le réécrire.

---

## 3. L'import YouTube

Aucune app ne fait l'extraction sur le téléphone. Chord AI, Chordify,
Songsterr analysent **côté serveur**, avec un outil type yt-dlp, et ne
restituent jamais de fichier audio — seulement des accords, une donnée
dérivée. C'est ce qui leur fait passer la review.

Les deux voies propres sont fermées, vérifié : le lecteur **IFrame officiel ne
donne aucun accès au flux audio**, et l'API YouTube Data ne rend que des
métadonnées.

**Précédent :** Musi, retirée de l'App Store en septembre 2024 (intervention
YouTube + IFPI, motif DMCA). Guideline Apple 5.2.3 : interdit de
`download media from third-party sources`.

| Plateforme | Import YouTube embarqué | Comment |
|---|---|---|
| **Android sideload** | **oui** | `youtubedl-android` (wrapper yt-dlp, actif en 2026) ou `NewPipeExtractor` |
| Android Play Store | non | NewPipe lui-même n'y est jamais allé, exprès |
| iOS | non | rejet quasi certain ; reste le partage de fichier depuis une autre app |

Réserve : yt-dlp embarqué hérite du même problème de PO-token et de cookies
qu'aujourd'hui sur le Mac. Ça marchera, et ça cassera régulièrement.

---

## 4. La pile recommandée

| Brique | Choix | Note |
|---|---|---|
| Réseaux (`beat_this`, `musx`) | `torch.export()` → **ExecuTorch**, backend XNNPACK | chemin cross-plateforme, utilisé en prod par Meta pour de l'audio ; PAS Chaquopy (torch y est bloqué à 1.13.1) |
| Glue Python | **Chaquopy** (gratuit/MIT depuis v12) | numpy officiellement supporté |
| UI | le JS existant dans une **WebView**, servi par Flask local | portage UI ≈ zéro |
| `nnls-chroma` | compilation NDK | inconnue principale |
| scipy | à retirer | **un seul usage** : `find_peaks`, `sections/similarity.py:53` — dix lignes de numpy |

Poids à embarquer : `beat_this` 9 Mo (checkpoint réduit `small0`) ou 97 Mo
(complet), `musx` 5 × 5,7 Mo = 28,7 Mo.

---

## 5. Phase 0 — 3 jours, avant de s'engager

Règle n°2 du projet : screener la prémisse avant d'implémenter. Trois tests,
chacun peut tuer le projet.

| # | Test | Où | Verdict si ça casse |
|---|---|---|---|
| 1 | `torch.export()` de `beat_this` (small0) et d'un fold `musx` — le STFT/mel survit-il ? | Mac | retomber sur un runtime torch embarqué, +2 semaines |
| 2 | compiler `nnls-chroma` au NDK Android, chronométré | Mac | réécrire la NNLS en numpy, +1 semaine |
| 3 | lancer les réseaux exportés sur l'Android réel, chronométrer un morceau de 4 min | **téléphone, semaine pro** | **> 2-3 min de calcul = le projet meurt**, retour à l'hybride |

Les tests 1 et 2 sont faisables immédiatement sur le Mac. Le 3 attend le
téléphone.

**Critère d'arrêt** : si les trois passent, les 3-4 semaines sont solides. Si
le 3 échoue, rien de ce qui précède ne sert.

---

## 6. Le périmètre v1, si on y va

**Dedans :** analyser un morceau (micro, fichier, YouTube en sideload),
écouter le chart, la bibliothèque locale.
**Dehors, sur le Mac :** annotation des sections, mode Jam, import iReal,
corrections, verdicts — 6 877 lignes qui attendent une v2.
