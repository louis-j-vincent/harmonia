# 30 — Le modèle entend en avance, et on le corrigeait en retard

*2026-09-15*

Le refactor de la veille avait reproduit les 44 charts de la bibliothèque à
l'octet près. Louis : « I didn't want you to have 44/44 identical, you could
have changed some charts. » Un refactor qui ne change aucun chart n'a, par
construction, corrigé aucune erreur musicale. D'où un audit — par le test,
pas par la lecture.

## Ce qu'on a mesuré

**Les caches sont vrais.** Battues, postérieures musx et chroma recalculés à
froid sur trois morceaux : identiques au cache. L'audit avait un sol.

**Le modèle entend en avance.** Sur 46 morceaux, l'énergie de changement des
postérieures musx, corrélée à la grille de battues, culmine 23 à 46 ms AVANT
la battue — une à deux trames de 23 ms, la fenêtre d'analyse qui chevauche
l'attaque. Jamais après. Or le code cherchait, morceau par morceau, une
compensation de 0 à +280 ms : « le modèle entend en retard », héritage d'un
autre traceur de battues. Bornée du mauvais côté, la recherche rendait 0
partout, sauf là où elle trouvait un *alias* : +280 ms, c'est −163 ms décalé
d'une battue à 135 bpm. Sur Another Day, 78 accords écrits un temps trop tôt.

**Et le repli refaisait la même recherche**, sur un gabarit synthétique où il
n'y a aucune latence à compenser. 89 décodages de gabarit sur 96 choisissaient
une latence non nulle, 8 saturaient à 280 ms, et le consensus réécrivait
toutes les mesures de la lettre un temps trop tôt : 18 morceaux, 239 mesures,
sans qu'aucun champ du chart ne le montre. `Urdlvw0SSEc`, un 6/8, affichait
des « BBm » superposés au temps 6 ; il lit maintenant Em | Bm | Em | Bm.

**Stand By Me à 50 % de N.C.** n'était pas le modèle : le décodeur disait A A
F#m F#m D E A A sur les couplets, à 0,9. C'est le rendu qui montrait la
première passe de la section — l'intro basse-voix où musx n'entend rien. Le
correctif existait depuis le 10 août… sur l'autre chemin de rendu. Deux
chemins, deux lois.

## Ce qu'on a livré

Plus de recherche de latence, nulle part : un décodage, sur les battues.
Une seule loi pour la passe affichée. 22 morceaux changent, Louis a écouté
la page avant/après et accepté. Baseline re-gelée, bibliothèque republiée.

## Ce qu'on n'a pas résolu

Le re-calage harmonique de la phase de mesure ne se déclenche sur aucun
morceau : ses seuils sont insatisfaisables. Sur deux des trois marques
« Set bar 1 » de Louis, le vote des accords pointait pourtant sur sa phase.
Question ouverte, quatre morceaux.
