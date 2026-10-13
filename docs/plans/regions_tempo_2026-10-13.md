# Régions de rythme — plan (2026-10-13)

Louis : « Attention sur The APL Song : l'intro a un tempo différent du reste du
morceau. Il faut une brique qui détecte les différentes portions avec des
rythmes différents. »

## Ce que la mesure a dit d'abord (règle 2)

- **Le tempo d'APL ne change pas** : 89,0 BPM dans l'intro (16,3 → 38,5 s) et
  dans le corps (pente des battues Beat This!, écart max 0,02 s).
- **La barre de mesure change** : l'intro dure 33 temps (8 mesures + 1). Les
  changements d'accords musx (lus image par image, indépendants des battues)
  tombent sur une grille dans l'intro, sur la même grille décalée d'un temps
  dans le corps ; les downbeats de Beat This! sautent d'un temps au même
  endroit. Retour sur la barre de l'intro vers 121 s.
- Le chart : `bars.layout_bars` pose UNE phase (celle de la marque mesure 1,
  ici `harmonia.debut` à 16,5 s) → mes. 17-48 écrites un temps trop tôt.
- Le garde-fou 007d51c n'y est pour rien : la grille rigide
  (`beats._poser_grille_rigide`) a déjà rendu les battues régulières, il n'a
  rien à juger (`meta.rythme = []`).
- Un vrai changement de tempo est DÉJÀ suivi : les mesures sont comptées en
  temps du traqueur (test : intro lente de 8 mesures pile → mesures de 3,2 s
  puis 2,4 s sans rien faire).

## Le principe

Une portion d'au moins une phrase (4 mesures) a sa propre barre de mesure si
**deux témoins indépendants** s'accordent : (1) ses changements d'accords
tombent sur des temps plus forts (1 > 3 > 2/4) sous sa phase que sous celle de
la voisine, test du signe unilatéral 5 % — et réciproquement pour la voisine ;
(2) les downbeats du traqueur y tombent le plus souvent sur cette même phase.
Aucun seuil réglé : 4 mesures = une phrase, 5 % = le test du garde-fou.

## Où ça se branche

- `harmonia/regions_rythme.py` : `detecter_regions(beats, downbeats, triad,
  frame_dt, bpb)` → `{"regions": [{i0, i1, phase, t0, t1, bpm, beatsPerBar}],
  "rapport", "rejets"}` ; découpe binaire (profondeur ≤ 3) puis consolidation
  (chaque frontière rejugée entre ses voisines réelles). `lignes_de_mesure`.
- `pipeline.analyze_steps` : étape 2c, **après** le garde-fou (il répare les
  battues, la brique ne touche qu'aux barres) ; les downbeats du re-décodage
  suivent les régions ; la marque mesure 1 ne re-phase que SA portion.
- `bars.layout_bars(..., regions=)` : barres portion par portion, la mesure de
  jonction est courte (1 à 3 temps) ; le re-calage harmonique global est sauté
  (les régions sont déjà le verdict de l'harmonie). Sans régions : code
  inchangé (63/69 morceaux identiques au rapport d'or).
- Chart : `tempoRegions = [{t0, t1, bpm, beatsPerBar, phase}]`, seulement
  quand il y a ≥ 2 portions.
- `SETTINGS.regions_rythme = True` (constante, pas de variable d'environnement).

## Ce que ça ne résout pas

- les portions SANS pulsation (APL 0-16 s : 6 mesures vides au tempo du corps) ;
- un verrou d'octave LOCAL (le traqueur à demi-tempo sur 8+ mesures : GYBR,
  Syracuse `fOw6_NaT24k`, `fd02pGJx0s0` au balayage) — le vrai « autre tempo »
  que la grille ne voit pas, prochain levier ;
- un changement de métrique (3/4), un rubato, un accelerando ;
- une portion de moins d'une phrase (une mesure de 2/4 isolée) ;
- la mesure courte s'affiche comme une case pleine ;
- une ré-analyse décale de +1 les numéros de mesure après chaque frontière :
  un découpage de sections validé À LA MAIN (en numéros de mesure) sur un
  morceau qui gagne une frontière serait à revoir (aucun des 6 concernés
  aujourd'hui n'en a).

## Tests (`tests/test_regions_rythme.py`)

APL synthétique (33 temps puis corps) → 2 portions, mesure d'un temps ;
intro lente 8 mesures + 1 temps → 3,2 s / 0,8 s / 2,4 s ; intro lente pile →
pas de région ; marque mesure 1 = sa portion seulement ; un seul témoin
(traqueur seul / harmonie seule) → rien ; triolets de That's Life → recousus
et sans région ; portion < 1 phrase → rien ; forme réelle d'APL sur ses
caches ; règle 1 : battues et images musx dans la durée réelle du fichier.
