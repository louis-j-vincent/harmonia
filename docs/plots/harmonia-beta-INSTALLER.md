# Harmonia sur ton téléphone — notice d'installation (beta 2026-09-23)

APK : `~/harmonia/apk/harmonia-beta-2026-09-23.apk` (160 Mio, signé). Hash à
côté (`.sha256`) si tu veux vérifier le transfert.

## a) Transférer l'APK

**Par câble (le plus fiable) :**
```
adb install ~/harmonia/apk/harmonia-beta-2026-09-23.apk
```
Le téléphone branché en USB, débogage USB activé une fois (Réglages →
À propos → tape 7× sur "Numéro de build" → Options développeur → Débogage USB).

**Sans câble** : envoie le fichier par Google Drive (ou AirDrop-like via un
partage de fichier) vers le téléphone, ouvre-le depuis l'app Fichiers.
Android bloque l'installation par défaut : au premier essai il propose
« Autoriser depuis cette source » — accepte, c'est la seule fois.

## b) Premier lancement

L'app s'ouvre directement sur ta bibliothèque, avec **3 morceaux déjà
analysés** (This Love, Stand By Me, She Will Be Loved) — pour vérifier tout
de suite que le son et le chart marchent, sans rien télécharger. Tape sur
l'un d'eux : le chart s'ouvre, le lecteur joue.

## c) Connecter YouTube

Nécessaire une seule fois avant d'analyser un lien YouTube.
Préférences → **« Connect YouTube »** → connecte-toi à ton compte Google
normalement dans l'écran qui s'ouvre → tape **« Terminé »** en haut.
*Si un import échoue ensuite avec « Sign in to confirm you're not a bot »*,
refais ces 3 gestes — la session dure environ 10-20 minutes d'usage, pas
plus, c'est mesuré et pas encore résolu autrement.

## d) Analyser un morceau

Trois façons, dans l'écran de recherche : colle un **lien YouTube**, choisis
un **fichier** sur le téléphone, ou lance le **micro**. À la fin, une ligne
apparaît sous "N mesures" : `Analyse : 1 min 43 s (battues 58 s, accords 40 s,
…)`. **Si elle commence par « ⚠ cache chaud — pas une mesure »**, le morceau
avait déjà été calculé avant (relu, pas recalculé) — n'en tiens pas compte.

## e) Le protocole T5 (le test qui compte)

Deux mesures FROIDES (sans le ⚠), à m'envoyer telles quelles :

1. **This Love** est déjà dans ta bibliothèque avec son chart, mais pas ses
   calculs intermédiaires — lance quand même une PREMIÈRE analyse dessus
   (les 3 points → réanalyser, ou recolle son lien YouTube) : sur un
   téléphone neuf, c'est froid. Recopie la ligne "Analyse : …".
2. Choisis **un morceau YouTube de ~4 minutes que tu ne connais pas encore**
   dans l'app, analyse-le (froid par construction). Recopie sa ligne.

Si l'une des deux commence par ⚠, dis-le-moi au lieu de me l'envoyer.

**Envoie-moi les deux lignes + une capture d'écran du chart obtenu.**

Le seuil : 4 minutes de musique doivent s'analyser en **moins de 2 minutes**.
Si l'une des deux lignes dépasse ce seuil, ce n'est pas cassé — c'est la
question ouverte de cette beta. La page des leviers connus (ce qu'on peut
accélérer et de combien) : `docs/portage_mobile/plots/fastmode_3songs.html`
(sur le Mac, pas sur le téléphone).

## f) Ce que cette version ne fait PAS encore

- Un morceau **neuf** sort avec **une seule section « A »** — pas de couplet/
  refrain détaillé (ça arrive en v2).
- Pas d'**annotation** manuelle des sections, pas de **Jam**, pas d'**import
  iReal**.
- Les **cookies YouTube** se dé-connectent après 10-20 minutes d'usage — à
  reconnecter via Préférences → Connect YouTube (voir c).

## g) Si ça plante — récupérer les logs

Téléphone branché en USB, sur le Mac :
```
adb logcat -s harmonia.js python.stdout python.stderr AndroidRuntime
```
Relance l'action qui plante pendant que ça tourne, puis envoie-moi ce qui
s'affiche (ou une capture).
