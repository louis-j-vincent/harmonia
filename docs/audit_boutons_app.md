# Audit des boutons de l'app (`harmonia_min`, :7772) — 2026-08-13

Louis : « j'aimerais faire un cleanup et virer tous les boutons qui sont
maintenant redondants : deprecated, fais moi un audit et je te dis quoi garder
/ enlever / +bouger ».

**137 contrôles** dans `harmonia_min/app_shell.html` (6538 lignes), répartis sur
56 fonctions d'écran. Aucun n'est orphelin : tous sont atteignables depuis un
autre bouton. Le tri ci-dessous ne porte donc pas sur du code mort, mais sur
**ce qui échoue à l'usage** et **ce qui fait doublon**.

## Méthode

1. Inventaire extrait du source (`kitButton` / `kitIcon` / `kitSegmented` /
   `seg(` / `el("button")` / `mkRow` / `.onclick`), avec la fonction
   englobante comme écran.
2. Remontée de la chaîne d'appels de chaque écran, pour distinguer
   « atteignable » de « atteignable en théorie ».
3. **Chaque endpoint appelé par l'app a été tapé sur le serveur qui tourne.**
   C'est ce qui a fait la différence : le tri par lecture du code aurait raté
   les entrées ci-dessous, qui ont l'air parfaitement vivantes dans le source.
4. **Puis la table de routage de Flask (`app.url_map`) comme juge de paix.**
   Ajoutée après coup, et pas par excès de zèle : mon premier passage tapait
   les routes avec une méthode HTTP devinée, et j'ai déclaré morte une
   suppression de chart qui marche très bien (elle est en DELETE, je l'avais
   appelée en GET). Une route absente et une route appelée de travers rendent
   le même 404 — seul `url_map` distingue les deux.

Le serveur répond aux inconnues par un 404 propre
(`api_unimplemented`, `server.py:1127`) : « /api/… is not part of harmonia_min
milestone 1 ». Rien ne plante, donc rien ne se voit.

## 1 — Ce qui est MORT (l'endpoint répond 404)

| # | Bouton | Où | Ce qui se passe quand tu tapes | Endpoint |
|---|--------|-----|-------------------------------|----------|
| 1 | **Enregistrer** | pied de la bibliothèque, 52 px pleine largeur | l'écran s'ouvre, `REC` échoue | `POST /api/record-analyze` |
| 2 | **Jam** | pied de la bibliothèque, 52 px pleine largeur | l'écran s'ouvre, `START` échoue | `POST /api/jam/start\|chunk\|stop` |
| ~~3~~ | ~~× supprimer un chart~~ | Mes charts | **ERREUR DE MESURE, corrigée le 2026-08-14 : ça marche.** J'avais tapé la route en GET alors que l'app l'appelle en DELETE, et le 404 venait de ma requête, pas du serveur. Vérifié depuis sur une copie jetable : `{"ok":true}`, le fichier disparaît. La route est même documentée en tête de `server.py`. | `DELETE /api/chart/<file>` ✅ |
| 4 | **Export to iReal Pro** | feuille *Share* | toast « Could not export » | `GET /api/irealb-export/<file>` |
| 5 | **Recherche, mode iReal** | écran de recherche | zéro résultat, toujours | `POST /api/irealb-search` |
| 6 | **Importer un chart iReal** | résultats de recherche | toast « Could not import that chart » | `POST /api/irealb-import` |
| 7 | **Suggestions de fusion de MESURES** | chart (`S.suggMode`) | le mode s'arme, aucune suggestion n'arrive jamais | `GET /api/bar-merge-candidates/<file>` |
| 8 | **Suggestions de fusion de SECTIONS** | chart | idem | `GET /api/section-merge-candidates/<file>` |
| 9 | **Écran Billboard / Training mode** | — | **aucun bouton n'y mène** : `go("billboard")` n'est appelé nulle part (`app_shell.html:1850` ne fait que re-rendre si on y est déjà) | `GET /api/billboard-corpus` |

Le n° 3 est le seul qui soit un **bug** et pas seulement du bois mort : la
suppression a l'air de marcher.

Les n° 7 et 8 méritent une note : leurs feuilles de confirmation (« Sounds the
same — pool & re-infer ») appellent, elles, un endpoint bien vivant
(`/api/reinfer`). C'est la LISTE de suggestions qui est morte, pas l'action.
Rebrancher les deux routes ressusciterait les deux modes tels quels.

## 2 — Ce qui MARCHE mais fait doublon

### Quatre outils de sections, trois endroits

| Outil | Où | Geste | État |
|-------|-----|-------|------|
| **Marquer les sections** | ruban Annotate, légende « les sections · au doigt » | on passe le doigt sur les mesures | vivant (`/api/section-repeats`, `/api/section-marks`) |
| **Check merges** | ruban Annotate, légende « the sections · by ear » | ouvre `/debug/section-merge-game` dans un **nouvel onglet** | vivant |
| **Souder les sections** | feuille `Aa` (préférences) | la bande de jetons, on soude deux voisins | vivant (ajouté le 2026-08-13) |
| Suggestions de sections | chart | badges sur la grille | mort (§1 n° 8) |

Les deux légendes du ruban sont côte à côte et disent la même chose dans deux
langues : « les sections » et « the sections ».

### Trois portes vers un même re-décodage

`POST /api/reinfer` (ou son cousin `/api/context_rescore`) est atteint par :

* **Re-infer ·** — le bandeau des corrections en attente (`renderChart`) ;
* **⋈ Pool two passes** → **Pool & re-infer** — la sélection au doigt de deux
  passages (`openMerge`) ;
* **Sounds the same — pool & re-infer** — les feuilles de suggestions (mortes).

Les deux premières sont deux gestes différents pour le même effet ; la
troisième ne s'ouvre jamais.

## 3 — Ce qui est MAL PLACÉ

* **Souder les sections** est dans la feuille `Aa`, sous l'intitulé
  PRÉFÉRENCES, alors que c'est un outil de sections : sa place est dans le
  ruban Annotate, à côté de « Marquer les sections ». (Il y est en tête de
  feuille pour l'instant, faute de quoi il tombait sous la ligne de flottaison
  derrière neuf réglages.)
* **Enregistrer** et **Jam** occupent les deux plus gros boutons de la
  bibliothèque (52 px, pleine largeur, pied d'écran) pour deux écrans qui
  échouent.

## 4 — Ce qui est sain

Le reste tient : la barre du chart (retour / tonalité-rotor / `Aa`), le dock
(Read · Analyse · Annotate · Practise), la rangée de lentilles d'Analyse
(Role / Local keys / Home key), la feuille `Aa` (thème, notation, couleurs,
accords, tap, follow, clic, piano, voicing), Set bar 1, le partage par lien,
les dossiers, la recherche YouTube et l'import de tabs.
