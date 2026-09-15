# Harmonia — handoff cleanup (2026-08-19)

**Fait.** Les cinq tâches sont livrées et commitées, une par commit, chacune
vérifiée à l'écran. Récit : `docs/blog/29-un-ecran-une-chose-a-faire.md`.

| Tâche | Commit | État |
| --- | --- | --- |
| T1 — la bande de forme à 22 px, groupée | `cd10166` | ✓ (le diff de `app_shell.html` a été emporté par `738b4bb`, voir plus bas) |
| T2 — Annotate : un seul « ··· » | `de0ee39` | ✓ |
| T4 — le doute rentre dans la case | `79f5c53` | ✓ |
| T5 — Practise : trois zones + le clavier | `189379d` | ✓ |
| T3 — l'éditeur de sections plein écran | `c1ff8ba` | ✓ **sans entrée** (voir plus bas) |
| suites du vrai rendu | `c79e3ed` | ✓ |

## Où voir le résultat

L'app en prod le sert déjà (`server.py` relit `app_shell.html` à chaque
requête, aucun redémarrage nécessaire) :

* le chart, la bande, Annotate, Practise — http://100.89.209.63:7772/
* l'éditeur de sections de T3, qui n'a **aucun bouton** :
  http://100.89.209.63:7772/?open=min_T64BgKEL-Sw&edit=sections

## T3 n'a pas d'entrée — pourquoi

La route serveur existe bien (`POST /api/soudure/valider/<file>` : elle écrit le
découpage ET ré-empile les accords en conséquence), donc l'écran est complet et
persiste pour de vrai. Mais il fait le même travail que l'outil au doigt du
chart (`buildSectionTool`), écrit en parallèle dans une autre session. Arbitrage
de Louis, 2026-08-19 : construire T3, ne pas lui donner de porte, trancher à
l'œil. Le brancher coûte une ligne dans `openAnnotateTools()`.

## Ce qui a dévié de la spec, et pourquoi

* **T2, « Vérifier les fusions »** n'est pas rendu : `/debug/section-merge-game`
  répond 404 « not in this build ». *Souder les sections* (`/soudure/<file>`)
  tient ce rôle depuis le 2026-08-14 et existe, elle. La feuille porte aussi
  *Voir la matrice*, que le ruban supprimé abritait.
* **T4, le « ✓ »** n'existe pas dans ce fichier : c'est le mot `locked` qui
  tenait ce rôle. Il devient le tiret vert de 14×2 que la spec décrit.
* **T5, « le style de voicing »** n'est pas dans la feuille : `S.coachStyle` ne
  touche pas cet écran (le prompter voice avec sa propre cascade). À sa place,
  *Mains* (deux/une), qui est le seul choix qui change ce que le clavier dessine.
* **T1, le trait de répétition** est en `T.line` (#e5dcc6), pas en #e2d9c3 :
  l'invariant « aucune couleur nouvelle » l'emporte, l'écart est invisible.

## Le harnais

| Fichier | À quoi ça sert |
| --- | --- |
| `CLEANUP_SPEC.md` | le document de travail (T1→T5) |
| `Harmonia Cleanup.dc.html` | la maquette, 5 frames à 390 px |
| `regen_mock.py` | régénère `mock_live.html` = `app_shell.html` + le prélude. **À relancer après chaque pull.** |
| `mock_prelude.html` | le prélude : `/api/*` mocké sur Autumn Leaves + les drapeaux de test |
| `check.py` | ouvre le mock à 375×667, exécute une sonde JS, prend une capture |

Drapeaux du mock (à ajouter à l'URL) :

* `?form=stress` — le pire cas de Louis : `A×3 B A×3 B A×3 B bridge B A×3`,
  17 passages, 9 groupes ;
* `?marks=1` — deux accords à `c=0,30` (le point) et deux verrouillés (le tiret) ;
* `?audio=1` — un WAV silencieux de 48 s fabriqué en JS : la lecture, la tête de
  lecture et le suivi d'accord du clavier deviennent exerçables sans serveur ;
* `?edit=sections` — l'éditeur de T3.

`regen_mock.py` injecte en plus, **dans le mock seulement**, un `API._t` qui
expose `S`, `formGroups`, `paintFormRail`, `setPlayhead`… — c'est par là que les
sondes mesurent. `app_shell.html` ne le porte jamais.

Exemple :

```
python regen_mock.py
python check.py out.png --query "?open=autumn_leaves&form=stress" --js sonde.js
```

## Un accident à connaître

Le diff `app_shell.html` de T1 a été **emporté par le commit d'une session
concurrente** (`738b4bb`, « folds: moyenne des logits ») dans la fenêtre entre
`git apply --cached` et `git commit` : l'arbre de travail est partagé, `git add`
d'une session voit l'index de l'autre. Rien n'est perdu, l'historique est
linéaire, mais le commit T1 (`cd10166`) ne porte que le harnais. Depuis, chaque
`apply --cached` et son `commit` partent dans **une seule commande**.
