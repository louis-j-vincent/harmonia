# Harmonia — handoff cleanup (2026-08-19)

## Contenu

| Fichier | À quoi ça sert |
| --- | --- |
| `CLEANUP_SPEC.md` | **Le document de travail.** 5 tâches (T1→T5), une par commit, avec valeurs exactes, ce qui est supprimé, ce qui ne doit pas bouger, et des critères d'acceptation vérifiables à l'écran. |
| `Harmonia Cleanup.dc.html` | La maquette, 5 frames à 390 px. À ouvrir à côté de la spec (`support.js` doit rester dans le même dossier). |
| `mock_live.html` | `harmonia_min/app_shell.html` @ `f00a123` + un prélude qui mocke `/api/*` sur Autumn Leaves. C'est là que les agents vérifient leur travail, sans serveur. À régénérer après chaque pull. |

## Ordre d'exécution

1. **T1** — la bande de forme à 22 px, groupée (`A×3 B A×3 …` sur une ligne). Tout le reste
   en dépend : c'est le composant réutilisé en T3 et T5.
2. **T2** — Annotate : le ruban d'outils devient un « ··· » + une feuille, et il n'existe qu'en
   Annotate.
3. **T4** — les « ? » rentrent dans la cellule (indépendant, peut partir en parallèle de T2).
4. **T5** — Practise : trois zones + le clavier.
5. **T3** — l'éditeur de sections plein écran. **En dernier**, et seulement si la route serveur
   existe : sinon écrire la demande dans `docs/handoff_mission3_ui_contract.md` et ne pas livrer
   d'entrée dans la feuille Outils.

## Le prompt à coller (un par tâche)

> Applique la tâche **T<n>** de `CLEANUP_SPEC.md` dans `harmonia_min/app_shell.html`. Ne touche à
> aucune fonction non nommée dans la tâche. Quand tu as fini : ouvre `mock_live.html`, va sur
> l'écran concerné à 375 px de large, et recopie les critères d'acceptation un par un avec ✓/✗.
> Si un critère est ✗, corrige avant de répondre. Si une contrainte est impossible sans toucher
> autre chose, ARRÊTE-TOI et dis-le au lieu d'improviser. Réponds par la liste ✓/✗, les fonctions
> touchées, et une capture — pas de résumé en prose.

## La règle qui gouverne tout le reste

Un écran = une chose à faire. Read lit, Annotate corrige, Practise joue. Rien d'un mode n'apparaît
dans un autre ; tout outil qui n'est pas l'action principale de l'écran vit derrière un bouton
unique. En cas de doute sur un ajout : il ne va pas sur le chart.
