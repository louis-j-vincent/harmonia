---
name: ship
description: Use whenever a change is finished and verified and it should reach Louis — and ALWAYS instead of waiting for him to say "push", "mets en prod", "commit". Runs the whole chain: bench, render check, log, commit, push, library rebake. Also use when he does say it, to make sure nothing in the chain is skipped.
---

# Shipper, c'est une chaîne — pas un `git push`

L'audit du 2026-08-12 (`docs/session_2026-08-12_diagnostic_boucle.md`) a compté
**~120 prompts de Louis** — 7 à 8 % de tout ce qu'il écrit, stable sur six
semaines — dont le contenu entier est « push », « mets en prod », « t'as
commité ? ». C'est le gaspillage le plus bête du projet : il paie un tour de
conversation pour de la plomberie que je devrais faire tout seul.

**La règle : je ship dès qu'un palier est vérifié, sans qu'on me le demande.**
Un palier = un changement bien défini qui a passé son contrôle. Pas à la fin
d'une séance, pas quand il demande.

**La seule exception** (déjà dans CLAUDE.md) : ne pas commiter au milieu d'un
aller-retour conceptuel — « ça veut dire quoi X ? », un désaccord de design. On
laisse le tour se poser d'abord.

## La chaîne, dans l'ordre

### 1. La branche — avant tout le reste

Charger le skill `branch-safety` et le suivre. Le dépôt est un seul arbre de
travail partagé par plusieurs sessions ; la branche courante peut avoir changé
sous les pieds depuis le dernier tour.

### 2. Est-ce que ça régresse ?

    .venv/bin/python scripts/bench.py --quick    # ~2 s, les 4 morceaux chauds
    .venv/bin/python scripts/bench.py            # les 18 annotés

Sort en code 1 sur tout recul de plus de 0,005. **Un recul arrête le ship** — on
le comprend ou on l'assume explicitement dans le message de commit, on ne le
découvre pas trois jours plus tard.

Si le banc dit « LA MÉTRIQUE A CHANGÉ » : les écarts ne mesurent pas le modèle.
Relancer `--save` et l'écrire dans le message de commit.

Si la modification ne touche pas les sections, sauter cette étape en le disant.

### 2 bis. Sauver les étiquettes de Louis

    cp harmonia_min/state/sections/*.json docs/ground_truth/sections/
    git status --short docs/ground_truth/sections/

`harmonia_min/state/` est **gitignoré**. Les annotations de sections y vivent, et
elles sont la contrainte qui dimensionne tout le projet — une annotation non
copiée est une annotation non sauvegardée. Le 2026-08-12, celle qu'il venait de
faire n'était dans aucun commit. Copier à chaque ship, et commiter le miroir avec
le reste.

### 3. Les tests du périmètre touché

    .venv/bin/python -m pytest tests/ -q --no-cov -k "<le sujet>"

### 4. Regarder le rendu réel — jamais la sortie du script

Playwright, 390 px, la vraie URL servie. Vérifier : contenu attendu présent,
zéro débordement horizontal, zéro erreur console. C'est une règle payée quatre
fois (mémoire `feedback-verify-rendered-chart`) : le script qui dit « ok » et la
page qui est cassée, c'est arrivé assez souvent.

### 5. Écrire ce qu'on a appris, pas seulement ce qu'on a changé

* un bug caractérisé ou résolu → `docs/known_issues.md`
* un résultat, même négatif → `docs/blog/` ou `docs/research_sessions/`
* un résultat négatif → charger le skill `negative-result` d'abord

### 6. Commiter — chemins explicites, jamais `-A`

    git add <chemin> <chemin>          # JAMAIS `git add -A` / `.` / `-a`
    git commit -F -                    # message multi-ligne par stdin

Le message dit, dans cet ordre : ce que Louis a demandé (ses mots si on les a),
ce qui a changé, **ce que ça ne résout pas**, et les chiffres avant/après quand
il y en a. Terminer par :

    Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>

### 7. Mettre en prod, si ça change ce que l'app sert

L'app live est `harmonia_min` sur le port **7772**. Une loi de merge, un moteur
de sections, un changement de chart réécrivent tous les charts servis :

    .venv/bin/python scripts/rebake_library.py

Le serveur tourne **sans reloader** — un changement de code Python n'est pas
servi tant qu'on n'a pas redémarré. Avant de redémarrer : **demander**, il est
peut-être en train d'annoter. Un process de 4 jours a déjà servi du code périmé
pendant toute une séance de débogage.

### 8. Pousser

    git push -u origin <branche>

### 9. Le dire en une ligne

Un lien cliquable en `http://100.89.209.63:7772/…` (le `file://` est mort sur
son téléphone), et **une prochaine étape nommée** — pas un menu.

## Le raccourci honnête

Si une étape ne s'applique pas, la sauter **en le disant**. Ce qui n'est pas
permis, c'est de la sauter en silence : « poussé » quand la bibliothèque n'a pas
été recuite, c'est un faux positif qu'il découvre sur son téléphone.
