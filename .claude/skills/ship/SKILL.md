---
name: ship
description: Use whenever a change is finished and verified and it should reach Louis — and ALWAYS instead of waiting for him to say "push", "mets en prod", "commit". Runs the whole chain: golden report, tests, render check, log, commit, push, library publish. Also use when he does say it, to make sure nothing in the chain is skipped.
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

### 2. Est-ce que ça régresse ? — le rapport d'or

    .venv/bin/python -m tools.golden --engine harmonia --out state/cache/golden/ship \
        --baseline state/cache/golden/baseline

Ce n'est pas une porte qui bloque : c'est un rapport que Louis arbitre (« a
different chart doesn't mean you did bad work, you will have to tell me what
the differences are and I'll arbitrate »). Zéro mesure changée = rien à
arbitrer, on continue. Sinon :

    .venv/bin/python -m tools.avant_apres --before state/cache/golden/baseline \
        --after state/cache/golden/ship

publie la page AVANT/APRÈS (mesure par mesure, cliquable pour écouter) — montrer
à Louis, jamais trancher à sa place. Un chart froid (cache manquant) est
signalé, jamais relancé pour le rapport.

Si la modification ne touche pas les sections/le chart, sauter cette étape en
le disant.

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

### 7. Republier la bibliothèque, si ça change ce que l'app sert

Un chart est cuit une fois puis servi tel quel : changer une loi du repli ne
change donc RIEN à ce que Louis voit tant que les charts ne sont pas republiés.

    .venv/bin/python -m tools.golden --engine harmonia --out state/cache/golden/pub \
        --publish --backup <dossier de sauvegarde existant>

Refuse de démarrer sans `--backup` (sauf `--dry-run`) — une session concurrente
peut travailler sur les mêmes fichiers. Publie chart par chart, jamais toute la
bibliothèque d'un coup : une interruption au milieu ne doit jamais laisser une
bibliothèque mi-ancienne mi-nouvelle.

### 8. Redémarrer le serveur, si le code Python a changé

Il tourne **sans reloader** — un changement de code n'est pas servi tant qu'on
n'a pas redémarré. **Demander avant** : Louis est peut-être en train d'annoter.

    kill $(lsof -ti tcp:7772)
    nohup .venv/bin/python -m harmonia.server >> /tmp/harmonia.log 2>&1 &

**Toujours par PID du PORT, jamais par motif de ligne de commande** (`pkill -f
…` a déjà tué le process VIVANT en tuant tout ce qui matchait le nom du module,
alors qu'une autre instance tournait sur un autre port). Un process de 4 jours
a déjà servi du code périmé pendant toute une séance de débogage.

### 9. Pousser

    git push -u origin <branche>

### 10. Le dire en une ligne

Un lien cliquable en `http://100.89.209.63:7772/…` (le `file://` est mort sur
son téléphone), et **une prochaine étape nommée** — pas un menu.

## Le raccourci honnête

Si une étape ne s'applique pas, la sauter **en le disant**. Ce qui n'est pas
permis, c'est de la sauter en silence : « poussé » quand la bibliothèque n'a pas
été republiée, c'est un faux positif qu'il découvre sur son téléphone.
