#!/bin/sh
# Hook Stop (fin de tour) : RAPPEL seulement, jamais de commit automatique —
# l'arbre est partagé entre plusieurs sessions Claude, un `git add -A` ici
# balaierait le travail d'une autre session (incident du 2026-08-07).
# Règle CLAUDE.md « Commits are automatic » (Louis, 2026-09-24).
cd "${CLAUDE_PROJECT_DIR:-.}" 2>/dev/null || exit 0
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0
n=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
[ "${n:-0}" -gt 0 ] || exit 0
b=$(git branch --show-current 2>/dev/null)
printf '{"systemMessage":"⚠ travail non commité : %s fichier(s) sur %s — CLAUDE.md exige un commit (chemins explicites, jamais add -A) avant de finir"}\n' "$n" "${b:-HEAD détaché}"
