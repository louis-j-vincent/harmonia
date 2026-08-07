---
name: branch-safety
description: Use before ANY git commit, push, checkout, branch creation, or before briefing a subagent that will touch git, in this repo. Louis runs several Claude sessions on one shared working tree, so the current branch is global mutable state that another session can change under you mid-task.
---

# The branch is global mutable state here

Louis, 2026-08-07, after finding our commits on a branch neither of us named:

> « Attends, deux agents sur deux sessions différentes vont envoyer dans la
> branche l'un de l'autre ? Si c'est le cas crée un skill dès que tu modifies et
> push pour s'assurer que c'est dans la bonne branche. »

## What actually happens

`/Users/vincente/Documents/Projets Perso/Code/harmonia` is **one working tree**
shared by every concurrent session and by every subagent that does not use an
isolated worktree. `git checkout` there is not local to a session: it moves HEAD
for everyone. The reflog of 2026-08-07 shows it plainly —

    19:14  checkout: feat/raw-chart-first  -> feat/intro-vocal-cues     (an agent)
    23:34  checkout: feat/intro-vocal-cues -> fix/annotation-musx-chords (another session)

Three of that session's commits landed on a branch created by a subagent, and
three more on a branch belonging to a different session. Nothing was lost — the
history stayed linear — but the branch names no longer meant anything.

## The rules

**1. Record the expected branch when a work thread starts.** One line, kept in
mind for the whole thread: *"we are on X"*.

**2. Check before every commit.** `git branch --show-current`. If it is not X,
STOP. Do not commit "since it is all the same history": say so to Louis, and let
him choose between switching back and adopting the new branch. A commit on the
wrong branch is cheap to make and expensive to explain a day later.

**3. Check again before every push, and check what you are about to send.**
`git log --oneline @{u}..HEAD` (or `origin/main..HEAD` with no upstream). Pushing
a branch that another session owns mixes two unrelated stories into one PR.
Never push a branch you did not create without Louis saying so explicitly.

**4. Never `git checkout` in the shared tree to "get out of the way".** It moves
everyone. If you need a different branch, either ask, or use a real worktree:

    git worktree add /tmp/wt-<name> -b feat/<name>

**5. Brief subagents accordingly — this is where the damage came from.** Telling
an agent "create your own branch" makes it run `git checkout -b` in the shared
tree and silently moves the parent session's branch. Either:
  * tell it explicitly to commit on the CURRENT branch and never to run
    `checkout`, `switch`, `reset`, `stash` or `rebase`; or
  * launch it with `isolation: "worktree"` so it gets its own checkout.
Say which one in the brief, every time.

**6. When you notice it has already happened, report it as a fact.** Give the
reflog lines, say which commits went where, and say plainly whether anything was
lost. Do not quietly `git checkout` back — that moves the other session again.

## Also true in this repo

`git add -A`, `git add .`, `git commit -a` and `--no-verify` are forbidden
(CLAUDE.md). Stage explicit paths only. On a shared tree this is not style: other
sessions have uncommitted work in the same directory, and `git add -A` sweeps it
into your commit. That has already happened here — a `/plots` route from another
session was nearly swept into an unrelated commit on 2026-08-07.
