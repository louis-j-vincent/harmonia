#!/usr/bin/env python3
"""« Toujours montrer l'illustration », rendu mécanique au lieu de déclaratif.

Hook `Stop`. Louis, 2026-10-09 : « skill -> always show illustration », puis
« make it a hook ». Le skill `show-illustration` est une consigne que je peux
oublier ; ce hook ne l'oublie pas.

Règle : si, depuis le dernier message de Louis, un fichier VISUEL a été
modifié (Edit/Write sur harmonia/static/, docs/plots/*.html, ou un outil qui
génère une page : tools/*page*.py, tools/*demo*.py) et qu'aucun `SendUserFile`
n'a suivi dans le même tour, la fin du tour est REFUSÉE une fois, avec la
consigne de capturer le vrai rendu, de le regarder, puis de l'envoyer.

Garde-fous : `stop_hook_active` (déjà relancé une fois → on laisse finir, pas de
boucle) ; toute erreur de lecture → on laisse finir (un hook qui plante ne doit
jamais coincer la session).

Ce que ça NE couvre PAS : les modifications faites par un sous-agent (elles ne
sont pas dans ce transcript) et un `git commit` seul (on ne voit pas ses
fichiers). Ce sont les cas où le skill reste la seule garantie.
"""
import json
import re
import sys

VISUEL = re.compile(
    r"(/harmonia/static/"
    r"|/docs/plots/[^/]*\.html$"
    r"|/tools/[^/]*(page|demo)[^/]*\.py$)"
)
OUTILS_ECRITURE = {"Edit", "Write", "NotebookEdit"}


def _est_prompt(e) -> bool:
    """Un message utilisateur qui n'est PAS que des résultats d'outils."""
    if e.get("type") != "user" or e.get("isSidechain"):
        return False
    c = (e.get("message") or {}).get("content")
    if isinstance(c, str):
        return bool(c.strip())
    if isinstance(c, list):
        return any(isinstance(b, dict) and b.get("type") == "text" for b in c)
    return False


def main() -> int:
    try:
        d = json.load(sys.stdin)
    except Exception:
        return 0
    if d.get("stop_hook_active"):
        return 0
    path = d.get("transcript_path")
    if not path:
        return 0

    visuels, envoye = [], False
    try:
        with open(path, errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if _est_prompt(e):                 # nouveau tour : on repart de zéro
                    visuels, envoye = [], False
                    continue
                if e.get("type") != "assistant" or e.get("isSidechain"):
                    continue
                c = (e.get("message") or {}).get("content")
                if not isinstance(c, list):
                    continue
                for b in c:
                    if not isinstance(b, dict) or b.get("type") != "tool_use":
                        continue
                    nom = b.get("name") or ""
                    if nom == "SendUserFile" or nom.endswith("__SendUserFile"):
                        envoye = True
                    elif nom in OUTILS_ECRITURE:
                        fp = (b.get("input") or {}).get("file_path") or ""
                        if VISUEL.search(fp):
                            visuels.append(fp)
    except OSError:
        return 0

    if not visuels or envoye:
        return 0

    noms = sorted({v.rsplit("/", 1)[-1] for v in visuels})
    print(json.dumps({
        "decision": "block",
        "reason": (
            "Skill show-illustration (Louis, 2026-10-09 : « always show "
            "illustration »). Ce tour a modifié du visuel (" + ", ".join(noms[:4])
            + ("…" if len(noms) > 4 else "") + ") sans envoyer d'image. "
            "Capture le VRAI rendu (Playwright, 390 px, ?open=<clé>), regarde-le "
            "toi-même, corrige s'il montre un défaut, puis envoie-le avec "
            "SendUserFile (display: render) — avant/après côte à côte dans UNE "
            "image — avec une légende d'une ligne. Ensuite seulement, "
            "1-2 phrases de texte."
        ),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
