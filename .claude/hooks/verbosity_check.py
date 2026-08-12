#!/usr/bin/env python3
"""Le contrat de clarté, rendu mesurable au lieu de déclaratif.

Hook `Stop`. Lit le transcript, mesure la DERNIÈRE réponse texte, et affiche un
rappel si elle dépasse la limite.

Pourquoi : l'audit du 2026-08-12 (`docs/session_2026-08-12_diagnostic_boucle.md`)
a compté qu'un message sur quatre dépasse 700 caractères — la limite que CLAUDE.md
fixe depuis des semaines — et que la proportion n'a pas bougé d'un point en six
semaines. Une règle qu'on relit sans jamais la mesurer ne se corrige pas.

700 caractères ≈ 120 mots, le contrat écrit dans CLAUDE.md. Les blocs de code et
les tableaux ne comptent pas : ils sont exactement ce que Louis préfère à de la
prose, et les pénaliser pousserait dans le mauvais sens.
"""
import json
import re
import sys

LIMIT = 700


def main() -> int:
    try:
        d = json.load(sys.stdin)
    except Exception:
        return 0
    path = d.get("transcript_path")
    if not path:
        return 0

    last = None
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
                if e.get("type") != "assistant" or e.get("isSidechain"):
                    continue
                content = (e.get("message") or {}).get("content")
                if not isinstance(content, list):
                    continue
                txt = "\n".join(c.get("text", "") for c in content
                                if isinstance(c, dict) and c.get("type") == "text")
                if txt.strip():
                    last = txt
    except OSError:
        return 0

    if not last:
        return 0

    # Ce qui ne compte pas : blocs de code, tableaux, liens (l'URL n'est pas de
    # la prose). C'est la PROSE qu'on mesure.
    prose = re.sub(r"```.*?```", "", last, flags=re.S)
    prose = "\n".join(l for l in prose.split("\n") if not l.lstrip().startswith("|"))
    prose = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", prose)
    n = len(prose.strip())
    if n <= LIMIT:
        return 0

    print(json.dumps({"systemMessage":
                      f"réponse : {n} caractères de prose (contrat : {LIMIT}). "
                      f"Le détail va dans la page ou le doc, pas dans le chat."}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
