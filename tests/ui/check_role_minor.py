"""Rôle d'un accord dans un morceau en MINEUR (vérif. visuelle lot 2 UX,
2026-10-01) : en sol mineur, Cm s'affichait « iv : emprunté au mineur »,
E♭ et B♭ « emprunt modal », F « emprunté au mixolydien » — alors que ce sont
les degrés naturels de la gamme mineure. Le majeur ne doit pas bouger.

    .venv/bin/python tests/ui/check_role_minor.py
"""
import sys
from playwright.sync_api import sync_playwright
G = 7
CASES = [  # (mode, root, q) -> mots attendus / interdits
    ("minor", 0, "-",  ["iv", "sous-dominante mineure"], ["emprunt"]),
    ("minor", 3, "",   ["III", "relatif majeur"], ["emprunt"]),
    ("minor", 10, "",  ["III"], []),  # placeholder replaced below
]
CASES = [
    ("minor", (G+5) % 12, "-",  ["iv", "sous-dominante mineure"], ["emprunt"]),
    ("minor", (G+3) % 12, "",   ["III", "relatif majeur"], ["emprunt"]),
    ("minor", (G+8) % 12, "",   ["VI"], ["emprunt"]),
    ("minor", (G+10) % 12, "",  ["VII"], ["emprunt", "mixolydien"]),
    ("minor", G, "-",           ["i", "tonique"], []),
    ("major", (G+5) % 12, "-",  ["iv", "emprunté au mineur"], []),
    ("major", (G+10) % 12, "",  ["♭VII", "mixolydien"], []),
    ("major", (G+8) % 12, "",   ["♭VI", "emprunt"], []),
]
fails = []
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(); pg.goto("http://localhost:7772/"); pg.wait_for_timeout(800)
    for mode, root, q, want, banned in CASES:
        r = pg.evaluate("""async([m,root,q])=>{const K=await import('/static/ui/kit.js'); const P=await import('/static/screens/practise.js');
            const r=K.roleOf(7,root,q,null,m); return {num:r.num, text:r.text, court:P.roleCourt(r.text)}}""", [mode, root, q])
        full = r["num"] + " " + r["text"] + " | " + r["court"]
        ok = all(w in full for w in want) and not any(w in full for w in banned)
        print(("ok   " if ok else "FAIL ") + f"{mode} root={root} q={q!r}: {r['text']}  →  {r['court']}")
        if not ok: fails.append(full)
    b.close()
sys.exit(1 if fails else 0)
