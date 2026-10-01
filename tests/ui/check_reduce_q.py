"""Détail des accords (Triades / Septièmes) : chaque qualité rencontrée dans la
bibliothèque doit être simplifiée, et « Septièmes » garde la couleur de la
septième (ø7, °7, m(maj7), 7sus4). Trouvé en vérifiant le lot 2 UX
(2026-10-01) : D7♭13 restait D7♭13 en Triades, Aø7 devenait A° en Septièmes.

    .venv/bin/python tests/ui/check_reduce_q.py
"""
import sys
from playwright.sync_api import sync_playwright

# qualité -> (Triades, Septièmes)
EXPECT = {
    "": ("", ""), "7": ("", "7"), "-": ("-", "-"), "-7": ("-", "-7"), "^7": ("", "^7"),
    "o": ("o", "o"), "o7": ("o", "o7"), "h7": ("o", "h7"), "h": ("o", "h7"), "-7b5": ("o", "h7"),
    "7sus4": ("sus4", "7sus4"), "7sus": ("sus4", "7sus4"), "9sus": ("sus4", "7sus4"), "sus4": ("sus4", "sus4"),
    "9": ("", "7"), "13": ("", "7"), "7b9": ("", "7"), "7#11": ("", "7"), "7b13": ("", "7"),
    "13b9": ("", "7"), "7#5": ("+", "7"), "+": ("+", "+"),
    "-9": ("-", "-7"), "-11": ("-", "-7"), "-6": ("-", "-"), "-^7": ("-", "-^7"), "-^9": ("-", "-^7"),
    "^": ("", "^7"), "^9": ("", "^7"), "^13": ("", "^7"), "6": ("", ""), "69": ("", ""),
}
fails = []
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page()
    pg.goto("http://localhost:7772/"); pg.wait_for_timeout(800)
    got = pg.evaluate("""async(qs)=>{const K=await import('/static/ui/kit.js');
        const o={}; for(const q of qs) o[q]=[K.reduceQ(q,1),K.reduceQ(q,2),K.reduceQ(q,3)]; return o;}""", list(EXPECT))
    b.close()
for q, (l1, l2) in EXPECT.items():
    g = got[q]
    if (g[0], g[1]) != (l1, l2) or g[2] != q:
        fails.append(q); print(f"FAIL {q!r}: got {g[:2]} want {[l1, l2]}")
print("OK" if not fails else f"{len(fails)} FAIL"); sys.exit(1 if fails else 0)
