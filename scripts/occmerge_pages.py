"""Pages d'écoute du chantier merge-d'occurrences (handoff 2026-08-08).

Lit les instantanés du harnais (`scripts/occmerge_harness.py`, un JSON par
morceau : bars première passe + chaque variante de fold rejouée) et écrit
les pages dans `harmonia_min/state/reports/` de l'ARBRE LIVE, pour que
:7772 les serve. Le pattern audio-cliquable vient de
`scripts/dictionary_audio.py` (un seul <audio>, seek t0, stop à t1).

    python scripts/occmerge_pages.py levier1
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SNAP = REPO / "harmonia_min" / "state" / "occmerge"
LIVE_REPORTS = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia"
                    "/harmonia_min/state/reports")
SCRATCH = Path(__file__).resolve().parents[2]   # …/scratchpad (worktree parent)

NOTES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def chord_txt(c: dict) -> str:
    if c.get("nc"):
        return "N"
    t = NOTES[c["root"]] + c["q"]
    if c.get("bass", -1) >= 0 and c["bass"] != c["root"]:
        t += "/" + NOTES[c["bass"]]
    return t


def bar_txt(bar: list[dict]) -> str:
    out = []
    for c in bar:
        t = chord_txt(c)
        if c.get("carry"):
            t = "(" + t + ")"
        out.append(t)
    return " ".join(out) or "·"


def bar_conf(bar: list[dict]) -> str:
    cs = [c.get("c") for c in bar if c.get("c") is not None
          and not c.get("carry")]
    return f"{min(cs):.2f}" if cs else "–"


CSS = """
*{box-sizing:border-box}
body{margin:0;padding:14px 10px 90px;background:#faf6ec;color:#2c2820;
     font:15px/1.45 system-ui,-apple-system,sans-serif;max-width:760px;
     margin-inline:auto}
h1{font-size:19px;margin:0 0 6px}
h2{font-size:16px;margin:26px 0 4px}
.lede{background:#fff7df;border:1px solid #e8d9a8;border-radius:10px;
      padding:10px 12px;font-size:14px;margin:10px 0}
.verdict{background:#eef4e6;border:1px solid #c6d9ab;border-radius:10px;
         padding:10px 12px;font-size:14px;margin:10px 0}
.note{color:#8a8371;font-size:12.5px}
.occ{margin:8px 0 2px;font-size:13px;color:#5c5648;font-weight:600}
.row{display:flex;flex-wrap:wrap;gap:4px;margin:2px 0 8px}
.chip{border:1px solid #d8cfb4;border-radius:8px;background:#fff;
      padding:3px 6px;min-width:52px;text-align:center;cursor:pointer;
      -webkit-user-select:none;user-select:none}
.chip .ch{font-weight:650;font-size:13.5px;white-space:nowrap}
.chip .cf{font-size:10.5px;color:#8a8371}
.chip.on{background:#8a2b2b;border-color:#8a2b2b;color:#fff}
.chip.on .cf{color:#f2d8d8}
.chip.tpl{background:#eef4e6;border-color:#a9c488}
.chip.skip{background:#f3efe4;color:#9a927e;border-style:dashed;cursor:default}
.chip.var{border-color:#c9762b;border-width:2px}
.pos{display:flex;flex-wrap:wrap;gap:4px;margin:2px 0 10px}
.cohchip{border-radius:8px;padding:3px 8px;font-size:12.5px;border:1px solid}
.cohok{background:#eef4e6;border-color:#a9c488}
.cohbad{background:#f8e3dc;border-color:#d99c88}
table{border-collapse:collapse;font-size:12.5px;margin:8px 0;width:100%;
      display:block;overflow-x:auto}
td,th{border:1px solid #ddd3b8;padding:3px 7px;text-align:left;
      white-space:nowrap}
th{background:#f3edda}
.legend{font-size:12px;color:#6b654f;margin:4px 0 14px}
.leg{display:inline-block;border-radius:6px;padding:1px 7px;margin-right:6px;
     border:1px solid #d8cfb4;background:#fff}
.leg.tpl{background:#eef4e6;border-color:#a9c488}
.leg.skip{background:#f3efe4;border-style:dashed}
.leg.var{border-color:#c9762b;border-width:2px}
"""

JS = """
const au=document.getElementById('au');let stopAt=null,cur=null;
au.addEventListener('timeupdate',()=>{
  if(stopAt!=null&&au.currentTime>=stopAt){au.pause();stopAt=null;
    if(cur){cur.classList.remove('on');cur=null;}}});
au.addEventListener('pause',()=>{if(cur){cur.classList.remove('on');cur=null;}});
function play(el,src,t0,t1){
  if(cur===el&&!au.paused){au.pause();return;}
  if(cur)cur.classList.remove('on');
  cur=el;el.classList.add('on');
  const go=()=>{try{au.currentTime=t0;}catch(e){}
    stopAt=t1;au.play().catch(()=>{el.classList.remove('on');cur=null;});};
  if(au.getAttribute('src')!==src){au.setAttribute('src',src);
    au.addEventListener('loadedmetadata',go,{once:true});au.load();}
  else go();}
"""


def chip(label, conf, src, t0, t1, cls="", extra=""):
    on = f" onclick=\"play(this,'{src}',{t0:.2f},{t1:.2f})\"" if src else ""
    cf = f"<div class=cf>{conf}</div>" if conf else ""
    return (f"<div class='chip {cls}'{on}><div class=ch>"
            f"{html.escape(label)}</div>{cf}{extra}</div>")


def page_shell(title, body):
    return (f"<meta charset=utf-8><meta name=viewport "
            f"content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style>"
            f"<body>{body}<audio id=au preload=auto playsinline></audio>"
            f"<script>{JS}</script>")


# ── page levier 1 ────────────────────────────────────────────────────────────

EXHIBITS = [  # (stem, lettre) — les 4 lettres où le gate bi-mesure agit
    ("maroon_5_she_will_be_loved_official_music_video", "B"),
    ("Ju8Hr50Ckwk", "D"),
    ("norah_jones_don_t_know_why", "C"),
    ("the_police_every_breath_you_take_official_music_video", "A"),
]


def levier1():
    probe = json.loads((SCRATCH / "lever1_probe.json").read_text())
    rescue = json.loads((SCRATCH / "lever1_rescue.json").read_text())
    shifted = [r for r in probe if r["s_star"] != 0]

    B = ["<h1>Levier 1 — aligner avant d'empiler : verdict, et ce qu'on "
         "propose à la place</h1>",
         "<div class=lede><b>La question à trancher</b> : le repli refuse "
         "aujourd'hui une lettre ENTIÈRE dès qu'une seule de ses positions "
         "diverge entre occurrences. Faut-il le laisser replier "
         "<b>bi-mesure par bi-mesure</b> (les positions vertes ci-dessous), "
         "en laissant les rouges déplié­es&nbsp;? Écoute les mesures : si les "
         "positions vertes sonnent bien comme la même musique à chaque "
         "occurrence, le repli par bi-mesure est juste.</div>",
         "<div class=verdict><b>Ce qui a été mesuré</b> (27 charts, "
         "scripts <code>occmerge_lever1_probe/rescue.py</code>) : "
         "l'hypothèse du handoff (« le modulo rigide désaligne, ≈71 % des "
         "dégâts ») est <b>réfutée</b>. 173 occurrences sur 182 s'alignent "
         "déjà au décalage 0 ; la pire désalignée était déjà écartée en "
         "variante ; et réaligner ne sauve <b>aucune</b> des 17 lettres "
         "refusées (détail en bas de page). Le vrai coût est ailleurs : "
         "<b>21 des 44 positions des lettres refusées sont individuellement "
         "cohérentes et perdent leur merge</b> à cause du veto par lettre."
         "</div>",
         "<div class=legend><span class=leg>mesure 1ʳᵉ passe (clique = "
         "écoute)</span><span class='leg tpl'>consensus mergé (gate "
         "bi-mesure)</span><span class='leg skip'>position refusée — chaque "
         "occurrence garde sa 1ʳᵉ passe</span><span class='leg var'>écartée "
         "en variante par le gate outlier</span> · le petit chiffre = "
         "confiance acoustique (0–1)</div>",
         "<p class=note>Vérité terrain : aucune référence iReal/tablature "
         "n'est affichée ici — sur ces 4 morceaux on n'en a pas de fiable "
         "sous la main ; c'est l'oreille qui tranche.</p>"]

    for stem, letter in EXHIBITS:
        snap = json.loads((SNAP / f"{stem}.json").read_text())
        grid, src = snap["grid"], snap["audio"]
        raw = snap["raw_bars"]
        rep_b = snap["variants"]["bibar"]["report"].get(letter, {})
        rep_p = snap["variants"]["prod"]["report"].get(letter, {})
        bars_b = snap["variants"]["bibar"]["bars"]
        P = rep_b.get("period")
        coh = rep_b.get("coh") or []
        pos_skip = set(rep_b.get("pos_skip") or [])
        cv_skip = set(rep_b.get("cv_skip") or [])
        variants = set(rep_b.get("variants") or [])
        occs = [o["range"] for o in snap["occurrences"]
                if o["label"] == letter]
        title = snap.get("title") or stem
        B.append(f"<h2>{html.escape(title)} — lettre {letter} "
                 f"(P={P}, {len(occs)} occurrences)</h2>")
        prod_txt = ("repli ACCEPTÉ" if "n_obs" in rep_p else
                    f"repli REFUSÉ en bloc — {rep_p.get('reason', '?')}")
        B.append(f"<p class=note>Aujourd'hui en prod : {prod_txt}.</p>")
        # cohérence par position
        chips = []
        for k, c in enumerate(coh):
            cls = "cohbad" if (c is None or c < 0.85) else "cohok"
            chips.append(f"<span class='cohchip {cls}'>pos {k} · "
                         f"{'–' if c is None else f'{c:.2f}'}</span>")
        B.append("<div class=pos>" + "".join(chips) + "</div>")
        # occurrences, première passe
        for i, (b0, b1) in enumerate(occs):
            B.append(f"<div class=occ>occurrence {i + 1} — mesures "
                     f"{b0}–{b1}</div><div class=row>")
            for b in range(b0, b1 + 1):
                if b >= len(raw):
                    continue
                cls = "var" if b in variants else ""
                B.append(chip(bar_txt(raw[b]), bar_conf(raw[b]), src,
                              grid[b], grid[b + 1], cls))
            B.append("</div>")
        # la ligne consensus (gate bibar)
        if P:
            b0 = occs[0][0]
            B.append("<div class=occ>ce que le gate bi-mesure écrit "
                     "(consensus ×N)</div><div class=row>")
            for k in range(P):
                b = b0 + k
                if k in pos_skip or k in cv_skip:
                    why = "CV" if k in cv_skip and k not in pos_skip else "coh"
                    B.append(chip(f"pos {k}", f"refusée ({why})", None,
                                  0, 0, "skip"))
                elif b < len(bars_b):
                    n = next((c.get("n_obs") for c in bars_b[b]
                              if c.get("n_obs")), "")
                    B.append(chip(bar_txt(bars_b[b]),
                                  f"{bar_conf(bars_b[b])} · ×{n}", src,
                                  grid[b], grid[b + 1], "tpl"))
            B.append("</div>")

    # tables de réfutation
    B.append("<h2>Annexe — la réfutation du levier 1 tel qu'énoncé</h2>")
    B.append("<p class=note>Décalage optimal par occurrence contre le "
             "centroïde des autres (balayage cyclique complet = un oracle "
             "sur toute la famille des décalages entiers) :</p>")
    B.append("<table><tr><th>morceau</th><th>lettre</th><th>occ</th>"
             "<th>P</th><th>meilleur décalage</th><th>gain cos</th></tr>")
    for r in sorted(shifted, key=lambda r: -r["gain"]):
        B.append(f"<tr><td>{html.escape(r['song'][:34])}</td>"
                 f"<td>{r['letter']}</td><td>{r['occ']}</td><td>{r['P']}</td>"
                 f"<td>{r['s_star']}</td><td>{r['gain']:+.3f}</td></tr>")
    B.append(f"</table><p class=note>… et les {len(probe) - len(shifted)} "
             "autres occurrences préfèrent le décalage 0.</p>")
    B.append("<p class=note>Lettres refusées « stack incoherent » : le "
             "réalignement n'en fait passer AUCUNE au-dessus du seuil 0,85 — "
             "lgHGU8gqz9U E 0,585→0,744 ; Chain of Fools A 0,823→0,838 ; "
             "Every Breath A <b>empire</b> 0,817→0,712 (la recherche colle "
             "au bruit). Les refus viennent du contenu, pas de la phase.</p>")
    for r in rescue:
        pass  # données complètes dans lever1_rescue.json, résumé ci-dessus
    out = LIVE_REPORTS / "occmerge_levier1.html"
    out.write_text(page_shell("Levier 1 — aligner avant d'empiler", "".join(B)))
    print(f"→ {out}")


# ── pages leviers 2/3/4 ─────────────────────────────────────────────────────

def _sig(bar):
    return [(c["root"], c["q"], c["nc"], c.get("carry", False)) for c in bar]


def _diff_bars(snap, va, vb):
    A, Bv = snap["variants"][va]["bars"], snap["variants"][vb]["bars"]
    return [b for b in range(len(A)) if _sig(A[b]) != _sig(Bv[b])]


def _letter_of(snap, b):
    for o in snap["occurrences"]:
        if o["range"][0] <= b <= o["range"][1]:
            return o["label"]
    return "?"


def _occ_rows(B, snap, letter, ranges, highlight=frozenset(), heat=None):
    """Lignes d'occurrences première passe (chips cliquables)."""
    grid, src, raw = snap["grid"], snap["audio"], snap["raw_bars"]
    for i, (b0, b1) in enumerate(ranges):
        B.append(f"<div class=occ>occurrence {i + 1} — mesures {b0}–{b1}"
                 f"</div><div class=row>")
        for b in range(b0, min(b1 + 1, len(raw))):
            cls = "var" if b in highlight else ""
            extra = ""
            if heat is not None and b in heat:
                extra = f"<div class=cf>H {heat[b]:.2f}</div>"
            B.append(chip(bar_txt(raw[b]), bar_conf(raw[b]), src,
                          grid[b], grid[b + 1], cls, extra))
        B.append("</div>")


def levier2():
    B = ["<h1>Levier 2 — moyenner robuste (médiane) plutôt que linéaire</h1>",
         "<div class=lede><b>La question à trancher</b> : sur TOUT le corpus "
         "(27 charts), remplacer la moyenne par la médiane ne change que "
         "<b>3 mesures, sur 2 morceaux</b> (la moyenne tronquée : zéro). "
         "Les voici. Écoute chaque paire : la version médiane est-elle "
         "meilleure&nbsp;? Si oui le levier est un micro-gain sûr ; sinon il "
         "est mort — dans les deux cas il n'est PAS le levier principal "
         "espéré par le handoff.</div>",
         "<p class=note>Vérité terrain : pas de référence fiable affichée "
         "ici ; l'oreille tranche. Le petit chiffre = confiance.</p>"]
    for stem in ("maroon_5_she_will_be_loved_official_music_video",
                 "maroon_5_this_love"):
        snap = json.loads((SNAP / f"{stem}.json").read_text())
        diffs = _diff_bars(snap, "prod", "median")
        if not diffs:
            continue
        grid, src = snap["grid"], snap["audio"]
        title = snap.get("title") or stem
        by_letter: dict[str, list[int]] = {}
        for b in diffs:
            by_letter.setdefault(_letter_of(snap, b), []).append(b)
        for letter, bs in by_letter.items():
            ranges = [o["range"] for o in snap["occurrences"]
                      if o["label"] == letter]
            rep = snap["variants"]["prod"]["report"].get(letter, {})
            B.append(f"<h2>{html.escape(title)} — lettre {letter} "
                     f"(P={rep.get('period')}), mesures changées : {bs}</h2>")
            _occ_rows(B, snap, letter, ranges, highlight=set(bs))
            B.append("<div class=occ>consensus : moyenne (prod) vs médiane"
                     "</div><div class=row>")
            for b in bs:
                pb = snap["variants"]["prod"]["bars"][b]
                mb = snap["variants"]["median"]["bars"][b]
                B.append(chip("moy " + bar_txt(pb), bar_conf(pb), src,
                              grid[b], grid[b + 1]))
                B.append(chip("méd " + bar_txt(mb), bar_conf(mb), src,
                              grid[b], grid[b + 1], "tpl"))
            B.append("</div>")
    out = LIVE_REPORTS / "occmerge_levier2.html"
    out.write_text(page_shell("Levier 2 — moyenne robuste", "".join(B)))
    print(f"→ {out}")


def levier3():
    import numpy as np
    rows, n_folded = [], 0
    for p in sorted(SNAP.glob("*.json")):
        snap = json.loads(p.read_text())
        assert not _diff_bars(snap, "prod", "logpool"), p.name
        pb = snap["variants"]["prod"]["bars"]
        lb = snap["variants"]["logpool"]["bars"]
        for b, (x, y) in enumerate(zip(pb, lb)):
            for cx, cy in zip(x, y):
                if cx.get("folded") and cx.get("c") is not None \
                        and cy.get("c") is not None and not cx.get("carry"):
                    n_folded += 1
                    rows.append((abs(cy["c"] - cx["c"]), cy["c"] - cx["c"],
                                 snap["stem"], b, chord_txt(cx),
                                 cx["c"], cy["c"], snap["audio"],
                                 snap["grid"][b], snap["grid"][b + 1]))
    rows.sort(reverse=True)
    dl = np.array([r[1] for r in rows])
    B = ["<h1>Levier 3 — produit des postérieures vs moyenne : "
         "AUCUN accord ne change</h1>",
         "<div class=lede><b>Le verdict, mesuré sur les 27 charts</b> : "
         "décoder le template sur le PRODUIT des postérieures (« le même "
         "accord vu N fois ») écrit exactement les mêmes accords que la "
         "moyenne, sur chaque mesure de chaque morceau. La loi de "
         "combinaison ne bouge pas l'argmax du Viterbi ici : les piles qui "
         "passent les gates sont déjà si d'accord que toutes les lois "
         "raisonnables désignent le même vainqueur. Il n'y a <b>rien à "
         "écouter</b> — ce qui est en soi la réponse du levier 3.</div>",
         f"<div class=verdict>{n_folded} accords mergés comparés · "
         f"écart de confiance produit−moyenne : médiane {np.median(dl):+.3f}, "
         f"pire {dl.min():+.3f}, meilleur {dl.max():+.3f}. Le produit rend "
         "les confiances plus TRANCHÉES (c'est sa nature de veto) sans "
         "changer un seul choix d'accord.</div>",
         "<p class=note>Les 12 plus gros écarts de confiance — même accord, "
         "confiance différente (clique = écoute) :</p>",
         "<table><tr><th>morceau</th><th>mesure</th><th>accord</th>"
         "<th>c moyenne</th><th>c produit</th><th>écouter</th></tr>"]
    for _, d, stem, b, ch, c0, c1, src, t0, t1 in rows[:12]:
        B.append(f"<tr><td>{html.escape(stem[:30])}</td><td>{b}</td>"
                 f"<td>{html.escape(ch)}</td><td>{c0:.2f}</td>"
                 f"<td>{c1:.2f}</td><td><span class=chip "
                 f"onclick=\"play(this,'{src}',{t0:.2f},{t1:.2f})\">▶"
                 f"</span></td></tr>")
    B.append("</table>")
    out = LIVE_REPORTS / "occmerge_levier3.html"
    out.write_text(page_shell("Levier 3 — produit vs moyenne", "".join(B)))
    print(f"→ {out}")


def levier4():
    import numpy as np
    import sys as _s
    _s.path.insert(0, str(REPO))
    from harmonia_min import musx as _musx
    stem = "Urdlvw0SSEc"
    snap = json.loads((SNAP / f"{stem}.json").read_text())
    diffs = _diff_bars(snap, "prod", "entropy")
    grid = snap["grid"]
    audio = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia"
                 "/docs/audio") / f"{stem}.m4a"
    probs = _musx.frame_posteriors(audio)
    tri = np.clip(probs[0], 1e-9, 1.0)
    Hf = -(tri * np.log(tri)).sum(1)          # entropie par frame
    heat = {}
    for b in range(len(grid) - 1):
        a = max(0, int(round(grid[b] / _musx.FRAME_DT)))
        z = min(len(Hf), int(round(grid[b + 1] / _musx.FRAME_DT)))
        if z > a:
            heat[b] = float(Hf[a:z].mean())
    diffs_by_letter: dict[str, list[int]] = {}
    for b in diffs:
        diffs_by_letter.setdefault(_letter_of(snap, b), []).append(b)
    B = ["<h1>Levier 4 — pondérer par l'entropie : inerte partout sauf UN "
         "morceau</h1>",
         "<div class=lede><b>La question à trancher</b> : sur 27 charts, "
         "pondérer chaque occurrence par la confiance de ses frames "
         "(entropie basse = poids fort) ne change RIEN sur 25, 3 mesures "
         "sur lgHGU8gqz9U… et <b>41 mesures ici</b> (Urdlvw0SSEc), le "
         "morceau au plus gros empilement du corpus (39–41 observations "
         "par position). Écoute les paires moyenne/pondérée : la version "
         "pondérée est-elle la bonne&nbsp;? C'est le seul endroit du corpus "
         "où ce levier a un effet.</div>",
         "<p class=note>« H » sous une mesure = entropie moyenne de ses "
         "frames (plus c'est HAUT, moins musx est sûr, moins la mesure "
         "pèse dans la version pondérée). Pas de référence extérieure "
         "fiable ici — l'oreille tranche.</p>"]
    for letter, bs in sorted(diffs_by_letter.items()):
        ranges = [o["range"] for o in snap["occurrences"]
                  if o["label"] == letter]
        rep = snap["variants"]["prod"]["report"].get(letter, {})
        B.append(f"<h2>{stem} — lettre {letter} (P={rep.get('period')}, "
                 f"n_obs {rep.get('n_obs')}) — {len(bs)} mesures changent"
                 f"</h2>")
        _occ_rows(B, snap, letter, ranges, highlight=set(bs), heat=heat)
        B.append("<div class=occ>consensus : moyenne (prod) vs pondérée "
                 "entropie — sur les mesures qui changent</div>")
        B.append("<div class=row>")
        src = snap["audio"]
        for b in bs[:24]:
            pb = snap["variants"]["prod"]["bars"][b]
            eb = snap["variants"]["entropy"]["bars"][b]
            B.append(chip(f"m{b} moy " + bar_txt(pb), bar_conf(pb), src,
                          grid[b], grid[b + 1]))
            B.append(chip(f"m{b} pond " + bar_txt(eb), bar_conf(eb), src,
                          grid[b], grid[b + 1], "tpl"))
        if len(bs) > 24:
            B.append(f"<div class=note>… et {len(bs) - 24} autres mesures "
                     "du même motif.</div>")
        B.append("</div>")
    out = LIVE_REPORTS / "occmerge_levier4.html"
    out.write_text(page_shell("Levier 4 — pondération par entropie",
                              "".join(B)))
    print(f"→ {out}")


# ── labo bi-mesures : 4 agrégations, à juger à l'oreille ────────────────────

def _events_to_bars(evts, dur, to_chord_fn):
    """Répartit les événements décodés d'une bi-mesure en 2 textes de mesure.

    Le décodage est périodique (template pavé ×3) : une mesure sans départ
    d'accord TIENT le dernier accord du cycle — affiché entre parenthèses,
    comme les reports de l'app."""
    half = dur / 2
    out = [[], []]
    for e in evts:
        ch = to_chord_fn(e["label"])
        txt = "N" if ch is None else chord_txt({**ch, "nc": False})
        out[0 if e["t0"] < half - 1e-6 else 1].append((txt, e["c"]))
    if evts:
        lch = to_chord_fn(evts[-1]["label"])
        last = "N" if lch is None else chord_txt({**lch, "nc": False})
        if not out[0]:
            out[0].append((f"({last})", None))
        if not out[1]:
            in_first = [t for t, _ in out[0] if not t.startswith("(")]
            hold = in_first[-1] if in_first else last
            out[1].append((f"({hold})", None))
    return out


def bibar_lab(stem):
    sys.path.insert(0, str(REPO))
    from harmonia_min.labels import to_chord
    d = json.loads((SNAP / f"lab_{stem}.json").read_text())
    title, src, bpb = d["title"], d["audio"], d["bpb"]
    B = [f"<h1>Bi-mesures superposées — {html.escape(title)}</h1>",
         "<div class=lede><b>La question</b> : quand la même bi-mesure (2 "
         "mesures) revient N fois, où faut-il additionner pour que musx "
         "prédise mieux&nbsp;? Quatre réponses possibles, montrées telles "
         "quelles, RIEN n'est scoré : chaque répétition décodée seule · "
         "les <b>audios superposés</b> (écoutables — le déphasage "
         "s'entend) · les <b>CQT moyennés</b> (le spectre que musx mange, "
         "pas de déphasage possible) · les <b>probabilités musx "
         "moyennées</b> (l'aval, ce que fait le repli actuel). Tout passe "
         "par le même décodage ensuite — seule l'agrégation change.</div>",
         f"<p class=note>Groupes = bi-mesures de la grille paire dont la "
         f"similarité (même substrat que la détection de sections) dépasse "
         f"{d['thr']:.2f}, lien complet ; {d['n_clusters_total']} groupes "
         f"trouvés, tous affichés s'il y en a ≤ 8, sinon les 8 plus gros. "
         "Pas de vérité terrain affichée — c'est l'oreille qui tranche. "
         "Le petit chiffre = confiance musx (0–1).</p>",
         "<div class=legend><span class=leg>répétition seule (clique = "
         "l'originale)</span><span class='leg tpl'>agrégé</span></div>"]
    for ci, c in enumerate(d["clusters"]):
        bars = sorted(2 * j for j in c["bibars"])
        B.append(f"<h2>groupe {ci + 1} — {len(c['bibars'])} répétitions "
                 f"(mesures {', '.join(str(b) for b in bars[:8])}"
                 f"{'…' if len(bars) > 8 else ''})</h2>")
        # chaque répétition seule
        for i, (t0, t1) in enumerate(c["spans"]):
            pair = _events_to_bars(c["solo"][i], c["dur"], to_chord)
            B.append(f"<div class=occ>mesures {bars[i]}–{bars[i] + 1}"
                     f"</div><div class=row>")
            for half in (0, 1):
                lbl = " ".join(t for t, _ in pair[half]) or "·"
                cf = min((cc for _, cc in pair[half] if cc is not None),
                         default=None)
                B.append(chip(lbl, f"{cf:.2f}" if cf else "", src,
                              t0 + half * (t1 - t0) / 2,
                              t0 + (half + 1) * (t1 - t0) / 2))
            B.append("</div>")
        # les trois agrégats
        rows = [("audios superposés", "audio",
                 f"/reports/{c['wav']}", 0.0, c["dur"]),
                ("CQT moyennés", "cqt", None, 0, 0),
                ("probabilités moyennées", "probs", None, 0, 0)]
        for name, key, wsrc, w0, w1 in rows:
            pair = _events_to_bars(c[key], c["dur"], to_chord)
            B.append(f"<div class=occ>{name}</div><div class=row>")
            if wsrc:
                B.append(chip("▶ écouter le mix", "", wsrc, w0, w1, "tpl"))
            for half in (0, 1):
                lbl = " ".join(t for t, _ in pair[half]) or "·"
                cf = min((cc for _, cc in pair[half] if cc is not None),
                         default=None)
                B.append(chip(lbl, f"{cf:.2f}" if cf else "",
                              wsrc, w0 + half * c["dur"] / 2,
                              w0 + (half + 1) * c["dur"] / 2, "tpl")
                         if wsrc else
                         chip(lbl, f"{cf:.2f}" if cf else "", None, 0, 0,
                              "tpl"))
            B.append("</div>")
    out = LIVE_REPORTS / f"bibar_lab_{stem}.html"
    out.write_text(page_shell(f"Bi-mesures superposées — {title}",
                              "".join(B)))
    print(f"→ {out}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "levier1"
    if which == "bibar_lab":
        bibar_lab(sys.argv[2])
    else:
        {"levier1": levier1, "levier2": levier2, "levier3": levier3,
         "levier4": levier4}[which]()
