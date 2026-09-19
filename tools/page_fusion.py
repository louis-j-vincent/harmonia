"""Ce que la fusion de lettres fait aux sections — morceau par morceau.

Louis, 2026-09-19 : « ok mets moi merge on et montres moi ce que donnes les
sections avec ».

CE QUE LA PAGE COMPARE, et c'est le point délicat : deux RÉ-ANALYSES du même
morceau par le même code, l'une avec `SETTINGS.merge_letters` éteint, l'autre
allumé. Pas la version en cache — un chart cuit il y a des semaines porte un
découpage qu'un autre détecteur a produit, et le comparer à une analyse
fraîche mélangerait deux changements. C'est l'erreur que j'ai faite la veille
et que le contrôle a rattrapée (voir known_issues, « fusion de lettres »).

Chaque lettre est JOUABLE : la fusion se juge à l'oreille, pas au décompte.
Les morceaux sont classés par ce qu'ils PERDENT en lettres, les plus gros
mouvements d'abord — c'est là que sont les bonnes et les mauvaises surprises.

    python -m tools.page_fusion       # → docs/plots/fusion_lettres.html
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from harmonia.folding import (MERGE_LETTERS_ACCORDS, _sig_repli, concordance,
                              tete_commune)
from harmonia.settings import SETTINGS

SANS = SETTINGS.repo / "state" / "cache" / "_sans_fusion"
AVEC = SETTINGS.repo / "state" / "cache" / "_avec_fusion"
SORTIE = SETTINGS.repo / "docs" / "plots" / "fusion_lettres.html"
FLAT = ["C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B"]
SHARP = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
FLAT_MAJ = {0, 1, 3, 5, 6, 8, 10}
TOK = {"": "", "6": "6", "^7": "maj7", "^": "maj7", "7": "7", "-": "m", "-7": "m7",
       "-^7": "mMaj7", "-6": "m6", "o": "dim", "o7": "dim7", "-7b5": "m7♭5",
       "h7": "ø7", "h": "ø", "9": "9", "-9": "m9", "^9": "maj9", "13": "13",
       "sus": "sus", "7sus": "7sus", "sus4": "sus4", "+": "+", "69": "6/9"}


def notes_du_ton(cle):
    t = int((cle or {}).get("tonic") or 0)
    maj = (t + 3) % 12 if (cle or {}).get("mode") == "minor" else t % 12
    return FLAT if maj in FLAT_MAJ else SHARP


def accord(c, notes):
    if c.get("nc"):
        return "N.C."
    q = c.get("q") or ""
    s = notes[c["root"] % 12] + TOK.get(q, q)
    b = c.get("bass")
    if b is not None and b >= 0 and b % 12 != c["root"] % 12:
        s += "/" + notes[b % 12]
    return s


def lignes(m):
    notes = notes_du_ton(m.get("key") or {})
    out = []
    for s in m.get("sections") or []:
        out.append({
            "label": s["label"], "reps": s.get("reps") or 1,
            "n": len(s.get("bars") or []),
            "acc": " | ".join(" ".join(accord(c, notes) for c in bar) or "%"
                              for bar in (s.get("bars") or [])),
            "spans": [list(x) for x in (s.get("spans") or [])],
        })
    return out


def vraies(sections):
    return [s for s in sections if str(s["label"]).lower() not in ("intro", "outro")]


def suspectes(m_sans):
    """Les paires de lettres que la fusion a réunies alors que leurs accords
    concordent MAL — c'est là qu'il faut écouter en premier.

    On les cherche dans la version SANS fusion (les lettres y sont encore
    distinctes) : toute paire dont la concordance est basse et qui pourtant
    finit ensemble n'a été réunie que par la règle de TÊTE."""
    secs = vraies(m_sans.get("sections") or [])
    s = {}
    for x in secs:
        s[x["label"]] = [_sig_repli(b) for b in (x.get("bars") or [])]
    out = []
    noms = list(s)
    for i, a in enumerate(noms):
        for b in noms[i + 1:]:
            c = concordance(s[a], s[b])
            t = tete_commune(s[a], s[b])
            if c < MERGE_LETTERS_ACCORDS and t >= 3:
                out.append((a, b, c, t))
    return out


def e(x):
    return html.escape(str(x))


def bloc(titre, lg, audio, cls):
    rows = "".join(
        f'<tr><td class="l">{e(x["label"])}</td><td class="n">×{x["reps"]}</td>'
        f'<td class="n">{x["n"]}&nbsp;mes.</td>'
        f'<td class="ac">{e(x["acc"][:96])}</td><td class="pl">'
        + "".join(f'<button class="p" onclick="joue(\'{audio}\','
                  f'{a:.2f},{b:.2f},this)">{i + 1}</button>'
                  for i, (a, b) in enumerate(x["spans"][:8]))
        + "</td></tr>" for x in lg)
    return (f'<div class="col {cls}"><h4>{titre}</h4>'
            f'<div class="tw"><table class="sec">{rows}</table></div></div>')


def carte(stem):
    a = json.loads((SANS / f"{stem}.json").read_text())
    b = json.loads((AVEC / f"{stem}.json").read_text())
    la, lb = len(vraies(a["sections"])), len(vraies(b["sections"]))
    audio = (b.get("audio_url") or "").lstrip("/")
    sus = suspectes(a)
    note = ""
    if sus:
        note = ('<p class="sus"><b>À écouter en premier :</b> '
                + ", ".join(f"{e(x)}/{e(y)} (accords {c:.2f}, {t} mesures de tête)"
                            for x, y, c, t in sus[:4])
                + " — réunies par la tête, pas par la ressemblance.</p>")
    delta = (f'<span class="d gain">−{la - lb} lettre{"s" if la - lb > 1 else ""}</span>'
             if lb < la else '<span class="d rien">inchangé</span>')
    return f"""
<section class="carte" data-delta="{la - lb}">
  <h2>{e(b.get('title') or stem)} {delta}</h2>
  {note}
  <div class="deux">
    {bloc(f"sans fusion — {la} lettres", lignes(a), audio, "av")}
    {bloc(f"avec fusion — {lb} lettres", lignes(b), audio, "ap")}
  </div>
</section>"""


PAGE = """<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Fusion de lettres : ce que ça donne</title>
<style>
 :root{--pap:#f7f3e9;--ink:#1c1c1c;--rule:#b9b09a;--faint:#8a8371;--line:#e5dcc6;--acc:#8a2b2b;--bg:#e7e0d0;--ok:#1f8a5b}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--ink);
   font:15px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif}
 .wrap{max-width:1180px;margin:0 auto;padding:18px 16px 60px}
 h1{font:600 italic 26px Georgia,serif;margin:0 0 8px}
 .chapo{color:var(--faint);max-width:76ch;margin:0 0 14px}
 .cle{background:var(--pap);border-left:3px solid var(--acc);padding:12px 14px;
   border-radius:0 8px 8px 0;margin:0 0 18px;max-width:76ch}
 .bilan{display:flex;gap:10px;flex-wrap:wrap;margin:0 0 18px}
 .kpi{background:var(--pap);border:1px solid var(--line);border-radius:10px;padding:10px 14px;min-width:150px}
 .kpi b{display:block;font:700 22px Georgia,serif}
 .kpi span{font-size:12px;color:var(--faint)}
 .carte{background:var(--pap);border:1px solid var(--line);border-radius:12px;padding:14px;margin:0 0 16px}
 .carte h2{font:600 italic 19px Georgia,serif;margin:0 0 8px}
 .d{font:600 12px -apple-system,system-ui,sans-serif;padding:2px 8px;border-radius:999px;margin-left:8px;vertical-align:2px}
 .d.gain{background:rgba(31,138,91,.14);color:var(--ok)} .d.rien{background:#efe9db;color:var(--faint)}
 .sus{margin:0 0 8px;padding:7px 10px;background:rgba(138,43,43,.09);border-radius:7px;font-size:12.5px}
 .deux{display:flex;gap:14px;flex-wrap:wrap}
 .col{flex:1 1 380px;min-width:0}
 .col h4{font:600 11.5px inherit;margin:0 0 6px;color:var(--faint);
   text-transform:uppercase;letter-spacing:.04em}
 .col.ap h4{color:var(--ok)}
 .tw{overflow-x:auto}
 table.sec{border-collapse:collapse;width:100%;font-size:12.5px}
 table.sec td{padding:4px 7px 4px 0;border-bottom:1px solid var(--line);vertical-align:top}
 td.l{font:700 13px Georgia,serif;white-space:nowrap}
 td.n{color:var(--faint);white-space:nowrap;font-variant-numeric:tabular-nums}
 td.ac{font:italic 12.5px Georgia,serif}
 td.pl{white-space:nowrap;text-align:right}
 .p{min-width:24px;min-height:24px;margin-left:2px;border:1px solid var(--line);
   background:#fffdf6;border-radius:5px;font:600 10.5px inherit;cursor:pointer;color:var(--ink)}
 .p.joue{background:var(--acc);color:var(--pap);border-color:var(--acc)}
 @media(max-width:620px){ td.ac{display:none} }
</style></head><body><div class="wrap">
<h1>Fusion de lettres : ce que ça donne</h1>
<p class="chapo">Deux ré-analyses du MÊME morceau par le MÊME code, l'une avec la
fusion éteinte, l'autre allumée. Pas la version en cache : un chart cuit il y a
des semaines porte un découpage qu'un autre détecteur a produit, et le comparer
à une analyse fraîche mélangerait deux changements.</p>
<div class="cle">__CLE__</div>
<div class="bilan">__BILAN__</div>
__CARTES__
</div>
<script>
const cache={}; let cour=null;
async function el(src){
  if(cache[src]) return cache[src];
  const a=new Audio();
  try{ const r=await fetch('/'+src); a.src=URL.createObjectURL(await r.blob()); }
  catch(e){ a.src='/'+src; }
  a.preload='auto'; cache[src]=a; return a;
}
function coupe(){
  if(!cour) return;
  try{ cour.a.pause(); }catch(e){}
  cour.a.removeEventListener('timeupdate',cour.stop);
  cour.b.classList.remove('joue'); cour=null;
}
async function joue(src,t0,t1,btn){
  const re=cour&&cour.b===btn; coupe(); if(re) return;
  const a=await el(src);
  const stop=()=>{ if(a.currentTime>=t1) coupe(); };
  a.addEventListener('timeupdate',stop);
  try{ a.currentTime=t0; }catch(e){}
  btn.classList.add('joue'); cour={a,b:btn,stop};
  a.play().catch(()=>{ btn.classList.remove('joue'); cour=null; });
}
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=SORTIE)
    ap.add_argument("--max", type=int, default=40, help="cartes affichées")
    a = ap.parse_args()
    paires = sorted({p.stem for p in SANS.glob("*.json")}
                    & {p.stem for p in AVEC.glob("*.json")})
    infos = []
    for stem in paires:
        try:
            x = json.loads((SANS / f"{stem}.json").read_text())
            y = json.loads((AVEC / f"{stem}.json").read_text())
        except Exception:
            continue
        infos.append((len(vraies(x["sections"])) - len(vraies(y["sections"])),
                      len(vraies(x["sections"])), len(vraies(y["sections"])), stem))
    infos.sort(key=lambda t: (-t[0], t[3]))
    bouge = [t for t in infos if t[0] > 0]
    av = sum(t[1] for t in infos)
    ap_ = sum(t[2] for t in infos)
    cle = (f"<b>{len(bouge)} morceaux sur {len(infos)}</b> voient des lettres "
           f"fusionner. Au total <b>{av} → {ap_}</b> lettres "
           f"({100 * (av - ap_) // max(av, 1)} % de moins). Les cartes sont "
           f"classées par ce qu'elles perdent : les plus gros mouvements "
           f"d'abord, c'est là que sont les surprises — bonnes et mauvaises. "
           f"Chaque lettre se joue.")
    bilan = "".join(
        f'<div class="kpi"><b>{v}</b><span>{k}</span></div>' for k, v in (
            ("morceaux ré-analysés", len(infos)),
            ("morceaux touchés", len(bouge)),
            ("lettres avant", av),
            ("lettres après", ap_),
        ))
    cartes = "".join(carte(t[3]) for t in infos[:a.max])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(PAGE.replace("__CLE__", cle).replace("__BILAN__", bilan)
                     .replace("__CARTES__", cartes))
    print(f"→ {a.out}")
    print(f"   http://100.89.209.63:7772/plots/{a.out.name}")
    print(f"   {len(bouge)}/{len(infos)} touchés, {av} → {ap_} lettres")


if __name__ == "__main__":
    main()
