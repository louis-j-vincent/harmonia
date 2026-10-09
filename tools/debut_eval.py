"""Le score de la règle du début — au temps près ET à la mesure près.

    python -m tools.debut_eval                 # les deux jeux, avant / après
    python -m tools.debut_eval --bibliotheque  # + les morceaux qui bougeraient

Deux jeux de vérité, tous deux de Louis :
* `state/human/debuts.json` — 42 débuts tranchés à l'oreille le 2026-09-17
  (41 avec audio : 191P7nIeECo n'en a pas) ;
* `state/human/marks/<stem>.json` ABSENTS de ce fichier — les marques
  « mesure 1 » posées depuis (dont La Jupe En Laine, 2026-10-08).

Deux scores :
* **à la mesure** — le protocole de `tools.debut_rates` : juste si la ligne
  de `barGrid` la plus proche de la prédiction est celle de la vérité (ou de
  `aussi_acceptable`). Une demi-mesure d'erreur s'y arrondit.
* **au temps** — juste si la battue retenue est celle de la vérité. C'est ce
  que fait vraiment le pipeline (`pipeline.py`, « LE VRAI DÉBUT DU
  MORCEAU ») : arbitrage « confirmé bon » → la mesure 1 du traqueur ; sinon
  la détection, si elle tombe à plus d'une demi-mesure après la 1re battue,
  et la battue la plus proche devient la mesure 1. Un temps d'erreur (une
  levée) n'existe que dans ce score — c'est le défaut de La Jupe En Laine.

« avant » = `debut_du_morceau` sans harmonie ; « après » = avec
(`cale_sur_harmonie`). Rien ici n'écrit dans `state/`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from harmonia import beats as _beats
from harmonia import musx as _musx
from harmonia.debut import arbitrage, debut_du_morceau
from harmonia.settings import SETTINGS


def _near(arr, t) -> int:
    return int(np.abs(np.asarray(arr, dtype=float) - float(t)).argmin())


def verites() -> list[tuple[str, dict, str]]:
    """[(stem, {t, aussi_acceptable?}, jeu)] — debuts.json puis les marques."""
    d = json.loads((SETTINGS.repo / "state/human/debuts.json")
                   .read_text(encoding="utf-8"))["debuts"]
    out = [(s, v, "debuts") for s, v in sorted(d.items())]
    for f in sorted((SETTINGS.repo / "state/human/marks").glob("*.json")):
        b = json.loads(f.read_text(encoding="utf-8")).get("bar1")
        if f.stem not in d and b is not None:
            out.append((f.stem, {"t": float(b)}, "marques"))
    return out


def mesure_un(stem: str, audio: Path, bt, bpb: int, probs, harmonie: bool) -> int:
    """L'indice de battue que le pipeline prendrait pour la mesure 1."""
    if arbitrage(stem) == "confirmé bon":
        db = _beats.track(audio)["downbeats"]
        return _near(bt, db[0]) if db else 0
    kw = dict(probs=probs, beat_times=bt, bpb=bpb) if harmonie else {}
    d = debut_du_morceau(audio, bt, probs_basse=probs[1], **kw)
    t_dec = d.get("t_basse", d["t"])        # même test que pipeline.py
    if t_dec is not None and t_dec - bt[0] > 0.5 * bpb * float(np.median(np.diff(bt))):
        return _near(bt, d["t"])
    db = _beats.track(audio)["downbeats"]
    return _near(bt, db[0]) if db else 0


def _chart(stem):
    cp = SETTINGS.charts_dir / f"min_{stem}.json"
    audio = SETTINGS.audio_dir / f"{stem}.m4a"
    if not cp.exists() or not audio.exists():
        return None, None
    return json.loads(cp.read_text(encoding="utf-8")), audio


def evalue() -> list[dict]:
    rows = []
    for stem, v, jeu in verites():
        c, audio = _chart(stem)
        if c is None:
            continue
        bt = [float(x) for x in c["beatTimes"]]
        g = [float(x) for x in c["barGrid"]]
        bpb = int(c.get("bpb") or 4)
        probs = _musx.frame_posteriors(audio)
        vrais_t = [v["t"]] + ([v["aussi_acceptable"]]
                              if v.get("aussi_acceptable") is not None else [])
        row = {"stem": stem, "jeu": jeu, "titre": c.get("title") or stem}
        for nom, h in (("avant", False), ("apres", True)):
            k = mesure_un(stem, audio, bt, bpb, probs, h)
            row[nom] = k
            row[nom + "_temps"] = k in {_near(bt, t) for t in vrais_t}
            row[nom + "_mesure"] = _near(g, bt[k]) in {_near(g, t) for t in vrais_t}
        row["vrai"] = _near(bt, v["t"])
        rows.append(row)
    return rows


def bibliotheque() -> list[dict]:
    """Les morceaux SANS marque ni arbitrage dont la mesure 1 bougerait."""
    out = []
    for cp in sorted(SETTINGS.charts_dir.glob("min_*.json")):
        stem = cp.stem[4:]
        if (SETTINGS.repo / "state/human/marks" / f"{stem}.json").exists():
            continue
        if arbitrage(stem) is not None:
            continue
        c, audio = _chart(stem)
        if c is None or len(c.get("beatTimes") or []) < 8:
            continue
        bt = [float(x) for x in c["beatTimes"]]
        bpb = int(c.get("bpb") or 4)
        try:
            probs = _musx.frame_posteriors(audio)
            a = mesure_un(stem, audio, bt, bpb, probs, False)
            b = mesure_un(stem, audio, bt, bpb, probs, True)
        except Exception as exc:                          # noqa: BLE001
            print(f"  ! {stem}: {exc}")
            continue
        if a != b:
            out.append({"stem": stem, "titre": c.get("title") or stem,
                        "avant": round(bt[a], 2), "apres": round(bt[b], 2),
                        "temps": b - a})
    return out


def page(rows: list[dict], out: Path) -> None:
    """Une bande de battues par morceau que la règle déplace : les accords
    entendus (décodage sans phase), la basse, l'ancienne et la nouvelle
    mesure 1, la vérité de Louis — et de quoi écouter depuis chacune."""
    import html as _h
    cartes = []
    for x in rows:
        if x["avant"] == x["apres"]:
            continue
        c, audio = _chart(x["stem"])
        bt = [float(t) for t in c["beatTimes"]]
        segs = _musx.redecode(bt, _musx.frame_posteriors(audio), downbeat_times=None)
        k0 = max(0, min(x["avant"], x["apres"], x["vrai"]) - 3)
        k1 = min(len(bt) - 1, k0 + 16)
        W = 34
        svg = [f'<div style="overflow-x:auto;position:relative"><svg width="{W * (k1 - k0) + 20}" height="96">']
        for k in range(k0, k1):
            xx = 10 + W * (k - k0)
            svg.append(f'<line x1="{xx}" y1="30" x2="{xx}" y2="62" stroke="#bbb"/>'
                       f'<text x="{xx + 2}" y="74" font-size="9" fill="#999">{k - x["vrai"]:+d}</text>')
        for t0, _t1, lab in segs:
            k = _near(bt, t0)
            if k0 <= k < k1 and lab != "N":
                xx = 10 + W * (k - k0)
                svg.append(f'<rect x="{xx}" y="34" width="3" height="24" fill="#b06a1f"/>'
                           f'<text x="{xx + 4}" y="50" font-size="10">{_h.escape(lab)}</text>')
        for k, coul, nom, y in ((x["vrai"], "#1f7a6b", "Louis", 12),
                                (x["avant"], "#b33", "avant", 24),
                                (x["apres"], "#2a5db0", "après", 88)):
            if k0 <= k < k1:
                xx = 10 + W * (k - k0)
                svg.append(f'<line x1="{xx}" y1="{min(y, 30) - 2}" x2="{xx}" y2="66" '
                           f'stroke="{coul}" stroke-width="2.5"/>'
                           f'<text x="{xx + 3}" y="{y}" font-size="11" font-weight="700" '
                           f'fill="{coul}">{nom}</text>')
        svg.append('<div class="tete" style="position:absolute;top:0;left:0;width:2px;'
                   'height:96px;background:#222;opacity:0;pointer-events:none;'
                   'transition:transform .22s linear"></div></div>')
        ok = lambda b: "✓" if b else "✗"
        dep = lambda k: bt[max(0, k - 2)]          # deux temps d'élan
        cartes.append(
            f'<div class="c" data-stem="{x["stem"]}" data-w="{W}" data-k0="{k0}" '
            f'data-bt="{",".join(f"{t:.3f}" for t in bt[k0:k1 + 1])}">'
            f'<b>{_h.escape(x["titre"])}</b> <span class="n">{x["stem"]}</span>'
            f'<div class="n">au temps : avant {ok(x["avant_temps"])} · après {ok(x["apres_temps"])}'
            f' — les chiffres sous la bande comptent les temps depuis la mesure 1 de Louis</div>'
            f'{"".join(svg)}<div>'
            f'<button onclick="joue(this,{dep(x["avant"]):.3f})">▶ depuis l’ancienne mesure 1</button> '
            f'<button onclick="joue(this,{dep(x["apres"]):.3f})">▶ depuis la nouvelle</button>'
            f'</div></div>')
    out.write_text(f"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mesure 1 : la levée</title>
<style>body{{font:15px system-ui;max-width:880px;margin:auto;padding:12px;background:#faf6ec}}
.c{{background:#fffdf7;border:1px solid #ddd3b8;border-radius:10px;padding:10px;margin:14px 0}}
.n{{color:#8a8371;font-size:12.5px}} button{{padding:8px 10px;margin-top:6px}}</style>
<h1>Mesure 1 : une basse en levée ne fait plus la mesure 1</h1>
<p>La règle : la basse dit <i>quelle</i> mesure, les changements d'accords (traits
orange) disent sur quel <i>temps</i> elle tombe. Si la basse entre sur un temps
faible, c'est une levée et la mesure 1 est le temps fort suivant.
Rouge = mesure 1 d'avant, bleu = d'après, vert = ta marque.</p>
{''.join(cartes)}
<script>
// Tout le son est chargé AVANT le premier clic (un blob par morceau) : « ▶ »
// démarre sur-le-champ. La tête de lecture suit `timeupdate` (pas de boucle
// rAF : elle empêche le moteur audio de WebKit de démarrer sur iPhone).
const AUD={{}}, PRET={{}};
function charge(stem){{
  return PRET[stem]||(PRET[stem]=fetch('../audio/'+stem+'.m4a').then(r=>r.blob()).then(b=>new Promise(ok=>{{
    const a=new Audio(); a.preload='auto'; a.src=URL.createObjectURL(b);
    a.addEventListener('loadedmetadata',()=>{{AUD[stem]=a; a.addEventListener('timeupdate',()=>{{ if(stem===EN_COURS) tete(stem,a); }}); ok(a);}},{{once:true}});
    a.addEventListener('ended',()=>cache(stem)); a.load();}})));
}}
function carte(stem){{return document.querySelector('.c[data-stem="'+stem+'"]');}}
function cache(stem){{const t=carte(stem).querySelector('.tete'); t.style.opacity=0;}}
function tete(stem,a){{
  const c=carte(stem), bt=c.dataset.bt.split(',').map(Number), W=+c.dataset.w, t=a.currentTime;
  let k=0; while(k<bt.length-2&&bt[k+1]<=t) k++;
  const f=Math.max(0,Math.min(1,(t-bt[k])/(bt[k+1]-bt[k]||1)));
  const el=c.querySelector('.tete'); el.style.opacity=(t<bt[0]||t>bt[bt.length-1])?0:1;
  el.style.transform='translateX('+(10+W*(k+f))+'px)';
}}
let EN_COURS=null;
async function joue(btn,t){{
  const stem=btn.closest('.c').dataset.stem;
  const a=await charge(stem), avant=EN_COURS; EN_COURS=stem;
  if(avant&&avant!==stem&&AUD[avant]){{AUD[avant].pause(); cache(avant);}}
  a.currentTime=Math.max(0,t); tete(stem,a); a.play();
}}
document.querySelectorAll('.c[data-stem]').forEach(c=>charge(c.dataset.stem));
</script>""", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bibliotheque", action="store_true")
    ap.add_argument("--page", type=Path, default=None,
                    help="écrire la page d'écoute des morceaux déplacés")
    a = ap.parse_args(argv)
    rows = evalue()
    if a.page:
        page(rows, a.page)
        print(f"page : {a.page}")
    for jeu in ("debuts", "marques"):
        r = [x for x in rows if x["jeu"] == jeu]
        for sc in ("temps", "mesure"):
            av = sum(x["avant_" + sc] for x in r)
            ap_ = sum(x["apres_" + sc] for x in r)
            gagnes = [x["stem"] for x in r if x["apres_" + sc] and not x["avant_" + sc]]
            perdus = [x["stem"] for x in r if x["avant_" + sc] and not x["apres_" + sc]]
            print(f"{jeu:8s} au {sc:7s} avant {av}/{len(r)}  après {ap_}/{len(r)}"
                  f"  gagnés {gagnes}  perdus {perdus}")
    print("\nencore faux au temps (après) :")
    for x in rows:
        if not x["apres_temps"]:
            print(f"  {x['stem'][:40]:40s} {x['apres'] - x['vrai']:+d} temps"
                  f"  ({'juste' if x['apres_mesure'] else 'faux'} à la mesure)")
    if a.bibliotheque:
        print("\nbibliothèque, sans marque ni arbitrage — la mesure 1 bougerait :")
        for x in bibliotheque():
            print(f"  {x['stem']:40s} {x['titre'][:30]:30s} {x['avant']:7.2f}s → "
                  f"{x['apres']:7.2f}s ({x['temps']:+d} temps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
