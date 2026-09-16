"""La page « faut-il changer de source de basse » : avant / après, et l'aveu.

Louis, 2026-09-16 : « il faut utiliser la tete de basse de musx qui est parfaite
enfaite », puis « oui re derive le plancher sur mes verdicts et montres avant
apres stp ».

Ce que la page montre, dans cet ordre, parce que c'est l'ordre où je me suis
trompé :

  1. Mon premier chiffre comparait les SIGNAUX (l'argmax brut de chaque source)
     et non les RÈGLES. Refait correctement : l'ancienne règle fait 12/12 sur
     ses verdicts, la nouvelle 11/12.
  2. Mais 12/12 n'est pas un score : `decide_bass` a été FABRIQUÉE sur ces
     douze cas. C'est de l'apprentissage, pas de la validation. Le 11/12 de
     musx, lui, est hors échantillon — il n'a jamais vu ces verdicts.
  3. Donc rien ne tranche sur cette preuve-là, et c'est l'oreille qui doit
     trancher les 76 mesures que le changement déplace.

    .venv/bin/python -m tools.page_basse_source
"""
from __future__ import annotations

import html
import json

from harmonia.settings import SETTINGS

DATA = SETTINGS.repo / "scratchpad" / "avant_apres_basse.json"
PLANCHER = SETTINGS.repo / "scratchpad" / "plancher_musx.json"
OUT = SETTINGS.reports_dir / "basse_source" / "avant_apres.html"


def page(D: dict, P: dict) -> str:
    e = html.escape
    L = D["lignes"]
    obs = sorted(P["obs"], key=lambda o: o["pRoot"])

    sep = "".join(
        f"<tr><td class='m'>{e(o['id'])}</td><td class='m'>{e(o['vrai'])}</td>"
        f"<td><span class='b b-{'warn' if o['slashVrai'] else 'mark'}'>"
        f"{'slash' if o['slashVrai'] else 'accord nu'}</span></td>"
        f"<td class='m num'>{o['pRoot']:.1f} %</td></tr>" for o in obs)

    cartes = "".join(f"""
<div class="cas {'ko' if not x['apresOk'] else ''}">
  <div class="h"><b>{e(x['vrai'])}</b>
    <span class="s">{e(x['song'])} &middot; mes. {x['bar']} &middot; {e(x['id'])}</span></div>
  <div class="cmp">
    <span class="c {'ok' if x['avantOk'] else 'no'}">avant&nbsp;: {e(x['avant'])}</span>
    <span class="c {'ok' if x['apresOk'] else 'no'}">après&nbsp;: {e(x['apres'])}</span>
  </div>
  <div class="d">musx&nbsp;: fondamentale {x['pRoot']:.0f}&nbsp;%, plus probable
    {e(x['argmax'])} à {x['pTop']:.0f}&nbsp;%</div>
  <div class="acts">
    <button class="pl" data-a="{e(x['audio'])}" data-t0="{x['t0']}" data-t1="{x['t1']}"
      type="button">&#9658; l'accord</button>
    <button class="pl" data-a="{e(x['audio'])}" data-t0="{max(0, x['t0'] - 1.6):.3f}"
      data-t1="{x['t1']}" type="button">&#9658; en contexte</button>
  </div>
</div>""" for x in L)

    return f"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Changer de Source de Basse</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=IBM+Plex+Mono:wght@400;500;600&family=Public+Sans:wght@400;500&display=swap">
<style>
 :root{{--bg:#F5F1E7;--surface:#fff;--surface-2:#FBF7EE;--rule:#E3DAC4;--rule-strong:#C9BE9F;
  --ink:#211C14;--ink-dim:#6F6555;--ink-faint:#A79C86;--accent:#B4632A;--accent-soft:#EFDFC5;
  --accent-ink:#5A3315;--mark:#1F6E5C;--mark-soft:#DCEDE7;--warn:#B23A33;--warn-soft:#F6E0DD;}}
 @media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
  --bg:#14120D;--surface:#1B1812;--surface-2:#211D16;--rule:#332D22;--rule-strong:#4A4130;
  --ink:#ECE4D3;--ink-dim:#A89D89;--ink-faint:#726858;--accent:#DE9251;--accent-soft:#3B2C1B;
  --accent-ink:#F3D2AC;--mark:#59B39C;--mark-soft:#1B302B;--warn:#E1837B;--warn-soft:#3A2220;}}}}
 *{{box-sizing:border-box}}
 body{{background:var(--bg);color:var(--ink);margin:0 auto;max-width:880px;
  font-family:'Public Sans',-apple-system,sans-serif;padding:22px 16px 80px;font-size:14.5px;line-height:1.55}}
 h1{{font-family:'Fraunces',Georgia,serif;font-size:26px;font-weight:600;margin:0;text-wrap:balance}}
 h2{{font-family:'Fraunces',Georgia,serif;font-size:18px;margin:30px 0 6px}}
 .eyebrow{{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:.09em;text-transform:uppercase;
  color:var(--accent-ink);background:var(--accent-soft);display:inline-block;padding:2px 7px;
  border-radius:3px;margin-bottom:8px}}
 .sub{{color:var(--ink-dim);font-size:13.5px;margin:5px 0 0}}
 .card{{background:var(--surface);border:1px solid var(--rule);border-radius:11px;padding:14px 16px;margin:14px 0}}
 .card.aveu{{border-left:3px solid var(--warn)}}
 .card p{{margin:9px 0;font-size:13.5px;color:var(--ink-dim)}}
 .card b{{color:var(--ink)}}
 code{{font-family:'IBM Plex Mono',monospace;font-size:12.5px;background:var(--surface-2);
  border:1px solid var(--rule);border-radius:4px;padding:1px 5px}}
 a{{color:var(--accent-ink)}}
 table{{border-collapse:collapse;width:100%;font-size:12.5px;margin:6px 0}}
 td,th{{padding:5px 7px;border-bottom:1px solid var(--rule);text-align:left}}
 td:first-child,th:first-child{{padding-left:0}}
 .m{{font-family:'IBM Plex Mono',monospace}} .num{{text-align:right;font-variant-numeric:tabular-nums}}
 .b{{font-family:'IBM Plex Mono',monospace;font-size:10px;padding:2px 6px;border-radius:4px}}
 .b-mark{{background:var(--mark-soft);color:var(--mark)}}
 .b-warn{{background:var(--warn-soft);color:var(--warn)}}
 .score{{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0 2px;font-family:'IBM Plex Mono',monospace}}
 .sc{{background:var(--surface-2);border:1px solid var(--rule);border-radius:8px;padding:8px 12px;font-size:13px}}
 .sc b{{font-size:17px}}
 .cas{{border:1px solid var(--rule);border-radius:10px;background:var(--surface);
  padding:11px 13px;margin-bottom:7px}}
 .cas.ko{{border-color:var(--warn);background:var(--warn-soft)}}
 .cas .h{{display:flex;gap:9px;align-items:baseline;flex-wrap:wrap;margin-bottom:6px}}
 .cas .h b{{font-family:'IBM Plex Mono',monospace;font-size:15px}}
 .cas .s{{font-size:11px;color:var(--ink-faint);font-family:'IBM Plex Mono',monospace;margin-left:auto}}
 .cmp{{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:6px}}
 .c{{font-family:'IBM Plex Mono',monospace;font-size:12.5px;border-radius:5px;padding:2px 8px;
  border:1px solid var(--rule);background:var(--surface-2);color:var(--ink-dim)}}
 .c.ok{{border-color:var(--mark);color:var(--mark);background:var(--mark-soft)}}
 .c.no{{border-color:var(--warn);color:var(--warn);background:var(--warn-soft);font-weight:600}}
 .d{{font-size:11.5px;color:var(--ink-faint);font-family:'IBM Plex Mono',monospace;margin-bottom:7px}}
 .acts{{display:flex;gap:6px;flex-wrap:wrap}}
 button{{font-family:'IBM Plex Mono',monospace;font-size:11.5px;border-radius:6px;padding:5px 10px;
  cursor:pointer;border:1px solid var(--rule);background:var(--surface-2);color:var(--ink-dim)}}
 button:hover{{border-color:var(--accent);color:var(--accent-ink);background:var(--accent-soft)}}
 button.playing{{background:var(--accent);border-color:var(--accent);color:var(--surface)}}
 footer{{margin-top:30px;padding-top:12px;border-top:1px solid var(--rule);font-size:11.5px;
  color:var(--ink-faint);font-family:'IBM Plex Mono',monospace;line-height:1.8}}
</style>

<span class="eyebrow">basse &middot; chroma NNLS contre tête musx</span>
<h1>Changer de source de basse&nbsp;?</h1>
<p class="sub">Tu as dit que la tête basse de musx était parfaite. Elle est meilleure
comme signal. Comme règle, elle perd un de tes verdicts.</p>

<div class="card aveu">
  <p><b>D'abord, je me corrige.</b> Mon premier chiffre — « chroma 6/16, musx 11/16 » —
  comparait les <b>signaux bruts</b>&nbsp;: l'accord le plus probable de chaque source,
  pris tel quel. Or la règle que tu as arbitrée ne fait pas ça. Elle cherche la
  fondamentale sur <b>tous les temps</b> de l'accord, et ne retombe sur la lecture du
  temps 1 que si elle ne la trouve pas. Elle compense donc le bruit du signal.</p>
  <p>Refait correctement, règle contre règle&nbsp;:</p>
  <div class="score">
    <span class="sc">ancienne règle (chroma) &nbsp;<b>{D['avantOk']}/{len(L)}</b></span>
    <span class="sc">nouvelle règle (musx) &nbsp;<b>{D['apresOk']}/{len(L)}</b></span>
  </div>
</div>

<div class="card">
  <p><b>Mais 12/12 n'est pas un score.</b> <code>decide_bass</code> a été fabriquée sur
  ces douze cas&nbsp;: c'est de l'apprentissage, pas de la validation. Le 11/12 de musx,
  lui, est <b>hors échantillon</b> — cette tête n'a jamais vu tes verdicts, personne ne
  l'a réglée dessus. Les deux nombres ne sont pas comparables, et <b>rien ne tranche
  sur cette preuve-là</b>.</p>
</div>

<h2>Ce que la re-dérivation a donné</h2>
<div class="card">
  <p>Tu m'as demandé de re-dériver le plancher. Fait — et <b>il ne sert à rien</b>.
  La part de la fondamentale selon musx sépare tes deux classes sans le moindre
  recouvrement&nbsp;:</p>
  <table>
    <tr><th>cas</th><th>vérité</th><th></th><th class="num">p(fondamentale)</th></tr>
    {sep}
  </table>
  <p>Un fossé vide de <b>42&nbsp;% à 80&nbsp;%</b>. Sauf que dans les <b>sept</b> cas sans
  slash, l'accord le plus probable selon musx <b>est déjà la fondamentale</b>&nbsp;: un
  plancher placé n'importe où entre 40 et 80 donne exactement le même résultat que pas
  de plancher du tout. Je l'ai donc retiré plutôt que de garder un nombre qui ne décide
  rien — la leçon de <code>DEFAULT_PENALTY</code>, ce matin.</p>
  <p>Il reste <b>une</b> règle&nbsp;: la basse la plus probable sur la durée de l'accord,
  écrite si elle diffère de la fondamentale et si son intervalle est l'un des cinq que
  tu as entendus. Plus de plancher, plus de fenêtre d'attaque, plus de seconde branche.</p>
</div>

<h2>Tes douze verdicts, avant et après</h2>
<p class="sub" style="margin-bottom:10px">Le seul que la nouvelle règle perd est en rouge.</p>
{cartes}

<h2>Ce que ça déplace dans la bibliothèque</h2>
<div class="card">
  <p><b>76 mesures sur 19 morceaux</b>&nbsp;: 65 slashes retirés, 12 ajoutés, 2 basses
  changées. La nouvelle règle est nettement plus prudente.</p>
  <p>Et elle retire <code>Eb-/G♭</code> sur Ready — précisément le cas que tu avais
  arbitré juste et que musx rate. C'est le risque en une ligne&nbsp;: on gagne en
  simplicité et en prudence, on perd les renversements que musx ne voit pas.</p>
  <p><a href="/reports/basse_musx/avant_apres.html">ouvrir les 19 morceaux, mesure par mesure</a></p>
</div>

<footer>
  ancienne&nbsp;: <code>bass_pc_onset</code> + <code>decide_bass</code>, 2 branches, plancher 30&nbsp;%<br>
  nouvelle&nbsp;: tête basse de musx moyennée sur l'accord, 1 règle, aucun plancher<br>
  vérité terrain&nbsp;: <code>state/human/bass_verdicts.json</code>, 2026-09-15, 12 cas localisables
</footer>

<script>
// fetch + blob : le lecteur de Safari ne bufferise pas les extraits servis en 206.
const cache = {{}};
let audio = null, playing = null, stopAt = null;
async function src(u){{
  if (!cache[u]) cache[u] = fetch(u).then(r => r.blob()).then(b => URL.createObjectURL(b));
  return cache[u];
}}
async function jouer(btn){{
  const u = btn.dataset.a, t0 = parseFloat(btn.dataset.t0), t1 = parseFloat(btn.dataset.t1);
  if (playing) playing.classList.remove('playing');
  if (playing === btn) {{ if (audio) audio.pause(); playing = null; return; }}
  playing = btn; btn.classList.add('playing');
  const url = await src(u);
  if (playing !== btn) return;
  if (!audio || audio.dataset.u !== u) {{
    if (audio) audio.pause();
    audio = new Audio(url); audio.dataset.u = u;
    audio.addEventListener('timeupdate', () => {{
      if (stopAt != null && audio.currentTime >= stopAt) {{
        audio.pause(); if (playing) playing.classList.remove('playing'); playing = null;
      }}
    }});
  }}
  stopAt = t1 + 0.25;
  const go = () => {{ audio.currentTime = Math.max(0, t0); audio.play().catch(()=>{{}}); }};
  if (audio.readyState >= 1) go(); else audio.addEventListener('loadedmetadata', go, {{once:true}});
}}
document.querySelectorAll('button.pl').forEach(b => b.addEventListener('click', () => jouer(b)));
</script>
"""


def main() -> int:
    D = json.loads(DATA.read_text(encoding="utf-8"))
    P = json.loads(PLANCHER.read_text(encoding="utf-8"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page(D, P), encoding="utf-8")
    print(f"→ {OUT}  ({len(D['lignes'])} cas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
