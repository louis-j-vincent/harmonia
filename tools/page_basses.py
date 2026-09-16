"""La page d'arbitrage des basses de slash, avec l'avis des tabs à côté.

Louis, 2026-09-16 : « C'est dur d'arbitrer les diffs comme ça, mais si tu as un
doute pour trancher, vas chercher les guitar tabs ou irealpro tabs s'ils sont
dispos et fais un matching pour comparer, en mettant dans la bonne tona[lité] ».

Le diff à arbitrer ne porte QUE sur des basses de slash : 85 accords sur 16
morceaux dont la racine et la qualité ne bougent pas, seule la basse change.

CE QUE LA TAB PEUT ET NE PEUT PAS DIRE — à lire avant de se réjouir des
chiffres. Une tab de guitare écrit ce que la MAIN DROITE joue ; la cible du
projet, depuis le 2026-07-16, est la basse SONNANTE, c'est-à-dire ce que joue
le bassiste. Les deux coïncident souvent et pas toujours. Donc :

  * « la tab écrit une AUTRE basse » est un vrai désaccord : elle nomme une
    note précise, et ce n'est pas la nôtre ;
  * « la tab met un slash là où on l'enlève » est un vrai soutien à l'ancienne
    version ;
  * « la tab écrit l'accord nu » ne prouve RIEN sur la basse sonnante — c'est
    une convention d'écriture, pas un témoignage. Rangé en bas de page.

Les lignes sont donc triées par ce que la tab apporte VRAIMENT, pas par le
nombre de mesures. L'écoute reste l'arbitre : chaque ligne joue l'attaque de
l'accord dans l'enregistrement.

    .venv/bin/python -m tools.page_basses
"""
from __future__ import annotations

import html
import json
from pathlib import Path

from harmonia.settings import SETTINGS

MATCH = SETTINGS.reports_dir / "tabmatch" / "tabmatch.json"
#: l'index artiste/titre fait à la main — les clés brutes (Oextk-If8HQ) sont
#: illisibles sur une page d'arbitrage.
_META_P = SETTINGS.repo / "state" / "human" / "chart_meta.json"
_META = json.loads(_META_P.read_text(encoding="utf-8")) if _META_P.exists() else {}

#: toutes les orthographes qu'on sait LIRE (l'écriture, elle, suit le ton)
_PCS = {n: i for i, n in enumerate(
    ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"])}
_PCS.update({"C#": 1, "D#": 3, "F#": 6, "G#": 8, "A#": 10,
             "D♭": 1, "E♭": 3, "G♭": 6, "A♭": 8, "B♭": 10,
             "C♯": 1, "D♯": 3, "F♯": 6, "G♯": 8, "A♯": 10})


AFTER = SETTINGS.repo / "state" / "cache" / "golden" / "jumeau_large"
OUT = SETTINGS.reports_dir / "basses" / "arbitrage.html"


def joli_titre(key: str, brut: str) -> str:
    m = _META.get(key) or {}
    if m.get("title"):
        return f"{m['title']} — {m['artist']}" if m.get("artist") else m["title"]
    return brut


# ── l'orthographe des notes (Louis, 2026-09-16 : « tu me les as écrits dans la
# mauvaise tona donc je peux pas ») ─────────────────────────────────────────
#
# J'écrivais tout en bémols, donc « A/D♭ » dans un morceau en la majeur, où
# cette note est un DO DIÈSE. Un accord mal orthographié n'est pas un détail
# cosmétique : il devient illisible, et Louis ne peut pas l'arbitrer.
#
# La règle est déjà dans l'app (`harmonia/static/ui/kit.js::setSpelling`, posée
# le 2026-07-19 sur un « G♭m7 affiché en mi majeur ») : les tons majeurs à
# bémols sont C D♭ E♭ F G♭ A♭ B♭, les autres s'écrivent en dièses, et un ton
# mineur suit son relatif majeur. On la reprend telle quelle, pour que la page
# d'arbitrage et le chart de l'app nomment la même note pareil.
FLAT = ["C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B"]
SHARP = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
FLAT_MAJ = {0, 1, 3, 5, 6, 8, 10}


def table_du_morceau(model: dict) -> list[str]:
    k = model.get("key") or {}
    tonic = int(k.get("tonic") or 0)
    maj = (tonic + 3) % 12 if k.get("mode") == "minor" else tonic % 12
    return FLAT if maj in FLAT_MAJ else SHARP


def respell(token: str, table: list[str]) -> str:
    """« Bb-7/Db » -> « B♭-7/C♯ » dans un ton à dièses. Parenthèses gardées."""
    t = token
    ouvre, ferme = ("(", ")") if t.startswith("(") and t.endswith(")") else ("", "")
    t = t.strip("()")
    if not t or t == "·":
        return token

    def _note(x: str) -> tuple[str, str]:
        """(nom réécrit, reste) — None-safe : rend (x, '') si illisible."""
        n = x[:2] if len(x) >= 2 and x[:2] in _PCS else (x[:1] if x[:1] in _PCS else None)
        if n is None:
            return x, ""
        return table[_PCS[n]], x[len(n):]

    if "/" in t:
        haut, bas = t.split("/", 1)
        r, q = _note(haut)
        b, _ = _note(bas)
        return f"{ouvre}{r}{q}/{b}{ferme}"
    r, q = _note(t)
    return f"{ouvre}{r}{q}{ferme}"

#: du plus parlant au moins parlant — c'est l'ordre de la page.
RANG = {"autre basse": 0, "la tab met un slash": 1, "soutient le slash": 2,
        "soutient l'accord nu": 3, "absent": 4, "muette": 5, None: 6}
COULEUR = {"autre basse": "warn", "la tab met un slash": "warn",
           "soutient le slash": "mark", "soutient l'accord nu": "dim",
           "absent": "faint", "muette": "faint"}


def collect() -> list[dict]:
    data = json.loads(MATCH.read_text(encoding="utf-8"))
    out = []
    for s in data:
        model = json.loads((AFTER / f"{s['song']}.json").read_text(encoding="utf-8"))
        grid = [float(x) for x in (model.get("barGrid") or [])]
        audio = model.get("audio_url") or ""
        stem = Path(audio).name or (s["song"].removeprefix("min_") + ".m4a")
        t = s.get("tab")
        table = table_du_morceau(model)
        for ln in (s.get("lignes") or []):
            k = int(ln["bar"])
            t0 = grid[k] if 0 <= k < len(grid) else None
            t1 = grid[k + 1] if 0 <= k + 1 < len(grid) else None
            v = (ln.get("tab") or {}).get("verdict")
            out.append({
                "song": s["song"],
                "title": joli_titre(s["song"], s.get("title") or s["song"]),
                "audio": "/audio/" + stem,
                "bar": k + 1, "t0": t0, "t1": t1,
                "avant": respell(ln["avant"], table),
                "apres": respell(ln["apres"], table),
                "basse_avant": ln.get("basse_avant"), "basse_apres": ln.get("basse_apres"),
                "verdict": v,
                "tab_avant": (ln.get("tab_avant") or {}).get("verdict"),
                "iv": intervalle(ln["apres"]),
                "ecrits": (ln.get("tab") or {}).get("ecrits") or [],
                "taux": (ln.get("tab") or {}).get("taux"),
                "tab_url": (t or {}).get("url"),
                "tab_note": None if not t else
                    f"{t.get('rating', 0):.2f}★ ×{t.get('votes', 0)}",
                "tab_score": (t or {}).get("score"),
                "rotation": (t or {}).get("rotation"),
                "capo": (t or {}).get("capo_declare"),
                "capo_ok": (t or {}).get("accord_rotation_capo"),
            })
    out.sort(key=lambda r: (RANG.get(r["verdict"], 6), r["title"], r["bar"]))
    for i, r in enumerate(out, 1):
        r["id"] = f"b{i:02d}"
    return out


NOM_IV = {0: "la fondamentale", 2: "la 9e", 3: "la 3ce mineure", 4: "la 3ce majeure",
          5: "la 4te", 7: "la quinte", 1: "la b9", 6: "la b5", 8: "la b13",
          9: "la 6te", 10: "la b7", 11: "la 7M"}


def intervalle(token: str) -> int | None:
    """L'intervalle basse-fondamentale d'un « Bb-7/Db », ou None s'il n'y a pas
    de slash. C'est la grandeur que les arbitrages de Louis ont tranchée."""
    t = token.strip("()")
    if "/" not in t:
        return None
    a, b = t.split("/", 1)
    r = a[:2] if a[:2] in _PCS else a[:1]
    return (_PCS[b] - _PCS[r]) % 12 if r in _PCS and b in _PCS else None


def render(rows: list[dict]) -> str:
    from collections import Counter

    from harmonia.bass_rules import IMPOSSIBLE, PLAUSIBLE

    c = Counter(r["verdict"] for r in rows)
    parlantes = sum(c[k] for k in ("autre basse", "la tab met un slash", "soutient le slash"))
    ivs = Counter(r["iv"] for r in rows if r["iv"] is not None)
    n_retire = sum(1 for r in rows if r["iv"] is None)
    illegaux = [r for r in rows if r["iv"] in IMPOSSIBLE]
    lignes_iv = " &middot; ".join(
        f"<b>{n}</b>× {NOM_IV.get(k, k)}" for k, n in ivs.most_common())
    verdict_regles = (
        f"<b>aucun</b> des {sum(ivs.values())} slashes écrits n'utilise un intervalle "
        f"que tu as écarté" if not illegaux else
        f"<b>{len(illegaux)} slashes</b> utilisent un intervalle que tu as écarté — "
        f"ce sont des bugs, pas des arbitrages")
    e = html.escape

    def carte(r: dict) -> str:
        col = COULEUR.get(r["verdict"], "faint")
        av, ap = e(r["avant"]), e(r["apres"])
        preuve = ""
        if r["verdict"] in ("autre basse", "la tab met un slash", "soutient le slash"):
            # quand la tab soutient explicitement l'AVANT, c'est ça qu'il faut
            # lire — inutile de laisser Louis le déduire des symboles.
            tete = ("<b>la tab soutient l'ancienne version</b> &mdash; elle écrit "
                    if r.get("tab_avant") in ("soutient le slash", "soutient l'accord nu")
                    else "<b>la tab écrit</b> ")
            preuve = (f"<div class='tab t-{col}'>{tete}"
                      f"{e(', '.join(r['ecrits']))}"
                      + (f" &middot; transposée de {r['rotation']:+d}" if r["rotation"] else "")
                      + (f" &middot; <a href='{e(r['tab_url'])}' target='_blank' rel='noopener'>"
                         f"{e(r['tab_note'] or 'la tab')}</a>" if r["tab_url"] else "")
                      + "</div>")
        elif r["verdict"] == "soutient l'accord nu":
            preuve = (f"<div class='tab t-dim'>la tab écrit {e(', '.join(r['ecrits']))} "
                      f"&mdash; sans basse. Elle note des renversements ailleurs "
                      f"({int((r['taux'] or 0) * 100)} % de ses accords), mais ce qu'elle "
                      f"écrit est l'accord joué à la guitare, pas la note du bassiste.</div>")
        elif r["verdict"] == "muette":
            preuve = "<div class='tab t-faint'>la tab n'écrit jamais de basse — elle ne dit rien ici.</div>"
        else:
            preuve = "<div class='tab t-faint'>cet accord n'est pas dans la tab.</div>"
        return f"""
<div class="row" id="r-{r['id']}" data-v="{e(r['verdict'] or '')}">
  <div class="head">
    <span class="song">{e(r['title'])}</span>
    <span class="bar">mes. {r['bar']}</span>
    <span class="chip c-{col}">{e(r['verdict'] or 'pas de tab')}</span>
  </div>
  <div class="pair">
    <span class="av">{av}</span><span class="arr">&rarr;</span><span class="ap">{ap}</span>
  </div>
  {preuve}
  <div class="acts">
    <button class="pl" data-a="{e(r['audio'])}" data-t0="{r['t0'] if r['t0'] is not None else ''}"
            data-t1="{r['t1'] if r['t1'] is not None else ''}" type="button">&#9658; écouter la mesure</button>
    <button class="pl" data-a="{e(r['audio'])}" data-t0="{r['t0'] if r['t0'] is not None else ''}"
            data-t1="{(r['t0'] + 0.9) if r['t0'] is not None else ''}" type="button">&#9658; l'attaque</button>
    <span class="sp"></span>
    <button class="v" data-id="{r['id']}" data-v="avant" type="button">{av}</button>
    <button class="v" data-id="{r['id']}" data-v="apres" type="button">{ap}</button>
    <button class="v" data-id="{r['id']}" data-v="autre" type="button">ni l'un ni l'autre</button>
  </div>
</div>"""

    cartes = "\n".join(carte(r) for r in rows)
    return f"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Les basses à trancher</title>
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
 body{{background:var(--bg);color:var(--ink);margin:0 auto;max-width:900px;
  font-family:'Public Sans',-apple-system,sans-serif;padding:22px 16px 90px;font-size:14.5px;line-height:1.55}}
 h1{{font-family:'Fraunces',Georgia,serif;font-size:26px;font-weight:600;margin:0;text-wrap:balance}}
 h2{{font-family:'Fraunces',Georgia,serif;font-size:16px;margin:26px 0 8px}}
 .eyebrow{{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:.09em;text-transform:uppercase;
  color:var(--accent-ink);background:var(--accent-soft);display:inline-block;padding:2px 7px;border-radius:3px;margin-bottom:8px}}
 .sub{{color:var(--ink-dim);font-size:13.5px;margin:5px 0 0}}
 .card{{background:var(--surface);border:1px solid var(--rule);border-radius:11px;padding:13px 15px;margin:14px 0}}
 code{{font-family:'IBM Plex Mono',monospace;font-size:12.5px;background:var(--surface-2);
  border:1px solid var(--rule);border-radius:4px;padding:1px 5px}}
 .prog{{position:sticky;top:0;z-index:20;background:var(--surface);border:1px solid var(--rule);
  border-radius:10px;padding:9px 13px;margin:16px 0;display:flex;gap:14px;flex-wrap:wrap;align-items:center;
  font-family:'IBM Plex Mono',monospace;font-size:12.5px;color:var(--ink-dim)}}
 .prog b{{color:var(--ink)}}
 .bar2{{flex:1;min-width:90px;height:6px;background:var(--surface-2);border:1px solid var(--rule);border-radius:3px;overflow:hidden}}
 .bar2 i{{display:block;height:100%;background:var(--mark);width:0}}
 .row{{border:1px solid var(--rule);border-radius:11px;background:var(--surface);padding:12px 14px;margin-bottom:8px}}
 .row.done{{border-color:var(--rule-strong);background:var(--surface-2)}}
 .head{{display:flex;gap:9px;align-items:baseline;flex-wrap:wrap;margin-bottom:6px}}
 .song{{font-weight:600;font-size:13.5px}}
 .bar{{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--ink-faint)}}
 .chip{{margin-left:auto;font-family:'IBM Plex Mono',monospace;font-size:10px;padding:2px 7px;border-radius:4px}}
 .c-warn{{background:var(--warn-soft);color:var(--warn)}} .c-mark{{background:var(--mark-soft);color:var(--mark)}}
 .c-dim{{background:var(--surface-2);color:var(--ink-dim);border:1px solid var(--rule)}}
 .c-faint{{background:var(--surface-2);color:var(--ink-faint);border:1px solid var(--rule)}}
 .pair{{font-family:'IBM Plex Mono',monospace;font-size:16px;font-weight:600;margin:5px 0 7px}}
 .pair .av{{color:var(--ink-dim)}} .pair .arr{{color:var(--ink-faint);margin:0 9px;font-weight:400}}
 .pair .ap{{color:var(--accent-ink)}}
 .tab{{font-size:12.5px;margin-bottom:9px;padding:7px 10px;border-radius:7px;background:var(--surface-2);
  border:1px solid var(--rule)}}
 .tab.t-warn{{background:var(--warn-soft);border-color:var(--warn);color:var(--warn)}}
 .tab.t-mark{{background:var(--mark-soft);border-color:var(--mark);color:var(--mark)}}
 .tab.t-dim{{color:var(--ink-dim)}} .tab.t-faint{{color:var(--ink-faint)}}
 .tab a{{color:inherit}}
 .acts{{display:flex;gap:7px;flex-wrap:wrap;align-items:center}}
 .sp{{flex:1;min-width:4px}}
 button{{font-family:'IBM Plex Mono',monospace;font-size:12px;border-radius:6px;padding:6px 10px;cursor:pointer;
  border:1px solid var(--rule);background:var(--surface-2);color:var(--ink-dim)}}
 button:hover{{border-color:var(--accent);color:var(--accent-ink);background:var(--accent-soft)}}
 button.playing{{background:var(--accent);border-color:var(--accent);color:var(--surface)}}
 button.v.on[data-v="avant"]{{background:var(--ink-dim);border-color:var(--ink-dim);color:var(--surface)}}
 button.v.on[data-v="apres"]{{background:var(--mark);border-color:var(--mark);color:#fff}}
 button.v.on[data-v="autre"]{{background:var(--warn);border-color:var(--warn);color:#fff}}
 details{{margin-top:18px}} summary{{cursor:pointer;font-family:'Fraunces',Georgia,serif;font-size:16px}}
 footer{{margin-top:30px;padding-top:12px;border-top:1px solid var(--rule);font-size:11.5px;
  color:var(--ink-faint);font-family:'IBM Plex Mono',monospace;line-height:1.8}}
</style>

<span class="eyebrow">basses de slash &middot; 16 morceaux &middot; {len(rows)} accords</span>
<h1>Les basses à trancher</h1>
<p class="sub">Aucun accord ne change de racine ni de qualité dans ce diff. Seule la basse bouge.</p>

<div class="card">
  <p><b>Ce que les tabs apportent, et ce qu'elles n'apportent pas.</b>
  J'ai récupéré les tabs Ultimate Guitar des 16 morceaux et je les ai remises dans ta
  tonalité — pas en croyant leur champ « capo », mais en essayant les douze
  transpositions et en gardant celle qui colle. Sur les quatre tabs capodastrées, la
  rotation trouvée retombe exactement sur la capo déclarée&nbsp;: le procédé se vérifie lui-même.</p>
  <p>Résultat honnête&nbsp;: <b>{parlantes} lignes sur {len(rows)}</b> reçoivent un vrai témoignage.
  Pour le reste, la tab est muette&nbsp;— et c'est structurel, pas un accident&nbsp;:
  <b>une tab écrit ce que gratte la main droite, notre cible est la note du bassiste.</b>
  Quand une tab écrit <code>Am</code> sans slash, elle ne dit pas que le bassiste ne joue
  pas un G&nbsp;; elle dit qu'on plaque un Am. Ces lignes-là sont rangées en bas, et
  seule ton oreille les tranche.</p>
</div>

<div class="card">
  <p><b>Ce que tes r\u00e8gles d'or disent d\u00e9j\u00e0.</b> Sur les {len(rows)} lignes,
  {sum(ivs.values())} \u00e9crivent une basse et {n_retire} en retirent une.
  Les intervalles \u00e9crits&nbsp;: {lignes_iv}.
  Compar\u00e9s \u00e0 ta table arbitr\u00e9e ({", ".join(NOM_IV.get(k, str(k)) for k in sorted(PLAUSIBLE))})&nbsp;:
  {verdict_regles}. La question qui reste n'est donc pas
  <i>cet intervalle est-il jouable</i> \u2014 tu l'as tranch\u00e9 \u2014 mais
  <i>la basse est-elle bien lue ICI</i>.</p>
</div>

<div class="prog">
  <span><b id="n">0</b>/{len(rows)}</span>
  <span class="bar2"><i id="pb"></i></span>
  <button id="copy" type="button">copier mes réponses</button>
  <button id="only" type="button">seulement les non répondues</button>
</div>

<h2>D'abord celles où la tab dit quelque chose</h2>
{cartes}

<footer>
  <code>tools/tab_match.py</code> &middot; transposition par les 12 rotations, contrôlée sur 4 capos
  &middot; garde-fou : une tab sous 5&nbsp;% d'accords slashés est considérée muette<br>
  diff : <code>golden/baseline</code> vs <code>golden/jumeau_large</code> &middot; basses seules, 0 accord changé
</footer>

<script>
const rows = {json.dumps([{k: r[k] for k in ("id", "song", "title", "bar", "avant", "apres", "verdict")} for r in rows], ensure_ascii=False)};
const KEY = 'basses_verdicts_v1';
let V = {{}};
try {{ V = JSON.parse(localStorage.getItem(KEY) || '{{}}'); }} catch (e) {{}}

let audio = null, playing = null, stopAt = null;
function play(btn){{
  const a = btn.dataset.a, t0 = parseFloat(btn.dataset.t0), t1 = parseFloat(btn.dataset.t1);
  if (!isFinite(t0)) return;
  if (playing) playing.classList.remove('playing');
  if (playing === btn) {{ audio.pause(); playing = null; return; }}
  if (!audio || audio.dataset.src !== a) {{
    if (audio) audio.pause();
    audio = new Audio(a); audio.dataset.src = a;
    audio.addEventListener('timeupdate', () => {{
      if (stopAt != null && audio.currentTime >= stopAt) {{
        audio.pause(); if (playing) playing.classList.remove('playing'); playing = null;
      }}
    }});
  }}
  stopAt = isFinite(t1) ? t1 + 0.25 : t0 + 3;
  const go = () => {{ audio.currentTime = Math.max(0, t0 - 0.15); audio.play().catch(()=>{{}}); }};
  if (audio.readyState >= 1) go(); else audio.addEventListener('loadedmetadata', go, {{once:true}});
  btn.classList.add('playing'); playing = btn;
}}
document.querySelectorAll('button.pl').forEach(b => b.addEventListener('click', () => play(b)));

function paint(){{
  let n = 0;
  rows.forEach(r => {{
    const el = document.getElementById('r-' + r.id);
    const v = V[r.id];
    if (v) n++;
    el.classList.toggle('done', !!v);
    el.querySelectorAll('button.v').forEach(b => b.classList.toggle('on', b.dataset.v === v));
  }});
  document.getElementById('n').textContent = n;
  document.getElementById('pb').style.width = (n / rows.length * 100) + '%';
}}
document.querySelectorAll('button.v').forEach(b => b.addEventListener('click', () => {{
  const id = b.dataset.id;
  V[id] = (V[id] === b.dataset.v) ? undefined : b.dataset.v;
  if (!V[id]) delete V[id];
  try {{ localStorage.setItem(KEY, JSON.stringify(V)); }} catch (e) {{}}
  paint();
}}));
document.getElementById('copy').addEventListener('click', async () => {{
  const txt = rows.filter(r => V[r.id])
    .map(r => `${{r.id}} ${{r.title}} mes.${{r.bar}} : ${{r.avant}} -> ${{r.apres}} = ${{V[r.id]}}`).join('\\n');
  const btn = document.getElementById('copy');
  try {{ await navigator.clipboard.writeText(txt || '(rien)'); btn.textContent = 'copi\\u00e9 \\u2713'; }}
  catch (e) {{ btn.textContent = 'copie refus\\u00e9e'; }}
  setTimeout(() => btn.textContent = 'copier mes r\\u00e9ponses', 1600);
}});
let filt = false;
document.getElementById('only').addEventListener('click', e => {{
  filt = !filt;
  e.currentTarget.classList.toggle('playing', filt);
  rows.forEach(r => {{ document.getElementById('r-' + r.id).hidden = filt && !!V[r.id]; }});
}});
paint();
</script>
"""


def main() -> int:
    rows = collect()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(rows), encoding="utf-8")
    print(f"→ {OUT}  ({len(rows)} lignes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
