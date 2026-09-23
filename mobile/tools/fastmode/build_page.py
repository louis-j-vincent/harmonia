"""T5-prep — assemble `docs/portage_mobile/plots/fastmode_3songs.html` à
partir des JSON produits par `run_lever.py` / `compare_levers.py` /
`speed_threads.py` (/ `int8.log` si le levier 4 a abouti).

Style repris de `tools/avant_apres.py` (mêmes couleurs, même widget
« cliquer une mesure pour l'écouter » via `<audio>` + `timeupdate`), mais
autonome — pas de republication dans la bibliothèque, pas d'import de ce
module (qui republie dans state/human, hors-sujet ici).

Liens audio en ABSOLU (`/audio/<stem>.m4a`), pas relatif `../audio/…` : cette
page vit sous `docs/portage_mobile/plots/`, pas `docs/plots/`, donc
`../audio/` pointerait au mauvais endroit. `/audio/<name>` est la route
Flask réelle (`harmonia/server/app.py:118`), qui sert depuis
`SETTINGS.audio_dir` quel que soit l'endroit d'où la page est chargée.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

SCRATCH = Path(sys.argv[1])
OUT = Path(sys.argv[2])

SONGS = ["maroon_5_this_love", "ben_e_king_stand_by_me_audio", "autumn_leaves"]
SONG_LABEL = {
    "maroon_5_this_love": "Maroon 5 — This Love",
    "ben_e_king_stand_by_me_audio": "Ben E. King — Stand By Me",
    "autumn_leaves": "Autumn Leaves (jazz)",
}
LEVER_LABEL = {
    "small0": "beat_this small0 (au lieu de final0)",
    "fold1": "musx 1 fold (au lieu de 5)",
    "fold3": "musx 3 folds (au lieu de 5)",
}

baseline = json.loads((SCRATCH / "baseline.json").read_text())
diffs = json.loads((SCRATCH / "diffs.json").read_text())
threads = json.loads((SCRATCH / "threads.json").read_text()) if (SCRATCH / "threads.json").exists() else None
int8_log = (SCRATCH / "int8.log").read_text() if (SCRATCH / "int8.log").exists() else ""
int8_parity = (json.loads((SCRATCH / "int8_parity.json").read_text())
              if (SCRATCH / "int8_parity.json").exists() else None)
diag_int8_log = (SCRATCH / "diag_int8.log").read_text() if (SCRATCH / "diag_int8.log").exists() else ""


def esc(s):
    return html.escape(str(s))


def fmt_s(x):
    return f"{x:.1f} s" if x is not None else "—"


def bar_rows_html(stem, lever_data):
    rows = lever_data["changed_bars"]
    if not rows:
        return "<p class='ok'>Aucune mesure différente.</p>"
    out = ["<table><tr><th>mesure</th><th>avant</th><th></th><th>après</th><th></th></tr>"]
    for r in rows[:400]:  # garde-fou d'affichage
        t0 = r.get("t0")
        btn = (f"<button class='play' data-src='/audio/{esc(stem)}.m4a' "
               f"data-t0='{t0}' data-t1='{(t0 or 0) + 6}'>&#9654;</button>") if t0 is not None else ""
        out.append(f"<tr><td>{r['bar']}</td><td class='chord'>{esc(r['before'])}</td>"
                   f"<td>{btn}</td><td class='chord new'>{esc(r['after'])}</td><td></td></tr>")
    if len(rows) > 400:
        out.append(f"<tr><td colspan=5>… {len(rows)-400} mesures de plus (tronqué à l'affichage)</td></tr>")
    out.append("</table>")
    return "".join(out)


def one_liner(stem, lever, d):
    n = d["n_changed"]
    nb0, nb1 = d["nBars_before"], d["nBars_after"]
    bt0, bt1 = d["n_beats_before"], d["n_beats_after"]
    if lever == "small0":
        if bt0 != bt1 or nb0 != nb1:
            return (f"small0 DÉPLACE la grille de battues : {bt0}→{bt1} battues, "
                    f"{nb0}→{nb1} mesures — {n} mesures d'accords changent aussi. "
                    f"Pas un simple arrondi : le morceau se redécoupe.")
        if n == 0:
            return "small0 ne change RIEN sur ce morceau : mêmes battues, mêmes accords."
        return f"small0 garde la même grille de battues mais change {n} mesure(s) d'accords sur {nb0}."
    else:
        nfold = "1 fold" if lever == "fold1" else "3 folds"
        pct = round(100 * n / max(1, nb0))
        return f"{nfold} au lieu de 5 change {n} mesure(s) sur {nb0} ({pct} %) — la grille de battues ne bouge pas."


def song_section(stem):
    base = baseline[stem]
    parts = [f"<h2>{esc(SONG_LABEL[stem])}</h2>",
             f"<p class='meta'>Référence (final0, 5 folds) : {base['nBars']} mesures, "
             f"{base['n_beats']} battues, {fmt_s(base['total_s'])} de calcul "
             f"(Mac, torch/MPS) — <code>{json.dumps(base['backend_timings'])}</code></p>"]
    for lever in ("small0", "fold1", "fold3"):
        d = diffs.get(lever, {}).get(stem)
        if d is None:
            continue
        gain = None
        if d["total_s_before"] and d["total_s_after"]:
            gain = d["total_s_before"] - d["total_s_after"]
        parts.append(f"<div class='lever-block'>")
        parts.append(f"<h3>{esc(LEVER_LABEL[lever])}</h3>")
        parts.append(f"<p class='tell'>{esc(one_liner(stem, lever, d))}</p>")
        parts.append(f"<p class='meta'>temps : {fmt_s(d['total_s_before'])} → {fmt_s(d['total_s_after'])}"
                     + (f" (gagné {fmt_s(gain)})" if gain is not None else "") + "</p>")
        parts.append(bar_rows_html(stem, d))
        parts.append("</div>")
    return "".join(parts)


def threads_table():
    if threads is None:
        return "<p class='ok'>mesure non disponible (script en cours ou échoué).</p>"
    rows = ["<table><tr><th>configuration</th><th>1 thread</th><th>2 threads</th><th>4 threads</th></tr>"]
    mt = threads.get("musx_threads", {})
    pt = threads.get("pte_threads", {})
    rows.append("<tr><td>musx numpy (OpenBLAS, ensemble 5 folds)</td>"
               f"<td>{mt.get('1','—')}s</td><td>{mt.get('2','—')}s</td><td>{mt.get('4','—')}s</td></tr>")
    rows.append("<tr><td>beat_this .pte (XNNPACK)</td>"
               f"<td>{pt.get('1','—')}s</td><td>{pt.get('2','—')}s</td><td>{pt.get('4','—')}s</td></tr>")
    rows.append("</table>")
    seq = threads.get("sequential_1thread_each", {})
    par = threads.get("parallel_2_python_threads_1lib_each", {})
    rows.append(f"<p class='meta'>Séquentiel 1 thread chacun : musx {seq.get('musx_s')}s + "
               f"pte {seq.get('pte_s')}s = {seq.get('total_s')}s. "
               f"En parallèle (2 threads Python, 1 thread lib chacun) : mur d'horloge "
               f"{par.get('wall_s')}s → accélération ×{par.get('speedup_vs_sequential')}.</p>")
    rows.append("<p class='tell'>Mesure prise EN MÊME TEMPS que le job de quantification int8 "
               "(contention CPU, gros consommateur) — la colonne .pte (51→32→22s, monotone) "
               "reste exploitable, la ligne musx numpy (24→62→93s, PLUS LENT avec plus de "
               "threads) est presque certainement un artefact de contention, pas une vraie "
               "propriété du code — à re-mesurer isolément avant toute décision. Le ×1,10 de "
               "la dernière ligne (attendu ×2) est un signal plus solide : la contention aurait "
               "plutôt masqué un vrai gain, pas créé un faux plafond.</p>")
    return "".join(rows)


def int8_summary():
    if not int8_log.strip():
        return "<p class='ok'>Script pas encore terminé au moment de la publication.</p>"
    if "non tenté" in int8_log or "BUDGET EXCEEDED" in int8_log or "FAILED" in int8_log:
        tail = "\n".join(int8_log.strip().splitlines()[-15:])
        return (f"<p class='tell'>Non abouti dans la boîte de temps (40 min) ou bloqué — "
               f"noté « non tenté ». Dernières lignes :</p><pre>{esc(tail)}</pre>")
    wrote_line = next((l for l in int8_log.splitlines() if l.startswith("WROTE")), "")
    total_line = next((l for l in int8_log.splitlines() if l.startswith("TOTAL TIME")), "")
    out = [f"<p class='tell'>Export réussi en {esc(total_line.replace('TOTAL TIME: ',''))}. "
          f"{esc(wrote_line)}.</p>"]
    out.append("<p class='tell'><b>Premier diff (indice par indice) trompeur, corrigé</b> : "
               "comparer battue N du float à battue N de l'int8 fabrique un écart énorme dès que "
               "les deux modèles n'ont pas le même compte — un seul décalage au début désaligne "
               "tout le reste. Le tableau ci-dessous est le diff CORRIGÉ (plus proche voisin, "
               "comme un vrai score de battues), sur This Love seulement (budget).</p>")
    if diag_int8_log.strip() and "nearest-neighbor" in diag_int8_log:
        nn_line = next((l for l in diag_int8_log.splitlines() if "nearest-neighbor" in l), "")
        counts_f = next((l for l in diag_int8_log.splitlines() if l.startswith("float:")), "")
        counts_i = next((l for l in diag_int8_log.splitlines() if l.startswith("int8:")), "")
        out.append(f"<table><tr><th></th><th>float32</th><th>int8</th></tr>"
                   f"<tr><td>compte</td><td>{esc(counts_f.replace('float: ',''))}</td>"
                   f"<td>{esc(counts_i.replace('int8:  ','').replace('int8: ',''))}</td></tr></table>")
        out.append(f"<p class='meta'>{esc(nn_line)}</p>")
        out.append("<p class='tell'>Mécanisme : les battues qui existent déjà bougent PEU "
                   "(médiane ~20 ms, bruit de quantification ordinaire) — le vrai défaut est "
                   "que l'int8 INSÈRE des battues fantômes à ~80 ms d'une vraie, trop loin pour "
                   "que le dédoublonnage (fenêtre 20 ms) les fusionne : le bruit crée un second "
                   "maximum local concurrent. Calibration = 2 fenêtres de 30 s seulement — "
                   "probablement insuffisant pour la tête de sortie (là où ce test compte). "
                   "Fix proposé, PAS implémenté (hors boîte de temps) : calibrer sur beaucoup "
                   "plus de fenêtres + garder la dernière couche (logits) en float32.</p>")
    else:
        out.append("<p class='ok'>Diagnostic plus-proche-voisin pas encore terminé.</p>")
    if int8_parity and "error" not in int8_parity:
        out.append("<p class='meta'>Diff BRUT indice-par-indice (trompeur, gardé pour mémoire) : "
                   "</p><table><tr><th>morceau</th><th>temps float32</th><th>temps int8</th>"
                   "<th>battues (n)</th><th>max|Δt| indice-par-indice</th></tr>")
        for stem, d in int8_parity.items():
            b = d["beats"]
            out.append(f"<tr><td>{esc(SONG_LABEL.get(stem, stem))}</td>"
                       f"<td>{d['t_float_s']}s</td><td>{d['t_int8_s']}s</td>"
                       f"<td>{b['n_a']}/{b['n_b']}</td>"
                       f"<td>{b['max_abs_diff_ms']/1000:.1f} s (artefact, voir ci-dessus)</td></tr>")
        out.append("</table>")
    return "".join(out)


def speed_table():
    b = baseline
    def tot(d, stem):
        return d[stem]["total_s"] if stem in d else None
    rows = []
    combos = [
        ("baseline (final0, 5 folds) — ce que sert :7772 aujourd'hui", baseline),
    ]
    for lever in ("small0", "fold1", "fold3"):
        p = SCRATCH / f"{lever}.json"
        if p.exists():
            combos.append((LEVER_LABEL[lever], json.loads(p.read_text())))
    out = ["<table><tr><th>combinaison</th>"] + [f"<th>{SONG_LABEL[s]}</th>" for s in SONGS] + ["<th>Mac 1 thread, total</th><th>extrapolation ×3 (téléphone)</th></tr>"]
    for name, data in combos:
        vals = [tot(data, s) for s in SONGS]
        total = sum(v for v in vals if v is not None)
        cells = "".join(f"<td>{fmt_s(v)}</td>" for v in vals)
        out.append(f"<tr><td>{esc(name)}</td>{cells}<td>{fmt_s(total)}</td><td>{fmt_s(total*3)}</td></tr>")
    out.append("</table>")
    return "".join(out)


HEAD = """<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>T5-prep — leviers de vitesse mobile</title>
<style>
*{box-sizing:border-box}
body{margin:0;padding:14px 10px 90px;background:#faf6ec;color:#2c2820;
 font:15px/1.45 system-ui,sans-serif;max-width:820px;margin-inline:auto}
h1{font-size:20px;margin:0 0 6px} h2{font-size:17px;margin:30px 0 4px;border-top:2px solid #d8cfb4;padding-top:14px}
h3{font-size:14.5px;margin:14px 0 2px;color:#5a4b2a}
.lede{background:#fff7df;border:1px solid #e8d9a8;border-radius:10px;padding:10px 12px;font-size:14px;margin:10px 0}
.meta{color:#8a8371;font-size:12.5px;margin:2px 0 8px}
.tell{background:#eef4e6;border:1px solid #a9c488;border-radius:8px;padding:6px 10px;font-size:13.5px;margin:6px 0}
.ok{color:#5a8a4a;font-size:13px}
table{border-collapse:collapse;font-size:12.5px;width:100%;display:block;overflow-x:auto;margin:6px 0}
td,th{border:1px solid #ddd3b8;padding:3px 7px;white-space:nowrap;text-align:left}
th{background:#f3edda}
.chord{font-weight:600}
.chord.new{color:#8a2b2b}
.lever-block{margin:8px 0 18px;padding:8px 10px;background:#fff;border:1px solid #e8e0c8;border-radius:8px}
button.play{border:1px solid #d8cfb4;border-radius:6px;background:#fff;padding:2px 8px;cursor:pointer}
button.play.on{background:#8a2b2b;border-color:#8a2b2b;color:#fff}
pre{background:#111;color:#c8c8c8;font-size:11px;padding:8px;border-radius:8px;overflow-x:auto}
.audiobar{position:fixed;left:0;right:0;bottom:0;background:#2c2820;padding:8px 10px}
.audiobar audio{width:100%}
</style></head><body>
"""

FOOT = """
<div class="audiobar"><audio id="au"></audio></div>
<script>
const au=document.getElementById('au');let stopAt=null,cur=null;
au.addEventListener('timeupdate',()=>{if(stopAt!=null&&au.currentTime>=stopAt){
 au.pause();stopAt=null;if(cur){cur.classList.remove('on');cur=null;}}});
document.querySelectorAll('button.play').forEach(el=>{
 el.addEventListener('click',()=>{
  const src=el.dataset.src,t0=parseFloat(el.dataset.t0),t1=parseFloat(el.dataset.t1);
  if(cur===el&&!au.paused){au.pause();cur.classList.remove('on');cur=null;return;}
  if(cur)cur.classList.remove('on');cur=el;el.classList.add('on');
  const go=()=>{try{au.currentTime=t0;}catch(e){}stopAt=t1;au.play().catch(()=>{el.classList.remove('on');cur=null;});};
  if(au.getAttribute('src')!==src){au.setAttribute('src',src);au.addEventListener('loadedmetadata',go,{once:true});au.load();}
  else go();
 });
});
</script>
</body></html>
"""

body = [HEAD]
body.append("<h1>T5-prep — leviers de vitesse (avant l'arrivée du téléphone)</h1>")
body.append("<div class='lede'>Objectif : 4 min de musique en &lt; 2 min de calcul sur le téléphone "
            "(porte T5). Ici, sur Mac 1 thread : ce que chaque raccourci change VRAIMENT dans le "
            "chart, mesure par mesure, sur 3 morceaux que tu connais. Les temps sont indicatifs "
            "(Mac ≠ téléphone) — la porte se joue sur le téléphone réel, à son arrivée. "
            "Clique un accord surligné pour écouter la mesure.</div>")

body.append("<h2>Tableau récapitulatif des temps</h2>")
body.append(speed_table())
body.append("<p class='meta'>« extrapolation ×3 » : facteur générique cité dans le plan pour "
            "un téléphone 2-4× plus lent par cœur qu'un Mac — PAS une mesure, à remplacer "
            "dès que T5 tourne sur le vrai appareil.</p>")

for stem in SONGS:
    body.append(song_section(stem))

body.append("<h2>Levier 1 — parallélisme (ne change aucun chart)</h2>")
body.append(threads_table())

body.append("<h2>Levier 4 — quantification int8 de beat_this</h2>")
body.append(int8_summary())

body.append("<h2>Ce que ça te dit</h2>")
body.append("<p class='tell'>Louis arbitre : la table du haut donne le temps gagné par "
            "combinaison ; chaque bloc au-dessus donne EXACTEMENT ce qui change dans le chart "
            "si tu acceptes ce levier. Aucune ligne ci-dessus ne décide à ta place.</p>")

body.append(FOOT)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("".join(body), encoding="utf-8")
print(f"wrote {OUT}")
