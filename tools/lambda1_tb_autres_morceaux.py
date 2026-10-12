"""λ1 t+b sur d'autres morceaux (Louis, 2026-10-12 : « je trouve λ1 t+b très
intéressant, fais voir ce qu'il donne sur d'autres morceaux »).

    .venv/bin/python -m tools.lambda1_tb_autres_morceaux

Écrit docs/plots/lambda1_tb_autres_morceaux_2026-10-12.html. Variante
EXACTEMENT celle d'Evening Star (tools/demi_mesures_delta_decodeur.py, `core`) :
par demi-mesure, q = norm(max(0, p + S)) avec S = 2p(h) − p(h−1) − p(h+1) sur
la triade et la basse seulement, extensions = p ; même décodeur de production.
Aucun réglage changé. On compare à λ0 (q = p, la postérieure seule).

Contrôle indépendant : chroma NNLS de l'audio brut (pas musx), moyenné sur la
demi-mesure ; cosinus entre les notes aiguës et les notes de chaque accord, et
la note de basse la plus forte. Piège connu : la sortie VAMP commence à la
(index 0 = A) → np.roll(−3) pour avoir do en 0 (vérifié ci-dessous sur un
accord tenu de chaque morceau, imprimé).
"""
from __future__ import annotations

import html

import numpy as np

from harmonia.nnls_features import extract_bothchroma
from harmonia.settings import SETTINGS
from tools.demi_mesures_delta_decodeur import NOTES, core, lab_name

OUT = SETTINGS.repo / "docs/plots/lambda1_tb_autres_morceaux_2026-10-12.html"
SONGS = [("ujgPfUZ7Whc", "La Jupe En Laine"),
         ("maroon_5_this_love", "This Love (Maroon 5)"),
         ("9zEl-FQLI4A", "San Francisco Street")]
MAX_SHOWN = 25

ROOT = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6,
        "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
QPCS = {"maj": [0, 4, 7], "min": [0, 3, 7], "aug": [0, 4, 8], "dim": [0, 3, 6],
        "hdim7": [0, 3, 6, 10], "dim7": [0, 3, 6, 9], "7": [0, 4, 7, 10], "maj7": [0, 4, 7, 11],
        "min7": [0, 3, 7, 10], "maj(6)": [0, 4, 7, 9], "min(6)": [0, 3, 7, 9], "sus4": [0, 5, 7],
        "sus2": [0, 2, 7], "sus4(b7)": [0, 5, 7, 10], "9": [0, 4, 7, 10, 2], "maj9": [0, 4, 7, 11, 2],
        "min9": [0, 3, 7, 10, 2], "11": [0, 4, 7, 10, 5], "13": [0, 4, 7, 10, 9]}
IV = {"2": 2, "b3": 3, "3": 4, "5": 7, "b7": 10}


def lab_pcs(lab: str):
    """'Bb:min/b3' → (notes {si♭ ré♭ fa}, basse ré♭)."""
    if lab == "N":
        return [], None
    r, _, q = lab.partition(":")
    q, _, b = q.partition("/")
    root = ROOT[r]
    pcs = [(root + k) % 12 for k in QPCS[q]]
    return pcs, (root + IV[b]) % 12 if b else root


def cos(v, pcs):
    t = np.zeros(12); t[pcs] = 1
    nv = np.linalg.norm(v)
    return float(v @ t / (nv * np.linalg.norm(t))) if nv > 1e-9 and pcs else 0.0


#: Lecture musicale de chaque changement, écrite APRÈS avoir lu les top 3 et le
#: chroma imprimés par collect(). (stem, demi-mesure) → (classe, texte).
#: classe : "ok" plausible, "q" à écouter, "x" artefact probable.
TAGS: dict[tuple[str, int], tuple[str, str]] = {
    # La Jupe En Laine (do mineur)
    ("ujgPfUZ7Whc", 98): ("ok", "Juste après un G7 : le si♮ est la note la plus forte, sur mi♭-sol. E♭+ = mi♭ sol si "
                                 "= Cm(maj7) sur basse mi♭ — le si du G7 qui reste, couleur classique en mineur."),
    ("ujgPfUZ7Whc", 154): ("x", "Le si♭ est la note la plus forte (B♭m/D♭ = D♭6 colle à 0,96, D♭ seul 0,62). "
                                 "Il était déjà là dans le B♭7 d'avant : la différence efface ce qui est commun "
                                 "avec la voisine, et le 6te disparaît. λ1 rejoint le chart, l'audio dit D♭6."),
    ("ujgPfUZ7Whc", 155): ("x", "Même chose, 2e moitié : si♭ toujours fort, D♭6 colle mieux que D♭ (0,84 contre 0,56)."),
    # This Love (do mineur)
    ("maroon_5_this_love", 0): ("x", "Toute première demi-mesure : une seule voisine, comptée deux fois. La basse "
                                     "si est la plus forte (sol 0,75) : le G/B du chart et de λ0 tient mieux ; "
                                     "λ1 perd le renversement et hésite même avec « pas d'accord » (22 %)."),
    ("maroon_5_this_love", 15): ("q", "Basse la♭ (pas ré, 0,23) avec fa, si♭, mi♭ en haut : quelque chose sur "
                                      "la♭ plutôt que Dø7. B♭/A♭ (= B♭7 avec la 7e à la basse) colle un peu mieux "
                                      "(0,76 contre 0,65). λ0 avait déjà B♭/A♭ en tête, le décodeur gardait Dø7."),
    ("maroon_5_this_love", 51): ("q", "Basse sol, puis sol♭, avant le Fm : la basse descend sol → sol♭ → fa. "
                                      "Cm/G note bien la basse ; mais l'audio a aussi sol♭ et ré♭ forts (ni Cm ni "
                                      "Cm/G ne collent, 0,56) : peut-être un accord de passage G♭ vers Fm."),
    ("maroon_5_this_love", 55): ("q", "λ1 rejoint le chart (A♭), mais l'audio a si♭ ré la♭ = B♭7, basse si♭ "
                                      "(B♭/A♭ 0,71, A♭ 0,54). La voisine suivante est un silence (N.C.), ce qui "
                                      "déforme la différence."),
    ("maroon_5_this_love", 65): ("ok", "Avant Cm : G/B → Cm, le V renversé qui ramène au i. L'audio n'a pas de fa "
                                       "(donc pas de B°) ; do et mi♭ forts = le Cm qui arrive un peu en avance."),
    ("maroon_5_this_love", 91): ("q", "Même endroit au 2e couplet, même lecture : basse sol puis sol♭ vers Fm, "
                                      "sol♭ et ré♭ forts. Deux couplets sur deux : ce n'est pas du bruit, mais "
                                      "le nom (Cm/G ? G♭ ?) est à écouter."),
    ("maroon_5_this_love", 101): ("q", "L'audio a bien si ré♯ fa♯ (B majeur colle à 0,84, B° 0,61). Un accord "
                                       "d'approche un demi-ton sous Cm ? Inattendu en do mineur, à écouter."),
    ("maroon_5_this_love", 127): ("ok", "Le chart et λ0 disent A♭, mais l'audio n'a presque pas de la♭ (basse 0,21) : "
                                        "basse si, sol et si en haut → G/B qui ramène à Cm (0,76 contre 0,38). "
                                        "Même geste que mes. 33."),
    ("maroon_5_this_love", 135): ("q", "Même accord, seule la basse change (si♭ → mi♭). La basse du chroma est "
                                       "floue (un mi♮ en tête, sans doute une note de passage) : rien ne tranche. "
                                       "Le si♭ était la basse de l'accord d'avant : la différence l'a effacé."),
    ("maroon_5_this_love", 143): ("ok", "Avant Cm, basse sol, avec sol ré fa en haut : une dominante sur sol "
                                        "(Fm/G ≈ G7sus4 avec ♭9) plutôt qu'A♭ (la♭ faible). Même rôle que mes. 33 "
                                        "et 64 ; le nom exact est à écouter."),
    # San Francisco Street (ré majeur)
    ("9zEl-FQLI4A", 62): ("ok", "Basse la très nette (sol 0,03) : G/A (= A9sus4, le son « sol sur la ») colle ; "
                                "λ1 rejoint le chart, λ0 avait oublié la basse."),
    ("9zEl-FQLI4A", 144): ("q", "La tierce manque : ni do (0,02) ni do♯ (0,11) dans l'audio. Am7, A7 ou A7sus4 "
                                "sont tous défendables ; l'oreille décide."),
    ("9zEl-FQLI4A", 145): ("q", "2e moitié : un peu de do (0,29) contre do♯ (0,17) — léger avantage à Am7 "
                                "(chart et λ0). λ1 garde A7 par continuité avec la 1re moitié."),
    ("9zEl-FQLI4A", 217): ("ok", "Pas de la à la basse (0,14) ; mi sol si ré = Em7 (0,94 contre 0,50). Avant "
                                 "Dmaj7 : le ii (Em7) qui déborde sur le V, ou un ii–I sans V. Plausible."),
    ("9zEl-FQLI4A", 220): ("x", "Aucune note d'Amaj7 dans l'audio (0,24 ; sol, si, ré, fa♯ = Gmaj7, 0,91). "
                                "Coincée entre Dmaj7 et Gmaj7, la demi-mesure n'a plus rien de « nouveau » : "
                                "la différence amplifie un reste de bruit."),
    ("9zEl-FQLI4A", 239): ("x", "Dernière demi-mesure du morceau : une seule voisine, comptée deux fois. Pas de "
                                "la à la basse (sol et mi) : Em7 (chart, 0,90) tient mieux que G/A."),
}


def collect():
    songs = []
    for stem, title in SONGS:
        c = core(stem, verbose=False)
        D, TOP, H, halves = c["DEC"], c["TOP"], c["H"], c["halves"]
        arr, times = extract_bothchroma(c["audio"])
        bassc = np.roll(arr[:, :12], -3, axis=1)
        treb = np.roll(arr[:, 12:], -3, axis=1)

        def chroma(i):
            a, e = halves[i]
            sel = (times >= a) & (times < e)
            return treb[sel].mean(0), bassc[sel].mean(0)

        # règle n°1 : le décalage −3 sur un accord tenu (le plus long tenu de λ0)
        if c["held"]:
            i0 = max(c["held"], key=lambda i: TOP["l0"][i][0][1])
            tv, bv = chroma(i0)
            print(f"[{stem}] contrôle chroma sur {lab_name(D['l0'][i0])} mes {i0 // 2 + 1} : "
                  f"notes={[NOTES[j] for j in np.argsort(-tv)[:4]]} basse={NOTES[int(bv.argmax())]} "
                  f"cos={cos(tv, lab_pcs(D['l0'][i0])[0]):.2f}")

        ch = []
        for i in range(H):
            a0, a1 = D["l0"][i], D["l1tb"][i]
            if a0 == a1:
                continue
            tv, bv = chroma(i)
            (p0, b0), (p1, b1) = lab_pcs(a0), lab_pcs(a1)
            bn = bv / max(bv.max(), 1e-9)
            r0, q0 = a0.split(":")[0] if a0 != "N" else "N", a0.partition(":")[2].partition("/")[0]
            r1, q1 = a1.split(":")[0] if a1 != "N" else "N", a1.partition(":")[2].partition("/")[0]
            kind = ("renversement (même accord, autre basse)" if (r0, q0) == (r1, q1)
                    else "couleur (même fondamentale)" if r0 == r1
                    else "autre fondamentale")
            conf = dict(TOP["l1tb"][i]).get(lab_name(a1), 0.0)
            ch.append(dict(i=i, bar=i // 2 + 1, half=i % 2, chart=c["chart_h"][i].replace("h7", "ø7"),
                           l0=lab_name(a0), l1=lab_name(a1), top0=TOP["l0"][i], top1=TOP["l1tb"][i],
                           cos0=cos(tv, p0), cos1=cos(tv, p1),
                           bass_top=[(NOTES[j], float(bn[j])) for j in np.argsort(-bv)[:2]],
                           bass0=NOTES[b0] if b0 is not None else "–", bass1=NOTES[b1] if b1 is not None else "–",
                           bassv0=float(bn[b0]) if b0 is not None else 0.0, bassv1=float(bn[b1]) if b1 is not None else 0.0,
                           notes=[NOTES[j] for j in np.argsort(-tv)[:4]],
                           prev=lab_name(D["l0"][i - 1]) if i else "–",
                           next=lab_name(D["l0"][i + 1]) if i + 1 < H else "–",
                           kind=kind, conf=conf,
                           t0=c["grid"][max(0, i // 2 - 1)], t1=c["grid"][i // 2 + 1]))
            d = ch[-1]
            print(f"[{stem}] h={i} {d['bar']}{'ab'[d['half']]} chart={d['chart']} {d['l0']}→{d['l1']} "
                  f"cos {d['cos0']:.2f}→{d['cos1']:.2f} basse {d['bass_top']} ({d['bass0']} {d['bassv0']:.2f} / "
                  f"{d['bass1']} {d['bassv1']:.2f}) notes={d['notes']} avant={d['prev']} après={d['next']}")
        hs = c["held_stats"]["l1tb"]
        songs.append(dict(stem=stem, title=title, H=H, changes=ch, held=hs, key=c["chart"].get("keyName", "").replace(" minor", " mineur").replace(" major", " majeur"),
                          n_irreg=c["n_irreg"]))
        print(f"[{stem}] {len(ch)}/{H} changées ; tenus gardés {hs[0]}/{hs[1]} ; mesures irrégulières {c['n_irreg']}")
    return songs


# ── page ─────────────────────────────────────────────────────────────────────

SUMMARY = [
    "<b>Le motif d'Evening Star se répète</b> : peu de changements (3, 10 et 6 demi-mesures, contre 13 sur "
    "Evening Star), et les accords tenus gardent tous leur nom.",
    "<b>Ce qui change est souvent musical</b> : sur This Love, trois fois une dominante sur sol (G/B, Fm/G) juste "
    "avant Cm là où le chart écrit A♭ ou B° ; sur San Francisco Street un Em7 (ii) au lieu d'A7 avant D ; "
    "sur La Jupe un E♭+ (= Cm maj7) juste après G7.",
    "<b>Pas de qualité inventée en série.</b> Deux artefacts reviennent : (1) la 1re et la dernière demi-mesure "
    "du morceau (une seule voisine) ; (2) une note déjà présente dans l'accord voisin est effacée — "
    "D♭6 → D♭ après B♭7, E♭/B♭ → E♭ après B♭. La différence punit les notes communes.",
]
SHOWS = {
    "ujgPfUZ7Whc": "Presque rien ne bouge (3 sur 182). Un E♭+ plausible après G7 ; les deux autres "
                   "changements sont le piège « note commune » : le 6te de D♭6 effacé parce que le B♭7 d'avant l'avait.",
    "maroon_5_this_love": "Le plus intéressant : là où le chart écrit A♭ ou B° juste avant Cm, λ1 trouve 3 fois "
                          "une dominante sur sol, et l'audio est d'accord (pas de la♭, pas de fa). "
                          "Le reste : un bord de morceau, des basses de passage à écouter.",
    "9zEl-FQLI4A": "Très stable (6 sur 240, 47 accords tenus sur 47). Deux changements plausibles (G/A avec "
                   "basse la, Em7 avant D), une tierce absente à trancher à l'oreille, deux artefacts "
                   "(Amaj7 inventé entre deux voisins proches, dernière demi-mesure).",
}


def _top(t):
    return " · ".join(f"{html.escape(n)} {100 * p:.0f} %" for n, p in t)


def write_page(songs):
    blocks = []
    for k, s in enumerate(songs):
        ch = sorted(s["changes"], key=lambda d: -d["conf"])
        shown, more = ch[:MAX_SHOWN], ch[MAX_SHOWN:]
        items = []
        for d in sorted(shown, key=lambda d: d["i"]):
            cls, txt = TAGS.get((s["stem"], d["i"]), ("q", "à écouter."))
            lab = {"ok": "plausible", "q": "à écouter", "x": "artefact probable"}[cls]
            better = ("λ1 t+b" if d["cos1"] > d["cos0"] + 0.02 else "λ0" if d["cos0"] > d["cos1"] + 0.02
                      else "égalité")
            items.append(f"""<div class="dv">
<div class="dh"><b>mes. {d['bar']}, temps {'1-2' if d['half'] == 0 else '3-4'}</b>
<button class="play" data-s="{k}" data-t0="{d['t0']:.3f}" data-t1="{d['t1']:.3f}">▶ écouter</button></div>
<div class="chg"><span class="mut">chart</span> {html.escape(d['chart'])} <span class="mut">· λ0</span> <b>{html.escape(d['l0'])}</b> → <span class="mut">λ1 t+b</span> <b class="new">{html.escape(d['l1'])}</b></div>
<div class="tp"><b>λ0</b> {_top(d['top0'])}</div>
<div class="tp"><b>λ1 t+b</b> {_top(d['top1'])}</div>
<div class="tp mut">audio brut : notes {' '.join(d['notes'])} ; basse {d['bass_top'][0][0]}
· accord des notes {html.escape(d['l0'])} {d['cos0']:.2f} / {html.escape(d['l1'])} {d['cos1']:.2f} → <b>{better}</b></div>
<p class="vd v{cls}"><b>{lab}.</b> {txt}</p></div>""")
        extra = ""
        if more:
            from collections import Counter
            cnt = Counter(d["kind"] for d in more)
            extra = f"<p class='mut'>+ {len(more)} autres : " + ", ".join(f"{v} {k}" for k, v in cnt.items()) + "</p>"
        kinds = {}
        for d in ch:
            kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
        blocks.append(f"""
<h2>{html.escape(s['title'])}</h2>
<p class="hd"><b>{len(ch)}</b> demi-mesures changent sur <b>{s['H']}</b> ; accords tenus qui gardent leur nom : <b>{s['held'][0]} / {s['held'][1]}</b>.</p>
<p class="mut">Tonalité du chart : {html.escape(s['key'])}. Genre de changement : {', '.join(f'{v} {k}' for k, v in kinds.items())}. Le bouton joue la mesure d'avant + la mesure.</p>
<div class="sh">{SHOWS.get(s['stem'], '')}</div>
{''.join(items)}{extra}""")

    auds = "".join(f'<audio id="au{k}" preload="none"></audio>' for k in range(len(songs)))
    srcs = ",".join(f"'../audio/{SETTINGS.audio_dir.joinpath(s['stem']).name}{_ext(s['stem'])}'" for s in songs)
    page = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>λ1 t+b, autres morceaux</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1d1d1f;--mut:#6b6b70;--line:#e3e1dc;--acc:#2a6fdb;--ok:#1d8a4e;--ko:#c0392b;--q:#b7791f;--card:#fff;--new:#9a5b00}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#151517;--fg:#ececef;--mut:#9a9aa2;--line:#2c2c31;--acc:#6ea0ff;--ok:#4cc38a;--ko:#ff7a6b;--q:#e0b050;--card:#1d1d21;--new:#f0b860}}}}
:root[data-theme="dark"]{{--bg:#151517;--fg:#ececef;--mut:#9a9aa2;--line:#2c2c31;--acc:#6ea0ff;--ok:#4cc38a;--ko:#ff7a6b;--q:#e0b050;--card:#1d1d21;--new:#f0b860}}
*{{box-sizing:border-box}} html,body{{margin:0;overflow-x:hidden}}
body{{background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,system-ui,sans-serif;padding:16px;max-width:820px;margin:auto}}
h1{{font-size:20px;margin:4px 0 8px}} h2{{font-size:17px;margin:28px 0 6px}}
.sum{{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--acc);border-radius:10px;padding:10px 12px}}
.sum p{{margin:5px 0}} .mut{{color:var(--mut);font-size:12.5px}} p{{margin:6px 0}}
.hd{{font-size:15px}} .sh{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 10px;font-size:14px}}
.dv{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px;margin:8px 0}}
.dh{{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:4px}}
.chg{{font-size:14px;margin:2px 0 4px;overflow-wrap:anywhere}} .new{{color:var(--new)}}
.tp{{font-size:12.5px;overflow-wrap:anywhere}}
.vd{{font-size:13.5px;margin:6px 0 0}} .vok b{{color:var(--ok)}} .vq b{{color:var(--q)}} .vx b{{color:var(--ko)}}
.play{{margin-left:auto;font:inherit;font-size:13px;padding:5px 12px;border-radius:14px;border:1px solid var(--acc);background:transparent;color:var(--acc);cursor:pointer}}
.play.on{{background:var(--acc);color:#fff}}
table{{border-collapse:collapse;width:100%}} td,th{{border-bottom:1px solid var(--line);padding:4px 3px;font-size:13px;text-align:left}}
th{{color:var(--mut);font-weight:600;font-size:12px}}
</style></head><body>
<h1>λ1 t+b sur trois autres morceaux</h1>
<p class="mut">2026-10-12. La même variante qu'Evening Star, sans aucun réglage : à chaque demi-mesure, on donne au décodeur la postérieure musx <i>plus</i> sa différence avec les deux demi-mesures voisines, seulement pour la triade et la basse (7e, 9e… inchangées). On compare à « λ0 » = la postérieure seule. Le chart actuel est montré pour repère ; ce n'est pas une vérité.</p>
<div class="sum">{''.join(f'<p><b>{n + 1}.</b> {t}</p>' for n, t in enumerate(SUMMARY))}</div>
<table><tr><th>morceau</th><th>changées</th><th>tenus gardés</th></tr>
<tr><td>Evening Star (rappel)</td><td>13 / 86</td><td>5 / 5</td></tr>
{''.join(f"<tr><td>{html.escape(s['title'])}</td><td>{len(s['changes'])} / {s['H']}</td><td>{s['held'][0]} / {s['held'][1]}</td></tr>" for s in songs)}
</table>
<p class="mut">Pour chaque changement : le top 3 du décodeur pour une trame (λ0 puis λ1 t+b) ; puis l'audio brut, indépendant de musx (chroma NNLS moyenné sur la demi-mesure) : les 4 notes les plus fortes, la basse la plus forte, et à quel point les notes de chaque accord collent (cosinus, 1 = parfait). Ce chiffre ne voit pas la basse : un renversement (G vs G/B) y fait toujours égalité, c'est la ligne « basse » qui tranche. Aucune grille iReal n'existe pour ces trois morceaux ; This Love a une tab Ultimate Guitar 4,8★ référencée mais son texte n'est pas en cache, donc pas de colonne de référence.</p>
{''.join(blocks)}
<h2>Ce que ça ne dit pas</h2>
<p>Trois morceaux, lectures du chroma seulement : chaque étiquette « plausible / à écouter / artefact » est une hypothèse pour ton oreille. Rien n'est branché dans le pipeline.</p>
<p class="mut">Scripts : tools/lambda1_tb_autres_morceaux.py (ce document) et tools/demi_mesures_delta_decodeur.py (le cœur commun, Evening Star inchangé).</p>
{auds}
<script>
(function(){{
  var SRC=[{srcs}], au=[], cur=null, stopAt=null, curAu=null;
  // iPhone : fetch()+blob (le chargeur média de la PWA ne bufferise pas), pas de boucle rAF : arrêt piloté par timeupdate
  SRC.forEach(function(src,k){{
    var a=document.getElementById('au'+k); au.push(a);
    fetch(src).then(function(r){{return r.blob();}}).then(function(b){{a.src=URL.createObjectURL(b);}}).catch(function(){{a.src=src;}});
    a.addEventListener('timeupdate',function(){{ if(curAu===a && stopAt!==null && a.currentTime>=stopAt){{a.pause();}} }});
    a.addEventListener('pause',function(){{ if(curAu===a){{ if(cur){{cur.classList.remove('on');cur=null;}} stopAt=null; curAu=null; }} }});
  }});
  document.querySelectorAll('.play').forEach(function(b){{
    b.addEventListener('click',function(ev){{
      ev.preventDefault(); ev.stopPropagation();
      if(cur===b){{curAu.pause();return;}}
      var a=au[+b.dataset.s];
      if(cur) cur.classList.remove('on');
      if(curAu && curAu!==a){{ var old=curAu; curAu=null; old.pause(); }}
      if(!a.src) a.src=SRC[+b.dataset.s];
      cur=b; curAu=a; b.classList.add('on');
      var t0=parseFloat(b.dataset.t0); stopAt=parseFloat(b.dataset.t1);
      try{{a.currentTime=t0;}}catch(e){{}}
      var p=a.play(); if(p&&p.catch) p.catch(function(){{}});
    }});
  }});
}})();
</script>
</body></html>"""
    OUT.write_text(page)
    print("écrit", OUT)


def _ext(stem):
    return next(SETTINGS.audio_dir.glob(f"{stem}.*")).suffix


def main():
    write_page(collect())


if __name__ == "__main__":
    main()
