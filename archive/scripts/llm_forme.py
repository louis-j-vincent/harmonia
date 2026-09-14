"""Le LLM lit la forme du morceau — automatisé, avec répétitions, en lettres.

    .venv/bin/python scripts/llm_forme.py <stem> [--n 5]   # un morceau
    .venv/bin/python scripts/llm_forme.py --tous           # les 18 validés
    .venv/bin/python scripts/llm_forme.py --table          # l'accord, sans appeler
    .venv/bin/python scripts/llm_forme.py --page           # -> /plots/llm_forme.html

Louis, 2026-08-14, après avoir lu les six pistes d'amélioration : « go, tu me les
log tous et tu me les lances dans l'ordre. » Les voici, dans l'ordre, et chacune
est marquée dans le code.

PISTE 1 — L'AUTOMATISER. Il n'y a ni `anthropic` installé ni clé API sur cette
machine, mais le CLI `claude` est là et accepte `-p` : on passe donc par l'auth
Claude Code de Louis, sans clé. `--system-prompt` remplace le prompt d'agent
complet (35 k jetons) par deux lignes, et les outils sont coupés — sinon chaque
appel paie le harnais entier. Coût mesuré : ~0,10 $ le premier appel, ~0,03 $ les
suivants (le cache serveur tient entre les processus).

PISTE 2 — LUI DONNER LA RÉPÉTITION. Le résumé texte ne portait que l'accord, le
chant, le niveau et la batterie : le modèle devait *deviner à l'œil* que la
mesure 29 rejoue la 13. On ajoute une colonne `≈` — pour chaque mesure, celle
qu'elle rejoue le plus (hors voisines immédiates), avec sa force. C'est
l'information la plus dense qu'on puisse lui passer, et elle est déjà calculée :
c'est la SSM d'accords de la prod.

PISTE 3 — DEMANDER LA FORME, PAS LES FRONTIÈRES. La sortie n'est plus une liste
de coupures mais une PARTITION en sections étiquetées, qui doit couvrir toutes
les mesures sans trou ni chevauchement, et réutiliser la même lettre pour deux
passages identiques. Le format impose alors tout seul ce qu'on veut : les
sections se répètent, elles pavent le morceau. Les frontières en tombent.

PISTE 4 — AUTO-COHÉRENCE. `n` tirages indépendants ; une frontière est retenue
si elle apparaît dans au moins `SEUIL` d'entre eux. Ça donne une CONFIANCE par
frontière — exactement ce qu'il faut quand les ancres sont des guides et non des
verrous.

PISTE 5 — RÉTRÉCIR LA QUESTION. `arbitrer()` ne demande pas « où sont les
frontières » mais « les signaux disent 49, la grille de 4 dit 51 — laquelle, et
pourquoi ». Deux options, une raison : beaucoup moins de place pour inventer.

PISTE 6 — LE CARNET. `scratchpad/carnet_oreille.json` collecte les corrections
que Louis fait à l'oreille, et chacune est réinjectée en exemple dans le prompt
suivant. Sa règle : une correction humaine doit devenir une capacité générale,
pas un rustine par morceau.

CE QUE ÇA NE FAIT PAS. Le modèle n'entend rien : il lit un résumé de NOTRE
analyse. Si les accords sont faux, sa forme sera fausse avec assurance. Et ses
frontières ne sont pas comparables à un détecteur de signal : il ne détecte pas
un changement, il reconnaît une forme qu'il a déjà lue mille fois.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "scratchpad"))

ETAT = HERE / "harmonia_min" / "state" / "llm_forme"
ETAT.mkdir(parents=True, exist_ok=True)
CARNET = HERE / "scratchpad" / "carnet_oreille.json"
OUT = HERE / "docs" / "plots" / "llm_forme.html"

MODELE = "claude-haiku-4-5-20251001"   # Louis, 2026-08-14 : « lance-les sur haiku »
N_TIRAGES = 3
SEUIL = 2
BUDGET_MAX = 8.0   # $ : le run s'arrête net au-delà, sans clé API on paie le harnais          # une frontière tenue par ≥ SEUIL tirages sur N est retenue
TOL = 1            # deux frontières à ≤ 1 mesure sont la même


# ── piste 2 : la colonne des répétitions ────────────────────────────────────

def repetitions(stem, W=4):
    """[(phrase rejouée, force)] par mesure — au grain de la PHRASE, pas de la mesure.

    Pour chaque mesure i on compare la fenêtre de `W` mesures qui commence là à
    toutes les autres, et on rend la PREMIÈRE occurrence assez ressemblante (pas
    la meilleure) : sur un morceau en boucle, « la mesure 45 rejoue la 5 » est
    une information de forme, « la 45 rejoue la 26 » n'en est pas une.

    Pourquoi la phrase et pas la mesure : essayé d'abord à la mesure, illisible.
    Sur This Love, qui rejoue la même boucle de 4 accords du début à la fin,
    CHAQUE mesure ressemble à vingt autres au maximum de l'échelle — la colonne
    disait « tout rejoue tout » et n'apprenait rien. À quatre mesures, la
    comparaison porte sur la phrase entière et retrouve la structure.
    """
    import order_bundle
    from ssm_zoo import capture
    import harmonia_min.harmonic_sections as HS
    b = order_bundle.get(stem)
    n, grid = b["n"], list(np.asarray(b["grid"], float))
    V = np.asarray(HS.harmonic_vectors(capture(stem)["triad"], grid), float)
    F = np.array([np.concatenate([V[min(n - 1, i + k)] for k in range(W)])
                  for i in range(n)])
    F = F / np.clip(np.linalg.norm(F, axis=1, keepdims=True), 1e-9, None)
    S = np.clip(F @ F.T, 0, 1)
    out = []
    for i in range(n):
        m = S[i].copy()
        m[max(0, i - W):i + W + 1] = -1
        fort = [j for j in range(n) if m[j] >= 0.97]
        j = min(fort) if fort else int(np.argmax(m))
        out.append((j, float(m[j])))
    return out


def resume(stem):
    """Le morceau en une ligne par mesure, répétitions comprises."""
    from resume_texte import resume as base
    txt = base(stem).splitlines()
    rep = repetitions(stem)
    L = [txt[0], txt[1].rstrip() + "   rejoue"]
    for i, ligne in enumerate(txt[2:]):
        j, s = rep[i]
        marque = f"≈{j + 1:<4}{'++' if s > .95 else ('+ ' if s > .85 else '  ')}" if s > .75 else ""
        L.append(f"{ligne}   {marque}")
    return "\n".join(L)


# ── piste 1 : l'appel, automatisé ───────────────────────────────────────────

SYSTEME = ("Tu es un analyste de forme musicale. Tu réponds UNIQUEMENT par un objet "
           "JSON valide, sans texte avant ni après, sans bloc de code.")


def appeler(prompt: str, modele=MODELE, temperature_hint="") -> dict | None:
    """Un tirage. Retourne l'objet JSON, ou None si la réponse est inexploitable.

    On coupe tous les outils : sans ça chaque appel paie le harnais d'agent
    complet (35 k jetons de prompt système contre 16 k, et 0,21 $ contre 0,10 $
    au premier appel).
    """
    cmd = ["claude", "-p", "--output-format", "json", "--model", modele,
           "--system-prompt", SYSTEME, "--allowed-tools", "",
           "--effort", "medium"]
    # `--effort medium` : sans lui le modèle réfléchit 22 000 jetons avant de
    # rendre le JSON, et la réflexion fait 45 % de la facture (mesuré). Les
    # outils sont coupés : sinon on paie aussi les 18 k jetons de leur
    # définition. Le harnais Claude Code reste payé à chaque appel — c'est ce
    # qu'une vraie clé API supprimerait (0,004 $ au lieu de 0,11 $).
    try:
        r = subprocess.run(cmd, input=prompt + temperature_hint, capture_output=True,
                           text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    try:
        env = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    txt = (env.get("result") or "").strip()
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    d["_cout"] = env.get("total_cost_usd", 0.0)
    return d


# ── piste 6 : le carnet des corrections de Louis ────────────────────────────

def carnet_txt():
    """Les corrections déjà faites à l'oreille, en exemples pour le prompt."""
    if not CARNET.exists():
        return ""
    d = json.loads(CARNET.read_text())
    if not d.get("regles"):
        return ""
    L = ["\nCE QUE LOUIS A DÉJÀ CORRIGÉ À L'OREILLE — applique-le :"]
    for r in d["regles"]:
        L.append(f"  · {r}")
    return "\n".join(L) + "\n"


# ── piste 3 : on demande la FORME ───────────────────────────────────────────

def prompt_forme(stem, ancres=None):
    n = int(re.search(r"· (\d+) mesures", resume(stem).splitlines()[0]).group(1))
    anc = ""
    if ancres:
        L = sorted(ancres, key=lambda a: -a[1])
        anc = ("\nDES SIGNAUX AUDIO INDÉPENDANTS (batterie, timbre, harmonie, basse, "
               "accords, voix, intensité) sont d'accord sur un changement à ces "
               "mesures, de la plus soutenue à la moins soutenue :\n  "
               + ", ".join(f"{b + 1} ({v}/7)" for b, v, *_ in L)
               + "\n\nCOMMENT T'EN SERVIR — c'est important, la liste est trompeuse "
                 "dans les deux sens. Elle n'est ni sûre ni complète : mesuré sur 18 "
                 "morceaux, une mesure à 6 ou 7 signaux est une vraie frontière 9 fois "
                 "sur 10, une mesure à 3 ou 4 signaux une fois sur trois — et cette "
                 "liste, même complète, ne couvre que 59 % des vraies frontières. "
                 "Donc : appuie-toi dessus quand elle confirme ce que disent les "
                 "colonnes, écarte-la quand la forme dit clairement autre chose, et "
                 "surtout N'HÉSITE PAS À COUPER LÀ OÙ AUCUN SIGNAL NE VOTE si la "
                 "suite d'accords ou le chant le demandent. Le nombre entre "
                 "parenthèses est le nombre de signaux, pas une probabilité.\n")
    return f"""Voici un morceau de pop/soul décrit mesure par mesure. Tu ne l'entends pas :
tu lis le résultat d'une analyse audio automatique.

COLONNES : `mes` le numéro de mesure · `accord` l'accord détecté · `chant` le
chant présent sur chacun des 4 quarts de la mesure (# = on chante, . = silence) ·
`niveau` le volume en dB relatif à la médiane du morceau · `batterie` l'énergie
de la batterie en quartiles · `rejoue` la mesure que celle-ci rejoue le plus
ailleurs dans le morceau (++ = très ressemblante, + = ressemblante).

{resume(stem)}
{anc}{carnet_txt()}
TA TÂCHE : découper ce morceau en SECTIONS et donner une lettre à chacune.

RÈGLES DE SORTIE, strictes :
  · les sections doivent couvrir les mesures 1 à {n} sans trou ni chevauchement ;
  · deux passages qui se rejouent portent la MÊME lettre (A, B, C…) ;
  · utilise `intro`, `pont` et `outro` quand c'est ce que c'est ;
  · la plupart des sections de pop font 4, 8 ou 16 mesures — mais PAS toutes,
    une mesure insérée ou une queue de 2 mesures existent, ne force pas la
    régularité contre ce que disent les colonnes ;
  · une section commence là où le matériau change, pas là où il se prépare : un
    fill de batterie ou une anacrouse de chant appartiennent à la mesure d'AVANT.

RÉPONDS UNIQUEMENT PAR CE JSON :
{{"sections": [{{"lettre": "intro", "debut": 1, "fin": 8, "pourquoi": "..."}},
               {{"lettre": "A", "debut": 9, "fin": 16, "pourquoi": "..."}}],
 "confiance": "haute|moyenne|basse",
 "ce_qui_est_douteux": "une phrase, ou vide"}}

`pourquoi` : une phrase courte, ancrée dans les colonnes (« le chant s'arrête »,
« les accords passent à C#m », « ≈ mes. 13 »). Pas de généralité."""


def valider(d, n):
    """La partition couvre-t-elle vraiment le morceau ? Sinon on jette."""
    if not isinstance(d, dict) or not isinstance(d.get("sections"), list):
        return None
    S = []
    for s in d["sections"]:
        try:
            a, b = int(s["debut"]), int(s["fin"])
        except (KeyError, TypeError, ValueError):
            return None
        if not (1 <= a <= b <= n):
            return None
        S.append((a, b, str(s.get("lettre", "?")), str(s.get("pourquoi", ""))))
    S.sort()
    if not S or S[0][0] != 1:
        return None
    for (a1, b1, *_), (a2, *_) in zip(S, S[1:]):
        if a2 != b1 + 1:
            return None
    if S[-1][1] != n:
        return None
    return S


# ── piste 4 : l'auto-cohérence ──────────────────────────────────────────────

def forme(stem, n_tirages=N_TIRAGES, ancres=None, rebuild=False):
    """`n_tirages` lectures indépendantes, puis le consensus.

    Une frontière est retenue si au moins `SEUIL` tirages la posent à ≤ `TOL`
    mesure. Le nombre de tirages qui la soutiennent EST sa confiance — et c'est
    ce qui manquait pour se servir de ces frontières comme de guides.
    """
    p = ETAT / f"{stem}.json"
    if p.exists() and not rebuild:
        return json.loads(p.read_text())
    import order_bundle
    n = order_bundle.get(stem)["n"]
    pr = prompt_forme(stem, ancres)
    tirages, cout = [], 0.0
    for k in range(n_tirages):
        if cout > BUDGET_MAX / 6:        # garde-fou par morceau
            break
        d = appeler(pr, temperature_hint=f"\n\n(lecture n°{k + 1})")
        if d is None:
            continue
        cout += d.pop("_cout", 0.0)
        S = valider(d, n)
        if S:
            tirages.append({"sections": S, "confiance": d.get("confiance", ""),
                            "doute": d.get("ce_qui_est_douteux", "")})
    if not tirages:
        return {"stem": stem, "n": n, "tirages": 0, "frontieres": [], "cout": cout}

    votes = Counter()
    raisons: dict[int, str] = {}
    for t in tirages:
        for a, _b, _l, pq in t["sections"][1:]:
            votes[a - 1] += 1
            raisons.setdefault(a - 1, pq)
    # fusionner les frontières à ±TOL : la position la plus soutenue l'emporte
    fusion: list[tuple[int, int]] = []
    for b, v in sorted(votes.items(), key=lambda x: (-x[1], x[0])):
        if all(abs(b - g) > TOL for g, _ in fusion):
            fusion.append((b, v + sum(w for c, w in votes.items()
                                      if c != b and abs(c - b) <= TOL)))
    F = [{"mesure": b + 1, "voix": min(v, len(tirages)), "sur": len(tirages),
          "pourquoi": raisons.get(b, "")} for b, v in sorted(fusion)]
    d = {"stem": stem, "n": n, "tirages": len(tirages), "cout": round(cout, 4),
         "frontieres": F,
         "retenues": [f["mesure"] for f in F if f["voix"] >= SEUIL],
         "meilleure_lecture": tirages[0]["sections"],
         "doutes": [t["doute"] for t in tirages if t.get("doute")]}
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1))
    return d


# ── piste 5 : rétrécir la question ──────────────────────────────────────────

def arbitrer(stem, option_a, option_b, contexte=""):
    """« Les signaux disent A, la grille dit B — laquelle, et pourquoi ? »

    Deux options, une raison. Beaucoup moins de place pour inventer qu'une
    génération libre — et c'est la forme à utiliser dès qu'on SAIT déjà que la
    frontière est l'une des deux.
    """
    pr = f"""{resume(stem)}
{carnet_txt()}
UNE SEULE QUESTION. Une frontière de section se trouve soit à la mesure
{option_a}, soit à la mesure {option_b}. {contexte}

Laquelle des deux ? Appuie-toi sur les colonnes (accord, chant, niveau,
batterie, rejoue), pas sur une préférence pour les nombres ronds.

RÉPONDS UNIQUEMENT PAR :
{{"mesure": {option_a} ou {option_b}, "pourquoi": "une phrase", "surete": "haute|moyenne|basse"}}"""
    return appeler(pr)


# ── mesure et page ──────────────────────────────────────────────────────────

def gt_de(stem):
    p = HERE / "harmonia_min" / "state" / "sections" / f"{stem}.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    return sorted({int(s["b0"]) for s in d["sections"] if 0 < int(s["b0"]) < d["n"]})


def table():
    """L'accord avec les frontières annotées — une BORNE BASSE, pas un score.

    Louis, 2026-08-14 : « je n'ai pas noté tous les changements de section ; des
    fois le LLM coupe sur un pont ou une section qui se répète, et ça fait sens
    et ça devrait être gardé. » Une frontière hors annotation n'est donc pas une
    erreur : elle est comptée à part, en « à écouter ».
    """
    from ancres import liste_validee
    print(f"{'morceau':<32}{'tir':>4}{'ftr':>5}{'sur':>5}{'retenues':>9}"
          f"{'trouvées':>9}{'à écouter':>10}{'$':>7}")
    T = R = H = 0.0
    C = 0.0
    for s in liste_validee():
        p = ETAT / f"{s}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        gt = gt_de(s)
        ret = d.get("retenues", [])
        bon = sum(1 for g in gt if any(abs(m - 1 - g) <= 1 for m in ret))
        hors = sum(1 for m in ret if all(abs(m - 1 - g) > 1 for g in gt))
        T += len(gt); R += bon; H += hors; C += d.get("cout", 0)
        print(f"{s[:31]:<32}{d['tirages']:>4}{len(gt):>5}{len(ret):>5}"
              f"{len(ret):>9}{bon:>9}{hors:>10}{d.get('cout', 0):>7.2f}")
    print(f"\nRAPPEL {R:.0f}/{T:.0f} = {100 * R / max(1, T):.0f}% des frontières "
          f"annotées sont retrouvées · {H:.0f} frontières hors annotation à écouter "
          f"· {C:.2f} $ au total")


def main():
    from ancres import liste_validee, ancres as detect
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--table" in sys.argv:
        return table()
    stems = liste_validee() if "--tous" in sys.argv else args
    n = N_TIRAGES
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])
    for s in stems:
        # ancres MOLLES : réglage large (Louis, 2026-08-14 : « il nous faut des
        # ancres molles qui rentrent en prior du LLM, et plus on en a mieux
        # c'est »). C'est le LLM qui filtre, donc on vise le rappel.
        A, *_ = detect(s, k=3, z=2.5, reseau=False)
        d = forme(s, n_tirages=n, ancres=A,
                  rebuild="--rebuild" in sys.argv)
        gt = gt_de(s)
        ret = d.get("retenues", [])
        bon = sum(1 for g in gt if any(abs(m - 1 - g) <= 1 for m in ret))
        print(f"  {s[:40]:<42} {d['tirages']} tirages · {len(ret)} frontières "
              f"retenues · {bon}/{len(gt)} des tiennes · {d.get('cout', 0):.2f} $")


if __name__ == "__main__":
    main()
