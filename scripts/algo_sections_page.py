"""The section algorithm, written out in full — every step, every metric.

Louis has asked three times for "le détail exact de comment on arrive à nos
sections". The per-song pages (`sections_steps_<stem>.html`) show one song's
numbers; this one explains the ALGORITHM, independent of any song.

Constants are read from `harmonia_min.sections` at generation time rather than
typed in, so the page cannot drift away from the code it documents.

    python scripts/algo_sections_page.py   ->  /reports/algo_sections.html
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
from harmonia_min import sections as hs   # noqa: E402

OUT = HERE / "harmonia_min/state/reports/algo_sections.html"

C = {k: getattr(hs, k) for k in
     ("KERNEL_HB", "BLUR_SIGMA", "PEAK_FRAC", "MIN_SEG_BARS",
      "LABEL_COS", "TILE_MIN", "RUN_COVERAGE_MIN")}

STEPS = [
 ("0", "Ce qui entre", """
<p>Trois choses, toutes produites avant :</p>
<ul>
<li>la <b>grille de mesures</b> — les temps réels des barres de mesure, issus des
downbeats de Beat This! (pas d'un tempo constant synthétique) ;</li>
<li>le <b>chroma NNLS brut</b> (<code>bothchroma</code>) : 24 nombres par trame,
12 pour la basse et 12 pour l'aigu ;</li>
<li>les <b>accords décodés par mesure</b>, qui ne servent qu'aux règles de
cellule et de mesure tenue (étapes 6, 9, 12, 13) — jamais à la similarité.</li>
</ul>
<p class=k>Aucune étape ne regarde la tonalité, ni le tempo, ni les paroles.</p>"""),

 ("1", "Le substrat : une empreinte par DEMI-MESURE", """
<p>Chaque mesure est coupée en deux. Pour chaque moitié on fait la moyenne du
chroma sur sa durée, puis on normalise <b>séparément</b> la partie basse et la
partie aigu (norme 1 chacune), et on divise le tout par √2.</p>
<p>Résultat : une matrice <code>F</code> de <code>2 × nb_mesures</code> lignes et
24 colonnes.</p>
<p class=k>Pourquoi le chroma brut et pas les accords : en v1 la similarité était
construite sur les accords décodés, et tout un morceau en do mineur s'effondrait
sur « famille de do mineur » — la structure disparaissait. Le chroma brut garde
la texture (voicings, mouvement de basse, densité), qui est ce qui sépare un
couplet d'un refrain jouant les mêmes accords.</p>
<p class=k>Pourquoi la demi-mesure : c'est le grain le plus fin où une frontière
de section a encore un sens, et il permet de détecter un changement qui tombe au
milieu d'une mesure (étape 8 le ramène ensuite sur une mesure entière).</p>"""),

 ("2", "La matrice de similarité", """
<p><code>S = F · Fᵀ</code> — le cosinus entre chaque paire de demi-mesures, donc
une matrice carrée de côté <code>2 × nb_mesures</code>.</p>
<p>La diagonale vaut 1 par construction. Les <b>blocs hors diagonale</b> sont les
répétitions du morceau.</p>
<p class=k><b>Plus de flou.</b> <code>BLUR_SIGMA = %(BLUR_SIGMA)s</code> depuis le
2026-08-05 (il valait 1,5 demi-mesure). Mesuré sur 285 morceaux Billboard et
2781 vraies frontières : l'enlever fait passer les pics de nouveauté de 27,6 %
à 30,1 % de frontières posées sur la bonne mesure. Le flou brouillait la
<i>position</i> du pic qu'il était censé aider à trouver. En revanche il ne
gênait pas la séparation des lettres (AUC 0,886 flouté contre 0,893 brut) —
c'était l'hypothèse, elle est fausse.</p>"""),

 ("3", "Les runs de tuilage — première source de coupes", """
<p>On fabrique d'abord un vecteur par <b>mesure</b> (moyenne de ses deux
demi-mesures, renormalisée).</p>
<p>Puis, pour chaque mesure <i>b</i>, on cherche la <b>plus petite</b> période
<i>P</i> parmi <b>2, 4, 8</b> telle que</p>
<p class=f>cos(<i>b</i>, <i>b</i>+P) ≥ %(TILE_MIN)s &nbsp;<b>ou</b>&nbsp;
cos(<i>b</i>, <i>b</i>−P) ≥ %(TILE_MIN)s</p>
<p>autrement dit : « est-ce que je ressemble à la mesure P avant ou P après moi ? »
La mesure reçoit cette période, ou 0 si aucune ne marche.</p>
<p>On regroupe ensuite les mesures <b>consécutives portant la même période</b> en
un <i>run</i>, et on ne garde le run que s'il fait au moins <b>2×P</b> mesures
(deux tuiles complètes).</p>
<p><b>Les coupes sont les bords des runs</b> : pour chaque run, sa première
mesure et la mesure juste après sa dernière.</p>
<p class=k>Limites connues, mesurées : les runs ne couvrent que 53 à 74 % d'un
morceau, donc ils ne peuvent pas fournir toutes les frontières ; un run se coupe
quand la <b>période</b> change, pas quand la musique change ; et seuls 22 de nos
40 runs sont un multiple de leur propre période. Imposer ce multiple a été testé
le 2026-08-05 et <b>fait baisser</b> la qualité des frontières (F 0,232 → 0,213),
donc ce n'est pas fait.</p>"""),

 ("4", "La courbe de nouveauté — deuxième source de coupes", """
<p>On fait glisser le long de la diagonale un <b>noyau en damier</b> de
<code>%(KERNEL_HB)s</code> demi-mesures de demi-largeur, soit 8 mesures de
contexte de chaque côté. Il vaut +1 sur les deux carrés diagonaux (avant/avant et
après/après) et −1 sur les deux carrés anti-diagonaux (avant/après), le tout
pondéré par une gaussienne.</p>
<p class=f>nouveauté(i) = Σ S[voisinage de i] × noyau</p>
<p>Elle est donc grande quand « avant » se ressemble, « après » se ressemble, mais
avant ≠ après — la définition d'une frontière.</p>
<p><b>Sélection des pics.</b> On masque les bords (sur <code>%(KERNEL_HB)s</code>
demi-mesures de chaque côté), où le noyau est tronqué et ses valeurs sont des
artefacts. Puis on garde les maxima locaux (sur ±3 demi-mesures) qui atteignent</p>
<p class=f>max( %(PEAK_FRAC)s × le plus fort pic intérieur , moyenne + 0,5 × écart-type )</p>
<p class=k>Ce seuil est <b>adaptatif par morceau</b> : un morceau sans structure
n'aura pas de pics artificiels, un morceau très structuré n'en perdra pas.</p>"""),

 ("5", "L'union des deux sources", """
<p>Les deux listes de coupes sont <b>additionnées</b>, pas choisies l'une ou
l'autre.</p>
<p class=k><b>C'est le changement du 2026-08-05</b> (demande de Louis : « les pics
recouvrent très bien, ils devraient corriger les tuilages, pas l'un ou l'autre
mais les deux ensemble »). Avant, le code choisissait : bords de runs si les runs
couvraient au moins <code>%(RUN_COVERAGE_MIN)s</code> du morceau, nouveauté sinon
— et il <b>supprimait</b> tous les pics tombant à l'intérieur d'un run. Mesuré :
runs seuls 35,8 % de frontières justes, pics seuls 30,1 %, les deux 41,8 %. Les
pics supprimés valaient à eux seuls 4,9 points de qualité de frontière.</p>"""),

 ("6", "Ramener un pic sur une mesure — et les cellules", """
<p>Un pic vit sur la demi-mesure <i>h</i>. On calcule <code>base = (h+1)/2</code>,
puis on choisit entre les <b>deux mesures paires</b> qui l'encadrent.</p>
<p>Pourquoi paires : les sections sont des multiples de 2 mesures, et la grille
est ancrée à la mesure 0.</p>
<p><b>Une coupe ne peut pas casser une cellule.</b> Une <i>cellule</i> est une
paire de mesures dont la signature d'accords revient au moins 2 fois dans le
morceau, avec au moins deux occurrences distantes de 4 mesures ou moins.</p>
<p class=k>La condition de proximité est essentielle : une <i>transition</i>
récurrente (fin de couplet → début de refrain) revient elle aussi, mais tous les
20 mesures. La traiter comme une cellule interdisait la vraie coupe de section et
avalait des refrains entiers — mesuré.</p>
<p>Entre les deux mesures paires candidates, on prend la plus proche de
<code>base</code> ; à égalité, celle où la nouveauté est la plus forte.</p>"""),

 ("7", "Des coupes aux segments", """
<p>La liste de coupes triée, plus 0 au début et le nombre de mesures à la fin,
découpe le morceau en segments contigus qui le couvrent entièrement.</p>
<p>Un segment fait au minimum <code>%(MIN_SEG_BARS)s</code> mesures.</p>"""),

 ("8", "Les orphelins rejoignent la section de gauche", """
<p>Un segment que <b>ne couvre aucun run</b> et qui est plus court que la période
du run précédent est recollé au segment de gauche.</p>
<p class=k>C'est de la matière de queue : la tenue après un refrain, les 2 mesures
de fin de couplet qui varient d'une passe à l'autre et tombent donc hors du
tuilage. Ce n'est pas une section.</p>"""),

 ("9", "Une section ne s'ouvre pas sur une résonance", """
<p>Si une section commence par le couple (attaque, mesure tenue) et qu'elle fait
plus de 4 mesures, elle démarre 2 mesures plus loin, et la section précédente
absorbe sa propre queue.</p>
<p class=k>Ce couple est la cadence de la phrase qui se termine, plus sa
résonance — pas le début de la suivante.</p>"""),

 ("10", "Les lettres", """
<p>Pour deux segments <i>i</i> et <i>j</i>, on prend la moyenne du <b>bloc
croisé</b> de la matrice de similarité (lignes de <i>i</i> × colonnes de <i>j</i>)
et on la normalise par leurs deux blocs internes :</p>
<p class=f>r(i,j) = M[i,j] / √( M[i,i] × M[j,j] )</p>
<p>Deux segments partagent une lettre si <code>r > %(LABEL_COS)s</code>. Le
regroupement est glouton : chaque segment rejoint le premier groupe dont la
moyenne des <i>r</i> le dépasse, sinon il ouvre son propre groupe.</p>
<p class=k>Pourquoi un ratio et pas un cosinus de moyennes : la moyenne de chroma
par segment échouait, tout un morceau en do mineur se ressemblant, le morceau
entier recevait une seule lettre. Le ratio compare « à quel point i ressemble à
j » à « à quel point i se ressemble à lui-même », ce qui discrimine bien mieux.</p>
<p class=warn><b>Défaut connu, non corrigé.</b> Ce ratio est calculé sur du chroma
absolu, donc il <b>n'est pas invariant par transposition</b>. Un morceau qui
module voit ses reprises éclatées en lettres neuves. Mesuré sur <i>Sunny</i> :
les sections F et G sont la même à un demi-ton près, r = 0,797 tel quel et 0,973
après rotation, contre un seuil de %(LABEL_COS)s. Prendre le meilleur des 12
rotations a été mesuré comme sûr (F 0,646 → 0,662) mais n'est pas encore
livré.</p>"""),

 ("11", "La mesure tenue frontalière", """
<p>Une mesure tenue qui <b>termine</b> une section, mais dont l'accord qui sonne
est celui sur lequel <b>ouvrent</b> les sections sœurs de la suivante, bascule
pour ouvrir la suivante.</p>
<p class=k>Cas réel : le sol tenu de This Love est le sol/si sur lequel ouvre A1,
donc il ouvre A2 et A3. Contre-exemple gardé correct : le la♭ tenu de She Will Be
Loved ne correspond à aucune ouverture de couplet, il reste avec son refrain.</p>"""),

 ("12", "Le garde-fou d'accord entre sœurs", """
<p>Pour chaque lettre, si l'ouverture d'un membre n'est pas <b>strictement égale</b>
à celle d'une sœur sur au moins 2 mesures, on essaie de la décaler de
<b>±2 mesures</b>, en respectant les cellules et les voisins.</p>
<p>S'il reste du désaccord, un avertissement part dans les logs :
<code>letter X members agree only 0.xx on their openings — cuts suspect</code>.</p>
<p class=warn><b>Limite structurelle mesurée.</b> Ce garde-fou ne peut corriger
que ±2 mesures, alors que <b>64 %</b> des paires d'occurrences d'une même lettre
demandent un décalage de 1 à 4 mesures. Il ne peut donc pas rattraper les erreurs
de 3 et 4 mesures.</p>"""),

 ("13", "Fusion des lettres jamais répétées", """
<p>Deux sections <b>adjacentes</b> jouées chacune une seule fois, dont au moins
une fait moins de 8 mesures, fusionnent.</p>
<p class=k>Un passage unique en deux morceaux est une seule section — le pont de
This Love, 6 + 2 mesures, est un pont de 8. Garde-fou mesuré sur Close to You :
deux passages uniques de 8 mesures ou plus sont de vraies sections autonomes
(ses deux moitiés dans deux tonalités), la fusion brute avalait sa modulation.</p>"""),
]

CONSTS = [
 ("TILE_MIN", C["TILE_MIN"], "similarité minimale pour qu'une mesure « tuile » à la période P (étape 3)"),
 ("RUN_COVERAGE_MIN", C["RUN_COVERAGE_MIN"], "ancien seuil de bascule entre runs et nouveauté — <b>plus utilisé pour choisir</b> depuis l'union (étape 5), seulement journalisé"),
 ("KERNEL_HB", C["KERNEL_HB"], "demi-largeur du damier en demi-mesures, soit 8 mesures de contexte (étape 4)"),
 ("BLUR_SIGMA", C["BLUR_SIGMA"], "flou gaussien sur la SSM — <b>désactivé</b> le 2026-08-05, valait 1,5"),
 ("PEAK_FRAC", C["PEAK_FRAC"], "fraction du plus fort pic qu'un pic doit atteindre (étape 4)"),
 ("MIN_SEG_BARS", C["MIN_SEG_BARS"], "longueur minimale d'un segment (étape 7)"),
 ("LABEL_COS", C["LABEL_COS"], "ratio croisé/interne au-dessus duquel deux segments partagent une lettre (étape 10)"),
]


def main():
    def fill(body):
        # str.replace, not %-formatting: the prose is full of literal "%"
        # (percentages) and %-formatting choked on them.
        for k, v in C.items():
            body = body.replace(f"%({k})s", str(v))
        return body

    steps = "".join(
        f'<section><h2><span class=n>{n}</span>{t}</h2>{fill(b)}</section>'
        for n, t, b in STEPS)
    crows = "".join(
        f"<tr><td><code>{a}</code></td><td><b>{v}</b></td><td>{d}</td></tr>"
        for a, v, d in CONSTS)
    OUT.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Comment on détecte les sections — l'algorithme complet</title><style>
*{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
body{{margin:0;background:#e7e0d0;font:16px/1.6 -apple-system,BlinkMacSystemFont,system-ui,sans-serif;color:#1c1c1c}}
.wrap{{max-width:720px;margin:0 auto;padding:22px 15px calc(50px + env(safe-area-inset-bottom))}}
h1{{font:italic 600 25px Georgia,serif;margin:0 0 4px}}
.lede{{color:#8a8371;font-size:13px;margin-bottom:22px}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:15px 17px;margin-bottom:13px}}
h2{{font:700 15px system-ui;margin:0 0 10px;color:#8a2b2b;display:flex;align-items:baseline;gap:9px}}
.n{{flex:0 0 auto;width:25px;height:25px;border-radius:50%;background:#8a2b2b;color:#fff;
   display:inline-flex;align-items:center;justify-content:center;font:800 12px system-ui}}
p{{margin:0 0 9px}} ul{{margin:0 0 9px;padding-left:20px}} li{{margin-bottom:4px}}
code{{background:#f7f3e9;padding:1px 5px;border-radius:4px;font-size:13px}}
.f{{background:#2a2622;color:#f4eee2;padding:9px 13px;border-radius:9px;font:600 14px system-ui;text-align:center}}
.k{{background:#f7f3e9;border-left:3px solid #1f8a5b;padding:8px 12px;border-radius:0 8px 8px 0;font-size:13.5px}}
.warn{{background:#f7f3e9;border-left:3px solid #c58a2e;padding:8px 12px;border-radius:0 8px 8px 0;font-size:13.5px}}
table{{border-collapse:collapse;font-size:13px;width:100%}}
th,td{{border:1px solid #e5dcc6;padding:5px 9px;text-align:left;vertical-align:top}}
th{{background:#f7f3e9}}
</style></head><body><div class=wrap>
<h1>Comment on détecte les sections</h1>
<div class=lede>L'algorithme complet, étape par étape, indépendamment d'un morceau.
Les constantes sont lues dans le code au moment de générer la page, elles ne
peuvent donc pas être périmées. Pour les chiffres d'un morceau précis, voir les
pages « toutes les étapes » de l'index.</div>
{steps}
<section><h2><span class=n>C</span>Toutes les constantes</h2>
<table><tr><th>nom</th><th>valeur</th><th>rôle</th></tr>{crows}</table></section>
<section><h2><span class=n>!</span>Ce que cet algorithme ne fait PAS</h2>
<ul>
<li>Il n'est <b>pas invariant par transposition</b> (étape 10) — tout morceau qui
module voit ses reprises éclatées.</li>
<li>Il ne corrige les décalages d'ouverture que de <b>±2 mesures</b> (étape 12),
alors que 64 % des paires en demandent 1 à 4.</li>
<li>Il ne force <b>pas</b> les occurrences d'une même lettre à avoir la même
longueur — d'où des lettres portant des occurrences de 8, 6, 12, 12, 16 et 8
mesures.</li>
<li>Il ne produit <b>pas</b> de chaîne de forme (AABA) : il produit une bande
d'étiquettes, la forme est lue après coup.</li>
<li>Ses seuils ont été calés sur <b>trois morceaux</b> (This Love, Let It Be,
Close to You), ce que le docstring du module qualifie lui-même d'hypothèse.</li>
</ul></section>
</div></body></html>""")
    print(f"wrote {OUT.relative_to(HERE)} ({OUT.stat().st_size // 1024} KB)")
    print("  constantes lues:", C)


if __name__ == "__main__":
    main()
