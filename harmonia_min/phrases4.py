"""harmonia_min/phrases4.py — l'algo des phrases à quatre mots, en pur.

Ces trois fonctions VIENNENT de `scripts/quatre_mots.py` et y sont restées
jusqu'au 2026-08-14. Elles déménagent ici sans changer d'une ligne, pour une
raison précise : Louis veut le bouton « Appliquer » dans l'outil Soudure
(« il faut un bouton appliquer qui re-infère les sections via l'algo des
phrases à 4 mots, mais avec nos sections déjà fixées »), donc le serveur doit
pouvoir les appeler — et `scripts/quatre_mots.py` tire `ssm_zoo`,
`order_bundle`, `matplotlib` et le reste du fourbi de recherche rien qu'à
l'import. En recopier une version « allégée » aurait fait deux algorithmes
pour une même question, celui de la page de recherche et celui de l'app, qui
auraient divergé au premier réglage. `scripts/quatre_mots.py` les importe
maintenant d'ici.

LES DEUX RÈGLES SONT DE LOUIS (2026-08-12, en majuscules dans son message) :

  1. **L'arrêt par la taille.** On soude la paire la plus fréquente, mais
     jamais au-delà de `cible` bi-mesures, et on s'arrête dès qu'aucune paire
     soudable ne se répète — « quand on n'arrive plus à trouver de sections de
     4 mots bi-barres ». C'est un critère de TAILLE, pas de contenu, et c'est
     ce qui manquait : l'agglomération libre ne savait pas s'arrêter et
     finissait par coller couplet et refrain.

  2. **La cadence, pas la phrase.** `abac` et `abad` sont le même A : trois
     bi-mesures identiques et une fin qui change — la première et la deuxième
     cadence d'une même période, l'ouvert et le clos. On écrit `A` et `A′`.
     Sans cette règle, la seconde prenait une lettre neuve et le morceau
     paraissait avoir deux fois plus de sections qu'il n'en a.
"""
from __future__ import annotations

CIBLES = (4, 6)    # une section fait quatre mots de deux mesures… ou six
CIBLE = 4
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def merges4(word, cible=CIBLE, max_steps=24, depart=None, geles=None):
    """[{paire, compte, jetons}] — l'agglomération plafonnée à `cible` mots.

    Identique à `bpe_lab.merges` sauf deux choses, qui sont les règles de
    Louis : une soudure ne peut pas dépasser `cible` bi-mesures, et on
    s'arrête dès qu'aucune paire soudable ne se répète.

    `depart` (2026-08-14) : les jetons DÉJÀ soudés à la main, sous la forme
    [(j0, j1, contenu)]. C'est ce que le bouton « Appliquer » de l'outil
    Soudure envoie — l'algorithme reprend alors le travail là où Louis l'a
    laissé au lieu de repartir des lettres nues, et ses soudures à lui ne
    peuvent plus être défaites, seulement prolongées.

    `geles` (2026-08-17) : parmi ces jetons de départ, ceux qui sont des
    sections ENTIÈRES et non des morceaux à prolonger — un ensemble de
    (j0, j1). Un jeton gelé ne fusionne avec personne, dans aucun sens.

    Les deux outils ne veulent pas la même chose du même geste, et c'est
    pourquoi le gel est un paramètre plutôt qu'une règle. SOUDURE recolle des
    bi-mesures : sa soudure est un bout de section, la prolonger est le
    travail. SECTIONS trace une section complète : la prolonger, c'est
    l'effacer. Sur Let It Be, sans gel, le couplet de 4 mots et le refrain de
    2 mots que Louis venait de tracer tenaient ensemble sous `cible=6` — ils
    fusionnaient, sa frontière disparaissait, et le bloc soudé ne
    correspondant plus à aucune de ses sections repartait sous une lettre de
    la machine : 15 sections envoyées, 1 rendue sous son nom.
    """
    toks = list(depart) if depart else [(j, j + 1, word[j]) for j in range(len(word))]
    fige = set(geles or ())
    steps = [{"paire": None, "compte": 0, "jetons": list(toks)}]
    for _ in range(max_steps):
        cnt: dict = {}
        for i in range(len(toks) - 1):
            if (toks[i][0], toks[i][1]) in fige or (toks[i + 1][0], toks[i + 1][1]) in fige:
                continue
            a, b = toks[i][2], toks[i + 1][2]
            if len(a) + len(b) > cible:
                continue
            cnt[(a, b)] = cnt.get((a, b), 0) + 1
        cnt = {k: v for k, v in cnt.items() if v >= 2}
        if not cnt:
            break
        (pa, pb), c = max(cnt.items(),
                          key=lambda kv: (kv[1], len(kv[0][0]) + len(kv[0][1])))
        out, i = [], 0
        while i < len(toks):
            if (i + 1 < len(toks) and toks[i][2] == pa and toks[i + 1][2] == pb
                    and (toks[i][0], toks[i][1]) not in fige
                    and (toks[i + 1][0], toks[i + 1][1]) not in fige):
                out.append((toks[i][0], toks[i + 1][1], pa + pb))
                i += 2
            else:
                out.append(toks[i])
                i += 1
        toks = out
        steps.append({"paire": (pa, pb), "compte": c, "jetons": list(toks)})
    return steps


def nommer(toks, cible=CIBLE, geles=None):
    """[{j0, j1, type, label, prime}] — les lettres, avec la règle du prime.

    Deux sections de même longueur qui ne diffèrent QUE par leur dernier mot
    sont la même lettre, la seconde marquée d'un prime.

    `geles` : un jeton tracé à la main n'est JAMAIS une queue, même court.
    Sans ça, l'intro de 2 mots que Louis vient de tracer repartait dans
    `grouper_restes`, collée à sa voisine — le gel de `merges4` l'aurait
    sauvée de la soudure pour la perdre au regroupement suivant.
    """
    fige = set(geles or ())
    base: list[tuple[str, str]] = []          # (type de référence, lettre)
    out = []
    for j0, j1, t in toks:
        lab, prime = None, False
        for u, L in base:
            if u == t:
                lab = L
                break
            if len(u) == len(t) and u[:-1] == t[:-1] and len(t) >= 2:
                lab, prime = L, True
                break
        if lab is None:
            lab = LETTERS[len({L for _u, L in base}) % len(LETTERS)]
            base.append((t, lab))
        out.append({"j0": j0, "j1": j1, "type": t, "label": lab,
                    "prime": prime,
                    "queue": (j1 - j0) < cible and (j0, j1) not in fige})
    return out


def grouper_restes(nom, word, cible):
    """Les jetons trop courts et VOISINS deviennent un bloc à eux.

    Louis, 2026-08-12 : « quand tu as fini de souder les sections, tu
    t'arrêtes à des blocs de 4, et tu mets en bloc les suites successives qui
    ne sont rentrées dans aucun bloc → dans This Love le queue et le bridge. »

    Ce qui ne rentre dans aucun bloc n'est pas du bruit, c'est le matériau qui
    n'arrive qu'une fois.
    """
    plafond = int(1.5 * cible)     # un reste plus long que ça se recoupe : sans
                                   # ce plafond, Chain of Fools sortait UN bloc de
                                   # 46 mesures et Sunny un de 86 — « ce qui n'est
                                   # rentré nulle part » finissait par manger le
                                   # morceau. 1,5 × la cible laisse passer la
                                   # queue+pont de This Love (6 mots) d'un bloc.
    out, i = [], 0
    while i < len(nom):
        if not nom[i]["queue"]:
            out.append(nom[i])
            i += 1
            continue
        j = i
        while j + 1 < len(nom) and nom[j + 1]["queue"]:
            j += 1
        run = nom[i:j + 1]
        while run:
            n_mots = sum(x["j1"] - x["j0"] for x in run)
            if n_mots <= plafond:
                bout, run = run, []
            else:
                bout, k = [], 0
                while run and k < cible:
                    k += run[0]["j1"] - run[0]["j0"]
                    bout.append(run.pop(0))
            out.append({"j0": bout[0]["j0"], "j1": bout[-1]["j1"],
                        "type": "".join(x["type"] for x in bout),
                        "label": None, "prime": False, "queue": True})
        i = j + 1
    # les blocs de reste identiques partagent une lettre, comme les autres —
    # mais on prend la première lettre LIBRE, pas la n-ième (corrigé le
    # 2026-08-14). Le comptage d'origine supposait les lettres posées
    # contiguës depuis A ; elles ne le sont plus dès qu'un reste a emporté la
    # sienne. Sur Lost Without U, A partait avec le premier reste, il ne
    # restait que B, le compteur repartait de la deuxième lettre — déjà prise
    # — et la bi-mesure isolée `a` sortait marquée B comme les huit sections
    # `baba`. Deux choses différentes, une seule lettre.
    base = {x["type"]: x["label"] for x in out if x["label"]}
    libres = sorted({x["type"] for x in out if x["label"] is None})
    prises = set(base.values())
    for t in libres:
        L = next((c for c in LETTERS if c not in prises), LETTERS[-1])
        base[t] = L
        prises.add(L)
    for x in out:
        if x["label"] is None:
            x["label"] = base[x["type"]]
    return out


def cout(nom, cible) -> float:
    """Le COÛT ÉPISTÉMIQUE d'une hypothèse : ce qu'il faut écrire pour dire la
    chanson entière.

    Louis, 2026-08-12 : « quand on a deux hypothèses (trancher ici ou ici, 4 ou
    6), il faut voir celle qui aura le meilleur coût épistémique → l'hypothèse
    qui explique le mieux la chanson. »

        dictionnaire : la somme des longueurs des types DISTINCTS
      + partition    : un renvoi par section
      + primes       : un mot de plus (la fin qui change)
      + restes       : leur longueur ENTIÈRE — un bloc qui n'explique rien se
                       paie au prix fort, il faut l'écrire tel quel.
    """
    types = {x["type"] for x in nom if not x["queue"]}
    d = sum(len(t) for t in types)
    p = len(nom)
    pr = sum(1 for x in nom if x["prime"])
    r = sum(x["j1"] - x["j0"] for x in nom if x["queue"])
    return d + p + pr + r


def phrases(word, depart=None, cibles=CIBLES, geles=None):
    """Le chemin complet : agglomérer, nommer, regrouper, puis ARBITRER.

    C'est ce que le bouton « Appliquer » de l'outil Soudure appelle, avec les
    jetons déjà soudés à la main comme point de départ.

    L'arbitrage 4-contre-6 est celui de `quatre_mots.sections` : une section
    fait quatre mots de deux mesures… ou six, et on garde l'hypothèse la moins
    chère à décrire. L'oublier — ce que faisait la première version de cette
    fonction — donnait à l'app un découpage différent de celui de la page de
    recherche sur le même morceau, ce qui est précisément ce qu'on cherche à
    éviter en partageant le code.

    CE QUI N'EST PAS REPRIS. La recherche départage les ex æquo par le score de
    VOIX (`score_voix`), qui demande la matrice de chant ; on ne l'a pas ici, et
    à coût égal on garde donc la plus petite cible. Louis a nommé deux morceaux
    exactement à égalité (Let It Be et Bein' Green, 29 contre 29) : sur
    ceux-là, et sur eux seuls, l'app peut choisir autrement que sa page.
    """
    essais = []
    for c in cibles:
        steps = merges4(word, cible=c, depart=depart, geles=geles)
        nom = grouper_restes(nommer(steps[-1]["jetons"], c, geles=geles), word, c)
        essais.append((cout(nom, c), c, nom))
    essais.sort(key=lambda e: (e[0], e[1]))
    return essais[0][2], {"cible": essais[0][1],
                          "couts": {c: k for k, c, _ in essais}}
