"""La recherche de « Mes charts » est globale (harmonia/static/recherche.js).

Le module est pur JS ; on l'exécute sous node (copié en .mjs pour qu'il soit
chargé comme module ES). Sauté si node est absent.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "harmonia" / "static" / "recherche.js"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node absent")


def _run(tmp_path, body):
    mod = tmp_path / "recherche.mjs"
    mod.write_text(SRC.read_text())
    script = tmp_path / "t.mjs"
    script.write_text(
        "import * as R from './recherche.mjs';\n"
        "const nomDe = c => ({artist: c.artist || '', title: c.title || ''});\n"
        + body
    )
    out = subprocess.run([NODE, str(script)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


LIB = [
    {"file": "a", "artist": "Beyoncé", "title": "Halo"},
    {"file": "b", "artist": "Joseph Kosma", "title": "Autumn Leaves"},
    {"file": "c", "artist": "Maroon 5", "title": "This Love"},
    {"file": "d", "artist": "Leaves Band", "title": "Something"},
    {"file": "e", "artist": "X", "title": "Falling Leaves"},
]
OF = {"b": "Jazz/Standards", "c": "Old charts"}


def test_tous_les_dossiers_et_hors_dossier(tmp_path):
    res = _run(tmp_path,
        f"const r = R.chercherPartout({json.dumps(LIB)}, {json.dumps(OF)}, 'leaves', nomDe);"
        "console.log(JSON.stringify(r.map(x => [x.chart.file, x.dossier, x.rang])));")
    # b (dossier imbriqué), e (hors dossier) : un mot du titre commence par
    # « leaves » (rang 1) ; d : c'est l'artiste (rang 2).
    assert res == [
        ["b", "Jazz › Standards", 1],
        ["e", "Hors dossier", 1],
        ["d", "Hors dossier", 2],
    ]


def test_accents_casse_et_mots(tmp_path):
    res = _run(tmp_path,
        f"const L={json.dumps(LIB)}, O={json.dumps(OF)};"
        "console.log(JSON.stringify(["
        " R.chercherPartout(L,O,'BEYONCE',nomDe).map(x=>x.chart.file),"
        " R.chercherPartout(L,O,'maroon love',nomDe).map(x=>x.chart.file),"
        " R.chercherPartout(L,O,'this',nomDe).map(x=>[x.chart.file,x.rang]),"
        " R.chercherPartout(L,O,'zzz',nomDe).length,"
        "]));")
    assert res == [["a"], ["c"], [["c", 0]], 0]


def test_chemin_dossier(tmp_path):
    res = _run(tmp_path,
        "console.log(JSON.stringify([R.cheminDossier(undefined), R.cheminDossier('Old charts'),"
        " R.cheminDossier('Jazz/ Standards')]));")
    assert res == ["Hors dossier", "Old charts", "Jazz › Standards"]
