"""`/api/import-audio` : un fichier audio importé part dans le même job que le
micro. Tout est en tmp_path ; `start_job` est remplacé (aucune analyse)."""
import io
import shutil
import subprocess

import pytest

from harmonia.server.routes import analyze

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "AUDIO_DIR", tmp_path / "audio")
    vus = []
    monkeypatch.setattr(analyze, "start_job",
                        lambda stem, title="": vus.append((stem, title)) or "j1")
    from harmonia.server import app as app_mod
    c = app_mod.create_app().test_client()
    c.vus, c.audio = vus, tmp_path / "audio"
    return c


def _wav(tmp_path):
    p = tmp_path / "Ma_chanson.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=1", str(p)], check=True)
    return p.read_bytes()


def test_import_wav_lance_un_job(client, tmp_path):
    r = client.post("/api/import-audio", content_type="multipart/form-data",
                    data={"audio": (io.BytesIO(_wav(tmp_path)), "Ma_chanson.wav")})
    assert r.get_json() == {"job_id": "j1"}
    (stem, titre), = client.vus
    assert stem.startswith("imp_") and titre == "Ma chanson"
    assert (client.audio / f"{stem}.m4a").stat().st_size > 0
    assert not list(client.audio.glob("*.upload"))


def test_import_sans_audio_400(client):
    r = client.post("/api/import-audio", data={})
    assert r.status_code == 400 and not client.vus


def test_import_illisible_400_et_rien_ne_reste(client):
    r = client.post("/api/import-audio", content_type="multipart/form-data",
                    data={"audio": (io.BytesIO(b"pas de l'audio"), "x.mp3")})
    assert r.status_code == 400 and not client.vus
    assert not list(client.audio.glob("*"))
