"""music-x-lab (ISMIR2019) chord-recognition over the REMAINING RWC songs (not yet in
musx_out). Reuses the already-cloned+patched repo. Stream-extract-delete, disk-safe,
incremental (.lab per song). Usage: python musx_extract_rest.py <N>
"""
import sys, shutil, subprocess, time
from pathlib import Path
sys.path.insert(0, '.')
from remotezip import RemoteZip

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
SCRATCH = Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-Code-harmonia/a6a4757d-a729-450a-a5be-b9970a43c412/scratchpad")
MUSX_DIR = SCRATCH / "nnls_bass_tools" / "ISMIR2019-Large-Vocabulary-Chord-Recognition"
MUSX_OUT = SCRATCH / "musx_out"; MUSX_OUT.mkdir(parents=True, exist_ok=True)
AUDIO_TMP = SCRATCH / "rwc_audio_tmp"; AUDIO_TMP.mkdir(parents=True, exist_ok=True)
VENV_PY = str(REPO / ".venv/bin/python3")
ZIP_URL = "https://zenodo.org/api/records/18656623/files/RWC-P.zip/content"


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 82
    ids = [f"RWC_P{i:03d}" for i in range(1, 101)]
    done = {p.stem for p in MUSX_OUT.glob("*.lab")}
    todo = [r for r in ids if r not in done][:N]
    print(f"already have {len(done)}; processing {len(todo)}", flush=True)
    if not todo: print("nothing to do"); return
    t_start = time.time(); ok_n = 0
    with RemoteZip(ZIP_URL) as z:
        names = {Path(i.filename).stem: i.filename for i in z.infolist() if i.filename.endswith(".wav")}
        for rid in todo:
            free = shutil.disk_usage(str(SCRATCH)).free / 1e9
            if free < 1.8: print(f"disk {free:.2f}GB floor STOP", flush=True); break
            zn = names.get(rid)
            if not zn: print(f"[{rid}] no wav, skip", flush=True); continue
            try:
                z.extract(zn, path=str(AUDIO_TMP)); wav = AUDIO_TMP / zn
            except Exception as e:
                print(f"[{rid}] extract FAIL {e}", flush=True); continue
            t0 = time.time()
            out_lab = MUSX_OUT / (rid + ".lab")
            r = subprocess.run([VENV_PY, "chord_recognition.py", str(wav), str(out_lab)],
                               cwd=str(MUSX_DIR), capture_output=True, text=True, timeout=600)
            wav.unlink(missing_ok=True)
            if r.returncode != 0:
                print(f"[{rid}] MUSX FAIL: {r.stderr[-300:]}", flush=True); continue
            ok_n += 1
            print(f"[{rid}] ok ({time.time()-t0:.0f}s, {free:.2f}GB, {(time.time()-t_start)/60:.1f}min elapsed, {ok_n} done)", flush=True)
    print(f"DONE_MUSX: {ok_n} new songs", flush=True)


if __name__ == "__main__":
    main()
