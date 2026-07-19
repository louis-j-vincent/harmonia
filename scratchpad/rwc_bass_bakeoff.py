"""RWC bass/root/quality bake-off: BTC-ISMIR19 vs music-x-lab Chord Structure
Decomposition, scored against sounding_bass_pc GT. Foreground, one song at a
time, wav deleted after each song (disk-tight machine, ~2.5GB free).

Results + analysis: docs/known_issues.md, "Pretrained bass-capable tools run
live on RWC (BTC-ISMIR19 vs music-x-lab) — 2026-07-17".

NOTE (not directly rerunnable as-is): TOOLS below points at a SESSION
scratchpad clone of the two tool repos, which is not preserved. To rerun,
`git clone` these first and update TOOLS accordingly, then apply the small
numpy/torch/yaml compat patches described in known_issues.md (np.float/np.int
-> float/int, yaml.load Loader=, torch.load weights_only=False):
  https://github.com/jayg996/BTC-ISMIR19
  https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition
Both ship their pretrained weights directly in the repo (no separate download).
"""
import sys, csv, io, json, shutil, subprocess, time
from pathlib import Path
import urllib.request
import numpy as np

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO))
from harmonia.data.corpus_schema import sounding_bass_pc
from scripts.build_jaah_corpus import parse_jaah as parse_harte

SCRATCH = Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-Code-harmonia/a6a4757d-a729-450a-a5be-b9970a43c412/scratchpad")
TOOLS = SCRATCH / "nnls_bass_tools"
BTC_DIR = TOOLS / "BTC-ISMIR19"
MUSX_DIR = TOOLS / "ISMIR2019-Large-Vocabulary-Chord-Recognition"
AUDIO_TMP = SCRATCH / "rwc_audio_tmp"
BTC_OUT = SCRATCH / "btc_out"
MUSX_OUT = SCRATCH / "musx_out"
VENV_PY = str(REPO / ".venv/bin/python3")

ZIP_URL = "https://zenodo.org/api/records/18656623/files/RWC-P.zip/content"
CHORD_BASE = ("https://raw.githubusercontent.com/rwc-music/rwc-annotations/"
              "main/01_annotations_preprocessed/chords/RWC-P")
UA = "harmonia-research/1.0 (louisjvincent@gmail.com)"
from remotezip import RemoteZip


def fetch_chords(rwcid: str):
    url = f"{CHORD_BASE}/{rwcid}.csv"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            text = r.read().decode("utf-8")
    except Exception:
        return None
    rows = []
    rd = csv.reader(io.StringIO(text), delimiter=";")
    next(rd, None)
    for row in rd:
        if len(row) != 3:
            continue
        try:
            rows.append((float(row[0]), float(row[1]), row[2].strip()))
        except ValueError:
            continue
    return rows


def load_lab(path: Path):
    """Return list of (t0, t1, label) from a .lab file (space or tab sep)."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            t0, t1 = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        out.append((t0, t1, parts[2]))
    return out


def label_at(intervals, t):
    for t0, t1, lab in intervals:
        if t0 <= t < t1:
            return lab
    return None


def run_btc(wav: Path, song_dir_out: Path):
    song_in = AUDIO_TMP / "btc_in"
    if song_in.exists():
        shutil.rmtree(song_in)
    song_in.mkdir(parents=True)
    link = song_in / wav.name
    shutil.copy(wav, link)
    song_dir_out.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [VENV_PY, "test.py", "--voca", "True", "--audio_dir", str(song_in),
         "--save_dir", str(song_dir_out)],
        cwd=str(BTC_DIR), capture_output=True, text=True, timeout=300)
    shutil.rmtree(song_in, ignore_errors=True)
    lab_path = song_dir_out / (wav.stem + ".lab")
    if r.returncode != 0:
        return None, r.stderr[-2000:]
    return lab_path, None


def run_musx(wav: Path, out_lab: Path):
    r = subprocess.run(
        [VENV_PY, "chord_recognition.py", str(wav), str(out_lab)],
        cwd=str(MUSX_DIR), capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        return None, r.stderr[-2000:]
    return out_lab, None


def score_song(rwcid, gt_rows, btc_rows, musx_rows, stats):
    for t0, t1, lab in gt_rows:
        gt_root, gt_fam, _ = parse_harte(lab)
        if gt_root is None:
            continue
        gt_bass = sounding_bass_pc(lab, gt_root)
        is_inv = (gt_bass is not None) and (gt_bass != gt_root)
        tm = 0.5 * (t0 + t1)
        for tool, rows in (("btc", btc_rows), ("musx", musx_rows)):
            plab = label_at(rows, tm)
            st = stats[tool]
            st["n"] += 1
            if is_inv:
                st["inv_n"] += 1
            else:
                st["rootpos_n"] += 1
            if plab is None:
                continue
            p_root, p_fam, _ = parse_harte(plab)
            if p_root is None:
                continue
            p_bass = sounding_bass_pc(plab, p_root)
            bass_ok = int(p_bass == gt_bass)
            st["root_correct"] += int(p_root == gt_root)
            st["quality_correct"] += int(p_fam == gt_fam)
            st["bass_correct"] += bass_ok
            if is_inv:
                st["inv_bass_correct"] += bass_ok
            else:
                st["rootpos_bass_correct"] += bass_ok
            st["by_fam_n"].setdefault(gt_fam, 0)
            st["by_fam_correct"].setdefault(gt_fam, 0)
            st["by_fam_n"][gt_fam] += 1
            st["by_fam_correct"][gt_fam] += int(p_fam == gt_fam)


def main():
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    max_attempt = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    ids = [f"RWC_P{i:03d}" for i in range(1, max_attempt + 1)]
    stats = {t: {"n": 0, "root_correct": 0, "quality_correct": 0, "bass_correct": 0,
                 "by_fam_n": {}, "by_fam_correct": {},
                 "inv_n": 0, "inv_bass_correct": 0,
                 "rootpos_n": 0, "rootpos_bass_correct": 0} for t in ("btc", "musx")}
    used_songs = []
    log = []
    t_start = time.time()
    AUDIO_TMP.mkdir(parents=True, exist_ok=True)
    with RemoteZip(ZIP_URL) as z:
        names = {Path(i.filename).stem: i.filename for i in z.infolist()
                 if i.filename.endswith(".wav")}
        for rwcid in ids:
            if len(used_songs) >= n_target:
                break
            free = shutil.disk_usage(str(SCRATCH)).free / 1e9
            if free < 1.8:
                print(f"!! disk {free:.2f}GB < 1.8GB floor -> STOP", flush=True)
                break
            rows = fetch_chords(rwcid)
            if not rows:
                print(f"[{rwcid}] no chords, skip"); continue
            zname = names.get(rwcid)
            if not zname:
                print(f"[{rwcid}] no wav in zip, skip"); continue
            print(f"[{rwcid}] ({len(rows)} chords, {free:.2f}GB free) extracting wav...", flush=True)
            try:
                z.extract(zname, path=str(AUDIO_TMP))
                wav = AUDIO_TMP / zname
            except Exception as e:
                print(f"   extract FAIL: {e}"); continue

            t0 = time.time()
            btc_lab, err = run_btc(wav, BTC_OUT / rwcid)
            if err:
                print(f"   BTC FAIL: {err[-500:]}")
                btc_rows = []
            else:
                btc_rows = load_lab(btc_lab)
                print(f"   BTC ok ({len(btc_rows)} segs, {time.time()-t0:.1f}s)")

            t0 = time.time()
            musx_lab, err = run_musx(wav, MUSX_OUT / (rwcid + ".lab"))
            if err:
                print(f"   MUSX FAIL: {err[-500:]}")
                musx_rows = []
            else:
                musx_rows = load_lab(musx_lab)
                print(f"   MUSX ok ({len(musx_rows)} segs, {time.time()-t0:.1f}s)")

            score_song(rwcid, rows, btc_rows, musx_rows, stats)
            used_songs.append(rwcid)
            log.append(rwcid)
            wav.unlink(missing_ok=True)
            print(f"   done. total songs={len(used_songs)}  ({(time.time()-t_start)/60:.1f}min elapsed)", flush=True)

    print("\n=== SONGS USED ===", used_songs)
    print(f"\n=== RESULTS (n songs={len(used_songs)}) ===")
    for tool in ("btc", "musx"):
        st = stats[tool]
        n = st["n"] or 1
        print(f"\n--- {tool} ---  segments scored={st['n']}")
        print(f"  root accuracy:    {st['root_correct']/n:.3f}")
        print(f"  quality accuracy: {st['quality_correct']/n:.3f}")
        print(f"  bass accuracy (all):      {st['bass_correct']/n:.3f}")
        inv_n = st["inv_n"] or 1
        rp_n = st["rootpos_n"] or 1
        print(f"  bass accuracy (inversions, n={st['inv_n']}): {st['inv_bass_correct']/inv_n:.3f}")
        print(f"  bass accuracy (root-pos, n={st['rootpos_n']}):  {st['rootpos_bass_correct']/rp_n:.3f}")
        recalls = []
        for fam, fn in st["by_fam_n"].items():
            if fn > 0:
                recalls.append(st["by_fam_correct"][fam] / fn)
        if recalls:
            print(f"  quality balanced (macro recall over {len(recalls)} families): {np.mean(recalls):.3f}")
        print(f"  per-family n: {st['by_fam_n']}")

    out = {"songs": used_songs, "stats": stats}
    (SCRATCH / "rwc_bass_bakeoff_result.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {SCRATCH / 'rwc_bass_bakeoff_result.json'}")


if __name__ == "__main__":
    main()
