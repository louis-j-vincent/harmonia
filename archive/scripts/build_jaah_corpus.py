"""Build a JAAH (113 jazz tracks) real-audio BP48 training corpus.

Phase 0 found AcoustID fingerprint verification BLOCKED (needs an interactive
MusicBrainz-OAuth API key, unavailable in a non-interactive agent session).
See docs/known_issues.md "JAAH real-audio corpus — Phase 0 screen".

Fallback verification gate (honestly NOT AcoustID-verified):
  (a) duration match to MusicBrainz's *authoritative* recording length
      (mbid -> WS/2, more precise than JAAH's own loose `duration` field), AND
  (b) chroma-template correlation of the downloaded audio against JAAH's own
      absolute-timestamp GT chord sequence (a genuine same-recording/harmonic
      -content check, cf. known_issues #3), with an abstain gate.
Songs below the gate are EXCLUDED, not silently kept.

Modes:
  --pilot   validate the chroma-fit gate on a few tracks (download 1 candidate,
            score true-vs-misaligned-vs-permuted alignment) then stop.
  --build   full corpus build with the gate; writes data/cache/jaah/jaah_bp48.npz

Feature extraction reuses harmonia.data.yt_chord_corpus.seg_feature[_abs] and
harmonia.models.chord_pipeline_v1.extract_beat_features exactly as the Billboard
builder does (no reimplementation). Disk discipline: WAV deleted per song.
"""
from __future__ import annotations
import sys, io, json, zipfile, argparse, subprocess, time, random, shutil, os
from pathlib import Path
import numpy as np
import urllib.request

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO))

from harmonia.models.chord_pipeline_v1 import extract_beat_features
from harmonia.models.stage1_pitch import PitchExtractor
from harmonia.data.yt_chord_corpus import (
    seg_feature, seg_feature_abs, seg_feature_clipped, seg_feature_abs_clipped,
    download_audio, QUALITY_IDX, QUALITIES,
)
from harmonia.data.corpus_schema import save_corpus

YTDLP = shutil.which("yt-dlp") or str(REPO / ".venv/bin/yt-dlp")
CACHE = REPO / "data/cache/jaah"
AUDIO_DIR = CACHE / "audio"
BP_CACHE = CACHE / "bp_cache"
LABS_DIR = CACHE / "labs"
UA = "harmonia-research/1.0 (louisjvincent@gmail.com)"

# ── JAAH Harte label -> (root_pc, 7-class family) ────────────────────────────
_NOTE_PC = {"C":0,"D":2,"E":4,"F":5,"G":7,"A":9,"B":11}
_DEG_SEMI = {  # scale-degree token -> semitone above root
    "1":0,"b2":1,"2":2,"#2":3,"b3":3,"3":4,"4":5,"#4":6,"b5":6,"5":7,"#5":8,
    "b6":8,"6":9,"bb7":9,"b7":10,"7":11,"b9":1,"9":2,"#9":3,"11":5,"#11":6,
    "b13":8,"13":9,"#13":10,
}

def _root_pc(tok: str):
    if not tok or tok[0] not in _NOTE_PC:
        return None
    pc = _NOTE_PC[tok[0]]
    for c in tok[1:]:
        if c == "#": pc += 1
        elif c == "b": pc -= 1
        else: break
    return pc % 12

def parse_jaah(label: str):
    """Return (root_pc, family, chord_tone_pcs) or (None, None, None) for N/X."""
    label = label.strip()
    if label in ("N", "X", ""):
        return None, None, None
    base = label.split("/")[0]              # drop bass inversion
    if ":" not in base:
        root = _root_pc(base)
        if root is None: return None, None, None
        return root, "maj", frozenset((root+i) % 12 for i in (0, 4, 7))
    root_str, tail = base.split(":", 1)
    root = _root_pc(root_str)
    if root is None: return None, None, None

    if tail.startswith("("):               # interval-list form
        degs = [d.strip() for d in tail.strip("()").split(",") if d.strip()]
        semis = {_DEG_SEMI[d] for d in degs if d in _DEG_SEMI}
        semis.add(0)                       # root always sounding
        # third quality from literal token, not semitone (#9 also == 3 semis)
        has_b3, has_3 = ("b3" in degs), ("3" in degs)
        has_b5 = ("b5" in degs)
        has_b7 = ("b7" in degs)
        has_s5 = ("#5" in degs)
        has_4 = ("4" in degs) and not has_3 and not has_b3
        if has_4:
            fam = "sus"
        elif has_b3:
            if has_b5 and has_b7: fam = "hdim"
            elif has_b5: fam = "dim"          # incl bb7 dim7
            else: fam = "min"
        elif has_3:
            if has_b7: fam = "dom"            # all altered dominants
            elif has_s5: fam = "aug"
            else: fam = "maj"
        else:
            fam = "maj"
        tones = frozenset((root + s) % 12 for s in semis)
        return root, fam, tones

    # shorthand form
    fam_map = {
        "7":"dom","9":"dom","13":"dom",
        "min":"min","min7":"min","min6":"min","min9":"min","minmaj7":"min","min7b5":"hdim",
        "maj":"maj","maj7":"maj","maj6":"maj","maj9":"maj","6":"maj","9maj":"maj",
        "dim":"dim","dim7":"dim","hdim7":"hdim","aug":"aug",
    }
    fam = fam_map.get(tail)
    if fam is None:
        if tail.startswith("min"): fam = "min"
        elif tail.startswith("maj") or tail in ("6","9"): fam = "maj"
        elif tail.startswith("dim"): fam = "dim"
        elif tail.startswith("hdim"): fam = "hdim"
        elif tail.startswith("aug"): fam = "aug"
        elif tail.startswith("sus"): fam = "sus"
        elif tail[0].isdigit(): fam = "dom"
        else: fam = None
    if fam is None:
        return None, None, None
    tone_map = {
        "maj":(0,4,7),"min":(0,3,7),"dom":(0,4,7,10),"hdim":(0,3,6,10),
        "dim":(0,3,6,9),"aug":(0,4,8),"sus":(0,5,7),
    }
    tones = frozenset((root + i) % 12 for i in tone_map[fam])
    return root, fam, tones


def load_lab(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) != 3: continue
        t0, t1, lab = parts
        rows.append((float(t0), float(t1), lab))
    return rows


# ── MusicBrainz authoritative length via mbid ────────────────────────────────
def mb_length_ms(mbid: str):
    url = f"https://musicbrainz.org/ws/2/recording/{mbid}?inc=isrcs&fmt=json"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            d = json.load(r)
        return d.get("length"), (d.get("isrcs") or [None])[0]
    except Exception as e:
        return None, None


# ── YouTube duration-matched search ──────────────────────────────────────────
def yt_search(query, n=6):
    cmd = [YTDLP, f"ytsearch{n}:{query}", "--print", "%(id)s\t%(duration)s\t%(title)s",
           "--skip-download", "--no-warnings"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    out = []
    for l in r.stdout.strip().split("\n"):
        p = l.split("\t")
        if len(p) != 3: continue
        vid, dur, title = p
        try: dur = float(dur)
        except ValueError: dur = None
        out.append((vid, dur, title))
    return out


# ── chroma-fit verification gate ─────────────────────────────────────────────
def chroma_fit(wav: Path, rows, *, shift=0.0, permute=False, seed=0):
    """Mean duration-weighted cosine between observed chroma and GT chord-tone
    template over each chord interval. Higher = audio harmonic content agrees
    with JAAH's timestamped chords (right recording, right alignment)."""
    import librosa
    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    hop = 2048
    C = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)  # (12, T)
    C = C / (np.linalg.norm(C, axis=0, keepdims=True) + 1e-9)
    times = librosa.frames_to_time(np.arange(C.shape[1]), sr=sr, hop_length=hop)
    parsed = [(t0, t1, parse_jaah(lab)) for t0, t1, lab in rows]
    parsed = [(t0, t1, p[2]) for t0, t1, p in parsed if p[0] is not None]
    if permute:
        rng = random.Random(seed)
        tones = [p[2] for p in parsed]; rng.shuffle(tones)
        parsed = [(t0, t1, tn) for (t0, t1, _), tn in zip(parsed, tones)]
    num = den = 0.0
    for t0, t1, tones in parsed:
        i0 = np.searchsorted(times, t0 + shift)
        i1 = np.searchsorted(times, t1 + shift)
        if i1 <= i0: continue
        obs = C[:, i0:i1].mean(1)
        obs = obs / (np.linalg.norm(obs) + 1e-9)
        tmpl = np.zeros(12);
        for pc in tones: tmpl[pc] = 1.0
        tmpl /= (np.linalg.norm(tmpl) + 1e-9)
        w = t1 - t0
        num += w * float(obs @ tmpl); den += w
    return num / den if den else 0.0


def fetch_labs():
    LABS_DIR.mkdir(parents=True, exist_ok=True)
    if list(LABS_DIR.glob("*.lab")): return
    req = urllib.request.Request("https://raw.githubusercontent.com/MTG/JAAH/master/labs.zip",
                                 headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    for n in z.namelist():
        if n.endswith(".lab"):
            (LABS_DIR / Path(n).name).write_bytes(z.read(n))


def ann_meta(slug):
    url = f"https://raw.githubusercontent.com/MTG/JAAH/master/annotations/{slug}.json"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=25) as r:
        d = json.load(r)
    return d.get("artist"), d.get("title"), d.get("mbid"), d.get("duration")


def source_candidate(artist, title, target_dur):
    cands = yt_search(f"{artist} {title}")
    best = None
    for vid, dur, ctitle in cands:
        if dur is None: continue
        diff = abs(dur - target_dur)
        tol = max(0.05 * target_dur, 6.0)
        if diff <= tol and (best is None or diff < best[1]):
            best = (vid, diff, dur, ctitle)
    return best, cands


# ── build one song's records ─────────────────────────────────────────────────
def build_song(slug, wav, rows, song_id):
    # Boundary-bleed fix (2026-07-16): pool frames clipped EXACTLY to [t0,t1)
    # instead of snapping to whole beats (which bled ~1 beat of the next chord
    # in — see docs/known_issues.md "boundary bleed"). PitchExtractor.extract is
    # cached, so this does not re-run Basic Pitch if extract_beat_features already
    # ran on this wav.
    acts = PitchExtractor(cache_dir=BP_CACHE).extract(wav)
    ft, onf, ntf = acts.frame_times, acts.onset_probs, acts.note_probs
    recs = []
    for t0, t1, lab in rows:
        root, fam, _ = parse_jaah(lab)
        if root is None: continue
        f48 = seg_feature_clipped(ft, onf, ntf, t0, t1, root)
        f48a = seg_feature_abs_clipped(ft, onf, ntf, t0, t1)
        if f48 is None or f48a is None: continue   # no frame in span
        recs.append({
            "feat48": f48, "feat48_abs": f48a,
            "root": int(root % 12), "quality": fam, "quality_idx": QUALITY_IDX[fam],
            "t0": float(t0), "t1": float(t1), "label": lab,
            "match": "exact", "song_id": song_id,
        })
    return recs


def slug_from_lab(p: Path):
    return p.stem


def run_pilot(slugs):
    fetch_labs()
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== PILOT: validate chroma-fit gate on {slugs} ===\n", flush=True)
    for slug in slugs:
        lab = LABS_DIR / f"{slug}.lab"
        if not lab.exists():
            print(f"[{slug}] no .lab, skip"); continue
        rows = load_lab(lab)
        artist, title, mbid, jdur = ann_meta(slug)
        mb_ms, isrc = mb_length_ms(mbid) if mbid else (None, None)
        target = (mb_ms / 1000.0) if mb_ms else jdur
        print(f"[{slug}] {artist} - {title}")
        print(f"    mbid={mbid} isrc={isrc} MB_len={target}s JAAH_dur={jdur}s n_chords={len(rows)}")
        best, cands = source_candidate(artist, title, target)
        if not best:
            print(f"    NO duration match among {len(cands)} candidates -> would EXCLUDE\n"); continue
        vid, diff, dur, ctitle = best
        print(f"    candidate {vid} dur={dur:.1f} diff={diff:.1f}  '{ctitle[:60]}'", flush=True)
        try:
            wav = download_audio(vid, AUDIO_DIR)
        except Exception as e:
            print(f"    download failed: {e}\n"); continue
        true_fit = chroma_fit(wav, rows, shift=0.0)
        sh5 = chroma_fit(wav, rows, shift=5.0)
        shm5 = chroma_fit(wav, rows, shift=-5.0)
        perm = np.mean([chroma_fit(wav, rows, permute=True, seed=s) for s in range(3)])
        print(f"    chroma-fit  TRUE={true_fit:.3f}  shift+5={sh5:.3f}  shift-5={shm5:.3f}  permuted={perm:.3f}")
        print(f"    -> margin(true - permuted)={true_fit-perm:+.3f}   {'PASS' if true_fit-perm>0.03 and true_fit>0.30 else 'WEAK'}\n", flush=True)
        wav.unlink(missing_ok=True)


def run_build(max_songs, seed, gate_abs, gate_margin, resume=False, floor_gb=1.0):
    fetch_labs()
    AUDIO_DIR.mkdir(parents=True, exist_ok=True); BP_CACHE.mkdir(parents=True, exist_ok=True)
    slugs = sorted(slug_from_lab(p) for p in LABS_DIR.glob("*.lab"))
    rng = random.Random(seed); rng.shuffle(slugs)
    if max_songs: slugs = slugs[:max_songs]

    prev_log = []
    prev_recs = {}  # song_id -> dict of arrays, loaded from existing npz for merge
    if resume:
        log_path = CACHE / "build_log.json"
        if log_path.exists():
            prev_log = json.loads(log_path.read_text())
        attempted = {l[0] for l in prev_log}
        before = len(slugs)
        slugs = [s for s in slugs if s not in attempted]
        print(f"[resume] {len(attempted)} slugs already attempted previously; "
              f"{before} -> {len(slugs)} remaining candidates", flush=True)
        corpus_path = CACHE / "jaah_bp48.npz"
        if corpus_path.exists():
            from harmonia.data.corpus_schema import load_corpus
            prev_recs = load_corpus(corpus_path)
            print(f"[resume] loaded existing corpus: {len(prev_recs['root'])} records, "
                  f"{len(set(prev_recs['song_id'].tolist()))} songs", flush=True)

    all_recs = []; log = []
    t0 = time.time()
    for i, slug in enumerate(slugs):
        free_gb = shutil.disk_usage(str(CACHE)).free / 1e9
        if free_gb < floor_gb:
            print(f"!! disk {free_gb:.2f}GB < {floor_gb}GB floor, stopping"); break
        rows = load_lab(LABS_DIR / f"{slug}.lab")
        try:
            artist, title, mbid, jdur = ann_meta(slug)
        except Exception as e:
            log.append((slug, "meta_fail", str(e))); continue
        mb_ms, isrc = mb_length_ms(mbid) if mbid else (None, None)
        target = (mb_ms / 1000.0) if mb_ms else jdur
        print(f"\n[{i+1}/{len(slugs)}] {slug}  {artist} - {title}  (target {target:.0f}s, {free_gb:.1f}GB free)", flush=True)
        best, cands = source_candidate(artist, title, target)
        if not best:
            log.append((slug, "no_dur_match", len(cands))); print("    EXCLUDE: no duration match"); continue
        vid, diff, dur, ctitle = best
        try:
            wav = download_audio(vid, AUDIO_DIR)
        except Exception as e:
            log.append((slug, "dl_fail", str(e)[:80])); print("    EXCLUDE: download failed"); continue
        try:
            fit = chroma_fit(wav, rows); perm = chroma_fit(wav, rows, permute=True, seed=1)
        except Exception as e:
            log.append((slug, "chroma_fail", str(e)[:80])); wav.unlink(missing_ok=True); continue
        margin = fit - perm
        passed = fit >= gate_abs and margin >= gate_margin
        print(f"    dur_diff={diff:.1f}s  chroma-fit={fit:.3f} perm={perm:.3f} margin={margin:+.3f}  "
              f"{'ACCEPT' if passed else 'EXCLUDE(gate)'}", flush=True)
        if not passed:
            log.append((slug, "gate_fail", round(fit,3), round(margin,3))); wav.unlink(missing_ok=True); continue
        try:
            recs = build_song(slug, wav, rows, f"jaah_{slug}")
        except Exception as e:
            log.append((slug, "feat_fail", str(e)[:80])); wav.unlink(missing_ok=True); continue
        all_recs += recs
        log.append((slug, "ACCEPT", len(recs), round(fit,3), round(margin,3), isrc))
        print(f"    +{len(recs)} records  (total {len(all_recs)}, {(time.time()-t0)/60:.1f}min)", flush=True)
        wav.unlink(missing_ok=True)

    print("\n\n=== BUILD SUMMARY (this run) ===")
    acc = [l for l in log if l[1] == "ACCEPT"]
    print(f"attempted={len(slugs)} accepted={len(acc)} excluded={len(slugs)-len(acc)} new_records={len(all_recs)}")
    for l in log: print("   ", l)

    merged_log = prev_log + log
    (CACHE / "build_log.json").write_text(json.dumps(merged_log, indent=2, default=str))

    if all_recs or prev_recs:
        out = {
            "feat48": np.stack([r["feat48"] for r in all_recs]) if all_recs else np.zeros((0, 48)),
            "feat48_abs": np.stack([r["feat48_abs"] for r in all_recs]) if all_recs else np.zeros((0, 48)),
            "root": np.array([r["root"] for r in all_recs], dtype=np.int32),
            "quality_idx": np.array([r["quality_idx"] for r in all_recs], dtype=np.int32),
            "quality": np.array([r["quality"] for r in all_recs]),
            "labels": np.array([r["label"] for r in all_recs]),
            "match": np.array([r["match"] for r in all_recs]),
            "t0": np.array([r["t0"] for r in all_recs]),
            "t1": np.array([r["t1"] for r in all_recs]),
            "song_id": np.array([r["song_id"] for r in all_recs]),
        }
        if prev_recs:
            keys = [k for k in out if k != "qualities"]
            merged = {}
            for k in keys:
                a = prev_recs[k] if k in prev_recs else np.array([])
                b = out[k]
                merged[k] = np.concatenate([a, b]) if len(a) or len(b) else a
            merged["qualities"] = np.array(QUALITIES)
            out = merged
        else:
            out["qualities"] = np.array(QUALITIES)
        outp = CACHE / "jaah_bp48.npz"
        save_corpus(outp, **out)
        n_songs = len(set(out["song_id"].tolist()))
        print(f"\nWrote {outp}  ({len(out['root'])} total records, {n_songs} total songs; "
              f"+{len(all_recs)} new records this run)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", nargs="*", default=None)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--max-songs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gate-abs", type=float, default=0.30)
    ap.add_argument("--gate-margin", type=float, default=0.03)
    ap.add_argument("--resume", action="store_true",
                     help="skip slugs already present in build_log.json and merge new "
                          "accepts into the existing jaah_bp48.npz instead of overwriting")
    ap.add_argument("--floor-gb", type=float, default=1.0,
                     help="stop (self-throttle) once free disk drops below this many GB")
    a = ap.parse_args()
    if a.pilot is not None:
        run_pilot(a.pilot or ["airegin", "bags_groove", "blue_monk"])
    elif a.build:
        run_build(a.max_songs, a.seed, a.gate_abs, a.gate_margin, resume=a.resume, floor_gb=a.floor_gb)
    else:
        ap.print_help()
