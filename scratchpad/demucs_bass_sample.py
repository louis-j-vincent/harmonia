"""STEP 3 — REAL Demucs bass-stem separation + pYIN, on a sample of the pYIN-covered
RWC songs. Compares real-Demucs-pYIN bass vs the 400Hz-low-pass PROXY pYIN (and the
NNLS estimators) — specifically on the HARD RESIDUAL (chords where the NNLS estimators
all fail). Decides whether a cleaner bass stem is worth a full 100-song run.

Stream-extract-delete (disk-safe): fetch one WAV, separate in-memory, pYIN, delete.
Incremental save to scratchpad/demucs_bass_cache.npz keyed by GLOBAL row index.
Usage: python demucs_bass_sample.py <N_songs>
"""
import sys, shutil, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from harmonia.data.corpus_schema import load_corpus
from remotezip import RemoteZip
import librosa, torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

REPO = Path('.').resolve(); AUDIO = REPO/'data/cache/rwc/audio'; AUDIO.mkdir(parents=True, exist_ok=True)
ZIP_URL = "https://zenodo.org/api/records/18656623/files/RWC-P.zip/content"
CACHE = REPO/'scratchpad/demucs_bass_cache.npz'
REL_FRAC = 0.30

_model = None
def model():
    global _model
    if _model is None:
        _model = get_model('htdemucs'); _model.eval()
    return _model

def demucs_bass_pc(wav, hop=512):
    m = model()
    y, sr = librosa.load(str(wav), sr=m.samplerate, mono=False)
    if y.ndim == 1: y = np.stack([y, y])
    wav_t = torch.tensor(y, dtype=torch.float32)[None]
    ref = wav_t.mean(0); wav_t = (wav_t - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        sources = apply_model(m, wav_t, device='cpu', progress=False, split=True, overlap=0.10)[0]
    sources = sources * ref.std() + ref.mean()
    bass_idx = m.sources.index('bass')
    bass = sources[bass_idx].mean(0).numpy()          # mono bass stem @ 44100
    br = librosa.resample(bass, orig_sr=m.samplerate, target_sr=11025)
    f0, vflag, vprob = librosa.pyin(br, fmin=41.2, fmax=392.0, sr=11025, hop_length=hop,
                                    frame_length=2048, fill_na=np.nan)
    t = librosa.times_like(f0, sr=11025, hop_length=hop)
    ok = np.isfinite(f0) & (f0 > 0) & vflag
    midi = np.where(ok, 69 + 12*np.log2(np.where(ok, f0, 440.0)/440.0), np.nan)
    pc = np.where(ok, np.round(midi) % 12, -1).astype(int)
    return t, pc, vprob, ok

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    d = load_corpus('data/cache/rwc/rwc_bp48_fixed.npz')
    sid = d['song_id']; t0 = d['t0']; t1 = d['t1']; n = len(sid)
    # restrict candidates to songs pYIN already covered (for a matched comparison)
    py = np.load('scratchpad/pyin_bass_cache.npz', allow_pickle=True)
    py_songs = list(py['songs_done'].tolist())
    store = {}; done = set()
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        for k in z.files: store[k] = z[k]
        done = set(store.get('songs_done', np.array([])).tolist())
    bass = store['bass_pc'].copy() if 'bass_pc' in store else np.full(n, -1, int)
    conf = store['conf'].copy() if 'conf' in store else np.full(n, np.nan)
    vfr = store['vfrac'].copy() if 'vfrac' in store else np.full(n, np.nan)
    todo = [s for s in py_songs if s not in done][:N]
    print(f"cached: {len(done)}; processing {len(todo)}: {todo}", flush=True)
    if not todo: print("nothing to do"); return

    def save():
        np.savez_compressed(CACHE, bass_pc=bass, conf=conf, vfrac=vfr, songs_done=np.array(sorted(done)))
    names = None
    for att in range(4):
        try:
            with RemoteZip(ZIP_URL) as zz:
                names = {Path(i.filename).stem: i.filename for i in zz.infolist() if i.filename.endswith('.wav')}
            break
        except Exception as e:
            print(f"infolist retry {att+1} ({str(e)[:40]})", flush=True)
    if names is None: print("no zip list"); return

    import time as _t
    for s in todo:
        free = shutil.disk_usage(str(AUDIO)).free/1e9
        if free < 1.5: print(f"disk {free:.1f}GB floor STOP"); break
        zn = names.get(s.replace('rwc_', ''))
        if not zn: print(f"[{s}] no wav"); continue
        print(f"[{s}] fetch ({free:.1f}GB free)...", flush=True); t_start = _t.time()
        wav = None
        for att in range(4):
            try:
                with RemoteZip(ZIP_URL) as zz: zz.extract(zn, path=str(AUDIO)); wav = AUDIO/zn; break
            except Exception as e: print(f"   fetch retry {att+1} ({str(e)[:40]})", flush=True); _t.sleep(3)
        if wav is None: print(f"[{s}] fetch FAILED"); continue
        try:
            print(f"[{s}] separating...", flush=True)
            t, pc, vprob, ok = demucs_bass_pc(wav)
        finally:
            try: wav.unlink()
            except: pass
        idx = np.where(sid == s)[0]; nr = 0
        for gi in idx:
            seg = (t >= t0[gi]) & (t < t1[gi]); nseg = max(int(seg.sum()), 1)
            so = seg & ok; vf = so.sum()/nseg; vfr[gi] = vf
            if so.sum() > 0:
                bass[gi] = int(np.bincount(pc[so], minlength=12).argmax())
                conf[gi] = float(np.mean(vprob[so])); nr += 1 if vf >= REL_FRAC else 0
            else:
                bass[gi] = -1; conf[gi] = 0.0
        done.add(s); save()
        print(f"   {len(idx)} chords ({_t.time()-t_start:.0f}s), reliable {nr}", flush=True)
    print(f"DONE: {len(done)} songs, {(bass>=0).sum()} chords", flush=True)

if __name__ == '__main__':
    main()
