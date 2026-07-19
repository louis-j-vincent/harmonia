"""Robust, CACHED pYIN sounding-bass extractor over RWC. Fetch one WAV at a time
(remotezip, deleted after), low-pass 400Hz, pYIN with voiced_prob. Per GT chord
span store: modal bass pc (mod-12 -> octave-error-immune), mean confidence
(voiced_prob), voiced fraction, and a 'reliable' flag. Appends to a cache npz keyed
by GLOBAL corpus row index so it aligns 1:1 with rwc_bp48_fixed/rwc_nnls24.

Usage: python pyin_extract_cache.py <N_songs>   (processes next N not-yet-cached songs)
"""
import sys, shutil, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
from harmonia.data.corpus_schema import load_corpus
from remotezip import RemoteZip
import librosa
from scipy.signal import butter, sosfiltfilt

REPO=Path('.').resolve(); AUDIO=REPO/'data/cache/rwc/audio'; AUDIO.mkdir(parents=True,exist_ok=True)
ZIP_URL="https://zenodo.org/api/records/18656623/files/RWC-P.zip/content"
CACHE=REPO/'scratchpad/pyin_bass_cache.npz'
REL_FRAC=0.30  # span needs >=30% voiced frames else flagged unreliable (fallback)

def pyin_song(wav, sr=11025, hop=512):
    y,_=librosa.load(str(wav),sr=sr,mono=True)
    sos=butter(4,400.0,btype='low',fs=sr,output='sos')
    yl=sosfiltfilt(sos,y).astype(np.float32)
    f0,vflag,vprob=librosa.pyin(yl,fmin=41.2,fmax=392.0,sr=sr,hop_length=hop,frame_length=2048,fill_na=np.nan)
    t=librosa.times_like(f0,sr=sr,hop_length=hop)
    # pYIN's own voicing decision (voiced_flag); vprob kept as SOFT confidence.
    ok=np.isfinite(f0)&(f0>0)&vflag
    midi=np.where(ok,69+12*np.log2(np.where(ok,f0,440.0)/440.0),np.nan)
    pc=np.where(ok,np.round(midi)%12,-1).astype(int)
    return t,pc,vprob,ok

def main():
    N=int(sys.argv[1]) if len(sys.argv)>1 else 10
    d=load_corpus('data/cache/rwc/rwc_bp48_fixed.npz')
    sid=d['song_id']; t0=d['t0']; t1=d['t1']
    allsongs=sorted(set(sid.tolist()))
    # load existing cache
    done=set()
    store={}
    if CACHE.exists():
        z=np.load(CACHE,allow_pickle=True)
        for k in z.files: store[k]=z[k]
        done=set(store.get('songs_done',np.array([])).tolist())
    todo=[s for s in allsongs if s not in done][:N]
    print(f"cached songs: {len(done)}; processing {len(todo)}: {todo}",flush=True)
    if not todo: print("nothing to do"); return
    # arrays we build/extend (indexed by global row); init from existing or fresh
    n=len(sid)
    def col(name,fill):
        return store[name].copy() if name in store else np.full(n,fill,float)
    bass=col('bass_pc',-1).astype(int) if 'bass_pc' in store else np.full(n,-1,int)
    conf=col('conf',np.nan); vfrac=col('vfrac',np.nan); rel=col('reliable',0).astype(int) if 'reliable' in store else np.zeros(n,int)
    import time as _time
    def save():
        np.savez_compressed(CACHE, bass_pc=bass, conf=conf, vfrac=vfrac, reliable=rel,
                            songs_done=np.array(sorted(done)))
    def fetch_wav(zn):
        # robust fetch: retry up to 4x, re-opening the RemoteZip on transient errors
        for att in range(4):
            try:
                with RemoteZip(ZIP_URL) as zz:
                    zz.extract(zn, path=str(AUDIO)); return AUDIO/zn
            except Exception as e:
                print(f"   fetch retry {att+1}/4 ({str(e)[:50]})",flush=True); _time.sleep(3*(att+1))
        return None
    names=None
    for att in range(4):
        try:
            with RemoteZip(ZIP_URL) as zz:
                names={Path(i.filename).stem:i.filename for i in zz.infolist() if i.filename.endswith('.wav')}
            break
        except Exception as e:
            print(f"infolist retry {att+1}/4 ({str(e)[:50]})",flush=True); _time.sleep(3*(att+1))
    if names is None: print("could not list zip; abort"); return

    for s in todo:
            free=shutil.disk_usage(str(AUDIO)).free/1e9
            if free<2.0: print(f"disk {free:.1f}GB floor STOP"); break
            zn=names.get(s.replace('rwc_',''))
            if not zn: print(f"[{s}] no wav"); continue
            print(f"[{s}] fetch ({free:.1f}GB)...",flush=True)
            wav=fetch_wav(zn)
            if wav is None: print(f"[{s}] fetch FAILED, skip"); continue
            try: t,pc,vprob,ok=pyin_song(wav)
            finally:
                try: wav.unlink()
                except: pass
            idx=np.where(sid==s)[0]; nr=0
            for gi in idx:
                seg=(t>=t0[gi])&(t<t1[gi])
                nseg=max(int(seg.sum()),1)
                segok=seg&ok
                vf=segok.sum()/nseg
                vfrac[gi]=vf
                if segok.sum()>0:
                    bass[gi]=int(np.bincount(pc[segok],minlength=12).argmax())
                    conf[gi]=float(np.mean(vprob[segok]))
                    rel[gi]=1 if vf>=REL_FRAC else 0
                    if rel[gi]: nr+=1
                else:
                    bass[gi]=-1; conf[gi]=0.0; rel[gi]=0
            done.add(s); save()  # incremental save after every song
            print(f"   {len(idx)} chords, reliable {nr}/{len(idx)}",flush=True)
    save()
    print(f"saved cache: {len(done)} songs done, {(bass>=0).sum()} chords with a bass pc")

if __name__=='__main__': main()
