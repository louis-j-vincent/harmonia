"""Run madmom's PRETRAINED deep-chroma maj/min chord recognizer on RWC songs,
score root + maj/min against GT on the maj/min-family subset — the cascade
'pretrained tool on the easy majority' number. Audio via remotezip, deleted after."""
import collections, collections.abc as abc, numpy as np
for nm in ("MutableSequence","Sequence","MutableMapping","Mapping","Iterable","Callable"):
    setattr(collections, nm, getattr(abc, nm))
for nm,ty in [("float",float),("int",int),("bool",bool),("complex",complex),("object",object)]:
    if not hasattr(np,nm): setattr(np,nm,ty)
import sys, shutil, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
from harmonia.data.corpus_schema import load_corpus
from remotezip import RemoteZip
from madmom.audio.chroma import DeepChromaProcessor
from madmom.features.chords import DeepChromaChordRecognitionProcessor

REPO=Path('.').resolve(); AUDIO=REPO/'data/cache/rwc/audio'; AUDIO.mkdir(parents=True,exist_ok=True)
ZIP_URL="https://zenodo.org/api/records/18656623/files/RWC-P.zip/content"
PC={'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'F':5,'F#':6,'Gb':6,'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'B':11}
def parse(lbl):
    lbl=lbl.strip()
    if lbl in ('N','X') or ':' not in lbl: return (-1,-1)
    r,q=lbl.split(':',1); r=r.split('/')[0]
    if r not in PC: return (-1,-1)
    return (PC[r], 0 if q.startswith('maj') else (1 if q.startswith('min') else -1))

d=load_corpus('data/cache/rwc/rwc_bp48.npz')
sid=d['song_id']; lab=d['labels']; root=d['root'].astype(int); t0=d['t0']; t1=d['t1']; q=d['quality']
majmin=np.isin(q,['maj','min'])
gt_mm = np.where(q=='maj',0,np.where(q=='min',1,-1))  # GT maj/min (family)
songs=["rwc_RWC_P040","rwc_RWC_P083","rwc_RWC_P099","rwc_RWC_P066","rwc_RWC_P020"]
dcp=DeepChromaProcessor(); rec=DeepChromaChordRecognitionProcessor()
rows=[]
with RemoteZip(ZIP_URL) as z:
    names={Path(i.filename).stem:i.filename for i in z.infolist() if i.filename.endswith('.wav')}
    for s in songs:
        free=shutil.disk_usage(str(AUDIO)).free/1e9
        if free<2.0: print("disk floor STOP"); break
        zn=names.get(s.replace('rwc_',''));
        if not zn: continue
        print(f"[{s}] fetch ({free:.1f}GB)...",flush=True); z.extract(zn,path=str(AUDIO)); wav=AUDIO/zn
        try:
            chroma=dcp(str(wav)); seg=rec(chroma)  # array of (start,end,label)
        finally:
            try: wav.unlink()
            except: pass
        segs=[(float(a),float(b),str(c)) for a,b,c in seg]
        m=np.where(sid==s)[0]
        nok_r=nok_mm=n=0
        for gi in m:
            mid=0.5*(t0[gi]+t1[gi]); pr,pmm=(-1,-1)
            for a,b,c in segs:
                if a<=mid<b: pr,pmm=parse(c); break
            rows.append((int(majmin[gi]), int(root[gi]), int(gt_mm[gi]), pr, pmm))
            n+=1; nok_r+= (pr==root[gi]); nok_mm+= (pmm==gt_mm[gi] and pmm>=0)
        print(f"   {n} chords: madmom root {nok_r/n:.3f} majmin {nok_mm/n:.3f}",flush=True)

R=np.array(rows); mm=R[:,0]==1; gr=R[:,1]; gmm=R[:,2]; pr=R[:,3]; pmm=R[:,4]
print(f"\n{'='*60}\nmadmom pretrained (deep-chroma maj/min CRF), pooled n={len(R)}")
for lbl,mask in [("maj/min subset",mm),("residual",~mm),("ALL",np.ones(len(R),bool))]:
    if mask.sum()==0: continue
    print(f"\n{lbl}: n={mask.sum()}")
    print(f"   madmom ROOT acc:          {(pr[mask]==gr[mask]).mean():.3f}")
    print(f"   madmom maj/min acc:       {(pmm[mask]==gmm[mask]).mean():.3f}")
    print(f"   madmom JOINT root&majmin: {((pr==gr)&(pmm==gmm))[mask].mean():.3f}")
