import json, warnings
from pathlib import Path
import numpy as np, librosa, mirdata
warnings.filterwarnings("ignore")
REPO=Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
AUDIO=REPO/"docs/audio"
vid=json.loads((REPO/"docs/plots/.yt_video_ids.json").read_text())
corpus={}
for n in ["billboard_search_results.json","billboard_search_results_60.json"]:
    p=REPO/"scratchpad"/n
    if p.exists():
        try:corpus.update(json.loads(p.read_text()))
        except:pass
vid2track={}
for tid,v in corpus.items():
    b=v.get("best") or []
    if b:vid2track[b[0]]=tid
ds=mirdata.initialize("billboard")

def analyze(af):
    stem=af.stem; v=vid.get(f"inferred_{stem}.html")
    tid=vid2track.get(v) if v else None
    if not tid: return None
    cf=ds.track(tid).chords_full
    bounds=np.array([float(a) for (a,b) in cf.intervals])  # chord change times
    labels=[str(l) for l in cf.labels]
    y,sr=librosa.load(str(af),sr=22050,mono=True)  # full
    hop=512
    oenv=librosa.onset.onset_strength(y=y,sr=sr,hop_length=hop)
    ot=librosa.times_like(oenv,sr=sr,hop_length=hop)
    dur=len(y)/sr
    # GT impulse train on same grid, weight boundaries where label actually changes
    imp=np.zeros_like(oenv)
    for i,t in enumerate(bounds):
        if t<dur and (i==0 or labels[i]!=labels[i-1]):
            k=int(round(t/(hop/sr)))
            if 0<=k<len(imp): imp[k]=1.0
    # smooth impulse a bit
    from scipy.ndimage import gaussian_filter1d
    imps=gaussian_filter1d(imp,sigma=2)
    oz=(oenv-oenv.mean())/(oenv.std()+1e-9)
    iz=(imps-imps.mean())/(imps.std()+1e-9)
    # xcorr over lag -3..+9 s  (positive lag = audio content LATER than GT => shift GT forward)
    fps=sr/hop
    lags=np.arange(int(-3*fps),int(9*fps))
    cc=[]
    for L in lags:
        a=oz; b=np.roll(iz,L)
        cc.append(np.dot(a,b))
    cc=np.array(cc); best=lags[np.argmax(cc)]; best_s=best/fps
    # per-boundary nearest-onset residual using best_s, first 12 real changes
    onsets=librosa.onset.onset_detect(onset_envelope=oenv,sr=sr,hop_length=hop,units='time',backtrack=True)
    real=[bounds[i] for i in range(len(bounds)) if i==0 or labels[i]!=labels[i-1]]
    resid=[]
    for t in real[:40]:
        tt=t+best_s
        if len(onsets):
            j=np.argmin(np.abs(onsets-tt)); resid.append(onsets[j]-tt)
    resid=np.array(resid)
    return dict(stem=stem,tid=tid,best_offset_s=round(float(best_s),3),
        dur_audio=round(dur,2),gt_dur=round(float(cf.intervals[-1][1]),2),
        resid_med=round(float(np.median(resid)),3),
        resid_early=round(float(np.median(resid[:8])),3) if len(resid)>=8 else None,
        resid_late=round(float(np.median(resid[-8:])),3) if len(resid)>=8 else None,
        resid_std=round(float(np.std(resid)),3), n_bounds=len(real))

out=[]
for af in sorted(AUDIO.glob("*.m4a")):
    r=analyze(af)
    if r: out.append(r); print(r)
json.dump(out,open(REPO/"scratchpad/offset_xcorr_results.json","w"),indent=2)
