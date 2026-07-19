import json, warnings
from pathlib import Path
import numpy as np, librosa, mirdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
REPO=Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
AUDIO=REPO/"docs/audio"; PLOTS=REPO/"docs/plots"
vid=json.loads((REPO/"docs/plots/.yt_video_ids.json").read_text())
corpus={}
for n in ["billboard_search_results.json","billboard_search_results_60.json"]:
    p=REPO/"scratchpad"/n
    if p.exists():
        try:corpus.update(json.loads(p.read_text()))
        except:pass
vid2track={t:v.get("best",[None])[0] for t,v in corpus.items()}
vid2track={v:t for t,v in vid2track.items() if v}
ds=mirdata.initialize("billboard")

fig,axes=plt.subplots(5,1,figsize=(14,16))
ai=0; summary=[]
for af in sorted(AUDIO.glob("*.m4a")):
    stem=af.stem; v=vid.get(f"inferred_{stem}.html"); tid=vid2track.get(v) if v else None
    if not tid: continue
    cf=ds.track(tid).chords_full
    labels=[str(l) for l in cf.labels]
    bounds=[float(a) for (a,b) in cf.intervals]
    real=[(bounds[i],labels[i]) for i in range(len(bounds)) if labels[i] not in("N","X") and (i==0 or labels[i]!=labels[i-1])]
    y,sr=librosa.load(str(af),sr=22050,mono=True)
    hop=512
    oenv=librosa.onset.onset_strength(y=y,sr=sr,hop_length=hop)
    onsets=librosa.onset.onset_detect(onset_envelope=oenv,sr=sr,hop_length=hop,units='time',backtrack=True)
    # first strong onset = first onset whose strength above 40% of head max
    ot=librosa.times_like(oenv,sr=sr,hop_length=hop)
    # strength at each detected onset
    strengths=np.interp(onsets,ot,oenv)
    thr=0.4*np.max(oenv[:int(30*sr/hop)])
    strong=onsets[strengths>thr]
    first_strong=strong[0] if len(strong) else onsets[0]
    gt_first=real[0][0] if real else 0.0
    offset=first_strong-gt_first   # +: audio later than GT; shift GT +offset to match
    # drift test: residual of (gt+offset) vs nearest onset, early vs late
    def resid_at(chords):
        r=[]
        for t,_ in chords:
            tt=t+offset; j=np.argmin(np.abs(onsets-tt)); d=onsets[j]-tt
            if abs(d)<0.35: r.append(d)
        return np.array(r)
    early=resid_at(real[:12]); late=resid_at(real[-15:])
    summary.append(dict(stem=stem[:32],tid=tid,gt_first=round(gt_first,2),
        first_strong_onset=round(float(first_strong),2),
        offset_s=round(float(offset),2),
        dur_mismatch=round(float(len(y)/sr-cf.intervals[-1][1]),2),
        resid_early_med=round(float(np.median(early)),3) if len(early) else None,
        resid_late_med=round(float(np.median(late)),3) if len(late) else None,
        drift=round(float(np.median(late)-np.median(early)),3) if len(early) and len(late) else None))
    # plot 0-22s
    ax=axes[ai]; ai+=1
    seg=int(22*sr); yt=y[:seg]; tt=np.arange(len(yt))/sr
    ax.plot(tt,yt/np.max(np.abs(yt)+1e-9)*0.9,color="#b9b09a",lw=0.4)
    for t,l in real:
        if t<22: ax.axvline(t,color="#1f8a5b",ls="-",lw=1.3,alpha=0.8)
    for t,l in real:
        tc=t+offset
        if tc<22: ax.axvline(tc,color="#8a2b2b",ls="--",lw=1.1,alpha=0.7)
    for o in onsets[onsets<22]: ax.axvline(o,color="#2b5f8a",ls=":",lw=0.6,alpha=0.5,ymin=0,ymax=0.15)
    ax.axvline(first_strong,color="orange",lw=2,alpha=0.7)
    ax.set_title(f"{stem[:40]}  offset={offset:+.2f}s  drift={summary[-1]['drift']}  durΔ={summary[-1]['dur_mismatch']:+.1f}s",fontsize=9)
    ax.set_xlim(0,22); ax.set_yticks([])
axes[0].legend(handles=[plt.Line2D([],[],color="#1f8a5b",label="GT raw"),
    plt.Line2D([],[],color="#8a2b2b",ls="--",label="GT+offset"),
    plt.Line2D([],[],color="orange",label="first strong onset"),
    plt.Line2D([],[],color="#2b5f8a",ls=":",label="onsets")],fontsize=8,loc="upper right")
plt.tight_layout(); plt.savefig(PLOTS/"billboard_gt_offset_first20s.png",dpi=90)
print(json.dumps(summary,indent=2))
json.dump(summary,open(REPO/"scratchpad/offset_final_results.json","w"),indent=2)
