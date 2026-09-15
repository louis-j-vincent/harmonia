"""Directive 2: multi-factor section-boundary model. Per matched (audio, iReal-GT)
song, generate bar-grid-locked candidate boundaries, extract per-song-normalized
features (chord recurrence, drum-fill/HPSS tail, energy/timbre novelty, N.C.
adjacency, phrase-position, harmonic-rhythm change), label by GT-boundary proximity,
and train a song-held-out logistic — reporting per-feature weights + F1 vs the
largest-unit-only baseline. REAL AUDIO ONLY.
"""
import sys, os
sys.path.insert(0, "/Users/vincente/Documents/Projets Perso/Code/harmonia")
os.environ["HARMONIA_MUSX_DIR"]="/Users/vincente/Documents/Projets Perso/Code/harmonia/harmonia/third_party/ISMIR2019-Large-Vocabulary-Chord-Recognition"
os.environ["HARMONIA_OCCAM_POSTPASS"]="1"
import logging; logging.disable(logging.WARNING)
from pathlib import Path
import numpy as np, librosa
from harmonia.models.chord_pipeline_v1 import infer_chords_v1
from scripts.render_youtube_chart import chart_to_interactive_inputs
from harmonia.output.chart_interactive import render_interactive
from harmonia.output.chart_model import payload_from_chart_html, to_chart_model, _bar_root_seq
from harmonia.data.ireal_corpus import load_playlist, sectionized_measures

SD = Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-Code-harmonia/8a011198-4935-4f2e-a73e-da83232ee2cd/scratchpad")
MATCHED = [
    ("billie_jean","Zi_XLOBDo_Y","billie jean","pop400"),
    ("let_it_be","QDYfEBY9NM4","let it be","pop400"),
    ("stand_by_me","hwZNL7QVJjE","stand by me","pop400"),
    ("easy","saaLW0jiiUE","easy","pop400"),
    ("chain_of_fools","5C4FnlftQt4","chain of fools","pop400"),
    ("autumn","zTVlrOk9a8M","autumn leaves","jazz1460"),
]
AUDIO={"Zi_XLOBDo_Y":"michael_jackson_billie_jean_official_video","QDYfEBY9NM4":"let_it_be_remastered_2009",
    "hwZNL7QVJjE":"ben_e_king_stand_by_me_audio","saaLW0jiiUE":"the_commodores_easy_1977",
    "5C4FnlftQt4":"aretha_franklin_chain_of_fools_official_lyric_video","zTVlrOk9a8M":"autumn_leaves"}
POP={t.title.lower():t for t in load_playlist(Path("data/ireal/pop400.txt"))}
JAZZ={t.title.lower():t for t in load_playlist(Path("data/ireal/jazz1460.txt"))}

def gt_boundary_fracs(sub, corpus):
    d = POP if corpus=="pop400" else JAZZ
    t=next((v for k,v in d.items() if sub in k),None)
    if not t: return None
    secs=sectionized_measures(t); n=len(secs)
    if n<2: return None
    bounds=[i for i in range(1,n) if secs[i][0]!=secs[i-1][0]]
    return [b/n for b in bounds]

FEAT_NAMES=["chord_recur","phrase_restart","drum_fill","energy_nov","timbre_nov","nc_adj","phrase_pos","harm_rhythm"]

def build(name, vid, sub, corpus):
    wav=SD/f"{vid}.wav"
    if not wav.exists():
        import subprocess; subprocess.run(["ffmpeg","-y","-i",f"docs/audio/{AUDIO[vid]}.m4a","-ac","2","-ar","44100",str(wav)],capture_output=True)
    ch=infer_chords_v1(wav, cache_dir=SD/"cache", feature_frontend="nnls24", bass_frontend="musx", quality_frontend="musx", segment_source="nnls")
    obj,cds=chart_to_interactive_inputs(ch,"x","t",bar1_offset_beats=int(getattr(ch,"grid_anchor_beats",0) or 0))
    out=SD/f"f_{vid}.html"; render_interactive(obj,cds,out,bars_per_row=4,sections=ch.sections)
    m=to_chart_model(payload_from_chart_html(out))
    bars=[bar for s in m["sections"] for bar in s["bars"]]; n=m["nBars"]
    R=_bar_root_seq(bars,n)
    # per-bar time from chord t0 (snapped real) — first chord in each bar
    bar_t=[None]*n
    for s in m["sections"]:
        for bar in s["bars"]:
            for c in bar:
                bi=c.get("bar");
                if bi is not None and 0<=bi<n and bar_t[bi] is None: bar_t[bi]=c["t0"]
    # fill missing bar times by interpolation
    dur=ch.duration_s
    known=[(i,t) for i,t in enumerate(bar_t) if t is not None]
    if len(known)<2: return None
    xs=[i for i,_ in known]; ts=[t for _,t in known]
    bar_time=np.interp(np.arange(n), xs, ts, left=ts[0], right=dur)
    nc_bar=[any(c.get("nc") for c in bar) for bar in bars]+[False]*(n-len(bars))
    # audio features per bar: percussive RMS (HPSS), full RMS, spectral centroid
    y,sr=librosa.load(str(wav),sr=22050)
    S=np.abs(librosa.stft(y,n_fft=2048,hop_length=512))
    H,P=librosa.decompose.hpss(S)
    hop=512/sr
    perc=np.sqrt((P**2).mean(0)); rms=np.sqrt((S**2).mean(0)); cent=librosa.feature.spectral_centroid(S=S,sr=sr)[0]
    def agg(arr, t0, t1):
        i0,i1=int(t0/hop),max(int(t0/hop)+1,int(t1/hop))
        seg=arr[i0:min(i1,len(arr))]; return float(seg.mean()) if len(seg) else 0.0
    perc_bar=np.array([agg(perc,bar_time[b],bar_time[b+1] if b+1<n else dur) for b in range(n)])
    rms_bar=np.array([agg(rms,bar_time[b],bar_time[b+1] if b+1<n else dur) for b in range(n)])
    cent_bar=np.array([agg(cent,bar_time[b],bar_time[b+1] if b+1<n else dur) for b in range(n)])
    # normalize per song
    def z(a): a=np.asarray(a,float); s=a.std(); return (a-a.mean())/s if s>1e-9 else a*0
    perc_z=z(perc_bar); rms_z=z(rms_bar); cent_z=z(cent_bar)
    gt=gt_boundary_fracs(sub,corpus) or []
    gt_bars=set(int(round(f*n)) for f in gt)
    # candidates: every bar that is a 4-bar multiple (bar-grid-locked)
    rows=[]
    for b in range(4,n-3,1):
        # chord recurrence contrast: dissimilarity of the 4 bars before vs the loop
        before=R[b-4:b]; after=R[b:b+4]
        recur=1.0 - (sum(1 for x,y in zip(before,after) if x==y)/4)
        # phrase RESTART: does the new content repeat the phrase from 8 bars ago?
        prestart=(sum(1 for x,y in zip(R[b:b+4],R[b-8:b-4]) if x==y)/4) if b>=8 else 0.0
        drum=perc_z[b-1]                        # percussive tail in the bar BEFORE
        enov=abs(rms_z[b]-rms_z[b-1])
        tnov=abs(cent_z[b]-cent_z[b-1])
        ncadj=1.0 if (nc_bar[b] or (b>0 and nc_bar[b-1]) or (b+1<n and nc_bar[b+1])) else 0.0
        phrase=-min(b%8,8-(b%8))                # 0 at 8-bar multiples
        # harmonic rhythm change: distinct roots in window after vs before
        hr=abs(len(set(R[b:b+4]))-len(set(R[b-4:b])))
        label=1 if any(abs(b-g)<=1 for g in gt_bars) else 0
        rows.append(([recur,prestart,drum,enov,tnov,ncadj,float(phrase),float(hr)],label,b))
    return dict(name=name,n=n,rows=rows,gt_bars=gt_bars)

data=[]
for name,vid,sub,corpus in MATCHED:
    d=build(name,vid,sub,corpus)
    if d and d["gt_bars"]:
        pos=sum(1 for _,l,_ in d["rows"] if l==1)
        print(f"{name}: {d['n']} bars, {len(d['rows'])} candidates, {pos} positive, {len(d['gt_bars'])} GT bounds")
        data.append(d)
import pickle
pickle.dump(data, open(SD/"secfeat.pkl","wb"))
print("saved", len(data), "songs")
