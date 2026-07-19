import os, sys, time, json
os.environ["HARMONIA_SECTION_FALLBACK"]="0"  # capture PURE symbolic §10b
from pathlib import Path
from harmonia.models.chord_pipeline_v1 import infer_chords_v1
from harmonia.models.section_structure import librosa_laplacian_sections

AUD={
 "chain_of_fools":"scratchpad/aretha_franklin_chain_of_fools_official_lyric_video.wav",
 "autumn_leaves":"scratchpad/autumn_leaves_easy_jazz_piano_piano_cover_sheets.wav",
 "goodbye_ybr":"scratchpad/elton_john_goodbye_yellow_brick_road_lyrics.wav",
}
name=sys.argv[1]
p=Path(AUD[name])
t0=time.time()
chart=infer_chords_v1(p, seventh_gate=0.0, cache_dir=Path("data/cache"))
sym=chart.sections or []
t1=time.time()
lap=librosa_laplacian_sections(p)
t2=time.time()
def fmt(secs):
    return [{"s":round(s["start_s"],1),"e":round(s["end_s"],1),"bars":s["n_bars"],"L":s["label"]} for s in secs]
res={"name":name,"infer_s":round(t1-t0,1),"lap_s":round(t2-t1,1),
     "symbolic":{"n":len(sym),"labels":"".join(s["label"] for s in sym),"secs":fmt(sym)},
     "librosa":{"n":len(lap),"labels":"".join(s["label"] for s in lap),"secs":fmt(lap)}}
Path("scratchpad/sec_%s.json"%name).write_text(json.dumps(res,indent=1))
print(json.dumps({k:res[k] for k in ("name","infer_s","lap_s")}, ))
print("SYM",res["symbolic"]["n"],res["symbolic"]["labels"])
print("LAP",res["librosa"]["n"],res["librosa"]["labels"])
