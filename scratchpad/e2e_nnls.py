import sys
from pathlib import Path
from harmonia.models.chord_pipeline_v1 import infer_chords_v1
AUD={"chain_of_fools":"scratchpad/aretha_franklin_chain_of_fools_official_lyric_video.wav",
 "autumn_leaves":"scratchpad/autumn_leaves_easy_jazz_piano_piano_cover_sheets.wav",
 "goodbye_ybr":"scratchpad/elton_john_goodbye_yellow_brick_road_lyrics.wav"}
name=sys.argv[1]
ch=infer_chords_v1(Path(AUD[name]), cache_dir=Path("data/cache"),
    feature_frontend="nnls24", bass_frontend="nnls24", quality_frontend="nnls24", segment_source="nnls")
s=ch.sections or []
print("RESULT",name,"nsec=%d"%len(s),"labels="+"".join(x["label"] for x in s))
