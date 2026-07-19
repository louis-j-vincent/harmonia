import sys
from pathlib import Path
from harmonia.models.chord_pipeline_v1 import infer_chords_v1
AUD={"chain_of_fools":"scratchpad/aretha_franklin_chain_of_fools_official_lyric_video.wav",
 "autumn_leaves":"scratchpad/autumn_leaves_easy_jazz_piano_piano_cover_sheets.wav",
 "goodbye_ybr":"scratchpad/elton_john_goodbye_yellow_brick_road_lyrics.wav"}
name=sys.argv[1]
ch=infer_chords_v1(Path(AUD[name]), seventh_gate=0.0, cache_dir=Path("data/cache"))
s=ch.sections or []
print("RESULT",name,len(s),"".join(x["label"] for x in s))
