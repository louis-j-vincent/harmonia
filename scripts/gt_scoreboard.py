"""GT self-correction scoreboard: decode each matched (audio, iReal-GT) pair and
score the LOOP READ against GT — key, chord-vocabulary Jaccard, loop period, loop
pattern.  Alignment-free (compares loop structure + vocab, not per-bar-time), so it
directly measures loop-detection quality without needing an aligner.
"""
import sys, os
sys.path.insert(0, "/Users/vincente/Documents/Projets Perso/Code/harmonia")
os.environ["HARMONIA_MUSX_DIR"]="/Users/vincente/Documents/Projets Perso/Code/harmonia/harmonia/third_party/ISMIR2019-Large-Vocabulary-Chord-Recognition"
os.environ["HARMONIA_OCCAM_POSTPASS"]="1"
import logging; logging.disable(logging.WARNING)
from pathlib import Path
from collections import Counter
import numpy as np
from harmonia.data.ireal_corpus import load_playlist, sectionized_measures, split_chords
from harmonia.models.chord_pipeline_v1 import infer_chords_v1, detect_loop_pattern, NOTE

SD = Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-Code-harmonia/8a011198-4935-4f2e-a73e-da83232ee2cd/scratchpad")
_L = {'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}

def _parse(label):
    from harmonia.models.chord_pipeline_v1 import _parse_harte_label
    return _parse_harte_label(label)[0]

def rpc(t):
    t=t.strip()
    if not t or t[0] not in _L: return None
    pc=_L[t[0]]; i=1
    while i<len(t) and t[i] in '#b': pc+=1 if t[i]=='#' else -1; i+=1
    return pc%12

def gt_bar_roots(tune):
    out=[]
    for _l,meas in sectionized_measures(tune):
        r=None
        for tok in split_chords(meas):
            s=tok.strip()
            if s and s[0] not in 'npW': r=rpc(s)
            if r is not None: break
        if r is not None: out.append(r)
    return out

# matched set: (name, video_id, gt_title_substr, corpus)
MATCHED = [
    ("billie_jean", "Zi_XLOBDo_Y", "billie jean", "pop400"),
    ("let_it_be", "QDYfEBY9NM4", "let it be", "pop400"),
    ("chiquitita", "p9Y3N_2xUsw", "chiquitita", "pop400"),
    ("autumn_leaves", "zTVlrOk9a8M", "autumn leaves", "jazz1460"),
    ("easy", "saaLW0jiiUE", "easy", "pop400"),
    ("chain_of_fools", "5C4FnlftQt4", "chain of fools", "pop400"),
]
AUDIO = {
    "Zi_XLOBDo_Y":"michael_jackson_billie_jean_official_video",
    "QDYfEBY9NM4":"let_it_be_remastered_2009",
    "p9Y3N_2xUsw":"abba_chiquitita_official_lyric_video",
    "zTVlrOk9a8M":"autumn_leaves",
    "saaLW0jiiUE":"the_commodores_easy_1977",
    "5C4FnlftQt4":"aretha_franklin_chain_of_fools_official_lyric_video",
}
POP = {t.title.lower(): t for t in load_playlist(Path("data/ireal/pop400.txt"))}
JAZZ = {t.title.lower(): t for t in load_playlist(Path("data/ireal/jazz1460.txt"))}

def find_gt(sub, corpus):
    d = POP if corpus=="pop400" else JAZZ
    for k,t in d.items():
        if sub in k: return t
    return None

import subprocess
rows=[]
for name, vid, sub, corpus in MATCHED:
    wav = SD/f"{vid}.wav"
    if not wav.exists():
        src = f"docs/audio/{AUDIO[vid]}.m4a"
        subprocess.run(["ffmpeg","-y","-i",src,"-ac","2","-ar","44100",str(wav)],capture_output=True)
    gt = find_gt(sub, corpus)
    if gt is None:
        print(f"{name}: GT not found"); continue
    gt_roots = gt_bar_roots(gt)
    m=len(gt_roots); post=np.full((m,12),1e-3)
    for i,r in enumerate(gt_roots): post[i,r]=1.0
    gt_loop = detect_loop_pattern(gt_roots, post, list(range(m)))
    gt_voc = set(gt_roots)
    # decode
    ch = infer_chords_v1(wav, cache_dir=SD/"cache", feature_frontend="nnls24",
        bass_frontend="musx", quality_frontend="musx", segment_source="nnls")
    dec_roots = [_parse(c["label"]) for c in ch.chords if c["label"]!="N"]
    dec_roots = [r for r in dec_roots if r is not None]
    dec_voc = set(dec_roots)
    md=len(dec_roots); dp=np.full((md,12),1e-3)
    for i,r in enumerate(dec_roots): dp[i,r]=1.0
    dec_loop = detect_loop_pattern(dec_roots, dp, list(range(md)))
    jac = len(gt_voc & dec_voc)/max(len(gt_voc|dec_voc),1)
    rows.append(dict(name=name, gt_key=str(getattr(gt,'key','?')),
        gt_period=gt_loop[0] if gt_loop else None,
        gt_pattern=[NOTE[r] for r in gt_loop[1]] if gt_loop else None,
        dec_period=dec_loop[0] if dec_loop else None,
        dec_pattern=[NOTE[r] for r in dec_loop[1]] if dec_loop else None,
        vocab_jaccard=round(jac,2), gt_voc=sorted(NOTE[r] for r in gt_voc),
        dec_voc=sorted(NOTE[r] for r in dec_voc)))

def _parse(label):
    from harmonia.models.chord_pipeline_v1 import _parse_harte_label
    return _parse_harte_label(label)[0]

for r in rows:
    print(f"\n{r['name']}:")
    print(f"  GT   period {r['gt_period']} pattern {r['gt_pattern']} vocab {r['gt_voc']}")
    print(f"  DEC  period {r['dec_period']} pattern {r['dec_pattern']} vocab {r['dec_voc']}")
    print(f"  vocab_jaccard {r['vocab_jaccard']}  period_match {r['gt_period']==r['dec_period']}")
import json
json.dump(rows, open(SD/"scoreboard.json","w"), indent=1)
