"""Real-audio gate: old hard-margin vs new Bayes arbitration — log the flips.
Reconstructs both decisions on the same per-bar inputs for henny/just_aint/abba.
"""
import sys, os
sys.path.insert(0, "/Users/vincente/Documents/Projets Perso/Code/harmonia")
os.environ["HARMONIA_MUSX_DIR"]="/Users/vincente/Documents/Projets Perso/Code/harmonia/harmonia/third_party/ISMIR2019-Large-Vocabulary-Chord-Recognition"
os.environ["HARMONIA_OCCAM_POSTPASS"]="1"
from pathlib import Path
import numpy as np
import logging; logging.disable(logging.WARNING)
from harmonia.models import chord_pipeline_v1 as cp

SD = Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-Code-harmonia/8a011198-4935-4f2e-a73e-da83232ee2cd/scratchpad")

# monkeypatch occam_compress_bars to capture inputs + emit both verdicts
orig = cp.occam_compress_bars
def spy(bar_root, bar_qual, bar_post, family_id, **kw):
    res = orig(bar_root, bar_qual, bar_post, family_id, **kw)
    _, _, decisions = res
    bar_conf = kw.get("bar_conf")
    logp = np.log(np.clip(bar_post, 1e-9, None))
    flips = 0; kept_new = 0; kept_old = 0
    for d in decisions:
        if "kept_deviation" not in d: continue
        b = d["bar"]; r_b = bar_root[b]
        # old rule: lr>log(4) and post[r_b]>=0.55
        lr = d["lr"]; post_rb = float(bar_post[b, r_b])
        old_keep = (lr > np.log(4.0)) and (post_rb >= 0.55)
        new_keep = d["kept_deviation"]
        kept_new += int(new_keep); kept_old += int(old_keep)
        if old_keep != new_keep:
            flips += 1
            print(f"    FLIP bar {b}: old={'keep' if old_keep else 'snap'} -> "
                  f"new={'keep' if new_keep else 'snap'}  conf={d['conf']} lr={d['lr']} "
                  f"post={post_rb:.2f} log_odds={d['log_odds']}")
    print(f"    kept: old={kept_old} new={kept_new}, flips={flips}")
    return res
cp.occam_compress_bars = spy

for name, vid in [("henny","gmfcYli6vV4"),("just_aint","pBKx8PyE5qQ"),("abba","p9Y3N_2xUsw")]:
    print(f"=== {name} ===")
    cp.infer_chords_v1(SD/f"{vid}.wav", cache_dir=SD/"cache", feature_frontend="nnls24",
        bass_frontend="musx", quality_frontend="musx", segment_source="nnls")
