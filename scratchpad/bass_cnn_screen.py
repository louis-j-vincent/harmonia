import numpy as np
from collections import Counter

d = np.load('data/cache/rwc/rwc_bp48_fixed.npz', allow_pickle=True)
labels = d['labels']; root = d['root'].astype(int); song = d['song_id']
fabs = d['feat48_abs']            # [ch_on, ch_nt, bass, treble], each L2-normed, ABSOLUTE
bass_abs = fabs[:, 24:36]         # 12-dim absolute bass chroma (pooled over snippet)

DEG = {'1':0,'b2':1,'2':2,'b3':3,'3':4,'4':5,'#4':6,'b5':6,'5':7,'#5':8,'b6':8,
       '6':9,'bb7':9,'b7':10,'7':11,'#7':0,'9':2,'b9':1,'#9':3,'11':5,'#11':6,'13':9,'b13':8}
def sounding_bass(lab, r):
    if '/' in lab:
        inv = lab.split('/')[1].strip()
        if inv in DEG:
            return (r + DEG[inv]) % 12
    return r % 12

bass_true = np.array([sounding_bass(l, root[i]) for i,l in enumerate(labels)])
argm = bass_abs.argmax(1)

print("=== NON-LEARNED SCREEN (n=%d) ==="%len(labels))
print("  bass-argmax == sounding-bass : %.4f" % (argm==bass_true).mean())
print("  bass-argmax == functional-root: %.4f" % (argm==(root%12)).mean())
# only on inversions
inv_mask = np.array(['/' in l and l.split('/')[1].strip() in DEG for l in labels])
print("  [inversions only n=%d] argmax==sounding: %.4f  argmax==func-root: %.4f"
      % (inv_mask.sum(), (argm[inv_mask]==bass_true[inv_mask]).mean(),
         (argm[inv_mask]==(root[inv_mask]%12)).mean()))
np.savez('scratchpad/bass_cnn_prep.npz', bass_abs=bass_abs, fabs=fabs,
         bass_true=bass_true, root=root%12, song=song, argm=argm)
print("saved prep")
