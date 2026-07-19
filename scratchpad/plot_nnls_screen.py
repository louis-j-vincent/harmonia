import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO=Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
b=json.load(open(REPO/"scratchpad/nnls_batch_results.json"))
s=json.load(open(REPO/"scratchpad/nnls_screen_results.json"))
rows=[]
for r in b: rows.append((r['sid'],r['nnls_pm'],r['bp48_pm'],r['nnls_bass_root'],r['bp48_bass_root']))
for r in s: rows.append((r['sid'],r['nnls_pm_full24'],r['bp48_pm_full48'],r['nnls_bass_root_acc'],r['bp48_bass_root_acc']))
rows.sort(key=lambda x:x[1])
sid=[r[0].replace('bb_','') for r in rows]
npm=np.array([r[1] for r in rows]); bpm=np.array([r[2] for r in rows])
nbr=np.array([r[3] for r in rows]); bbr=np.array([r[4] for r in rows])
x=np.arange(len(sid)); w=0.4

fig,ax=plt.subplots(1,3,figsize=(17,5.5))
ax[0].bar(x-w/2,npm,w,label='NNLS (ours)',color='#2a9d8f')
ax[0].bar(x+w/2,bpm,w,label='BP48',color='#e76f51')
ax[0].axhline(4.42,ls='--',c='#264653',lw=1); ax[0].text(0,4.5,'McGill NNLS 4.42',fontsize=8)
ax[0].axhline(2.77,ls=':',c='#8a5a44',lw=1); ax[0].text(0,2.85,'McGill-report BP48 2.77',fontsize=8)
ax[0].set_title('Chroma sharpness: peak/mean (higher=sharper)\nNNLS on OUR youtube audio wins 15/15')
ax[0].set_xticks(x); ax[0].set_xticklabels(sid,rotation=90,fontsize=7); ax[0].legend(); ax[0].set_ylabel('peak/mean')

ax[1].bar(x-w/2,nbr,w,label='NNLS bass-argmax',color='#2a9d8f')
ax[1].bar(x+w/2,bbr,w,label='BP48 bass-argmax',color='#e76f51')
ax[1].set_title('Untrained bass-argmax -> root acc\nNNLS +10pp mean (0.576 vs 0.476)')
ax[1].set_xticks(x); ax[1].set_xticklabels(sid,rotation=90,fontsize=7); ax[1].legend(); ax[1].set_ylabel('root acc')

# trained CV summary
names=['NNLS-24\nLR','BP48-48\nLR','NNLS-24\nMLP','BP48-48\nMLP','NNLS\nargmax','BP48\nargmax']
vals=[0.485,0.467,0.454,0.447,0.530,0.439]
cols=['#2a9d8f','#e76f51','#2a9d8f','#e76f51','#2a9d8f','#e76f51']
ax[2].bar(range(6),vals,color=cols)
for i,v in enumerate(vals): ax[2].text(i,v+0.005,f'{v:.2f}',ha='center',fontsize=8)
ax[2].set_xticks(range(6)); ax[2].set_xticklabels(names,fontsize=8)
ax[2].set_title('Root acc, grouped 5-fold CV (25 songs)\nTRAINED head nearly closes gap: +1.8pp LR')
ax[2].set_ylabel('root acc'); ax[2].set_ylim(0,0.6)
fig.suptitle('NNLS vs BP48 on identical OUR-OWN Billboard audio — confound broken (sharpness=algorithm), but trained-head lever marginal',fontsize=12)
fig.tight_layout()
out=REPO/"docs/plots/nnls_vs_bp48_our_audio.png"; fig.savefig(out,dpi=110,bbox_inches='tight')
print('saved',out)
