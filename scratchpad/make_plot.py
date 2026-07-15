import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
ROOT=Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")

QUALS=['maj','min','dom','hdim','dim']
# per-class recall rows (test set, from runs)
rows = {
 'abs no-rot (baseline)'      : [0.66,0.73,0.53,0.40,0.92],
 'root-rel (oracle root)'     : [0.74,0.79,0.70,0.40,0.95],
 'root-rel + trigram ctx (oracle)': [0.73,0.87,0.70,0.40,0.97],
 'cascade hard (pred root)'   : [0.70,0.81,0.68,0.40,0.89],
 'cascade marginalized'       : [0.73,0.82,0.67,0.40,0.97],
 'cascade marg + dom-w1.8 (SHIP)': [0.64,0.79,0.78,0.40,0.95],
}
bal = {k: np.mean(v) for k,v in rows.items()}

C = ['#4C78A8','#F58518','#54A24B','#B279A2','#9D755D']  # brand-neutral categorical
fig, axes = plt.subplots(1,2, figsize=(13,5.2), gridspec_kw={'width_ratios':[2.2,1]})

ax=axes[0]
labels=list(rows.keys()); x=np.arange(5); w=0.13
for i,lab in enumerate(labels):
    ax.bar(x+(i-2.5)*w, rows[lab], w, label=lab, color=C[i%len(C)],
           edgecolor='white', linewidth=.4)
ax.axhline(0.70, ls='--', lw=1, color='#666'); ax.text(4.35,0.71,'dom target 0.70',fontsize=8,color='#666')
ax.set_xticks(x); ax.set_xticklabels(QUALS); ax.set_ylabel('per-class recall (test)')
ax.set_ylim(0,1.0); ax.set_title('Quality head: per-class recall by architecture / normalization')
ax.legend(fontsize=7.5, ncol=2, loc='upper left', framealpha=.9)
ax.grid(axis='y', alpha=.25)

ax=axes[1]
ks=list(bal.keys()); vals=[bal[k] for k in ks]
yy=np.arange(len(ks))
ax.barh(yy, vals, color='#4C78A8', edgecolor='white')
for i,v in enumerate(vals): ax.text(v+.005,i,f'{v:.3f}',va='center',fontsize=8)
ax.set_yticks(yy); ax.set_yticklabels([k.replace(' (','\n(') for k in ks], fontsize=7)
ax.invert_yaxis(); ax.set_xlim(0,0.85); ax.set_xlabel('balanced acc (mean recall)')
ax.set_title('Balanced accuracy')
ax.grid(axis='x', alpha=.25)
fig.suptitle('Structured multi-head quality: root-relative rotation + learned trigram context + root-marginalization\n'
             'Billboard NNLS (114,741 chords / 884 songs, song-stratified 80/10/10). Root head 89.0%.', fontsize=10)
fig.tight_layout(rect=[0,0,1,0.94])
fig.savefig(ROOT/"docs/plots/architecture_comparison.png", dpi=130)
print("saved docs/plots/architecture_comparison.png")
