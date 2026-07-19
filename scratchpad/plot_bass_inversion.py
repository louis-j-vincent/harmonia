"""Compact diagnostic for the RWC bass/inversion head 6-seed CV."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# per-seed values parsed from scratchpad/bass_inversion_full.log
rootpos = [.693,.709,.654,.700,.655,.729]
inv     = [.361,.205,.276,.300,.330,.208]
err_b   = [.476,.606,.456,.536,.540,.675]   # err on bass BEFORE
basspc  = [.733,.648,.579,.574,.686,.762]
inv_orc = [.459,.404,.394,.429,.438,.370]   # inv root acc AFTER oracle redirect
err_a   = [.338,.500,.370,.415,.436,.580]   # err on bass AFTER oracle redirect
rp_afterblind = [.620,.646,.580,.615,.588,.654]
inv_afterblind= [.410,.283,.339,.385,.352,.254]

def ms(x): return np.mean(x), np.std(x)

fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))

# Panel 1: root acc split, + billboard reference
labels = ["root-pos", "inversion"]
m = [ms(rootpos)[0], ms(inv)[0]]; s = [ms(rootpos)[1], ms(inv)[1]]
ax[0].bar([0,1], m, yerr=s, color=["#4C72B0","#C44E52"], capsize=5, width=.55)
ax[0].scatter([1.35,1.35],[0.553,0.161], color="gray", marker="_", s=400)
ax[0].text(1.45,0.553,"BB 55.3%",fontsize=8,va="center")
ax[0].text(1.45,0.161,"BB 16.1%",fontsize=8,va="center")
ax[0].set_xticks([0,1]); ax[0].set_xticklabels(labels)
ax[0].set_ylim(0,0.85); ax[0].set_ylabel("root accuracy")
ax[0].set_title(f"ROOT acc by position\n(−41pp gap; Billboard −39pp)")
for i,v in enumerate(m): ax[0].text(i,v+s[i]+.01,f"{v:.0%}",ha="center",fontsize=9)

# Panel 2: bass-pc head own accuracy vs chance / vs "== root"
ax[1].bar([0,1,2],[ms(basspc)[0],0.085,1/12],
          yerr=[ms(basspc)[1],0.030,0],color=["#55A868","#999","#ccc"],capsize=5,width=.55)
ax[1].set_xticks([0,1,2]); ax[1].set_xticklabels(["bass-pc\nacc","== root\n(sanity)","chance\n1/12"],fontsize=8)
ax[1].set_ylim(0,0.85); ax[1].set_ylabel("accuracy on true inversions")
ax[1].set_title("BASS-PC head: NEW capability\n(learns sounding bass, not root)")
ax[1].text(0,ms(basspc)[0]+ms(basspc)[1]+.01,f"{ms(basspc)[0]:.0%}",ha="center",fontsize=9)

# Panel 3: interaction — inv root acc & err-on-bass, before vs after (oracle gate)
x = np.arange(2); w=0.35
before = [ms(inv)[0], ms(err_b)[0]]
after  = [ms(inv_orc)[0], ms(err_a)[0]]
be = [ms(inv)[1], ms(err_b)[1]]; ae=[ms(inv_orc)[1],ms(err_a)[1]]
ax[2].bar(x-w/2, before, w, yerr=be, label="root head alone", color="#C44E52", capsize=4)
ax[2].bar(x+w/2, after,  w, yerr=ae, label="+ bass head redirect\n(oracle inv-gate)", color="#4C72B0", capsize=4)
ax[2].set_xticks(x); ax[2].set_xticklabels(["inv root acc\n↑ better","err→bass\n↓ better"])
ax[2].set_ylim(0,0.75); ax[2].legend(fontsize=7,loc="upper right")
ax[2].set_title("Interaction: bass head rescues\nroot-on-inversion (+13.5pp)")
ax[2].annotate("",xy=(0-w/2,before[0]+.03),xytext=(0+w/2,after[0]+.03),
               arrowprops=dict(arrowstyle="->",color="green"))
ax[2].text(0,0.52,"+13.5pp",color="green",ha="center",fontsize=9,fontweight="bold")

fig.suptitle("RWC BP48 bass/inversion head (6-seed CV) — hypothesis: root errors on inversions land on the SOUNDING BASS (54.8%)",
             fontsize=10)
fig.tight_layout(rect=[0,0,1,0.94])
out = "docs/plots/bass_inversion_rwc.png"
fig.savefig(out, dpi=110)
print("wrote", out)
