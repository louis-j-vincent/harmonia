"""Plot SIMPLE unconditional bass-PC CV results vs prior attempts."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
res = json.loads((REPO / "scratchpad/bass_simple_result.json").read_text())
S = res["summary"]
def g(k): return S.get(k, [float("nan"), 0.0])

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

# --- Panel 1: bass-PC accuracy, raw vs renorm, split rootpos/inv ---
groups = ["all", "root-pos", "inversions"]
raw = [g("raw_bass_acc_all"), g("raw_bass_acc_rootpos"), g("raw_bass_acc_inv")]
rr  = [g("rr_bass_acc_all"),  g("rr_bass_acc_rootpos"),  g("rr_bass_acc_inv")]
x = np.arange(3); w = 0.36
ax[0].bar(x - w/2, [m for m,_ in raw], w, yerr=[s for _,s in raw], capsize=4,
          label="RAW (absolute)", color="#4C72B0")
ax[0].bar(x + w/2, [m for m,_ in rr], w, yerr=[s for _,s in rr], capsize=4,
          label="RENORM (root-anchored)", color="#DD8452")
# prior references (bass-pc on true inversions)
ax[0].axhline(0.681, ls="--", c="#55A868", lw=1.2, label="v1 pooled-gated (inv) 0.681")
ax[0].axhline(0.578, ls=":",  c="#C44E52", lw=1.2, label="v2 temporal (inv) 0.578")
ax[0].axhline(1/12, ls="-", c="gray", lw=0.8, alpha=0.6)
ax[0].set_xticks(x); ax[0].set_xticklabels(groups)
ax[0].set_ylabel("bass-PC accuracy"); ax[0].set_ylim(0, 1)
ax[0].set_title("Unconditional bass-PC (12-way)"); ax[0].legend(fontsize=8, loc="lower left")
for i,(m,_) in enumerate(raw): ax[0].text(i-w/2, m+0.02, f"{m:.2f}", ha="center", fontsize=8)
for i,(m,_) in enumerate(rr):  ax[0].text(i+w/2, m+0.02, f"{m:.2f}", ha="center", fontsize=8)

# --- Panel 2: root accuracy — does bass head help? ---
labels = ["baseline", "S1 b0.5", "S2 ensemble"]
allk = [g("root_acc_all"), g("s1_root_all_b0.5"), g("s2_root_all")]
rpk  = [g("root_acc_rootpos"), g("s1_root_rp_b0.5"), g("s2_root_rp")]
ivk  = [g("root_acc_inv"), g("s1_root_inv_b0.5"), g("s2_root_inv")]
x2 = np.arange(3); w2 = 0.26
ax[1].bar(x2-w2, [m for m,_ in allk], w2, yerr=[s for _,s in allk], capsize=3, label="all", color="#4C72B0")
ax[1].bar(x2,    [m for m,_ in rpk],  w2, yerr=[s for _,s in rpk],  capsize=3, label="root-pos", color="#55A868")
ax[1].bar(x2+w2, [m for m,_ in ivk],  w2, yerr=[s for _,s in ivk],  capsize=3, label="inversions", color="#C44E52")
ax[1].axhline(allk[0][0], ls="--", c="#4C72B0", lw=1, alpha=0.7)
ax[1].set_xticks(x2); ax[1].set_xticklabels(labels)
ax[1].set_ylabel("root accuracy"); ax[1].set_ylim(0, 1)
ax[1].set_title("Root accuracy: does the bass head help?"); ax[1].legend(fontsize=8)

fig.suptitle(f"RWC {res['corpus']} — simple unconditional bass head, {res['seeds']}-seed CV", fontsize=12)
fig.tight_layout()
out = REPO / "docs/plots/bass_simple_rwc.png"
fig.savefig(out, dpi=110); print("wrote", out)
