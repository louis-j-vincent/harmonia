import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent

fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

# Q1
q1n = ["12\nonset", "24\nonset+note", "24\nbass+treble", "48\nall blocks"]
q1v = [0.908, 0.932, 0.954, 0.971]
ax[0].bar(range(4), q1v, color=["#bbb", "#89c", "#59a", "#27a"])
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(q1n, fontsize=8)
ax[0].set_ylim(0.85, 1.0); ax[0].set_title("Q1: dimensionality (q5 balanced acc)\nroot-relative frame")
for i, v in enumerate(q1v): ax[0].text(i, v + .002, f"{v:.3f}", ha="center", fontsize=8)
ax[0].axhline(0.932, ls=":", c="gray"); ax[0].text(2, 0.936, "register-24 > channel-24 (equal dim)", fontsize=7, ha="center")

# Q2
q2n = ["raw\nabsolute", "relative\nKEY (C)", "relative\nROOT (A)"]
q2v = [0.811, 0.819, 0.968]
ax[1].bar(range(3), q2v, color=["#bbb", "#c96", "#2a6"])
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(q2n, fontsize=8)
ax[1].set_ylim(0.75, 1.0); ax[1].set_title("Q2: normalization (48-dim)\nOption A wins +15pp")
for i, v in enumerate(q2v): ax[1].text(i, v + .003, f"{v:.3f}", ha="center", fontsize=8)

# Q3
q3n = ["BP48\nin-domain", "NNLS\nin-domain", "CROSS\nNNLS->BP48", "CROSS\n+z-norm"]
q3v = [0.923, 0.745, 0.765, 0.673]
ax[2].bar(range(4), q3v, color=["#27a", "#888", "#c86", "#a44"])
ax[2].set_xticks(range(4)); ax[2].set_xticklabels(q3n, fontsize=8)
ax[2].set_ylim(0.6, 1.0); ax[2].set_title("Q3: bridge (train->test)\nnative BP48 >> ported NNLS")
for i, v in enumerate(q3v): ax[2].text(i, v + .004, f"{v:.3f}", ha="center", fontsize=8)

for a in ax: a.set_ylabel("balanced acc (5-way q5)")
plt.tight_layout()
out = REPO / "docs/plots/bridge_nnls_bp48.png"
plt.savefig(out, dpi=110)
print("wrote", out)
