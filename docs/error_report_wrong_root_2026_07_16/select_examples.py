"""Select ~18 wrong-root examples from the FIXED-corpus test manifest,
stratified by error category (P4/P5, inversion-related, other), and write
examples_manifest.json for the fetch/plot/build-html steps.

Categories (best-effort, from the label string + root probs):
  - "p4p5": predicted root is a perfect 4th or 5th (±5 or ±7 semitones) from GT root
  - "inversion": GT label carries a "/bass" tag AND the bass PC != GT root
    (i.e. this chord itself is an inversion in the ground truth — a known
    distinct failure mode per docs/known_issues.md bass-anchor diagnostic)
  - "other": neither of the above
Both tags can co-occur (e.g. inverted chord whose error is also P4/P5); we
report primary tag by priority p4p5 > inversion > other but store both flags.
"""
import json, re
from pathlib import Path
import numpy as np

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
IN = REPO / "docs/error_report_wrong_root_2026_07_16/fixed_test_predictions.json"
OUT = REPO / "docs/error_report_wrong_root_2026_07_16/examples_manifest.json"

PC_NAMES = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]

d = json.loads(IN.read_text())
records = d["records"]
qualities = d["qualities"]

wrong = [r for r in records if r["pred_root"] != r["gt_root"]]
print(f"{len(wrong)} / {len(records)} wrong-root test examples")

def interval_pc(a, b):
    return (b - a) % 12

for r in wrong:
    iv = interval_pc(r["gt_root"], r["pred_root"])
    r["interval_semitones"] = iv
    r["is_p4p5"] = iv in (5, 7)
    m = re.search(r"/(\w+)$", r["label"])
    r["gt_has_bass_tag"] = bool(m)
    r["gt_bass_degree"] = m.group(1) if m else None
    # inversion-related: GT label itself specifies a non-root bass (this
    # chord is an inversion in the ground truth, independent of the model's
    # bass-argmax reading)
    r["is_inversion_gt"] = bool(m) and m.group(1) not in ("1",)
    if r["is_p4p5"]:
        r["category"] = "p4p5"
    elif r["is_inversion_gt"]:
        r["category"] = "inversion"
    else:
        r["category"] = "other"
    r["gt_root_name"] = PC_NAMES[r["gt_root"]]
    r["pred_root_name"] = PC_NAMES[r["pred_root"]]
    r["bass_argmax_name"] = PC_NAMES[r["bass_argmax_pc"]]
    r["bass_agrees_gt"] = r["bass_argmax_pc"] == r["gt_root"]
    r["bass_agrees_pred"] = r["bass_argmax_pc"] == r["pred_root"]

cats = {}
for r in wrong:
    cats.setdefault(r["category"], []).append(r)
for k, v in cats.items():
    print(f"  {k}: {len(v)}")

# Target ~18, proportionally stratified but guarantee coverage of all 3
# categories and a mix of bass-agreement patterns for diagnostic contrast.
TARGET = 18
rng = np.random.RandomState(0)

def pick(lst, n):
    lst = sorted(lst, key=lambda r: (r["song_id"], r["t0"]))
    if len(lst) <= n:
        return lst
    idx = rng.choice(len(lst), size=n, replace=False)
    return [lst[i] for i in sorted(idx)]

n_p4p5 = min(len(cats.get("p4p5", [])), 8)
n_inv = min(len(cats.get("inversion", [])), 5)
n_other = TARGET - n_p4p5 - n_inv
n_other = min(n_other, len(cats.get("other", [])))

selected = (pick(cats.get("p4p5", []), n_p4p5)
            + pick(cats.get("inversion", []), n_inv)
            + pick(cats.get("other", []), n_other))

# top up to TARGET if short (borrow from largest remaining pool)
if len(selected) < TARGET:
    remaining = [r for r in wrong if r not in selected]
    need = TARGET - len(selected)
    selected += pick(remaining, min(need, len(remaining)))

# dedupe (song_id,t0) just in case
seen = set(); dedup = []
for r in selected:
    k = (r["song_id"], r["t0"])
    if k in seen: continue
    seen.add(k); dedup.append(r)
selected = dedup

print(f"\nSelected {len(selected)} examples:")
for r in selected:
    print(f"  {r['song_id']} t=[{r['t0']:.2f},{r['t1']:.2f}) GT={r['label']} "
          f"pred={r['pred_root_name']}:{r['pred_quality']} cat={r['category']} "
          f"bass_argmax={r['bass_argmax_name']} (agrees_gt={r['bass_agrees_gt']})")

OUT.write_text(json.dumps({"qualities": qualities, "examples": selected}, indent=1))
print(f"\nwrote {OUT}")
