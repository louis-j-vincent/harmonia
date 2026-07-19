"""PREMISE CHECK (2026-07-16): does the BASS-12 chroma argmax anchor concentrate
on I/IV/V (0/5/7 semitones) relative to the SONG KEY?

User's refined claim (refinement #1): anchoring on argmax of the bass-12 chroma
block (feat48_abs[:,24:36]) yields an easier classification surface because that
argmax usually lands near tonic/subdominant/dominant of the actual key -> few
effective anchor classes following the circle-of-fifths bias of real harmony.

This is the EXACT feature the previously-rejected argmax-renorm used
(scratchpad/oracle_bass_family.py line 50: bass_argmax = fabs[:,24:36].argmax(1)).

Per-song KEY tonic is estimated from the GROUND-TRUTH chord labels (higher-trust
than audio): duration-weighted chord-tone chroma -> Krumhansl infer_key.
Cheap, falsifiable, run BEFORE training anything.
"""
from __future__ import annotations
import sys
from pathlib import Path
from collections import Counter
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from harmonia.data.corpus_schema import sounding_bass_pc
from harmonia.theory.key_profiles import infer_key

d = np.load(REPO / "data/cache/rwc/rwc_bp48_fixed.npz", allow_pickle=True)
labels = d["labels"]
root = d["root"].astype(int) % 12
song = d["song_id"]
qual = d["quality"]                      # <U4 string quality
fabs = d["feat48_abs"].astype(np.float32)
t0 = d["t0"].astype(float); t1 = d["t1"].astype(float)
dur = np.clip(t1 - t0, 1e-3, None)
N = len(labels)

# --- the anchor under test: argmax of the BASS-12 chroma block (cols 24:36) ---
bass_block = fabs[:, 24:36]
anchor_pc = bass_block.argmax(1)

# --- true sounding bass (resolver) and inversion mask ---
bass_true = np.array([sounding_bass_pc(l, int(root[i])) for i, l in enumerate(labels)]).astype(int)
inv = np.array(["/" in l for l in labels])

# --- chord-tone intervals per 7-way quality (for symbolic key estimation) ---
QTONES = {"maj": (0,4,7), "min": (0,3,7), "dom": (0,4,7,10), "hdim": (0,3,6,10),
          "dim": (0,3,6), "aug": (0,4,8), "sus": (0,5,7)}

# --- per-song key tonic from GT labels (duration-weighted chord-tone chroma) ---
song_key = {}
for s in np.unique(song):
    m = song == s
    chroma = np.zeros(12)
    for r, q, w in zip(root[m], qual[m], dur[m]):
        for iv_ in QTONES.get(str(q), (0,4,7)):
            chroma[(int(r) + iv_) % 12] += w
    song_key[s] = infer_key(chroma).tonic
key_tonic = np.array([song_key[s] for s in song])

# --- offsets relative to KEY ---
off_anchor = (anchor_pc - key_tonic) % 12       # <-- the premise variable
off_bass   = (bass_true - key_tonic) % 12       # true sounding bass rel key
off_root   = (root - key_tonic) % 12            # functional root rel key

def hist12(x, mask=None):
    v = x if mask is None else x[mask]
    c = Counter(v.tolist())
    tot = len(v)
    return [(k, c.get(k, 0), c.get(k, 0)/tot) for k in range(12)], tot

DEG = ["I","bII","II","bIII","III","IV","bV","V","bVI","VI","bVII","VII"]

print(f"N={N} chords, {len(song_key)} songs, inversions={inv.sum()} ({100*inv.mean():.1f}%)")
print(f"anchor(bass-argmax) == sounding-bass: {(anchor_pc==bass_true).mean():.4f}")
print(f"anchor(bass-argmax) == functional-root: {(anchor_pc==root).mean():.4f}")
print(f"per-song key modes: {Counter('maj' if True else '' for _ in [0])}")  # placeholder

def show(name, x, mask=None):
    rows, tot = hist12(x, mask)
    print(f"\n--- {name}  (n={tot}) ---")
    print("deg:   " + " ".join(f"{DEG[k]:>4s}" for k in range(12)))
    print("pct:   " + " ".join(f"{p*100:4.1f}" for _,_,p in rows))
    iivv = rows[0][2] + rows[5][2] + rows[7][2]
    diat = sum(rows[k][2] for k in (0,2,4,5,7,9,11))
    print(f"  I+IV+V (0,5,7) = {iivv*100:.1f}%   diatonic(0,2,4,5,7,9,11) = {diat*100:.1f}%")

print("\n============ PREMISE: bass-argmax ANCHOR offset relative to SONG KEY ============")
show("ANCHOR (bass-12 argmax) - KEY  [ALL]", off_anchor)
show("ANCHOR (bass-12 argmax) - KEY  [ROOT-POSITION]", off_anchor, ~inv)
show("ANCHOR (bass-12 argmax) - KEY  [INVERSIONS]", off_anchor, inv)

print("\n============ CONTEXT: true sounding-bass & functional-root rel KEY ============")
show("TRUE sounding-bass - KEY  [ALL]", off_bass)
show("FUNCTIONAL root - KEY  [ALL]", off_root)

# effective anchor-class count: entropy / #classes covering 90% mass
def eff(x):
    c = np.array([ (x==k).mean() for k in range(12) ])
    ent = -np.sum(c[c>0]*np.log2(c[c>0]))
    order = np.sort(c)[::-1]; cum = np.cumsum(order)
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    return ent, n90
for nm, x in [("anchor-rel-key", off_anchor), ("anchor-ABSOLUTE(no key)", anchor_pc)]:
    ent, n90 = eff(x)
    print(f"\n[{nm}] entropy={ent:.2f} bits (max 3.58), #classes for 90% mass = {n90}")
