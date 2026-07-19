"""PHASE 2 lever: monophonic pitch tracking of the low register (pYIN on a
low-pass-filtered mix) as a sounding-bass estimator, vs the BP48 bass-argmax
baseline, scored against RWC GT sounding-bass pc on the SAME chord spans.

This is a CONSERVATIVE LOWER BOUND on the Demucs-bass -> pYIN lever from the
literature review: source separation (not installed / disk-tight) would only
clean the bass isolation further. Low-pass + pYIN tests the core mechanism —
"read the lowest sounding pitch with a monophonic per-onset tracker" — which the
error analysis said should fix the three BP48 bass failure modes (short spans,
fourth/fifth slips, unreadable songs).

Audio: RWC-P WAVs fetched one at a time via remotezip from Zenodo (record
18656623), deleted after use. GT spans + BP48 baseline come from rwc_bp48.npz.
"""
from __future__ import annotations
import sys, os, shutil
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"scripts"))
from harmonia.data.corpus_schema import load_corpus
from remotezip import RemoteZip

ZIP_URL = "https://zenodo.org/api/records/18656623/files/RWC-P.zip/content"
AUDIO = REPO/"data/cache/rwc/audio"; AUDIO.mkdir(parents=True, exist_ok=True)
BS = {"b2":1,"2":2,"b3":3,"3":4,"4":5,"b5":6,"5":7,"b6":8,"6":9,"b7":10,"7":11,"b9":1,"9":2}
NOTE = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]


def derive_bass(label, root):
    label = str(label).strip()
    if "/" not in label or label in ("N","X",""):
        return 0, root % 12  # root-position: sounding bass == root
    b = label.split("/",1)[1].strip()
    if b not in BS:
        return 0, root % 12
    return 1, (root + BS[b]) % 12


def pyin_bass_pc(wav, sr=11025, fmin=41.2, fmax=392.0, hop=512):
    import librosa
    from scipy.signal import butter, sosfiltfilt
    y, _ = librosa.load(str(wav), sr=sr, mono=True)
    sos = butter(4, 400.0, btype="low", fs=sr, output="sos")
    yl = sosfiltfilt(sos, y).astype(np.float32)
    f0, voiced, _ = librosa.pyin(yl, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop,
                                 frame_length=2048, fill_na=np.nan)
    times = librosa.times_like(f0, sr=sr, hop_length=hop)
    pc = np.full(len(f0), -1, int)
    v = voiced & np.isfinite(f0) & (f0 > 0)
    midi = 69 + 12*np.log2(np.where(v, f0, 440.0)/440.0)
    pc[v] = (np.round(midi[v]).astype(int)) % 12
    return times, pc


def main():
    d = load_corpus(REPO/"data/cache/rwc/rwc_bp48.npz")
    sid = d["song_id"]; lab = d["labels"]; root = d["root"].astype(int)
    t0 = d["t0"]; t1 = d["t1"]; feat = d["feat48_abs"]
    dur = t1 - t0
    is_inv = np.zeros(len(lab), int); gbass = np.zeros(len(lab), int)
    for i in range(len(lab)):
        iv, b = derive_bass(lab[i], root[i]); is_inv[i]=iv; gbass[i]=b
    bp48_bass = feat[:, 24:36].argmax(1)  # BP48 bass-block argmax (baseline)

    songs = ["rwc_RWC_P040","rwc_RWC_P083","rwc_RWC_P099","rwc_RWC_P066","rwc_RWC_P020"]
    print(f"Phase2 pYIN-bass vs BP48 baseline on {len(songs)} RWC songs\n")

    rows = []  # (is_inv, dur, gbass, bp48_pred, pyin_pred)
    with RemoteZip(ZIP_URL) as z:
        names = {Path(i.filename).stem: i.filename for i in z.infolist() if i.filename.endswith(".wav")}
        for s in songs:
            free = shutil.disk_usage(str(AUDIO)).free/1e9
            if free < 2.0:
                print(f"!! disk {free:.2f}GB < 2GB floor -> STOP"); break
            rwcid = s.replace("rwc_","")
            zname = names.get(rwcid)
            if not zname:
                print(f"[{s}] no wav"); continue
            print(f"[{s}] fetch ({free:.1f}GB free)...", flush=True)
            z.extract(zname, path=str(AUDIO)); wav = AUDIO/zname
            try:
                times, ppc = pyin_bass_pc(wav)
            finally:
                try: wav.unlink()
                except Exception: pass
            m = sid == s
            idx = np.where(m)[0]
            n_ok_p = n_ok_b = n = nvoiced = 0
            for gi in idx:
                seg = (times >= t0[gi]) & (times < t1[gi]) & (ppc >= 0)
                if seg.sum() == 0:
                    pred = -1
                else:
                    pred = int(np.bincount(ppc[seg], minlength=12).argmax()); nvoiced += 1
                rows.append((is_inv[gi], dur[gi], gbass[gi], int(bp48_bass[gi]), pred))
                n += 1
                if pred == gbass[gi]: n_ok_p += 1
                if bp48_bass[gi] == gbass[gi]: n_ok_b += 1
            print(f"   {n} chords, voiced-cover {nvoiced}/{n}; "
                  f"pyin bass-acc {n_ok_p/n:.3f}  BP48 bass-acc {n_ok_b/n:.3f}", flush=True)

    R = np.array(rows)
    if len(R) == 0:
        print("no data"); return
    inv, du, gb, bpp, pyp = R[:,0], R[:,1], R[:,2], R[:,3].astype(int), R[:,4].astype(int)
    have = pyp >= 0  # pyin produced a voiced prediction

    def acc(mask, pred, gt):
        mask = mask & (pred >= 0)
        return (pred[mask]==gt[mask]).mean() if mask.sum() else float("nan"), int(mask.sum())

    print("\n" + "="*64)
    print(f"Pooled {len(R)} chords ({int(inv.sum())} inversions). "
          f"pYIN voiced-coverage {have.mean():.2f}")
    for name, mask in [("ALL chords", np.ones(len(R),bool)),
                       ("INVERSIONS only", inv==1),
                       ("root-position", inv==0)]:
        ap, na = acc(mask, pyp, gb); ab,_ = acc(mask, bpp, gb)
        # fair head-to-head: only where pyin has a prediction
        fair = mask & have
        apf,_ = acc(fair, pyp, gb); abf,_ = acc(fair, bpp, gb)
        print(f"\n{name}: n={mask.sum()}")
        print(f"   pYIN bass-acc (where voiced, n={na}): {ap:.3f}")
        print(f"   BP48 bass-acc (same subset)         : {abf:.3f}")
        print(f"   BP48 bass-acc (all)                 : {ab:.3f}")
    # duration split (the BP48 short-span weakness)
    print("\nBy duration (fair subset where pyin voiced):")
    med = np.median(du)
    for lbl, mm in [("short (<median)", du<med), ("long (>=median)", du>=med)]:
        fair = mm & have
        ap,_ = acc(fair, pyp, gb); ab,_ = acc(fair, bpp, gb)
        print(f"   {lbl:16s} n={fair.sum():4d}  pYIN {ap:.3f}  BP48 {ab:.3f}")
    # error interval for pyin (fifth/fourth slip check)
    e = have & (pyp != gb)
    ivl = (pyp[e]-gb[e])%12
    top = sorted([(100*(ivl==k).sum()/max(e.sum(),1),k) for k in range(12)],reverse=True)[:4]
    print("\npYIN error intervals: " + ", ".join(f"+{k}({NOTE[k]}) {p:.0f}%" for p,k in top))


if __name__ == "__main__":
    main()
