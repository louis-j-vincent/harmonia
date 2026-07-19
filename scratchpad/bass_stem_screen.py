"""Part 2 cheap screen: does a Demucs-separated BASS stem sharpen the
per-block argmax->root accuracy vs the current mixed-audio bass block?

Baseline (finding #4, corpus-wide, root-position chords): mixed-audio bass
block argmax->root = 0.458. Here we compute BOTH mixed and separated on the
SAME 1-2 songs (controls for song identity, per mandate) plus a chroma
sharpness proxy (peak/mean of the folded 12-vector, comparable to the
'peak/mean 2.77' muddiness finding).

Disk discipline: 44.1k stems + wavs deleted immediately after BP extraction.
"""
from __future__ import annotations
import sys, subprocess, shutil, json, tempfile
from pathlib import Path
import numpy as np
import torch, torchaudio
import soundfile as sf

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO))
from harmonia.models.stage1_pitch import PitchExtractor

PC = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
TMP = Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-Code-harmonia/a6a4757d-a729-450a-a5be-b9970a43c412/scratchpad/bass_tmp"); TMP.mkdir(parents=True, exist_ok=True)
SEPBP = REPO/"data/cache/billboard_60/bp_sep_cache"; SEPBP.mkdir(parents=True, exist_ok=True)
MIXBP = REPO/"data/cache/billboard_60/bp_cache"

SONGS = {  # tid -> (video_id, title)
    "362":  ("JRiAMe1zsQ0", "Wednesday - Last Kiss (hard, 0%inv)"),
    "1111": ("3joI5VtuNV0", "Chris Kenner - Land of 1000 Dances (clean, 0%inv)"),
}

d = np.load(REPO/"data/cache/billboard_bp48_60_fixed_beatgrid.npz", allow_pickle=True)
C = {k: d[k] for k in d.keys()}

def ytdlp(vid, out):
    ytd = shutil.which("yt-dlp") or str(Path(sys.executable).parent/"yt-dlp")
    subprocess.run([ytd,"-x","--audio-quality","0","-o",str(out/f"{vid}.%(ext)s"),
                    f"https://www.youtube.com/watch?v={vid}"], capture_output=True, text=True, check=True)
    src = next(p for p in out.glob(f"{vid}.*") if p.suffix!=".wav")
    return src

def to_wav(src, dst, sr, ch):
    subprocess.run(["ffmpeg","-y","-i",str(src),"-ar",str(sr),"-ac",str(ch),str(dst)],
                   capture_output=True, text=True, check=True)

# ---- Demucs (torchaudio HDemucs) ----
_bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB
_model = None
def get_model():
    global _model
    if _model is None:
        _model = _bundle.get_model().eval()
    return _model

def separate_bass(wav_path):
    """Return mono bass stem at 44100 as np.float32."""
    model = get_model()
    arr, sr = sf.read(str(wav_path))  # (T, ch)
    wav = torch.tensor(np.atleast_2d(arr.T), dtype=torch.float32)  # (ch, T)
    if sr != _bundle.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, _bundle.sample_rate)
    ref = wav.mean(0); wav = (wav - ref.mean())/ (ref.std()+1e-8)
    sources = model.sources  # ['drums','bass','other','vocals']
    bi = sources.index("bass")
    seg = int(_bundle.sample_rate*10.0); overlap = int(_bundle.sample_rate*0.5)
    T = wav.shape[-1]; out = torch.zeros(len(sources), T)
    with torch.no_grad():
        pos = 0
        while pos < T:
            end = min(pos+seg, T)
            chunk = wav[:, pos:end].unsqueeze(0)
            o = model(chunk)[0]  # (S, ch, t)
            o = o*ref.std()+ref.mean()
            out[:, pos:end] += o.mean(1)[:, :end-pos]  # mono, no fancy blend (screen)
            pos = end
    bass = out[bi].numpy().astype(np.float32)
    # drums-removed mix (lit-standard chroma cleanup): sum all non-drum stems
    di = sources.index("drums")
    nodrum = out[[i for i in range(len(sources)) if i!=di]].sum(0).numpy().astype(np.float32)
    return bass, nodrum, _bundle.sample_rate

def seg_argmax_acc(acts, t0, t1, gt, lo, hi):
    """Per chord: fold onset frames in [t0,t1) to 12pc in MIDI register [lo,hi),
    argmax->root; also peak/mean sharpness. Returns (acc, mean_sharp, preds)."""
    ft = acts.frame_times; on = acts.onset_probs
    preds=[]; sharps=[]; hit=0; n=0
    for a,b,g in zip(t0,t1,gt):
        msk = (ft>=a)&(ft<b)
        if not msk.any(): continue
        v = on[msk].sum(0)  # (88,)
        c = np.zeros(12)
        for k in range(88):
            m=21+k
            if lo<=m<hi: c[m%12]+=v[k]
        s=c.sum()
        if s<1e-9: continue
        p=int(c.argmax()); preds.append(p); n+=1; hit+= (p==g)
        sharps.append(c.max()/(c.mean()+1e-12))
    return (hit/n if n else 0.0), (float(np.mean(sharps)) if sharps else 0.0), n

results=[]
for tid,(vid,title) in SONGS.items():
    sid=f"bb_{tid}"; print(f"\n=== {sid} {title} ===", flush=True)
    m = C["song_id"]==sid
    inv = np.array(["/" in str(l) for l in C["labels"][m]])
    t0=C["t0"][m][~inv]; t1=C["t1"][m][~inv]; gt=C["root"][m][~inv].astype(int)  # root-position only
    # 1) MIXED baseline: 22050 mono, hits existing BP cache
    src = ytdlp(vid, TMP)
    mixwav = TMP/f"{vid}_mix.wav"; to_wav(src, mixwav, 22050, 1)
    acts_mix = PitchExtractor(cache_dir=MIXBP).extract(mixwav)
    mix_acc, mix_sharp, n = seg_argmax_acc(acts_mix, t0, t1, gt, 0, 52)
    # 2) SEPARATED bass: 44100 stereo -> demucs bass -> mono wav -> BP
    stereo = TMP/f"{vid}_44.wav"; to_wav(src, stereo, 44100, 2)
    bass, nodrum, bsr = separate_bass(stereo)
    basswav = TMP/f"{vid}_bass.wav"; sf.write(basswav, bass, bsr)
    ndwav = TMP/f"{vid}_nodrum.wav"; sf.write(ndwav, nodrum, bsr)
    acts_sep = PitchExtractor(cache_dir=SEPBP).extract(basswav)
    acts_nd  = PitchExtractor(cache_dir=SEPBP).extract(ndwav)
    # isolated bass stem: register 0-52 and full 0-200
    sep_acc_lo, sep_sharp_lo, _ = seg_argmax_acc(acts_sep, t0, t1, gt, 0, 52)
    sep_acc_full, sep_sharp_full, _ = seg_argmax_acc(acts_sep, t0, t1, gt, 0, 200)
    # drums-removed mix, bass register 0-52 (lit-standard chroma cleanup)
    nd_acc_lo, nd_sharp_lo, _ = seg_argmax_acc(acts_nd, t0, t1, gt, 0, 52)
    # cleanup immediately
    for f in [src, mixwav, stereo, basswav, ndwav]:
        Path(f).unlink(missing_ok=True)
    r=dict(sid=sid, title=title, n_rootpos=int(n),
           mix_bass_acc=round(mix_acc,3), mix_sharp=round(mix_sharp,2),
           sep_bass_acc_lo=round(sep_acc_lo,3), sep_sharp_lo=round(sep_sharp_lo,2),
           sep_bass_acc_full=round(sep_acc_full,3), sep_sharp_full=round(sep_sharp_full,2),
           nodrum_bass_acc_lo=round(nd_acc_lo,3), nodrum_sharp_lo=round(nd_sharp_lo,2))
    print("  ", json.dumps(r), flush=True)
    results.append(r)

(Path(__file__).parent/"bass_stem_screen_results.json").write_text(json.dumps(results,indent=2))
shutil.rmtree(TMP, ignore_errors=True)
print("\nDONE"); print(json.dumps(results, indent=2))
