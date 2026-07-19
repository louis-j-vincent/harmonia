"""
Compute _estimate_gt_offset() auto-guesses for a representative sample of the
Billboard GT-offset triage corpus, WITHOUT saving to billboard_gt_offsets.json
(per task instructions — track 341/Commodores is under separate review, and
blind-saving 59 auto-guesses risks propagating bad guesses per the earlier
"different edit" finding).

Downloads audio one song at a time to a scratch dir, runs the exact same
heuristic as scripts/harmonia_server.py's _estimate_gt_offset(), computes a
cheap confidence signal, deletes the audio, moves to next song.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import harmonia_server as hs  # noqa: E402
import mirdata  # noqa: E402
import numpy as np  # noqa: E402
import librosa  # noqa: E402

SCRATCH = Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-Code-harmonia/"
                "a6a4757d-a729-450a-a5be-b9970a43c412/scratchpad/gt_offset_dl")
SCRATCH.mkdir(parents=True, exist_ok=True)

CACHED_AUDIO = REPO / "data" / "cache" / "billboard_60" / "audio"

# Sample: representative mix across severity buckets, EXCLUDING track 341
# (Commodores, owned by parallel verification agent).
SAMPLE_TRACK_IDS = [
    # wrong-edit bucket (>2s duration mismatch), spanning the mismatch range
    "1027",  # 12.71s mismatch
    "647",   # 11.43s
    "329",   # 8.2s
    "145",   # 7.37s
    "521",   # 4.84s
    "306",   # 4.25s
    "334",   # 2.66s
    "168",   # 2.35s
    # check bucket (<2s), spanning the range
    "354",   # 1.65s
    "217",   # 1.5s
    "183",   # 1.32s -- audio already cached locally (p9Y3N_2xUsw.wav)
    "159",   # 0.97s
    "246",   # 0.66s
    "640",   # 0.02s
    "153",   # 0.05s
]


def load_corpus() -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for p in hs._BILLBOARD_CORPUS_FILES:
        merged.update(json.loads(p.read_text(encoding="utf-8")))
    return merged


def mismatch_and_severity(v: dict) -> tuple[float | None, str]:
    best = v.get("best") or []
    audio_dur = best[2] if len(best) > 2 else None
    gt_dur = v.get("gt_dur")
    if audio_dur is None or gt_dur is None:
        return None, "unknown"
    m = abs(audio_dur - gt_dur)
    return m, ("wrong-edit" if m > 2.0 else "check")


def download_audio(video_id: str, dest_dir: Path) -> Path | None:
    import yt_dlp
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    url = f"https://www.youtube.com/watch?v={video_id}"
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)
    for f in dest_dir.iterdir():
        if f.suffix.lower() in {".wav", ".m4a", ".opus", ".mp3"}:
            return f
    return None


def onset_confidence(audio_path: Path) -> dict:
    """Extra diagnostics beyond the raw _estimate_gt_offset() return, so we
    can judge how trustworthy the anchored onset was."""
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    hop = 512
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    onsets = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, hop_length=hop,
                                         units="time", backtrack=True)
    if len(onsets) == 0:
        return {"n_onsets_30s": 0, "first_onset_prominence": 0.0}
    ot = librosa.times_like(oenv, sr=sr, hop_length=hop)
    strengths = np.interp(onsets, ot, oenv)
    head = oenv[: int(30 * sr / hop)]
    thr = 0.4 * float(np.max(head)) if len(head) else 0.0
    strong = onsets[strengths > thr]
    n_onsets_30s = int((onsets < 30).sum())
    # prominence: how far above threshold the chosen onset's strength is,
    # relative to the head's max — higher = more unambiguous "this is THE
    # downbeat", lower = could easily be a false pick among many similar
    # peaks (e.g. a drum fill / count-in).
    if len(strong):
        first_strength = float(np.interp(strong[0], ot, oenv))
    else:
        first_strength = float(np.interp(onsets[0], ot, oenv))
    prom = (first_strength - thr) / (float(np.max(head)) - thr + 1e-9) if len(head) else 0.0
    return {"n_onsets_30s": n_onsets_30s, "first_onset_prominence": round(prom, 3)}


def main():
    corpus = load_corpus()
    results = []
    for i, track_id in enumerate(SAMPLE_TRACK_IDS):
        v = corpus.get(track_id)
        if v is None:
            print(f"[{track_id}] not in corpus, skip")
            continue
        best = v["best"]
        video_id = best[0]
        mismatch, severity = mismatch_and_severity(v)
        artist, title = v.get("artist", ""), v.get("title", "")
        print(f"\n=== [{i+1}/{len(SAMPLE_TRACK_IDS)}] track {track_id} ({severity}, "
              f"mismatch {mismatch:.2f}s): {artist} - {title} ({video_id}) ===")

        t0 = time.time()
        # reuse cached audio if we have it, else download fresh
        cached = CACHED_AUDIO / f"{video_id}.wav"
        song_dir = SCRATCH / track_id
        song_dir.mkdir(exist_ok=True)
        if cached.exists():
            audio_path = cached
            print(f"  using cached audio {cached}")
            downloaded_fresh = False
        else:
            try:
                audio_path = download_audio(video_id, song_dir)
            except Exception as e:
                print(f"  DOWNLOAD FAILED: {e}")
                results.append(dict(track_id=track_id, artist=artist, title=title,
                                     video_id=video_id, severity=severity, mismatch=mismatch,
                                     error=f"download failed: {e}"))
                shutil.rmtree(song_dir, ignore_errors=True)
                continue
            downloaded_fresh = True
            if audio_path is None:
                print("  DOWNLOAD FAILED: no audio file produced")
                results.append(dict(track_id=track_id, artist=artist, title=title,
                                     video_id=video_id, severity=severity, mismatch=mismatch,
                                     error="download failed: no file"))
                shutil.rmtree(song_dir, ignore_errors=True)
                continue
        dl_t = time.time() - t0

        try:
            _, gt_raw = hs._gt_chords_for_video_raw(video_id)
            if not gt_raw:
                raise RuntimeError("no GT chords loaded")
            t1 = time.time()
            offset = hs._estimate_gt_offset(audio_path, gt_raw)
            conf = onset_confidence(audio_path)
            est_t = time.time() - t1
            print(f"  auto-guess offset: {offset:+.3f}s  (dl {dl_t:.1f}s, est {est_t:.1f}s)  conf={conf}")
            results.append(dict(track_id=track_id, artist=artist, title=title, video_id=video_id,
                                 severity=severity, mismatch=round(mismatch, 2) if mismatch else mismatch,
                                 offset_guess=offset, **conf))
        except Exception as e:
            print(f"  ESTIMATE FAILED: {e}")
            results.append(dict(track_id=track_id, artist=artist, title=title, video_id=video_id,
                                 severity=severity, mismatch=mismatch, error=f"estimate failed: {e}"))
        finally:
            if downloaded_fresh:
                shutil.rmtree(song_dir, ignore_errors=True)

    out_path = SCRATCH.parent / "gt_offset_guess_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n\nWrote {out_path}")
    return results


if __name__ == "__main__":
    main()
