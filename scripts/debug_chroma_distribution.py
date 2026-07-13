"""debug_chroma_distribution.py — show absolute pitch distribution per song,
diagnose key bias in beat_seq model training vs POP909."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from harmonia.models.chord_pipeline_v1 import _reg_raw
from harmonia.models.stage1_pitch import PitchExtractor
from harmonia.data.pop909_parser import POP909Parser

NOTE = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
DATA_ROOT = REPO / "data"


def song_chroma(acts):
    chroma = np.zeros(12)
    for f in range(len(acts.onset_probs)):
        chroma += _reg_raw(acts.onset_probs[f])
    chroma /= chroma.sum() + 1e-9
    return chroma


def chroma_bar(chroma):
    return "  ".join(f"{NOTE[i]}:{chroma[i]:.3f}" for i in range(12))


def main():
    ex = PitchExtractor(cache_dir=DATA_ROOT / "cache")
    parser = POP909Parser(DATA_ROOT / "pop909" / "POP909")

    # ── POP909 ────────────────────────────────────────────────────────────────
    print("=== POP909 mean onset chroma (absolute) ===")
    pop_chromas = []
    for sid in ["001", "002", "003", "004", "005"]:
        wav = DATA_ROOT / "renders" / "pop909" / sid / f"{sid}_v005_musescoregeneral.wav"
        if not wav.exists():
            continue
        acts = ex.extract(wav)
        ch = song_chroma(acts)
        pop_chromas.append(ch)
        peak = ch.argmax()
        print(f"  {sid}  peak={NOTE[peak]:>3} ({ch[peak]:.3f})  [{chroma_bar(ch)}]")

    # ── iReal training corpus key distribution ────────────────────────────────
    print("\n=== iReal corpus GT key distribution ===")
    DB = DATA_ROOT / "accomp_db" / "db.jsonl"
    recs = [json.loads(l) for l in open(DB)]
    jazz = [r for r in recs if r.get("corpus") == "jazz1460" and "key" in r]

    key_counts = {}
    for r in jazz:
        k = r["key"]
        key_counts[k] = key_counts.get(k, 0) + 1
    total = sum(key_counts.values())

    # map key names to root pitch class
    KEY_PC = {}
    for k in key_counts:
        root = k.replace("b", "").replace("#", "").replace("m", "")
        root = root.strip()
        if root in NOTE:
            pc = NOTE.index(root)
        else:
            continue
        if "b" in k.split("m")[0]:
            pc = (pc - 1) % 12
        elif "#" in k.split("m")[0]:
            pc = (pc + 1) % 12
        KEY_PC[k] = pc

    pc_counts = [0] * 12
    for k, cnt in key_counts.items():
        if k in KEY_PC:
            pc_counts[KEY_PC[k]] += cnt

    print(f"  {total} songs, {len(set(jazz[0].keys()))} fields")
    print("  Root distribution (major+minor combined):")
    for i in range(12):
        bar = "█" * int(pc_counts[i] / total * 100)
        print(f"    {NOTE[i]:>3}: {pc_counts[i]:4d} ({pc_counts[i]/total:.1%}) {bar}")

    # ── what beat_seq sees at train time: rendered onset chroma ─────────────
    # Sample 3 iReal songs and show their onset chroma
    print("\n=== Sample iReal songs: GT key vs onset chroma peak ===")
    from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig
    import tempfile
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    sampled = [r for r in jazz if (REPO / r["midi_path"]).exists()][:8]
    for rec in sampled:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            tmp = Path(wf.name)
        try:
            renderer.render(REPO / rec["midi_path"], tmp, RenderConfig(soundfont_path=sf2))
            acts = ex.extract(tmp, use_cache=False)
            ch = song_chroma(acts)
            peak = ch.argmax()
            gt_key = rec.get("key", "?")
            print(f"  {rec['song_id']:30s}  key={gt_key:>5}  peak={NOTE[peak]:>3} ({ch[peak]:.3f})")
        except Exception as e:
            print(f"  {rec['song_id']}: skip ({e})")
        finally:
            tmp.unlink(missing_ok=True)

    # ── key bias in beat_seq errors ───────────────────────────────────────────
    print("\n=== Key-agnostic check: would uniform transposition fix things? ===")
    print("  If beat_seq had uniform key training, per-root accuracy should be flat.")
    print("  Current per-root accuracies from debug_root_model.py showed:")
    print("    C=27%, C#=34% (bad) vs G=93%, F=81% (good)")
    print("  → This is classic corpus key-distribution bias, NOT a representation issue.")
    print("  The 48d features are absolute pitch class (not rolled), so the LR")
    print("  implicitly learns 'feature at position 7 (G) = root' more often than 'position 0 (C)'")
    print("  Fix: transpose-augment training data to uniform 12-key distribution.")


if __name__ == "__main__":
    main()
