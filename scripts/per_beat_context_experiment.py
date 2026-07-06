"""Attack per-beat evidence (the recurring bottleneck). Per-BEAT root is ~67% but
per-SEGMENT ~91% — the segment just has more context. Question: does giving a per-beat
prediction its NEIGHBOURS' features (±context, the soft/learned version of pooling, the
±4-chord LSTM idea) recover that gap WITHOUT needing segment boundaries?

For window w, the feature is the beat's 48d chroma concatenated with beats [b-w..b+w].
5-fold by song. If per-beat root climbs 67% → ~85% with context, a beat-sequence model
(BiLSTM over per-beat features, soft outputs) is the lever — and it fixes segmentation
(cleaner per-cell labels) and labeling at once.

Usage: .venv/bin/python scripts/per_beat_context_experiment.py --n-songs 15 [--degrade]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from analyze_accomp_emission import song_chord_spans  # noqa: E402
from build_accomp_audio_hard import time_varying_degrade  # noqa: E402
from build_audio_chord_features import BUCKET_FAMILY  # noqa: E402
from analyze_accomp_emission import parse_chord  # noqa: E402
from root_model_experiment import chroma88  # noqa: E402
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig  # noqa: E402
from harmonia.models.stage1_pitch import PitchExtractor  # noqa: E402
from harmonic_rhythm_probe import pool_beats  # noqa: E402

DB = REPO / "data" / "accomp_db" / "db.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-songs", type=int, default=15)
    ap.add_argument("--degrade", action="store_true")
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(DB)]
    songs = [r for r in recs if r["corpus"] == "jazz1460" and r["beats_per_bar"] == 4
             and (REPO / r["midi_path"]).exists()]
    songs = songs[:: max(len(songs) // args.n_songs, 1)][: args.n_songs]
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    ex = PitchExtractor(cache_dir=None); rng = np.random.default_rng(4)

    feats, roots, grp = [], [], []          # per-beat 48d feature, GT root, song
    for rec in songs:
        spb = 60.0 / rec["tempo"]; bpb = rec["beats_per_bar"]; nb = rec["n_bars"]
        n_beats = nb * bpb
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            tmp = Path(wf.name)
        try:
            renderer.render(REPO / rec["midi_path"], tmp, RenderConfig(soundfont_path=sf2))
            y, sr = sf.read(tmp); y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
            if args.degrade:
                y = time_varying_degrade(y, sr, rng); sf.write(tmp, y, sr)
            acts = ex.extract(tmp, use_cache=False)
        finally:
            tmp.unlink(missing_ok=True)
        bt = np.arange(n_beats + 1) * spb
        onb = pool_beats(acts.frame_times, acts.onset_probs, bt)
        ntb = pool_beats(acts.frame_times, acts.note_probs, bt)

        def gtroot(t):
            for t0, t1, root, _q in song_chord_spans(rec):
                if t0 <= t < t1:
                    return root % 12
            return None
        for b in range(n_beats):
            g = gtroot((b + 0.5) * spb)
            if g is None:
                continue
            on, nt = onb[b], ntb[b]
            feats.append(np.concatenate([chroma88(on), chroma88(nt),
                                         chroma88(on, 0, 52), chroma88(on, 60, 200)]))
            roots.append(g); grp.append(rec["song_id"] + f"_{b}")   # song id + beat for context
    feats = np.array(feats); roots = np.array(roots)
    song_of = np.array([g.rsplit("_", 1)[0] for g in grp])
    beat_of = np.array([int(g.rsplit("_", 1)[1]) for g in grp])

    def windowed(w):
        """Concatenate each beat's feature with ±w neighbours (same song, zero-pad edges)."""
        X = []
        by_song = {}
        for i, (s, b) in enumerate(zip(song_of, beat_of)):
            by_song.setdefault(s, {})[b] = feats[i]
        for s, b in zip(song_of, beat_of):
            row = []
            for d in range(-w, w + 1):
                row.append(by_song[s].get(b + d, np.zeros(48)))
            X.append(np.concatenate(row))
        return np.array(X)

    print(f"\n=== per-beat root vs temporal context, {len(songs)} "
          f"{'DEGRADED' if args.degrade else 'clean'} songs, {len(roots)} beats ===\n")
    print(f"  {'window':>8} {'feat dim':>9} {'per-beat root':>14}")
    for w in (0, 1, 2, 3):
        X = windowed(w)
        pred = np.zeros(len(roots), int)
        for tr, te in GroupKFold(5).split(X, roots, song_of):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), roots[tr])
            pred[te] = clf.predict(sc.transform(X[te]))
        print(f"  {f'±{w}':>8} {X.shape[1]:>9} {(pred == roots).mean():>14.1%}")
    print("\n  climbing with window => temporal context (a beat-sequence model) is the lever.")


if __name__ == "__main__":
    main()
