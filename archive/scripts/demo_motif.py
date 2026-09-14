"""Show a tune reduced to its recurring motifs (exact + transpose-invariant)."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

from harmonia.models.motif import Chord, PC_NAMES, find_motifs, reduce_song  # noqa
from analyze_accomp_emission import parse_chord  # noqa

DB = REPO / "data" / "accomp_db" / "db.jsonl"


def load_chords(title: str) -> list[Chord]:
    rec = [r for r in map(json.loads, open(DB)) if r["title"] == title][0]
    chords = []
    for e in rec["chord_timeline"]:
        p = parse_chord(e["mma"])
        if p is None:
            continue
        chords.append(Chord(root=p[0], qual=p[1], label=e["ireal"], bar=e["bar"]))
    return rec, chords


def show(title: str):
    rec, chords = load_chords(title)
    print(f"\n{'='*66}\n{title}  ({rec['form']})  —  {len(chords)} chords\n{'='*66}")

    for shape in (False, True):
        head = "SHAPE motifs (transpose-invariant — same pattern, any key)" if shape \
               else "EXACT motifs (literal repeated chords)"
        print(f"\n{head}")
        motifs = find_motifs(chords, shape=shape, min_len=2, max_len=8, min_count=2)
        for m in motifs[:8]:
            where = f"  in keys: {', '.join(m.keys)}" if shape else ""
            print(f"  {m.display:14s} x{m.count:2d}  (saves {m.saving:2d} slots){where}")

        timeline, used = reduce_song(chords, shape=shape, min_len=2, max_len=8)
        toks = []
        for kind, obj, _ in timeline:
            toks.append(f"⟨{obj.display}⟩" if kind == "motif" else obj.label)
        n_units = len(timeline)
        print(f"  reduced: {len(chords)} chords → {n_units} units "
              f"({len(used)} unique motifs)")
        print("   ", "  ".join(toks))


if __name__ == "__main__":
    titles = sys.argv[1:] or ["Anthropology"]
    for t in titles:
        show(t)
