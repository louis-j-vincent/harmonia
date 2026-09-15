"""Premise check for issue #20/#23: is per-section *local* key a real problem on
the iReal corpus, or does it collapse to a global-key problem?

Runs the rules-based oracle section-key labeler over the whole corpus and reports
the fraction of section instances whose oracle local key differs from the song's
global key (the "true modulation" rate), split by sub-corpus (jazz vs pop/blues)
and swept over the modulation-margin gate to show sensitivity.

Usage:
    python scripts/check_local_key_premise.py
"""
from __future__ import annotations

from collections import Counter

from harmonia.models.local_key_data import (
    DEFAULT_DB, build_examples, idx_to_key,
)
from harmonia.theory.local_key import key_name


def collection_root(idx: int) -> int:
    """Major-key root of a key's diatonic collection (minor -> relative major).
    Two keys with the same collection_root share the exact same 7 diatonic
    pitch classes -> identical diatonic *quality* prior per root."""
    tonic, mode = idx_to_key(idx)
    return tonic if mode == "major" else (tonic + 3) % 12


def report(margin: float) -> None:
    ex = build_examples(DEFAULT_DB, margin=margin)

    print(f"\n=== margin = {margin} nats ===")
    print(f"{'corpus':<12} {'sections':>9} {'mod(24cls)':>11} {'rate':>7} "
          f"{'mod(coll)':>10} {'rate':>7}")
    groups = {
        "jazz1460": ["jazz1460"],
        "pop400": ["pop400"],
        "blues50": ["blues50"],
        "ALL": ["jazz1460", "pop400", "blues50"],
    }
    for name, corps in groups.items():
        sub = [e for e in ex if e["corpus"] in corps]
        if not sub:
            continue
        nmod = sum(e["modulated"] for e in sub)
        ncoll = sum(collection_root(e["y"]) != collection_root(e["y_global"])
                    for e in sub)
        print(f"{name:<12} {len(sub):>9} {nmod:>11} {nmod / len(sub):>6.1%} "
              f"{ncoll:>10} {ncoll / len(sub):>6.1%}")
    return ex


def main() -> None:
    print("Loading corpus & running rules-based oracle section-key labeler...")
    for margin in (3.0, 6.0, 10.0):
        ex = report(margin)

    # detail at the default margin
    ex = build_examples(DEFAULT_DB, margin=6.0)
    jazz = [e for e in ex if e["corpus"] == "jazz1460"]
    mod = [e for e in jazz if e["modulated"]]
    print("\n--- example modulated jazz sections (margin 6.0) ---")
    for e in mod[:12]:
        gt = key_name(*idx_to_key(e["y_global"]))
        lk = key_name(*idx_to_key(e["y"]))
        seq = " ".join(f"{r}:{q}" for r, q in e["seq"][:8])
        print(f"  {e['label']}: global {gt:>9} -> local {lk:>9} | {seq}")

    print("\n--- section length stats ---")
    lens = [len(e["seq"]) for e in ex]
    print(f"  sections={len(ex)}  median chords/section={sorted(lens)[len(lens)//2]}"
          f"  mean={sum(lens)/len(lens):.1f}")
    print(f"  songs with >=1 modulated section: "
          f"{len({e['song_idx'] for e in ex if e['modulated']})}"
          f" / {len({e['song_idx'] for e in ex})}")


if __name__ == "__main__":
    main()
