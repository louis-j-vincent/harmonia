#!/usr/bin/env python3
"""ug_rebake.py — re-decode the comparison songs with the SHIPPED config and
re-score them against their UG tabs, so "où on se trompe" is measured on the
pipeline that actually ships rather than on stale baked charts.

    .venv/bin/python scratchpad/ug_rebake.py            # all 6 + quarantine
    .venv/bin/python scratchpad/ug_rebake.py --slugs X Y

WHY
---
Every number in ``docs/ug_score_report.md`` up to now was measured on
``docs/plots/inferred_*.html``, which ``docs/postmusx_segment_loss.md`` showed
to be STALE: Let It Be's chart predates the pipeline it was compared against and
was produced by a decoder in which music-x-lab never ran.  And commit c602a93
changed the splitter.  So the whole table has to be re-measured.

Payloads are written to ``scratchpad/rebake/`` — **never** to
``docs/plots/*.html``, which the app serves; a production re-bake is Louis's
call, not this script's.

WHAT IS AND IS NOT RE-RUN
-------------------------
* Our chord sequence: **re-decoded**, SHIPPED_CONFIG, once per song.
* The UG alignment: **re-run** from scratch by ``ug_score.run``.  It fits the
  TAB to the AUDIO chroma and never reads our chart, so it is deterministic and
  unaffected by the re-decode — but it is genuinely re-run rather than replayed
  from stored JSON, which also recomputes ``t_intro`` and the UG audio support
  that ``find_anchors`` consults (both of which the earlier replay had to hold
  fixed).
* Cached and reused: the UG tab HTML and the Whisper transcripts.  Neither
  depends on our decoder.

THE TWO TRAPS, BOTH LIVE IN THIS WORKTREE
-----------------------------------------
Inherited verbatim from ``scratchpad/bass_recovery_fresh.py``, because both have
actually fired here:

1. ``harmonia/models/nnls24_heads.npz`` is gitignored, so a fresh worktree has
   no trained heads and ``chord_head`` degrades to a ONE-CHORD chart for the
   whole song.  It looks like a decode.  It parses.  It scores.  It is nothing.
2. ``harmonia/third_party/.../data/`` is absent, so ``musx_redecode`` fails to a
   *warning*, ``_used_redecode`` stays False, the **Occam gate silently flips
   ON**, and two runs of "the shipped config" differ (114 vs 120 chords).

Both are resolved by symlinking the main checkout's copies; ``_preflight``
refuses to run if either is missing, and ``decode`` asserts afterwards that no
degradation warning fired.  A silent fallback here would invalidate the whole
comparison (CLAUDE.md #1 / #6).  Each song is decoded ONCE, deterministically;
nothing is averaged over a nondeterministic gate.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRATCH = REPO / "scratchpad"
REBAKE = SCRATCH / "rebake"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRATCH))

# Louis 2026-07-31: Chain of Fools is OUT of the comparison base (its harmony
# cannot time itself).  Kept only as a quarantined chord-vs-no-chord stress case.
BASE = {
    "maroon_5_this_love": "this_love_786697",
    "carpenters_close_to_you": "close_1044073",
    "let_it_be_remastered_2009": "let_it_be_17427",
    "the_police_every_breath_you_take_official_music_video": "every_breath_1087239",
    "ben_e_king_stand_by_me_audio": "stand_by_me_1724608",
    "katy_perry_hot_n_cold_official_music_video": "hot_n_cold_733932",
}
QUARANTINE = {
    "aretha_franklin_chain_of_fools_official_lyric_video": "chain_1212253",
}


def _preflight() -> dict:
    """Refuse to run unless the shipped pipeline can really run. Loud, not warn."""
    heads = REPO / "harmonia" / "models" / "nnls24_heads.npz"
    if not heads.exists():
        raise SystemExit(
            f"MISSING {heads} (gitignored; symlink the main checkout's copy). "
            "Without it chord_head emits a one-chord chart that scores fine and "
            "means nothing — see docs/postmusx_segment_loss.md.")
    musx = (REPO / "harmonia" / "third_party"
            / "ISMIR2019-Large-Vocabulary-Chord-Recognition")
    if not (musx / "data" / "submission_chord_list.txt").exists():
        raise SystemExit(
            f"MISSING {musx}/data/submission_chord_list.txt. musx_redecode "
            "would fail to a WARNING, the Occam gate would silently flip ON, "
            "and the decode would be nondeterministic. Refusing.")
    return {"heads": str(heads.resolve()), "musx": str(musx.resolve())}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slugs", nargs="*", default=None)
    ap.add_argument("--quarantine", action="store_true",
                    help="also score Chain of Fools, kept out of aggregates")
    a = ap.parse_args()

    env = _preflight()
    REBAKE.mkdir(parents=True, exist_ok=True)

    BRF = _load("bass_recovery_fresh", SCRATCH / "bass_recovery_fresh.py")
    import ug_score as US

    print(f"[config] SHIPPED_CONFIG = {json.dumps(BRF.SHIPPED_CONFIG)}")
    print(f"[config] heads = {env['heads']}")
    print(f"[config] musx  = {env['musx']}")
    print("[config] one decode per song, no averaging; any degradation "
          "warning aborts the song.\n")

    todo = dict(BASE)
    if a.quarantine:
        todo.update(QUARANTINE)
    if a.slugs:
        todo = {s: (BASE | QUARANTINE)[s] for s in a.slugs}

    manifest = {"shipped_config": BRF.SHIPPED_CONFIG, "env": env, "songs": {}}
    for slug, tab in todo.items():
        print(f"=== {slug}")
        chords = BRF.decode(slug)              # guarded; raises on degradation
        ours = BRF.to_evs(chords)
        (REBAKE / f"{slug}.json").write_text(json.dumps(
            {"slug": slug, "shipped_config": BRF.SHIPPED_CONFIG,
             "n_raw": len(chords), "n_collapsed": len(ours),
             "chords": chords}, indent=1))

        # score with the fresh sequence swapped in for the baked payload
        US.our_sequence_override = ours
        try:
            out = US.run(slug, str(SCRATCH / "ug_cache" / f"{tab}.html"),
                         use_asr=True, tag="fresh")
        finally:
            US.our_sequence_override = None
        US.print_song(out)
        manifest["songs"][slug] = {
            "n_raw": len(chords), "n_collapsed": len(ours),
            "counts": out["counts"], "agreement_pct": out["agreement_pct"],
            "quarantined": slug in QUARANTINE}
    (REBAKE / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nwrote {REBAKE/'manifest.json'}")


if __name__ == "__main__":
    main()
