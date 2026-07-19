"""group_pool_graceful_verify.py — 2026-07-19 (★ CHORD-ROBUSTNESS / BAR-MERGE)

Re-runs the EXACT group-pooling scenario that failed tonight (the ORIGINAL,
non-workaround whole-8-bar-block-span encoding — one merge group per cluster
letter, spans = block_times_s) through the FIXED `pool_beat_evidence`, on all
3 real songs, via the production in-process path (`infer_chords_v1`, exactly
what `/api/reinfer` invokes when merges are present).

Before this fix the whole-block spec was MECHANICALLY BROKEN: any single
off-by-one span killed its ENTIRE group (aretha 0/2, abba 0/3, autumn_leaves
1/5 groups applied). This measures, per song: how many groups now pool FULLY
(all spans equal — what the old code would also have applied), how many pool
PARTIALLY (mode pooled, weak-link spans excluded — the NEW win), and how many
remain UNPOOLABLE (reported, not silently dropped), plus real before/after
chord-change counts.
"""
from __future__ import annotations
import sys, json, re, shutil, subprocess, tempfile, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from group_pool_section_clusters import SONGS, load_clusters, AUDIO_DIR

from harmonia.models.chord_pipeline_v1 import infer_chords_v1
from harmonia.models.user_constraints import SectionMerge


class _CaptureLogs(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs = []
    def emit(self, record):
        self.msgs.append(record.getMessage())


def transcode(audio_path, wav_path):
    subprocess.run(["ffmpeg", "-y", "-i", str(audio_path), "-ac", "1", "-ar", "22050",
                    str(wav_path)], check=True, capture_output=True, timeout=180)


def full_cluster_groups(letters, block_times_s):
    """One merge group per multi-block cluster letter; spans = whole-block
    [t0,t1] time ranges (the ORIGINAL failing spec)."""
    groups = []
    for letter, members in sorted(letters.items()):
        if len(members) < 2:
            continue
        spans = [tuple(block_times_s[b]) for b in sorted(members)]
        groups.append((letter, spans))
    return groups


def chord_at(chords, t):
    for c in chords:
        if c["start_s"] <= t < c["end_s"]:
            return c
    return None


def main():
    out = {"generated": "2026-07-19", "songs": {}}
    for slug in SONGS:
        letters, blocks, block_times_s, _am, _sm = load_clusters(slug)
        groups = full_cluster_groups(letters, block_times_s)
        merges = [SectionMerge(spans=list(spans)) for _l, spans in groups]

        cap = _CaptureLogs()
        plog = logging.getLogger("harmonia.models.chord_pipeline_v1")
        old_level = plog.level
        plog.setLevel(logging.INFO)
        plog.addHandler(cap)
        tmp = Path(tempfile.mkdtemp(prefix="harmonia_gracefulverify_"))
        try:
            wav = tmp / "a.wav"
            transcode(AUDIO_DIR / SONGS[slug]["audio_name"], wav)
            base = infer_chords_v1(wav, cache_dir=tmp, joint_transition_weight=0.0)
            cap.msgs.clear()  # only the merges (cons) call's logs matter
            cons = infer_chords_v1(
                wav, cache_dir=tmp, joint_transition_weight=0.0,
                user_constraints={"confirms": [],
                                  "merges": [{"spans": list(sp)} for _l, sp in groups]})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            plog.removeHandler(cap)
            plog.setLevel(old_level)

        # Faithful classification from the pipeline's OWN log lines (computed on
        # the exact `bt` grid by the exact pool_beat_evidence call — no fragile
        # grid reconstruction). chord_pipeline_v1 emits:
        #   "pooled A/B merge group(s)"             -> n_applied=A (full+partial)
        #   "PARTIALLY applied for K/B ... excluded: [...]"
        #   "rejected for R/B group(s) ..."
        total = len(groups)
        n_applied = new_partial = new_unpoolable = 0
        excluded_spans = 0
        lines = {"pooled": None, "partial": None, "rejected": None}
        for msg in cap.msgs:
            m = re.search(r"pooled (\d+)/(\d+) merge group", msg)
            if m:
                n_applied = int(m.group(1)); lines["pooled"] = msg
            m = re.search(r"PARTIALLY applied for (\d+)/", msg)
            if m:
                new_partial = int(m.group(1)); lines["partial"] = msg
                excluded_spans = msg.count("'span'")  # one 'span' key per excluded entry
            m = re.search(r"rejected for (\d+)/", msg)
            if m:
                new_unpoolable = int(m.group(1)); lines["rejected"] = msg
        new_full = n_applied - new_partial
        old_applied = new_full  # old all-or-nothing applied exactly the all-equal groups
        per_group = lines

        # real before/after chord changes at block midpoints
        base_ch = [c for c in base.chords if c["end_s"] > c["start_s"]]
        cons_ch = [c for c in cons.chords if c["end_s"] > c["start_s"]]
        changed = 0
        touched = 0
        for _l, spans in groups:
            for (t0, t1) in spans:
                mid = 0.5 * (t0 + t1)
                bc, cc = chord_at(base_ch, mid), chord_at(cons_ch, mid)
                if bc and cc:
                    touched += 1
                    if bc["label"] != cc["label"] or abs(
                            bc.get("confidence", 0) - cc.get("confidence", 0)) > 1e-9:
                        changed += 1

        rec = {
            "n_groups": len(groups),
            "OLD_all_or_nothing_groups_applied": old_applied,
            "NEW_full_pool": new_full,
            "NEW_partial_pool": new_partial,
            "NEW_unpoolable": new_unpoolable,
            "NEW_total_applied": new_full + new_partial,
            "spans_excluded_in_partials": excluded_spans,
            "block_midpoints_touched": touched,
            "block_midpoints_changed": changed,
            "per_group": per_group,
        }
        out["songs"][slug] = rec
        print(f"\n=== {slug} ===")
        print(json.dumps(rec, indent=2, default=str))

    agg = {}
    for k in ["n_groups", "OLD_all_or_nothing_groups_applied", "NEW_full_pool",
              "NEW_partial_pool", "NEW_unpoolable", "NEW_total_applied",
              "spans_excluded_in_partials"]:
        agg[k] = sum(out["songs"][s][k] for s in out["songs"])
    out["aggregate"] = agg
    print("\n=== AGGREGATE ===")
    print(json.dumps(agg, indent=2))
    p = Path(__file__).resolve().parent / "group_pool_graceful_verify_results.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
