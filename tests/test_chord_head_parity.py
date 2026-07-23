"""tests/test_chord_head_parity.py — Phase 3 ChordHead byte-identity gate.

Proves the ported ``harmonia.stages.chord_head.NNLS24ChordHead`` reproduces the
live nnls24 chord stage byte-for-byte on the 8 frozen parity songs, for the three
net-covered stages (features / pre-coalesce labels / symbolic sections).

Two independent comparisons per song, both from REAL runs (no fabricated numbers):

  A. mine vs LIVE-fresh (same process, SAME beat grid ``bt``) — pure PORT
     correctness: my re-implemented ``_label_segments`` / ``_coalesce_labeled`` /
     ``_root_change_segs`` / ``_pool_root_proba_to_bars`` / section gating must
     equal chord_pipeline_v1's on identical inputs. bt-reproduction independent.

  B. mine vs the committed GOLDEN (harmonia/eval/golden/frozen_parity/*.golden.json)
     — the frozen-oracle gate: my output must match the frozen labels/vectors.
     Requires my recomputed ``bt`` to equal the golden's (checked via the beats
     grid sha256 as an explicit INPUT-IDENTITY guard).

Metadata-only keys (``provenance``, ``__audio_dependent_not_captured__``) are
excluded from the diff — they are annotation strings, not chord-stage output.

Run as a script for the per-song table:  python tests/test_chord_head_parity.py
Run under pytest:                         pytest -o addopts="" tests/test_chord_head_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from harmonia.eval import parity
from harmonia.eval.benchmark_set import LIVE_ORACLE_KWARGS, parity_songs
from harmonia.stages.chord_head import ChordHeadConfig, NNLS24ChordHead

CACHE_DIR = REPO / "data" / "cache"
WAV_DIR = CACHE_DIR / "_chordhead_parity_wav_tmp"
_META_KEYS = ("provenance", "__audio_dependent_not_captured__")


# ── build the golden-schema three-stage dict from MY ChordStageResult ─────────
def _my_stages_dict(res) -> dict:
    """Assemble the exact field layout parity._nnls24_stages emits, from the
    ported ChordHead's raw outputs, so parity._diff_value can compare directly."""
    n_beats = int(len(res.feat))
    features = {
        "n_frames": int(res.arr.shape[0]),
        "n_beats": n_beats,
        "bothchroma_arr": parity._arr_summary(res.arr),
        "bothchroma_times": parity._arr_summary(res.times),
        "feat": parity._arr_summary(res.feat),
        "beat_proba": parity._arr_summary(res.beat_proba),
        "key_name": res.key_name,
        "key_confidence": round(float(res.key_confidence), 6),
    }
    precoalesce = {
        "n_segs": int(len(res.segs)),
        "n_precoalesce": int(len(res.labeled)),
        "n_coalesced": int(len(res.coalesced)),
        "seg_boundaries": [[int(s), int(e)] for (s, e) in res.segs],
        "labels": [{"start_s": round(float(t0), 3), "end_s": round(float(t1), 3),
                    "label": lab, "conf": round(float(conf), 6)}
                   for (t0, t1, lab, conf) in res.labeled],
    }
    secs = res.sections
    sections = {
        "barlocked_fired": bool(secs is not None),
        "n_bars": int(len(res.bar_root)),
        "bar_root": parity._arr_summary(res.bar_root),
        "sections": [{"start_s": s.get("start_s"), "end_s": s.get("end_s"),
                      "n_bars": s.get("n_bars"), "label": s.get("label")}
                     for s in (secs or [])],
    }
    return {"nnls24_features": features,
            "nnls24_precoalesce": precoalesce,
            "nnls24_sections": sections}


def _strip_meta(stage: dict) -> dict:
    return {k: v for k, v in stage.items() if k not in _META_KEYS}


def _diff_stage(name: str, ref: dict, mine: dict) -> list:
    out: list = []
    parity._diff_value(name, _strip_meta(ref), _strip_meta(mine), 1e-6, out)
    return out


# ── one song: decode → beats → my head + live capture → diffs ─────────────────
def _run_one(entry: dict) -> dict:
    song_id = entry["song_id"]
    wav = WAV_DIR / f"{song_id}.wav"
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    parity.decode_to_wav(Path(entry["audio_path"]), wav)
    try:
        # 1. beat grid — exactly as capture() derives it (beatthis + bestfit)
        beats = parity._beats_stage(wav, LIVE_ORACLE_KWARGS.get("beat_period_mode", "bestfit"))
        bt = beats.pop("_bt")
        period = beats.pop("_period")
        duration_s = beats.pop("_duration_s")
        golden = parity.load_golden(song_id)
        g_stages = golden["stages"]

        # INPUT-IDENTITY guard: did beatthis reproduce the golden's grid?
        grid_match = (beats["grid"]["__sha256__"]
                      == g_stages["beats"]["grid"]["__sha256__"])

        # 2. MY ChordHead (the port) on that grid
        head = NNLS24ChordHead(ChordHeadConfig.live_defaults())
        res = head.run(wav, bt, period, duration_s)
        mine = _my_stages_dict(res)

        # 3. LIVE-fresh capture of the same three stages on the SAME grid
        live = parity._nnls24_stages(wav, bt, period, duration_s, CACHE_DIR)

        # comparison A: mine vs live-fresh (port correctness, bt-independent)
        # comparison B: mine vs committed golden (frozen-oracle gate)
        diffs_A, diffs_B = {}, {}
        for st in ("nnls24_features", "nnls24_precoalesce", "nnls24_sections"):
            diffs_A[st] = _diff_stage(st, live[st], mine[st])
            diffs_B[st] = _diff_stage(st, g_stages[st], mine[st])

        # per-stage max float delta on the array stats (for the report)
        def _max_delta(ref, mine_):
            d = 0.0
            for st in ("nnls24_features", "nnls24_sections"):
                for arr_key in ("feat", "beat_proba", "bothchroma_arr", "bar_root"):
                    if arr_key in ref[st] and arr_key in mine_[st]:
                        rs, ms = ref[st][arr_key].get("stats"), mine_[st][arr_key].get("stats")
                        if rs and ms:
                            for k in ("sum", "mean", "l2"):
                                if rs.get(k) is not None and ms.get(k) is not None:
                                    d = max(d, abs(rs[k] - ms[k]))
            return d

        return {
            "song_id": song_id,
            "grid_match": grid_match,
            "n_precoalesce": mine["nnls24_precoalesce"]["n_precoalesce"],
            "n_coalesced": mine["nnls24_precoalesce"]["n_coalesced"],
            "precoalesce_labels_match_live": len(diffs_A["nnls24_precoalesce"]) == 0,
            "precoalesce_labels_match_golden": len(diffs_B["nnls24_precoalesce"]) == 0,
            "features_match_golden": len(diffs_B["nnls24_features"]) == 0,
            "sections_match_golden": len(diffs_B["nnls24_sections"]) == 0,
            "all_match_live": all(len(v) == 0 for v in diffs_A.values()),
            "all_match_golden": all(len(v) == 0 for v in diffs_B.values()),
            "max_delta_vs_golden": round(_max_delta(g_stages, mine), 9),
            "diffs_A": diffs_A,
            "diffs_B": diffs_B,
        }
    finally:
        wav.unlink(missing_ok=True)


def run_all() -> list[dict]:
    songs = [s for s in parity_songs() if s["capturable"]]
    return [_run_one(s) for s in songs]


def _print_table(rows: list[dict]) -> bool:
    hdr = (f"{'song_id':<48} {'grid':<5} {'pre#':>5} {'coa#':>5} "
           f"{'preLbl=live':<12} {'preLbl=gold':<12} {'feat=gold':<10} "
           f"{'sec=gold':<9} {'maxΔ':>10}")
    print(hdr)
    print("-" * len(hdr))
    all_green = True
    for r in rows:
        all_green &= (r["all_match_live"] and r["all_match_golden"])
        print(f"{r['song_id']:<48} {'ok' if r['grid_match'] else 'DIFF':<5} "
              f"{r['n_precoalesce']:>5} {r['n_coalesced']:>5} "
              f"{('MATCH' if r['precoalesce_labels_match_live'] else 'DIFF'):<12} "
              f"{('MATCH' if r['precoalesce_labels_match_golden'] else 'DIFF'):<12} "
              f"{('MATCH' if r['features_match_golden'] else 'DIFF'):<10} "
              f"{('MATCH' if r['sections_match_golden'] else 'DIFF'):<9} "
              f"{r['max_delta_vs_golden']:>10.2e}")
    print("-" * len(hdr))
    print(f"VERDICT: {'GREEN (all 8 byte-identical vs live AND golden)' if all_green else 'RED (divergence — see detail)'}")
    for r in rows:
        for tag, diffs in (("live", r["diffs_A"]), ("golden", r["diffs_B"])):
            for st, dv in diffs.items():
                if dv:
                    print(f"  ✗ {r['song_id']} [{tag}] {st}: {len(dv)} divergence(s)")
                    for (p, kind, o, n) in dv[:6]:
                        print(f"      {p} [{kind}] {str(o)[:44]} → {str(n)[:44]}")
    return all_green


# ── pytest entry ─────────────────────────────────────────────────────────────
def test_chord_head_byte_identical():
    rows = run_all()
    assert rows, "no capturable parity songs found"
    for r in rows:
        assert r["all_match_live"], (
            f"{r['song_id']}: ChordHead != live code on same grid: {r['diffs_A']}")
        assert r["all_match_golden"], (
            f"{r['song_id']}: ChordHead != frozen golden: {r['diffs_B']}")


if __name__ == "__main__":
    ok = _print_table(run_all())
    sys.exit(0 if ok else 1)
