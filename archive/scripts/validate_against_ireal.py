"""validate_against_ireal.py — honest chord accuracy against iReal Pro GT.

Unblocks docs/known_issues.md #35 ("GT-eval blocked by a chart timeline
mismatch").  The fix #35 asked for was: put inference + iReal GT on a *shared
audio clock via a single DTW pass*.  ``harmonia.irealb_aligner`` already does
exactly that — it transfers timestamps from the inferred chart (real audio
clock) onto the iReal GT sequence.  The stale ``irealb_<slug>.html`` artifacts
were produced by an older aligner that under-detected repeats (autumn_leaves GT
span 160s vs inferred 422s); the *current* aligner tiles correctly (span 422 =
422, 8 choruses).  So we re-run the alignment fresh here rather than trusting
the cached HTML match fields.

Data sources (no pipeline re-run needed):
  * inferred chords  ← embedded ``const P = {...}`` in docs/plots/inferred_<slug>.html
                       (this is the *current production model's* output on the
                       real audio — root pc, quality dict lv.*, t0/t1, conf)
  * iReal GT         ← corpus tune (data/ireal/{jazz1460,pop400}.txt) → tune_to_mma

Alignment: align_irealb_to_inferred(mma, p_chords) → each GT chord gets t0/t1 on
the inferred clock.  We then pair each GT chord with the inferred chord active at
its midpoint (that midpoint lands inside the DTW-matched inferred segment) and
score root / quality-family / joint.

PREMISE GATE (CLAUDE.md #2, #35): a song is only counted toward the headline if
its GT alignment covers the track (gt_span / inf_span >= COVERAGE_MIN) and the
fraction of GT chords that received a timestamp (non-gap) >= ALIGNED_MIN.
Songs failing the gate are reported separately, never folded into the corpus
number — a bad corpus match (e.g. a jazz standard vs a pop cover of the same
title) or a mis-tiled alignment would otherwise fabricate an accuracy number.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.data.ireal_corpus import load_playlist, tune_to_mma  # noqa: E402
from harmonia.irealb_aligner import align_irealb_to_inferred  # noqa: E402
from harmonia.tab_aligner import _parse_ireal, _family  # noqa: E402

PLOTS = REPO / "docs" / "plots"
COVERAGE_MIN = 0.70   # gt_span / inf_span
ALIGNED_MIN = 0.70    # fraction of GT chords with a timestamp (non-gap)

# inferred_<slug> → (corpus_file, tune title).  Only songs whose iReal chart is
# genuinely the same tune as the audio.  "adele_hello" maps to the jazz standard
# "Hello" which is NOT Adele's song — left in deliberately as a gate negative
# control (it should fail coverage/cost and be excluded).
SONG_MAP: dict[str, tuple[str, str]] = {
    "adele_hello_official_music_video": ("jazz1460", "Hello"),
    "anthropology": ("jazz1460", "Anthropology"),
    "anthropology_phone": ("jazz1460", "Anthropology"),
    "autumn_leaves": ("jazz1460", "Autumn Leaves"),
    "autumn_leaves_remastered": ("jazz1460", "Autumn Leaves"),
    "blue_bossa": ("jazz1460", "Blue Bossa"),
    "blue_bossa_150bpm_backing_track": ("jazz1460", "Blue Bossa"),
    "blue_skies": ("jazz1460", "Blue Skies"),
    "bye_bye_blackbird": ("jazz1460", "Bye Bye Blackbird"),
    "muppets_kermit_its_not_easy_being_green_original": ("jazz1460", "Bein' Green"),
    "my_baby_just_cares_for_me": ("jazz1460", "My Baby Just Cares For Me"),
    "ray_charles_georgia_on_my_mind_official_video": ("jazz1460", "Georgia On My Mind"),
    "satin_doll": ("jazz1460", "Satin Doll"),
    "the_beatles_the_beatles_let_it_be_official_music_video_remas": ("pop400", "Let It Be"),
}

_dec = json.JSONDecoder()


def load_inferred(slug: str) -> list[dict] | None:
    fp = PLOTS / f"inferred_{slug}.html"
    if not fp.exists():
        return None
    h = fp.read_text()
    i = h.find("const P = ")
    if i < 0:
        return None
    i += len("const P = ")
    return _dec.raw_decode(h[i:])[0]["chords"]


def build_tune_index() -> dict[tuple[str, str], object]:
    idx: dict[tuple[str, str], object] = {}
    for corpus in ("jazz1460", "pop400"):
        fp = REPO / "data" / "ireal" / f"{corpus}.txt"
        if not fp.exists() or fp.stat().st_size == 0:
            continue
        for t in load_playlist(fp):
            idx.setdefault((corpus, t.title), t)
    return idx


def inferred_at(t: float, inf: list[dict]) -> dict | None:
    """Inferred chord whose [t0,t1] contains t (or nearest by t0)."""
    best = None
    for c in inf:
        t0 = c.get("t0")
        t1 = c.get("t1")
        if t0 is None:
            continue
        if t1 is None:
            t1 = t0 + 1
        if t0 <= t <= t1:
            return c
        if best is None or abs(c["t0"] - t) < abs(best["t0"] - t):
            best = c
    return best


def inf_quality(c: dict) -> str:
    """The inferred chord's exact quality token (lv.exact.q), e.g. '-7', '7'."""
    return (c.get("lv", {}).get("exact", {}) or {}).get("q", "") or ""


def score_song(slug: str, tune, inf: list[dict]) -> dict:
    mma = tune_to_mma(tune)
    res = align_irealb_to_inferred(mma, inf)
    inf_span = max((c["t1"] for c in inf if c.get("t1") is not None), default=0.0)

    aligned = [c for c in res.chords if c.get("t0") is not None]
    gt_span = max((c["t1"] for c in aligned), default=0.0)
    n_gt = len(res.chords)
    coverage = gt_span / inf_span if inf_span else 0.0
    aligned_frac = len(aligned) / n_gt if n_gt else 0.0

    pairs = []          # (gt_pc, gt_fam, inf_pc, inf_fam, conf)
    for gc in aligned:
        gt_pc, gt_q = _parse_ireal(gc["label"])
        if gt_pc < 0:               # N.C. — skip
            continue
        mid = 0.5 * (gc["t0"] + gc["t1"])
        ic = inferred_at(mid, inf)
        if ic is None:
            continue
        inf_pc = ic.get("root", -1)
        inf_q = inf_quality(ic)
        conf = (ic.get("lv", {}).get("exact", {}) or {}).get("c", 0.0)
        pairs.append((gt_pc, _family(gt_q), inf_pc, _family(inf_q), float(conf),
                      gt_q))

    n = len(pairs)
    root_ok = sum(1 for g, _, i, _, _, _ in pairs if g == i)
    fam_ok = sum(1 for _, gf, _, ifm, _, _ in pairs if gf == ifm)
    joint_ok = sum(1 for g, gf, i, ifm, _, _ in pairs if g == i and gf == ifm)
    # majmin: collapse everything to {maj-ish, min-ish} for a coarse family read
    def majmin(f):
        return "min" if f in ("min", "hdim", "dim") else "maj"
    mm_ok = sum(1 for _, gf, _, ifm, _, _ in pairs if majmin(gf) == majmin(ifm))
    # root-conditioned quality: quality accuracy among root-correct chords
    root_correct = [(gf, ifm) for g, gf, i, ifm, _, _ in pairs if g == i]
    q_given_root = (sum(1 for gf, ifm in root_correct if gf == ifm)
                    / len(root_correct)) if root_correct else 0.0

    # per-GT-family breakdown (how well is each true family recognised, root+fam)
    by_fam: dict[str, list[int]] = defaultdict(list)
    for g, gf, i, ifm, _, _ in pairs:
        by_fam[gf].append(1 if (g == i and gf == ifm) else 0)

    return {
        "slug": slug,
        "tune": tune.title,
        "n_gt_chords": n_gt,
        "n_scored": n,
        "coverage": round(coverage, 3),
        "aligned_frac": round(aligned_frac, 3),
        "n_repeats": res.n_repeats,
        "transpose": res.transpose_semitones,
        "dtw_cost": res.dtw_cost,
        "root_acc": round(root_ok / n, 3) if n else 0.0,
        "family_acc": round(fam_ok / n, 3) if n else 0.0,
        "joint_acc": round(joint_ok / n, 3) if n else 0.0,
        "majmin_acc": round(mm_ok / n, 3) if n else 0.0,
        "q_given_root": round(q_given_root, 3),
        "by_family": {k: [sum(v), len(v)] for k, v in by_fam.items()},
        "pairs": [(g, gf, i, ifm, round(c, 4)) for g, gf, i, ifm, c, _ in pairs],
        "passes_gate": coverage >= COVERAGE_MIN and aligned_frac >= ALIGNED_MIN,
    }


# Deliberately-wrong tunes for the spurious-alignment floor (CLAUDE.md #2/#3).
# align_irealb_to_inferred picks the best of 12 transpositions and warps to fit,
# so even a WRONG chart aligns to some baseline root agreement.  We must subtract
# this floor before claiming recognition.
CONTROL_TUNES = [("jazz1460", "Blue Bossa"), ("jazz1460", "Autumn Leaves"),
                 ("jazz1460", "Giant Steps"), ("pop400", "Let It Be")]


def shuffled_floor(slug: str, inf: list[dict], tune_idx, own_title: str) -> float:
    """Mean root accuracy of this inferred chart aligned to *wrong* tunes."""
    accs = []
    for corpus, title in CONTROL_TUNES:
        if title == own_title:
            continue
        tune = tune_idx.get((corpus, title))
        if tune is None:
            continue
        try:
            r = score_song(slug, tune, inf)
        except Exception:
            continue
        if r["n_scored"] >= 20:
            accs.append(r["root_acc"])
    return round(sum(accs) / len(accs), 3) if accs else 0.0


def main():
    tune_idx = build_tune_index()
    results = []
    for slug, (corpus, title) in SONG_MAP.items():
        tune = tune_idx.get((corpus, title))
        inf = load_inferred(slug)
        if tune is None or inf is None:
            print(f"  skip {slug}: tune={tune is not None} inf={inf is not None}")
            continue
        r = score_song(slug, tune, inf)
        if r["passes_gate"]:
            r["root_floor"] = shuffled_floor(slug, inf, tune_idx, title)
            r["root_lift"] = round(r["root_acc"] - r["root_floor"], 3)
        else:
            r["root_floor"] = None
            r["root_lift"] = None
        results.append(r)
        flag = "OK " if r["passes_gate"] else "GATE-FAIL"
        fl = f"floor={r['root_floor']:.2f} lift={r['root_lift']:+.2f}" if r["passes_gate"] else ""
        print(f"  [{flag}] {slug[:40]:40} cov={r['coverage']:.2f} "
              f"n={r['n_scored']:3} root={r['root_acc']:.2f} {fl} "
              f"fam={r['family_acc']:.2f} joint={r['joint_acc']:.2f} reps={r['n_repeats']}")

    # ── corpus aggregate over gate-passing songs (pooled chord-level) ──
    passing = [r for r in results if r["passes_gate"]]
    pool = [p for r in passing for p in r["pairs"]]  # (g,gf,i,ifm,c)
    def _majmin(f):
        return "min" if f in ("min", "hdim", "dim") else "maj"
    N = len(pool)
    agg = {
        "n_songs_total": len(results),
        "n_songs_passing": len(passing),
        "n_chords_pooled": N,
        "root_acc": round(sum(1 for g, _, i, _, _ in pool if g == i) / N, 3) if N else 0,
        "family_acc": round(sum(1 for _, gf, _, ifm, _ in pool if gf == ifm) / N, 3) if N else 0,
        "joint_acc": round(sum(1 for g, gf, i, ifm, _ in pool if g == i and gf == ifm) / N, 3) if N else 0,
        "majmin_acc": round(sum(1 for _, gf, _, ifm, _ in pool if _majmin(gf) == _majmin(ifm)) / N, 3) if N else 0,
    }
    # per-family pooled
    fam_pool: dict[str, list[int]] = defaultdict(list)
    for g, gf, i, ifm, _ in pool:
        fam_pool[gf].append(1 if (g == i and gf == ifm) else 0)
    agg["by_family"] = {k: [sum(v), len(v), round(sum(v) / len(v), 3)]
                        for k, v in sorted(fam_pool.items())}
    # confidence correlation: high-conf (>=0.5) vs low-conf root accuracy
    hi = [(g, i) for g, _, i, _, c in pool if c >= 0.5]
    lo = [(g, i) for g, _, i, _, c in pool if c < 0.5]
    agg["conf_split"] = {
        "high_conf_n": len(hi),
        "high_conf_root_acc": round(sum(1 for g, i in hi if g == i) / len(hi), 3) if hi else 0,
        "low_conf_n": len(lo),
        "low_conf_root_acc": round(sum(1 for g, i in lo if g == i) / len(lo), 3) if lo else 0,
    }

    floors = [r["root_floor"] for r in passing if r.get("root_floor") is not None]
    lifts = [r["root_lift"] for r in passing if r.get("root_lift") is not None]
    agg["mean_root_floor"] = round(sum(floors) / len(floors), 3) if floors else None
    agg["mean_root_lift"] = round(sum(lifts) / len(lifts), 3) if lifts else None

    print("\n=== CORPUS (gate-passing songs, pooled) ===")
    print(f"  songs {agg['n_songs_passing']}/{agg['n_songs_total']}  chords {N}")
    print(f"  root={agg['root_acc']:.3f}  family={agg['family_acc']:.3f}  "
          f"joint={agg['joint_acc']:.3f}  majmin={agg['majmin_acc']:.3f}")
    print(f"  spurious-align floor(mean)={agg['mean_root_floor']}  "
          f"mean per-song lift={agg['mean_root_lift']}")
    print(f"  by-family(root+fam): {agg['by_family']}")
    print(f"  conf split: {agg['conf_split']}")

    out = {"aggregate": agg, "per_song": results,
           "gate": {"coverage_min": COVERAGE_MIN, "aligned_min": ALIGNED_MIN}}
    outp = REPO / "data" / "ireal_gt_validation_set.json"
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {outp}")
    return out


if __name__ == "__main__":
    main()
