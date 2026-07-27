"""Bucket every GT region: AUTO-OK / EXTENSION-ONLY / DRIFT / NEEDS-EAR."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SCRATCH = Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-Code-harmonia/"
               "22c6b747-cea4-410f-85ae-f6ed58a9d2eb/scratchpad")
sys.path.insert(0, str(SCRATCH))
from triage_analyze import PC                                # noqa: E402

MUSX_LATENCY = 0.113          # measured 2026-07-27: the third-party model runs late
SHARP_MIN = 0.30              # a boundary measurement we are willing to trust

# ---- thresholds (all stated on the page) -----------------------------------
# Calibrated against the observed spread over all 194 regions, not guessed:
# musx_root_agree has p25/p50/p75 = 0.60/0.78/0.94, ext_gap 0.098/0.137/0.179.
T_ROOT_OK = 0.70      # third-party root agreement for "root confirmed"
T_ROOT_SOME = 0.50    # ... for "root corroborated"
T_TRIAD_RATIO = 0.70  # of the frames that agree on the root, this share must
                      #   also agree on major-vs-minor
T_MARGIN_ROOT = 0.06  # chroma: a chord on a DIFFERENT root fits better by this
T_EXT_GAP = 0.25      # chroma clearly prefers another 7th on the same root+triad;
                      #   per-chord p75/p90 over all 736 chords is 0.21/0.33, so this
                      #   is the top ~20% - not an everyday amount
T_SIB = 0.15          # fits worse than its own repeats elsewhere by this much
T_DRIFT = 0.30        # seconds of systematic boundary offset (>= half a beat
                      #   at these tempi)
T_NC = 0.40           # third-party P(no chord): above this it is abstaining
T_QUIET = -12.0       # dB below the song's own median level


def label_of(root, q, bass):
    if root is None:
        return "no chord"
    s = f"{PC[root]}:{q}"
    if bass is not None and bass != root:
        s += f"/{PC[bass]}"
    return s


def main():
    regions = json.loads((SCRATCH / "regions.json").read_text())
    bounds = json.loads((SCRATCH / "musx_bounds.json").read_text())
    cbounds = json.loads((SCRATCH / "offsets.json").read_text())

    songs = {}
    for sid, S in regions.items():
        bs = [b for b in bounds[sid] if b["sharp"] >= SHARP_MIN]
        off = np.array([b["offset"] - MUSX_LATENCY for b in bs])
        tt = np.array([b["t_gt"] for b in bs])
        slope, icept = (np.polyfit(tt, off, 1) if len(bs) >= 6 else (0.0, 0.0))
        S["timing"] = {
            "n": len(bs), "n_all": len(bounds[sid]),
            "median": float(np.median(off)) if len(off) else None,
            "mad": float(np.median(np.abs(off - np.median(off)))) if len(off) else None,
            "slope_s_per_min": float(slope * 60),
            "start": float(icept + slope * tt.min()) if len(bs) >= 6 else None,
            "end": float(icept + slope * tt.max()) if len(bs) >= 6 else None,
            "span_min": float((tt.max() - tt.min()) / 60) if len(bs) else None,
        }
        S["bounds"] = bs
        S["chroma_bounds"] = [b for b in cbounds[sid] if b.get("conf", 0) >= 0.15]
        songs[sid] = S

    for sid, S in songs.items():
        for r in S["regions"]:
            inb = [b for b in S["bounds"]
                   if r["t0"] - 0.05 <= b["t_gt"] <= r["t1"] + 0.05]
            offs = [b["offset"] - MUSX_LATENCY for b in inb]
            r["off_n"] = len(offs)
            r["off_med"] = float(np.median(offs)) if offs else None
            incb = [b for b in S["chroma_bounds"]
                    if r["t0"] - 0.05 <= b["t_gt"] <= r["t1"] + 0.05]
            r["coff_med"] = (float(np.median([b["offset"] for b in incb]))
                             if incb else None)
            r["off_detail"] = [
                {"t": b["t_gt"], "off": round(b["offset"] - MUSX_LATENCY, 3),
                 "from": b["from"], "to": b["to"], "sharp": round(b["sharp"], 2)}
                for b in inb]

            no_ev = ((r["musx_nc"] is not None and r["musx_nc"] > T_NC)
                     or r["rms_db"] < T_QUIET)
            r["no_evidence"] = bool(no_ev)

            ra = r["musx_root_agree"] or 0.0
            ta = r["musx_triad_agree"] or 0.0
            marg_root = r["margin_root"] if r["margin_root"] is not None else 0.0
            ext_gap = r["ext_gap"] if r["ext_gap"] is not None else 0.0

            # Q1 - is the ROOT right?  third-party frame agreement, corroborated
            #      by whether raw chroma prefers a chord on a DIFFERENT root.
            # a reading the third-party model spells on another note is NOT a
            # root disagreement when its notes fit inside the written chord plus
            # one colour tone (Em7/D read as G major)
            ext_cons = r.get("ext_cons_frac") or 0.0
            root_ok = (ra >= T_ROOT_OK
                       or (ra >= T_ROOT_SOME and marg_root <= T_MARGIN_ROOT)
                       or (ra + ext_cons >= T_ROOT_OK and ext_cons >= 0.30))
            root_via_ext = (not (ra >= T_ROOT_OK
                                 or (ra >= T_ROOT_SOME
                                     and marg_root <= T_MARGIN_ROOT))) and root_ok
            # Q2 - is major-vs-minor right?  of the frames that agree on the root,
            #      what share also agrees on the triad.
            triad_ok = ((ta / ra >= T_TRIAD_RATIO) if ra > 0.05 else False) \
                or root_via_ext
            # Q3 - is the TIMING right?
            drifting = (r["off_med"] is not None and r["off_n"] >= 2
                        and abs(r["off_med"]) >= T_DRIFT)
            sib_bad = (r["sib_gap"] or 0) >= T_SIB
            conflict = (r["off_med"] is not None and r["coff_med"] is not None
                        and abs(r["off_med"] - r["coff_med"]) >= 1.0)
            r["timing_conflict"] = bool(conflict)
            r["drifting"] = bool(drifting)
            r["root_ok"], r["triad_ok"] = bool(root_ok), bool(triad_ok)

            reasons = []
            if no_ev:
                bucket = "needs-ear"
                reasons.append("no independent evidence reaches this passage")
            elif not root_ok:
                bucket = "needs-ear"
                reasons.append("the independent model hears a different root here "
                               f"(it backs the written one {ra:.0%} of the time)")
                if marg_root > T_MARGIN_ROOT:
                    reasons.append("the raw spectrum also prefers a chord built on "
                                   "another note")
            elif not triad_ok:
                bucket = "needs-ear"
                reasons.append("the root holds but major-vs-minor does not: only "
                               f"{ta / max(ra, 1e-9):.0%} of the frames that back the "
                               "root also back the written quality")
            elif sib_bad:
                bucket = "needs-ear"
                reasons.append("the very same bars elsewhere in this song match the "
                               f"recording {r['sib_gap']:.2f} better than they do here")
            elif conflict:
                bucket = "needs-ear"
                reasons.append("the two independent timing measurements disagree by "
                               "more than a second")
            elif drifting:
                bucket = "drift"
                reasons.append("the chord names hold, but the bar lines sit "
                               f"{abs(r['off_med']):.2f} s "
                               f"{'late' if r['off_med'] < 0 else 'early'} "
                               "against the recording")
            elif (r.get("ext_gap_frac") or 0) >= 0.18 or ext_gap >= T_EXT_GAP:
                bucket = "extension-only"
                reasons.append("root and quality confirmed; only the 7th / colour "
                               "tone is in question")
            else:
                bucket = "auto-ok"
            r["bucket"] = bucket
            r["reasons"] = reasons

            d = 0.0
            d += 2.4 * max(0.0, 1.0 - min(1.0, ra + 0.7 * ext_cons))
            d += 1.4 * max(0.0, 1.0 - (ta / ra if ra > 0.05 else 0.0))
            d += 1.2 * min(max(marg_root, 0.0), 0.4) / 0.1
            d += 1.0 * max(0.0, (r["sib_gap"] or 0.0)) / 0.3
            d += 0.9 * (abs(r["off_med"]) / 0.5 if r["off_med"] is not None else 0.0)
            d += 1.5 if conflict else 0.0
            d += 1.6 if no_ev else 0.0
            d += 0.6 * max(0.0, (0.35 - (r["gt_fit"] or 0.0)) / 0.35)
            r["doubt"] = float(d)

    order = ("auto-ok", "extension-only", "drift", "needs-ear")
    tot = {}
    print(f"{'song':<24}" + "".join(f"{k:>16}" for k in order) + "     total")
    for sid, S in songs.items():
        agg = {}
        for r in S["regions"]:
            agg[r["bucket"]] = agg.get(r["bucket"], 0.0) + (r["t1"] - r["t0"])
            tot[r["bucket"]] = tot.get(r["bucket"], 0.0) + (r["t1"] - r["t0"])
        T = sum(agg.values())
        print(f"{sid:<24}" + "".join(f"{agg.get(k,0)/T:>15.0%} " for k in order)
              + f"    {T/60:.1f} min")
    T = sum(tot.values())
    print(f"{'POOLED (N=7)':<24}" + "".join(f"{tot.get(k,0)/T:>15.0%} " for k in order)
          + f"    {T/60:.1f} min")
    print("minutes:", {k: round(tot.get(k, 0) / 60, 1) for k in order})
    n = {}
    for S in songs.values():
        for r in S["regions"]:
            n[r["bucket"]] = n.get(r["bucket"], 0) + 1
    print("regions:", n, "total", sum(n.values()))
    print("\ntiming per song (latency-corrected):")
    for sid, S in songs.items():
        t = S["timing"]
        print(f"  {sid:<24} median {t['median']:+.3f}s  MAD {t['mad']:.3f}  "
              f"slope {t['slope_s_per_min']:+.3f} s/min  "
              f"start {t['start']:+.2f} -> end {t['end']:+.2f}")

    # ---- the colour-tone list, at CHORD level -------------------------------
    # A region carries one verdict, but "the 7th is wrong" is a property of a
    # single chord, so this list is built per chord and is deliberately not
    # bucketed by region.
    colour = []
    for sid, S in songs.items():
        for r in S["regions"]:
            if r["bucket"] == "needs-ear" and not r["triad_ok"]:
                continue
            for c in r["chords"]:
                if c.get("ext_gap") is None or c["ext_gap"] < T_EXT_GAP:
                    continue
                if (c.get("musx_root_agree") or 0) < T_ROOT_SOME \
                        and not c.get("ext_consistent"):
                    continue
                eb = c.get("ext_best")
                colour.append({
                    "song": sid, "t0": c["t0"], "t1": c["t1"],
                    "gt": c["label"],
                    "suggest": (f"{PC[eb[0]]}:{eb[1]}" if eb else None),
                    "gap": round(c["ext_gap"], 3),
                    "gt_fit": round(c["gt_fit"], 3),
                    "ext_fit": round(c["ext_fit"], 3),
                    "musx": (f"{PC[c['musx_top_root']]}:{c['musx_top_triad']}"
                             if c.get("musx_top_root") is not None else None),
                    "musx_root_agree": round(c.get("musx_root_agree") or 0, 2),
                    "rootless": bool(c.get("ext_consistent")),
                    "region": r["region_idx"],
                })
    colour.sort(key=lambda x: -x["gap"])
    print(f"\ncolour-tone flags (per chord): {len(colour)}")
    for x in colour[:12]:
        print(f"   {x['song']:<22} {x['t0']:7.2f}  {x['gt']:<10} -> {x['suggest']:<10} "
              f"gap {x['gap']:+.3f}  third-party hears {x['musx']}")
    (SCRATCH / "colour.json").write_text(json.dumps(colour))
    (SCRATCH / "gt_triage.json").write_text(json.dumps(songs))


if __name__ == "__main__":
    main()
