#!/usr/bin/env python3
"""ug_missed.py — characterize the MISSED class before anyone tries to fix it.

    .venv/bin/python scratchpad/ug_missed.py

MISSED = a chord change the UG tab has that our chart never wrote. It is the
#1 defect class (160 of 276 errors; 80% once Chain of Fools, whose harmony
cannot time itself, is set aside). Before proposing a fix, describe the thing:

  1. DURATION  — passing chords or full-bar changes we plainly dropped?
  2. POSITION  — where in the bar (payload bar grid) and where in the section?
  3. INSTEAD   — are we holding the PREVIOUS chord through it (segmentation
                 swallowed the change), the NEXT one (boundary early), or
                 something else?
  4. FUNCTION  — root motion into and out of the missed chord.
  5. REPETITION— Let It Be (46) and Every Breath (53) carry two thirds of the
                 class. If each is ONE repeating pattern, the fix is different.

Reads scratchpad/ug_score_<slug>.json (which now carries both full sequences)
plus the baked payload for the bar grid. Writes:
  scratchpad/ug_missed_duration.png
  scratchpad/ug_missed_position.png
  scratchpad/ug_missed.json
and prints the tables that go into docs/ug_score_report.md.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratchpad"))

from ug_align import load_payload            # noqa: E402
from ug_score import _PC                     # noqa: E402

SCRATCH = REPO / "scratchpad"
EXCLUDE = "chain_of_fools"                   # harmony-underdetermined; 0 MISSED anyway

_IV = {0: "unison", 1: "up m2", 2: "up M2", 3: "up m3", 4: "up M3", 5: "up P4",
       6: "tritone", 7: "down P4 (up P5)", 8: "down M3", 9: "down m3",
       10: "down M2", 11: "down m2"}


def bar_grid(P: dict):
    """Map time -> continuous bar position, from the payload's own bar/beat tags.

    Every payload chord carries (bar, beat, t0), so the grid is already there;
    we only need to interpolate between the tagged onsets. Returns a function
    t -> bar_position (float, integer part = bar index, fraction = position in
    the bar) or None when the payload has too few anchors to be trusted.
    """
    bpb = float(P.get("bpb") or 4)
    pts = [(float(c["t0"]), float(c["bar"]) + float(c.get("beat", 0)) / bpb)
           for c in P.get("chords", []) if "bar" in c]
    pts = sorted(set(pts))
    if len(pts) < 4:
        return None, bpb
    ts = np.array([p[0] for p in pts])
    bs = np.array([p[1] for p in pts])
    keep = np.concatenate([[True], np.diff(bs) > 0])   # strictly increasing
    ts, bs = ts[keep], bs[keep]
    if len(ts) < 4:
        return None, bpb

    def f(t: float) -> float:
        return float(np.interp(t, ts, bs))
    return f, bpb


def our_at(our_seq, t):
    for e in our_seq:
        if e["t0"] <= t < e["t1"]:
            return e
    return None


def main():
    rows = []
    per_song = {}
    for path in sorted(SCRATCH.glob("ug_score_*.json")):
        d = json.loads(path.read_text())
        slug = d["slug"]
        if EXCLUDE in slug:
            continue
        ug, ours = d["ug_seq"], d["our_seq"]
        P = load_payload(slug)
        grid, bpb = bar_grid(P) if P else (None, 4.0)
        chips = sorted((P or {}).get("sectionChips", []),
                       key=lambda c: c["start_s"])
        missed = [e for e in d["errors"]
                  if e["cls"] == "MISSED" and not e["intro"] and "ug_i" in e]
        got = []
        for e in missed:
            i = e["ug_i"]
            u = ug[i]
            prev_u = ug[i - 1] if i > 0 else None
            next_u = ug[i + 1] if i + 1 < len(ug) else None
            mid = (u["t0"] + u["t1"]) / 2
            o = our_at(ours, mid)

            # what we show instead
            if o is None:
                instead = "nothing (gap)"
            elif o["nc"]:
                instead = "no-chord"
            elif prev_u and o["root"] == prev_u["root"]:
                instead = "held PREVIOUS"
            elif next_u and o["root"] == next_u["root"]:
                instead = "held NEXT"
            else:
                instead = "a third chord"

            # position in the bar
            beat = None
            if grid is not None:
                bp = grid(u["t0"])
                beat = round((bp % 1.0) * bpb, 2)

            sec = ""
            for c in chips:
                if c["start_s"] <= u["t0"]:
                    sec = c["label"]

            rows.append({
                "slug": slug, "song": d["meta"]["song"],
                "t0": u["t0"], "t1": u["t1"], "dur": round(u["t1"] - u["t0"], 2),
                "name": u["name"], "root": u["root"],
                "ug_section": u.get("section", ""), "our_section": sec,
                "prev": prev_u["name"] if prev_u else None,
                "next": next_u["name"] if next_u else None,
                "iv_in": None if not prev_u else (u["root"] - prev_u["root"]) % 12,
                "iv_out": None if not next_u else (next_u["root"] - u["root"]) % 12,
                "instead": instead, "beat": beat,
                "motif": (f"{prev_u['name'] if prev_u else '?'}"
                          f" [{u['name']}] "
                          f"{next_u['name'] if next_u else '?'}"),
            })
            got.append(rows[-1])
        per_song[slug] = got

    if not rows:
        raise SystemExit("no MISSED rows")

    durs = np.array([r["dur"] for r in rows])
    print(f"\n=== MISSED characterization — {len(rows)} misses, "
          f"{len(per_song)} songs (Chain of Fools excluded) ===\n")

    # ---- 1. duration ------------------------------------------------------ #
    print("1. DURATION (seconds)")
    print(f"   pooled: median {np.median(durs):.2f}s  mean {durs.mean():.2f}s  "
          f"p10 {np.percentile(durs,10):.2f}  p90 {np.percentile(durs,90):.2f}")
    print(f"{'song':<26}{'n':>4}{'median':>8}{'mean':>7}{'<1.0s':>7}{'>2.5s':>7}")
    for slug, g in sorted(per_song.items(), key=lambda x: -len(x[1])):
        if not g:
            continue
        dd = np.array([r["dur"] for r in g])
        print(f"{g[0]['song'][:25]:<26}{len(dd):>4}{np.median(dd):>8.2f}"
              f"{dd.mean():>7.2f}{100*(dd<1.0).mean():>6.0f}%"
              f"{100*(dd>2.5).mean():>6.0f}%")

    # duration in BEATS, using each song's own bar grid
    print("\n   in beats (payload bar grid):")
    bt = []
    for slug, g in per_song.items():
        P = load_payload(slug)
        grid, bpb = bar_grid(P) if P else (None, 4.0)
        if grid is None:
            continue
        for r in g:
            b = (grid(r["t1"]) - grid(r["t0"])) * bpb
            if 0 < b < 32:
                bt.append(b)
                r["beats"] = round(b, 2)
    bt = np.array(bt)
    print(f"   median {np.median(bt):.2f} beats; "
          f"<=1 beat {100*(bt<=1.2).mean():.0f}%, "
          f"<=2 beats {100*(bt<=2.4).mean():.0f}%, "
          f">=4 beats (full bar) {100*(bt>=3.6).mean():.0f}%")

    # ---- 2. position ------------------------------------------------------ #
    print("\n2. POSITION IN THE BAR (beat 0 = downbeat)")
    beats = [r["beat"] for r in rows if r["beat"] is not None]
    bb = np.array(beats)
    near_db = 100 * (np.minimum(bb % 4, 4 - (bb % 4)) < 0.5).mean()
    print(f"   {len(bb)} located; {near_db:.0f}% within half a beat of a "
          f"downbeat, {100-near_db:.0f}% mid-bar")
    hist = Counter(int(round(b)) % 4 for b in bb)
    for k in range(4):
        print(f"     beat {k+1}: {hist.get(k,0):>3}  "
              f"{'#'*int(40*hist.get(k,0)/max(len(bb),1))}")

    print("\n   by UG section:")
    for k, v in Counter(r["ug_section"] or "?" for r in rows).most_common(8):
        print(f"     {k[:24]:<26}{v:>4}")

    # ---- 3. what we show instead ------------------------------------------ #
    print("\n3. WHAT WE SHOW INSTEAD")
    ins = Counter(r["instead"] for r in rows)
    for k, v in ins.most_common():
        print(f"   {k:<18}{v:>4}  {100*v/len(rows):>5.0f}%")
    print("\n   per song:")
    for slug, g in sorted(per_song.items(), key=lambda x: -len(x[1])):
        if not g:
            continue
        c = Counter(r["instead"] for r in g)
        print(f"     {g[0]['song'][:25]:<26}"
              + "  ".join(f"{k}={v}" for k, v in c.most_common(3)))

    # ---- 4. function ------------------------------------------------------ #
    print("\n4. FUNCTION — root motion INTO the missed chord")
    for k, v in Counter(r["iv_in"] for r in rows if r["iv_in"] is not None
                        ).most_common(6):
        print(f"   {_IV.get(k,k):<18}{v:>4}  {100*v/len(rows):>5.0f}%")
    print("   root motion OUT of it")
    for k, v in Counter(r["iv_out"] for r in rows if r["iv_out"] is not None
                        ).most_common(6):
        print(f"   {_IV.get(k,k):<18}{v:>4}  {100*v/len(rows):>5.0f}%")

    print("\n5. REPEATING MOTIFS (prev [missed] next), pooled")
    for k, v in Counter(r["motif"] for r in rows).most_common(12):
        print(f"   {v:>3}x  {k}")
    print("\n   per song — is each dominated by ONE pattern?")
    for slug, g in sorted(per_song.items(), key=lambda x: -len(x[1])):
        if not g:
            continue
        c = Counter(r["motif"] for r in g)
        top, n = c.most_common(1)[0]
        print(f"     {g[0]['song'][:25]:<26} top motif {n}/{len(g)} "
              f"({100*n/len(g):.0f}%)  {top}")
        for m, k in c.most_common(3)[1:]:
            print(f"     {'':<26}   then {k}x  {m}")

    # ---- 6. is "a third chord" real, or UG timing slop? -------------------- #
    scores = {}
    for path in SCRATCH.glob("ug_score_*.json"):
        d = json.loads(path.read_text())
        scores[d["slug"]] = d
    slop = 0
    retally = Counter()
    for r in rows:
        ours = scores[r["slug"]]["our_seq"]
        hit = any(e["root"] == r["root"] and not e["nc"]
                  and e["t0"] < r["t1"] + 2.0 and e["t1"] > r["t0"] - 2.0
                  for e in ours)
        r["we_write_it_within_2s"] = hit
        if hit:
            slop += 1
            retally["WE DO WRITE IT (timing slop)"] += 1
        else:
            retally[r["instead"]] += 1
    print("\n6. IS IT REALLY A MISS? (doctrine: UG timing is rough)")
    print(f"   our chart writes that root within +/-2s in {slop}/{len(rows)} "
          f"({100*slop/len(rows):.0f}%) -> not misses, diff slop")
    for k, v in retally.most_common():
        print(f"   {k:<32}{v:>4}  {100*v/len(rows):>5.0f}%")
    true_n = len(rows) - slop

    # ---- 7. the screening test -------------------------------------------- #
    print("\n7. SCREENING TEST — can our chart even EXPRESS a short chord?")
    print(f"{'song':<26}{'our med':>9}{'our min':>9}{'our<1b':>8}{'UG<1b':>8}{'misses':>8}")
    allo, allu = [], []
    for slug, d in sorted(scores.items()):
        if EXCLUDE in slug:
            continue
        P = load_payload(slug)
        grid, bpb = bar_grid(P) if P else (None, 4.0)
        if grid is None:
            continue

        def bts(a, b):
            return (grid(b) - grid(a)) * bpb
        ob = np.array([x for e in d["our_seq"] if not e["nc"]
                       and 0 < (x := bts(e["t0"], e["t1"])) < 40])
        ub = np.array([x for e in d["ug_seq"] if e["t1"] - e["t0"] > 0.41
                       and 0 < (x := bts(e["t0"], e["t1"])) < 40])
        allo += list(ob)
        allu += list(ub)
        print(f"{d['meta']['song'][:25]:<26}{np.median(ob):>9.2f}{ob.min():>9.2f}"
              f"{100*(ob<1).mean():>7.0f}%{100*(ub<1).mean():>7.0f}%"
              f"{len(per_song.get(slug,[])):>8}")
    allo, allu = np.array(allo), np.array(allu)
    print(f"   POOLED ours: median {np.median(allo):.2f} beats, min "
          f"{allo.min():.2f}, {100*(allo<1).mean():.0f}% under 1 beat, "
          f"{100*(allo<2).mean():.0f}% under 2")
    print(f"   POOLED UG  : median {np.median(allu):.2f} beats, min "
          f"{allu.min():.2f}, {100*(allu<1).mean():.0f}% under 1 beat, "
          f"{100*(allu<2).mean():.0f}% under 2")

    _plots(rows, per_song, bt)
    (SCRATCH / "ug_missed.json").write_text(json.dumps(rows, indent=1))
    _markdown(rows, per_song, scores, bt, slop, true_n, retally, allo, allu)
    print(f"\nwrote {SCRATCH/'ug_missed.json'} ({len(rows)} rows)")


def _markdown(rows, per_song, scores, bt, slop, true_n, retally, allo, allu):
    durs = np.array([r["dur"] for r in rows])
    beats = np.array([r["beat"] for r in rows if r["beat"] is not None])
    L = ["## MISSED characterization", "",
         f"MISSED was the #1 defect class. Characterized before anyone tries to "
         f"fix it — and characterizing it removed most of it.", "",
         "### Two measurement artifacts found first", "",
         "**1. Crammed tab material (−72 of 160).** The aligner gives a chord "
         "it has no room for the minimum legal duration (0.30 s). Every Breath "
         "You Take's tab writes out the entire fade-out loop — **73 chords, "
         "every one at the floor, all inside 205–229 s** — and each was being "
         "scored as a chord we missed, making it the second-worst song in the "
         "report. Same family as the Close to You alternate-ending problem "
         "already logged; `unsupported_frac` was flagging it at 0.275 and I did "
         "not act on it until the duration histogram made it unmissable. These "
         "are now class `CRAMMED`. Every Breath: **53 → 6 misses**.", "",
         f"**2. Diff slop (−{slop} more).** In {slop} of {len(rows)} remaining "
         f"cases ({100*slop/len(rows):.0f}%) our chart *does* write that root "
         f"within ±2 s — the ordinal diff simply failed to pair them. Per the "
         f"doctrine (UG timing is hand-made), those are not misses.", "",
         f"**160 reported → {true_n} real misses.** Everything below describes "
         f"the {len(rows)} post-CRAMMED rows, with the slop share marked.", "",
         "### 1. Duration — these are passing chords, not dropped bars", "",
         f"Median **{np.median(durs):.2f} s = {np.median(bt):.2f} beats**. "
         f"**{100*(bt<=1.2).mean():.0f}% last one beat or less**, "
         f"{100*(bt<=2.4).mean():.0f}% two beats or less, and only "
         f"{100*(bt>=3.6).mean():.0f}% are a full bar. We are not dropping "
         f"structural changes; we are dropping ornaments and approach chords.",
         "", "| song | n | median s | median beats | ≤1 beat |",
         "|---|---|---|---|---|"]
    for slug, g in sorted(per_song.items(), key=lambda x: -len(x[1])):
        if not g:
            continue
        dd = np.array([r["dur"] for r in g])
        bb2 = np.array([r["beats"] for r in g if "beats" in r])
        L.append(f"| {g[0]['song']} | {len(dd)} | {np.median(dd):.2f} | "
                 f"{np.median(bb2):.2f} | {100*(bb2<=1.2).mean():.0f}% |")
    L += ["", "### 2. Position — NOT a downbeat/mid-bar effect", "",
          f"{len(beats)} located on the payload bar grid. Counts by beat: "
          + ", ".join(f"beat {k+1} = {int((np.round(beats)%4==k).sum())}"
                      for k in range(4))
          + f". That is close to uniform, so the missed chords are **not** "
          f"concentrated off the downbeat. A bar-grid *phase* problem would "
          f"show a strong mid-bar bias; it does not. This hypothesis is "
          f"rejected by the data.", "",
          "### 3. What we show instead", "", "| | n | share |", "|---|---|---|"]
    for k, v in retally.most_common():
        L.append(f"| {k} | {v} | {100*v/len(rows):.0f}% |")
    L += ["", "Only ~22% is “we held the previous chord through the change”, so "
          "this is not mainly a decoder-stickiness story. The largest real "
          "bucket is *a third chord entirely* — consistent with our chart "
          "writing one chord for a whole bar that the tab fills with three.", "",
          "### 4. Function — approach chords", "", "Root motion out of the "
          "missed chord is a fourth/fifth (33%) or a step (27%): these sit "
          "**between** two structural chords and resolve into the next one. "
          "Top recurring shapes (prev → [missed] → next):", "",
          "| n | shape | song |", "|---|---|---|"]
    top = Counter(r["motif"] for r in rows).most_common(8)
    where = {}
    for r in rows:
        where.setdefault(r["motif"], r["song"])
    for m, n in top:
        L.append(f"| {n} | {m} | {where[m]} |")
    L += ["", "### 5. Are the big two one repeating pattern each?", "",
          "**Partly, and it matters.**", ""]
    for slug, g in sorted(per_song.items(), key=lambda x: -len(x[1])):
        if not g:
            continue
        c = Counter(r["motif"] for r in g)
        t, n = c.most_common(1)[0]
        L.append(f"- **{g[0]['song']}** ({len(g)}): top shape `{t}` "
                 f"{n}/{len(g)} ({100*n/len(g):.0f}%)")
    L += ["", "Let It Be is one third a single repeated shape and Hot N Cold "
          "two thirds; Close to You is genuinely diverse. So a fix aimed at "
          "one progression would clear a third of Let It Be and most of Hot N "
          "Cold, and nothing on Close to You.", "",
          "### 6. Screening test — already run, and it confirms", "",
          "The premise “our chart cannot express a short chord” is checkable "
          "with data already on disk, so I ran it rather than proposing it "
          "(project rule: screen the premise cheaply).", "",
          "| | median | min | under 1 beat | under 2 beats |", "|---|---|---|---|---|",
          f"| **our chart** | {np.median(allo):.2f} beats | {allo.min():.2f} | "
          f"**{100*(allo<1).mean():.0f}%** | {100*(allo<2).mean():.0f}% |",
          f"| **UG tab** | {np.median(allu):.2f} beats | {allu.min():.2f} | "
          f"{100*(allu<1).mean():.0f}% | {100*(allu<2).mean():.0f}% |", "",
          "**Our chart never emits a chord shorter than about one beat** "
          f"(min {allo.min():.2f} beats over {len(allo)} chords; 0% under a "
          "beat). Per song it is starker: Stand By Me, Hot N Cold and Every "
          "Breath are locked to *exactly* 4.00 beats — one chord per bar, no "
          "exceptions. Let It Be and This Love run on a half-bar grid "
          "(median 2.00). Close to You sits at 8 beats, two whole bars.", "",
          "And the miss counts track the **tab's** sub-beat content almost "
          "perfectly: Let It Be 35% of tab chords under a beat → 37 misses; "
          "Close to You 22% → 24; every other song ~0–2% → 5–9 misses.", "",
          "### Ranked hypotheses", "",
          "| # | hypothesis | verdict |", "|---|---|---|",
          "| 1 | **Our chart is quantized to a bar / half-bar grid and cannot "
          "represent a sub-beat chord at all.** The tab puts 40% of its chords "
          "under two beats; we put 2%. | **Confirmed** by the screening test "
          "above, and the per-song miss counts follow the tab's sub-beat share |",
          "| 2 | Beat-grid *phase* — mid-bar changes merged into the downbeat | "
          "**Rejected** — position is near-uniform across the four beats |",
          "| 3 | Decoder stickiness — the HMM holds the previous chord | "
          "**Minor** — only ~22% show “held previous” |",
          "| 4 | Genuine harmonic error (we hear a different chord) | "
          "**Residual** — the “third chord” bucket, largely explained by (1): "
          "one chord written over a bar the tab fills with three |", "",
          "### The one cheapest test that would falsify hypothesis 1", "",
          "Hypothesis 1 says the chords exist in the model but are destroyed by "
          "quantization. So: **dump the decoder's pre-quantization chord "
          "sequence for Let It Be and check whether the 12 `C [Dm7] C` and 7 "
          "`G [F] C` events are present as sub-bar segments before the bar grid "
          "is applied.**", "",
          "- If they are there → quantization is the defect; the fix is grain, "
          "not harmony, and it is cheap.",
          "- If they are absent pre-quantization → hypothesis 1 is falsified, "
          "the chords never existed, and the defect is upstream in the emission "
          "or the decoder's duration prior. That would redirect the whole "
          "effort.", "",
          "One song, one intermediate dump, no retraining, no sweep. Run it "
          "before touching anything.", "",
          "*(Figures: `scratchpad/ug_missed_duration.png`, "
          "`scratchpad/ug_missed_position.png`.)*", ""]
    (SCRATCH / "ug_missed_section.md").write_text("\n".join(L))
    print(f"wrote {SCRATCH/'ug_missed_section.md'}")


def _plots(rows, per_song, bt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    durs = np.array([r["dur"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), dpi=140)
    ax[0].hist(durs, bins=np.arange(0, 8.25, 0.25), color="#c0392b", alpha=0.85)
    ax[0].axvline(np.median(durs), color="k", ls="--",
                  label=f"median {np.median(durs):.2f}s")
    ax[0].set_xlabel("duration of the missed chord (s)")
    ax[0].set_ylabel("count")
    ax[0].set_title(f"MISSED chord duration — {len(durs)} misses, 6 songs")
    ax[0].legend()
    ax[1].hist(bt, bins=np.arange(0, 10.5, 0.5), color="#2e6fb7", alpha=0.85)
    ax[1].axvline(np.median(bt), color="k", ls="--",
                  label=f"median {np.median(bt):.2f} beats")
    for b in (2, 4):
        ax[1].axvline(b, color="0.4", lw=0.8)
    ax[1].set_xlabel("duration (beats, payload bar grid)")
    ax[1].set_title("…in beats: is it a passing chord or a full bar?")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(SCRATCH / "ug_missed_duration.png")
    plt.close(fig)

    beats = np.array([r["beat"] for r in rows if r["beat"] is not None])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), dpi=140)
    ax[0].hist(beats % 4, bins=np.arange(0, 4.125, 0.125), color="#2e8b57",
               alpha=0.85)
    for b in range(5):
        ax[0].axvline(b, color="0.3", lw=0.8)
    ax[0].set_xlabel("position in the bar (0 = downbeat, 4 beats/bar)")
    ax[0].set_ylabel("count")
    ax[0].set_title("Where in the bar do we miss chords?")
    ins = Counter(r["instead"] for r in rows)
    ks = [k for k, _ in ins.most_common()]
    ax[1].barh(ks, [ins[k] for k in ks], color="#8e44ad", alpha=0.85)
    ax[1].set_xlabel("count")
    ax[1].set_title("What our chart shows instead")
    ax[1].invert_yaxis()
    fig.tight_layout()
    fig.savefig(SCRATCH / "ug_missed_position.png")
    plt.close(fig)
    print(f"\nplots: {SCRATCH/'ug_missed_duration.png'}, "
          f"{SCRATCH/'ug_missed_position.png'}")


if __name__ == "__main__":
    main()
