"""bass_premise_check.py — does ANY bass source show D where the decoder wrote F?

PREMISE UNDER TEST (Louis, 2026-07-30).  Half of Let It Be's MISSED chords are
"decoder-level absorption of shared-tone passing chords": all 13 D-7 passing
chords decode as their neighbour F:maj (Dm7 = D-F-A-C contains F-A-C, so the
ONLY spectral discriminator is the D in the bass).  Before building any repair,
check the cheapest falsifier: **at those 13 spots, does a bass source actually
read D?**  If none does at a majority of spots, the premise is dead and no
amount of decode-time machinery can recover them (CLAUDE.md rule #2).

Bass sources tested, in the order the mission names them:

1. ``musx_post``  — music-x-lab's own frame BASS posterior (``musx_probs/*.npz``
   stream 1, shape ``(n_frame, 13)``).  **Column 0 is "no bass"; column i>=1 is
   pitch class i-1** (verified in the vendored decoder:
   ``xhmm_ismir.get_chord_tag_obs`` does ``result_array[:,1] += 1  # bass
   adjust`` on a ``Chord.bass`` field that ``shift_complex_chord_array`` rotates
   mod 12, i.e. an ABSOLUTE pitch class).  This is a *trained* bass head, not a
   chroma blob, so it is the one source the bass doctrine
   (``docs/harmonic_key_investigation.md``) never banned.
2. ``musx_lab``   — the sounding bass of music-x-lab's decoded label
   (``musx_bass.bass_pc_of_label``).  Known-dead going in (the label IS F:maj);
   included as the control that the harness reads the right timestamps.
3. ``nnls``       — NNLS-24 bothchroma bass half, argmax, with the v3
   ``BASS_MARGIN = 1.5`` decisiveness guard.  Bass as a NOTE, never as mass.
4. ``musx_triad`` — not a bass source: the triad posterior, asked whether
   ``D:min`` carries secondary probability mass where the argmax says ``F:maj``.
   Tests the alternative story "the chord is in the posterior, the Viterbi
   duration penalty crushes it" rather than "only the bass knows".

WHAT THIS SCRIPT DOES NOT DO
----------------------------
* It does not measure whether a repair would help; it only asks whether the
  evidence exists.  A source can read D at every spot and still be useless if it
  also reads D all over the neighbouring F chords — so the script ALSO measures
  the false-positive side: how often each source reads the substitute bass on
  the *unambiguous* F spans that must not be touched.
* UG timings are hand-made (``docs/ug_alignment_brick.md``), so an exact-span
  reading can miss by a beat.  Both the exact-span reading and the best reading
  in a +-0.4 s window are reported; the window figure is an UPPER bound.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harmonia.models.musx_bass import bass_pc_of_label, _parse_root  # noqa: E402
from harmonia.models.musx_redecode import FRAME_DT  # noqa: E402

PC = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
TRIADS = ["maj", "min", "sus4", "sus2", "dim", "aug"]
BASS_MARGIN = 1.5          # v3's decisiveness guard, harmonic_key_investigation
WINDOW = 0.4               # s of UG timing slop allowed in the "best in window"

# NNLS bothchroma index 0 is A; roll to a C-first frame.  IMPORTED, not
# retyped: the first version of this script hard-coded ``-9`` instead of ``+9``,
# which differs by a tritone (18 == 6 mod 12) and silently relabelled every NNLS
# reading — D was printed as Ab, F as B.  CLAUDE.md error-pattern #1: unit-test
# the load-bearing constant against the upstream reference instead of restating
# it.
from harmonia.models.nnls_features import _ROLL_TO_C  # noqa: E402


# ── loaders ──────────────────────────────────────────────────────────────────

def load_score(slug: str) -> dict:
    return json.loads((REPO / "scratchpad" / f"ug_score_{slug}.json").read_text())


def load_musx_probs(slug: str) -> list[np.ndarray]:
    z = np.load(REPO / "data" / "cache" / "musx_probs" / f"{slug}.npz")
    return [z[n] for n in ("triad", "bass", "s7", "s9", "s11", "s13")]


def load_musx_lab(slug: str) -> list[tuple[float, float, str]]:
    p = REPO / "data" / "cache" / "musx_infer" / f"{slug}_submission.lab"
    out = []
    for line in p.read_text().splitlines():
        f = line.split()
        if len(f) >= 3:
            out.append((float(f[0]), float(f[1]), f[2]))
    return out


def load_nnls(slug: str):
    z = np.load(REPO / "data" / "cache" / "nnls_infer" / f"{slug}.npz")
    arr, times = z["arr"], z["times"]
    bass = np.roll(np.asarray(arr[:, :12], np.float64), _ROLL_TO_C, axis=1)
    return bass, np.asarray(times, np.float64)


# ── per-span readings ────────────────────────────────────────────────────────

def musx_bass_read(prob_bass: np.ndarray, t0: float, t1: float):
    """(top pc, margin, mass on top, mass on runner-up) over [t0,t1) frames.

    Column 0 ("no bass") is dropped: the question is *which* pitch class, and a
    high N mass just means the frame is quiet, which the caller sees as a low
    absolute mass on the winner anyway.
    """
    a = max(0, int(round(t0 / FRAME_DT)))
    b = min(prob_bass.shape[0], max(a + 1, int(round(t1 / FRAME_DT))))
    m = prob_bass[a:b, 1:13].mean(0)
    o = np.argsort(m)[::-1]
    top, run = int(o[0]), int(o[1])
    margin = float(m[top] / max(m[run], 1e-9))
    return top, margin, float(m[top]), float(m[run])


def musx_triad_read(prob_triad: np.ndarray, t0: float, t1: float, k: int = 3):
    """Top-k of the 73-way triad posterior over [t0,t1), as (name, mass)."""
    a = max(0, int(round(t0 / FRAME_DT)))
    b = min(prob_triad.shape[0], max(a + 1, int(round(t1 / FRAME_DT))))
    m = prob_triad[a:b].mean(0)
    out = []
    for i in np.argsort(m)[::-1][:k]:
        i = int(i)
        out.append(("N" if i == 0 else
                    f"{PC[(i - 1) % 12]}:{TRIADS[(i - 1) // 12]}", float(m[i])))
    return out


def triad_mass_of(prob_triad: np.ndarray, t0: float, t1: float,
                  root: int, kind: str) -> float:
    a = max(0, int(round(t0 / FRAME_DT)))
    b = min(prob_triad.shape[0], max(a + 1, int(round(t1 / FRAME_DT))))
    col = 1 + TRIADS.index(kind) * 12 + root
    return float(prob_triad[a:b, col].mean())


def nnls_bass_read(bass: np.ndarray, times: np.ndarray, t0: float, t1: float):
    m = (times >= t0) & (times < t1)
    if not m.any():
        m = np.argmin(np.abs(times - 0.5 * (t0 + t1)))
        v = bass[m]
    else:
        v = bass[m].mean(0)
    o = np.argsort(v)[::-1]
    top, run = int(o[0]), int(o[1])
    return top, float(v[top] / max(v[run], 1e-9))


def lab_at(labels, t: float) -> str | None:
    for t0, t1, s in labels:
        if t0 <= t < t1:
            return s
    return None


# ── the check ────────────────────────────────────────────────────────────────

def q_root(name: str) -> int | None:
    """Root pc of a UG/chart chord name like 'D-7', 'Bb', 'F^7'."""
    return _parse_root(name)


def run(slug: str, ug_filter: str | None, verbose: bool = True) -> dict:
    sc = load_score(slug)
    probs = load_musx_probs(slug)
    triad, pbass = probs[0], probs[1]
    labels = load_musx_lab(slug)
    nb, nt = load_nnls(slug)

    rows = []
    for e in sc["errors"]:
        if e["cls"] != "MISSED" or e.get("ug") is None:
            continue
        if ug_filter and e["ug"] != ug_filter:
            continue
        t0, t1 = float(e["t0"]), float(e["t1"])
        want = q_root(e["ug"])
        if want is None:
            continue
        lab = lab_at(labels, 0.5 * (t0 + t1))
        lab_bass = bass_pc_of_label(lab) if lab else None
        lab_root = _parse_root(lab.split("/")[0].split(":")[0]) if lab else None

        mtop, mmarg, mtm, mrm = musx_bass_read(pbass, t0, t1)
        wtop, wmarg = None, 0.0
        # best reading inside the UG-slop window (upper bound)
        for sh in np.arange(-WINDOW, WINDOW + 1e-9, 0.1):
            tp, mg, _, _ = musx_bass_read(pbass, t0 + sh, t1 + sh)
            if tp == want and mg > wmarg:
                wtop, wmarg = tp, mg
        ntop, nmarg = nnls_bass_read(nb, nt, t0, t1)

        rows.append({
            "t0": t0, "t1": t1, "dur": round(t1 - t0, 2), "ug": e["ug"],
            "want_pc": want, "musx_lab": lab, "lab_root": lab_root,
            "lab_bass": lab_bass,
            "musx_post_pc": mtop, "musx_post_margin": round(mmarg, 2),
            "musx_post_mass": round(mtm, 3), "musx_post_run": round(mrm, 3),
            "musx_post_hit": mtop == want,
            "musx_post_decisive": mtop == want and mmarg >= BASS_MARGIN,
            "musx_post_win_hit": wtop == want,
            "musx_post_win_margin": round(wmarg, 2),
            "nnls_pc": ntop, "nnls_margin": round(nmarg, 2),
            "nnls_hit": ntop == want,
            "nnls_decisive": ntop == want and nmarg >= BASS_MARGIN,
            "lab_bass_hit": lab_bass == want,
            "triad_top3": musx_triad_read(triad, t0, t1),
        })

    # ── taxonomy ────────────────────────────────────────────────────────────
    # Which MISSED chords could a BASS discriminator possibly reach?  Three
    # disjoint classes, and only the middle one is this mission's business.
    for r in rows:
        if r["lab_root"] == r["want_pc"]:
            r["klass"] = "PRESENT"        # musx already decoded it; lost later
        elif r["musx_post_decisive"] or r["nnls_decisive"]:
            r["klass"] = "BASS"           # bass names the root musx did not
        elif r["musx_post_hit"] or r["nnls_hit"]:
            r["klass"] = "BASS-WEAK"      # right pc, margin under the guard
        else:
            r["klass"] = "NO-EVIDENCE"    # no bass source names it at all
    kl = Counter(r["klass"] for r in rows)

    n = len(rows)
    summ = {
        "klass": dict(kl),
        "slug": slug, "filter": ug_filter or "ALL", "n": n,
        "musx_post_hit": sum(r["musx_post_hit"] for r in rows),
        "musx_post_decisive": sum(r["musx_post_decisive"] for r in rows),
        "musx_post_win_hit": sum(r["musx_post_win_hit"] for r in rows),
        "nnls_hit": sum(r["nnls_hit"] for r in rows),
        "nnls_decisive": sum(r["nnls_decisive"] for r in rows),
        "lab_bass_hit": sum(r["lab_bass_hit"] for r in rows),
    }
    if verbose:
        print(f"\n=== {slug}   MISSED where ug={summ['filter']}   n={n} ===")
        hdr = ("  when      dur ug    musx.lab      | musx-post   marg  hit "
               "dec | win | nnls      marg hit dec")
        print(hdr)
        for r in rows:
            print(f"  {r['t0']:6.1f}   {r['dur']:.2f} {r['ug']:5s} "
                  f"{str(r['musx_lab']):12s} | "
                  f"{PC[r['musx_post_pc']]:2s} {r['musx_post_mass']:.3f} "
                  f"{r['musx_post_margin']:5.2f} "
                  f"{'Y' if r['musx_post_hit'] else '.':>3s} "
                  f"{'Y' if r['musx_post_decisive'] else '.':>3s} | "
                  f"{'Y' if r['musx_post_win_hit'] else '.':>3s} | "
                  f"{PC[r['nnls_pc']]:2s} {r['nnls_margin']:8.2f} "
                  f"{'Y' if r['nnls_hit'] else '.':>3s} "
                  f"{'Y' if r['nnls_decisive'] else '.':>3s}")
        if n:
            print(f"\n  CLASS   n   what it means")
            print(f"  PRESENT     {kl['PRESENT']:3d}  musx already decoded this "
                  f"root -- lost downstream, no bass work can help")
            print(f"  BASS        {kl['BASS']:3d}  a bass source names the UG "
                  f"root decisively where musx's label does not")
            print(f"  BASS-WEAK   {kl['BASS-WEAK']:3d}  right pitch class, "
                  f"margin under the {BASS_MARGIN}x guard")
            print(f"  NO-EVIDENCE {kl['NO-EVIDENCE']:3d}  no bass source names "
                  f"it -- premise cannot reach these")
            for k in ("BASS", "BASS-WEAK"):
                sh = Counter(f"{r['ug']} in {r['musx_lab']}"
                             for r in rows if r["klass"] == k)
                if sh:
                    print(f"    {k}: " + ", ".join(f"{s} x{c}"
                                                   for s, c in sh.most_common()))
            print()
            print(f"  -- musx bass posterior: {summ['musx_post_hit']}/{n} hit, "
                  f"{summ['musx_post_decisive']}/{n} decisive (>= {BASS_MARGIN}x), "
                  f"{summ['musx_post_win_hit']}/{n} hit within +-{WINDOW}s")
            print(f"  -- NNLS bass argmax:    {summ['nnls_hit']}/{n} hit, "
                  f"{summ['nnls_decisive']}/{n} decisive")
            print(f"  -- musx .lab bass:      {summ['lab_bass_hit']}/{n} hit")
    return {"summary": summ, "rows": rows}


def false_positive_check(slug: str, sub_pc: int, host_root: int,
                         host_kind: str = "maj", guard_spans=()) -> dict:
    """How often does each bass source read ``sub_pc`` on host spans that are
    RIGHT?  The recovery only pays if the evidence is SPECIFIC — a source that
    shouts D all over the F chords is worthless however well it scores on the 13
    spots.

    Walks every music-x-lab segment labelled ``host``, splits it into 0.5 s
    windows, and counts windows where the source's argmax is ``sub_pc`` with
    margin >= BASS_MARGIN.  ``guard_spans`` (t0, t1) are excluded: on Let It Be
    the tab writes the passing D-7 in the verses but NOT in the solo/chorus,
    while the piano plays the same F-E-D-C fill throughout, so windows near a
    genuine fill are not honest negatives.  Both figures are reported —
    ``fired`` over all host windows and ``fired_guarded`` over the guarded
    subset — because which one is "the" false-positive rate depends on whether
    you trust the tab's omissions, and the tab is inconsistent here.
    """
    probs = load_musx_probs(slug)
    pbass = probs[1]
    labels = load_musx_lab(slug)
    nb, nt = load_nnls(slug)
    out = {"host": f"{PC[host_root]}:{host_kind}", "sub": PC[sub_pc],
           "windows": 0, "windows_guarded": 0,
           "musx_fired": 0, "musx_fired_guarded": 0,
           "nnls_fired": 0, "nnls_fired_guarded": 0}
    for t0, t1, lab in labels:
        r = _parse_root(lab.split("/")[0].split(":")[0]) if lab not in ("N", "X") else None
        q = lab.split(":")[1].split("/")[0] if ":" in lab else None
        if r != host_root or (q or "maj") != host_kind:
            continue
        t = t0
        while t + 0.5 <= t1:
            guarded = all(not (a - 1.0 <= t <= b + 1.0) for a, b in guard_spans)
            out["windows"] += 1
            out["windows_guarded"] += int(guarded)
            mt, mg, _, _ = musx_bass_read(pbass, t, t + 0.5)
            if mt == sub_pc and mg >= BASS_MARGIN:
                out["musx_fired"] += 1
                out["musx_fired_guarded"] += int(guarded)
            nt_, ng = nnls_bass_read(nb, nt, t, t + 0.5)
            if nt_ == sub_pc and ng >= BASS_MARGIN:
                out["nnls_fired"] += 1
                out["nnls_fired_guarded"] += int(guarded)
            t += 0.5
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="let_it_be_remastered_2009")
    ap.add_argument("--ug", default=None, help="only MISSED with this UG name")
    ap.add_argument("--fp", nargs=3, metavar=("SUBPC", "HOSTPC", "KIND"),
                    default=None, help="false-positive check on host spans")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    res = run(a.slug, a.ug)
    if a.fp:
        sub, host = int(a.fp[0]), int(a.fp[1])
        sc = load_score(a.slug)
        # every UG chord on the substitute root, MISSED or not — the tab's own
        # occurrences of the fill are not negatives
        guard = [(float(e["t0"]), float(e["t1"])) for e in sc["ug_seq"]
                 if e.get("root") == sub]
        fp = false_positive_check(a.slug, sub, host, a.fp[2], guard)
        print(f"\n  FALSE-POSITIVE on {fp['host']} spans "
              f"(0.5s windows, argmax=={PC[sub]} and margin>={BASS_MARGIN}):")
        print(f"    all host windows      n={fp['windows']:4d}   "
              f"musx {fp['musx_fired']:3d} "
              f"({100 * fp['musx_fired'] / max(fp['windows'], 1):4.1f}%)   "
              f"nnls {fp['nnls_fired']:3d} "
              f"({100 * fp['nnls_fired'] / max(fp['windows'], 1):4.1f}%)")
        print(f"    guarded (>=1s from any UG {PC[sub]}) n="
              f"{fp['windows_guarded']:4d}   "
              f"musx {fp['musx_fired_guarded']:3d} "
              f"({100 * fp['musx_fired_guarded'] / max(fp['windows_guarded'], 1):4.1f}%)   "
              f"nnls {fp['nnls_fired_guarded']:3d} "
              f"({100 * fp['nnls_fired_guarded'] / max(fp['windows_guarded'], 1):4.1f}%)")
        res["fp"] = fp
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
