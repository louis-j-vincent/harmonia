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

# NNLS bothchroma index 0 is A; roll to a C-first frame.
_ROLL_TO_C = -9


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

    n = len(rows)
    summ = {
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
            print(f"  -- musx bass posterior: {summ['musx_post_hit']}/{n} hit, "
                  f"{summ['musx_post_decisive']}/{n} decisive (>= {BASS_MARGIN}x), "
                  f"{summ['musx_post_win_hit']}/{n} hit within +-{WINDOW}s")
            print(f"  -- NNLS bass argmax:    {summ['nnls_hit']}/{n} hit, "
                  f"{summ['nnls_decisive']}/{n} decisive")
            print(f"  -- musx .lab bass:      {summ['lab_bass_hit']}/{n} hit")
    return {"summary": summ, "rows": rows}


def false_positive_check(slug: str, sub_pc: int, host_root: int,
                         host_kind: str = "maj") -> dict:
    """How often does the musx bass posterior read ``sub_pc`` on host spans that
    are RIGHT?  The recovery only pays if the evidence is specific.

    Walks every music-x-lab segment whose label is ``host``, splits it into
    beat-ish 0.5 s windows, and counts the windows where the bass posterior's
    argmax is ``sub_pc`` with margin >= BASS_MARGIN.  These are the spans a
    naive "bass != root => relabel" rule would corrupt.
    """
    probs = load_musx_probs(slug)
    pbass = probs[1]
    labels = load_musx_lab(slug)
    tot = fired = 0
    for t0, t1, lab in labels:
        r = _parse_root(lab.split("/")[0].split(":")[0]) if lab not in ("N", "X") else None
        q = lab.split(":")[1].split("/")[0] if ":" in lab else None
        if r != host_root or (q or "maj") != host_kind:
            continue
        t = t0
        while t + 0.5 <= t1:
            tot += 1
            top, marg, _, _ = musx_bass_read(pbass, t, t + 0.5)
            if top == sub_pc and marg >= BASS_MARGIN:
                fired += 1
            t += 0.5
    return {"host": f"{PC[host_root]}:{host_kind}", "windows": tot,
            "fired": fired, "rate": (fired / tot) if tot else 0.0}


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
        fp = false_positive_check(a.slug, int(a.fp[0]), int(a.fp[1]), a.fp[2])
        print(f"\n  FALSE-POSITIVE on {fp['host']} spans: bass reads "
              f"{PC[int(a.fp[0])]} decisively in {fp['fired']}/{fp['windows']} "
              f"0.5s windows ({100 * fp['rate']:.1f}%)")
        res["fp"] = fp
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
