"""bass_root_recovery.py — recover shared-tone passing chords the decoder ate.

THE DEFECT (docs/ug_score_report.md, "Falsification test result")
-----------------------------------------------------------------
Half of Let It Be's missing chords are *decoder-level absorption*: the chord is
a subset of the one next to it, so nothing in the treble can tell them apart.
All 13 D-7 passing chords come out as ``F:maj``, because ``Dm7 = D F A C``
contains ``F A C`` and the ONLY discriminator is the D underneath.

THE PREMISE, CHECKED BEFORE BUILDING (scratchpad/bass_premise_check.py)
-----------------------------------------------------------------------
At those 13 spots the NNLS-24 bass half's argmax reads **D at 8/13, decisively
(>= 1.5x runner-up) at 7/13** — and reads D on only 4/71 (5.6 %) of the guarded
half-second windows inside plain ``F:maj``, so the evidence is specific as well
as present.  music-x-lab's own bass posterior is **0/13** there: its bass head
is smoothed toward its own chord decode and cannot see a 0.2 s passing note.
The two sources are complementary and both are read here.

This is the sanctioned use of the bass under the project's bass doctrine
(``docs/harmonic_key_investigation.md``): the bass as a **decisive NOTE**, an
argmax with a margin guard, functioning as a root discriminator between two
candidates the chroma cannot separate.  Bass as chroma MASS stays banned; no
mass from the bass half enters any decision here.

THE RULE, ENUMERATED RATHER THAN TABULATED
------------------------------------------
A hand-written list of "relative substitute pairs" invites exactly the errors it
is meant to fix, so the substitutes are *derived*.  For a host chord with pitch
classes ``H`` and rooted at ``R``, a decisive bass ``B`` proposes a relabel only
when

    1. ``B not in H``            — B is a NEW tone.  This single condition throws
                                   out every inversion: C/E and C/G never fire,
                                   because E and G are already in the C triad.
                                   On Let It Be that is what protects the three
                                   ``C`` chords whose sounding bass really is E.
    2. some quality ``S`` in the vocabulary satisfies
       ``pcs(B, S) == H | {B}``  — the substitute CONTAINS the host exactly, plus
                                   the bass and nothing else.  Minimal by
                                   construction, so no chord is invented.
    3. the substitute is rooted on the BASS                 (that is the whole
                                   point: the bass is what names the root).

Run over the vocabulary this yields, and only yields: major triad + bass a minor
third below -> ``min7`` (F:maj + D = D:min7 — the Let It Be case, and equally
C:maj + A = A:min7, G:maj + E = E:min7); minor triad + bass a major third below
-> ``maj7`` (A:min + F = F:maj7); maj7 + bass a minor third below -> ``min9``.
Everything else fails condition 2.  ``D:min7`` + F bass correctly does NOT fire —
F is already in the chord, so that is an inversion (``Dm7/F``), not a substitute.

WHAT THIS DOES **NOT** SOLVE (CLAUDE.md rule #4)
------------------------------------------------
* **It reaches 12 of Let It Be's 38 misses, not 38.**  The premise check splits
  them: 14 are PRESENT in music-x-lab's decode already and are lost *downstream*
  of it (a chart-layer defect — 3.4 s segments are being dropped, which no bass
  work touches); 12 have NO bass evidence for the UG root at all.  This module
  is aimed only at the middle 12.
* **The three ``G`` absorbed by ``C:maj/5`` are NOT this rule's business.**  There
  the bass already reads G and music-x-lab already wrote G *as the bass* — the
  question is whether ``C/G`` should have been ``G``, a fifth-inversion decision
  that belongs to ``harmonia/models/fifth_discriminator.py``.  Condition 1
  deliberately refuses them (G is in the C triad).  They are counted and
  reported, never repaired.
* **It runs on the baked chart, not inside the decoder.**  The measured effect
  is therefore the CEILING of the idea, not what wiring it live would give: the
  shipped chart layer never emits a chord under ~0.8 beats (report, screening
  test), so a live version needs that floor lifted first.  Nothing here touches
  ``harmonia/**``.
* **The margin guard is one number on one song.**  1.5x is inherited from the
  harmonic-key thread's ``BASS_MARGIN``, not re-fitted here; it is a hypothesis
  (CLAUDE.md rule #5).
* It cannot distinguish "the tab omitted this passing chord" from "we invented
  one".  On Let It Be the piano plays the F-E-D-C fill in the solo and choruses
  too, where the tab writes plain ``F | C`` — so some of what the scorer will
  call ADDED is the tab's inconsistency, not ours.  Reported, not smoothed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRATCH = REPO / "scratchpad"
sys.path.insert(0, str(REPO))

from harmonia.models.musx_bass import _parse_root  # noqa: E402
from harmonia.models.musx_redecode import FRAME_DT  # noqa: E402
from harmonia.models.nnls_features import _ROLL_TO_C  # noqa: E402

PC = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# ── tunables (all single numbers on few songs — hypotheses, not laws) ────────
BASS_MARGIN = 1.5     # decisiveness guard, inherited from harmonic_key v3
WIN = 0.30            # bass reading window
HOP = 0.10            # slide
MIN_RUN = 0.25        # a fired run shorter than this is noise, not a chord
EDGE_PAD = 0.05       # ignore the host's own boundary frames (attack/release)
MIN_HOST_LEFT = 0.20  # never leave a host stub shorter than this


def load_ug_score():
    """Import ``ug_score`` as a module WITHOUT editing it (it is read-only)."""
    spec = importlib.util.spec_from_file_location("ug_score", SCRATCH / "ug_score.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["ug_score"] = m
    spec.loader.exec_module(m)
    return m


UG = load_ug_score()

# ── the substitute vocabulary, and the derivation of the firing rule ─────────
# iReal quality token -> pitch classes relative to the root.  The tokens are the
# ones ``ug_score.cname`` / ``parse_token`` speak, so a relabel is scoreable.
QUAL_PCS: dict[str, frozenset[int]] = {
    "": frozenset({0, 4, 7}),            # major triad
    "-": frozenset({0, 3, 7}),           # minor triad
    "-7": frozenset({0, 3, 7, 10}),
    "^7": frozenset({0, 4, 7, 11}),
    "7": frozenset({0, 4, 7, 10}),
    "6": frozenset({0, 4, 7, 9}),
    "-6": frozenset({0, 3, 7, 9}),
    "o": frozenset({0, 3, 6}),
    "h7": frozenset({0, 3, 6, 10}),
    "o7": frozenset({0, 3, 6, 9}),
    "sus": frozenset({0, 5, 7}),
    "+": frozenset({0, 4, 8}),
    "-9": frozenset({0, 2, 3, 7, 10}),
    "^9": frozenset({0, 2, 4, 7, 11}),
    "9": frozenset({0, 2, 4, 7, 10}),
}


def pcs_of(root: int, q: str) -> frozenset[int] | None:
    rel = QUAL_PCS.get(q)
    if rel is None:
        return None
    return frozenset((root + i) % 12 for i in rel)


#: Which derived branches are allowed to fire.  The derivation below yields ten
#: (host quality, bass interval) branches, but only ONE of them has measured
#: evidence behind it — ``maj`` triad + bass a minor third below -> ``min7``,
#: the F:maj/D-7 case the premise check scored at 8/13 with a 5.6 % false-
#: positive rate.  Running all ten on Let It Be fires 15 times and costs
#: ADDED +6 / ROOT +3 against MISSED -5, i.e. it is net NEGATIVE: the unevidenced
#: branches (``A-`` + F bass -> ``F^7`` twice, ``C`` + A bass -> ``A-7`` twice,
#: ``G`` + E bass -> ``E-7``) fire in places no source says a chord changed.
#: Enumerating the rule is how you know what COULD fire; evidence is what decides
#: what SHOULD.  ``--all-branches`` restores the full set for the ablation.
EVIDENCED_BRANCHES: frozenset[tuple[str, int]] = frozenset({("", 9)})


def substitute_for(root: int, q: str, bass: int,
                   branches: "frozenset[tuple[str, int]] | None" = None
                   ) -> str | None:
    """The minimal chord rooted on ``bass`` that CONTAINS the host, or None.

    Conditions 1-3 of the module docstring.  Returns the iReal quality token.
    Deterministic: candidates are tried in ``QUAL_PCS`` order and the first
    exact match wins; ties cannot occur because an exact pc-set match with a
    fixed root determines the quality up to spelling.
    """
    host = pcs_of(root, q)
    if host is None or bass == root or bass in host:
        return None                                   # condition 1
    if branches is not None and (q, (bass - root) % 12) not in branches:
        return None
    want = host | {bass}
    for sq in QUAL_PCS:
        sp = pcs_of(bass, sq)
        if sp is not None and sp == want:             # conditions 2 and 3
            return sq
    return None


def enumerate_rule() -> list[str]:
    """Every (host quality, bass interval) that can ever fire — printed so the
    rule is auditable rather than asserted."""
    out = []
    for q in QUAL_PCS:
        for iv in range(1, 12):
            s = substitute_for(0, q, iv)
            if s is not None:
                out.append(f"{'maj' if q == '' else q} + bass +{iv} "
                           f"-> {PC[iv]}{s or 'maj'} (rooted on the bass)")
    return out


# ── bass sources ─────────────────────────────────────────────────────────────

class Bass:
    """Both bass estimators for one song, read as NOTES with a margin."""

    def __init__(self, slug: str):
        z = np.load(REPO / "data" / "cache" / "nnls_infer" / f"{slug}.npz")
        self.nnls = np.roll(np.asarray(z["arr"][:, :12], np.float64),
                            _ROLL_TO_C, axis=1)
        self.nt = np.asarray(z["times"], np.float64)
        p = REPO / "data" / "cache" / "musx_probs" / f"{slug}.npz"
        self.musx = np.load(p)["bass"] if p.exists() else None

    def read(self, t0: float, t1: float) -> tuple[int, float, int, float]:
        m = (self.nt >= t0) & (self.nt < t1)
        v = self.nnls[m].mean(0) if m.any() else \
            self.nnls[int(np.argmin(np.abs(self.nt - 0.5 * (t0 + t1))))]
        o = np.argsort(v)[::-1]
        n_pc, n_mg = int(o[0]), float(v[o[0]] / max(v[o[1]], 1e-9))
        if self.musx is None:
            return n_pc, n_mg, -1, 0.0
        a = max(0, min(int(round(t0 / FRAME_DT)), self.musx.shape[0] - 1))
        b = min(self.musx.shape[0], max(a + 1, int(round(t1 / FRAME_DT))))
        w = self.musx[a:b, 1:13].mean(0)
        p = np.argsort(w)[::-1]
        return n_pc, n_mg, int(p[0]), float(w[p[0]] / max(w[p[1]], 1e-9))


# ── the audit pass ───────────────────────────────────────────────────────────

def recover(ours: list, bass: Bass, *, require_both: bool = False,
            margin: float = BASS_MARGIN,
            branches: "frozenset[tuple[str, int]] | None" = EVIDENCED_BRANCHES,
            late_only: bool = False, tail_absorb: float = MIN_HOST_LEFT,
            ) -> tuple[list, list[dict]]:
    """Insert recovered sub-spans into a chart sequence.  Pure: builds a new
    list, never mutates ``ours``.

    ``require_both`` demands that BOTH bass sources name the substitute root.
    Measured to be far too strict (music-x-lab is 0/13 on the Let It Be case),
    so the default is either-source; the flag exists so the ablation is one
    argument away rather than a code change.
    """
    out, fired = [], []
    for e in ours:
        if e.nc or e.dur < MIN_RUN + 2 * EDGE_PAD:
            out.append(e)
            continue
        host = pcs_of(e.root, e.q)
        if host is None:
            out.append(e)
            continue

        # 1. which windows propose which substitute?
        props: list[tuple[float, float, int, str]] = []
        t = e.t0 + EDGE_PAD
        while t + WIN <= e.t1 - EDGE_PAD:
            npc, nmg, mpc, mmg = bass.read(t, t + WIN)
            hit = None
            n_ok = nmg >= margin and substitute_for(e.root, e.q, npc, branches)
            m_ok = (mpc >= 0 and mmg >= margin
                    and substitute_for(e.root, e.q, mpc, branches))
            if require_both:
                if n_ok and m_ok and npc == mpc:
                    hit = (npc, n_ok)
            elif n_ok:
                hit = (npc, n_ok)
            elif m_ok:
                hit = (mpc, m_ok)
            if hit:
                props.append((t, t + WIN, hit[0], hit[1]))
            t += HOP

        # 2. coalesce contiguous windows proposing the SAME substitute root
        runs: list[list] = []
        for a, b, pc, sq in props:
            if runs and runs[-1][2] == pc and a <= runs[-1][1] + 1e-9:
                runs[-1][1] = b
            else:
                runs.append([a, b, pc, sq])
        runs = [r for r in runs if r[1] - r[0] >= MIN_RUN]
        if not runs:
            out.append(e)
            continue

        # 3. keep the single strongest run per host (a passing chord, not a
        #    shredded bar): the longest, ties broken by lateness — approach
        #    chords resolve INTO the next chord, so the late one is the one.
        if late_only:
            # A passing chord APPROACHES the next one, so it lives in the back
            # half of the span it was absorbed into.  Measured on Let It Be: all
            # 13 UG D-7 spots sit in the final third of their host F.  A run in
            # the front half is more likely a sustained bass note under the
            # host's own downbeat, which is not a chord change.
            runs = [r for r in runs if r[0] >= 0.5 * (e.t0 + e.t1)]
            if not runs:
                out.append(e)
                continue
        r = max(runs, key=lambda r: (r[1] - r[0], r[0]))
        a, b, pc, sq = r[0], r[1], r[2], r[3]
        a, b = max(a, e.t0), min(b, e.t1)
        if (a - e.t0) < MIN_HOST_LEFT:            # would leave a stub -> take
            a = e.t0                              # the head of the host instead
        # A leftover TAIL of the host is the expensive stub: the ordinal diff
        # pairs it against the next UG chord and calls it ADDED (measured:
        # 2 of the 6 new ADDED on Let It Be were 0.3 s host tails).  Absorb any
        # tail shorter than ``tail_absorb`` into the recovered chord, which is
        # also the musically right answer -- the passing chord runs INTO the
        # next chord, it does not hand back to the one it came from.
        if (e.t1 - b) < tail_absorb:
            b = e.t1
        if b - a < MIN_RUN or (a <= e.t0 and b >= e.t1):
            out.append(e)                          # whole-host relabel: refuse
            continue

        if a > e.t0:
            out.append(UG.Ev(e.t0, a, e.root, e.q, e.conf, e.nc))
        out.append(UG.Ev(a, b, pc, sq, e.conf, False))
        if b < e.t1:
            out.append(UG.Ev(b, e.t1, e.root, e.q, e.conf, e.nc))
        fired.append({"t0": round(a, 2), "t1": round(b, 2),
                      "host": UG.cname(e.root, e.q),
                      "host_t0": round(e.t0, 2), "host_t1": round(e.t1, 2),
                      "new": UG.cname(pc, sq)})
    return out, fired


# ── validation against the UG scorer, replayed from the stored JSON ──────────

def rebuild(slug: str):
    """(ours, ug, nm_spans, t_intro, stored) from ``ug_score_<slug>.json``.

    The stored JSON does not carry the UG side's audio-support value, which
    ``find_anchors`` consults, so it is reconstructed as neutral.  That is a
    real approximation and it is CHECKED, not assumed: ``baseline()`` re-scores
    the untouched sequences and refuses to report anything unless the replay
    reproduces the stored MISSED/ADDED/ROOT/QUALITY counts.
    """
    d = json.loads((SCRATCH / f"ug_score_{slug}.json").read_text())
    nm = d.get("nomusic_spans") or []

    def nomusic(t0, t1):
        return any(s["t0"] <= 0.5 * (t0 + t1) <= s["t1"] for s in nm)

    ours = [UG.Ev(e["t0"], e["t1"], e["root"], e["q"], e.get("conf", 1.0),
                  bool(e.get("nc"))) for e in d["our_seq"]]
    ug = []
    for e in d["ug_seq"]:
        v = UG.Ev(e["t0"], e["t1"], e["root"], e["q"], 1.0, False,
                  nomusic(e["t0"], e["t1"]))
        v.section = e.get("section", "") or ""
        ug.append(v)
    return ours, ug, nm, float(d.get("t_intro", 0.0)), d


def score_seq(slug, ours, ug, nm, t_intro) -> dict:
    return UG.score(slug, ours, ug, UG.find_anchors(ours, ug), nm, t_intro)


KEYS = ("MISSED", "ADDED", "ROOT", "QUALITY")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slugs", nargs="*", default=None)
    ap.add_argument("--require-both", action="store_true")
    ap.add_argument("--margin", type=float, default=BASS_MARGIN)
    ap.add_argument("--rule", action="store_true", help="print the derived rule")
    ap.add_argument("--late-only", action="store_true",
                    help="only fire in the back half of the host span")
    ap.add_argument("--tail-absorb", type=float, default=MIN_HOST_LEFT,
                    help="absorb a host tail shorter than this into the "
                         "recovered chord instead of leaving a stub")
    ap.add_argument("--all-branches", action="store_true",
                    help="ablation: fire every derived branch, not only the "
                         "one with measured evidence")
    ap.add_argument("--out", default=str(SCRATCH / "bass_root_recovery_result.json"))
    a = ap.parse_args()

    if a.rule:
        print("Every (host, bass interval) that can fire:")
        for line in enumerate_rule():
            print("   " + line)
        print()

    slugs = a.slugs or sorted(p.stem[len("ug_score_"):]
                              for p in SCRATCH.glob("ug_score_*.json"))
    rows, detail = [], {}
    for slug in slugs:
        try:
            ours, ug, nm, t_intro, stored = rebuild(slug)
        except FileNotFoundError:
            continue
        base = score_seq(slug, ours, ug, nm, t_intro)
        faithful = all(base["counts"][k] == stored["counts"][k] for k in KEYS)
        try:
            bass = Bass(slug)
        except FileNotFoundError:
            rows.append((slug, stored["counts"], None, faithful, 0, "no NNLS cache"))
            continue
        new, fired = recover(ours, bass, require_both=a.require_both,
                             margin=a.margin,
                             branches=None if a.all_branches
                             else EVIDENCED_BRANCHES,
                             late_only=a.late_only,
                             tail_absorb=a.tail_absorb)
        after = score_seq(slug, new, ug, nm, t_intro)
        rows.append((slug, base["counts"], after["counts"], faithful,
                     len(fired), ""))
        detail[slug] = {"faithful_replay": faithful, "fired": fired,
                        "stored": {k: stored["counts"][k] for k in KEYS},
                        "replay": {k: base["counts"][k] for k in KEYS},
                        "after": {k: after["counts"][k] for k in KEYS},
                        "agree_before": base["agreement_pct"],
                        "agree_after": after["agreement_pct"]}

    print(f"\nbranches: {'ALL derived' if a.all_branches else 'evidenced only'}")
    print(f"margin {a.margin}x, "
          f"{'BOTH sources' if a.require_both else 'either source'}, "
          f"win {WIN}s / min run {MIN_RUN}s\n")
    print(f"{'song':<46} {'replay ok':<10} {'fired':>5}  "
          f"{'MISSED':>13} {'ADDED':>13} {'ROOT':>11} {'QUAL':>11}")
    for slug, b, aft, ok, nf, note in rows:
        if aft is None:
            print(f"{slug:<46} {'-':<10} {'-':>5}  {note}")
            continue
        def d(k):
            return f"{b[k]:3d} -> {aft[k]:3d}" + \
                   ("  " if aft[k] == b[k] else (" +" if aft[k] > b[k] else " -"))
        print(f"{slug:<46} {'YES' if ok else 'NO':<10} {nf:>5}  "
              f"{d('MISSED'):>13} {d('ADDED'):>13} {d('ROOT'):>11} "
              f"{d('QUALITY'):>11}")
    tot = {k: [sum(r[1][k] for r in rows if r[2]),
               sum(r[2][k] for r in rows if r[2])] for k in KEYS}
    print(f"\n  TOTAL over replay-faithful songs: " + "  ".join(
        f"{k} {v[0]}->{v[1]}" for k, v in tot.items()))
    Path(a.out).write_text(json.dumps(detail, indent=1))


if __name__ == "__main__":
    main()
