"""pool_denoise_premise_check.py — 2026-07-18, chord-robustness reframe,
Step 1 (mandatory premise check BEFORE building any UI, per CLAUDE.md rule
#2 and the research-loop skill).

Question: for a pair of harmonically-IDENTICAL bars found by the 1-bar SSM,
does AVERAGING their per-beat evidence (pool_beat_evidence's mechanism) move
the pooled vector CLOSER to the clean underlying chord than either
individual noisy bar was? Or does real audio's noise have a shared/
systematic component (confirmed tonight: the "elevated similarity floor",
see "Step 1 RETRY: floor/multiplicative-blend noise model" in
known_issues.md) that averaging can't remove?

No bar-level ground truth exists for real audio (repeatedly confirmed
tonight), so this uses the ALREADY-VALIDATED floor-blend noise model
(scratchpad/noise_calibrate_floor.py, best_alpha combined=0.40, calibrated
to match real audio's actual off-diagonal similarity floor to within 0.004)
as the realism-checked proxy, applied to CLEAN iReal chord vectors (which DO
have known ground truth — the template itself).

Two noise components, both physically motivated and applied together:
  1. SHARED bias (the calibrated floor-blend, alpha=0.40 toward a fixed
     `generic` vector) — same direction added to every bar, does NOT average
     out by construction (this is the user's own worry, stated explicitly).
  2. INDEPENDENT per-bar jitter (small Gaussian, sigma swept) — represents
     genuine per-instance extraction randomness (onset timing / transient
     contamination / overtone variance across two different real playings of
     "the same" chord) — THIS component should average out like the
     video-frame-stacking intuition predicts, if the intuition holds at all.

For many random clean chords, simulate 2 independently-jittered noisy copies
under the shared bias, then compare:
  - individual: mean cosine(noisy_i, clean) over i=1,2
  - pooled:     cosine(normalize(noisy_1 + noisy_2), clean)
sweeping jitter sigma from 0 (pure shared bias, no random component — the
worst case for pooling) up through the sigma that reproduces real audio's
actual per-bar variance (estimated below from repeated-bar variance in the
3 real-audio songs' own bar_ssm_rawchroma_*.json outputs, so the sigma isn't
just guessed).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from noise_calibrate import load_corpus_registers, N_CORPUS_SAMPLE
from noise_calibrate_floor import corpus_generic_vector, blend_vecs

OUT_DIR = Path(__file__).resolve().parent
RNG = np.random.default_rng(0)
ALPHA = 0.40   # calibrated combined-register best alpha from noise_calibrate_floor


def jitter(v, sigma, rng):
    if sigma <= 0:
        return v
    nv = v + rng.normal(0, sigma, size=v.shape)
    n = np.linalg.norm(nv)
    return nv / n if n > 1e-9 else nv


def estimate_real_audio_jitter_sigma():
    """Cheap real-audio anchor: for each song's finest-grain (size=1, i.e.
    per-bar) SSM, find pairs the model itself scores as near-identical
    (sim > 0.97, i.e. bars the SSM is confident are repeats) and treat
    residual sub-1.0 similarity there as an upper bound on 'random jitter'
    once the shared floor is accounted for — NOT a rigorous estimate (no bar
    GT), just a sanity anchor so the sigma sweep isn't purely arbitrary."""
    sims = []
    for song in ("aretha_chain_of_fools", "autumn_leaves", "abba_chiquitita"):
        p = OUT_DIR / f"bar_ssm_rawchroma_{song}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        grains = d.get("grains", {})
        if "1" not in grains:
            continue
        sim = np.array(grains["1"]["similarity_matrix"])
        n = sim.shape[0]
        for i in range(n):
            for j in range(i + 2, n):  # exclude trivial neighbors
                if sim[i, j] > 0.97:
                    sims.append(sim[i, j])
    if not sims:
        return None
    return float(np.mean(sims)), len(sims)


def main():
    print("=== Real-audio anchor for jitter sigma (near-identical bar pairs) ===")
    anchor = estimate_real_audio_jitter_sigma()
    if anchor:
        print(f"  {anchor[1]} pairs with sim>0.97 at grain=1, mean sim={anchor[0]:.4f}")
    else:
        print("  no bar_ssm_rawchroma_*.json size=1 grain found — skipping anchor")

    print("\n=== Loading iReal corpus + calibrated generic vector ===")
    corpus = load_corpus_registers(max_tunes=N_CORPUS_SAMPLE)
    generic = corpus_generic_vector(corpus, "treble")  # treble = fuller chord-tone signal
    print(f"  corpus: {len(corpus)} tunes, generic vector = "
          f"{np.round(generic, 3).tolist()}")

    # Flatten all clean per-bar treble vectors as candidate "clean chords"
    clean_vecs = [v for c in corpus for v in c["treble"] if np.linalg.norm(v) > 1e-6]
    print(f"  {len(clean_vecs)} candidate clean chord vectors")

    sigmas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    n_trials = 2000
    idxs = RNG.integers(0, len(clean_vecs), size=n_trials)

    print(f"\n=== Sweep: shared bias alpha={ALPHA} (fixed, calibrated) "
          f"x independent jitter sigma ===")
    print(f"{'sigma':>6} {'mean_individual_sim':>20} {'mean_pooled_sim':>16} "
          f"{'delta':>8} {'pooled_wins_%':>14}")
    results = []
    for sigma in sigmas:
        indiv_sims, pooled_sims, wins = [], [], 0
        for idx in idxs:
            clean = clean_vecs[idx]
            clean_n = clean / np.linalg.norm(clean)
            b1 = blend_vecs([clean], ALPHA, generic)[0]
            b2 = blend_vecs([clean], ALPHA, generic)[0]  # same shared bias
            n1 = jitter(b1, sigma, RNG)
            n2 = jitter(b2, sigma, RNG)
            s1 = float(np.dot(n1, clean_n))
            s2 = float(np.dot(n2, clean_n))
            pooled = n1 + n2
            pn = np.linalg.norm(pooled)
            pooled = pooled / pn if pn > 1e-9 else pooled
            sp = float(np.dot(pooled, clean_n))
            indiv_sims.append((s1 + s2) / 2)
            pooled_sims.append(sp)
            if sp > max(s1, s2):
                wins += 1
        mi, mp = float(np.mean(indiv_sims)), float(np.mean(pooled_sims))
        win_pct = 100.0 * wins / n_trials
        results.append({"sigma": sigma, "mean_individual": mi, "mean_pooled": mp,
                         "delta": mp - mi, "pooled_beats_best_individual_pct": win_pct})
        print(f"{sigma:6.2f} {mi:20.4f} {mp:16.4f} {mp - mi:8.4f} {win_pct:14.1f}")

    out = {"alpha": ALPHA, "n_trials": n_trials, "real_audio_anchor": anchor,
           "sweep": results}
    (OUT_DIR / "pool_denoise_premise_check_results.json").write_text(
        json.dumps(out, indent=2))
    print("\nwrote pool_denoise_premise_check_results.json")

    print("\n=== Verdict ===")
    zero = results[0]
    if zero["delta"] <= 0.001:
        print(f"  At sigma=0 (PURE shared bias, no random component): delta="
              f"{zero['delta']:.4f} — pooling gives ~ZERO benefit when noise is "
              f"purely the systematic floor bias. Matches the user's stated worry: "
              f"a shared bias does not cancel under averaging.")
    nonzero = [r for r in results if r["sigma"] > 0]
    if nonzero and all(r["delta"] > 0.005 for r in nonzero):
        print(f"  For sigma>0 (adding a random per-bar component): delta ranges "
              f"{nonzero[0]['delta']:.4f} (sigma={nonzero[0]['sigma']}) to "
              f"{nonzero[-1]['delta']:.4f} (sigma={nonzero[-1]['sigma']}) — pooling "
              f"DOES help once there's a genuine random component to average out, "
              f"even with the shared bias present underneath.")


if __name__ == "__main__":
    main()
