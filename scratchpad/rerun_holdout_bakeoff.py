"""Driver: re-run the aligned_corpus vs musx held-out bake-off at CURRENT
corpus scale, multi-seed, honest (CLAUDE.md rule #11 -- every number traces
to a real script run).

For each seed:
  1. train_aligned_corpus_heads.main()-equivalent (song-level split,
     flag_suspect_songs filter applied) -> in-house root/qual accuracy on
     the held-out aligned_corpus songs (fast, no audio re-download).
  2. musx_vs_inhouse_scaled.run_musx_compare() on THE SAME held-out songs
     (slow, one YouTube re-download per song) -> musx root/qual accuracy on
     the identical rows.

This tracks whether the 5-song +6pp in-house-root-beats-musx lead
(docs/known_issues.md, "LOOP: aligned_corpus.npz close-the-loop") holds,
grows, shrinks, or reverses as the corpus grows -- the actual question this
autonomous session is trying to answer.

Usage:
  .venv/bin/python scratchpad/rerun_holdout_bakeoff.py --seeds 0 1 2 --max-musx-songs 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scratchpad"))

import train_aligned_corpus_heads as tach  # noqa: E402
from musx_vs_inhouse_scaled import run_musx_compare  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--max-musx-songs", type=int, default=25,
                     help="cap on held-out songs to re-download+run musx on, per seed (network/time bound)")
    a = ap.parse_args()

    nn_rwc, r_rwc, q_rwc, sid_rwc = tach.load_rwc()
    nn_al, r_al, q_al, sid_al = tach.load_aligned()
    print(f"RWC: {len(nn_rwc)} rows / {len(np.unique(sid_rwc))} songs")
    print(f"aligned_corpus (pre-filter): {len(nn_al)} rows / {len(np.unique(sid_al))} songs")

    suspect = tach.flag_suspect_songs(nn_al, r_al, sid_al)
    print(f"flag_suspect_songs: dropping {len(suspect)}: {sorted(suspect)}")
    keep = ~np.isin(sid_al, list(suspect))
    nn_al, r_al, q_al, sid_al = nn_al[keep], r_al[keep], q_al[keep], sid_al[keep]
    print(f"aligned_corpus (post-filter): {len(nn_al)} rows / {len(np.unique(sid_al))} songs")

    all_summaries = []
    for seed in a.seeds:
        print(f"\n{'#'*70}\n# SEED {seed}\n{'#'*70}")
        tr_mask, te_mask, test_songs = tach.song_level_split(sid_al, seed, a.test_frac)
        test_songs = sorted(test_songs)
        print(f"held-out songs ({len(test_songs)}): {test_songs}")

        nn_al_tr, r_al_tr, q_al_tr = nn_al[tr_mask], r_al[tr_mask], q_al[tr_mask]
        nn_al_te, r_al_te, q_al_te = nn_al[te_mask], r_al[te_mask], q_al[te_mask]

        # A. rwc_only (shipped recipe)
        rm_a, qm_a = tach.train_heads(nn_rwc, r_rwc, q_rwc, seed=seed)
        res_a = tach.eval_heads(rm_a, qm_a, nn_al_te, r_al_te, q_al_te)

        # B. merged
        nn_m = np.concatenate([nn_rwc, nn_al_tr], 0)
        r_m = np.concatenate([r_rwc, r_al_tr], 0)
        q_m = np.concatenate([q_rwc, q_al_tr], 0)
        rm_b, qm_b = tach.train_heads(nn_m, r_m, q_m, seed=seed)
        res_b = tach.eval_heads(rm_b, qm_b, nn_al_te, r_al_te, q_al_te)

        print(f"\n[seed {seed}] in-house rwc_only : root={res_a['root_acc']:.3f} "
              f"qual={res_a['qual_acc']:.3f} fam={res_a['qual_family_acc']:.3f}")
        print(f"[seed {seed}] in-house merged   : root={res_b['root_acc']:.3f} "
              f"qual={res_b['qual_acc']:.3f} fam={res_b['qual_family_acc']:.3f}")

        # musx comparison -- capped subset of held-out songs (network/time bound)
        musx_songs = test_songs[: a.max_musx_songs]
        print(f"\nrunning musx on {len(musx_songs)}/{len(test_songs)} held-out songs...")
        rows, musx_summary = run_musx_compare(
            musx_songs, out_npz=REPO / f"scratchpad/musx_bakeoff_seed{seed}.npz")

        if musx_summary:
            # Restrict in-house eval to the SAME rows musx actually covered
            # (apples-to-apples -- some songs/segments may have failed musx
            # download/labeling and must not silently pad the in-house side).
            covered_titles = {r[0] for r in rows}
            m_sub = np.isin(sid_al[te_mask], list(covered_titles))
            if m_sub.sum() > 0:
                res_a_sub = tach.eval_heads(rm_a, qm_a, nn_al_te[m_sub], r_al_te[m_sub], q_al_te[m_sub])
                res_b_sub = tach.eval_heads(rm_b, qm_b, nn_al_te[m_sub], r_al_te[m_sub], q_al_te[m_sub])
                print(f"\n[seed {seed}] APPLES-TO-APPLES on {m_sub.sum()} rows / {len(covered_titles)} songs:")
                print(f"  musx (production)     : root={musx_summary['root_acc']:.3f} "
                      f"qual={musx_summary['qual_acc']:.3f} fam={musx_summary['qual_family_acc']:.3f}")
                print(f"  in-house rwc_only     : root={res_a_sub['root_acc']:.3f} "
                      f"qual={res_a_sub['qual_acc']:.3f} fam={res_a_sub['qual_family_acc']:.3f}")
                print(f"  in-house merged       : root={res_b_sub['root_acc']:.3f} "
                      f"qual={res_b_sub['qual_acc']:.3f} fam={res_b_sub['qual_family_acc']:.3f}")
                all_summaries.append(dict(
                    seed=seed, n_rows=int(m_sub.sum()), n_songs=len(covered_titles),
                    musx=musx_summary,
                    rwc_only=dict(root_acc=res_a_sub['root_acc'], qual_acc=res_a_sub['qual_acc'],
                                  qual_family_acc=res_a_sub['qual_family_acc']),
                    merged=dict(root_acc=res_b_sub['root_acc'], qual_acc=res_b_sub['qual_acc'],
                                qual_family_acc=res_b_sub['qual_family_acc']),
                ))

    print(f"\n\n{'='*70}\nFINAL SUMMARY across {len(all_summaries)} seeds\n{'='*70}")
    for s in all_summaries:
        print(f"seed={s['seed']} n_rows={s['n_rows']} n_songs={s['n_songs']}")
        print(f"  musx      root={s['musx']['root_acc']:.3f} qual={s['musx']['qual_acc']:.3f} fam={s['musx']['qual_family_acc']:.3f}")
        print(f"  rwc_only  root={s['rwc_only']['root_acc']:.3f} qual={s['rwc_only']['qual_acc']:.3f} fam={s['rwc_only']['qual_family_acc']:.3f}")
        print(f"  merged    root={s['merged']['root_acc']:.3f} qual={s['merged']['qual_acc']:.3f} fam={s['merged']['qual_family_acc']:.3f}")
        lead = s['rwc_only']['root_acc'] - s['musx']['root_acc']
        print(f"  in-house-root LEAD over musx: {lead:+.3f}")

    if all_summaries:
        import json
        (REPO / "scratchpad/bakeoff_final_summary.json").write_text(json.dumps(all_summaries, indent=1))
        print("\nsaved scratchpad/bakeoff_final_summary.json")


if __name__ == "__main__":
    main()
