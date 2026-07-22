"""harmonia/serving/runtime.py — runtime settings + shared in-memory serving state.

Extracted out of ``scripts/harmonia_server.py`` (serving refactor, runtime
round, following ``cache.py``/``config.py``/``state.py``/``render.py``/
``templates.py``/``api.py``). Behavior-preserving MOVE: same values at request
time, same import-time env reads, same in-place mutation semantics.

Two reassignment/mutation regimes live here, and they are handled differently
so the server always sees the current value:

1. ``ARGS`` (was ``_ARGS`` in the server) is **reassigned at startup**: the
   server's ``main()`` did ``global _ARGS; _ARGS = parse_args()`` AFTER import,
   so anything that captured the name at import time captured the pre-startup
   ``None`` forever. It therefore CANNOT be re-imported into the server as a
   plain name — that binding would freeze at ``None``. Instead every request-
   time read in the server goes through ``runtime.ARGS.<attr>`` (module-
   attribute access, resolved live), and ``main()`` does
   ``runtime.ARGS = parse_args()`` (sets the attribute on THIS module). Reading
   ``runtime.ARGS`` always yields the latest assignment.

2. ``jobs`` / ``jobs_lock`` and ``jam_sessions`` / ``jam_sessions_lock`` (was
   ``_jobs`` / ``_jam_sessions`` + their locks) are **mutated in place only**
   (``d[k] = v``, ``d.get``, ``d.pop``, ``d[k].update``) — never reassigned.
   So the server re-imports each object once (``from ...runtime import jobs as
   _jobs``) and shares the single live instance; a mutation through either
   module's name is visible through the other (``server._jobs is
   runtime.jobs``).

3. The ``_ANALYZE_*`` env-config values are read-only, set once from the
   environment at import, never reassigned. The server re-imports them by
   value; identical value, identical behavior.

Deliberately LEFT in the server this round: ``_billboard_ds`` /
``_billboard_gt_cache`` — a lazily-(re)assigned in-memory GT cache (out of
scope; moving it cleanly would need the same attribute-access treatment as
``ARGS`` for a pure cache with no request-time settings value, so it stays put).
"""

from __future__ import annotations

import argparse
import os
import threading

# ── CLI args, set at startup by the server's main() via ``runtime.ARGS =
# parse_args()``. Read at request time as ``runtime.ARGS.<attr>`` so the live
# value is always seen (see module docstring #1). None until main() runs —
# identical to the server's old ``_ARGS: ... = None`` default.
ARGS: argparse.Namespace | None = None

# ── Acoustic front-end for fresh /api/analyze requests ───────────────────────
# 2026-07-17: production deploy of the NNLS-24 feature front-end + music-x-lab
# routed-bass pipeline (docs/known_issues.md "music-x-lab BASS FRONT-END
# DEPLOYED"). This is the new default for freshly-analysed audio. Fully
# reversible WITHOUT a code change:
#   HARMONIA_ANALYZE_FRONTEND=bp48    → revert to the prior Billboard/BP48 chain
#   HARMONIA_ANALYZE_BASS=nnls24      → keep NNLS-24 features but drop music-x-lab bass
#   HARMONIA_ANALYZE_QUALITY=nnls24   → keep the in-house NNLS-24 root/quality heads
# The analyze route also try/excepts this path and falls back to the exact prior
# Billboard→infer_chords_v1 chain if the NNLS-24/musx pipeline raises, so a
# new-pipeline bug can never hard-break analysis for users.
#
# 2026-07-17 (DEPLOY-3): music-x-lab's OWN root/quality replace the NNLS-24 heads
# by default (FAIR bake-off: +7.3pp root / +13.5pp quality / +13.9pp joint on
# RWC). It reuses the same music-x-lab .lab already loaded for the routed bass, so
# there is no extra inference cost. NNLS-24 stays the bass root-veto only.
_ANALYZE_FEATURE_FRONTEND = os.environ.get("HARMONIA_ANALYZE_FRONTEND", "nnls24")
_ANALYZE_BASS_FRONTEND = os.environ.get("HARMONIA_ANALYZE_BASS", "musx")
_ANALYZE_QUALITY_FRONTEND = os.environ.get("HARMONIA_ANALYZE_QUALITY", "musx")
# Segmentation source (chord-CHANGE timing). "nnls" (default, unchanged): cut at
# every beat where the per-beat NNLS root argmax flips. "musx": use music-x-lab's
# OWN chord-change times snapped to the beat grid (boundary-F1 0.90 vs RWC GT,
# vs the NNLS argmax mechanism's documented over-segmentation). Opt-in via
# HARMONIA_ANALYZE_SEGSOURCE=musx; falls back to NNLS segs if musx is unavailable.
_ANALYZE_SEGMENT_SOURCE = os.environ.get("HARMONIA_ANALYZE_SEGSOURCE", "nnls")
# Beat-grid period (2026-07-19, "BAR-GRID vs REAL-MUSIC DRIFT"): "librosa"
# (default, bit-identical grid) vs "bestfit" (whole-song LSQ period; removes
# the systematic multi-bar drift, madmom-corroborated 11/14 songs — see
# scratchpad/beatgrid_madmom_validate.json). Staged rollout: opt-in only.
# Default flipped to "bestfit" 2026-07-19 with the pipeline default (commit
# cf0d1d1) — the env fallback had stayed "librosa" and silently overrode the
# shipped pipeline default on the analyze route (caught by the barlocked
# section-pass debugging). Rollback: HARMONIA_BEAT_PERIOD_MODE=librosa.
_ANALYZE_BEAT_PERIOD_MODE = os.environ.get("HARMONIA_BEAT_PERIOD_MODE", "bestfit")

# ── In-progress jobs: {job_id: {"status": ..., "url": ..., "out": ...}} ─────
# Mutated in place only; re-imported into the server as ``_jobs`` (same object).
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()

# ── Jam Mode sessions (2026-07-20): {session_id: JamSession} — in-memory,
# single-process (see JamSession's own docstring on thread-safety scope).
# No TTL/eviction yet: a session lives until the process restarts or the
# client explicitly stops it — fine for a personal dev server, would leak
# under real multi-user, long-uptime deployment. Mutated in place only;
# re-imported into the server as ``_jam_sessions`` (same object).
jam_sessions: dict[str, "object"] = {}
jam_sessions_lock = threading.Lock()
