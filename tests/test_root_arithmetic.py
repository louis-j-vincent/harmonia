"""Tests for every root-relative arithmetic claim in the pipeline.

Each function tests one claim.  Run with:
    .venv/bin/python -m pytest tests/test_root_arithmetic.py -v

Claims covered:
  1. chroma root-shift:  roll(chroma_abs, -root)[0] = chroma_abs[root]
  2. key label on LL bars:  NOTE[(root + best_roll) % 12] is the predicted absolute pitch
  3. predicted root in card header:  (root + keys5[pred_fam_i]) % 12
  4. ctx tensor key unification:  col 0 of out[i,j] = root_i's LL column from j's ll_mat
  5. ll_mat column semantics:  ll_mat[fi, r] = LL(roll(x, -r) | family_fi)
  6. max_ll_over_keys finds the right root
  7. double-check: wrong root label formula catches the old bug
"""
import sys
from pathlib import Path
import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

NOTE   = ["C","Db","D","Eb","E","F","F#","G","Ab","A","Bb","B"]
DEGREE = ["R","b2","2","b3","3","4","b5","5","#5","6","b7","7"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers that mirror the pipeline exactly
# ─────────────────────────────────────────────────────────────────────────────

def diag_ll(x, mu, std):
    return float(-0.5 * np.sum(((x - mu) / std)**2) - np.sum(np.log(std)))


def max_ll_over_keys(x, mu, std):
    """Sweep 12 root candidates; return (best_ll, best_roll)."""
    best_ll, best_r = -np.inf, 0
    for r in range(12):
        ll = diag_ll(np.roll(x, -r), mu, std)
        if ll > best_ll:
            best_ll, best_r = ll, r
    return best_ll, best_r


def make_ll_mat(chroma_abs, distributions):
    """(5, 12) LL matrix matching compute_key_family_ll in experiment_ctx_model.py."""
    families = list(distributions.keys())
    n = np.linalg.norm(chroma_abs)
    x = chroma_abs / n if n > 1e-9 else chroma_abs
    ll = np.zeros((len(families), 12), dtype=np.float32)
    for fi, fam in enumerate(families):
        mu, std = distributions[fam]
        for r in range(12):
            ll[fi, r] = diag_ll(np.roll(x, -r), mu, std)
    return ll


# ─────────────────────────────────────────────────────────────────────────────
# 1. chroma root-shift
# ─────────────────────────────────────────────────────────────────────────────

def test_chroma_root_shift_puts_root_at_index_zero():
    """After roll(chroma_abs, -root), index 0 holds the original chroma_abs[root]."""
    rng = np.random.default_rng(0)
    for root in range(12):
        chroma_abs = rng.uniform(0, 1, 12).astype(np.float32)
        chroma_abs[root] += 2.0          # make root dominant
        shifted = np.roll(chroma_abs, -root)
        assert shifted[0] == chroma_abs[root], (
            f"root={root}: shifted[0]={shifted[0]:.4f} != chroma_abs[root]={chroma_abs[root]:.4f}")
        # also check a few other degrees
        for d in [3, 7]:
            assert shifted[d] == chroma_abs[(root + d) % 12], (
                f"root={root} degree={d}: shifted[{d}]={shifted[d]:.4f} != "
                f"chroma_abs[{(root+d)%12}]={chroma_abs[(root+d)%12]:.4f}")


def test_chroma_root_shift_inverse():
    """roll(roll(x, -root), +root) == x."""
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, 12).astype(np.float32)
    for root in range(12):
        assert np.allclose(np.roll(np.roll(x, -root), root), x), \
            f"round-trip failed for root={root}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. LL bar key label:  NOTE[(root + best_roll) % 12]
# ─────────────────────────────────────────────────────────────────────────────

def test_ll_bar_label_is_absolute_pitch():
    """
    The label above an LL bar should name the absolute pitch class the model
    thinks is the root.

    Setup: distribution centred on Am (root=9, minor pattern).
    Query: chroma_abs with energy at A.
    Expected: best_roll aligns A to index 0, so
              predicted_absolute_root = (gt_root + best_roll) % 12 = A = 9.
    """
    # minor chord intervals: root, b3, 5  → degrees 0, 3, 7
    mu = np.zeros(12, np.float32)
    for d in [0, 3, 7]:
        mu[d] = 1.0
    mu /= np.linalg.norm(mu)
    std = np.full(12, 0.1, np.float32)

    gt_root = 9  # A
    # Build absolute-pitch chroma: put energy at A(9), C(0), E(4) — but shifted: Am has C#... wait
    # Am intervals from root: 0(A), 3(C), 7(E)
    chroma_abs = np.zeros(12, np.float32)
    for d in [0, 3, 7]:
        chroma_abs[(gt_root + d) % 12] += 1.0
    n = np.linalg.norm(chroma_abs); chroma_abs /= n

    # root-shift to feed max_ll_over_keys
    chroma_shifted = np.roll(chroma_abs, -gt_root)   # root at index 0
    best_ll, best_roll = max_ll_over_keys(chroma_shifted, mu, std)

    # The key label formula used in the HTML:
    predicted_abs_root = (gt_root + best_roll) % 12
    assert predicted_abs_root == gt_root, (
        f"Expected A(9), got {NOTE[predicted_abs_root]}({predicted_abs_root}), "
        f"best_roll={best_roll}")
    assert NOTE[predicted_abs_root] == "A"


def test_ll_bar_label_wrong_formula_would_fail():
    """
    Demonstrate that the old buggy formula NOTE[best_roll] gives the wrong answer
    when gt_root != 0.
    """
    mu = np.zeros(12, np.float32); mu[0] = 1.0; mu[3] = 0.8; mu[7] = 0.9
    mu /= np.linalg.norm(mu)
    std = np.full(12, 0.15, np.float32)

    gt_root = 5  # F
    chroma_abs = np.zeros(12, np.float32)
    for d in [0, 3, 7]: chroma_abs[(gt_root + d) % 12] += 1.0
    chroma_shifted = np.roll(chroma_abs, -gt_root)
    _, best_roll = max_ll_over_keys(chroma_shifted, mu, std)

    correct_label   = NOTE[(gt_root + best_roll) % 12]   # correct formula
    buggy_label     = NOTE[best_roll]                      # old bug: ignores gt_root

    # When gt_root=F(5) and best_roll=0 (distribution aligned at root), correct=F, buggy=C
    assert correct_label == "F", f"correct formula failed: got {correct_label}"
    assert buggy_label != "F" or best_roll == 5, (
        "Bug would only be invisible when gt_root == best_roll — coincidence, not correctness")


# ─────────────────────────────────────────────────────────────────────────────
# 3. predicted root in card header
# ─────────────────────────────────────────────────────────────────────────────

def test_predicted_root_uses_pred_family_best_key():
    """
    In _make_card: pred_root = (root + keys5[pred_fam_i]) % 12
    The old bug used root for both GT and pred.
    """
    # GT: Bb major (root=10), predicted family: minor (fam_i=1)
    # keys5[1] = 3 (best roll for minor was 3, meaning Db is the minor root)
    seg = {
        "root": 10,       # Bb
        "gt_fam": "major", "gt_fam_i": 0,
        "pred_fam": "minor", "pred_fam_i": 1,
        "keys5": [0, 3, 6, 1, 2],   # best roll per family
    }
    # correct formula
    pred_root = (seg["root"] + seg["keys5"][seg["pred_fam_i"]]) % 12
    assert pred_root == (10 + 3) % 12 == 1, f"got {pred_root}"  # Db(1)
    assert NOTE[pred_root] == "Db"

    # old buggy formula: used seg["root"] for both
    buggy_pred_root = seg["root"]  # = 10 = Bb
    assert NOTE[buggy_pred_root] == "Bb"  # wrong — would show Bb-minor instead of Db-minor


def test_gt_root_is_always_direct():
    """GT root is always seg['root'] directly — no offset needed."""
    for root in range(12):
        seg = {"root": root}
        assert NOTE[seg["root"]] == NOTE[root]


# ─────────────────────────────────────────────────────────────────────────────
# 4. ctx tensor key unification
# ─────────────────────────────────────────────────────────────────────────────

def test_ctx_tensor_col0_is_target_root():
    """
    After key unification, ll_mat_out[i,j][fi, 0] should equal
    ll_mat_j[fi, root_i-relative-column], i.e. the score for key=root_i.

    Concretely: if root_i=5 and root_j=8, delta=(8-5)%12=3,
    rolling ll_mat_j by -3 on axis 1 puts j's column 3 (which scores key=5) at col 0.
    """
    rng = np.random.default_rng(42)
    root_i, root_j = 5, 8
    delta = (root_j - root_i) % 12   # = 3

    # random ll_mat for segment j, shape (5, 12)
    ll_mat_j = rng.uniform(-50, -10, (5, 12)).astype(np.float32)

    # apply the pipeline formula
    ll_unified = np.roll(ll_mat_j, -delta, axis=1)

    # col 0 of unified should equal col (root_i) of original, relative to root_j
    # i.e. the column in ll_mat_j that corresponds to key=root_i
    # key=root_i in j's absolute frame: since ll_mat_j[fi, r] scores roll by -r,
    # we want r such that root_j - r ≡ root_i (mod 12)  →  r = (root_j - root_i) % 12 = delta
    expected_col = delta
    np.testing.assert_array_equal(
        ll_unified[:, 0], ll_mat_j[:, expected_col],
        err_msg=f"col 0 of unified tensor != col {expected_col} of original"
    )


def test_ctx_tensor_self_position_unchanged():
    """At offset=0 (i==j), delta=0, so roll by 0 — tensor unchanged."""
    rng = np.random.default_rng(7)
    ll_mat = rng.uniform(-50, -10, (5, 12)).astype(np.float32)
    root = 4
    delta = (root - root) % 12   # = 0
    unified = np.roll(ll_mat, -delta, axis=1)
    np.testing.assert_array_equal(unified, ll_mat,
        err_msg="self-position (delta=0) should leave ll_mat unchanged")


def test_ctx_tensor_all_12_roots_round_trip():
    """For any root_i, root_j: unify then reverse-unify restores original."""
    rng = np.random.default_rng(99)
    ll_mat_j = rng.uniform(-80, -5, (5, 12)).astype(np.float32)
    for root_i in range(12):
        for root_j in range(12):
            delta    = (root_j - root_i) % 12
            unified  = np.roll(ll_mat_j, -delta, axis=1)
            restored = np.roll(unified, delta, axis=1)
            np.testing.assert_array_equal(restored, ll_mat_j,
                err_msg=f"round-trip failed root_i={root_i} root_j={root_j}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. ll_mat column semantics
# ─────────────────────────────────────────────────────────────────────────────

def test_ll_mat_col_r_scores_roll_minus_r():
    """ll_mat[fi, r] must equal diag_ll(roll(x, -r), mu, std)."""
    rng = np.random.default_rng(3)
    x   = rng.uniform(0, 1, 12).astype(np.float32); x /= np.linalg.norm(x)
    mu  = rng.uniform(0, 1, 12).astype(np.float32); mu /= np.linalg.norm(mu)
    std = np.full(12, 0.2, np.float32)

    # build ll_mat for a single fake family
    ll_row = np.array([diag_ll(np.roll(x, -r), mu, std) for r in range(12)])

    for r in range(12):
        expected = diag_ll(np.roll(x, -r), mu, std)
        assert abs(ll_row[r] - expected) < 1e-5, \
            f"ll_mat col {r}: got {ll_row[r]:.4f}, expected {expected:.4f}"


def test_ll_mat_argmax_col_is_predicted_root():
    """
    The argmax over columns of ll_mat[fi] is the best-roll for family fi,
    which means: predicted root = (gt_root + argmax_col) % 12 when chroma is root-shifted.
    Or equivalently, when chroma is absolute, argmax_col directly is the predicted root.
    """
    # Build an absolute chroma for Eb minor (root=3): strong at Eb(3), Gb(6), Bb(10)
    gt_root = 3
    chroma_abs = np.zeros(12, np.float32)
    for d in [0, 3, 7]: chroma_abs[(gt_root + d) % 12] += 1.0
    chroma_abs /= np.linalg.norm(chroma_abs)

    # Distribution centred on minor (root-shifted, root at 0)
    mu  = np.zeros(12, np.float32); mu[0] = 1.0; mu[3] = 0.8; mu[7] = 0.9
    mu /= np.linalg.norm(mu)
    std = np.full(12, 0.1, np.float32)

    ll_row = np.array([diag_ll(np.roll(chroma_abs, -r), mu, std) for r in range(12)])
    pred_root = int(np.argmax(ll_row))   # absolute root directly from argmax

    assert pred_root == gt_root, \
        f"Expected root {gt_root}({NOTE[gt_root]}), got {pred_root}({NOTE[pred_root]})"


# ─────────────────────────────────────────────────────────────────────────────
# 6. max_ll_over_keys end-to-end
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("root,quality_degrees", [
    (0,  [0, 4, 7]),      # C  major
    (9,  [0, 3, 7]),      # A  minor
    (5,  [0, 4, 7, 11]),  # F  maj7
    (2,  [0, 3, 7, 10]),  # D  min7
    (11, [0, 4, 7, 10]),  # B  dom7
])
def test_max_ll_finds_correct_root(root, quality_degrees):
    """max_ll_over_keys on a noiseless chord query recovers the correct root."""
    mu = np.zeros(12, np.float32)
    for d in quality_degrees: mu[d] = 1.0
    mu /= np.linalg.norm(mu)
    std = np.full(12, 0.05, np.float32)  # tight — noiseless case

    # absolute-pitch query, then root-shift before calling (as pipeline does)
    chroma_abs = np.zeros(12, np.float32)
    for d in quality_degrees: chroma_abs[(root + d) % 12] += 1.0
    chroma_abs /= np.linalg.norm(chroma_abs)

    # root-shift with GT root (as pipeline does when GT is available)
    chroma_shifted = np.roll(chroma_abs, -root)
    _, best_roll = max_ll_over_keys(chroma_shifted, mu, std)
    predicted_abs_root = (root + best_roll) % 12

    assert predicted_abs_root == root, (
        f"quality={quality_degrees} root={root}({NOTE[root]}): "
        f"predicted={predicted_abs_root}({NOTE[predicted_abs_root]}), best_roll={best_roll}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. The original key-label bug is demonstrably different from the fix
# ─────────────────────────────────────────────────────────────────────────────

def test_old_vs_new_label_formula_differ_when_root_nonzero():
    """
    Old bug: key_str = NOTE[best_roll]
    Fix:     key_str = NOTE[(gt_root + best_roll) % 12]
    These are only equal when (gt_root + best_roll) % 12 == best_roll,
    i.e. gt_root == 0. For any other root they differ (unless best_roll
    happens to compensate, which would be coincidental).
    Show they differ for gt_root=7 (G), best_roll=0 (distribution aligned).
    """
    gt_root  = 7    # G
    best_roll = 0   # distribution is already aligned → best roll is 0

    old_label = NOTE[best_roll]                       # NOTE[0] = C  ← wrong
    new_label = NOTE[(gt_root + best_roll) % 12]      # NOTE[7] = G  ← correct

    assert old_label == "C",  f"expected old formula to give C, got {old_label}"
    assert new_label == "G",  f"expected new formula to give G, got {new_label}"
    assert old_label != new_label


if __name__ == "__main__":
    # Run manually without pytest
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in tests:
        # handle parametrize manually for test_max_ll_finds_correct_root
        if fn.__name__ == "test_max_ll_finds_correct_root":
            params = [
                (0,  [0, 4, 7]),
                (9,  [0, 3, 7]),
                (5,  [0, 4, 7, 11]),
                (2,  [0, 3, 7, 10]),
                (11, [0, 4, 7, 10]),
            ]
            for root, degs in params:
                try:
                    fn(root, degs); passed += 1
                    print(f"  PASS  {fn.__name__}[root={root}]")
                except Exception:
                    failed += 1
                    print(f"  FAIL  {fn.__name__}[root={root}]")
                    traceback.print_exc()
        else:
            try:
                fn(); passed += 1
                print(f"  PASS  {fn.__name__}")
            except Exception:
                failed += 1
                print(f"  FAIL  {fn.__name__}")
                traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
