"""
Experiment: How does D(r) affect the comparison W_bar^nested vs W_bar^iid?

The model has two information structures:
  - Nested:  r^nested(alpha) <= r^iid(alpha) everywhere (lower rates)
  - IID:     r^iid(alpha)    >= r^nested(alpha) everywhere (higher rates)

Proposition (r-comparison): r^iid > r^nested for alpha != alpha0.

Question: Does W_bar^nested > W_bar^iid (as the lemma says for D=1)?
         Or can steep D reverse this?

W_bar = total capital = integral of w(alpha) * D(r(alpha)) dalpha.
When D=1 this is just total borrowers served; nested wins because it
serves everyone (including bad) through Region III.

With D(r) decreasing (borrower demand): nested has LOWER r -> higher D
per borrower, nested should still win... but does it?

This script sweeps D(r) forms and reports the comparison.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import credit_model as cm
from credit_model import Parameters, solve_nested_analytical, solve_iid, find_alpha2
from scipy.integrate import quad

# Use strict uniform priors
params = Parameters(
    Pi=0.05, beta=0.1, BperG=0.2,
    Delta=0.001, delom=0.0001,
    a_g=0.0, a_b=0.0   # uniform
)

# Patch priors to exactly uniform (a_g=a_b=0 already does this)
cm._current_params = params

# ============================================================
# Demand functions to test
# ============================================================
demand_specs = [
    ("D=1 (borrower count)",   lambda r: 1.0),
    ("D=1/r^0.25",             lambda r: r**(-0.25)),
    ("D=1/r^0.5",              lambda r: r**(-0.5)),
    ("D=1/r (current)",        lambda r: 1.0 / r),
    ("D=1/r^2",                lambda r: r**(-2.0)),
    ("D=1/r^5",                lambda r: r**(-5.0)),
    ("D=r^0.5 (increasing)",   lambda r: r**0.5),
    ("D=r (increasing)",       lambda r: r),
    ("D=r^2 (increasing)",     lambda r: r**2.0),
]

# ============================================================
# Solve the structural equilibrium ONCE (D only affects scale
# of w, not the alpha0/rp/pool-quality equilibrium objects,
# because the zero-profit condition is gamma*(1+r)=1+K which
# is independent of D).
# ============================================================
# We need to monkey-patch dfun for the solvers (it affects w).
# But alpha0, rp, alpha1, alpha2 are determined by the zero-profit
# condition and pool dynamics which DON'T depend on D(r) directly:
# the solver uses dfun inside the loop, so we must patch per run.

def total_nested_capital(params, dfun_fn):
    """Solve nested with given dfun and return total capital."""
    cm.dfun = dfun_fn
    n = solve_nested_analytical(params)
    return n['W_R2_cumsum'][-1] + n['WNS'], n

def total_iid_capital(params, dfun_fn):
    """Solve IID with given dfun and return total capital."""
    cm.dfun = dfun_fn
    iid = solve_iid(params)
    return iid['W_cumsum'][-1], iid

print("=" * 75)
print("EXPERIMENT: W_bar^nested vs W_bar^iid as D(r) varies")
print(f"Parameters: Pi={params.Pi}, beta={params.beta}, BperG={params.BperG} (uniform priors)")
print("=" * 75)

# Solve structural objects once (with D=1 to get alpha0, rp, etc.)
cm.dfun = lambda r: 1.0
n_ref = solve_nested_analytical(params)
iid_ref = solve_iid(params)
alpha0_n = n_ref['alpha0']
alpha0_i = iid_ref['alpha0']
rp_n     = n_ref['rp']
r0_i     = iid_ref['r0']
alpha1   = n_ref['alpha1']
alpha2   = n_ref['alpha2']
alphabar = iid_ref['alpha_bar']

print(f"\nEquilibrium (D-independent objects):")
print(f"  alpha0 (nested) = {alpha0_n:.5f}")
print(f"  alpha0 (IID)    = {alpha0_i:.5f}   (should equal nested)")
print(f"  rp     (nested) = {rp_n:.5f}")
print(f"  r0     (IID)    = {r0_i:.5f}   (should equal rp)")
print(f"  alpha1          = {alpha1:.5f}")
print(f"  alpha2          = {alpha2:.5f}")
print(f"  alpha_bar (IID) = {alphabar:.5f}")
print(f"  r_NS (nested)   = {n_ref['r_NS']:.5f}")
print(f"  r_perfect (IID) = {iid_ref['r_perfect']:.5f}")

# Borrower counts (D-independent)
G0, B0 = iid_ref['G0'], iid_ref['B0']
G_end_iid = iid_ref['G_eq'][-1]
B_end_iid = iid_ref['B_eq'][-1]
borrowers_iid     = (G0 - G_end_iid) + (B0 - B_end_iid)
borrowers_nested  = G0 + B0  # all served

print(f"\nBorrower counts (D-independent):")
print(f"  Nested: {borrowers_nested:.5f}  (= g_bar + b_bar = {G0+B0:.5f})")
print(f"  IID:    {borrowers_iid:.5f}  (excluded bad = {B_end_iid:.5f})")
print(f"  Nested > IID: {borrowers_nested > borrowers_iid}")

print(f"\n{'D function':<25}  {'W_nested':>10}  {'W_iid':>10}  {'ratio N/I':>10}  {'nested>iid':>11}")
print("-" * 75)

for name, dfn in demand_specs:
    try:
        W_n, n_sol = total_nested_capital(params, dfn)
        W_i, i_sol = total_iid_capital(params, dfn)
        ratio = W_n / W_i if W_i > 1e-12 else float('inf')
        print(f"  {name:<23}  {W_n:>10.4f}  {W_i:>10.4f}  {ratio:>10.4f}  {str(W_n > W_i):>11}")
    except Exception as e:
        print(f"  {name:<23}  ERROR: {e}")

# ============================================================
# Deeper analysis with D=1/r (the current default)
# ============================================================
cm.dfun = lambda r: 1.0 / r
W_n_dflt, n_dflt = total_nested_capital(params, lambda r: 1.0/r)
W_i_dflt, i_dflt = total_iid_capital(params, lambda r: 1.0/r)

print(f"\n{'='*75}")
print(f"DECOMPOSITION with D(r) = 1/r")
print(f"{'='*75}")

# Nested decomposition
W_R1  = n_dflt['W_cumsum_R1'][-1]
W_R2  = n_dflt['W_R2_cumsum'][-1] - W_R1
WNS   = n_dflt['WNS']
r_NS  = n_dflt['r_NS']

print(f"\nNested total capital = {W_n_dflt:.5f}")
print(f"  Region I  (alpha in [{alpha0_n:.3f}, {alpha1:.3f}], flat rate {rp_n:.4f}): {W_R1:.5f}")
print(f"  Region II (alpha in [{alpha1:.3f}, {alpha2:.3f}], rising rate): {W_R2:.5f}")
print(f"  Region III (non-selective, rate {r_NS:.4f}): {WNS:.5f}")
print(f"    WNS * D(r_NS) = {WNS:.5f}  where D(r_NS) = {1.0/r_NS:.5f}")
print(f"    badleftover = {n_dflt['badleftover']:.5f}")

print(f"\nIID total capital = {W_i_dflt:.5f}")
print(f"  alpha range: [{alpha0_i:.3f}, {alphabar:.3f}]")
print(f"  rate range:  [{r0_i:.4f}, {iid_ref['r_perfect']:.4f}]")
print(f"  D(r) range:  [{1/iid_ref['r_perfect']:.4f}, {1/r0_i:.4f}]")

# Per-borrower average D(r) under each regime
D_avg_nested = W_n_dflt / borrowers_nested if borrowers_nested > 1e-12 else 0
D_avg_iid    = W_i_dflt / borrowers_iid    if borrowers_iid > 1e-12 else 0
print(f"\nAverage capital per borrower:")
print(f"  Nested: {W_n_dflt:.5f} / {borrowers_nested:.5f} = {D_avg_nested:.5f}")
print(f"  IID:    {W_i_dflt:.5f} / {borrowers_iid:.5f} = {D_avg_iid:.5f}")
print(f"  (Nested has lower rates -> higher D per borrower: {D_avg_nested > D_avg_iid})")

# ============================================================
# The puzzle: which component drives W_iid > W_nested?
# Decompose as: W = (borrowers served) x (avg D)
# ============================================================
print(f"\nDecomposition W = borrowers x avg_D:")
print(f"  Nested: {borrowers_nested:.5f} x {D_avg_nested:.5f} = {W_n_dflt:.5f}")
print(f"  IID:    {borrowers_iid:.5f} x {D_avg_iid:.5f} = {W_i_dflt:.5f}")
print(f"  -> Nested serves MORE borrowers: {borrowers_nested:.4f} > {borrowers_iid:.4f}")
print(f"  -> Nested has higher avg D:      {D_avg_nested:.4f} > {D_avg_iid:.4f}  (if true)")
print(f"  => Both effects should favor nested... but W_nested < W_iid? {W_n_dflt < W_i_dflt}")

# ============================================================
# Sweep beta and BperG to see when/if nested wins
# ============================================================
print(f"\n{'='*75}")
print("SWEEP: beta and BperG with D=1/r")
print(f"{'='*75}")
print(f"{'beta':>6}  {'BperG':>6}  {'W_nested':>10}  {'W_iid':>10}  {'ratio':>8}  {'nested>iid':>11}")
print("-" * 60)

for beta_v in [0.1, 0.3, 0.5, 0.7]:
    for bpg_v in [0.1, 0.3, 0.5, 1.0]:
        try:
            p = Parameters(Pi=0.05, beta=beta_v, BperG=bpg_v,
                          Delta=0.001, delom=0.0001, a_g=0.0, a_b=0.0)
            cm.dfun = lambda r: 1.0/r
            W_n, _ = total_nested_capital(p, lambda r: 1.0/r)
            W_i, _ = total_iid_capital(p, lambda r: 1.0/r)
            ratio = W_n / W_i if W_i > 1e-12 else float('inf')
            print(f"  {beta_v:>4.1f}  {bpg_v:>6.1f}  {W_n:>10.4f}  {W_i:>10.4f}  {ratio:>8.4f}  {str(W_n > W_i):>11}")
        except Exception as e:
            print(f"  {beta_v:>4.1f}  {bpg_v:>6.1f}  ERROR: {e}")

print(f"\n{'='*75}")
print("CONCLUSION")
print(f"{'='*75}")
print("With D=1: nested always > iid (borrower count, see lemma in proof)")
print("With D=1/r (decreasing): ??? -- see above")
print("With D=r^k (increasing): see above -- higher rates attract more capital")
print("Key: if D is the SUPPLY of lender capital (increasing in r),")
print("     IID's higher rates attract more capital, possibly W_iid > W_nested.")
print("If D is BORROWER demand (decreasing), both effects favor nested,")
print("     so W_nested > W_iid always -- unless there is a SCALING ISSUE in the code.")