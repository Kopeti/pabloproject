"""
Find exact BperG* where W_nested = W_iid for baseline parameters.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import credit_model as cm
from credit_model import Parameters, solve_nested_analytical, solve_iid
from scipy.optimize import brentq

def compute_diff(beta_v, bpg_v, dfun_fn):
    """Return W_nested - W_iid for given parameters."""
    p = Parameters(Pi=0.05, beta=beta_v, BperG=bpg_v,
                   Delta=0.001, delom=0.0001, a_g=0.0, a_b=0.0)
    cm.dfun = dfun_fn
    cm._current_params = p
    n = solve_nested_analytical(p)
    iid = solve_iid(p)
    W_n = n['W_R2_cumsum'][-1] + n['WNS']
    W_i = iid['W_cumsum'][-1]
    return W_n - W_i

# D(r) = 1/r
dfun_1_r = lambda r: 1.0 / r

print("=" * 80)
print("FIND EXACT CROSSING BperG* where W_nested = W_iid")
print("=" * 80)
print(f"Parameters: Pi=0.05, uniform priors (a_g=0, a_b=0), D(r) = 1/r")
print()

# For beta=0.1: two crossings (based on fine sweep)
print("beta = 0.1  (default)")
print("-" * 80)

# Check values around suspected crossings
test_bpg = [0.10, 0.13, 0.14, 0.15, 0.18, 0.20, 0.70, 0.80, 0.85, 0.90]
print(f"  {'BperG':>7}  {'W_nested':>10}  {'W_iid':>10}  {'diff':>10}  {'ratio':>10}")
for bpg in test_bpg:
    try:
        diff = compute_diff(0.1, bpg, dfun_1_r)
        p = Parameters(Pi=0.05, beta=0.1, BperG=bpg, a_g=0.0, a_b=0.0)
        cm.dfun = dfun_1_r
        cm._current_params = p
        n = solve_nested_analytical(p)
        iid = solve_iid(p)
        W_n = n['W_R2_cumsum'][-1] + n['WNS']
        W_i = iid['W_cumsum'][-1]
        ratio = W_n / W_i if W_i > 1e-12 else float('inf')
        print(f"  {bpg:>7.2f}  {W_n:>10.4f}  {W_i:>10.4f}  {diff:>10.4f}  {ratio:>10.4f}")
    except Exception as e:
        print(f"  {bpg:>7.2f}  ERROR: {e}")

# Find first crossing (nested → iid)
try:
    bpg_cross1 = brentq(lambda b: compute_diff(0.1, b, dfun_1_r), 0.13, 0.16, xtol=1e-5)
    print(f"\n  First crossing (nested→iid):  BperG* = {bpg_cross1:.6f}")
except Exception as e:
    print(f"\n  First crossing: FAILED {e}")

# Find second crossing (iid → nested)
try:
    bpg_cross2 = brentq(lambda b: compute_diff(0.1, b, dfun_1_r), 0.70, 0.90, xtol=1e-5)
    print(f"  Second crossing (iid→nested): BperG* = {bpg_cross2:.6f}")
except Exception as e:
    print(f"  Second crossing: FAILED {e}")

# For beta=0.3: single crossing (cleaner)
print("\n\nbeta = 0.3")
print("-" * 80)
test_bpg_3 = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70]
print(f"  {'BperG':>7}  {'W_nested':>10}  {'W_iid':>10}  {'diff':>10}  {'ratio':>10}")
for bpg in test_bpg_3:
    try:
        diff = compute_diff(0.3, bpg, dfun_1_r)
        p = Parameters(Pi=0.05, beta=0.3, BperG=bpg, a_g=0.0, a_b=0.0)
        cm.dfun = dfun_1_r
        cm._current_params = p
        n = solve_nested_analytical(p)
        iid = solve_iid(p)
        W_n = n['W_R2_cumsum'][-1] + n['WNS']
        W_i = iid['W_cumsum'][-1]
        ratio = W_n / W_i if W_i > 1e-12 else float('inf')
        print(f"  {bpg:>7.2f}  {W_n:>10.4f}  {W_i:>10.4f}  {diff:>10.4f}  {ratio:>10.4f}")
    except Exception as e:
        print(f"  {bpg:>7.2f}  ERROR: {e}")

try:
    bpg_cross_3 = brentq(lambda b: compute_diff(0.3, b, dfun_1_r), 0.50, 0.70, xtol=1e-5)
    print(f"\n  Crossing (iid→nested): BperG* = {bpg_cross_3:.6f}")
except Exception as e:
    print(f"\n  Crossing: FAILED {e}")

# For the default BperG=0.2 with beta=0.1
print("\n\n" + "=" * 80)
print("AT DEFAULT: beta=0.1, BperG=0.2")
print("=" * 80)
diff_default = compute_diff(0.1, 0.2, dfun_1_r)
p = Parameters(Pi=0.05, beta=0.1, BperG=0.2, a_g=0.0, a_b=0.0)
cm.dfun = dfun_1_r
cm._current_params = p
n = solve_nested_analytical(p)
iid = solve_iid(p)
W_n = n['W_R2_cumsum'][-1] + n['WNS']
W_i = iid['W_cumsum'][-1]
ratio = W_n / W_i if W_i > 1e-12 else float('inf')
print(f"  W_nested = {W_n:.5f}")
print(f"  W_iid    = {W_i:.5f}")
print(f"  diff     = {diff_default:.5f}  ({'nested wins' if diff_default > 0 else 'iid WINS'})")
print(f"  ratio    = {ratio:.5f}")
print(f"\n  For beta=0.1, BperG=0.2 is {'ABOVE' if 0.2 > bpg_cross1 else 'BELOW'} first crossing ({bpg_cross1:.4f})")
print(f"  and {'BELOW' if 0.2 < bpg_cross2 else 'ABOVE'} second crossing ({bpg_cross2:.4f})")
print(f"  → At BperG=0.2, {'NESTED' if diff_default > 0 else 'IID'} wins (in the middle region)")