"""
Find parameters where:
1. W_nested = W_iid (crossing point)
2. Region III exists (alpha2 < 1)
3. alpha2 is relatively close to 1

Search over: Pi, beta, BperG, and cost function c(alpha) = c2*alpha^2 + c1*alpha

STRATEGY: Start from known working cases and explore systematically.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import credit_model as cm
from credit_model import Parameters, solve_nested_analytical, solve_iid
from scipy.optimize import brentq
import warnings
warnings.filterwarnings('ignore')

def set_cost_function(c2, c1):
    """Set cost function c(alpha) = c2*alpha^2 + c1*alpha."""
    cm.cfun = lambda alpha: c2 * alpha**2 + c1 * alpha
    cm.cfun_prime_exact = lambda alpha: 2*c2*alpha + c1
    cm.cfun_prime2_exact = lambda alpha: 2*c2

def compute_W_diff(Pi, beta, BperG, c2, c1):
    """Return W_nested - W_iid and equilibrium details."""
    try:
        set_cost_function(c2, c1)
        p = Parameters(Pi=Pi, beta=beta, BperG=BperG, a_g=0.0, a_b=0.0)
        cm.dfun = lambda r: 1.0 / r
        cm._current_params = p

        n = solve_nested_analytical(p)
        iid = solve_iid(p)

        return {
            'diff': n['W_R2_cumsum'][-1] + n['WNS'] - iid['W_cumsum'][-1],
            'alpha2': n['alpha2'],
            'alpha1': n['alpha1'],
            'alpha0': n['alpha0'],
            'WNS': n['WNS'],
            'W_nested': n['W_R2_cumsum'][-1] + n['WNS'],
            'W_iid': iid['W_cumsum'][-1],
            'rp': n['rp']
        }
    except:
        return None

print("=" * 90)
print("TARGETED SEARCH: W_nested = W_iid + Region III + alpha2 close to 1")
print("=" * 90)
print()

solutions = []

# PHASE 1: Test cost functions with current beta=0.3
print("PHASE 1: Varying cost function with beta=0.3, Pi=0.05")
print("-" * 90)

cost_params = [
    (0.5, 0.4), (0.5, 0.6), (0.5, 0.8),
    (0.8, 0.3), (0.8, 0.5), (0.8, 0.7),
    (1.0, 0.4), (1.0, 0.6), (1.0, 0.8),
    (1.5, 0.5), (1.5, 0.7),
    (2.0, 0.6), (2.0, 0.8)
]

for c2, c1 in cost_params:
    # Scan BperG to find crossing
    bpg_test = np.linspace(0.15, 1.0, 15)
    diffs = [compute_W_diff(0.05, 0.3, b, c2, c1) for b in bpg_test]

    # Find sign changes
    for i in range(len(diffs)-1):
        if diffs[i] and diffs[i+1]:
            if diffs[i]['diff'] * diffs[i+1]['diff'] < 0:
                try:
                    bpg_cross = brentq(
                        lambda b: compute_W_diff(0.05, 0.3, b, c2, c1)['diff'],
                        bpg_test[i], bpg_test[i+1], xtol=1e-4
                    )
                    result = compute_W_diff(0.05, 0.3, bpg_cross, c2, c1)
                    if result['alpha2'] > 0.85 and result['alpha2'] < 0.999:
                        solutions.append({
                            'Pi': 0.05, 'beta': 0.3, 'BperG': bpg_cross,
                            'c2': c2, 'c1': c1, **result
                        })
                        print(f"  c(a)={c2}a^2+{c1}a: BperG={bpg_cross:.5f}, alpha2={result['alpha2']:.5f}")
                except:
                    pass

print()

# PHASE 2: Vary Pi with best cost from Phase 1 (or default)
print("PHASE 2: Varying Pi with beta=0.3, c(a)=0.8a^2+0.5a")
print("-" * 90)

for Pi in [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]:
    bpg_test = np.linspace(0.15, 1.0, 15)
    diffs = [compute_W_diff(Pi, 0.3, b, 0.8, 0.5) for b in bpg_test]

    for i in range(len(diffs)-1):
        if diffs[i] and diffs[i+1]:
            if diffs[i]['diff'] * diffs[i+1]['diff'] < 0:
                try:
                    bpg_cross = brentq(
                        lambda b: compute_W_diff(Pi, 0.3, b, 0.8, 0.5)['diff'],
                        bpg_test[i], bpg_test[i+1], xtol=1e-4
                    )
                    result = compute_W_diff(Pi, 0.3, bpg_cross, 0.8, 0.5)
                    if result['alpha2'] > 0.85 and result['alpha2'] < 0.999:
                        solutions.append({
                            'Pi': Pi, 'beta': 0.3, 'BperG': bpg_cross,
                            'c2': 0.8, 'c1': 0.5, **result
                        })
                        print(f"  Pi={Pi:.2f}: BperG={bpg_cross:.5f}, alpha2={result['alpha2']:.5f}")
                except:
                    pass

print()

# PHASE 3: Vary beta with best Pi (or default)
print("PHASE 3: Varying beta with Pi=0.05, c(a)=0.8a^2+0.5a")
print("-" * 90)

for beta in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.32, 0.35]:
    bpg_test = np.linspace(0.15, 1.2, 15)
    diffs = [compute_W_diff(0.05, beta, b, 0.8, 0.5) for b in bpg_test]

    for i in range(len(diffs)-1):
        if diffs[i] and diffs[i+1]:
            if diffs[i]['diff'] * diffs[i+1]['diff'] < 0:
                try:
                    bpg_cross = brentq(
                        lambda b: compute_W_diff(0.05, beta, b, 0.8, 0.5)['diff'],
                        bpg_test[i], bpg_test[i+1], xtol=1e-4
                    )
                    result = compute_W_diff(0.05, beta, bpg_cross, 0.8, 0.5)
                    if result['alpha2'] > 0.85 and result['alpha2'] < 0.999:
                        solutions.append({
                            'Pi': 0.05, 'beta': beta, 'BperG': bpg_cross,
                            'c2': 0.8, 'c1': 0.5, **result
                        })
                        print(f"  beta={beta:.2f}: BperG={bpg_cross:.5f}, alpha2={result['alpha2']:.5f}")
                except:
                    pass

print()
print("=" * 90)
print(f"TOTAL: {len(solutions)} SOLUTIONS FOUND")
print("=" * 90)
print()

if solutions:
    # Sort by alpha2 descending (closest to 1)
    solutions.sort(key=lambda x: -x['alpha2'])

    print("TOP SOLUTIONS (alpha2 closest to 1):")
    print()
    print(f"{'#':>3} {'Pi':>6} {'beta':>6} {'BperG':>9} {'c2':>6} {'c1':>6} "
          f"{'alpha2':>8} {'alpha1':>8} {'W':>9} {'WNS/W':>7}")
    print("-" * 85)

    for i, s in enumerate(solutions[:10], 1):
        wns_frac = s['WNS'] / s['W_nested'] if s['W_nested'] > 0 else 0
        print(f"{i:>3} {s['Pi']:>6.3f} {s['beta']:>6.3f} {s['BperG']:>9.6f} "
              f"{s['c2']:>6.2f} {s['c1']:>6.2f} {s['alpha2']:>8.5f} "
              f"{s['alpha1']:>8.5f} {s['W_nested']:>9.4f} {wns_frac:>7.3f}")

    # Best solution
    print()
    print("=" * 90)
    print("★ BEST SOLUTION (alpha2 closest to 1):")
    print("=" * 90)
    best = solutions[0]
    print(f"""
  Pi       = {best['Pi']}
  beta     = {best['beta']}
  BperG    = {best['BperG']:.7f}
  c(a)     = {best['c2']}*a^2 + {best['c1']}*a

  alpha0   = {best['alpha0']:.6f}
  alpha1   = {best['alpha1']:.6f}
  alpha2   = {best['alpha2']:.6f}  ← Region III exists!
  rp       = {best['rp']:.6f}

  W_nested = {best['W_nested']:.6f}
  W_iid    = {best['W_iid']:.6f}
  WNS      = {best['WNS']:.6f}  ({100*best['WNS']/best['W_nested']:.1f}% of total capital)

To use these parameters, update credit_model.py:
  Pi = {best['Pi']}
  beta = {best['beta']}
  BperG = {best['BperG']:.7f}

And modify cfun (lines ~17-19):
  def cfun(alpha):
      return {best['c2']} * alpha**2 + {best['c1']} * alpha
    """)

else:
    print("No solutions found satisfying all constraints.")
    print("Consider: lower alpha2 threshold, wider BperG range, or different cost forms.")

print("=" * 90)
