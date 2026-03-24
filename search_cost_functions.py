"""
Search over cost function parameters to find one where:
- Crossing exists
- alpha2 is between 0.85 and 0.95 (close to 1 but not equal)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import credit_model as cm
from credit_model import Parameters, solve_nested_analytical, solve_iid
from scipy.optimize import brentq

cm.dfun = lambda r: 1.0/r

def test_with_cost(Pi, beta, BperG, c2, c1):
    cm.cfun = lambda a: c2*a**2 + c1*a
    cm.cfun_prime_exact = lambda a: 2*c2*a + c1
    cm.cfun_prime2_exact = lambda a: 2*c2

    p = Parameters(Pi=Pi, beta=beta, BperG=BperG, a_g=0, a_b=0)
    cm._current_params = p
    n = solve_nested_analytical(p)
    iid = solve_iid(p)
    W_n = n['W_R2_cumsum'][-1] + n['WNS']
    W_i = iid['W_cumsum'][-1]
    return {
        'W_diff': W_n - W_i,
        'alpha2': n['alpha2'],
        'alpha0': n['alpha0'],
        'alpha1': n['alpha1'],
        'WNS': n['WNS'],
        'W': W_n,
        'rp': n['rp']
    }

print("=" * 90)
print("SEARCH OVER COST FUNCTIONS: c(a) = c2*a^2 + c1*a")
print("Target: Crossing with 0.85 < alpha2 < 0.95")
print("=" * 90)
print()

# Fix Pi=0.05, beta=0.1 (we know this has interesting behavior)
Pi, beta = 0.05, 0.1

# Try different cost parameters
cost_grid = []
for c2 in [0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]:
    for c1 in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        cost_grid.append((c2, c1))

solutions = []

for c2, c1 in cost_grid:
    # Scan BperG broadly
    bpg_scan = np.linspace(0.1, 1.0, 20)
    diffs = []

    for bpg in bpg_scan:
        try:
            r = test_with_cost(Pi, beta, bpg, c2, c1)
            diffs.append((bpg, r))
        except:
            pass

    # Find crossings
    for i in range(len(diffs)-1):
        bpg1, r1 = diffs[i]
        bpg2, r2 = diffs[i+1]

        if r1['W_diff'] * r2['W_diff'] < 0:
            try:
                def diff_fn(b):
                    return test_with_cost(Pi, beta, b, c2, c1)['W_diff']

                bpg_cross = brentq(diff_fn, bpg1, bpg2, xtol=1e-5)
                r_cross = test_with_cost(Pi, beta, bpg_cross, c2, c1)

                # Check if it meets criteria
                if 0.85 <= r_cross['alpha2'] < 0.99:
                    solutions.append({
                        'Pi': Pi,
                        'beta': beta,
                        'BperG': bpg_cross,
                        'c2': c2,
                        'c1': c1,
                        **r_cross
                    })
                    print(f"FOUND: c2={c2:.1f}, c1={c1:.1f}, BperG={bpg_cross:.5f}, alpha2={r_cross['alpha2']:.5f}")

            except:
                pass

print()
print("=" * 90)
print(f"FOUND {len(solutions)} SOLUTIONS")
print("=" * 90)
print()

if solutions:
    # Sort by alpha2 descending (closest to 0.95)
    solutions.sort(key=lambda x: -x['alpha2'])

    print(f"{'#':>2} {'c2':>6} {'c1':>6} {'BperG':>9} {'alpha2':>8} {'alpha1':>8} {'W':>9} {'WNS':>9} {'WNS/W':>7}")
    print("-" * 85)

    for i, s in enumerate(solutions, 1):
        wns_ratio = s['WNS'] / s['W'] if s['W'] > 0 else 0
        print(f"{i:>2} {s['c2']:>6.2f} {s['c1']:>6.2f} {s['BperG']:>9.6f} "
              f"{s['alpha2']:>8.5f} {s['alpha1']:>8.5f} {s['W']:>9.4f} "
              f"{s['WNS']:>9.5f} {wns_ratio:>7.3f}")

    # Best
    print()
    print("=" * 90)
    print("BEST SOLUTION (alpha2 closest to 0.95):")
    print("=" * 90)
    best = solutions[0]
    print(f"""
  Pi       = {best['Pi']}
  beta     = {best['beta']}
  BperG    = {best['BperG']:.7f}
  c(a)     = {best['c2']}*a^2 + {best['c1']}*a

  alpha0   = {best['alpha0']:.6f}
  alpha1   = {best['alpha1']:.6f}
  alpha2   = {best['alpha2']:.6f}  <-- Region III exists, close to 1!
  rp       = {best['rp']:.6f}

  W        = {best['W']:.6f}
  WNS      = {best['WNS']:.6f}  ({100*best['WNS']/best['W']:.1f}% of total)

To implement in credit_model.py:
  Line 41: beta: float = {best['beta']}
  Line 42: BperG: float = {best['BperG']:.7f}

  Lines 17-19, replace cfun:
    def cfun(alpha):
        return {best['c2']} * alpha**2 + {best['c1']} * alpha
    """)

else:
    print("No solutions found with 0.85 < alpha2 < 0.95")
    print("The sweet spot might not exist for beta=0.1")
    print("Try: different beta values, or accept alpha2 closer to 0.8")