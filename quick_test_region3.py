"""
Quick test of a few hand-picked parameter combinations to find:
1. W_nested = W_iid
2. Region III exists (alpha2 < 1, close to 1)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import credit_model as cm
from credit_model import Parameters, solve_nested_analytical, solve_iid
from scipy.optimize import brentq

def set_cost(c2, c1):
    cm.cfun = lambda a: c2*a**2 + c1*a
    cm.cfun_prime_exact = lambda a: 2*c2*a + c1
    cm.cfun_prime2_exact = lambda a: 2*c2

def test_params(Pi, beta, BperG, c2, c1):
    """Test specific parameter combination."""
    set_cost(c2, c1)
    p = Parameters(Pi=Pi, beta=beta, BperG=BperG, a_g=0, a_b=0)
    cm.dfun = lambda r: 1.0/r
    cm._current_params = p

    n = solve_nested_analytical(p)
    iid = solve_iid(p)

    W_n = n['W_R2_cumsum'][-1] + n['WNS']
    W_i = iid['W_cumsum'][-1]

    return {
        'W_diff': W_n - W_i,
        'W_ratio': W_n / W_i,
        'alpha2': n['alpha2'],
        'alpha1': n['alpha1'],
        'alpha0': n['alpha0'],
        'rp': n['rp'],
        'WNS': n['WNS'],
        'W_n': W_n,
        'W_i': W_i
    }

print("=" * 80)
print("QUICK TEST: Hand-picked parameter combinations")
print("=" * 80)
print()

# Test cases based on intuition:
# - Higher beta usually pushes alpha2 closer to 1
# - Need to find the right BperG for crossing
# - Cost function affects where alpha2 ends up

test_cases = [
    # (Pi, beta, c2, c1, description)
    # Try LOWER beta - Region III exists more at low beta
    (0.05, 0.10, 0.8, 0.5, "Low beta (0.1)"),
    (0.05, 0.15, 0.8, 0.5, "Low beta (0.15)"),
    (0.05, 0.20, 0.8, 0.5, "Low beta (0.2)"),
    (0.05, 0.25, 0.8, 0.5, "Med-low beta (0.25)"),
    # Try different cost with low beta
    (0.05, 0.15, 1.0, 0.6, "Low beta, steeper cost"),
    (0.05, 0.20, 1.2, 0.6, "Low beta, steep cost"),
    # Try different Pi with low beta
    (0.07, 0.15, 0.8, 0.5, "Low beta, high Pi"),
    (0.03, 0.20, 0.8, 0.5, "Low beta, low Pi"),
]

candidates = []

for Pi, beta, c2, c1, desc in test_cases:
    print(f"\nTesting: {desc}")
    print(f"  Pi={Pi}, beta={beta}, c2={c2}, c1={c1}")
    print(f"  Scanning BperG to find crossing...")

    # Quick scan - lower range for Region III
    bpg_vals = np.linspace(0.05, 0.6, 15)
    diffs = []

    for bpg in bpg_vals:
        try:
            result = test_params(Pi, beta, bpg, c2, c1)
            diffs.append(result['W_diff'])
        except:
            diffs.append(np.nan)

    # Find crossing
    found_crossing = False
    for i in range(len(diffs)-1):
        if not np.isnan(diffs[i]) and not np.isnan(diffs[i+1]):
            if diffs[i] * diffs[i+1] < 0:
                # Found crossing
                try:
                    def diff_fn(b):
                        return test_params(Pi, beta, b, c2, c1)['W_diff']

                    bpg_cross = brentq(diff_fn, bpg_vals[i], bpg_vals[i+1], xtol=1e-4)
                    result = test_params(Pi, beta, bpg_cross, c2, c1)

                    print(f"  -> CROSSING at BperG = {bpg_cross:.6f}")
                    print(f"     alpha2 = {result['alpha2']:.6f}")
                    print(f"     WNS = {result['WNS']:.5f}")
                    print(f"     W_n = W_i = {result['W_n']:.5f}")

                    if result['alpha2'] > 0.85 and result['alpha2'] < 0.999:
                        print(f"     *** GOOD: Region III exists and alpha2 close to 1!")
                        candidates.append({
                            'Pi': Pi, 'beta': beta, 'BperG': bpg_cross,
                            'c2': c2, 'c1': c1, 'desc': desc,
                            **result
                        })
                    found_crossing = True
                except Exception as e:
                    print(f"     Error refining: {e}")

    if not found_crossing:
        print(f"  -> No crossing found in BperG range [0.05, 0.6]")

print()
print("=" * 80)
print(f"FOUND {len(candidates)} GOOD CANDIDATES")
print("=" * 80)
print()

if candidates:
    # Sort by alpha2
    candidates.sort(key=lambda x: -x['alpha2'])

    print(f"{'#':>2} {'Pi':>6} {'beta':>6} {'BperG':>9} {'c2':>6} {'c1':>6} "
          f"{'alpha2':>8} {'W':>9} {'WNS':>9}")
    print("-" * 80)

    for i, c in enumerate(candidates, 1):
        print(f"{i:>2} {c['Pi']:>6.3f} {c['beta']:>6.3f} {c['BperG']:>9.6f} "
              f"{c['c2']:>6.2f} {c['c1']:>6.2f} {c['alpha2']:>8.5f} "
              f"{c['W_n']:>9.4f} {c['WNS']:>9.5f}")

    # Best
    best = candidates[0]
    print()
    print("BEST SOLUTION:")
    print(f"  {best['desc']}")
    print(f"  Pi = {best['Pi']}, beta = {best['beta']}, BperG = {best['BperG']:.7f}")
    print(f"  c(a) = {best['c2']}*a^2 + {best['c1']}*a")
    print(f"  alpha2 = {best['alpha2']:.6f} (Region III!)")
    print(f"  WNS/W = {100*best['WNS']/best['W_n']:.1f}%")

else:
    print("No suitable candidates found. Try different parameters or wider BperG range.")
