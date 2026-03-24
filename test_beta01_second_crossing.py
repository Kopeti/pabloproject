"""
Test beta=0.1 more carefully - we know it has TWO crossings.
The second crossing might have alpha2 closer to 1.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import credit_model as cm
from credit_model import Parameters, solve_nested_analytical, solve_iid
from scipy.optimize import brentq

# Set default cost
cm.cfun = lambda a: 0.8*a**2 + 0.5*a
cm.cfun_prime_exact = lambda a: 1.6*a + 0.5
cm.cfun_prime2_exact = lambda a: 1.6
cm.dfun = lambda r: 1.0/r

def test_params(Pi, beta, BperG):
    p = Parameters(Pi=Pi, beta=beta, BperG=BperG, a_g=0, a_b=0)
    cm._current_params = p
    n = solve_nested_analytical(p)
    iid = solve_iid(p)
    W_n = n['W_R2_cumsum'][-1] + n['WNS']
    W_i = iid['W_cumsum'][-1]
    return {
        'W_diff': W_n - W_i,
        'alpha2': n['alpha2'],
        'alpha1': n['alpha1'],
        'alpha0': n['alpha0'],
        'WNS': n['WNS'],
        'W_n': W_n,
        'W_i': W_i,
        'rp': n['rp']
    }

print("=" * 80)
print("DETAILED ANALYSIS: beta=0.1 (two crossings expected)")
print("=" * 80)
print()

# Scan full range
BperG_vals = np.linspace(0.05, 1.2, 30)
results = []

print("Scanning BperG from 0.05 to 1.2...")
for bpg in BperG_vals:
    try:
        r = test_params(0.05, 0.1, bpg)
        results.append((bpg, r))
        print(f"  BperG={bpg:.4f}: W_diff={r['W_diff']:>8.4f}, alpha2={r['alpha2']:.5f}, WNS={r['WNS']:.4f}")
    except Exception as e:
        print(f"  BperG={bpg:.4f}: ERROR - {e}")

print()
print("-" * 80)
print("Finding crossings (where W_diff changes sign)...")
print()

crossings = []
for i in range(len(results)-1):
    bpg1, r1 = results[i]
    bpg2, r2 = results[i+1]

    if r1['W_diff'] * r2['W_diff'] < 0:
        # Crossing found
        try:
            def diff_fn(b):
                return test_params(0.05, 0.1, b)['W_diff']

            bpg_cross = brentq(diff_fn, bpg1, bpg2, xtol=1e-5)
            r_cross = test_params(0.05, 0.1, bpg_cross)
            crossings.append((bpg_cross, r_cross))

            print(f"CROSSING #{len(crossings)} at BperG = {bpg_cross:.7f}")
            print(f"  W_nested = W_iid = {r_cross['W_n']:.6f}")
            print(f"  alpha0 = {r_cross['alpha0']:.6f}")
            print(f"  alpha1 = {r_cross['alpha1']:.6f}")
            print(f"  alpha2 = {r_cross['alpha2']:.6f}")
            print(f"  WNS    = {r_cross['WNS']:.6f}")
            print(f"  rp     = {r_cross['rp']:.6f}")

            if r_cross['alpha2'] > 0.85 and r_cross['alpha2'] < 0.999:
                print(f"  >>> GOOD! Region III exists and alpha2 close to 1!")
            elif r_cross['alpha2'] < 0.999:
                print(f"  >>> Region III exists but alpha2 too small ({r_cross['alpha2']:.3f})")
            else:
                print(f"  >>> No Region III (alpha2 = 1)")
            print()

        except Exception as e:
            print(f"  Error refining crossing: {e}")
            print()

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Found {len(crossings)} crossing(s)")

good_ones = [(bpg, r) for bpg, r in crossings if 0.85 < r['alpha2'] < 0.999]
if good_ones:
    print(f"{len(good_ones)} crossing(s) meet criteria (alpha2 in [0.85, 0.999])")
    print()
    for bpg, r in good_ones:
        print(f"USE: Pi=0.05, beta=0.1, BperG={bpg:.7f}")
        print(f"     alpha2 = {r['alpha2']:.6f}")
        print(f"     WNS/W = {100*r['WNS']/r['W_n']:.1f}%")
else:
    print("No crossings meet the criteria.")
    print("Consider adjusting cost function or trying different parameters.")
