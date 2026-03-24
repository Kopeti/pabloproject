"""
Fine sweep of BperG to find where W_bar^nested vs W_bar^iid flips direction.

With D=1/r:
  - BperG large: nested wins (Region III serves many bad borrowers)
  - BperG small: ???

Also checks D(r) = r^{-eta} for several eta to find cleanest statement.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import credit_model as cm
from credit_model import Parameters, solve_nested_analytical, solve_iid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_pair(beta_v, bpg_v, dfun_fn):
    p = Parameters(Pi=0.05, beta=beta_v, BperG=bpg_v,
                   Delta=0.001, delom=0.0001, a_g=0.0, a_b=0.0)
    cm.dfun = dfun_fn
    cm._current_params = p
    n = solve_nested_analytical(p)
    iid = solve_iid(p)
    W_n = n['W_R2_cumsum'][-1] + n['WNS']
    W_i = iid['W_cumsum'][-1]
    # Also compute borrower counts (D-independent)
    borrowers_n = p.BperG + 1.0          # all served under nested
    borrowers_i = iid['G0'] + iid['B0'] - iid['G_eq'][-1] - iid['B_eq'][-1]
    return W_n, W_i, borrowers_n, borrowers_i, n, iid

# ============================================================
# 1. Fine sweep of BperG at several fixed beta values, D=1/r
# ============================================================
bpg_grid = np.concatenate([
    np.linspace(0.02, 0.15, 8),
    np.linspace(0.15, 0.5, 10),
    np.linspace(0.5, 2.0, 10),
    np.linspace(2.0, 5.0, 5),
])
bpg_grid = np.unique(bpg_grid)

dfun_1_r = lambda r: 1.0 / r

print("=" * 80)
print("FINE SWEEP: BperG  (D = 1/r)")
print("=" * 80)

for beta_v in [0.1, 0.3, 0.5, 0.7]:
    print(f"\nbeta = {beta_v}")
    print(f"  {'BperG':>7}  {'W_nested':>10}  {'W_iid':>10}  {'ratio N/I':>10}  {'N>I?':>6}  "
          f"{'WNS/W_n':>8}  {'alpha0':>7}  {'rp':>7}")
    print("  " + "-" * 72)

    W_ns, W_is, ratios, bpg_vals = [], [], [], []
    for bpg in bpg_grid:
        try:
            W_n, W_i, bn, bi, n_sol, iid_sol = run_pair(beta_v, bpg, dfun_1_r)
            ratio = W_n / W_i if W_i > 1e-12 else float('inf')
            wns_frac = n_sol['WNS'] / W_n if W_n > 1e-12 else 0
            W_ns.append(W_n); W_is.append(W_i); ratios.append(ratio); bpg_vals.append(bpg)
            flag = "TRUE " if W_n > W_i else "FALSE"
            print(f"  {bpg:>7.3f}  {W_n:>10.4f}  {W_i:>10.4f}  {ratio:>10.4f}  {flag:>6}  "
                  f"{wns_frac:>8.3f}  {n_sol['alpha0']:>7.4f}  {n_sol['rp']:>7.4f}")
        except Exception as e:
            print(f"  {bpg:>7.3f}  ERROR: {e}")

# ============================================================
# 2. At beta=0.3, check limiting behavior as BperG -> 0
# ============================================================
print("\n\n" + "=" * 80)
print("LIMITING BEHAVIOR: BperG -> 0  (beta=0.3, D=1/r)")
print("=" * 80)
print(f"  {'BperG':>8}  {'W_nested':>10}  {'W_iid':>10}  {'ratio':>10}  {'WNS':>10}  {'badleft':>10}")
for bpg in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]:
    try:
        W_n, W_i, _, _, n_sol, _ = run_pair(0.3, bpg, dfun_1_r)
        ratio = W_n / W_i if W_i > 1e-12 else float('inf')
        print(f"  {bpg:>8.4f}  {W_n:>10.4f}  {W_i:>10.4f}  {ratio:>10.4f}  "
              f"{n_sol['WNS']:>10.4f}  {n_sol['badleftover']:>10.5f}")
    except Exception as e:
        print(f"  {bpg:>8.4f}  ERROR: {e}")

# ============================================================
# 3. At beta=0.3, vary D elasticity eta (D = r^{-eta}) for BperG=0.2
# ============================================================
print("\n\n" + "=" * 80)
print("VARY D ELASTICITY: D(r) = r^{-eta}, beta=0.3, BperG=0.2")
print("=" * 80)
print(f"  {'eta':>6}  {'W_nested':>10}  {'W_iid':>10}  {'ratio':>10}  {'N>I?':>6}")
for eta in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]:
    try:
        fn = (lambda e: lambda r: r**(-e))(eta)
        W_n, W_i, _, _, _, _ = run_pair(0.3, 0.2, fn)
        ratio = W_n / W_i if W_i > 1e-12 else float('inf')
        flag = "TRUE " if W_n > W_i else "FALSE"
        print(f"  {eta:>6.2f}  {W_n:>10.4f}  {W_i:>10.4f}  {ratio:>10.4f}  {flag:>6}")
    except Exception as e:
        print(f"  {eta:>6.2f}  ERROR: {e}")

# ============================================================
# 4. Find crossing BperG* for each (beta, eta) pair
# ============================================================
print("\n\n" + "=" * 80)
print("CROSSING POINTS BperG* where W_nested = W_iid  (for various beta and eta)")
print("=" * 80)
print(f"  {'beta':>5}  {'eta':>5}  {'BperG*':>8}  {'direction at 0':>16}  {'direction at inf':>16}")

from scipy.optimize import brentq

for beta_v in [0.1, 0.3, 0.5, 0.7]:
    for eta in [0.5, 1.0, 2.0]:
        fn = (lambda e: lambda r: max(r, 1e-10)**(-e))(eta)
        def diff(bpg):
            try:
                W_n, W_i, _, _, _, _ = run_pair(beta_v, bpg, fn)
                return W_n - W_i
            except:
                return float('nan')

        # Check direction at extremes
        d_lo = diff(0.02)
        d_hi = diff(3.0)
        dir_lo = "nested>iid" if d_lo > 0 else "iid>nested"
        dir_hi = "nested>iid" if d_hi > 0 else "iid>nested"

        # Find crossing if it exists
        if d_lo * d_hi < 0:
            try:
                bpg_star = brentq(diff, 0.02, 3.0, xtol=1e-4)
                print(f"  {beta_v:>5.1f}  {eta:>5.1f}  {bpg_star:>8.4f}  {dir_lo:>16}  {dir_hi:>16}")
            except Exception as e:
                print(f"  {beta_v:>5.1f}  {eta:>5.1f}  CROSSING FIND FAILED: {e}")
        else:
            print(f"  {beta_v:>5.1f}  {eta:>5.1f}  {'NO CROSSING':>8}  {dir_lo:>16}  {dir_hi:>16}")

# ============================================================
# 5. Plot W_nested and W_iid vs BperG for beta=0.3, D=1/r
# ============================================================
bpg_plot = np.concatenate([np.linspace(0.02, 0.5, 25), np.linspace(0.5, 3.0, 20)])
bpg_plot = np.unique(bpg_plot)
W_ns_plot, W_is_plot = [], []
for bpg in bpg_plot:
    try:
        W_n, W_i, _, _, _, _ = run_pair(0.3, bpg, dfun_1_r)
        W_ns_plot.append(W_n); W_is_plot.append(W_i)
    except:
        W_ns_plot.append(np.nan); W_is_plot.append(np.nan)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(bpg_plot, W_ns_plot, 'b-o', ms=4, label=r'$\bar{W}^{\rm nested}$')
ax1.plot(bpg_plot, W_is_plot, 'r-s', ms=4, label=r'$\bar{W}^{\rm iid}$')
ax1.set_xlabel('BperG'); ax1.set_ylabel('Total capital W')
ax1.set_title(r'$\beta=0.3$, $D(r)=1/r$')
ax1.legend(); ax1.grid(alpha=0.3)
ax1.set_xlim([0, 3])

ratio_plot = [n/i if i > 1e-12 else np.nan for n, i in zip(W_ns_plot, W_is_plot)]
ax2.plot(bpg_plot, ratio_plot, 'k-', lw=2)
ax2.axhline(1.0, color='gray', ls='--', alpha=0.6)
ax2.set_xlabel('BperG'); ax2.set_ylabel(r'$\bar{W}^{\rm nested}/\bar{W}^{\rm iid}$')
ax2.set_title(r'Ratio $\bar{W}^{\rm nested}/\bar{W}^{\rm iid}$, $\beta=0.3$, $D(r)=1/r$')
ax2.grid(alpha=0.3); ax2.set_xlim([0, 3])

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'sweep_BperG.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to {out}")