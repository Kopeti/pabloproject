"""Ironing demo — a calibration that activates the Region-I ironing branch.

The stock incumbent cost C(a) = 9a^2 + 0.2a never produces a negative
closed-form entry density, so the ironing branch (Appendix "Ironing" of the
draft; `credit_model.iron_region1`) is inactive on all shipped calibrations.
This script runs the one calibration where it has real work to do: the same
cost with its slope shrunk to 15% inside a smooth window [0.15, 0.25]
(`credit_model.plateau_cost_factory`).  C stays strictly increasing, yet the
closed-form density w(alpha) goes negative on an interior interval, and the
equilibrium instead has a no-entry interval [a_L, a_R] with an undepleted pool.

Outputs:
  1. naive-clip vs ironed equilibrium comparison table;
  2. internal consistency checks (zero profit outside, slack inside, frozen
     depletion, flat G_leftover, borrower accounting);
  3. cross-check against the discrete solver (`solve_nested`).  NOTE: the
     discrete w_opt = 0 branch is forward-greedy — it stops entry only where
     the pool quality already overshoots the free-entry bound (letting
     positive profits stand locally) and resumes late, i.e. it approximates
     ironing in state space rather than implementing the N-set construction.
     Expect agreement in aggregates (W, badleftover) but not in the location
     of the no-entry interval;
  4. the mainE_python.run_baseline mirror (config IRONING_demo);
  5. ironing_demo.png — w(alpha) naive vs ironed, and the frozen-pool quality
     against the free-entry bound.

Usage:  python ironing_demo.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import credit_model as cm


def run_analytical(params, ironed=True):
    """solve_nested_analytical with the ironing branch on or off."""
    if ironed:
        return cm.solve_nested_analytical(params)
    saved = cm.iron_region1
    # Passthrough reproduces the pre-fix behavior (naive clip at w >= 0).
    cm.iron_region1 = lambda alphas, K, rp, beta, gt, B0, T, G, B, E, w, da, **kw: {
        'active': False, 'w': w, 'T': T, 'G': G, 'B': B, 'E': E,
        'gamma': np.where(T > 1e-15, G / T, 1.0), 'theta': None,
        'intervals': [], 'diagnostics': []}
    try:
        return cm.solve_nested_analytical(params)
    finally:
        cm.iron_region1 = saved


def main():
    params = cm.Parameters(Pi=0.235, beta=0.5, BperG=1.0)

    # Patch in the plateau cost (established monkey-patch idiom, cf.
    # find_crossing_with_region3.py).
    cfun_p, cfun_prime_p = cm.plateau_cost_factory(lam=0.15, lo=0.15, hi=0.25,
                                                   s=0.02)
    saved_cost = (cm.cfun, cm.cfun_prime_exact)
    cm.cfun = cfun_p
    cm.cfun_prime_exact = cfun_prime_p
    try:
        print("=" * 70)
        print("1. Naive clip vs ironed equilibrium (analytical solver)")
        print("=" * 70)
        naive = run_analytical(params, ironed=False)
        ironed = run_analytical(params, ironed=True)

        rows = [('alpha0', 'alpha0'), ('rp', 'rp'), ('alpha1', 'alpha1'),
                ('alpha2', 'alpha2'), ('r_NS', 'r_NS'),
                ('badleftover', 'badleftover'), ('WNS', 'WNS')]
        print(f"{'':14s}{'naive clip':>14s}{'ironed':>14s}{'diff':>12s}")
        for label, key in rows:
            a, b = naive[key], ironed[key]
            print(f"{label:14s}{a:14.6f}{b:14.6f}{b - a:12.2e}")
        W_naive = naive['W_cumsum_R1'][-1]
        W_iron = ironed['W_cumsum_R1'][-1]
        print(f"{'W(alpha1)':14s}{W_naive:14.6f}{W_iron:14.6f}"
              f"{W_iron - W_naive:12.2e}")
        print(f"\n  no-entry intervals: {ironed['no_entry_intervals']}")
        for d in ironed['ironing_diagnostics']:
            print(f"  a_L={d['a_L']:.4f}  a_R={d['a_R']:.4f}  "
                  f"first w<0 at {d['first_w_neg']:.4f}  "
                  f"state mismatch at a_R (G,B): "
                  f"({d['state_mismatch_at_aR'][0]:.2e}, "
                  f"{d['state_mismatch_at_aR'][1]:.2e})")
        naive_w_neg = (naive['ws_R1'] <= 0) & (naive['alphas_R1'] > naive['alpha0'] + 1e-3) \
                      & (naive['alphas_R1'] < naive['alpha1'] - 1e-3)
        if naive_w_neg.any():
            a = naive['alphas_R1'][naive_w_neg]
            print(f"  naive clip had zeroed w on [{a.min():.4f}, {a.max():.4f}] "
                  "(clipped, pools left inconsistent)")

        print()
        print("=" * 70)
        print("2. Internal consistency of the ironed equilibrium")
        print("=" * 70)
        al = ironed['alphas_R1']
        gam = ironed['gammas_R1']
        w = ironed['ws_R1']
        K = cm.cfun(al) + params.Pi
        resid = gam * (1 + ironed['rp']) - 1 - K
        ok = True
        for (aL, aR) in ironed['no_entry_intervals']:
            inside = (al > aL) & (al < aR)
            outside = ~inside
            checks = [
                ('zero profit outside (|resid|)',
                 np.abs(resid[outside]).max(), 1e-8),
                ('free-entry slack inside (max resid <= tol)',
                 resid[inside].max(), 1e-6),
                ('w == 0 inside', np.abs(w[inside]).max(), 1e-12),
                ('E frozen inside (ptp)', np.ptp(ironed['E_R1'][inside]), 1e-12),
                ('G_leftover flat inside (ptp)',
                 np.ptp(ironed['GLOs_R1'][inside]), 1e-10),
            ]
            print(f"  component ({aL:.4f}, {aR:.4f}):")
            for name, val, tol in checks:
                status = "PASS" if val <= tol else "FAIL"
                ok &= (val <= tol)
                print(f"    {name:44s} {val:10.2e}  {status}")
        print(f"  min w over Region I: {w.min():.2e} (>= 0 required)")
        ok &= (w.min() >= 0)

        print()
        print("=" * 70)
        print("3. Cross-check vs discrete solver (forward-greedy w_opt=0 branch:")
        print("   approximates ironing; compare aggregates, not the interval)")
        print("=" * 70)
        cm.compare_analytical_vs_discrete(params)

        print()
        print("=" * 70)
        print("4. mainE_python.run_baseline mirror (config IRONING_demo)")
        print("=" * 70)
        import mainE_python as me
        me.ACTIVE_CONFIG = 'IRONING_demo'
        me.run_baseline()
        print(f"  mainE ironing_active = {me.g.ironing_active}, "
              f"intervals = {[(round(a, 4), round(b, 4)) for (a, b) in me.g.no_entry_intervals]}")
        for key, cm_val, me_val in [
                ('alpha0', ironed['alpha0'], me.g.alpha0),
                ('rp', ironed['rp'], me.g.rp),
                ('alpha1', ironed['alpha1'], me.g.alpha1),
                ('badleftover', ironed['badleftover'], me.g.badleftover)]:
            print(f"  {key:12s} credit_model={cm_val:.6f}  mainE={me_val:.6f}  "
                  f"diff={me_val - cm_val:.2e}")

        # ---- figure ------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        ax1.plot(naive['alphas_R1'], naive['ws_R1'], 'r--', lw=1.2,
                 label='naive clip')
        ax1.plot(al, w, 'b-', lw=1.4, label='ironed')
        for (aL, aR) in ironed['no_entry_intervals']:
            ax1.axvspan(aL, aR, color='0.9')
        ax1.set_xlabel(r'$\alpha$'); ax1.set_ylabel(r'$w(\alpha)$')
        ax1.set_title('Region-I entry density'); ax1.legend()

        # frozen-pool quality vs bound from the component's left edge
        if ironed['no_entry_intervals']:
            aL, aR = ironed['no_entry_intervals'][0]
            iL = int(np.argmin(np.abs(al - aL)))
            beta = params.beta
            g_t = np.ones_like(al)   # demo priors are uniform
            da = ironed['da_R1']
            cumG = np.zeros_like(al)
            cumG[1:] = np.cumsum((1 - beta) * g_t[:-1]) * da
            G_ir = ironed['G_R1'][iL] + (cumG - cumG[iL])
            B0t = params.BperG * (1 - (1 - beta + al * beta))
            B_ir = ironed['E_R1'][iL] * B0t
            gam_ir = G_ir / (G_ir + B_ir)
            bound = (1 + K) / (1 + ironed['rp'])
            sel = al >= aL - 0.02
            ax2.plot(al[sel], gam_ir[sel], 'b-', lw=1.4,
                     label=r'frozen-pool $\gamma(\alpha;\alpha_L)$')
            ax2.plot(al[sel], bound[sel], 'k--', lw=1.2,
                     label=r'$(1+K)/(1+r_p)$')
            ax2.axvspan(aL, aR, color='0.9')
            ax2.set_xlabel(r'$\alpha$')
            ax2.set_title('No-entry interval: quality vs free-entry bound')
            ax2.legend()
        fig.tight_layout()
        fig.savefig('ironing_demo.png', dpi=150)
        print("\n  Figure saved to ironing_demo.png")

        print("\nOVERALL:", "PASS" if ok else "CHECK FAILURES ABOVE")
    finally:
        cm.cfun, cm.cfun_prime_exact = saved_cost


if __name__ == '__main__':
    main()
