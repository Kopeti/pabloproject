"""
Compare baseline equilibrium across three solvers:
  1. mainE_python.run_baseline()          — discrete iterative (older algorithm)
  2. credit_model.solve_nested()          — discrete (newer, cleaner)
  3. credit_model.solve_nested_analytical() — analytical closed-form

All run with the same parameters (credit_model defaults:
  Pi=0.05, beta=0.1, BperG=0.2197, c(α) = 0.2α² + 1.0α).
"""

import numpy as np
import sys
import os

# Ensure this script can import siblings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mainE_python as me
import credit_model as cm


def run_comparison():
    # ------------------------------------------------------------------
    # 1. Run mainE_python with 'credit_model' parameterization
    # ------------------------------------------------------------------
    me.ACTIVE_CONFIG = 'credit_model'
    me.run_baseline()

    mainE = {
        'alpha0': me.g.alpha0,
        'alpha1': me.g.alpha1,
        'alpha2': me.g.alpha2,
        'rp':     me.g.rp,
        'r_NS':   me.g.alpha2 and (float(me.cfun(me.g.alpha2)) + me.g.Pi),
        'WNS':    me.g.WNS,
        'badleftover': me.g.badleftover,
        'W_total': me.g.W_total if hasattr(me.g, 'W_total') else (me.g.W[-1] if hasattr(me.g, 'W') and len(me.g.W) > 0 else np.nan),
    }

    # ------------------------------------------------------------------
    # 2. Run credit_model discrete solver
    # ------------------------------------------------------------------
    params = cm.Parameters()  # uses defaults: Pi=0.05, beta=0.1, BperG=0.2197
    print("\n" + "=" * 60)
    print("Running credit_model.solve_nested (discrete)")
    print("=" * 60)
    nested_disc = cm.solve_nested(params)

    cm_disc = {
        'alpha0': nested_disc['alpha0'],
        'alpha1': nested_disc['alpha1'],
        'alpha2': nested_disc['alpha2'],
        'rp':     nested_disc['rp'],
        'r_NS':   nested_disc['r_NS'],
        'WNS':    nested_disc['WNS'],
        'badleftover': nested_disc['badleftover'],
        'W_total': nested_disc['W_R2_cumsum'][-1] + nested_disc['WNS'],
    }

    # ------------------------------------------------------------------
    # 3. Run credit_model analytical solver
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Running credit_model.solve_nested_analytical")
    print("=" * 60)
    nested_ana = cm.solve_nested_analytical(params)

    cm_ana = {
        'alpha0': nested_ana['alpha0'],
        'alpha1': nested_ana['alpha1'],
        'alpha2': nested_ana['alpha2'],
        'rp':     nested_ana['rp'],
        'r_NS':   nested_ana['r_NS'],
        'WNS':    nested_ana['WNS'],
        'badleftover': nested_ana['badleftover'],
        'W_total': nested_ana['W_R2_cumsum'][-1] + nested_ana['WNS'],
    }

    # ------------------------------------------------------------------
    # 4. Print comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BASELINE EQUILIBRIUM — SIDE-BY-SIDE COMPARISON")
    print(f"Config: Pi={params.Pi}, beta={params.beta}, BperG={params.BperG}")
    print(f"Cost:   c(a) = 0.2*a^2 + 1.0*a")
    print("=" * 70)

    header = f"{'Variable':<16} {'mainE(disc)':>14} {'cm(disc)':>14} {'cm(analyt)':>14} {'max diff':>12}"
    print(header)
    print("-" * len(header))

    keys = ['alpha0', 'alpha1', 'alpha2', 'rp', 'r_NS', 'WNS', 'badleftover', 'W_total']
    all_close = True
    for k in keys:
        v1, v2, v3 = mainE[k], cm_disc[k], cm_ana[k]
        vals = [v for v in [v1, v2, v3] if not np.isnan(v)]
        max_diff = max(vals) - min(vals) if vals else np.nan
        flag = "" if max_diff < 1e-3 else " <<<"
        if max_diff >= 1e-3:
            all_close = False
        print(f"{k:<16} {v1:>14.6f} {v2:>14.6f} {v3:>14.6f} {max_diff:>12.2e}{flag}")

    print("-" * len(header))
    if all_close:
        print("ALL MATCH (within 1e-3)")
    else:
        print("DIFFERENCES FOUND (marked with <<<)")

    # ------------------------------------------------------------------
    # 5. Breakdown of W by region
    # ------------------------------------------------------------------
    print("\n--- W breakdown by region ---")

    # mainE: W is cumsum of [WNS, wmass..., wcim...]
    W_R1_mainE = np.sum(me.g.wmass) if len(me.g.wmass) > 0 else 0
    cim_a = np.linspace(me.g.alpha1, me.g.alpha2, 100)
    W_R2_mainE = np.sum(me.wcim(cim_a))
    print(f"mainE:    W_NS={me.g.WNS:.6f}  W_R1={W_R1_mainE:.6f}  W_R2={W_R2_mainE:.6f}  total={me.g.WNS + W_R1_mainE + W_R2_mainE:.6f}")

    # cm discrete
    W_R1_cmd = np.sum(nested_disc['ws_R1']) * nested_disc['alphas_R1'][1] - nested_disc['alphas_R1'][0] if len(nested_disc['alphas_R1']) > 1 else 0
    # Actually W_R2_cumsum already has R1+R2 cumulative
    W_R1R2_cmd = nested_disc['W_R2_cumsum'][-1]
    print(f"cm(disc): W_NS={nested_disc['WNS']:.6f}  W_R1+R2={W_R1R2_cmd:.6f}  total={nested_disc['WNS'] + W_R1R2_cmd:.6f}")

    # cm analytical
    W_R1_cma = nested_ana['W_cumsum_R1'][-1]
    W_R2_cma = nested_ana['W_R2_cumsum'][-1] - nested_ana['W_cumsum_R1'][-1]
    print(f"cm(ana):  W_NS={nested_ana['WNS']:.6f}  W_R1={W_R1_cma:.6f}  W_R2={W_R2_cma:.6f}  total={nested_ana['WNS'] + nested_ana['W_R2_cumsum'][-1]:.6f}")

    return mainE, cm_disc, cm_ana


if __name__ == '__main__':
    run_comparison()
