"""Parallel-shift demo: the entry equilibrium is NOT the shifted baseline.

Corollary "Entry Equilibrium with Homogeneous Technology" claims that when
entrants differ from incumbents only by a parallel downward shift of the total
cost of entry, K^E = K - Delta, the Entry Equilibrium coincides with the
*baseline* equilibrium computed under K^E.  This script shows that it does not,
and why.

Three equilibria are solved and overlaid:

  (1) baseline under K            -- credit_model.solve_nested_analytical
  (2) entry equilibrium, K^E=K-D  -- mainE_python.solve_for_config
  (3) baseline under K^E          -- credit_model.solve_nested_analytical

The coincidence (2) = (3) would require the K^E-baseline to want at least as
much capital as the incumbents already hold, at every precision -- incumbent
capital is sunk and cannot exit.  That fails in a boundary layer just below
alpha_1: as alpha -> alpha_1 the acceptance margin B(alpha) vanishes
quadratically, so the baseline pooling density tends to 2*D(r_p)*g*(1-beta),
*twice* the slice-clearing level, while above its own (lower) alpha_1 the
shifted baseline is already in its CIM region at the slice-clearing density.
Dominance at alpha_1 would need D(r_p - Delta) >= 2*D(r_p), i.e. Delta >= r_p/2
-- far larger than any sensible shift.  So entrants stay out on that interval,
the stranded incumbent capital prices those markets by cash-in-the-market
clearing, and the two equilibria part company there and downstream.

This script does NOT modify credit_model.py or mainE_python.py.  It drives them
from outside via the idioms they already support (monkey-patching the cost
functions of credit_model, as search_cost_functions.py does; injecting a config
into mainE_python.PARAM_CONFIGS, as that module does to itself for IRONING_demo).

Usage:  python parallel_shift_demo.py
Output: parallel_shift_demo.png + a console comparison table.
Note:   mainE_python.run_mainE always rewrites mainE_results.png next to the
        module; restore it with `git checkout -- mainE_results.png` afterwards.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad

import credit_model as cm
import mainE_python as me

# ---------------------------------------------------------------- parameters
PI      = 0.235          # incumbent fixed cost of entry K(0)
SHIFT   = 0.10           # parallel downward shift:  K^E = K - SHIFT
PIE     = PI - SHIFT
BETA    = 0.5
BPERG   = 1.0
C2, C1  = 9.0, 0.2       # C(alpha) = C2 alpha^2 + C1 alpha  (shared by both)

# numpy-safe, array-in/array-out (required by both modules)
C       = lambda a: C2 * np.asarray(a, dtype=float)**2 + C1 * np.asarray(a, dtype=float)
C_prime = lambda a: 2 * C2 * np.asarray(a, dtype=float) + C1


# ------------------------------------------------------------------ helpers
def solve_baseline(Pi):
    """Baseline equilibrium under K(alpha) = Pi + C(alpha) (credit_model)."""
    cm.cfun = C
    cm.cfun_prime_exact = C_prime
    cm.cfun_prime2_exact = lambda a: 2 * C2 * np.ones_like(np.asarray(a, dtype=float))
    cm.dfun = lambda r: 1.0 / np.asarray(r, dtype=float)
    p = cm.Parameters(Pi=Pi, beta=BETA, BperG=BPERG, a_g=0, a_b=0)
    cm._current_params = p
    res = cm.solve_nested_analytical(p)
    res['Pi'] = Pi
    return res


def baseline_curves(res, grid):
    """(r, gamma, w, W_cumulative) of a baseline equilibrium on `grid`.

    W is cumulative *selective* capital (Regions I and II); the non-selective
    atom sits at alpha = 0 and is reported separately.  NaN outside the support
    so the curves break rather than interpolate across empty ranges.
    """
    a0, a1, a2, Pi = res['alpha0'], res['alpha1'], res['alpha2'], res['Pi']
    r = np.full_like(grid, np.nan)
    gam = np.full_like(grid, np.nan)
    w = np.full_like(grid, np.nan)

    inI = (grid >= a0) & (grid <= a1)
    r[inI] = res['rp']
    gam[inI] = np.interp(grid[inI], res['alphas_R1'], res['gammas_R1'])
    w[inI] = np.interp(grid[inI], res['alphas_R1'], res['ws_R1'])

    inII = (grid > a1) & (grid <= a2)
    r[inII] = C(grid[inII]) + Pi
    gam[inII] = 1.0
    w[inII] = np.interp(grid[inII], res['alphas_R2'], res['ws_R2'])

    # cumulative selective capital: Region I then Region II (already continuous)
    al_c = np.concatenate([res['alphas_R1'], res['alphas_R2']])
    W_c = np.concatenate([res['W_cumsum_R1'], res['W_R2_cumsum']])
    W = np.interp(grid, al_c, W_c, left=0.0, right=W_c[-1])

    gamma_NS = (res['goodleftover_alpha2']
                / (res['goodleftover_alpha2'] + res['badleftover']))
    return dict(r=r, gamma=gam, w=w, W=W,
                atom_r=res['r_NS'], atom_W=res['WNS'], atom_gamma=gamma_NS)


def entry_curves(res, grid, base_K):
    """(r, gamma, w, W_cumulative) of the entry equilibrium on `grid`.

    Densities and quality in the pooling range come from the solver's own
    analytical Region-I object (me.g.entry_analytical); the realized total
    density is max(w_total, w_incumbent) -- on suspended stretches the
    zero-profit path asks for less than the incumbents already supply, so the
    incumbents' sunk capital is what is actually there.
    """
    a0E, a1E, a2E = res['alpha0E'], res['alpha1E'], res['alpha2E']
    a0, a1, a2 = res['alpha0'], res['alpha1'], res['alpha2']

    # --- rate: the module's own post-entry schedule (ACTIVE_CONFIG still set)
    r = me.rfunE(grid, max(a0, a0E), a1E, a2E)
    r = np.where(grid > max(a2E, a2), np.nan, r)   # NS lenders live at alpha=0

    ea = me.g.entry_analytical
    gam = np.full_like(grid, np.nan)
    w = np.full_like(grid, np.nan)

    if ea is not None:
        w_real = np.maximum(ea['w_total'], ea['w_incumbent'])
        inI = (grid >= ea['alphas'][0]) & (grid <= ea['alphas'][-1])
        w[inI] = np.interp(grid[inI], ea['alphas'], w_real)
        gam[inI] = np.interp(grid[inI], ea['alphas'], ea['gammaE'])

    # CIM region of the entry equilibrium: total capital clears each slice at r^E
    inII = (grid > a1E) & (grid <= max(a2E, a2))
    w[inII] = (1.0 / r[inII]) * (1 - BETA)      # D(r) * g(omega_g) * (1-beta), g == 1
    gam[inII] = 1.0

    # --- cumulative selective capital = incumbent (sunk, unchanged) + entrant
    W_inc = np.interp(grid,
                      np.concatenate([base_K['alphas_R1'], base_K['alphas_R2']]),
                      np.concatenate([base_K['W_cumsum_R1'], base_K['W_R2_cumsum']]),
                      left=0.0, right=base_K['W_R2_cumsum'][-1])
    # WE = cumsum([pooling masses, CIM masses]); its i-th entry is the total
    # entered by the RIGHT endpoint of step i, i.e. at almassE[i+1] -- and the
    # curve must start from 0 at alpha_0^E, so prepend that point explicitly.
    almassE, WE = res['almassE'], res['WE']
    n_pool = len(WE) - 100 if len(WE) > 100 else len(WE)
    n_pool = min(n_pool, len(almassE) - 1)
    x_E = np.concatenate([[almassE[0]],
                          almassE[1:n_pool + 1],
                          np.linspace(a1E, a2E, len(WE) - n_pool)])
    W_E = np.concatenate([[0.0], WE])
    n = min(len(x_E), len(W_E))
    order = np.argsort(x_E[:n], kind='stable')
    W_ent = np.interp(grid, x_E[:n][order], W_E[:n][order],
                      left=0.0, right=W_E[n - 1])
    return dict(r=r, gamma=gam, w=w, W=W_inc + W_ent,
                atom_r=res['rnsE'], atom_W=np.nan, atom_gamma=np.nan)


# --------------------------------------------------------------------- main
def main():
    grid = np.linspace(0.0, 1.0, 2000)

    print("=" * 74)
    print("PARALLEL-SHIFT DEMO   K^E(a) = K(a) - %.2f   (Pi %.3f -> %.3f)"
          % (SHIFT, PI, PIE))
    print("=" * 74)

    # ---- (2) entry equilibrium FIRST: ACTIVE_CONFIG is global and sticky, so
    #      everything that depends on it is read out before anything else runs.
    me.PARAM_CONFIGS['PARALLEL_SHIFT'] = {
        'description': 'Parallel downward shift K^E = K - %.2f' % SHIFT,
        'Pi': PI, 'beta': BETA, 'BperG': BPERG,
        'cfun': C,
        'has_entry': True,
        'PiE': PIE,
        'cfunE_kind': 'polynomial',
        'cfunE_params': {'coeffs': C},      # array-in/array-out: required
    }
    print("\n--- (2) entry equilibrium ---")
    res_E = me.solve_for_config('PARALLEL_SHIFT')
    ea = me.g.entry_analytical
    susp = ea.get('suspended_intervals', []) if ea is not None else []
    susp_diag = ea.get('suspension_diags', []) if ea is not None else []
    inc_from_mainE = dict(alpha0=res_E['alpha0'], alpha1=res_E['alpha1'],
                          alpha2=res_E['alpha2'], rp=res_E['rp'])

    # ---- (1) and (3): the two baselines
    print("\n--- (1) baseline under K, (3) baseline under K^E ---")
    base_K = solve_baseline(PI)
    base_KE = solve_baseline(PIE)

    # cross-check: mainE's own incumbent solve must equal curve (1)
    print("\nCross-check (mainE incumbent vs credit_model baseline under K):")
    for k in ('alpha0', 'alpha1', 'alpha2', 'rp'):
        d = abs(inc_from_mainE[k] - base_K[k])
        print("   %-8s mainE=%.6f  credit_model=%.6f  diff=%.2e  %s"
              % (k, inc_from_mainE[k], base_K[k], d, "OK" if d < 1e-6 else "MISMATCH"))

    c1 = baseline_curves(base_K, grid)
    c3 = baseline_curves(base_KE, grid)
    c2 = entry_curves(res_E, grid, base_K)

    # ---------------------------------------------------------------- table
    print("\n" + "=" * 74)
    print("Equilibrium comparison")
    print("=" * 74)
    print("%-22s %14s %14s %14s" % ("", "(1) base K", "(2) entry", "(3) base K^E"))
    rows = [
        ("alpha_0", base_K['alpha0'], res_E['alpha0E'], base_KE['alpha0']),
        ("alpha_1", base_K['alpha1'], res_E['alpha1E'], base_KE['alpha1']),
        ("alpha_2", base_K['alpha2'], res_E['alpha2E'], base_KE['alpha2']),
        ("pooling rate r_p", base_K['rp'], res_E['rpE'], base_KE['rp']),
        ("non-selective rate", base_K['r_NS'], res_E['rnsE'], base_KE['r_NS']),
        ("selective capital", base_K['W_R2_cumsum'][-1], c2['W'][-1],
         base_KE['W_R2_cumsum'][-1]),
    ]
    for name, v1, v2, v3 in rows:
        print("%-22s %14.6f %14.6f %14.6f" % (name, v1, v2, v3))
    print("%-22s %14.6f %14s %14.6f"
          % ("NS atom (alpha=0)", base_K['WNS'], "n/a", base_KE['WNS']))

    print("\nEntry solver Step-6 suspended stretches (no entrants there):")
    if susp:
        for iv, d in zip(susp, susp_diag):
            print("   [%.4f, %.4f]   max entrant-profit residual inside: %.3e"
                  % (iv[0], iv[1], d['max_entrant_profit_resid']))
    else:
        print("   (none reported)")

    # ------------------------------------------------- the decisive comparison
    lo, hi = base_KE['alpha1'], base_K['alpha1']
    print("\n" + "=" * 74)
    print("Do (2) and (3) coincide?   Focus interval [alpha_1(K^E), alpha_1(K)]"
          " = [%.4f, %.4f]" % (lo, hi))
    print("=" * 74)
    band = (grid >= lo) & (grid <= hi)
    for name in ('r', 'gamma', 'w', 'W'):
        d = np.abs(c2[name] - c3[name])
        ok = np.isfinite(d)
        d_band = d[band & ok]
        print("   %-6s  max|2-3| overall = %10.4f   on the interval: max = %8.4f"
              "   mean = %8.4f" % (name, np.nanmax(d[ok]) if ok.any() else np.nan,
                                   d_band.max() if d_band.size else np.nan,
                                   d_band.mean() if d_band.size else np.nan))
    KE_of = lambda a: C(a) + PIE
    below = np.isfinite(c2['r']) & band & (c2['r'] < KE_of(grid) - 1e-9)
    print("\n   entry-equilibrium rate strictly below K^E(alpha) on the interval "
          "at %d of %d grid points" % (below.sum(), band.sum()))
    print("   -> capital clearing on stranded incumbent capital, not K^E pricing")

    # --------------------------------------------------------------- figure
    styles = [(c1, 'baseline under $K$', 'tab:green', '--'),
              (c2, 'entry equilibrium ($K^E=K-%.2f$)' % SHIFT, 'tab:blue', '-'),
              (c3, 'baseline under $K^E$', 'tab:red', ':')]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))
    panels = [('r', r'$r(\alpha)$', 'interest rate'),
              ('gamma', r'$\gamma(\alpha)$', 'pool quality'),
              ('W', r'$W(\alpha)$', 'cumulative selective capital'),
              ('w', r'$w(\alpha)$', 'density of lenders')]

    for ax, (key, ylab, title) in zip(axes.ravel(), panels):
        for c, label, color, ls in styles:
            ax.plot(grid, c[key], color=color, linestyle=ls, lw=1.8, label=label)
            if key == 'r' and np.isfinite(c['atom_r']):
                ax.plot(0.0, c['atom_r'], color=color, marker='o', ms=7,
                        linestyle='')
        ax.axvspan(lo, hi, color='0.85', zorder=0)
        ax.axvline(base_K['alpha1'], color='0.4', lw=0.7, ls=':')
        ax.set_xlabel(r'$\alpha$'); ax.set_ylabel(ylab)
        ax.set_title(title); ax.grid(alpha=0.3)
        ax.set_xlim(-0.02, 1.0)

    axes[1, 1].set_ylim(0, 3)      # density: clip the alpha_0 spike
    axes[0, 0].legend(fontsize=8, loc='upper left')
    axes[0, 0].text(0.5 * (lo + hi), axes[0, 0].get_ylim()[1] * 0.55,
                    'no entry\nhere', ha='center', fontsize=8, color='0.3')
    fig.suptitle('Entry equilibrium vs. the shifted baseline: a parallel shift '
                 r'$K^E = K - %.2f$ does not reproduce the $K^E$ baseline' % SHIFT)
    fig.tight_layout()
    fig.savefig('parallel_shift_demo.png', dpi=150)
    print("\nFigure saved to parallel_shift_demo.png")
    print("(remember: mainE_results.png was overwritten by the entry solve;"
          " restore it with git)")


if __name__ == '__main__':
    main()
