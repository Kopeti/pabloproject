"""Clearing cutoff in the post-entry pooling region: a read-only diagnostic.

Background (Notes/Entry_equilibrium_proof_explained.tex, revision of Sept 2026).
The entry construction of Steps 5-6 lets entrants follow their zero-profit path
as far as it goes.  The incumbents above the last entrant then either run out
of borrowers before alpha_1 (part of their capital is idle, and they would
rather undercut the pooling rate) or leave a residual of recognizable borrowers
at alpha_1 (and the top incumbent would rather raise her rate).  Equilibrium
needs neither: entry must stop at the cutoff alpha* from which the incumbents'
capital exhausts the acceptable pool exactly at alpha_1.

For each appendix example this script
  * solves the model with the existing solver (nothing in it is changed),
  * walks the incumbent-only continuation of Step 6 from every candidate cutoff
    WITHOUT clipping the pool at zero, and records where the pool is exhausted,
  * locates alpha*, checks free entry above it, and
  * re-solves the non-selective margin (Step 7) and the maintained conditions
    with entry truncated at alpha*.

It saves no figure and writes no file.

Usage:  python clearing_cutoff_check.py [--n GRID]

GRID is the number of points of the pooling-region grid on which Steps 5-6 are
re-solved here (default 32000; the solver's own default is 500).  The fine grid
matters: where entry is concentrated on a very narrow interval, as in the
'advantage at alpha = 0' example (entrant density about 30), one cell of the
500-point grid carries eight times the capital that separates 'a residual is
left' from 'the pool is exhausted early', and the coarse grid wrongly reports
that no clearing cutoff exists.

The marginal entrant is located exactly here.  Step 2 of the proof defines
alpha_0^E as the minimiser of the entrants' break-even rate.  Below alpha_0 the
solver takes that minimum on a grid of 60 points (spacing 0.0024), which is
wider than the whole interval on which entrants are active in the 'advantage
at alpha = 0' example (0.0009).  r_p^E is hardly affected (the error is second
order at a minimum), but the entrants' active range then starts at the wrong
place, and with a density of 30 the capital lent on it, hence the leftover
bads, is off by more than the effect the example is meant to show.  The patch
below replaces the grid minimum by a bounded scalar minimisation, in memory
only; pass --grid-a0E to switch it off.

Since 2026-09-29 the solver itself locates the marginal entrant exactly, uses
the 32000-point grid and stops entry at the clearing cutoff (Step 5(c) of the
proof).  This script re-solves Steps 4-5 with clearing_cutoff=False, so that
it still sees the path before the cutoff and can locate the cutoff itself.
"""
import contextlib
import io
import os

os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

# The solver saves diagnostic figures as a side effect; switch that off here.
plt.savefig = lambda *a, **k: None
plt.show = lambda *a, **k: None
matplotlib.figure.Figure.savefig = lambda *a, **k: None

import mainE_python as m
with contextlib.redirect_stdout(io.StringIO()):
    import make_figure11      # registers FIG11_parallel_shift
    import make_figure10ab    # registers FIG10a_pool_improve, FIG10b_pool_worsen

_gap_on_grid = m._entry_breakeven_gap
_low_min_cache = {}


def _breakeven_low(a):
    """Entrant's break-even rate on the fresh pool (no incumbent lends below alpha_0)."""
    return (1.0 + m.g.PiE + m._scalar(m.cfunE(a))) / m._scalar(m.gam0(a)) - 1.0


def _entry_breakeven_gap_exact(r):
    """m._entry_breakeven_gap with the minimum below alpha_0 taken exactly."""
    f, a = _gap_on_grid(r)
    key = (m.ACTIVE_CONFIG, m.g.PiE, m.g.alpha0)
    if key not in _low_min_cache:
        al = m.g._al_low
        be = np.array([_breakeven_low(x) for x in al])
        k = int(np.argmin(be))
        res = minimize_scalar(_breakeven_low, method='bounded', options={'xatol': 1e-12},
                              bounds=(al[max(k - 1, 0)], al[min(k + 1, len(al) - 1)]))
        _low_min_cache[key] = (float(res.fun), float(res.x)) if res.fun <= be[k] else (float(be[k]), float(al[k]))
    be_min, a_min = _low_min_cache[key]
    if be_min - r <= f + 1e-12:
        return be_min - r, a_min
    return f, a


EXAMPLES = [
    ('FIG11_parallel_shift', 'parallel shift'),
    ('FIG8_bigdata', 'big data (advantage at high alpha)'),
    ('FIG10a_pool_improve', 'advantage at alpha = 0'),
    ('FIG10b_pool_worsen', 'advantage at intermediate alpha'),
]


def continuation(i, GE, BE, w_inc, B0t, gt, bt, Greq, beta, D, da):
    """Incumbent-only continuation of Step 6 from cutoff index i, not clipped.

    Returns where the acceptable pool is first exhausted (index or None), the
    residual pool at the top of the grid, the incumbent capital left idle above
    the exhaustion point, the largest entrant margin gamma - (1+K^E)/(1+r_p^E)
    on the stretch where an entrant could break even, and the (G, B) paths.
    """
    n = len(Greq)
    G, B = GE[i], BE[i]
    exhausted_at, idle, margin = None, 0.0, -np.inf
    Gp, Bp = np.zeros(n), np.zeros(n)
    for k in range(i + 1, n):
        T = G + B
        inflow = (1 - beta) * gt[k] * da
        if exhausted_at is None:
            th = w_inc[k] / (D * T) if T > 1e-15 else np.inf
            if np.isfinite(th):
                E = B / max(B0t[k], 1e-15)
                G_new = (1 - th * da) * G + inflow
                B_new = (1 - th * da) * B - beta * bt[k] * E * da
            else:
                G_new, B_new = -1.0, 0.0
            if G_new < 0.0 or G_new + max(B_new, 0.0) <= 0.0:
                exhausted_at = k
                idle += max(w_inc[k] * da - D * (T + inflow), 0.0)
                G, B = 0.0, 0.0
            else:
                G, B = G_new, max(B_new, 0.0)
        else:
            idle += max(w_inc[k] * da - D * inflow, 0.0)
            G, B = 0.0, 0.0
        Gp[k], Bp[k] = G, B
        if exhausted_at is None and Greq[k] < 1.0:
            T_k = G + B
            margin = max(margin, (G / T_k if T_k > 1e-15 else 1.0) - Greq[k])
    return dict(exhausted_at=exhausted_at, resid_G=G, resid_B=B, idle=idle,
                margin=margin, G=Gp, B=Bp)


def _margin_and_checks():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ns = m._solve_ns_margin()
        ok = m._verification_checks(ns)
    fails = [ln.strip() for ln in buf.getvalue().splitlines()
             if (ln.strip().startswith('[V') and 'FAIL' in ln) or ln.strip().startswith('[branch]')]
    return ns, ok, fails


def check(name, label, n_grid):
    with contextlib.redirect_stdout(io.StringIO()):
        m.solve_for_config(name)
    g = m.g
    if g.entry_analytical is not None and n_grid:
        # re-solve Steps 5-6 on the fine grid; everything downstream uses it
        with contextlib.redirect_stdout(io.StringIO()):
            ea_fine = m.solve_entry_pooling_analytical(
                g.rpE, g.alpha0E, g.alpha1E, n_pts=n_grid,
                cfunE_poly_info=getattr(g, 'cfunE_poly_info', None),
                clearing_cutoff=False)
            g.entry_analytical = ea_fine
            g.badleftoverE = m._leftover_bads_after_pooling(ea_fine, g.rpE)
    gam0_a1 = m._scalar(m.gam0(g.alpha1))
    share = m._scalar(m.dfun(g.rpE)) / (2.0 * m._scalar(m.dfun(g.rp)))
    print(f'\n=== {label}  [{name}]')
    print(f'  alpha0={g.alpha0:.4f}  alpha1={g.alpha1:.4f}  r_p={g.rp:.4f}  r_p^E={g.rpE:.4f}  '
          f'alpha0^E={g.alpha0E:.4f}')
    print(f'  share of capital lent by a rationed incumbent near alpha1, D(r_p^E)/(2D(r_p)) = {share:.3f}; '
          f'gamma_0(alpha1) = {gam0_a1:.3f}  ->  a rationed incumbent '
          f'{"undercuts" if share < gam0_a1 else "stays"}')
    ns0, ok0, fails0 = _margin_and_checks()
    print(f'  Steps 5-6 as they stand: case {ns0["case"]}, R={ns0["R"]:.5f}, r_NS^E={ns0["rnsE"]:.5f}, '
          f'B^NS,E={g.badleftoverE:.6f} (baseline {g.badleftover:.6f}), conditions ok={ok0} {fails0 or ""}')

    ea = g.entry_analytical
    if ea is None or not np.any(ea['wE'] > 0):
        print('  no entrant capital in the pooling region: baseline exhaustion at alpha1, alpha* is void')
        return
    al, da, beta, rpE = ea['alphas'], ea['da'], g.beta, g.rpE
    D = m._scalar(m.dfun(rpE))
    og, ob = beta + al * (1 - beta), 1 - beta + al * beta
    gt = np.array([m.gpriorfun_scalar(x) for x in og])
    bt = np.array([m.bpriorfun_scalar(x) for x in ob])
    B0t = np.array([quad(m.bpriorfun_scalar, x, 1)[0] for x in ob])
    Greq = (1 + ea['KE']) / (1 + rpE)
    args = (ea['GE'], ea['BE'], ea['w_incumbent'], B0t, gt, bt, Greq, beta, D, da)
    act = np.where(ea['wE'] > 0)[0]
    a1pp = al[np.where(ea['KE'] < rpE)[0][-1]]
    WE = float(np.sum(ea['wE']) * da)
    top = continuation(act[-1], *args)
    print(f"  entrants active on [{al[act[0]]:.5f}, {al[act[-1]]:.5f}] (alpha_1''^E = {a1pp:.4f}), pooling capital W^E={WE:.5f}")
    if top['exhausted_at'] is None:
        q = top['resid_G'] / max(top['resid_G'] + top['resid_B'], 1e-300)
        KE1 = ea['KE'][-1]
        print(f'  NO clearing cutoff: even with all admissible entry the pool is not exhausted at alpha1.')
        print(f'    residual at alpha1: goods {top["resid_G"]:.5f}, bads {top["resid_B"]:.5f}, quality {q:.4f}')
        print(f'    top pooling incumbent: stays {q * (1 + rpE):.4f}, follows the residual to the first '
              f'market above at min(r_p, K^E(alpha1)) = {min(g.rp, KE1):.4f}: {q * (1 + min(g.rp, KE1)):.4f}')
        return
    x_full = al[top["exhausted_at"]]
    print(f'  with that entry the pool is exhausted at {x_full:.4f} < alpha1; '
          f'idle incumbent capital {top["idle"]:.5f}')

    # Regularity condition (M5) of the note.
    # (a) K^E is strictly increasing at alpha_1''^E.
    k1pp = int(np.where(ea['KE'] < rpE)[0][-1])
    print(f"  (M5a) (K^E)'(alpha_1''^E) = {ea['KE_prime'][k1pp]:.3f}  ->  "
          f"{'ok' if ea['KE_prime'][k1pp] > 0 else 'FAIL'}")
    # (b) On [alpha_1''^E, alpha_1) the incumbents hold more capital than their
    #     own slices absorb at the pooling rate: D(r_p^E)(1-beta)g(omega_g)/w < 1.
    top_mask = (al >= a1pp) & (al < g.alpha1 - 5e-4)
    lent = D * (1 - beta) * gt[top_mask] / np.maximum(ea['w_incumbent'][top_mask], 1e-300)
    print(f"  (M5b) share of incumbent capital that own slices absorb on [alpha_1''^E, alpha_1): "
          f'{lent.min():.3f} to {lent.max():.3f}  ->  {"ok" if lent.max() < 1 else "FAIL"}')
    # (c) If the path of Steps 5-6 ends with a no-entry interval, the entrants'
    #     density is positive at its left edge.  (The solver switches the closed
    #     form off where r_p^E - K^E < 0.005; an interval that starts there is
    #     not a no-entry interval of the construction.)
    i_last = act[-1]
    if rpE - ea['KE'][i_last] < 0.006:
        print('  (M5c) entrants are active up to the end of the path: nothing to check')
    else:
        print(f'  (M5c) the path ends with a no-entry interval from {al[i_last]:.5f}; entrant density at its '
              f'left edge {ea["wE"][i_last]:.2f} against incumbent density {ea["w_incumbent"][i_last]:.2f}  ->  '
              f'{"ok" if ea["wE"][i_last] > 0 else "FAIL"}')
    # What the incumbents above the exhaustion point would do on the full path.
    rat = (al >= x_full) & (al < g.alpha1 - 5e-4)
    share_r = D * (1 - beta) * gt[rat] / np.maximum(ea['w_incumbent'][rat], 1e-300)
    gam0_r = np.array([m._scalar(m.gam0(a)) for a in al[rat][::max(1, rat.sum() // 50)]])
    print(f'  on the full path the incumbents on [{x_full:.4f}, alpha_1) lend a share {share_r.min():.2f} to '
          f'{share_r.max():.2f} of their capital: staying pays {share_r.min() * (1 + rpE):.2f} to '
          f'{share_r.max() * (1 + rpE):.2f}, undercutting {gam0_r.min() * (1 + rpE):.2f} to '
          f'{gam0_r.max() * (1 + rpE):.2f}, non-selective lending R = {ns0["R"]:.3f}')

    # The exhaustion flag is monotone in the cutoff (comparison lemma): check
    # that on a sample of cutoffs, then locate the switch by bisection.
    sample = act[np.unique(np.linspace(0, len(act) - 1, min(len(act), 60)).astype(int))]
    flags = [continuation(i, *args)['exhausted_at'] is not None for i in sample]
    switches = int(np.sum(np.diff(np.array(flags, dtype=int)) != 0))
    if flags[0]:
        print('  every cutoff exhausts early: no clearing cutoff on the grid')
        return
    lo, hi = 0, len(act) - 1            # act[lo] does not exhaust early, act[hi] does
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if continuation(act[mid], *args)['exhausted_at'] is None:
            lo = mid
        else:
            hi = mid
    i_s = act[lo]
    star = continuation(i_s, *args)
    print(f'  clearing cutoff alpha* in [{al[i_s]:.5f}, {al[i_s + 1]:.5f}]  '
          f'(exhaustion flag switches {switches} time(s) on a sample of {len(sample)} cutoffs);')
    print(f'    residual at alpha1 {star["resid_G"] + star["resid_B"]:.5f} (one grid step of inflow is '
          f'{(1 - beta) * gt[-1] * da:.5f}); largest entrant margin above alpha*: {star["margin"]:.1e} '
          f'(grid tolerance {da:.1e})')
    WE2 = float(np.sum(ea['wE'][:i_s + 1]) * da)
    print(f'    entrant pooling capital {WE:.5f} -> {WE2:.5f} ({100 * (WE2 / WE - 1):+.1f}%)')

    ea2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ea.items()}
    ea2['wE'][i_s + 1:] = 0.0
    ea2['GE'][i_s + 1:] = star['G'][i_s + 1:]
    ea2['BE'][i_s + 1:] = star['B'][i_s + 1:]
    ea2['TE'] = ea2['GE'] + ea2['BE']
    ea2['clearing_cutoff'] = float(al[i_s])
    with contextlib.redirect_stdout(io.StringIO()):
        g.badleftoverE = m._leftover_bads_after_pooling(ea2, rpE)
    g.entry_analytical = ea2
    ns1, ok1, fails1 = _margin_and_checks()
    rns0 = m._scalar(m.cfun(g.alpha2)) + g.Pi
    print(f'  with entry stopped at alpha*: case {ns1["case"]}, R={ns1["R"]:.5f}, r_NS^E={ns1["rnsE"]:.5f} '
          f'(baseline {rns0:.5f}), B^NS,E={g.badleftoverE:.6f} (baseline {g.badleftover:.6f}), '
          f'alpha_2^E={ns1["alpha2E"]:.4f} (alpha_2={g.alpha2:.4f}), conditions ok={ok1} {fails1 or ""}')
    print(f'    change: R {ns1["R"] - ns0["R"]:+.1e}, r_NS^E {ns1["rnsE"] - ns0["rnsE"]:+.1e}, '
          f'alpha_2^E {ns1["alpha2E"] - ns0["alpha2E"]:+.1e}')


if __name__ == '__main__':
    import sys
    n_grid = 32000
    if '--n' in sys.argv:
        n_grid = int(sys.argv[sys.argv.index('--n') + 1])
    if '--grid-a0E' not in sys.argv:
        m._entry_breakeven_gap = _entry_breakeven_gap_exact
    print(f'pooling-region grid: {n_grid} points; marginal entrant below alpha_0 located '
          f'{"on the 60-point grid of the solver" if "--grid-a0E" in sys.argv else "exactly"}')
    for cfg, lab in EXAMPLES:
        check(cfg, lab, n_grid)
