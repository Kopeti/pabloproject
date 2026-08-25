"""Figure 11 — a parallel downward shift of the total cost of entry.

Same baseline as figures 10a/10b: C(alpha) = 0.2 alpha + 16 alpha^3,
Pi = 0.235, beta = 0.5, BperG = 1.0 (alpha_0 = 0.140, alpha_1 = 0.394,
alpha_2 = 0.612, r_p = 1.293, r_NS = 4.018).

Entrants differ from incumbents only by a parallel shift of the total cost of
entry,

    K^E(alpha) = K(alpha) - Delta,   i.e.   C^E = C   and   Pi^E = Pi - Delta,

with Delta = 0.10.  The screening technology is untouched; only the fixed cost
of entry falls.  This is the pure "access" channel and the complement to
figures 10a/10b: there the fixed cost was held equal (Pi^E = Pi) and only the
SHAPE of K^E differed, so r_NS moved through pool composition alone and the
effect was tiny.  Here the shape is identical and the entire effect comes
through the level -- including at alpha = 0, where the non-selective lenders
sit -- so the effect is an order of magnitude larger:

    r_NS 4.018 -> 3.597 (drop 0.421), Region IIb of width 0.036,
    alpha_0^E = 0.132 (entry below the incumbent alpha_0 = 0.140),
    r_p 1.293 -> 1.117.

The branch is the same one as figure 10a ("NS rate goes down, with NS entry
before alpha_2"), so the two figures isolate the level and shape channels of
the same mechanism.  Note the effect is not monotone in Delta: past
Delta ~ 0.10 it partly reverses (at Delta = 0.15 the drop falls back to 0.28
and Region IIb shrinks to 0.015), and Delta = 0.10 is near the widest IIb.

Produces three PNGs in the paper figures folder (same place as fig9a/fig10a):
    fig_r_alpha_fig11.png    r(alpha) for incumbent + post-entry.
    fig_r_omega_fig11.png    r(omega) for incumbent + post-entry.
    fig_K_alpha_fig11.png    K(alpha) and K^E(alpha).
"""
import numpy as np
import mainE_python as m
from panel_plots import (panel_r_alpha, panel_r_omega, panel_K_alpha,
                         omega_g, find_omega_H)


PI = 0.235
SHIFT = 0.10

# Shared with figures 10a/10b.  Must be numpy-safe (array in, array out): the
# 'polynomial' cfunE kind evaluates it on an array, and here the entrant's cost
# curve IS the incumbent's, which is exactly what makes the shift parallel.
def CFUN(alpha):
    a = np.asarray(alpha, dtype=float)
    return 0.2 * a + 16.0 * a**3


m.PARAM_CONFIGS['FIG11_parallel_shift'] = {
    'description': f'Figure 11 - parallel shift K^E = K - {SHIFT}.',
    'Pi': PI, 'beta': 0.5, 'BperG': 1.0,
    'cfun': CFUN,
    'has_entry': True,
    'PiE': PI - SHIFT,                  # the whole shift lives in the level
    'cfunE_kind': 'polynomial',
    'cfunE_params': {'coeffs': CFUN},   # same variable cost as the incumbent
}


def main():
    print("=" * 60)
    print(f"Building Figure 11 (parallel shift, Delta = {SHIFT})")
    print("=" * 60)
    res = m.solve_for_config('FIG11_parallel_shift')

    beta = res['config']['beta']
    omega_1 = omega_g(res['alpha1'], beta)
    omega_2 = omega_g(res['alpha2'], beta)
    has_IIb = res['alpha2E'] < res['alpha2'] - 1e-6
    omega_2E = omega_g(res['alpha2E'], beta) if has_IIb else None
    # Too narrow to label in place: the omega_2^E tick would collide with
    # omega_2.  Mark it with a callout arrow instead (as in figure 10a).
    IIb_is_sliver = has_IIb and (omega_2 - omega_2E) < 0.02
    # A parallel shift keeps K^E strictly below K, so there is no crossing
    # inside the CIM region; find_omega_H returns None here.
    omega_H = find_omega_H(res)

    vlines = [(omega_1, r'$\omega_1$'), (omega_2, r'$\omega_2$')]
    if has_IIb:
        # Always draw the IIa/IIb border at omega_2^E.  When the band is a
        # sliver the tick LABEL is suppressed (it would collide with omega_2's)
        # but the line itself stays; the callout arrow names the region.
        vlines.append((omega_2E, '' if IIb_is_sliver else r"$\omega_2^E$"))
    if omega_H is not None:
        vlines.append((omega_H, r'$\omega_H$'))

    callouts = []
    if not has_IIb:
        region_labels = [(omega_1 / 2, 'I'),
                         ((omega_1 + omega_2) / 2, 'II'),
                         ((omega_2 + 1) / 2, 'III')]
    elif IIb_is_sliver:
        region_labels = [(omega_1 / 2, 'I'),
                         ((omega_1 + omega_2E) / 2, 'IIa'),
                         ((omega_2 + 1) / 2 + 0.03, 'III')]
        callouts = [((omega_2E + omega_2) / 2, 'IIb', omega_2 + 0.055)]
    else:
        region_labels = [(omega_1 / 2, 'I'),
                         ((omega_1 + omega_2E) / 2, 'IIa'),
                         ((omega_2E + omega_2) / 2, 'IIb'),
                         ((omega_2 + 1) / 2, 'III')]
    omega_annotations = {
        'vlines': sorted(vlines, key=lambda t: t[0]),
        'regions': region_labels,
        'callouts': callouts,
    }

    specs = [
        {'res': res, 'kind': 'incumbent', 'color': 'g', 'linestyle': '--',
         'label': 'incumbent'},
        {'res': res, 'kind': 'entrant',   'color': 'b', 'linestyle': '-',
         'label': 'post-entry'},
    ]

    panel_r_alpha(specs, 'Figure 11 — r(alpha)', 'fig_r_alpha_fig11.png')
    panel_r_omega(specs, 'Figure 11 — r(omega)', 'fig_r_omega_fig11.png',
                  annotations=omega_annotations)
    panel_K_alpha(specs, 'Figure 11 — K(alpha)', 'fig_K_alpha_fig11.png')

    print(f"  alpha0E={res['alpha0E']:.4f}  alpha2E={res['alpha2E']:.4f}  "
          f"IIb={res['alpha2']-res['alpha2E']:.4f}  rpE={res['rpE']:.4f}  "
          f"rnsE={res['rnsE']:.4f}  (r_NS drop {res['rNS_inc']-res['rnsE']:+.4f})")


if __name__ == '__main__':
    main()
