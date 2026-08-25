"""Figures 10a/10b — the pool channel with equal fixed costs (Pi^E = Pi).

One baseline, two entrant cost functions, both monotone with K^E(0) = K(0):
the sign of the NS-rate spillover is decided purely by WHERE the entrant's
cost advantage sits on the skill axis.

    Baseline:  C(alpha) = 0.2*alpha + 16*alpha^3, Pi = 0.235
               (flat early section pushes alpha_0 up to 0.140, leaving room
               for entry below it; the cubic convexity keeps Region III alive:
               alpha_1 = 0.394, alpha_2 = 0.612, r_NS = 4.018)
    10a (pool improvement): multiplicative Gaussian dip centered at alpha = 0.
               Entry starts below alpha_0 (0.134 < 0.140), entrants absorb bad
               borrowers in the band omega in [omega_b(alpha_0^E),
               omega_b(alpha_0)) that no incumbent accepts; badleftover falls
               and r_NS drops (4.018 -> 4.010).
    10b (pool worsening): same dip form centered at alpha = 0.27 (mid-pool).
               Entrants cream-skim good borrowers from the pooling segment;
               badleftover rises (+0.043) and r_NS rises (4.018 -> 4.237).
               Note the asymmetry: the worsening channel scales almost freely
               with the dip (this calibration is near the monotonicity frontier
               of K^E), while the improvement channel of 10a is structurally
               capped near -0.007 (advantage below alpha_0 bounded by C(alpha)
               since K^E(0)=K(0); gamma_0 gradient penalizes low-alpha entry;
               Region-I absorption is anchored at alpha_1).

Produces eight PNGs in the paper figures folder (same place as fig9a/fig9b):
    fig_r_alpha_fig10a.png   fig_r_omega_fig10a.png   fig_K_alpha_fig10a.png
    fig_r_alpha_fig10b.png   fig_r_omega_fig10b.png   fig_K_alpha_fig10b.png
    fig_K_alpha_zoom_fig10a.png     zoom of K(alpha) on the band where
                                    K^E < K (the entrant advantage at the
                                    bottom of the skill range)
    fig_r_omega_IIbzoom_fig10a.png  zoom of r(omega) at the top of Region II:
                                    the sliver-thin Region IIb and the lower
                                    post-entry NS rate become visible
"""
import numpy as np
import mainE_python as m
from panel_plots import (panel_r_alpha, panel_r_omega, panel_K_alpha,
                         omega_g, find_omega_H)


# Registered here rather than in mainE_python.PARAM_CONFIGS so the example
# stays self-contained; move them into mainE_python once adopted.
_CUBIC_BASELINE = dict(
    Pi=0.235, beta=0.5, BperG=1.0,
    cfun=lambda alpha: 0.2 * alpha + 16.0 * alpha**3,
    has_entry=True,
    PiE=0.235,                      # equal fixed costs: K^E(0) = K(0)
    cfunE_kind='cost_dip_multiplicative',
)

m.PARAM_CONFIGS['FIG10a_pool_improve'] = {
    'description': 'Figure 10a - pool improvement: dip at the bottom, PiE=Pi.',
    **_CUBIC_BASELINE,
    # m(0) = 1 + 0.13 - 1.13 = 0 -> C^E ~ flat at zero, K^E(0) = Pi exactly;
    # advantage on (0, 0.21), max 0.023 at alpha = 0.13.
    'cfunE_params': {'alpha_center': 0.0, 'sigma': 0.102,
                     'delta_dip': 1.13, 'delta_baseline': 0.13},
}

m.PARAM_CONFIGS['FIG10b_pool_worsen'] = {
    'description': 'Figure 10b - pool worsening: dip at mid-pool, PiE=Pi.',
    **_CUBIC_BASELINE,
    # Advantage band (0.159, 0.381) inside (alpha_0, alpha_1) = (0.140, 0.394),
    # max 0.237 at alpha = 0.30; C(0) = 0 so K^E(0) = Pi exactly.  delta_dip =
    # 0.68 is near the monotonicity frontier of K^E = Pi + C*m: the min slope
    # on the left flank is +0.12 (at 0.70 it is +0.03; at 0.75 K^E dips).
    # Outcome: badleftoverE 0.595 -> 0.638, r_NS 4.018 -> 4.237 (+0.219).
    'cfunE_params': {'alpha_center': 0.27, 'sigma': 0.06,
                     'delta_dip': 0.68, 'delta_baseline': 0.12},
}


def build_figure(config_name, tag, title_prefix):
    """Solve one config and emit the three standard panels (as in fig9a/9b)."""
    res = m.solve_for_config(config_name)

    beta = res['config']['beta']
    omega_1 = omega_g(res['alpha1'], beta)
    omega_2 = omega_g(res['alpha2'], beta)
    # IIa/IIb boundary as in make_figure9a: omega_g(alpha2E) when Region IIb
    # has positive width — however tiny (in 10a the NS-entry branch produces a
    # sliver-thin IIb, alpha2E = 0.6107 vs alpha2 = 0.6116, which we still mark).
    has_IIb = res['alpha2E'] < res['alpha2'] - 1e-6
    omega_2E = omega_g(res['alpha2E'], beta) if has_IIb else None
    # A IIb narrower than this (in omega units) cannot be labelled in place:
    # its own tick label would collide with omega_2 and the region label would
    # not fit between the vlines.  Use a callout arrow instead.
    IIb_is_sliver = has_IIb and (omega_2 - omega_2E) < 0.02
    # K^E/K crossing inside the CIM region as in make_figure9b (if any).
    omega_H = find_omega_H(res)

    vlines = [(omega_1, r'$\omega_1$'), (omega_2, r'$\omega_2$')]
    if has_IIb and not IIb_is_sliver:
        vlines.append((omega_2E, r"$\omega_2^E$"))
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

    panel_r_alpha(specs, f'{title_prefix} — r(alpha)', f'fig_r_alpha_{tag}.png')
    panel_r_omega(specs, f'{title_prefix} — r(omega)', f'fig_r_omega_{tag}.png',
                  annotations=omega_annotations)
    panel_K_alpha(specs, f'{title_prefix} — K(alpha)', f'fig_K_alpha_{tag}.png')
    return res


def build_fig10a_zooms(res, specs):
    """Two zoom panels for Figure 10a.

    (1) K(alpha) on the band where K^E < K: the entrant's cost advantage at
        the bottom of the skill range, invisible at full scale.
    (2) r(omega) at the top of Region II: the sliver-thin Region IIb
        (omega_2^E to omega_2) and the lower post-entry NS rate.
    """
    beta = res['config']['beta']

    # --- zoom 1: K^E < K band -------------------------------------------
    alphas = res['alphas_plot']
    diff = res['K_E_plot'] - res['K_inc_plot']
    neg = np.where(diff < 0)[0]
    a_cross = alphas[neg[-1]]              # K^E crosses K from below (~0.21)
    a_hi = a_cross + 0.06
    in_win = alphas <= a_hi
    y_hi = max(res['K_inc_plot'][in_win].max(),
               res['K_E_plot'][in_win].max()) + 0.02
    y_lo = res['K_inc_plot'][0] - 0.03
    panel_K_alpha(specs, r'Figure 10a — K(alpha), zoom: $K^E<K$ band',
                  'fig_K_alpha_zoom_fig10a.png',
                  xlim=(0.0, a_hi), ylim=(y_lo, y_hi))

    # --- zoom 2: Region IIb and the NS-rate drop ------------------------
    omega_2E = omega_g(res['alpha2E'], beta)
    omega_2 = omega_g(res['alpha2'], beta)
    band = omega_2 - omega_2E
    x_lo, x_hi = omega_2E - 6.0 * band, omega_2 + 10.0 * band
    gap = res['rNS_inc'] - res['rnsE']
    y_lo, y_hi = res['rnsE'] - 2.2 * gap, res['rNS_inc'] + 1.2 * gap
    zoom_annotations = {
        # omega_2E's tick label stays empty: at this separation the two tick
        # texts would still collide; the IIb callout identifies the band.
        'vlines': [(omega_2E, ''), (omega_2, r'$\omega_2$')],
        'regions': [((x_lo + omega_2E) / 2, 'IIa'),
                    (x_hi - 1.5 * band, 'III')],
        'callouts': [((omega_2E + omega_2) / 2, 'IIb', omega_2 + 4.0 * band)],
    }
    panel_r_omega(specs, 'Figure 10a — r(omega), zoom: Region IIb and NS rates',
                  'fig_r_omega_IIbzoom_fig10a.png',
                  annotations=zoom_annotations,
                  xlim=(x_lo, x_hi), ylim=(y_lo, y_hi))


def main():
    print("=" * 60)
    print("Building Figure 10a (pool improvement, PiE = Pi)")
    print("=" * 60)
    res_a = build_figure('FIG10a_pool_improve', 'fig10a', 'Figure 10a')
    specs_a = [
        {'res': res_a, 'kind': 'incumbent', 'color': 'g', 'linestyle': '--',
         'label': 'incumbent'},
        {'res': res_a, 'kind': 'entrant',   'color': 'b', 'linestyle': '-',
         'label': 'post-entry'},
    ]
    build_fig10a_zooms(res_a, specs_a)

    print("=" * 60)
    print("Building Figure 10b (pool worsening, PiE = Pi)")
    print("=" * 60)
    build_figure('FIG10b_pool_worsen', 'fig10b', 'Figure 10b')


if __name__ == '__main__':
    main()
