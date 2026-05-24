"""Figure 9a (fig:OB left panel) — Open Banking, limited adoption.

Produces three PNGs:
    fig_r_alpha_fig9a.png    r(alpha) for incumbent + post-entry.
    fig_r_omega_fig9a.png    r(omega) for incumbent + post-entry.
    fig_K_alpha_fig9a.png    K(alpha) and K^E(alpha).
"""
import mainE_python as m
from panel_plots import (panel_r_alpha, panel_r_omega, panel_K_alpha,
                         omega_g, find_omega_H)


def main():
    print("=" * 60); print("Building Figure 9a (OB, limited adoption)"); print("=" * 60)
    res = m.solve_for_config('FIG9_OB_limited')

    beta = res['config']['beta']
    omega_1 = omega_g(res['alpha1'], beta)
    omega_2 = omega_g(res['alpha2'], beta)
    # IIa/IIb boundary: ω_g(α₂^E).  In Region IIa (ω₁, ω₂^E) the entrant
    # tracks the incumbent's CIM rate; in Region IIb (ω₂^E, ω₂) the entrant
    # has switched to its NS rate, while the incumbent's CIM rate keeps
    # rising — visible divergence of the curves.
    has_IIb = res['alpha2E'] < res['alpha2'] - 1e-3
    omega_2E = omega_g(res['alpha2E'], beta) if has_IIb else None

    vlines = [(omega_1, r'$\omega_1$'), (omega_2, r'$\omega_2$')]
    if has_IIb:
        vlines.append((omega_2E, r"$\omega_2^E$"))

    if has_IIb:
        region_labels = [(omega_1 / 2, 'I'),
                         ((omega_1 + omega_2E) / 2, 'IIa'),
                         ((omega_2E + omega_2) / 2, 'IIb'),
                         ((omega_2 + 1) / 2, 'III')]
    else:
        region_labels = [(omega_1 / 2, 'I'),
                         ((omega_1 + omega_2) / 2, 'II'),
                         ((omega_2 + 1) / 2, 'III')]
    omega_annotations = {
        'vlines': sorted(vlines, key=lambda t: t[0]),
        'regions': region_labels,
    }

    specs = [
        {'res': res, 'kind': 'incumbent', 'color': 'g', 'linestyle': '--',
         'label': 'incumbent'},
        {'res': res, 'kind': 'entrant',   'color': 'b', 'linestyle': '-',
         'label': 'post-entry'},
    ]

    panel_r_alpha(specs, 'Figure 9a — r(alpha)',  'fig_r_alpha_fig9a.png')
    panel_r_omega(specs, 'Figure 9a — r(omega)',  'fig_r_omega_fig9a.png',
                  annotations=omega_annotations)
    panel_K_alpha(specs, 'Figure 9a — K(alpha)',  'fig_K_alpha_fig9a.png')


if __name__ == '__main__':
    main()
