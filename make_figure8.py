"""Figure 8 (fig:AIintermediateInnovation) — Big-Data Innovation.

Produces fig_panels_fig8.png -- the four panels as a 2x2 grid (cost function,
density of lenders, r(omega), r(alpha)) -- plus the same four as single-panel
PNGs, which the main text still uses.
"""
import mainE_python as m
from panel_plots import (panel_grid, panel_r_alpha, panel_r_omega,
                         panel_K_alpha, panel_w_alpha, omega_g, find_omega_H)


def main():
    print("=" * 60); print("Building Figure 8 (Big-Data Innovation)"); print("=" * 60)
    res = m.solve_for_config('FIG8_bigdata')

    beta = res['config']['beta']
    omega_1 = omega_g(res['alpha1'], beta)
    omega_2 = omega_g(res['alpha2'], beta)
    omega_H = find_omega_H(res)
    omega_hat_2 = (omega_g(res['alpha2E'], beta)
                   if res['alpha2E'] > res['alpha2'] else None)
    vlines = [(omega_1, r'$\omega_1$'), (omega_2, r'$\omega_2$')]
    if omega_H is not None:
        vlines.append((omega_H, r'$\omega_H$'))
    if omega_hat_2 is not None:
        vlines.append((omega_hat_2, r'$\hat\omega_2$'))
    omega_annotations = {
        'vlines': sorted(vlines, key=lambda t: t[0]),
        'regions': [(omega_1 / 2, 'I'),
                    ((omega_1 + omega_2) / 2, 'II'),
                    ((max(omega_2, omega_hat_2 or omega_2) + 1) / 2, 'III')],
    }

    specs = [
        {'res': res, 'kind': 'incumbent', 'color': 'g', 'linestyle': '--',
         'label': 'incumbent'},
        {'res': res, 'kind': 'entrant',   'color': 'b', 'linestyle': '-',
         'label': 'post-entry'},
    ]

    panel_grid(specs, 'fig_panels_fig8.png', annotations=omega_annotations)
    # The single-panel PNGs too: the main text still sets r(omega) and
    # K(alpha) as standalone figures (fig:AIintermediateInnovation).
    panel_r_alpha(specs, None, 'fig_r_alpha_fig8.png')
    panel_r_omega(specs, None, 'fig_r_omega_fig8.png',
                  annotations=omega_annotations)
    panel_K_alpha(specs, None, 'fig_K_alpha_fig8.png')
    panel_w_alpha(specs, None, 'fig_w_alpha_fig8.png')


if __name__ == '__main__':
    main()
