"""Figure 8 (fig:AIintermediateInnovation) — Big-Data Innovation.

Produces three PNGs that LaTeX assembles into Figure 8:
    fig_r_alpha_fig8.png    r(alpha) for incumbent + post-entry.
    fig_r_omega_fig8.png    r(omega) for incumbent + post-entry.
    fig_K_alpha_fig8.png    K(alpha) and K^E(alpha).
"""
import mainE_python as m
from panel_plots import (panel_r_alpha, panel_r_omega, panel_K_alpha,
                         omega_g, find_omega_H)


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

    panel_r_alpha(specs, 'Figure 8 — r(alpha)',  'fig_r_alpha_fig8.png')
    panel_r_omega(specs, 'Figure 8 — r(omega)',  'fig_r_omega_fig8.png',
                  annotations=omega_annotations)
    panel_K_alpha(specs, 'Figure 8 — K(alpha)',  'fig_K_alpha_fig8.png')


if __name__ == '__main__':
    main()
