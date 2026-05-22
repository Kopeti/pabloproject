"""Figure 8 (fig:AIintermediateInnovation) — Big-Data Innovation.

Produces three PNGs that LaTeX assembles into Figure 8:
    fig_r_alpha_fig8.png    r(alpha) for incumbent + post-entry.
    fig_r_omega_fig8.png    r(omega) for incumbent + post-entry.
    fig_K_alpha_fig8.png    K(alpha) and K^E(alpha).
"""
import mainE_python as m
from panel_plots import panel_r_alpha, panel_r_omega, panel_K_alpha


def main():
    print("=" * 60); print("Building Figure 8 (Big-Data Innovation)"); print("=" * 60)
    res = m.solve_for_config('FIG8_bigdata')

    specs = [
        {'res': res, 'kind': 'incumbent', 'color': 'g', 'linestyle': '--',
         'label': 'incumbent'},
        {'res': res, 'kind': 'entrant',   'color': 'b', 'linestyle': '-',
         'label': 'post-entry'},
    ]

    panel_r_alpha(specs, 'Figure 8 — r(alpha)',  'fig_r_alpha_fig8.png')
    panel_r_omega(specs, 'Figure 8 — r(omega)',  'fig_r_omega_fig8.png')
    panel_K_alpha(specs, 'Figure 8 — K(alpha)',  'fig_K_alpha_fig8.png')


if __name__ == '__main__':
    main()
