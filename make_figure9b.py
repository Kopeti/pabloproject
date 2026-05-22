"""Figure 9b (fig:OB right panel) — Open Banking, broad adoption.

Produces three PNGs:
    fig_r_alpha_fig9b.png    r(alpha) for incumbent + post-entry.
    fig_r_omega_fig9b.png    r(omega) for incumbent + post-entry.
    fig_K_alpha_fig9b.png    K(alpha) and K^E(alpha).
"""
import mainE_python as m
from panel_plots import panel_r_alpha, panel_r_omega, panel_K_alpha


def main():
    print("=" * 60); print("Building Figure 9b (OB, broad adoption)"); print("=" * 60)
    res = m.solve_for_config('FIG9_OB_broad')

    specs = [
        {'res': res, 'kind': 'incumbent', 'color': 'g', 'linestyle': '--',
         'label': 'incumbent'},
        {'res': res, 'kind': 'entrant',   'color': 'b', 'linestyle': '-',
         'label': 'post-entry'},
    ]

    panel_r_alpha(specs, 'Figure 9b — r(alpha)',  'fig_r_alpha_fig9b.png')
    panel_r_omega(specs, 'Figure 9b — r(omega)',  'fig_r_omega_fig9b.png')
    panel_K_alpha(specs, 'Figure 9b — K(alpha)',  'fig_K_alpha_fig9b.png')


if __name__ == '__main__':
    main()
