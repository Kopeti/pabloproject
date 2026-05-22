"""Figure 9a (fig:OB left panel) — Open Banking, limited adoption.

Produces three PNGs:
    fig_r_alpha_fig9a.png    r(alpha) for incumbent + post-entry.
    fig_r_omega_fig9a.png    r(omega) for incumbent + post-entry.
    fig_K_alpha_fig9a.png    K(alpha) and K^E(alpha).
"""
import mainE_python as m
from panel_plots import panel_r_alpha, panel_r_omega, panel_K_alpha


def main():
    print("=" * 60); print("Building Figure 9a (OB, limited adoption)"); print("=" * 60)
    res = m.solve_for_config('FIG9_OB_limited')

    specs = [
        {'res': res, 'kind': 'incumbent', 'color': 'g', 'linestyle': '--',
         'label': 'incumbent'},
        {'res': res, 'kind': 'entrant',   'color': 'b', 'linestyle': '-',
         'label': 'post-entry'},
    ]

    panel_r_alpha(specs, 'Figure 9a — r(alpha)',  'fig_r_alpha_fig9a.png')
    panel_r_omega(specs, 'Figure 9a — r(omega)',  'fig_r_omega_fig9a.png')
    panel_K_alpha(specs, 'Figure 9a — K(alpha)',  'fig_K_alpha_fig9a.png')


if __name__ == '__main__':
    main()
