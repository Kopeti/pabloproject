"""Figure 7 (fig:SPT) — Selection-Preserving Technology.

Produces three PNGs that LaTeX assembles into Figure 7:
    fig_r_alpha_fig7.png   r(alpha) overlay of incumbent + SP_highPiE + SP_lowPiE.
    fig_r_omega_fig7.png   r(omega) overlay (same three curves).
    fig_K_alpha_fig7.png   K(alpha) overlay (incumbent + entrant; K_SP is
                            independent of Pi^E so one entrant curve suffices).
"""
import mainE_python as m
from panel_plots import panel_r_alpha, panel_r_omega, panel_K_alpha


def main():
    print("=" * 60); print("Building Figure 7 (fig:SPT)"); print("=" * 60)
    incumbent  = m.solve_for_config('FIG7_incumbent')
    sp_highPiE = m.solve_for_config('FIG7_SP_highPiE')
    sp_lowPiE  = m.solve_for_config('FIG7_SP_lowPiE')

    rate_specs = [
        {'res': incumbent,  'kind': 'incumbent', 'color': 'g', 'linestyle': '--',
         'label': 'incumbent'},
        {'res': sp_highPiE, 'kind': 'entrant',   'color': 'b', 'linestyle': '-',
         'label': r'SP entry, $\Pi^E > \bar\Pi^E$'},
        {'res': sp_lowPiE,  'kind': 'entrant',   'color': 'r', 'linestyle': '-',
         'label': r'SP entry, $\Pi^E < \bar\Pi^E$'},
    ]

    # K^E = K_SP(alpha; kappa) is independent of Pi^E, so the high-PiE and
    # low-PiE curves coincide. Plot one of them.
    K_specs = [
        {'res': incumbent,  'kind': 'incumbent', 'color': 'g', 'linestyle': '--',
         'label': r'$K(\alpha) = \Pi + C(\alpha)$ (incumbent)'},
        {'res': sp_highPiE, 'kind': 'entrant',   'color': 'b', 'linestyle': '-',
         'label': r'$K^E(\alpha) = K_{SP}(\alpha; \kappa)$ (entrant)'},
    ]

    panel_r_alpha(rate_specs, 'Figure 7 — r(alpha)',  'fig_r_alpha_fig7.png')
    panel_r_omega(rate_specs, 'Figure 7 — r(omega)',  'fig_r_omega_fig7.png')
    panel_K_alpha(K_specs,    'Figure 7 — K(alpha)',  'fig_K_alpha_fig7.png')


if __name__ == '__main__':
    main()
