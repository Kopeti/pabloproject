"""Figure 7 (fig:SPT) — Selection-Preserving Technology.

Runs three configs:
  FIG7_incumbent    — green dashed (incumbent only)
  FIG7_SP_highPiE   — blue  (SP entry, PiE > PiE_bar, no Region IIb)
  FIG7_SP_lowPiE    — red   (SP entry, PiE < PiE_bar, Region IIb)

Output: fig7.png — 2 panels (r and K).
"""
import os
import matplotlib.pyplot as plt
import mainE_python as m


def main():
    print("=" * 60); print("Building Figure 7 (fig:SPT)"); print("=" * 60)
    incumbent  = m.solve_for_config('FIG7_incumbent')
    sp_highPiE = m.solve_for_config('FIG7_SP_highPiE')
    sp_lowPiE  = m.solve_for_config('FIG7_SP_lowPiE')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel 1: r(alpha) ---
    ax = axes[0]
    ax.plot(incumbent['alphas_plot'], incumbent['r_inc_plot'],
            'g--', lw=2, label='incumbent')
    ax.plot(sp_highPiE['alphas_plot'], sp_highPiE['r_E_plot'],
            'b-',  lw=2, label=r'SP entry, $\Pi^E > \bar\Pi^E$')
    ax.plot(sp_lowPiE['alphas_plot'],  sp_lowPiE['r_E_plot'],
            'r-',  lw=2, label=r'SP entry, $\Pi^E < \bar\Pi^E$')
    for cfg, color in [(incumbent, 'g'), (sp_highPiE, 'b'), (sp_lowPiE, 'r')]:
        ax.axvline(cfg['alpha0'], color=color, ls=':', lw=0.5, alpha=0.5)
    ax.set_xlabel(r'$\alpha$'); ax.set_ylabel(r'$r(\alpha)$')
    ax.set_title('Interest-rate schedule'); ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # --- Panel 2: K(alpha) ---
    # K^E = Π^E + C^E_SP = K_SP(α; κ) is independent of Π^E by construction,
    # so the high-/low-Π^E entry K curves coincide. Plot one of them.
    ax = axes[1]
    ax.plot(incumbent['alphas_plot'], incumbent['K_inc_plot'],
            'g--', lw=2, label=r'$K(\alpha) = \Pi + C(\alpha)$ (incumbent)')
    ax.plot(sp_highPiE['alphas_plot'], sp_highPiE['K_E_plot'],
            'b-',  lw=2,
            label=r'$K^E(\alpha) = K_{SP}(\alpha;\kappa)$ (entrant, both $\Pi^E$)')
    ax.set_xlabel(r'$\alpha$'); ax.set_ylabel(r'$K(\alpha)$')
    ax.set_title('Modified cost'); ax.grid(alpha=0.3); ax.legend(fontsize=9)

    fig.suptitle('Figure 7 — Selection-Preserving Technology', fontsize=12)
    fig.tight_layout()

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig7.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
