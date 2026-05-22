"""Figure 9a (fig:OB left panel) — Open Banking, limited adoption.

Entrants get a cost reduction for very low alpha (C^E < C for alpha < alpha_hat,
with alpha_hat small).
Output: fig9a.png — 2 panels (r and K).
"""
import os
import matplotlib.pyplot as plt
import mainE_python as m


def main():
    print("=" * 60); print("Building Figure 9a (Open Banking, limited)"); print("=" * 60)
    res = m.solve_for_config('FIG9_OB_limited')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel 1: r(alpha) ---
    ax = axes[0]
    ax.plot(res['r_inc_alphas'], res['r_inc_plot'], 'g--', lw=2, label='incumbent')
    ax.plot(res['r_E_alphas'],   res['r_E_plot'],   'b-',  lw=2, label='post-entry')
    ax.axvline(res['alpha0'],  color='g', ls=':', lw=0.6, alpha=0.5)
    ax.axvline(res['alpha0E'], color='b', ls=':', lw=0.6, alpha=0.5)
    ax.set_xlabel(r'$\alpha$'); ax.set_ylabel(r'$r(\alpha)$')
    ax.set_title('Interest-rate schedule'); ax.grid(alpha=0.3); ax.legend(fontsize=9)

    # --- Panel 2: K(alpha) ---
    ax = axes[1]
    ax.plot(res['alphas_plot'], res['K_inc_plot'], 'g--', lw=2,
            label=r'$K(\alpha) = \Pi + C(\alpha)$ (incumbent)')
    ax.plot(res['alphas_plot'], res['K_E_plot'],   'b-',  lw=2,
            label=r'$K^E(\alpha) = \Pi^E + C^E(\alpha)$ (entrant)')
    ax.set_xlabel(r'$\alpha$'); ax.set_ylabel(r'$K(\alpha)$')
    ax.set_title('Modified cost'); ax.grid(alpha=0.3); ax.legend(fontsize=9)

    fig.suptitle('Figure 9a — Open Banking, limited adoption', fontsize=12)
    fig.tight_layout()

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig9a.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
