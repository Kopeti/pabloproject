"""
Generate the paper figure (fig:nested-vs-iid) with only the analytical nested
solution (labelled "Nested") and Ind Inf. No discrete comparison curves.

Saves to: Peter-Pablo-Maryam/figures/credit_model.png
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt

# Import from credit_model.py in the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from credit_model import (
    Parameters, solve_nested_analytical, solve_iid,
    compute_avg_good_rate, cfun
)

def plot_paper_figure(params, nested, iid):
    """Plot nested vs Ind Inf comparison without discrete curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    alpha0 = nested['alpha0']
    alpha1 = nested['alpha1']
    alpha2 = nested['alpha2']
    rp = nested['rp']
    r_NS = nested['r_NS']
    WNS = nested['WNS']

    prior_desc = "uniform priors" if params.a_g == 0 and params.a_b == 0 else f"a_g={params.a_g}, a_b={params.a_b}"

    # =====================================================================
    # Panel (A): Interest Rate r(alpha)
    # =====================================================================
    alpha_r1 = np.linspace(alpha0, alpha1, 100)
    r_r1 = rp * np.ones_like(alpha_r1)
    alpha_r2 = np.linspace(alpha1, alpha2, 100)
    r_r2 = cfun(alpha_r2) + params.Pi

    axes[0,0].plot(alpha_r1, r_r1, 'b-', lw=2, label='Nested')
    axes[0,0].plot(alpha_r2, r_r2, 'b-', lw=2)
    axes[0,0].plot(0, r_NS, 'bo', markersize=10, markerfacecolor='blue', label='Nested (non-selective)')
    axes[0,0].plot(iid['alpha_eq'], iid['r_eq'], 'r-', lw=2, label='Ind Inf')

    # Average interest rate for good borrowers
    avg_r_nested, avg_r_iid, _, _ = compute_avg_good_rate(params, nested, iid)
    axes[0,0].axhline(avg_r_nested, color='blue', ls='--', lw=1.2, alpha=0.7)
    axes[0,0].axhline(avg_r_iid, color='red', ls='--', lw=1.2, alpha=0.7)
    # Labels near right edge of panel
    axes[0,0].text(0.9, avg_r_nested + 0.02, f'Nested avg = {avg_r_nested:.3f}', fontsize=7,
                   color='blue', ha='center', va='bottom')
    axes[0,0].text(0.9, avg_r_iid - 0.02, f'Ind Inf avg = {avg_r_iid:.3f}', fontsize=7,
                   color='red', ha='center', va='top')

    axes[0,0].axvline(alpha0, color='black', ls=':', alpha=0.3)
    axes[0,0].axvline(alpha1, color='blue', ls=':', alpha=0.5)
    axes[0,0].axvline(alpha2, color='blue', ls=':', alpha=0.5)
    axes[0,0].set_xlabel(r'$\alpha$'); axes[0,0].set_ylabel(r'$r(\alpha)$')
    axes[0,0].set_title('(A) Interest Rate')
    axes[0,0].legend(loc='upper left', fontsize=8, bbox_to_anchor=(0, 0.88))
    axes[0,0].grid(alpha=0.3)
    r_top = max(1.5, 1.08 * iid.get('r_perfect', 0.0), 1.08 * r_NS)
    axes[0,0].set_xlim([-0.03, 1.03]); axes[0,0].set_ylim([0, r_top])
    axes[0,0].text((alpha0+alpha1)/2, rp - 0.06, 'I', fontsize=12, color='blue')
    axes[0,0].text((alpha1+alpha2)/2, 0.45, 'II', fontsize=12, color='blue')
    axes[0,0].text(0.03, r_NS + 0.05, 'III', fontsize=12, color='blue')

    # =====================================================================
    # Panel (B): Pool Quality gamma(alpha)
    # =====================================================================
    axes[0,1].plot(nested['alphas_R1'], nested['gammas_R1'], 'b-', lw=2, label='Nested')
    axes[0,1].plot(nested['alphas_R2'], nested['gammas_R2'], 'b-', lw=2)
    axes[0,1].plot(iid['alpha_eq'], iid['gamma_eq'], 'r-', lw=2, label='Ind Inf')
    # Non-selective lender gamma at alpha=0
    goodleft = nested.get('goodleftover_alpha2', 0)
    badleft = nested['badleftover']
    gamma_NS = goodleft / (goodleft + badleft) if (goodleft + badleft) > 1e-12 else 0.0
    axes[0,1].plot(0, gamma_NS, 'bo', markersize=10, markerfacecolor='blue',
                   label='Nested (non-selective)')

    axes[0,1].axvline(alpha0, color='black', ls=':', alpha=0.3)
    axes[0,1].axvline(alpha1, color='blue', ls=':', alpha=0.5)
    axes[0,1].axvline(alpha2, color='blue', ls=':', alpha=0.5)
    axes[0,1].set_xlabel(r'$\alpha$'); axes[0,1].set_ylabel(r'$\gamma(\alpha)$')
    axes[0,1].set_title('(B) Pool Quality (selective lenders)')
    axes[0,1].legend(fontsize=9, loc='center left')
    axes[0,1].grid(alpha=0.3)
    axes[0,1].set_xlim([-0.03, 1.03])

    # =====================================================================
    # Panel (C): Cumulative Capital W(alpha)
    # =====================================================================
    # Jump at alpha=0
    axes[1,0].plot([0, 0], [0, WNS], 'b-', lw=2)
    axes[1,0].plot(0, 0, 'bo', markersize=8, markerfacecolor='white', markeredgewidth=2)
    axes[1,0].plot(0, WNS, 'bo', markersize=8, markerfacecolor='blue', label='Nested (non-selective)')
    alpha_flat = np.linspace(0, alpha0, 20)
    axes[1,0].plot(alpha_flat, WNS * np.ones_like(alpha_flat), 'b-', lw=2, label='Nested')
    # Region I
    axes[1,0].plot(nested['alphas_R1'], WNS + nested['W_cumsum_R1'], 'b-', lw=2)
    # Region II
    axes[1,0].plot(nested['alphas_R2'], WNS + nested['W_R2_cumsum'], 'b-', lw=2)

    # Ind Inf
    alpha_before_iid = np.linspace(0, iid['alpha0'], 20)
    axes[1,0].plot(alpha_before_iid, np.zeros_like(alpha_before_iid), 'r-', lw=2, label='Ind Inf')
    axes[1,0].plot(iid['alpha_eq'], iid['W_cumsum'], 'r-', lw=2)

    axes[1,0].axvline(alpha0, color='black', ls=':', alpha=0.3)
    axes[1,0].axvline(alpha1, color='blue', ls=':', alpha=0.5)
    axes[1,0].axvline(alpha2, color='blue', ls=':', alpha=0.5)
    axes[1,0].set_xlabel(r'$\alpha$'); axes[1,0].set_ylabel(r'$W(\alpha)$')
    axes[1,0].set_title('(C) Cumulative Capital')
    axes[1,0].legend(fontsize=9)
    axes[1,0].grid(alpha=0.3)
    axes[1,0].set_xlim([-0.03, 1.03])

    # =====================================================================
    # Panel (D): Remaining Borrowers
    # =====================================================================
    axes[1,1].plot(nested['alphas_R1'], nested['GLOs_R1'], 'b-', lw=2, label='Nested G')
    axes[1,1].plot(nested['alphas_R1'], nested['BLOs_R1'], 'b--', lw=2, label='Nested B')
    axes[1,1].plot(nested['alphas_R2'], nested['GLOs_R2'], 'b-', lw=2)
    axes[1,1].plot(nested['alphas_R2'], nested['BLOs_R2'], 'b--', lw=2)
    axes[1,1].plot(iid['alpha_eq'], iid['G_eq'], 'r-', lw=2, label='Ind Inf G')
    axes[1,1].plot(iid['alpha_eq'], iid['B_eq'], 'r--', lw=2, label='Ind Inf B')

    # Nested: remaining masses at alpha2, cleared by NS lenders (drop to 0)
    G_rem_a2 = nested['GLOs_R2'][-1]
    B_rem_a2 = nested['BLOs_R2'][-1]
    # G: filled dot at top, vertical drop, open dot at 0
    axes[1,1].plot([alpha2, alpha2], [G_rem_a2, 0], 'b-', lw=2)
    axes[1,1].plot(alpha2, G_rem_a2, 'bo', markersize=8, markerfacecolor='blue', markeredgewidth=2)
    axes[1,1].plot(alpha2, 0, 'bo', markersize=8, markerfacecolor='white', markeredgewidth=2)
    # B: filled dot at top, vertical drop, open dot at 0
    axes[1,1].plot([alpha2, alpha2], [B_rem_a2, 0], 'b--', lw=2)
    axes[1,1].plot(alpha2, B_rem_a2, 'bo', markersize=8, markerfacecolor='blue', markeredgewidth=2)
    axes[1,1].plot(alpha2, 0, 'bo', markersize=8, markerfacecolor='white', markeredgewidth=2)

    # Ind Inf: good borrowers clear at alpha_bar; bad remain unserved
    alpha_bar = iid['alpha_bar']
    G_rem_iid = iid['G_eq'][-1]
    axes[1,1].plot(alpha_bar, G_rem_iid, 'ro', markersize=8, markerfacecolor='red', markeredgewidth=2)

    axes[1,1].axvline(alpha0, color='black', ls=':', alpha=0.3)
    axes[1,1].axvline(alpha1, color='blue', ls=':', alpha=0.5)
    axes[1,1].axvline(alpha2, color='blue', ls=':', alpha=0.5)
    axes[1,1].set_xlabel(r'$\alpha$'); axes[1,1].set_ylabel('Mass')
    axes[1,1].set_title('(D) Remaining Borrowers')
    axes[1,1].legend(fontsize=8)
    axes[1,1].grid(alpha=0.3)
    axes[1,1].set_xlim([-0.03, 1.03])

    # =====================================================================
    # Add alpha labels to all panels
    # =====================================================================
    for ax in axes.flat:
        yl = ax.get_ylim()
        yt = yl[1] - 0.02 * (yl[1] - yl[0])
        nudge = 0.01
        ax.text(alpha0 + nudge, yt, r'$\alpha_0$', ha='left', va='top', fontsize=9, color='black')
        ax.text(alpha1 + nudge, yt, r'$\alpha_1$', ha='left', va='top', fontsize=9, color='blue')
        ax.text(alpha2 + nudge, yt, r'$\alpha_2$', ha='left', va='top', fontsize=9, color='blue')

    plt.tight_layout()

    # Compute quantities for the caption
    G0, B0 = 1.0, params.BperG
    G_rem_nested_a2 = nested['GLOs_R2'][-1]
    B_rem_nested_a2 = nested['BLOs_R2'][-1]
    B_rem_iid = iid['B_eq'][-1]
    print(f"\n  [Caption quantities]")
    print(f"    Nested: remaining G at alpha2 = {G_rem_nested_a2:.3f}, remaining B at alpha2 = {B_rem_nested_a2:.3f}")
    print(f"    These are served at the non-selective rate r_NS = {r_NS:.4f}")
    print(f"    Ind Inf: unserved bad at alpha=1 (dashed red) = {B_rem_iid:.3f}")

    return fig


if __name__ == "__main__":
    params = Parameters()

    print("Solving nested (analytical)...")
    nested = solve_nested_analytical(params)
    print(f"  alpha0={nested['alpha0']:.4f}, alpha1={nested['alpha1']:.4f}, alpha2={nested['alpha2']:.4f}, rp={nested['rp']:.4f}")

    print("Solving Ind Inf...")
    iid = solve_iid(params)
    print(f"  alpha0={iid['alpha0']:.4f}, alpha_bar={iid['alpha_bar']:.4f}, r0={iid['r0']:.4f}")

    avg_r_nested, avg_r_iid, _, _ = compute_avg_good_rate(params, nested, iid)
    print(f"Avg rate (good): nested={avg_r_nested:.4f}, iid={avg_r_iid:.4f}")

    fig = plot_paper_figure(params, nested, iid)

    # Save to paper figures directory
    output_path = r"c:\Dropbox\projects-Dropbox\The-pablo-project\Peter-Pablo-Maryam\figures\credit_model.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {output_path}")
