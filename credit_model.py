"""
Credit Market Equilibrium: Nested vs IID Information Structures
Translation from MATLAB code integrated.m
"""
import numpy as np
from scipy.optimize import minimize_scalar, brentq
from scipy.integrate import quad
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PRIMITIVE FUNCTIONS (easily modifiable)
# =============================================================================

def cfun(alpha):
    """Screening cost function c(alpha)."""
    return 0.8 * alpha**2 + 0.5 * alpha

def cfun_prime(alpha, eps=1e-5):
    """First derivative of cost function (numerical)."""
    return (cfun(alpha + eps) - cfun(alpha)) / eps

def cfun_prime2(alpha, eps=1e-5):
    """Second derivative of cost function (numerical)."""
    return (cfun_prime(alpha + eps) - cfun_prime(alpha)) / eps

def dfun(r0):
    """Loan demand function D(r)."""
    return 1.0 / r0

def gpriorfun(om):
    """Prior density of good borrowers (uniform on [0,1])."""
    return np.ones_like(np.atleast_1d(om)).astype(float) if not np.isscalar(om) else 1.0

def bpriorfun(om, BperG):
    """Prior density of bad borrowers (uniform, scaled by BperG)."""
    return BperG * np.ones_like(np.atleast_1d(om)).astype(float) if not np.isscalar(om) else BperG

# =============================================================================
# PARAMETERS
# =============================================================================

@dataclass
class Parameters:
    """Model parameters."""
    Pi: float = 0.05      # Profit margin
    beta: float = 0.1     # Signal precision parameter was 0.3
    BperG: float = 0.2    # Ratio of bad to good borrowers
    Delta: float = 0.001  # Step size for alpha iteration
    delom: float = 0.0001 # Step size for omega grid

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def gam0(alpha, params):
    """Gamma at alpha with fresh pool."""
    beta, BperG = params.beta, params.BperG
    omega_g = beta + alpha * (1 - beta)
    omega_b = 1 - beta + alpha * beta
    G, _ = quad(gpriorfun, 0, omega_g)
    B, _ = quad(lambda x: bpriorfun(x, BperG), omega_b, 1)
    return G / (G + B)

def NSfun(al, beta, badleftover, Pi, BperG):
    """No-screening equilibrium condition."""
    omega_g = beta + al * (1 - beta)
    goodleftover, _ = quad(gpriorfun, omega_g, 1)
    if goodleftover + badleftover < 1e-12:
        return 1e10
    gammaNS = goodleftover / (goodleftover + badleftover)
    return (gammaNS * (1 + cfun(al) + Pi) - (1 + Pi)) ** 2

# =============================================================================
# NESTED INFORMATION STRUCTURE
# =============================================================================

def solve_nested(params):
    """
    Solve the model under nested information structure.
    
    Returns equilibrium with three regions:
    - Region I: alpha in [alpha0, alpha1], flat rate rp
    - Region II: alpha in [alpha1, alpha2], rate r(alpha) = c(alpha) + Pi
    - Region III: non-selective lenders (alpha=0) at rate r_NS
    """
    Pi, beta, BperG = params.Pi, params.beta, params.BperG
    Delta, delom = params.Delta, params.delom
    
    # =========================================================================
    # Find alpha0 and rp (entry margin)
    # =========================================================================
    res = minimize_scalar(
        lambda a: (Pi + cfun(a) + 1) / gam0(a, params) if 0 < a < 1 else 1e10,
        bounds=(0.01, 0.99), method='bounded'
    )
    alpha0 = res.x
    rp = (Pi + cfun(alpha0) + 1) / gam0(alpha0, params) - 1
    
    # =========================================================================
    # Find alpha1: where c(alpha) = rp - Pi
    # =========================================================================
    if rp - cfun(1) > Pi:
        alpha1 = 1.0
    else:
        alpha1 = brentq(lambda a: cfun(a) - (rp - Pi), 0.01, 0.99)
    
    # =========================================================================
    # Region I: Solve for w(alpha) distribution
    # =========================================================================
    n_om = int(1 / delom)
    omvec = np.linspace(0, 1, n_om)
    g, b = gpriorfun(omvec), bpriorfun(omvec, BperG)
    D_rp = dfun(rp)
    
    alphas_R1, ws_R1, gammas_R1, GLOs_R1, BLOs_R1 = [alpha0], [], [], [], []
    alpha = alpha0
    
    while alpha + Delta <= alpha1 + 1e-8:
        omega_g_alp = beta + alpha * (1 - beta)
        omega_b_alp = 1 - beta + alpha * beta
        mask_g, mask_b = omvec <= omega_g_alp, omvec >= omega_b_alp
        
        G_alp = np.sum(delom * g[mask_g])
        B_alp = np.sum(delom * b[mask_b])
        T_alp = G_alp + B_alp
        if T_alp < 1e-10:
            break
        
        gammas_R1.append(G_alp / T_alp)
        alpha_next = min(alpha + Delta, alpha1)
        req_gamma = (1 + Pi + cfun(alpha_next)) / (1 + rp)
        
        omega_g_next = beta + alpha_next * (1 - beta)
        omega_b_next = 1 - beta + alpha_next * beta
        
        def gamma_after_w(w, mask_g=mask_g, mask_b=mask_b, T_alp=T_alp):
            scale = w / (T_alp * D_rp)
            if scale >= 1: return 1.0
            g_new = g * (1 - mask_g * scale)
            b_new = b * (1 - mask_b * scale)
            G_n = np.sum(delom * g_new[omvec <= omega_g_next])
            B_n = np.sum(delom * b_new[omvec >= omega_b_next])
            return G_n / (G_n + B_n) if G_n + B_n > 1e-10 else 1.0
        
        g0, gmax = gamma_after_w(0), gamma_after_w(T_alp * D_rp * 0.999)
        if req_gamma <= g0:
            w_opt = 0
        elif req_gamma >= gmax:
            w_opt = T_alp * D_rp * 0.999
        else:
            w_opt = brentq(lambda w: gamma_after_w(w) - req_gamma, 0, T_alp * D_rp * 0.999)
        
        ws_R1.append(w_opt)
        scale = w_opt / (T_alp * D_rp)
        g, b = g * (1 - mask_g * scale), b * (1 - mask_b * scale)
        GLOs_R1.append(np.sum(delom * g))
        BLOs_R1.append(np.sum(delom * b))
        alphas_R1.append(alpha_next)
        alpha = alpha_next
    
    # State at end of Region I
    G_end_R1 = GLOs_R1[-1] if GLOs_R1 else np.sum(delom * g)
    B_end_R1 = BLOs_R1[-1] if BLOs_R1 else np.sum(delom * b)
    badleftover = B_end_R1
    
    # =========================================================================
    # Find alpha2 and WNS (non-selective lenders)
    # =========================================================================
    res = minimize_scalar(
        lambda al: NSfun(al, beta, badleftover, Pi, BperG),
        bounds=(alpha1, 1.0), method='bounded'
    )
    alpha2 = res.x
    
    if NSfun(alpha2, beta, badleftover, Pi, BperG) < Delta:
        omega_g_alpha2 = beta + alpha2 * (1 - beta)
        goodleftover_alpha2, _ = quad(gpriorfun, omega_g_alpha2, 1)
        WNS = dfun(cfun(alpha2) + Pi) * (badleftover + goodleftover_alpha2)
    else:
        WNS, alpha2 = 0, 1.0
    
    r_NS = cfun(alpha2) + Pi
    
    # =========================================================================
    # Region II: alpha1 to alpha2
    # =========================================================================
    n_R2 = max(int((alpha2 - alpha1) / Delta), 10)
    alphas_R2 = np.linspace(alpha1, alpha2, n_R2)
    da_R2 = alphas_R2[1] - alphas_R2[0] if n_R2 > 1 else Delta
    
    gammas_R2 = []
    GLOs_R2 = []
    BLOs_R2 = []
    ws_R2 = []
    
    G_remaining = G_end_R1
    B_remaining = B_end_R1
    
    for i, al in enumerate(alphas_R2):
        # In Region II, gamma = 1 (only good borrowers served)
        gammas_R2.append(1.0)
        
        # Capital density: w(alpha) = D(r(alpha)) * g(omega_g(alpha)) * (1-beta)
        r_al = cfun(al) + Pi
        D_al = dfun(r_al)
        w_al = D_al * (1 - beta) * 1.0  # g=1 for uniform
        ws_R2.append(w_al)
        
        # Update remaining good: borrowers served = w/D * dalpha = (1-beta) * dalpha
        if i > 0:
            borrowers_served = (1 - beta) * da_R2
            G_remaining -= borrowers_served
        
        GLOs_R2.append(max(G_remaining, 0))
        BLOs_R2.append(B_remaining)
    
    ws_R2 = np.array(ws_R2)
    
    # =========================================================================
    # Compute cumulative capital
    # =========================================================================
    W_cumsum_R1 = np.cumsum(ws_R1)
    W_R2_cumsum = W_cumsum_R1[-1] + np.cumsum(ws_R2) * da_R2 if len(W_cumsum_R1) > 0 else np.cumsum(ws_R2) * da_R2
    
    return {
        'alpha0': alpha0, 'alpha1': alpha1, 'alpha2': alpha2,
        'rp': rp, 'r_NS': r_NS,
        'WNS': WNS,
        # Region I
        'alphas_R1': np.array(alphas_R1),
        'ws_R1': np.array(ws_R1),
        'gammas_R1': np.array(gammas_R1),
        'GLOs_R1': np.array(GLOs_R1),
        'BLOs_R1': np.array(BLOs_R1),
        'W_cumsum_R1': W_cumsum_R1,
        # Region II
        'alphas_R2': np.array(alphas_R2),
        'ws_R2': ws_R2,
        'gammas_R2': np.array(gammas_R2),
        'GLOs_R2': np.array(GLOs_R2),
        'BLOs_R2': np.array(BLOs_R2),
        'W_R2_cumsum': W_R2_cumsum,
        # Leftover
        'badleftover': badleftover,
        'G_end_R1': G_end_R1,
        'B_end_R1': B_end_R1,
    }

# =============================================================================
# IID INFORMATION STRUCTURE
# =============================================================================

def solve_iid(params):
    """Solve the model under IID information structure using scalar ODE."""
    Pi, beta, BperG = params.Pi, params.beta, params.BperG
    
    z0 = 1 / (1 + BperG)
    G0, B0 = 1.0, BperG
    
    h = lambda a, z: beta * (1 - a) + a * z
    mu = lambda a: beta + a * (1 - beta)
    
    def gamma_f(a, z):
        denom = z * mu(a) + (1 - z) * beta * (1 - a)
        return z * mu(a) / denom if denom > 1e-12 else 0
    
    def r_f(a, z):
        gam = gamma_f(a, z)
        return (Pi + 1 + cfun(a)) / gam - 1 if gam > 1e-12 else 1e10
    
    r_perfect = cfun(1) + Pi
    
    def zprime(a, z):
        Cp, Cpp = cfun_prime(a), cfun_prime2(a)
        if abs(Cp) < 1e-12:
            return 0
        return (Cpp / Cp * h(a, z) + 2 * (z - beta)) * (z - 1) / mu(a)
    
    # Find alpha0
    alpha0 = minimize_scalar(
        lambda a: (Pi + 1 + cfun(a)) / gamma_f(a, z0) if gamma_f(a, z0) > 1e-12 else 1e10,
        bounds=(0.01, 0.99), method='bounded'
    ).x
    r0 = r_f(alpha0, z0)
    
    # Euler method for ODE
    n_steps = 50000
    alpha_path = np.linspace(alpha0, 0.9999, n_steps)
    da = alpha_path[1] - alpha_path[0]
    
    z_path = np.zeros(n_steps)
    r_path = np.zeros(n_steps)
    G_path = np.zeros(n_steps)
    B_path = np.zeros(n_steps)
    w_over_D = np.zeros(n_steps)
    
    z_path[0], r_path[0], G_path[0], B_path[0] = z0, r0, G0, B0
    alpha_bar_idx = n_steps - 1
    
    for k in range(1, n_steps):
        a, z, G, B = alpha_path[k-1], z_path[k-1], G_path[k-1], B_path[k-1]
        
        zp = zprime(a, z)
        z_path[k] = z + zp * da
        r_path[k] = r_f(alpha_path[k], z_path[k])
        
        if r_path[k] >= r_perfect and alpha_bar_idx == n_steps - 1:
            alpha_bar_idx = k
        
        gam = gamma_f(a, z)
        wD = zp * (G + B) / (z - gam) if abs(z - gam) > 1e-12 and (G + B) > 1e-12 else 0
        w_over_D[k-1] = wD
        
        G_path[k] = max(G - wD * gam * da, 0)
        B_path[k] = max(B - wD * (1 - gam) * da, 0)
    
    # Truncate to barrier
    idx = alpha_bar_idx + 1
    alpha_eq = alpha_path[:idx]
    z_eq = z_path[:idx]
    r_eq = r_path[:idx]
    G_eq = G_path[:idx]
    B_eq = B_path[:idx]
    
    gamma_eq = np.array([gamma_f(alpha_eq[i], z_eq[i]) for i in range(len(alpha_eq))])
    w_eq = np.array([w_over_D[k] * dfun(r_eq[k]) if r_eq[k] > 1e-12 else 0 for k in range(len(alpha_eq))])
    W_iid_cumsum = np.cumsum(w_eq) * da
    
    return {
        'alpha0': alpha0, 'alpha_bar': alpha_path[alpha_bar_idx],
        'r0': r0, 'r_perfect': r_perfect,
        'alpha_eq': alpha_eq, 'z_eq': z_eq, 'r_eq': r_eq,
        'G_eq': G_eq, 'B_eq': B_eq,
        'gamma_eq': gamma_eq, 'w_eq': w_eq,
        'W_cumsum': W_iid_cumsum,
        'z0': z0, 'G0': G0, 'B0': B0, 'da': da
    }

# =============================================================================
# PLOTTING
# =============================================================================

def plot_comparison(params, nested, iid):
    """Create comparison plots for nested vs IID equilibria."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    alpha0 = nested['alpha0']
    alpha1 = nested['alpha1']
    alpha2 = nested['alpha2']
    rp = nested['rp']
    r_NS = nested['r_NS']
    WNS = nested['WNS']
    
    # =========================================================================
    # Panel 1: Interest Rate r(alpha)
    # =========================================================================
    # Region I: flat at rp
    alpha_r1 = np.linspace(alpha0, alpha1, 100)
    r_r1 = rp * np.ones_like(alpha_r1)
    # Region II: r = c(alpha) + Pi
    alpha_r2 = np.linspace(alpha1, alpha2, 100)
    r_r2 = cfun(alpha_r2) + params.Pi
    
    axes[0,0].plot(alpha_r1, r_r1, 'b-', lw=2, label='Nested (selective)')
    axes[0,0].plot(alpha_r2, r_r2, 'b-', lw=2)
    # Region III: atom at alpha=0
    axes[0,0].plot(0, r_NS, 'bo', markersize=10, markerfacecolor='blue', label='Nested (non-selective)')
    # IID
    axes[0,0].plot(iid['alpha_eq'], iid['r_eq'], 'r-', lw=2, label='IID')
    
    axes[0,0].axvline(alpha0, color='black', ls=':', alpha=0.3)
    axes[0,0].axvline(alpha1, color='blue', ls=':', alpha=0.5)
    axes[0,0].axvline(alpha2, color='blue', ls=':', alpha=0.5)
    axes[0,0].set_xlabel(r'$\alpha$'); axes[0,0].set_ylabel(r'$r(\alpha)$')
    axes[0,0].set_title('Interest Rate')
    axes[0,0].legend(loc='upper left', fontsize=9)
    axes[0,0].grid(alpha=0.3)
    axes[0,0].set_xlim([0, 1]); axes[0,0].set_ylim([0, 1.5])
    axes[0,0].text((alpha0+alpha1)/2, rp - 0.06, 'I', fontsize=12, color='blue')
    axes[0,0].text((alpha1+alpha2)/2, 0.45, 'II', fontsize=12, color='blue')
    axes[0,0].text(0.03, r_NS + 0.05, 'III', fontsize=12, color='blue')
    
    # =========================================================================
    # Panel 2: Pool Quality gamma(alpha)
    # =========================================================================
    axes[0,1].plot(nested['alphas_R1'][:-1], nested['gammas_R1'], 'b-', lw=2, label='Nested')
    axes[0,1].plot(nested['alphas_R2'], nested['gammas_R2'], 'b-', lw=2)
    axes[0,1].plot(iid['alpha_eq'], iid['gamma_eq'], 'r-', lw=2, label='IID')
    
    axes[0,1].axvline(alpha0, color='black', ls=':', alpha=0.3)
    axes[0,1].axvline(alpha1, color='blue', ls=':', alpha=0.5)
    axes[0,1].axvline(alpha2, color='blue', ls=':', alpha=0.5)
    axes[0,1].set_xlabel(r'$\alpha$'); axes[0,1].set_ylabel(r'$\gamma(\alpha)$')
    axes[0,1].set_title('Pool Quality (selective lenders)')
    axes[0,1].legend()
    axes[0,1].grid(alpha=0.3)
    axes[0,1].set_xlim([0, 1])
    
    # =========================================================================
    # Panel 3: Cumulative Capital W(alpha)
    # =========================================================================
    # Nested: atom at alpha=0, then flat until alpha0, then accumulate
    # Jump at alpha=0
    axes[1,0].plot([0, 0], [0, WNS], 'b-', lw=2)
    axes[1,0].plot(0, 0, 'bo', markersize=8, markerfacecolor='white', markeredgewidth=2)
    axes[1,0].plot(0, WNS, 'bo', markersize=8, markerfacecolor='blue', label='Nested')
    # Flat from 0 to alpha0
    alpha_flat = np.linspace(0, alpha0, 20)
    axes[1,0].plot(alpha_flat, WNS * np.ones_like(alpha_flat), 'b-', lw=2)
    # Region I
    axes[1,0].plot(nested['alphas_R1'][1:], WNS + nested['W_cumsum_R1'], 'b-', lw=2)
    # Region II
    axes[1,0].plot(nested['alphas_R2'], WNS + nested['W_R2_cumsum'], 'b-', lw=2)
    
    # IID
    alpha_before_iid = np.linspace(0, iid['alpha0'], 20)
    axes[1,0].plot(alpha_before_iid, np.zeros_like(alpha_before_iid), 'r-', lw=2, label='IID')
    axes[1,0].plot(iid['alpha_eq'], iid['W_cumsum'], 'r-', lw=2)
    
    axes[1,0].axvline(alpha0, color='black', ls=':', alpha=0.3)
    axes[1,0].axvline(alpha1, color='blue', ls=':', alpha=0.5)
    axes[1,0].axvline(alpha2, color='blue', ls=':', alpha=0.5)
    axes[1,0].set_xlabel(r'$\alpha$'); axes[1,0].set_ylabel(r'$W(\alpha)$')
    axes[1,0].set_title('Cumulative Capital')
    axes[1,0].legend()
    axes[1,0].grid(alpha=0.3)
    axes[1,0].set_xlim([0, 1])
    
    # =========================================================================
    # Panel 4: Remaining Borrowers
    # =========================================================================
    axes[1,1].plot(nested['alphas_R1'][:-1], nested['GLOs_R1'], 'b-', lw=2, label='Nested G')
    axes[1,1].plot(nested['alphas_R1'][:-1], nested['BLOs_R1'], 'b--', lw=2, label='Nested B')
    axes[1,1].plot(nested['alphas_R2'], nested['GLOs_R2'], 'b-', lw=2)
    axes[1,1].plot(nested['alphas_R2'], nested['BLOs_R2'], 'b--', lw=2)
    axes[1,1].plot(iid['alpha_eq'], iid['G_eq'], 'r-', lw=2, label='IID G')
    axes[1,1].plot(iid['alpha_eq'], iid['B_eq'], 'r--', lw=2, label='IID B')
    
    axes[1,1].axvline(alpha0, color='black', ls=':', alpha=0.3)
    axes[1,1].axvline(alpha1, color='blue', ls=':', alpha=0.5)
    axes[1,1].axvline(alpha2, color='blue', ls=':', alpha=0.5)
    axes[1,1].set_xlabel(r'$\alpha$'); axes[1,1].set_ylabel('Mass')
    axes[1,1].set_title('Remaining Borrowers')
    axes[1,1].legend()
    axes[1,1].grid(alpha=0.3)
    axes[1,1].set_xlim([0, 1])
    
    plt.tight_layout()
    return fig

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function to run the model comparison."""
    params = Parameters()
    
    print("=" * 60)
    print("CREDIT MARKET: NESTED vs IID INFORMATION STRUCTURES")
    print("=" * 60)
    print(f"Parameters: Pi={params.Pi}, beta={params.beta}, BperG={params.BperG}")
    print("=" * 60)
    
    # Solve nested
    print("\n--- Solving Nested Model ---")
    nested = solve_nested(params)
    print(f"alpha0 = {nested['alpha0']:.4f}")
    print(f"alpha1 = {nested['alpha1']:.4f}")
    print(f"alpha2 = {nested['alpha2']:.4f}")
    print(f"rp = {nested['rp']:.4f}")
    print(f"r_NS = {nested['r_NS']:.4f}")
    print(f"WNS = {nested['WNS']:.4f}")
    print(f"Total W (Regions I+II) = {nested['W_R2_cumsum'][-1]:.4f}")
    print(f"Total W (including NS) = {nested['W_R2_cumsum'][-1] + nested['WNS']:.4f}")
    
    # Solve IID
    print("\n--- Solving IID Model ---")
    iid = solve_iid(params)
    print(f"alpha0 = {iid['alpha0']:.4f}")
    print(f"alpha_bar = {iid['alpha_bar']:.4f}")
    print(f"r0 = {iid['r0']:.4f}")
    print(f"r_perfect = {iid['r_perfect']:.4f}")
    print(f"Total W = {iid['W_cumsum'][-1]:.4f}")
    
    # Plot
    print("\n--- Creating Plot ---")
    fig = plot_comparison(params, nested, iid)
    
    # Save in same folder as script (works on Windows and Linux)
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'credit_model.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    return params, nested, iid

if __name__ == "__main__":
    params, nested, iid = main()
