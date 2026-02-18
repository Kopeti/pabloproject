"""
Credit Market Equilibrium Model: Nested vs IID Information Structures

Translated from MATLAB code integrated.m
"""

import numpy as np
from scipy.optimize import fsolve, minimize_scalar
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
    Ca = 0.8   # Quadratic coefficient
    Cb = 0.5   # Linear coefficient
    Cpower = 2
    return Ca * np.power(alpha, Cpower) + Cb * alpha


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
    return np.ones_like(np.atleast_1d(om)).astype(float)


def bpriorfun(om, BperG):
    """Prior density of bad borrowers (uniform, scaled by BperG)."""
    return BperG * np.ones_like(np.atleast_1d(om)).astype(float)


# =============================================================================
# MODEL PARAMETERS
# =============================================================================

@dataclass
class Parameters:
    """Model parameters."""
    Pi: float = 0.05       # Required profit margin
    beta: float = 0.3      # Signal precision parameter
    BperG: float = 0.2     # Proportion of bad to good borrowers
    Delta: float = 0.001   # Grid step for alpha
    delom: float = 0.0001  # Grid step for omega


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def gam0(alpha, params):
    """Fraction of good borrowers at alpha when pool is fresh."""
    beta = params.beta
    BperG = params.BperG
    
    omega_g = beta + alpha * (1 - beta)
    omega_b = 1 - beta + alpha * beta
    
    G_acc, _ = quad(gpriorfun, 0, omega_g)
    B_acc, _ = quad(lambda x: bpriorfun(x, BperG), omega_b, 1)
    
    return G_acc / (G_acc + B_acc)


def gamfun(w, al, alp, gfunprev, bfunprev, omvec, delom, beta, rp):
    """Fraction of good borrowers for alpha-type if at alp capital w comes in."""
    omega_g_alp = beta + alp * (1 - beta)
    omega_b_alp = 1 - beta + alp * beta
    
    G_acc = np.sum(delom * gfunprev[omvec <= omega_g_alp])
    B_acc = np.sum(delom * bfunprev[omvec >= omega_b_alp])
    T_acc = G_acc + B_acc
    
    scale = w / (T_acc * dfun(rp))
    g = gfunprev * (1 - (omvec <= omega_g_alp) * scale)
    b = bfunprev * (1 - (omvec >= omega_b_alp) * scale)
    
    omega_g_al = beta + al * (1 - beta)
    omega_b_al = 1 - beta + al * beta
    
    G_new = np.sum(delom * g[omvec <= omega_g_al])
    B_new = np.sum(delom * b[omvec >= omega_b_al])
    
    if G_new + B_new < 1e-12:
        return 0.0
    return G_new / (G_new + B_new)


def profit(w, alpha, alp, gfunprev, bfunprev, omvec, delom, beta, rp, Pi):
    """Gross profit (1+pi) as function of alpha."""
    gam = gamfun(w, alpha, alp, gfunprev, bfunprev, omvec, delom, beta, rp)
    return (1 + rp) * gam - cfun(alpha)


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
    """Solve the model under nested information structure."""
    Pi, beta, BperG = params.Pi, params.beta, params.BperG
    Delta, delom = params.Delta, params.delom
    
    # Find alpha0 and rp
    def obj_alpha0(alpha):
        if alpha <= 0 or alpha >= 1:
            return 1e10
        g = gam0(alpha, params)
        return (Pi + cfun(alpha) + 1) / g if g > 1e-12 else 1e10
    
    result = minimize_scalar(obj_alpha0, bounds=(0.01, 0.99), method='bounded')
    alpha0 = result.x
    rp = (Pi + cfun(alpha0) + 1) / gam0(alpha0, params) - 1
    
    print(f"Nested: alpha0 = {alpha0:.6f}, rp = {rp:.6f}")
    
    # Find alpha1
    if (1 + rp) - 1 - cfun(1) > Pi:
        alpha1 = 1.0
    else:
        alpha1 = fsolve(lambda a: cfun(a) - (rp - Pi), alpha0)[0]
    
    print(f"Nested: alpha1 = {alpha1:.6f}")
    
    # Set up omega grid
    omvec = np.linspace(0, 1, int(1/delom))
    gfunprev = gpriorfun(omvec)
    bfunprev = bpriorfun(omvec, BperG)
    
    wmass, almass = [1.0], [alpha0]
    GLO, BLO = [1.0], [BperG]
    gammab = []
    
    # Initial gamma
    omega_g0 = beta + alpha0 * (1 - beta)
    omega_b0 = 1 - beta + alpha0 * beta
    G_acc = np.sum(delom * gfunprev[omvec <= omega_g0])
    B_acc = np.sum(delom * bfunprev[omvec >= omega_b0])
    gammab.append(G_acc / (G_acc + B_acc))
    
    n = 1
    gnext, bnext = gfunprev.copy(), bfunprev.copy()
    
    # Main iteration loop
    while almass[-1] + Delta <= alpha1:
        if n >= 2:
            gfunprev, bfunprev = gnext.copy(), bnext.copy()
        
        alp = almass[-1]
        omega_g_alp = beta + alp * (1 - beta)
        omega_b_alp = 1 - beta + alp * beta
        
        G_acc = np.sum(delom * gfunprev[omvec <= omega_g_alp])
        B_acc = np.sum(delom * bfunprev[omvec >= omega_b_alp])
        T_acc = G_acc + B_acc
        
        if n >= 2:
            gammab.append(G_acc / (G_acc + B_acc) if T_acc > 1e-12 else 0)
        
        # Find wmax
        def bfunmin(w):
            scale = w / (T_acc * dfun(rp))
            return np.min(bfunprev * (1 - (omvec >= omega_b_alp) * scale))
        
        try:
            wmax = max(0.01, min(fsolve(bfunmin, 1.0)[0], 10.0))
        except:
            wmax = 1.0
        
        # Find optimal w and alpha
        alopt3 = alp + Delta
        
        def obj_w3(w):
            p = profit(w, alopt3, alp, gfunprev, bfunprev, omvec, delom, beta, rp, Pi)
            return 1000 * (p - (1 + Pi)) ** 2
        
        wopt3 = minimize_scalar(obj_w3, bounds=(0.001, wmax), method='bounded').x
        
        def obj_al3b(alpha):
            gam = gamfun(wopt3, alpha, alp, gfunprev, bfunprev, omvec, delom, beta, rp)
            return 1000 * (cfun(alpha) - (1 + rp) * gam) ** 2
        
        alopt3b = minimize_scalar(obj_al3b, bounds=(alp + Delta, alpha1), method='bounded').x
        
        wopt = wopt3
        alopt = alopt3 if abs(alopt3 - alopt3b) < 10 * Delta else alopt3b
        
        almass.append(alopt)
        
        # Update distributions
        scale = wopt / (T_acc * dfun(rp))
        gnext = gfunprev * (1 - (omvec <= omega_g_alp) * scale)
        bnext = bfunprev * (1 - (omvec >= omega_b_alp) * scale)
        
        GLO.append(np.sum(delom * gnext))
        BLO.append(np.sum(delom * bnext))
        wmass.append(wopt)
        
        n += 1
        if n % 100 == 0:
            print(f"  Iteration {n}, alpha = {alopt:.4f}")
    
    # Final gamma
    G_acc = np.sum(delom * gfunprev[omvec <= omega_g_alp])
    B_acc = np.sum(delom * bfunprev[omvec >= omega_b_alp])
    gammab.append(G_acc / (G_acc + B_acc) if G_acc + B_acc > 1e-12 else 0)
    
    # CIM and NS segment
    badleftover = np.sum(delom * bnext)
    
    result = minimize_scalar(lambda al: NSfun(al, beta, badleftover, Pi, BperG),
                             bounds=(alpha1, 1.0), method='bounded')
    alpha2 = result.x
    
    if NSfun(alpha2, beta, badleftover, Pi, BperG) < Delta:
        omega_g_alpha2 = beta + alpha2 * (1 - beta)
        goodleftover, _ = quad(gpriorfun, omega_g_alpha2, 1)
        WNS = dfun(cfun(alpha2) + Pi) * (badleftover + goodleftover)
    else:
        WNS, alpha2 = 0, 1.0
    
    print(f"Nested: alpha2 = {alpha2:.6f}, WNS = {WNS:.4f}")
    
    return {
        'alpha0': alpha0, 'alpha1': alpha1, 'alpha2': alpha2, 'rp': rp,
        'wmass': np.array(wmass[1:]), 'almass': np.array(almass),
        'GLO': np.array(GLO), 'BLO': np.array(BLO), 'gammab': np.array(gammab),
        'WNS': WNS, 'badleftover': badleftover,
        'gnext': gnext, 'bnext': bnext, 'omvec': omvec, 'delom': delom
    }


# =============================================================================
# IID INFORMATION STRUCTURE
# =============================================================================

def solve_iid(params):
    """Solve the model under IID information structure using scalar ODE."""
    Pi, beta, BperG = params.Pi, params.beta, params.BperG
    
    z0 = 1 / (1 + BperG)
    G0, _ = quad(gpriorfun, 0, 1)
    B0, _ = quad(lambda x: bpriorfun(x, BperG), 0, 1)
    
    print(f"\nIID: z0 = {z0:.4f}, G0 = {G0:.2f}, B0 = {B0:.2f}")
    
    h_fun = lambda a, z: beta * (1 - a) + a * z
    mu_fun = lambda a: beta + a * (1 - beta)
    
    def gamma_fun(a, z):
        num = z * (beta + a * (1 - beta))
        denom = num + (1 - z) * beta * (1 - a)
        return num / denom if denom > 1e-12 else 0
    
    def r_fun(a, z):
        gam = gamma_fun(a, z)
        return (Pi + 1 + cfun(a)) / gam - 1 if gam > 1e-12 else 1e10
    
    r_perfect = cfun(1) + Pi
    print(f"IID: r_perfect = {r_perfect:.4f}")
    
    def z_prime_fun(a, z):
        Cp, Cpp = cfun_prime(a), cfun_prime2(a)
        if abs(Cp) < 1e-12:
            return 0
        return (Cpp / Cp * h_fun(a, z) + 2 * (z - beta)) * (z - 1) / mu_fun(a)
    
    # Find alpha0
    def obj_alpha0(a):
        gam = gamma_fun(a, z0)
        return (Pi + 1 + cfun(a)) / gam if gam > 1e-12 else 1e10
    
    alpha0 = minimize_scalar(obj_alpha0, bounds=(0.01, 0.99), method='bounded').x
    r0 = r_fun(alpha0, z0)
    print(f"IID: alpha0 = {alpha0:.6f}, r0 = {r0:.6f}")
    
    # Euler method for ODE
    n_steps = 50000
    alpha_path = np.linspace(alpha0, 0.9999, n_steps)
    da = alpha_path[1] - alpha_path[0]
    
    z_path = np.zeros(n_steps)
    r_path = np.zeros(n_steps)
    G_path = np.zeros(n_steps)
    B_path = np.zeros(n_steps)
    w_over_D_path = np.zeros(n_steps)
    zprime_path = np.zeros(n_steps)
    
    z_path[0], r_path[0], G_path[0], B_path[0] = z0, r0, G0, B0
    alpha_bar_idx = n_steps - 1
    
    for k in range(1, n_steps):
        a, z, G, B = alpha_path[k-1], z_path[k-1], G_path[k-1], B_path[k-1]
        
        zprime = z_prime_fun(a, z)
        zprime_path[k-1] = zprime
        z_path[k] = z + zprime * da
        r_path[k] = r_fun(alpha_path[k], z_path[k])
        
        if r_path[k] >= r_perfect and alpha_bar_idx == n_steps - 1:
            alpha_bar_idx = k
            print(f"IID: Barrier at alpha_bar = {alpha_path[k]:.6f}")
        
        gam = gamma_fun(a, z)
        w_over_D = zprime * (G + B) / (z - gam) if abs(z - gam) > 1e-12 and (G + B) > 1e-12 else 0
        w_over_D_path[k-1] = w_over_D
        
        G_path[k] = max(G - w_over_D * gam * da, 0)
        B_path[k] = max(B - w_over_D * (1 - gam) * da, 0)
    
    zprime_path[-1] = z_prime_fun(alpha_path[-1], z_path[-1])
    
    # Truncate to barrier
    idx = alpha_bar_idx + 1
    alpha_eq, z_eq, r_eq = alpha_path[:idx], z_path[:idx], r_path[:idx]
    G_eq, B_eq = G_path[:idx], B_path[:idx]
    w_over_D_eq, zprime_eq = w_over_D_path[:idx], zprime_path[:idx]
    
    gamma_eq = np.array([gamma_fun(alpha_eq[i], z_eq[i]) for i in range(len(alpha_eq))])
    
    w_eq = np.zeros_like(alpha_eq)
    for k in range(len(alpha_eq) - 1):
        if r_eq[k] > 1e-12:
            w_eq[k] = w_over_D_eq[k] * dfun(r_eq[k])
    
    print(f"IID: Equilibrium alpha in [{alpha0:.4f}, {alpha_path[alpha_bar_idx]:.4f}]")
    
    return {
        'alpha0': alpha0, 'alpha_bar': alpha_path[alpha_bar_idx],
        'r0': r0, 'r_perfect': r_perfect,
        'alpha_eq': alpha_eq, 'z_eq': z_eq, 'r_eq': r_eq,
        'G_eq': G_eq, 'B_eq': B_eq, 'w_over_D_eq': w_over_D_eq,
        'gamma_eq': gamma_eq, 'w_eq': w_eq, 'zprime_eq': zprime_eq,
        'z0': z0, 'G0': G0, 'B0': B0, 'da': da
    }


# =============================================================================
# COMPARISON AND PLOTTING
# =============================================================================

def compare_w_at_alpha0(params, nested, iid):
    """Compare w(alpha_0) under each information structure."""
    beta, BperG, Pi, Delta = params.beta, params.BperG, params.Pi, params.Delta
    alpha0, rp, z0 = nested['alpha0'], nested['rp'], iid['z0']
    
    omega_g0 = beta + alpha0 * (1 - beta)
    gbar = float(gpriorfun(omega_g0)[0])
    bbar = float(bpriorfun(1 - beta + alpha0 * beta, BperG)[0])
    D_rp = dfun(rp)
    h0 = beta * (1 - alpha0) + alpha0 * z0
    
    w_nested_formula = ((1 - beta) * gbar - beta * bbar) * D_rp
    w_nested_code = nested['wmass'][0] / Delta if len(nested['wmass']) > 0 else 0
    
    R0 = cfun_prime2(alpha0) / cfun_prime(alpha0)
    phi_0 = R0 * h0 + 2 * (z0 - beta)
    w_iid_formula = phi_0 * h0 * (gbar + bbar) * D_rp / (z0 * alpha0 * omega_g0)
    w_iid_code = iid['w_eq'][0] if len(iid['w_eq']) > 0 else 0
    
    print("\n" + "="*60)
    print("w(alpha_0) COMPARISON")
    print("="*60)
    print(f"Nested: formula = {w_nested_formula:.6f}, code = {w_nested_code:.6f}")
    print(f"IID: formula = {w_iid_formula:.6f}, code = {w_iid_code:.6f}")


def plot_comparison(params, nested, iid):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Interest rate
    axes[0,0].axhline(nested['rp'], color='blue', lw=2, label='Nested')
    axes[0,0].plot(iid['alpha_eq'], iid['r_eq'], 'r-', lw=2, label='IID')
    axes[0,0].axhline(iid['r_perfect'], color='gray', ls='--', label='r_perfect')
    axes[0,0].set_xlabel(r'$\alpha$'); axes[0,0].set_ylabel(r'$r(\alpha)$')
    axes[0,0].set_title('Interest Rate'); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)
    
    # Pool quality
    axes[0,1].plot(nested['almass'][:-1], nested['gammab'][:-1], 'b-', lw=2, label=r'Nested $\gamma$')
    axes[0,1].plot(iid['alpha_eq'], iid['gamma_eq'], 'r-', lw=2, label=r'IID $\gamma$')
    axes[0,1].plot(iid['alpha_eq'], iid['z_eq'], 'r--', lw=1, label='IID z')
    axes[0,1].set_xlabel(r'$\alpha$'); axes[0,1].set_ylabel('Fraction good')
    axes[0,1].set_title('Pool Quality'); axes[0,1].legend(); axes[0,1].grid(alpha=0.3)
    
    # Capital density
    w_nested = nested['wmass'] / np.diff(nested['almass'])
    axes[1,0].plot(nested['almass'][:-1], w_nested, 'b-', lw=2, label='Nested')
    axes[1,0].plot(iid['alpha_eq'], iid['w_eq'], 'r-', lw=2, label='IID')
    axes[1,0].set_xlabel(r'$\alpha$'); axes[1,0].set_ylabel(r'$w(\alpha)$')
    axes[1,0].set_title('Capital Density'); axes[1,0].legend(); axes[1,0].grid(alpha=0.3)
    
    # Remaining borrowers
    axes[1,1].plot(nested['almass'][:-1], nested['GLO'][:-1], 'b-', lw=2, label='Nested G')
    axes[1,1].plot(nested['almass'][:-1], nested['BLO'][:-1], 'b--', lw=2, label='Nested B')
    axes[1,1].plot(iid['alpha_eq'], iid['G_eq'], 'r-', lw=2, label='IID G')
    axes[1,1].plot(iid['alpha_eq'], iid['B_eq'], 'r--', lw=2, label='IID B')
    axes[1,1].set_xlabel(r'$\alpha$'); axes[1,1].set_ylabel('Mass')
    axes[1,1].set_title('Remaining Borrowers'); axes[1,1].legend(); axes[1,1].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """Main function."""
    params = Parameters(Pi=0.05, beta=0.3, BperG=0.2, Delta=0.001, delom=0.0001)
    
    print("="*60)
    print("CREDIT MARKET: NESTED vs IID INFORMATION")
    print("="*60)
    
    nested = solve_nested(params)
    iid = solve_iid(params)
    compare_w_at_alpha0(params, nested, iid)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Nested: α₀={nested['alpha0']:.4f}, α₁={nested['alpha1']:.4f}, rp={nested['rp']:.4f}")
    print(f"IID: α₀={iid['alpha0']:.4f}, ᾱ={iid['alpha_bar']:.4f}, r₀={iid['r0']:.4f}")
    
    fig = plot_comparison(params, nested, iid)
    plt.savefig('credit_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return params, nested, iid


if __name__ == "__main__":
    params, nested, iid = main()