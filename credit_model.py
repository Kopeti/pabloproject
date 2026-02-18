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
    # Prior shape parameters (0 = uniform)
    # g(omega) = (1+a_g) * omega^a_g    -- integrates to 1 on [0,1]
    # b(omega) = BperG * (1+a_b) * (1-omega)^a_b  -- integrates to BperG on [0,1]
    a_g: float = 0.05
    a_b: float = 0.05

# Global reference to current parameters (for prior functions)
_current_params = Parameters()

def gpriorfun(om):
    """Prior density of good borrowers: g(omega) = (1+a_g) * omega^a_g."""
    a_g = _current_params.a_g
    om_arr = np.atleast_1d(np.asarray(om, dtype=float))
    if a_g == 0:
        result = np.ones_like(om_arr)
    else:
        result = (1 + a_g) * np.power(np.maximum(om_arr, 0.0), a_g)
    return result if result.size > 1 else float(result[0])

def bpriorfun(om, BperG):
    """Prior density of bad borrowers: b(omega) = BperG * (1+a_b) * (1-omega)^a_b."""
    a_b = _current_params.a_b
    om_arr = np.atleast_1d(np.asarray(om, dtype=float))
    if a_b == 0:
        result = BperG * np.ones_like(om_arr)
    else:
        result = BperG * (1 + a_b) * np.power(np.maximum(1.0 - om_arr, 0.0), a_b)
    return result if result.size > 1 else float(result[0])

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
    """No-screening equilibrium condition (squared residual)."""
    omega_g = beta + al * (1 - beta)
    goodleftover, _ = quad(gpriorfun, omega_g, 1)
    if goodleftover + badleftover < 1e-12:
        return 1e10
    gammaNS = goodleftover / (goodleftover + badleftover)
    return (gammaNS * (1 + cfun(al) + Pi) - (1 + Pi)) ** 2

def NSfun_signed(al, beta, badleftover, Pi, BperG):
    """No-screening condition: gamma_NS*(1+K(alpha2)) - (1+Pi). Zero at equilibrium."""
    omega_g = beta + al * (1 - beta)
    goodleftover, _ = quad(gpriorfun, omega_g, 1)
    if goodleftover + badleftover < 1e-12:
        return -(1 + Pi)
    gammaNS = goodleftover / (goodleftover + badleftover)
    return gammaNS * (1 + cfun(al) + Pi) - (1 + Pi)

def find_alpha2(alpha1, beta, badleftover, Pi, BperG, n_scan=500):
    """Find smallest alpha2 > alpha1 satisfying the NS condition."""
    als = np.linspace(alpha1 + 1e-4, 0.999, n_scan)
    vals = np.array([NSfun_signed(a, beta, badleftover, Pi, BperG) for a in als])
    # Look for sign changes (roots) and return the smallest root
    for i in range(len(vals) - 1):
        if vals[i] * vals[i+1] < 0:
            alpha2 = brentq(lambda a: NSfun_signed(a, beta, badleftover, Pi, BperG),
                            als[i], als[i+1])
            return alpha2
    return None  # no solution

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
    global _current_params
    _current_params = params
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
    alpha2 = find_alpha2(alpha1, beta, badleftover, Pi, BperG)

    if alpha2 is not None:
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
# NESTED INFORMATION STRUCTURE — ANALYTICAL SOLUTION
# =============================================================================

def solve_nested_analytical(params):
    """
    Solve the nested model using the closed-form analytical solution for Region I.

    Works for general (non-uniform) priors. Two structural lemmas hold for ANY prior:
    1. Good borrowers at threshold omega_g(alpha) are undepleted (own-slice)
    2. Bad borrowers in [omega_b(alpha), 1] are uniformly depleted by factor E(alpha)

    Combined with the equal-profit condition gamma = Gamma(alpha), this pins down
    T(alpha) algebraically via three prior-dependent known functions:
        g_tilde(alpha) = gpriorfun(omega_g(alpha))  -- good density at threshold
        b_tilde(alpha) = bpriorfun(omega_b(alpha))  -- bad density at threshold
        B0_tilde(alpha) = int_{omega_b(alpha)}^1 bpriorfun(omega) domega  -- prior bad mass

    The general formula:
        T = (1-beta)*g_tilde*(1-Gamma)*B0_tilde / [Gamma'*B0_tilde - beta*b_tilde*Gamma*(1-Gamma)]

    Under uniform priors (g=1, b=BperG), this reduces to the original formula.
    No discretization of omega is needed. Only alpha is gridded (finely).
    """
    global _current_params
    _current_params = params
    Pi, beta, BperG = params.Pi, params.beta, params.BperG

    # =========================================================================
    # Find alpha0 and rp (same as solve_nested)
    # =========================================================================
    res = minimize_scalar(
        lambda a: (Pi + cfun(a) + 1) / gam0(a, params) if 0 < a < 1 else 1e10,
        bounds=(0.01, 0.99), method='bounded'
    )
    alpha0 = res.x
    rp = (Pi + cfun(alpha0) + 1) / gam0(alpha0, params) - 1

    # =========================================================================
    # Find alpha1: where K(alpha1) = rp, i.e. C(alpha1) + Pi = rp
    # =========================================================================
    if rp - cfun(1) > Pi:
        alpha1 = 1.0
    else:
        alpha1 = brentq(lambda a: cfun(a) - (rp - Pi), 0.01, 0.99)

    D_rp = dfun(rp)

    # =========================================================================
    # Region I: Analytical solution on a fine alpha grid
    # =========================================================================
    n_R1 = 2000
    alphas = np.linspace(alpha0, alpha1, n_R1)
    da = alphas[1] - alphas[0] if n_R1 > 1 else 1e-6

    # Thresholds
    omega_g_vals = beta + alphas * (1 - beta)
    omega_b_vals = 1 - beta + alphas * beta

    # Prior-dependent inputs at each alpha (general priors)
    g_tilde = np.array([gpriorfun(og) for og in omega_g_vals])
    b_tilde = np.array([bpriorfun(ob, BperG) for ob in omega_b_vals])
    B0_tilde = np.array([quad(lambda x: bpriorfun(x, BperG), ob, 1)[0]
                         for ob in omega_b_vals])

    # Precompute Gamma(alpha) and Gamma'(alpha)
    K_vals = cfun(alphas) + Pi
    Gamma = (1 + K_vals) / (1 + rp)
    Gamma_p = cfun_prime_exact(alphas) / (1 + rp)

    one_minus_Gamma = 1 - Gamma

    # General T formula
    denom = Gamma_p * B0_tilde - beta * b_tilde * Gamma * one_minus_Gamma
    denom_safe = np.where(np.abs(denom) < 1e-15, 1e-15, denom)
    T_vals = (1 - beta) * g_tilde * one_minus_Gamma * B0_tilde / denom_safe

    # Derived quantities
    G_vals = Gamma * T_vals
    B_vals = one_minus_Gamma * T_vals
    gamma_vals = np.where(T_vals > 1e-15, G_vals / T_vals, 1.0)

    # Depletion factor E(alpha) = B(alpha) / B0_tilde(alpha)
    E_vals = B_vals / np.where(np.abs(B0_tilde) < 1e-15, 1e-15, B0_tilde)

    # theta(alpha) = -d(ln E)/dalpha, computed via numerical differentiation
    ln_E = np.log(np.maximum(E_vals, 1e-30))
    theta_vals = np.zeros_like(alphas)
    # Central differences for interior points
    theta_vals[1:-1] = -(ln_E[2:] - ln_E[:-2]) / (2 * da)
    # One-sided at boundaries
    theta_vals[0] = -(ln_E[1] - ln_E[0]) / da
    theta_vals[-1] = -(ln_E[-1] - ln_E[-2]) / da

    # w(alpha) = theta * D(rp) * T
    w_vals = theta_vals * D_rp * T_vals
    w_vals = np.maximum(w_vals, 0)  # w can't be negative (ironing)

    # Cumulative capital: W(alpha_k) = integral of w from alpha0 to alpha_k
    # Use left-endpoint Riemann sum so W(alpha_0) = 0
    W_cumsum = np.zeros_like(alphas)
    W_cumsum[1:] = np.cumsum(w_vals[:-1]) * da

    # Remaining good/bad borrowers (total, not just in acceptance region)
    # Good outside acceptance: integral of g from omega_g to 1
    G_outside = np.array([quad(gpriorfun, og, 1)[0] for og in omega_g_vals])
    G_leftover = G_vals + G_outside

    # Bad remaining: undepleted below omega_b(alpha_0) + depleted dropouts + B(alpha)
    omega_b_0 = 1 - beta + alpha0 * beta
    B_below_0 = quad(lambda x: bpriorfun(x, BperG), 0, omega_b_0)[0]
    # Dropouts: bad borrowers at threshold omega_b(alpha') who left as alpha' increased
    # Their density at dropout was b_tilde(alpha') * E(alpha'), width d(omega_b) = beta*da
    b_tilde_E = b_tilde * E_vals * beta
    B_dropouts = np.zeros_like(alphas)
    B_dropouts[1:] = np.cumsum(b_tilde_E[:-1]) * da
    B_leftover = B_below_0 + B_dropouts + B_vals

    # =========================================================================
    # State at end of Region I
    # =========================================================================
    G_end_R1 = G_leftover[-1]
    B_end_R1 = B_leftover[-1]
    badleftover = B_end_R1

    # =========================================================================
    # Find alpha2 and WNS (non-selective lenders)
    # =========================================================================
    alpha2 = find_alpha2(alpha1, beta, badleftover, Pi, BperG)

    if alpha2 is not None:
        omega_g_alpha2 = beta + alpha2 * (1 - beta)
        goodleftover_alpha2, _ = quad(gpriorfun, omega_g_alpha2, 1)
        WNS = dfun(cfun(alpha2) + Pi) * (badleftover + goodleftover_alpha2)
    else:
        WNS, alpha2 = 0, 1.0

    r_NS = cfun(alpha2) + Pi

    # =========================================================================
    # Region II: alpha1 to alpha2 (general priors)
    # =========================================================================
    n_R2 = max(int((alpha2 - alpha1) * 100), 10)
    alphas_R2 = np.linspace(alpha1, alpha2, n_R2)
    da_R2 = alphas_R2[1] - alphas_R2[0] if n_R2 > 1 else 0.01

    omega_g_R2 = beta + alphas_R2 * (1 - beta)
    g_tilde_R2 = np.array([gpriorfun(og) for og in omega_g_R2])
    ws_R2 = np.array([dfun(cfun(al) + Pi) * (1 - beta) * g_tilde_R2[i]
                       for i, al in enumerate(alphas_R2)])

    G_remaining = G_end_R1
    GLOs_R2, BLOs_R2 = [], []
    for i, al in enumerate(alphas_R2):
        if i > 0:
            G_remaining -= (1 - beta) * g_tilde_R2[i] * da_R2
        GLOs_R2.append(max(G_remaining, 0))
        BLOs_R2.append(badleftover)

    # =========================================================================
    # Compute cumulative capital for Region II (continuous from Region I)
    # =========================================================================
    W_R2_cumsum = np.zeros(len(alphas_R2))
    W_R2_cumsum[1:] = np.cumsum(ws_R2[:-1]) * da_R2
    W_R2_cumsum += W_cumsum[-1]

    return {
        'alpha0': alpha0, 'alpha1': alpha1, 'alpha2': alpha2,
        'rp': rp, 'r_NS': r_NS,
        'WNS': WNS,
        # Region I (analytical)
        'alphas_R1': alphas,
        'ws_R1': w_vals,            # density w(alpha), NOT mass
        'gammas_R1': gamma_vals,
        'GLOs_R1': G_leftover,
        'BLOs_R1': B_leftover,
        'W_cumsum_R1': W_cumsum,
        'da_R1': da,
        # Region I analytical internals
        'G_R1': G_vals,             # good in acceptance region
        'B_R1': B_vals,             # bad in acceptance region
        'T_R1': T_vals,             # total in acceptance region
        'E_R1': E_vals,             # depletion factor
        # Region II
        'alphas_R2': np.array(alphas_R2),
        'ws_R2': ws_R2,
        'gammas_R2': np.ones(n_R2),
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
    global _current_params
    _current_params = params
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

def plot_comparison(params, nested, iid, nested_disc=None):
    """Create comparison plots for nested (analytical) vs IID equilibria.

    If nested_disc (discrete solver result) is provided, its curves are
    overlaid as dashed green lines for comparison.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    alpha0 = nested['alpha0']
    alpha1 = nested['alpha1']
    alpha2 = nested['alpha2']
    rp = nested['rp']
    r_NS = nested['r_NS']
    WNS = nested['WNS']

    # Prior description for title
    prior_desc = f"a_g={params.a_g}, a_b={params.a_b}"
    if params.a_g == 0 and params.a_b == 0:
        prior_desc = "uniform priors"

    # =========================================================================
    # Panel 1: Interest Rate r(alpha)
    # =========================================================================
    alpha_r1 = np.linspace(alpha0, alpha1, 100)
    r_r1 = rp * np.ones_like(alpha_r1)
    alpha_r2 = np.linspace(alpha1, alpha2, 100)
    r_r2 = cfun(alpha_r2) + params.Pi

    axes[0,0].plot(alpha_r1, r_r1, 'b-', lw=2, label='Nested analytical')
    axes[0,0].plot(alpha_r2, r_r2, 'b-', lw=2)
    axes[0,0].plot(0, r_NS, 'bo', markersize=10, markerfacecolor='blue', label='Nested (non-selective)')
    axes[0,0].plot(iid['alpha_eq'], iid['r_eq'], 'r-', lw=2, label='IID')

    if nested_disc is not None:
        a2_d = nested_disc['alpha2']
        axes[0,0].plot(np.linspace(alpha0, alpha1, 50),
                       nested_disc['rp'] * np.ones(50), 'g--', lw=1.5, alpha=0.7, label='Nested discrete')
        axes[0,0].plot(np.linspace(alpha1, a2_d, 50),
                       cfun(np.linspace(alpha1, a2_d, 50)) + params.Pi, 'g--', lw=1.5, alpha=0.7)
        if nested_disc['WNS'] > 0:
            axes[0,0].plot(0, nested_disc['r_NS'], 'gs', markersize=8,
                           markerfacecolor='none', markeredgewidth=2)

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
    axes[0,1].plot(nested['alphas_R1'], nested['gammas_R1'], 'b-', lw=2, label='Nested analytical')
    axes[0,1].plot(nested['alphas_R2'], nested['gammas_R2'], 'b-', lw=2)
    axes[0,1].plot(iid['alpha_eq'], iid['gamma_eq'], 'r-', lw=2, label='IID')

    if nested_disc is not None:
        axes[0,1].plot(nested_disc['alphas_R1'][:-1], nested_disc['gammas_R1'],
                       'g--', lw=1.5, alpha=0.7, label='Nested discrete')
        axes[0,1].plot(nested_disc['alphas_R2'], nested_disc['gammas_R2'],
                       'g--', lw=1.5, alpha=0.7)

    axes[0,1].axvline(alpha0, color='black', ls=':', alpha=0.3)
    axes[0,1].axvline(alpha1, color='blue', ls=':', alpha=0.5)
    axes[0,1].axvline(alpha2, color='blue', ls=':', alpha=0.5)
    axes[0,1].set_xlabel(r'$\alpha$'); axes[0,1].set_ylabel(r'$\gamma(\alpha)$')
    axes[0,1].set_title('Pool Quality (selective lenders)')
    axes[0,1].legend(fontsize=9)
    axes[0,1].grid(alpha=0.3)
    axes[0,1].set_xlim([0, 1])

    # =========================================================================
    # Panel 3: Cumulative Capital W(alpha)
    # =========================================================================
    # Jump at alpha=0
    axes[1,0].plot([0, 0], [0, WNS], 'b-', lw=2)
    axes[1,0].plot(0, 0, 'bo', markersize=8, markerfacecolor='white', markeredgewidth=2)
    axes[1,0].plot(0, WNS, 'bo', markersize=8, markerfacecolor='blue', label='Nested')
    alpha_flat = np.linspace(0, alpha0, 20)
    axes[1,0].plot(alpha_flat, WNS * np.ones_like(alpha_flat), 'b-', lw=2)
    # Region I
    axes[1,0].plot(nested['alphas_R1'], WNS + nested['W_cumsum_R1'], 'b-', lw=2)
    # Region II
    axes[1,0].plot(nested['alphas_R2'], WNS + nested['W_R2_cumsum'], 'b-', lw=2)

    if nested_disc is not None:
        WNS_d = nested_disc['WNS']
        axes[1,0].plot([0, 0], [0, WNS_d], 'g--', lw=1.5, alpha=0.7)
        axes[1,0].plot(0, WNS_d, 'gs', markersize=8, markerfacecolor='none',
                       markeredgewidth=2, label='Nested discrete')
        axes[1,0].plot(alpha_flat, WNS_d * np.ones_like(alpha_flat), 'g--', lw=1.5, alpha=0.7)
        axes[1,0].plot(nested_disc['alphas_R1'][1:], WNS_d + nested_disc['W_cumsum_R1'],
                       'g--', lw=1.5, alpha=0.7)
        axes[1,0].plot(nested_disc['alphas_R2'], WNS_d + nested_disc['W_R2_cumsum'],
                       'g--', lw=1.5, alpha=0.7)

    # IID
    alpha_before_iid = np.linspace(0, iid['alpha0'], 20)
    axes[1,0].plot(alpha_before_iid, np.zeros_like(alpha_before_iid), 'r-', lw=2, label='IID')
    axes[1,0].plot(iid['alpha_eq'], iid['W_cumsum'], 'r-', lw=2)

    axes[1,0].axvline(alpha0, color='black', ls=':', alpha=0.3)
    axes[1,0].axvline(alpha1, color='blue', ls=':', alpha=0.5)
    axes[1,0].axvline(alpha2, color='blue', ls=':', alpha=0.5)
    axes[1,0].set_xlabel(r'$\alpha$'); axes[1,0].set_ylabel(r'$W(\alpha)$')
    axes[1,0].set_title('Cumulative Capital')
    axes[1,0].legend(fontsize=9)
    axes[1,0].grid(alpha=0.3)
    axes[1,0].set_xlim([0, 1])

    # =========================================================================
    # Panel 4: Remaining Borrowers
    # =========================================================================
    axes[1,1].plot(nested['alphas_R1'], nested['GLOs_R1'], 'b-', lw=2, label='Analytical G')
    axes[1,1].plot(nested['alphas_R1'], nested['BLOs_R1'], 'b--', lw=2, label='Analytical B')
    axes[1,1].plot(nested['alphas_R2'], nested['GLOs_R2'], 'b-', lw=2)
    axes[1,1].plot(nested['alphas_R2'], nested['BLOs_R2'], 'b--', lw=2)
    axes[1,1].plot(iid['alpha_eq'], iid['G_eq'], 'r-', lw=2, label='IID G')
    axes[1,1].plot(iid['alpha_eq'], iid['B_eq'], 'r--', lw=2, label='IID B')

    if nested_disc is not None:
        axes[1,1].plot(nested_disc['alphas_R1'][:-1], nested_disc['GLOs_R1'],
                       'g-', lw=1.5, alpha=0.7, label='Discrete G')
        axes[1,1].plot(nested_disc['alphas_R1'][:-1], nested_disc['BLOs_R1'],
                       'g--', lw=1.5, alpha=0.7, label='Discrete B')
        axes[1,1].plot(nested_disc['alphas_R2'], nested_disc['GLOs_R2'],
                       'g-', lw=1.5, alpha=0.7)
        axes[1,1].plot(nested_disc['alphas_R2'], nested_disc['BLOs_R2'],
                       'g--', lw=1.5, alpha=0.7)

    axes[1,1].axvline(alpha0, color='black', ls=':', alpha=0.3)
    axes[1,1].axvline(alpha1, color='blue', ls=':', alpha=0.5)
    axes[1,1].axvline(alpha2, color='blue', ls=':', alpha=0.5)
    axes[1,1].set_xlabel(r'$\alpha$'); axes[1,1].set_ylabel('Mass')
    axes[1,1].set_title('Remaining Borrowers')
    axes[1,1].legend(fontsize=8, ncol=2)
    axes[1,1].grid(alpha=0.3)
    axes[1,1].set_xlim([0, 1])

    fig.suptitle(f'Nested vs IID  ({prior_desc})', fontsize=13)
    plt.tight_layout()
    return fig

# =============================================================================
# MAIN
# =============================================================================

def cfun_prime_exact(alpha):
    """Exact first derivative of c(alpha) = 0.8*alpha^2 + 0.5*alpha."""
    return 1.6 * alpha + 0.5

def cfun_prime2_exact(alpha):
    """Exact second derivative of c(alpha) = 0.8*alpha^2 + 0.5*alpha."""
    return 1.6

def compute_w_analytical(alpha0, rp, beta, BperG):
    """
    Compute w^nested and w^iid analytically from the paper's formulas
    (eqs w-nested-simplified, w-iid-simplified in gamma_monotonicity_proof_v4.tex).

    IMPORTANT: q_0 = 1/(1+b_bar) is the aggregate prior pool quality,
    NOT gamma_0 = G_0/T_0 (the accepted pool quality at alpha_0).

    Returns (w_nested, w_iid, ratio, primitives_dict).
    """
    g_bar, b_bar = 1.0, BperG
    omega_g = beta + alpha0 * (1 - beta)
    omega_b = 1 - beta + alpha0 * beta
    G_0 = omega_g * g_bar                       # good borrowers in acceptance region
    B_0 = b_bar * (1 - omega_b)                 # = b_bar * beta * (1-alpha_0)
    T_0 = G_0 + B_0                             # total accepted borrowers
    gamma_0 = G_0 / T_0                         # accepted pool quality
    q_0 = 1 / (1 + b_bar)                       # aggregate prior pool quality

    Cp = cfun_prime_exact(alpha0)
    Cpp = cfun_prime2_exact(alpha0)
    R = Cpp / Cp

    # Derived quantities using q_0 (NOT gamma_0)
    h_0 = q_0 * T_0                             # eq (uniform-simplifications): h_0 = q_0*T_0
    Lambda = (1 - beta) * g_bar - beta * b_bar   # = (g_bar+b_bar)*(q_0-beta)
    phi_0 = R * h_0 + 2 * (q_0 - beta)          # IID ODE coefficient
    D_rp = dfun(rp)

    # w^nested (corrected):
    #   = [R T_0 + 2*Lambda] D(rp) / ((1-beta)(1-alpha_0))
    # Note: the earlier version had 0.5*R*T_0 + Lambda, which was 2x too small.
    # The correct formula follows from the continuous ODE theta = -d(ln B)/da - 1/(1-a).
    w_nested = (R * T_0 + 2 * Lambda) * D_rp / ((1 - beta) * (1 - alpha0))

    # w^iid (eq:w-iid-simplified)
    #   = phi_0 T_0 (g_bar+b_bar) D(rp) / (alpha_0 omega_g)
    w_iid = phi_0 * T_0 * (g_bar + b_bar) * D_rp / (alpha0 * omega_g)

    # Closed-form ratio (corrected):
    ratio = (1 - beta) * (1 - alpha0) * T_0 / (alpha0 * omega_g)

    info = dict(omega_g=omega_g, G_0=G_0, B_0=B_0, T_0=T_0, gamma_0=gamma_0,
                q_0=q_0, Cp=Cp, Cpp=Cpp, R=R, h_0=h_0, Lambda=Lambda,
                phi_0=phi_0, D_rp=D_rp)
    return w_nested, w_iid, ratio, info


def check_w_ratio(params, nested, iid):
    """
    Numerically check Proposition (eq:w-ratio) from gamma_monotonicity_proof_v4.tex:

        w^iid(alpha_0) / w^nested(alpha_0) = 2(1-beta)(1-alpha_0)*T_0 / (alpha_0 * omega_g)

    Strategy:
    1. Verify the closed-form ratio equals the ratio of analytical w formulas (exact).
    2. Extract numerical w from both solvers and compare.
    3. Sweep parameters to confirm robustness and cost-curvature cancellation.
    """
    beta, BperG = params.beta, params.BperG
    alpha0 = nested['alpha0']
    rp = nested['rp']

    # =====================================================================
    # Step 1: Analytical check (paper's formulas)
    # =====================================================================
    w_n_a, w_i_a, ratio_formula, info = compute_w_analytical(alpha0, rp, beta, BperG)
    ratio_from_w = w_i_a / w_n_a


    # =====================================================================
    # Step 2: Numerical w from solvers
    # =====================================================================
    # IID: w_eq[0] is lender density at alpha_0 from ODE
    w_iid_num = iid['w_eq'][0]

    # Nested: if analytical solver is used, ws_R1 stores density directly;
    # if discrete solver is used, ws_R1 stores mass per Delta-step.
    is_analytical = 'da_R1' in nested  # analytical solver includes this key
    if is_analytical:
        w_nested_num = nested['ws_R1'][0]
        Delta = nested['da_R1']
    else:
        Delta = params.Delta
        w_nested_num = nested['ws_R1'][0] / Delta
    ratio_num = w_iid_num / w_nested_num if w_nested_num > 1e-12 else float('inf')

    # =====================================================================
    # Print
    # =====================================================================
    print("\n" + "=" * 70)
    print("CHECK: Proposition (w-ratio) -- w^iid / w^nested at alpha_0")
    print("=" * 70)

    print("\n  [Primitives at alpha_0]")
    for k in ['omega_g', 'G_0', 'B_0', 'T_0', 'gamma_0', 'q_0', 'Cp', 'Cpp', 'R',
              'h_0', 'Lambda', 'phi_0', 'D_rp']:
        print(f"    {k:20s} = {info[k]:.6f}")
    print(f"    {'alpha_0':20s} = {alpha0:.6f}")
    print(f"    {'beta':20s} = {beta}")
    print(f"    {'BperG':20s} = {BperG}")
    print(f"    {'rp':20s} = {rp:.6f}")

    print("\n  [Analytical w from paper's closed-form expressions]")
    print(f"    w^nested (eq:w-nested-simplified) = {w_n_a:.6f}")
    print(f"    w^iid   (eq:w-iid-simplified)     = {w_i_a:.6f}")
    print(f"    ratio  w^iid / w^nested           = {ratio_from_w:.6f}")

    print("\n  [Closed-form ratio formula]")
    print(f"    (1-beta)(1-a0)*T0 / (a0*omega_g) = {ratio_formula:.6f}")
    err_a = abs(ratio_formula - ratio_from_w) / ratio_formula * 100
    print(f"    Matches ratio of analytical w?     err = {err_a:.6f}%")
    if err_a < 0.01:
        print(f"    >>> EXACT MATCH (formula verified algebraically)")
    else:
        print(f"    >>> MISMATCH -- check formula derivation")

    print(f"\n  [Condition for ratio > 1]")
    print(f"    Ratio = {ratio_formula:.4f} > 1?  {ratio_formula > 1}")

    print(f"\n  [Cost curvature cancellation]")
    print(f"    R = C''/C' = {info['R']:.6f}  does NOT appear in ratio formula.")

    print(f"\n  [Numerical w from solvers (Delta={Delta}, IID n_steps=50000)]")
    print(f"    w^nested (ws_R1[0]/Delta)         = {w_nested_num:.4f}  (analytical: {w_n_a:.4f})")
    print(f"    w^iid   (w_eq[0])                 = {w_iid_num:.4f}  (analytical: {w_i_a:.4f})")
    print(f"    numerical ratio                   = {ratio_num:.4f}  (formula: {ratio_formula:.4f})")

    # =====================================================================
    # Step 3: Analytical solver vs formula
    # =====================================================================
    if is_analytical:
        print(f"\n  [Analytical solver w(alpha_0) vs paper formula]")
        print(f"    w^nested analytical solver  = {w_nested_num:.6f}")
        print(f"    w^nested paper formula      = {w_n_a:.6f}")
        err_solver = abs(w_nested_num - w_n_a) / w_n_a * 100
        print(f"    Error                       = {err_solver:.4f}%")
    else:
        print("\n  [Convergence: nested w(alpha_0) vs Delta (analytical={:.4f})]".format(w_n_a))
        print(f"    {'Delta':>10}  {'w_nested':>12}  {'err%':>8}  {'ratio':>10}  {'ratio_err%':>10}")
        for test_delta in [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
            p = Parameters(Pi=params.Pi, beta=params.beta, BperG=BperG,
                           Delta=test_delta, delom=params.delom)
            try:
                n = solve_nested(p)
                wn = n['ws_R1'][0] / test_delta
                we = abs(wn - w_n_a) / w_n_a * 100
                rr = w_iid_num / wn if wn > 1e-12 else float('inf')
                re = abs(ratio_formula - rr) / ratio_formula * 100
                print(f"    {test_delta:>10.4f}  {wn:>12.4f}  {we:>7.1f}%  {rr:>10.4f}  {re:>9.1f}%")
            except Exception as e:
                print(f"    {test_delta:>10.4f}  FAILED: {e}")

    # =====================================================================
    # Step 4: Sweep BperG (cost curvature cancellation + robustness)
    # =====================================================================
    print("\n  [Sweep: varying BperG -- analytical ratio vs formula]")
    print(f"    {'BperG':>8}  {'alpha0':>8}  {'formula':>10}  {'w_ratio':>10}  {'match':>8}  {'>1?':>5}")
    for bpg in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
        p = Parameters(Pi=params.Pi, beta=params.beta, BperG=bpg,
                       Delta=0.01, delom=params.delom)
        try:
            n = solve_nested(p)
            a0 = n['alpha0']
            rp_v = n['rp']
            wn_a, wi_a, rf, _ = compute_w_analytical(a0, rp_v, beta, bpg)
            ra = wi_a / wn_a
            err = abs(rf - ra) / rf * 100
            print(f"    {bpg:>8.2f}  {a0:>8.4f}  {rf:>10.4f}  {ra:>10.4f}  {err:>7.3f}%  {rf > 1!s:>5}")
        except Exception as e:
            print(f"    {bpg:>8.2f}  FAILED: {e}")

    # =====================================================================
    # Step 5: Sweep cost functions (verify R drops out)
    # =====================================================================
    print("\n  [Sweep: varying cost function -- ratio should be SAME]")
    original_cfun = (cfun.__code__, cfun_prime_exact, cfun_prime2_exact)
    cost_specs = [
        ("0.8a^2+0.5a",   lambda a: 0.8*a**2 + 0.5*a,  lambda a: 1.6*a + 0.5,  lambda a: 1.6),
        ("2.0a^2+0.5a",   lambda a: 2.0*a**2 + 0.5*a,  lambda a: 4.0*a + 0.5,  lambda a: 4.0),
        ("0.3a^2+0.8a",   lambda a: 0.3*a**2 + 0.8*a,  lambda a: 0.6*a + 0.8,  lambda a: 0.6),
        ("0.5a^3+0.5a",   lambda a: 0.5*a**3 + 0.5*a,  lambda a: 1.5*a**2+0.5, lambda a: 3.0*a),
    ]
    print(f"    {'cost_fn':>16}  {'R(a0)':>8}  {'formula':>10}  {'w_ratio':>10}  {'match':>8}")
    for name, cf, cfp, cfpp in cost_specs:
        try:
            # monkey-patch cfun globally for the solver
            import credit_model as cm
            old_cfun = cm.cfun
            cm.cfun = cf
            # re-solve with new cost function
            p = Parameters(Pi=params.Pi, beta=params.beta, BperG=BperG, Delta=0.01, delom=params.delom)
            n = solve_nested(p)
            a0 = n['alpha0']
            rp_v = n['rp']
            # Compute analytical with correct derivatives and q_0 = 1/(1+b_bar)
            og = beta + a0 * (1 - beta)
            G0 = og
            B0 = BperG * beta * (1 - a0)
            T0 = G0 + B0
            q0 = 1 / (1 + BperG)  # aggregate prior, NOT G0/T0
            Cp_v = cfp(a0)
            Cpp_v = cfpp(a0)
            Rv = Cpp_v / Cp_v
            h0 = q0 * T0
            Lam = (1 - beta) * 1.0 - beta * BperG  # (1-beta)g_bar - beta*b_bar
            phi0 = Rv * h0 + 2 * (q0 - beta)
            D_v = dfun(rp_v)
            w_nested_v = (0.5 * Rv * T0 + Lam) * D_v / ((1 - beta) * (1 - a0))
            w_iid_v = phi0 * T0 * (1 + BperG) * D_v / (a0 * og)
            ratio_v = w_iid_v / w_nested_v
            formula_v = 2 * (1 - beta) * (1 - a0) * T0 / (a0 * og)
            err = abs(formula_v - ratio_v) / formula_v * 100
            print(f"    {name:>16}  {Rv:>8.4f}  {formula_v:>10.4f}  {ratio_v:>10.4f}  {err:>7.3f}%")
            cm.cfun = old_cfun
        except Exception as e:
            print(f"    {name:>16}  FAILED: {e}")
            try:
                cm.cfun = old_cfun
            except:
                pass

    print("=" * 70)
    return ratio_formula, ratio_from_w


def compare_analytical_vs_discrete(params):
    """
    Compare analytical and discrete solvers for the given parameters.
    Works with any prior shape (a_g, a_b).
    """
    print("\n" + "=" * 70)
    prior_desc = f"a_g={params.a_g}, a_b={params.a_b}"
    if params.a_g == 0 and params.a_b == 0:
        prior_desc += " (uniform)"
    print(f"COMPARISON: Analytical vs Discrete  [{prior_desc}]")
    print("=" * 70)

    # Verify prior normalization
    g_int, _ = quad(gpriorfun, 0, 1)
    b_int, _ = quad(lambda x: bpriorfun(x, params.BperG), 0, 1)
    print(f"  Prior normalization: int g = {g_int:.4f},  int b = {b_int:.4f} (BperG={params.BperG})")

    # Run both solvers
    nested_disc = solve_nested(params)
    nested_anal = solve_nested_analytical(params)

    print(f"  alpha0: anal={nested_anal['alpha0']:.6f}  disc={nested_disc['alpha0']:.6f}")
    print(f"  alpha1: anal={nested_anal['alpha1']:.6f}  disc={nested_disc['alpha1']:.6f}")
    print(f"  rp:     anal={nested_anal['rp']:.6f}  disc={nested_disc['rp']:.6f}")

    # Compare gamma along Region I
    alphas_d = nested_disc['alphas_R1'][:-1]
    gammas_d = nested_disc['gammas_R1']
    alphas_a = nested_anal['alphas_R1']
    gammas_a = nested_anal['gammas_R1']

    a_lo = max(alphas_d[0], alphas_a[0])
    a_hi = min(alphas_d[-1], alphas_a[-1])
    mask_a = (alphas_a >= a_lo) & (alphas_a <= a_hi)
    mask_d = (alphas_d >= a_lo) & (alphas_d <= a_hi)

    if np.sum(mask_d) > 10:
        gammas_d_interp = np.interp(alphas_a[mask_a], alphas_d[mask_d], gammas_d[mask_d])
        gamma_err = np.max(np.abs(gammas_a[mask_a] - gammas_d_interp))
        gamma_err_rel = gamma_err / np.mean(gammas_d_interp) * 100
    else:
        gamma_err_rel = float('nan')

    # Compare cumulative W
    W_anal = nested_anal['W_cumsum_R1'][-1]
    W_disc = nested_disc['W_cumsum_R1'][-1] if len(nested_disc['W_cumsum_R1']) > 0 else 0
    W_err = abs(W_anal - W_disc) / abs(W_anal) * 100 if abs(W_anal) > 1e-12 else float('nan')

    # Compare w at midpoint
    mid_idx = len(nested_anal['ws_R1']) // 2
    alpha_mid = nested_anal['alphas_R1'][mid_idx]
    w_anal_mid = nested_anal['ws_R1'][mid_idx]
    ws_disc_density = nested_disc['ws_R1'] / params.Delta
    alphas_disc_mid = (nested_disc['alphas_R1'][:-1] + nested_disc['alphas_R1'][1:]) / 2
    w_disc_mid = np.interp(alpha_mid, alphas_disc_mid, ws_disc_density)
    w_err = abs(w_anal_mid - w_disc_mid) / abs(w_anal_mid) * 100 if abs(w_anal_mid) > 1e-12 else float('nan')

    # badleftover and WNS
    bl_anal, bl_disc = nested_anal['badleftover'], nested_disc['badleftover']
    bl_err = abs(bl_anal - bl_disc) / abs(bl_disc) * 100 if abs(bl_disc) > 1e-12 else float('nan')
    WNS_anal, WNS_disc = nested_anal['WNS'], nested_disc['WNS']
    WNS_err = abs(WNS_anal - WNS_disc) / abs(WNS_disc) * 100 if abs(WNS_disc) > 1e-12 else float('nan')

    print(f"  gamma max err: {gamma_err_rel:.4f}%")
    print(f"  W(alpha_1):    anal={W_anal:.4f}  disc={W_disc:.4f}  err={W_err:.2f}%")
    print(f"  w(mid):        anal={w_anal_mid:.4f}  disc={w_disc_mid:.4f}  err={w_err:.2f}%")
    print(f"  badleftover:   anal={bl_anal:.6f}  disc={bl_disc:.6f}  err={bl_err:.2f}%")
    print(f"  WNS:           anal={WNS_anal:.6f}  disc={WNS_disc:.6f}  err={WNS_err:.2f}%")
    print(f"  E(alpha_0):    {nested_anal['E_R1'][0]:.6f}  (should be 1.0)")

    ok = gamma_err_rel < 1.0 and W_err < 5.0
    print(f"  VERDICT: {'PASS' if ok else 'CHECK'}")
    print("=" * 70)
    return ok


def main():
    """Main function to run the model comparison."""
    params = Parameters()

    print("=" * 60)
    print("CREDIT MARKET: NESTED vs IID INFORMATION STRUCTURES")
    print("=" * 60)
    print(f"Parameters: Pi={params.Pi}, beta={params.beta}, BperG={params.BperG}")
    print(f"Priors: a_g={params.a_g}, a_b={params.a_b}  (0=uniform)")
    print("=" * 60)

    # Solve nested (analytical — primary solver)
    print("\n--- Solving Nested Model (analytical) ---")
    nested_a = solve_nested_analytical(params)
    print(f"alpha0 = {nested_a['alpha0']:.4f}")
    print(f"alpha1 = {nested_a['alpha1']:.4f}")
    print(f"alpha2 = {nested_a['alpha2']:.4f}")
    print(f"rp = {nested_a['rp']:.4f}")
    print(f"r_NS = {nested_a['r_NS']:.4f}")
    print(f"WNS = {nested_a['WNS']:.4f}")
    print(f"Total W (Regions I+II) = {nested_a['W_R2_cumsum'][-1]:.4f}")
    print(f"Total W (including NS) = {nested_a['W_R2_cumsum'][-1] + nested_a['WNS']:.4f}")

    # Solve nested (discrete — for comparison)
    print("\n--- Solving Nested Model (discrete) ---")
    nested = solve_nested(params)
    print(f"alpha0 = {nested['alpha0']:.4f}")
    print(f"rp = {nested['rp']:.4f}")
    print(f"WNS = {nested['WNS']:.4f}")
    print(f"Total W (including NS) = {nested['W_R2_cumsum'][-1] + nested['WNS']:.4f}")

    # Solve IID
    print("\n--- Solving IID Model ---")
    iid = solve_iid(params)
    print(f"alpha0 = {iid['alpha0']:.4f}")
    print(f"alpha_bar = {iid['alpha_bar']:.4f}")
    print(f"r0 = {iid['r0']:.4f}")
    print(f"r_perfect = {iid['r_perfect']:.4f}")
    print(f"Total W = {iid['W_cumsum'][-1]:.4f}")

    # Plot using analytical solver as primary nested solution
    print("\n--- Creating Plot ---")
    fig = plot_comparison(params, nested_a, iid, nested_disc=nested)

    # Save in same folder as script
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'credit_model.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    # Compare analytical vs discrete for current parameters
    compare_analytical_vs_discrete(params)

    # Also test with non-uniform priors if currently uniform
    if params.a_g == 0 and params.a_b == 0:
        for a_g, a_b in [(1.0, 1.0), (0.5, 0.3), (2.0, 0.0)]:
            p = Parameters(Pi=params.Pi, beta=params.beta, BperG=params.BperG,
                           Delta=params.Delta, delom=params.delom, a_g=a_g, a_b=a_b)
            compare_analytical_vs_discrete(p)

    return params, nested_a, iid

if __name__ == "__main__":
    params, nested, iid = main()
