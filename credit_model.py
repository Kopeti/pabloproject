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

# Cost function: c(alpha) = COST_C2 * alpha**COST_P2 + COST_C1 * alpha**COST_P1
# Change these four parameters — cfun, cfun_prime_exact, cfun_prime2_exact
# are all derived from them and stay consistent automatically.
COST_C2 = 0.2
COST_P2 = 2.0
COST_C1 = 1.0
COST_P1 = 1.0

def cfun(alpha):
    """Screening cost function c(alpha) = C2*alpha^P2 + C1*alpha^P1."""
    return COST_C2 * alpha**COST_P2 + COST_C1 * alpha**COST_P1

def cfun_prime_exact(alpha):
    """Exact first derivative: c'(alpha) = C2*P2*alpha^(P2-1) + C1*P1*alpha^(P1-1)."""
    return COST_C2 * COST_P2 * alpha**(COST_P2 - 1) + COST_C1 * COST_P1 * alpha**(COST_P1 - 1)

def cfun_prime2_exact(alpha):
    """Exact second derivative: c''(alpha) = C2*P2*(P2-1)*alpha^(P2-2) + C1*P1*(P1-1)*alpha^(P1-2)."""
    t2 = COST_C2 * COST_P2 * (COST_P2 - 1) * alpha**(COST_P2 - 2) if COST_P2 > 1 else 0.0
    t1 = COST_C1 * COST_P1 * (COST_P1 - 1) * alpha**(COST_P1 - 2) if COST_P1 > 1 else 0.0
    return t2 + t1

def cfun_prime(alpha, eps=1e-5):
    """First derivative of cost function (numerical fallback)."""
    return (cfun(alpha + eps) - cfun(alpha)) / eps

def cfun_prime2(alpha, eps=1e-5):
    """Second derivative of cost function (numerical fallback)."""
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
    BperG: float = 0.2197  # Ratio of bad to good borrowers
    Delta: float = 0.001  # Step size for alpha iteration
    delom: float = 0.0001 # Step size for omega grid
    # Prior shape parameters (0 = uniform)
    # g(omega) = (1+a_g) * omega^a_g    -- integrates to 1 on [0,1]
    # b(omega) = BperG * (1+a_b) * (1-omega)^a_b  -- integrates to BperG on [0,1]
    a_g: float = 0
    a_b: float = 0

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

def find_alpha0(params, n_scan=400, lo=0.01, hi=0.99):
    """Global argmin of the fresh-pool break-even rate (Pi + C(a) + 1)/gam0(a).

    Runs the original bounded Brent search first and keeps its result verbatim
    (bit-identical to the legacy behavior) unless a grid scan finds a strictly
    better value — i.e. Brent, which assumes unimodality, landed in a local
    minimum (e.g. under the plateau cost used to demonstrate ironing).  Only
    then is the grid minimum refined and used.
    """
    obj = lambda a: (params.Pi + cfun(a) + 1) / gam0(a, params) if 0 < a < 1 else 1e10
    res = minimize_scalar(obj, bounds=(lo, hi), method='bounded')
    alpha0 = res.x
    grid = np.linspace(lo, hi, n_scan)
    vals = np.array([obj(a) for a in grid])
    k = int(np.argmin(vals))
    if vals[k] < obj(alpha0) - 1e-12:
        # Brent missed the global minimum: refine around the grid argmin.
        res2 = minimize_scalar(
            obj, bounds=(grid[max(k - 1, 0)], grid[min(k + 1, n_scan - 1)]),
            method='bounded')
        if obj(res2.x) < obj(alpha0):
            print(f"  [find_alpha0] Brent local minimum at {alpha0:.4f} "
                  f"overridden by global minimum near {res2.x:.4f}")
            alpha0 = res2.x
    rp = obj(alpha0) - 1
    return alpha0, rp

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
# IRONING (Region I, nested) — Appendix "Ironing" of the draft
# =============================================================================

def _frozen_tail_violation(i, G_vals, E_vals, B0_tilde, cumG, Gamma_req,
                           j_min=None):
    """Max violation of the free-entry bound along the frozen-pool tail from i.

    If entry follows the closed form up to alphas[i] and then stops, the pool
    at a > alphas[i] evolves with no depletion:
        G(a; a_i) = G(a_i) + [calG(omega_g(a)) - calG(omega_g(a_i))]
        B(a; a_i) = E(a_i) * B0_tilde(a)
    (general priors: the own-slice increment enters through cumG, the bad side
    through B0_tilde — no uniformity assumed).

    Returns (m, j_tan): m is the max violation of
    gamma(a; a_i) <= Gamma_req(a) over the whole tail a > a_i (membership test
    for the set N); j_tan is the argmax restricted to indices >= j_min — the
    tangency/resumption candidate.  The restriction matters at the component's
    left edge, where the global max is ~0 and float noise can otherwise pick a
    spurious adjacent point instead of the interior tangency; the no-entry
    interval must cover the first infeasible point, so passing j_min = first
    index with w < 0 is the correct restriction.
    """
    G_ir = G_vals[i] + (cumG[i + 1:] - cumG[i])
    B_ir = E_vals[i] * B0_tilde[i + 1:]
    gam = G_ir / (G_ir + B_ir)
    viol = gam - Gamma_req[i + 1:]
    m = float(np.max(viol))
    if j_min is None or j_min <= i + 1:
        j_tan = i + 1 + int(np.argmax(viol))
    else:
        j_tan = j_min + int(np.argmax(viol[j_min - (i + 1):]))
    return m, j_tan


def iron_region1(alphas, K_vals, rp, beta, g_tilde, B0_tilde,
                 T_vals, G_vals, B_vals, E_vals, w_vals, da,
                 w_tol_rel=1e-8):
    """Ironing for the closed-form Region-I solution (nested information).

    Where the closed-form entry density w(alpha) would be negative, the
    equilibrium instead has no-entry intervals: entry stops at a_L, the pool
    evolves undepleted (own slices accumulate, the bad depletion factor stays
    frozen at E(a_L)), and entry resumes at a_R.  Per the draft's Ironing
    appendix, the no-entry intervals are the connected components of
        N = { a~ : gamma(a; a~) > (1+K(a))/(1+rp) for some a > a~ },
    where gamma(a; a~) is the frozen-pool quality.  At the left edge a_L the
    frozen tail stays weakly below the bound with a tangency at a_R; at the
    tangency both the quality and its slope match the zero-profit path, so the
    closed form resumes there with (numerically) the same state.

    All prior dependence enters through the input arrays (g_tilde, B0_tilde);
    no uniform-prior assumption is made.

    Returns a dict with 'active', corrected arrays ('w','T','G','B','E',
    'gamma','theta'), 'intervals' [(a_L, a_R), ...] and 'diagnostics'.
    """
    n = len(alphas)
    D_rp = dfun(rp)
    scale = max(1.0, float(np.max(np.abs(w_vals))))
    if np.min(w_vals) >= -w_tol_rel * scale:
        return {'active': False, 'w': w_vals, 'T': T_vals, 'G': G_vals,
                'B': B_vals, 'E': E_vals,
                'gamma': np.where(T_vals > 1e-15, G_vals / T_vals, 1.0),
                'theta': None, 'intervals': [], 'diagnostics': []}

    Gamma_req = (1 + K_vals) / (1 + rp)
    # Own-slice cumulative integral (left-endpoint, house convention):
    # cumG[i] ~ calG(omega_g(alphas[i])) - calG(omega_g(alphas[0]))
    cumG = np.zeros(n)
    cumG[1:] = np.cumsum((1 - beta) * g_tilde[:-1]) * da

    G_new = G_vals.copy(); B_new = B_vals.copy(); E_new = E_vals.copy()
    intervals = []          # [(i_L, i_R)] index pairs
    diagnostics = []

    start = 0
    while True:
        neg = np.where(w_vals[start:] < -w_tol_rel * scale)[0]
        if len(neg) == 0:
            break
        i_neg = start + int(neg[0])

        # Scan candidate stopping points downward from the first violation for
        # the left edge of the N-component (first candidate NOT in N).  The
        # tangency/resumption point is the restricted argmax at or beyond the
        # first infeasible index, so the interval always covers it.
        i_L, i_R = None, None
        for i in range(i_neg, start - 1, -1):
            m, j = _frozen_tail_violation(i, G_vals, E_vals, B0_tilde,
                                          cumG, Gamma_req, j_min=i_neg)
            if m <= 0.0:
                i_L, i_R = i, j
                break
        if i_L is None:
            # Component extends to the current segment start — stop there.
            print("  [ironing] WARNING: no consistent stopping point found "
                  "above the segment start; ironing from the segment start.")
            i_L = start
            _, i_R = _frozen_tail_violation(i_L, G_vals, E_vals, B0_tilde,
                                            cumG, Gamma_req, j_min=i_neg)
        if i_R >= n - 1:
            raise RuntimeError(
                "iron_region1: a no-entry interval reaches alpha_1 "
                f"(a_L={alphas[i_L]:.4f}).  This variant (entry never resumes "
                "in Region I) is not implemented.")

        # Overwrite the state on the interior of the component with the
        # frozen-pool propagation.  Edge points keep closed-form values
        # (state is continuous at a_L by construction, at a_R by tangency).
        ks = np.arange(i_L + 1, i_R)
        E_new[ks] = E_vals[i_L]
        B_new[ks] = E_vals[i_L] * B0_tilde[ks]
        G_new[ks] = G_vals[i_L] + (cumG[ks] - cumG[i_L])

        # Diagnostics: does the pointwise-negative set start after a_L?  How
        # good is the state match at the tangency point a_R?
        mismatch_G = abs((G_vals[i_L] + (cumG[i_R] - cumG[i_L]) - G_vals[i_R])
                         / max(abs(G_vals[i_R]), 1e-15))
        mismatch_B = abs((E_vals[i_L] * B0_tilde[i_R] - B_vals[i_R])
                         / max(abs(B_vals[i_R]), 1e-15))
        diagnostics.append({
            'a_L': float(alphas[i_L]), 'a_R': float(alphas[i_R]),
            'first_w_neg': float(alphas[i_neg]),
            'state_mismatch_at_aR': (float(mismatch_G), float(mismatch_B)),
        })
        intervals.append((i_L, i_R))
        start = i_R + 1

    T_new = G_new + B_new
    gamma_new = np.where(T_new > 1e-15, G_new / T_new, 1.0)

    # Segment-aware theta = -d(ln E)/da: one-sided differences at every
    # regime boundary so the kink in ln E does not smear into w.
    ln_E = np.log(np.maximum(E_new, 1e-30))
    theta_new = np.zeros(n)
    edges = [0] + [e for (iL, iR) in intervals for e in (iL, iR)] + [n - 1]
    seg_bounds = list(zip(edges[:-1], edges[1:]))
    for k_seg, (s, e) in enumerate(seg_bounds):
        inside_component = any(iL == s and iR == e for (iL, iR) in intervals)
        if inside_component:
            theta_new[s + 1:e] = 0.0            # E frozen => theta = 0
            continue
        if e - s >= 2:
            theta_new[s + 1:e] = -(ln_E[s + 2:e + 1] - ln_E[s:e - 1]) / (2 * da)
        if e > s:
            theta_new[s] = -(ln_E[s + 1] - ln_E[s]) / da
            theta_new[e] = -(ln_E[e] - ln_E[e - 1]) / da

    w_new = theta_new * D_rp * T_new
    for (iL, iR) in intervals:
        w_new[iL + 1:iR] = 0.0

    return {'active': True, 'w': w_new, 'T': T_new, 'G': G_new, 'B': B_new,
            'E': E_new, 'gamma': gamma_new, 'theta': theta_new,
            'intervals': [(float(alphas[iL]), float(alphas[iR]))
                          for (iL, iR) in intervals],
            'diagnostics': diagnostics}


def plateau_cost_factory(lam=0.15, lo=0.15, hi=0.25, s=0.02,
                         c2=9.0, c1=0.2, n_grid=20001):
    """Cost function with a flattened-slope window — activates ironing.

    C'(a) = (2*c2*a + c1) * (1 - (1-lam)*win(a)) with a smooth logistic window
    on [lo, hi]; C by cumulative (trapezoid) integration of C' on a fine grid.
    With the stock incumbent parameters (Pi=0.235, beta=0.5, B/G=1) and
    lam=0.15 the closed-form Region-I density goes negative on an interior
    interval, so iron_region1 has real work to do, while C stays strictly
    increasing.  Returns (cfun, cfun_prime) callables (numpy-vectorized).
    """
    grid = np.linspace(0.0, 1.0, n_grid)
    win = (1.0 / (1.0 + np.exp(-(grid - lo) / s))
           * 1.0 / (1.0 + np.exp((grid - hi) / s)))
    cp = (2.0 * c2 * grid + c1) * (1.0 - (1.0 - lam) * win)
    C = np.concatenate([[0.0],
                        np.cumsum((cp[1:] + cp[:-1]) * 0.5 * np.diff(grid))])

    def cfun_plateau(a):
        return np.interp(np.asarray(a, dtype=float), grid, C)

    def cfun_prime_plateau(a):
        return np.interp(np.asarray(a, dtype=float), grid, cp)

    return cfun_plateau, cfun_prime_plateau


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
    alpha0, rp = find_alpha0(params)
    
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
    alpha0, rp = find_alpha0(params)

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

    # Ironing (Appendix "Ironing" of the draft): where the closed-form density
    # would be negative, the equilibrium instead has no-entry intervals with an
    # undepleted pool.  iron_region1 detects the intervals (connected
    # components of the set N) and rebuilds the state arrays consistently; on
    # calibrations where w >= 0 everywhere it is a no-op passthrough.
    _ironing = iron_region1(alphas, K_vals, rp, beta, g_tilde, B0_tilde,
                            T_vals, G_vals, B_vals, E_vals, w_vals, da)
    if _ironing['active']:
        w_vals = _ironing['w']
        T_vals, G_vals, B_vals = _ironing['T'], _ironing['G'], _ironing['B']
        E_vals, gamma_vals = _ironing['E'], _ironing['gamma']
        theta_vals = _ironing['theta']
        print(f"  [ironing] active: no-entry intervals "
              f"{[(round(a, 4), round(b, 4)) for (a, b) in _ironing['intervals']]}")
    w_vals = np.maximum(w_vals, 0)  # numerical dust only; negativity handled above

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
        goodleftover_alpha2 = 0.0
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
        'goodleftover_alpha2': goodleftover_alpha2,
        'G_end_R1': G_end_R1,
        'B_end_R1': B_end_R1,
        # Ironing (Appendix "Ironing"): no-entry intervals in Region I
        'ironing_active': _ironing['active'],
        'no_entry_intervals': _ironing['intervals'],
        'ironing_diagnostics': _ironing['diagnostics'],
    }


# =============================================================================
# IID INFORMATION STRUCTURE
# =============================================================================

def solve_iid(params, n_steps=50000, tie_grid_n=2001, max_segments=12):
    """Solve the model under IID information structure using the scalar ODE,
    with the equilibrium configurations of the ironing note
    (Notes/ironing_iid.tex) handled:

      Case 1 (fully separating): the ODE runs to alpha ~ 1 -- identical to the
          legacy behavior (same Euler updates, same grid, same output arrays).
      Case 2 (top jump): r(alpha) reaches r_perfect = K(1) at alpha_bar < 1;
          the schedule truncates there and the atom of perfect screeners at
          alpha = 1 absorbs the remaining good borrowers.
      Case 3 (interior ironing): when a second skill ties on the break-even
          frontier b(alpha, z) = (Pi + 1 + C(alpha))/gamma(alpha, z) - 1, the
          support jumps: entry stops at m1, resumes at m2 > m1 with the SAME
          rate and pool (r and z continuous across the gap), and the ODE
          restarts from (m2, z_tie).  Detected by a per-step global scan of
          the frontier; gaps are returned in 'gaps'.
      Corner (complete pooling): if alpha = 0 is the strict argmin of the
          fresh-pool break-even rate, the unique equilibrium is a single
          pooling market at r_0(0) serving all borrowers (no schedule).

    The atom's capital W_atom = D(r_perfect) * G_end is reported in every
    non-pooling case (it was previously ignored).

    Backward compatibility: all legacy keys are returned with unchanged
    meaning; in Case 1 the arrays are numerically identical to the legacy
    solver.  New keys: 'gaps' [(a_L, a_R, z_tie, r_tie), ...],
    'segment_slices' (index slices of the continuous stretches, for plotting),
    'W_atom', 'case' (1, 2, 3 or 'pooling'), 'pooling'.
    """
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

    # Break-even frontier b(alpha, z) on a fixed alpha grid (vectorized).
    A_GRID = np.linspace(0.0, 1.0, tie_grid_n)
    MU_GRID = beta + A_GRID * (1 - beta)
    KP1_GRID = Pi + 1 + cfun(A_GRID)

    def b_grid(z):
        denom = z * MU_GRID + (1 - z) * beta * (1 - A_GRID)
        gam = np.where(denom > 1e-15, z * MU_GRID / denom, 0.0)
        return np.where(gam > 1e-12, KP1_GRID / gam - 1.0, 1e10)

    # Find alpha0 (legacy bounded Brent kept verbatim; grid scan only detects
    # the pooling corner or overrides a missed global minimum, as find_alpha0
    # does for the nested solver).
    alpha0 = minimize_scalar(
        lambda a: (Pi + 1 + cfun(a)) / gamma_f(a, z0) if gamma_f(a, z0) > 1e-12 else 1e10,
        bounds=(0.01, 0.99), method='bounded'
    ).x
    b_fresh = b_grid(z0)
    k_min = int(np.argmin(b_fresh))
    if k_min == 0 and b_fresh[0] < r_f(alpha0, z0) - 1e-12:
        # Corner: zero skill is the strict cheapest server of the pristine
        # pool.  Zero-skill draws are quality-neutral, so the pool never
        # deteriorates and the unique equilibrium is complete pooling at
        # r_0(0) (ironing note, Section "The corner").
        r_pool = r_f(0.0, z0)
        W_pool = dfun(r_pool) * (G0 + B0)
        print(f"  [solve_iid] CORNER: alpha=0 is the cheapest skill for the "
              f"fresh pool. Complete-pooling equilibrium at r={r_pool:.4f}, "
              f"W={W_pool:.4f} (all borrowers served, no separation, no atom).")
        one = np.array([0.0])
        return {
            'alpha0': 0.0, 'alpha_bar': 0.0,
            'r0': r_pool, 'r_perfect': r_perfect,
            'alpha_eq': one, 'z_eq': np.array([z0]),
            'r_eq': np.array([r_pool]),
            'G_eq': np.array([0.0]), 'B_eq': np.array([0.0]),
            'gamma_eq': np.array([z0]), 'w_eq': np.array([0.0]),
            'W_cumsum': np.array([W_pool]),
            'z0': z0, 'G0': G0, 'B0': B0, 'da': 0.0,
            'gaps': [], 'segment_slices': [slice(0, 1)],
            'W_atom': 0.0, 'case': 'pooling', 'pooling': True,
        }
    if b_fresh[k_min] < r_f(alpha0, z0) - 1e-12:
        res2 = minimize_scalar(
            lambda a: r_f(a, z0),
            bounds=(A_GRID[max(k_min - 1, 0)], A_GRID[min(k_min + 1, tie_grid_n - 1)]),
            method='bounded')
        if r_f(res2.x, z0) < r_f(alpha0, z0):
            print(f"  [solve_iid] Brent local minimum at {alpha0:.4f} "
                  f"overridden by global minimum near {res2.x:.4f}")
            alpha0 = res2.x
    r0 = r_f(alpha0, z0)

    # Euler method for ODE, in segments separated by ironing jumps.  Segment 0
    # uses exactly the legacy grid (linspace alpha0..0.9999, n_steps) and the
    # legacy step da = alpha_path[1] - alpha_path[0], so in Case 1 -- no ties,
    # no crossing -- the output is bit-identical to the legacy solver.  Later
    # segments reuse the same step size da.
    _grid0 = np.linspace(alpha0, 0.9999, n_steps)
    da = _grid0[1] - _grid0[0]
    TIE_MASK_W = 0.03   # ignore the frontier within this window of the
                        # current skill (the path itself sits at b = r there);
                        # jumps shorter than this are not detected.

    seg_arrays = []     # list of dicts of truncated per-segment arrays
    gaps = []           # (a_L, a_R, z_tie, r_tie)
    a_start, z_start, G_start, B_start = alpha0, z0, G0, B0
    crossed = False
    alpha_bar = 0.9999
    fold_warning = False

    for seg in range(max_segments):
        n_seg = n_steps if seg == 0 else max(int(np.floor((0.9999 - a_start) / da)) + 1, 2)
        alpha_path = _grid0 if seg == 0 else a_start + da * np.arange(n_seg)

        z_path = np.zeros(n_seg)
        r_path = np.zeros(n_seg)
        G_path = np.zeros(n_seg)
        B_path = np.zeros(n_seg)
        w_over_D = np.zeros(n_seg)

        z_path[0], r_path[0] = z_start, r_f(a_start, z_start)
        G_path[0], B_path[0] = G_start, B_start
        alpha_bar_idx = n_seg - 1
        tie_k, tie_j = None, None

        for k in range(1, n_seg):
            a, z, G, B = alpha_path[k-1], z_path[k-1], G_path[k-1], B_path[k-1]

            zp = zprime(a, z)
            z_path[k] = z + zp * da
            r_path[k] = r_f(alpha_path[k], z_path[k])

            if r_path[k] >= r_perfect and alpha_bar_idx == n_seg - 1:
                alpha_bar_idx = k

            gam = gamma_f(a, z)
            wD = zp * (G + B) / (z - gam) if abs(z - gam) > 1e-12 and (G + B) > 1e-12 else 0
            w_over_D[k-1] = wD

            G_path[k] = max(G - wD * gam * da, 0)
            B_path[k] = max(B - wD * (1 - gam) * da, 0)

            # Ironing tie check: has another skill become strictly cheaper on
            # the break-even frontier at the current pool?  (Skip the first
            # few steps of a resumed segment: the landing point is the argmin
            # there by construction, up to refinement noise.)
            if alpha_bar_idx == n_seg - 1 and (seg == 0 or k > 10):
                bvec = b_grid(z_path[k])
                far = np.abs(A_GRID - alpha_path[k]) >= TIE_MASK_W
                far &= A_GRID < 0.999   # the corner alpha=1 (b = r_perfect)
                                        # is the atom, handled by the crossing
                if np.any(far):
                    jj = np.where(far)[0][int(np.argmin(bvec[far]))]
                    if bvec[jj] < r_path[k] - 1e-12:
                        tie_k, tie_j = k, jj
                        break

        if tie_k is None:
            idx = alpha_bar_idx + 1
            crossed = r_path[alpha_bar_idx] >= r_perfect
            alpha_bar = alpha_path[alpha_bar_idx]
        else:
            # Backfill the w_over_D of the tie point (the legacy loop sets
            # w_over_D[k-1] during iteration k, and the break at tie_k means
            # iteration tie_k+1 never ran).  Same state, same formula.
            idx = tie_k + 1
            a, z, G, B = (alpha_path[tie_k], z_path[tie_k],
                          G_path[tie_k], B_path[tie_k])
            zp = zprime(a, z)
            gam = gamma_f(a, z)
            w_over_D[tie_k] = (zp * (G + B) / (z - gam)
                               if abs(z - gam) > 1e-12 and (G + B) > 1e-12 else 0)

        seg_arrays.append({
            'alpha': alpha_path[:idx], 'z': z_path[:idx], 'r': r_path[:idx],
            'G': G_path[:idx], 'B': B_path[:idx], 'wD': w_over_D[:idx],
        })

        if tie_k is None:
            break

        # Ironing jump: refine the landing skill m2 = argmin b(., z_tie) near
        # the grid minimizer, then restart the ODE from (m2, z_tie) with the
        # pool masses unchanged (no entry inside the gap).
        z_t, r_t = z_path[tie_k], r_path[tie_k]
        dg = A_GRID[1] - A_GRID[0]
        m2 = minimize_scalar(
            lambda aa: r_f(aa, z_t),
            bounds=(max(A_GRID[tie_j] - 2 * dg, 0.0),
                    min(A_GRID[tie_j] + 2 * dg, 1.0)),
            method='bounded').x
        gaps.append((float(alpha_path[tie_k]), float(m2), float(z_t), float(r_t)))
        print(f"  [solve_iid] ironing: skill support jumps "
              f"{alpha_path[tie_k]:.4f} -> {m2:.4f} at pool z={z_t:.4f} "
              f"(rate {r_t:.4f} continuous across the gap)")
        a_start, z_start = m2, z_t
        G_start, B_start = G_path[tie_k], B_path[tie_k]
    else:
        print("  [solve_iid] WARNING: max_segments reached; schedule truncated.")

    # Concatenate segments
    alpha_eq = np.concatenate([s['alpha'] for s in seg_arrays])
    z_eq = np.concatenate([s['z'] for s in seg_arrays])
    r_eq = np.concatenate([s['r'] for s in seg_arrays])
    G_eq = np.concatenate([s['G'] for s in seg_arrays])
    B_eq = np.concatenate([s['B'] for s in seg_arrays])
    wD_eq = np.concatenate([s['wD'] for s in seg_arrays])
    lens = [len(s['alpha']) for s in seg_arrays]
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    segment_slices = [slice(int(s), int(s + l)) for s, l in zip(starts, lens)]

    gamma_eq = np.array([gamma_f(alpha_eq[i], z_eq[i]) for i in range(len(alpha_eq))])
    w_eq = np.array([wD_eq[k] * dfun(r_eq[k]) if r_eq[k] > 1e-12 else 0
                     for k in range(len(alpha_eq))])
    if np.min(w_eq) < -1e-6 * max(1.0, float(np.max(np.abs(w_eq)))):
        fold_warning = True
        print("  [solve_iid] WARNING: negative entry density detected on the "
              "kept path -- an undetected fold/tie (raise tie_grid_n or lower "
              "TIE_MASK_W).")
    W_iid_cumsum = np.cumsum(w_eq) * da

    # Atom of perfect screeners at alpha = 1: absorbs the remaining good
    # borrowers at r_perfect = K(1).  (In Case 1 the schedule reaches alpha ~ 1
    # with a small residual; in Case 2 the residual is large.)
    G_end, B_end = float(G_eq[-1]), float(B_eq[-1])
    W_atom = dfun(r_perfect) * G_end if r_perfect > 1e-12 else 0.0

    case = 3 if gaps else (2 if crossed and alpha_bar < 0.98 else 1)

    return {
        'alpha0': alpha0, 'alpha_bar': alpha_bar,
        'r0': r0, 'r_perfect': r_perfect,
        'alpha_eq': alpha_eq, 'z_eq': z_eq, 'r_eq': r_eq,
        'G_eq': G_eq, 'B_eq': B_eq,
        'gamma_eq': gamma_eq, 'w_eq': w_eq,
        'W_cumsum': W_iid_cumsum,
        'z0': z0, 'G0': G0, 'B0': B0, 'da': da,
        # New: ironing / atom / classification
        'gaps': gaps, 'segment_slices': segment_slices,
        'W_atom': W_atom, 'case': case, 'pooling': False,
        'fold_warning': fold_warning,
    }

# =============================================================================
# PLOTTING
# =============================================================================

def _iid_segments(iid):
    """Index slices of the continuous stretches of the iid schedule (one slice
    for legacy results without the 'segment_slices' key)."""
    return iid.get('segment_slices') or [slice(0, len(iid['alpha_eq']))]


def _plot_iid_curve(ax, iid, yvals, style='r-', lw=2, label=None):
    """Plot an iid equilibrium object segment by segment so that lines break
    at ironing gaps instead of bridging them."""
    for i, s in enumerate(_iid_segments(iid)):
        ax.plot(iid['alpha_eq'][s], yvals[s], style, lw=lw,
                label=label if i == 0 else None)


def _shade_iid_gaps(ax, iid):
    """Light shading over skill intervals never chosen in the iid equilibrium
    (interior ironing gaps and, in the top-jump case, [alpha_bar, 1))."""
    for (aL, aR, _z, _r) in iid.get('gaps', []):
        ax.axvspan(aL, aR, color='0.85', alpha=0.6, lw=0, zorder=0)
    if iid.get('case') == 2:
        ax.axvspan(iid['alpha_bar'], 1.0, color='0.85', alpha=0.6, lw=0, zorder=0)


def plot_comparison(params, nested, iid, nested_disc=None):
    """Create comparison plots for nested (analytical) vs IID equilibria.

    If nested_disc (discrete solver result) is provided, its curves are
    overlaid as dashed green lines for comparison.  IID curves are drawn
    segment by segment (broken at ironing gaps, which are shaded), and the
    atom of perfect screeners at alpha = 1 is marked where present.
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
    _plot_iid_curve(axes[0,0], iid, iid['r_eq'], 'r-', lw=2, label='IID')
    _shade_iid_gaps(axes[0,0], iid)
    if iid.get('pooling'):
        axes[0,0].plot(0, iid['r0'], 'r^', markersize=10,
                       label='IID (complete pooling)')
    elif iid.get('W_atom', 0) > 0:
        axes[0,0].plot(1, iid['r_perfect'], 'ro', markersize=7,
                       label='IID atom at $\\alpha=1$')

    if nested_disc is not None:
        a2_d = nested_disc['alpha2']
        axes[0,0].plot(np.linspace(alpha0, alpha1, 50),
                       nested_disc['rp'] * np.ones(50), 'g--', lw=1.5, alpha=0.7, label='Nested discrete')
        axes[0,0].plot(np.linspace(alpha1, a2_d, 50),
                       cfun(np.linspace(alpha1, a2_d, 50)) + params.Pi, 'g--', lw=1.5, alpha=0.7)
        if nested_disc['WNS'] > 0:
            axes[0,0].plot(0, nested_disc['r_NS'], 'gs', markersize=8,
                           markerfacecolor='none', markeredgewidth=2)

    # Average interest rate for good borrowers
    avg_r_nested, avg_r_iid, _, _ = compute_avg_good_rate(params, nested, iid)
    axes[0,0].axhline(avg_r_nested, color='blue', ls='--', lw=1.2, alpha=0.7,
                      label=f'Nested avg r (good) = {avg_r_nested:.3f}')
    axes[0,0].axhline(avg_r_iid, color='red', ls='--', lw=1.2, alpha=0.7,
                      label=f'IID avg r (good) = {avg_r_iid:.3f}')

    axes[0,0].axvline(alpha0, color='black', ls=':', alpha=0.3)
    axes[0,0].axvline(alpha1, color='blue', ls=':', alpha=0.5)
    axes[0,0].axvline(alpha2, color='blue', ls=':', alpha=0.5)
    axes[0,0].set_xlabel(r'$\alpha$'); axes[0,0].set_ylabel(r'$r(\alpha)$')
    axes[0,0].set_title('Interest Rate')
    axes[0,0].legend(loc='upper left', fontsize=8)
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
    _plot_iid_curve(axes[0,1], iid, iid['gamma_eq'], 'r-', lw=2, label='IID')
    _shade_iid_gaps(axes[0,1], iid)
    if not iid.get('pooling') and iid.get('W_atom', 0) > 0:
        axes[0,1].plot(1, 1.0, 'ro', markersize=7)

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
    _plot_iid_curve(axes[1,0], iid, iid['W_cumsum'], 'r-', lw=2)
    for s in _iid_segments(iid)[1:]:
        # cumulative capital is flat across an ironing gap
        prev_end = s.start - 1
        axes[1,0].plot([iid['alpha_eq'][prev_end], iid['alpha_eq'][s.start]],
                       [iid['W_cumsum'][prev_end]] * 2, 'r:', lw=1.5)
    if not iid.get('pooling') and iid.get('W_atom', 0) > 0:
        W_end_iid = iid['W_cumsum'][-1]
        axes[1,0].plot([1, 1], [W_end_iid, W_end_iid + iid['W_atom']], 'r-', lw=2)
        axes[1,0].plot(1, W_end_iid + iid['W_atom'], 'ro', markersize=7)

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
    _plot_iid_curve(axes[1,1], iid, iid['G_eq'], 'r-', lw=2, label='IID G')
    _plot_iid_curve(axes[1,1], iid, iid['B_eq'], 'r--', lw=2, label='IID B')

    if nested_disc is not None:
        axes[1,1].plot(nested_disc['alphas_R1'][:-1], nested_disc['GLOs_R1'],
                       'g-', lw=1.5, alpha=0.7, label='Discrete G')
        axes[1,1].plot(nested_disc['alphas_R1'][:-1], nested_disc['BLOs_R1'],
                       'g--', lw=1.5, alpha=0.7, label='Discrete B')
        axes[1,1].plot(nested_disc['alphas_R2'], nested_disc['GLOs_R2'],
                       'g-', lw=1.5, alpha=0.7)
        axes[1,1].plot(nested_disc['alphas_R2'], nested_disc['BLOs_R2'],
                       'g--', lw=1.5, alpha=0.7)

    # Total demand satisfied: initial mass minus final remaining
    G0 = 1.0
    B0 = params.BperG
    # Nested: final remaining after R1+R2 (NS lenders then serve all remainder)
    G_rem_nested = nested['GLOs_R2'][-1]
    B_rem_nested = nested['BLOs_R2'][-1]
    G_served_nested = G0 - G_rem_nested
    B_served_nested = B0 - B_rem_nested
    # IID: final remaining
    G_rem_iid = iid['G_eq'][-1]
    B_rem_iid = iid['B_eq'][-1]
    G_served_iid = G0 - G_rem_iid
    B_served_iid = B0 - B_rem_iid

    axes[1,1].axhline(G_served_nested, color='blue', ls='--', lw=1.2, alpha=0.7,
                      label=f'Nested G served = {G_served_nested:.3f}')
    axes[1,1].axhline(B_served_nested, color='blue', ls=':', lw=1.2, alpha=0.7,
                      label=f'Nested B served = {B_served_nested:.3f}')
    axes[1,1].axhline(G_served_iid, color='red', ls='--', lw=1.2, alpha=0.7,
                      label=f'IID G served = {G_served_iid:.3f}')
    axes[1,1].axhline(B_served_iid, color='red', ls=':', lw=1.2, alpha=0.7,
                      label=f'IID B served = {B_served_iid:.3f}')

    axes[1,1].axvline(alpha0, color='black', ls=':', alpha=0.3)
    axes[1,1].axvline(alpha1, color='blue', ls=':', alpha=0.5)
    axes[1,1].axvline(alpha2, color='blue', ls=':', alpha=0.5)
    axes[1,1].set_xlabel(r'$\alpha$'); axes[1,1].set_ylabel('Mass')
    axes[1,1].set_title('Remaining Borrowers')
    axes[1,1].legend(fontsize=7, ncol=2)
    axes[1,1].grid(alpha=0.3)
    axes[1,1].set_xlim([0, 1])

    fig.suptitle(f'Nested vs IID  ({prior_desc})', fontsize=13)
    plt.tight_layout()
    return fig

# =============================================================================
# MAIN
# =============================================================================

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


def compute_avg_good_rate(params, nested, iid):
    """
    Compute the average interest rate paid by good borrowers in each regime.

    Weighted average: sum(rate * capital_to_good) / sum(capital_to_good)

    Nested regions:
      I   (alpha0..alpha1): rate = rp,            capital_to_good = gamma(alpha)*w(alpha)*da
      II  (alpha1..alpha2): rate = cfun(al)+Pi,   capital_to_good = w(alpha)*da  [gamma=1]
      III (non-selective):  rate = r_NS,           capital_to_good = gamma_NS * WNS
    IID:
      rate = r(alpha),  capital_to_good = gamma(alpha)*w(alpha)*da
    """
    Pi = params.Pi

    # --- Nested ---
    da_R1 = nested['da_R1']
    # Region I
    cap_good_R1 = nested['gammas_R1'] * nested['ws_R1'] * da_R1
    rate_R1 = np.full_like(cap_good_R1, nested['rp'])

    # Region II (gamma = 1 throughout)
    alphas_R2 = nested['alphas_R2']
    da_R2 = alphas_R2[1] - alphas_R2[0] if len(alphas_R2) > 1 else 0.0
    cap_good_R2 = nested['ws_R2'] * da_R2          # gamma = 1
    rate_R2 = np.array([cfun(al) + Pi for al in alphas_R2])

    # Region III (non-selective)
    goodleft = nested['goodleftover_alpha2']
    badleft  = nested['badleftover']
    gamma_NS = goodleft / (goodleft + badleft) if (goodleft + badleft) > 1e-12 else 0.0
    cap_good_NS = gamma_NS * nested['WNS']
    rate_NS     = nested['r_NS']

    total_good_nested = np.sum(cap_good_R1) + np.sum(cap_good_R2) + cap_good_NS
    avg_rate_nested   = (np.sum(rate_R1 * cap_good_R1)
                         + np.sum(rate_R2 * cap_good_R2)
                         + rate_NS * cap_good_NS) / total_good_nested

    # --- IID ---
    da_iid = iid['da']
    cap_good_iid = iid['gamma_eq'] * iid['w_eq'] * da_iid
    total_good_iid = np.sum(cap_good_iid)
    avg_rate_iid   = np.sum(iid['r_eq'] * cap_good_iid) / total_good_iid

    return avg_rate_nested, avg_rate_iid, total_good_nested, total_good_iid


def plot_cumulative_lending(params, nested, iid):
    """
    Plot cumulative total lending quantity to good and bad borrowers as alpha
    increases from alpha=0 (non-selective lenders in nested) upward.

    Quantity = D(r) * borrower mass served per unit alpha.  Since w(alpha)*da
    already equals D(r)*mass (the solvers build D(r) into w), we can split
    w*da into good and bad using gamma:
        good contribution at alpha = gamma(alpha) * w(alpha) * da
        bad  contribution at alpha = (1-gamma(alpha)) * w(alpha) * da

    Nested ordering (lowest to highest alpha):
        alpha=0  : non-selective (Region III) lenders — jump of WNS
        0..alpha0: flat (no additional lenders in nested)
        alpha0..alpha1: Region I  (pooling rate rp, mixed pool)
        alpha1..alpha2: Region II (screening, gamma=1, only good borrowers)

    IID ordering:
        0..alpha0: zero lending
        alpha0..alpha_bar: IID selective lenders
    """
    # -------------------------------------------------------------------------
    # Nested: Region III (non-selective, at alpha=0)
    # -------------------------------------------------------------------------
    goodleft = nested['goodleftover_alpha2']
    badleft  = nested['badleftover']
    gamma_NS = goodleft / (goodleft + badleft) if (goodleft + badleft) > 1e-12 else 0.0
    WNS       = nested['WNS']
    Q_NS_good = WNS * gamma_NS
    Q_NS_bad  = WNS * (1.0 - gamma_NS)

    # -------------------------------------------------------------------------
    # Nested: Region I (alpha0 to alpha1)
    # -------------------------------------------------------------------------
    da_R1     = nested['da_R1']
    good_R1   = nested['gammas_R1'] * nested['ws_R1'] * da_R1   # per-step quantities
    bad_R1    = (1.0 - nested['gammas_R1']) * nested['ws_R1'] * da_R1

    cum_good_R1 = Q_NS_good + np.concatenate([[0.0], np.cumsum(good_R1[:-1])])
    cum_bad_R1  = Q_NS_bad  + np.concatenate([[0.0], np.cumsum(bad_R1[:-1])])

    # -------------------------------------------------------------------------
    # Nested: Region II (alpha1 to alpha2, gamma=1 so all lending to good)
    # -------------------------------------------------------------------------
    alphas_R2 = nested['alphas_R2']
    da_R2     = alphas_R2[1] - alphas_R2[0] if len(alphas_R2) > 1 else 0.01
    good_R2   = nested['ws_R2'] * da_R2
    bad_R2    = np.zeros_like(good_R2)

    cum_good_R2 = cum_good_R1[-1] + np.concatenate([[0.0], np.cumsum(good_R2[:-1])])
    cum_bad_R2  = cum_bad_R1[-1]  + np.concatenate([[0.0], np.cumsum(bad_R2[:-1])])

    # -------------------------------------------------------------------------
    # IID (alpha0 to alpha_bar)
    # -------------------------------------------------------------------------
    da_iid      = iid['da']
    good_iid    = iid['gamma_eq'] * iid['w_eq'] * da_iid
    bad_iid     = (1.0 - iid['gamma_eq']) * iid['w_eq'] * da_iid

    cum_good_iid = np.concatenate([[0.0], np.cumsum(good_iid[:-1])])
    cum_bad_iid  = np.concatenate([[0.0], np.cumsum(bad_iid[:-1])])

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    alpha0 = nested['alpha0']
    alpha1 = nested['alpha1']
    alpha2 = nested['alpha2']
    alpha_ns_flat  = np.linspace(0.0, alpha0, 30)
    alpha_pre_iid  = np.linspace(0.0, iid['alpha0'], 20)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    prior_desc = ("uniform priors" if params.a_g == 0 and params.a_b == 0
                  else f"a_g={params.a_g}, a_b={params.a_b}")

    for ax, (cum_n_R1, cum_n_R2, cum_i, Q_ns, title) in zip(
        axes,
        [
            (cum_good_R1, cum_good_R2, cum_good_iid, Q_NS_good, 'Good Borrowers'),
            (cum_bad_R1,  cum_bad_R2,  cum_bad_iid,  Q_NS_bad,  'Bad Borrowers'),
        ]
    ):
        # Nested: vertical jump at alpha=0 (NS lenders), then flat to alpha0
        ax.plot([0, 0], [0, Q_ns], 'b-', lw=2)
        ax.plot(alpha_ns_flat, Q_ns * np.ones_like(alpha_ns_flat), 'b-', lw=2,
                label='Nested')
        # Region I
        ax.plot(nested['alphas_R1'], cum_n_R1, 'b-', lw=2)
        # Region II
        ax.plot(nested['alphas_R2'], cum_n_R2, 'b-', lw=2)

        # IID: zero until alpha0, then cumulative (broken at ironing gaps;
        # flat dotted bridge across each gap, no lending there)
        ax.plot(alpha_pre_iid, np.zeros_like(alpha_pre_iid), 'r-', lw=2, label='IID')
        _plot_iid_curve(ax, iid, cum_i, 'r-', lw=2)
        for s in _iid_segments(iid)[1:]:
            prev_end = s.start - 1
            ax.plot([iid['alpha_eq'][prev_end], iid['alpha_eq'][s.start]],
                    [cum_i[prev_end]] * 2, 'r:', lw=1.5)
        if title == 'Good Borrowers' and iid.get('W_atom', 0) > 0:
            # atom of perfect screeners: all-good lending jump at alpha = 1
            ax.plot([1, 1], [cum_i[-1], cum_i[-1] + iid['W_atom']], 'r-', lw=2)
            ax.plot(1, cum_i[-1] + iid['W_atom'], 'ro', markersize=6)

        # Boundary markers
        ax.axvline(alpha0, color='blue', ls=':', alpha=0.4, label=f'alpha0={alpha0:.3f}')
        ax.axvline(alpha1, color='blue', ls='--', alpha=0.4, label=f'alpha1={alpha1:.3f}')
        ax.axvline(alpha2, color='blue', ls='-.', alpha=0.4, label=f'alpha2={alpha2:.3f}')

        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel('Cumulative lending quantity  D(r) x mass')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])

    fig.suptitle(f'Cumulative Lending by Borrower Type  ({prior_desc})', fontsize=13)
    plt.tight_layout()
    return fig


# =============================================================================
# IID CASE DEMOS  (the three configurations of Notes/ironing_iid.tex)
# =============================================================================

def run_iid_case_demos(save_dir=None, show_titles=True):
    """Solve and plot the three iid equilibrium configurations of the ironing
    note (Notes/ironing_iid.tex), using its exact parameterizations mapped
    into this module's convention K(alpha) = Pi + C(alpha):

      Case 1 (fully separating):  K = 0.10 + 0.50 a^2,                q0 = 0.60
      Case 2 (top jump):          K = 0.10 + 0.48(3a^2-2a^3) + 0.02a, q0 = 0.75
      Case 3 (interior ironing):  K = 0.05 + 0.55 a + 0.02 sin(4 pi a), q0 = 0.70

    all with beta = 0.5 and uniform priors (q0 pins BperG = (1-q0)/q0).
    The module cost function is overridden for the duration and restored on
    exit (cfun_prime / cfun_prime2 are numerical, so they follow along).
    Saves credit_model_iid_cases.png; returns the list of solve_iid results.
    """
    import os
    global cfun
    old_cfun = cfun
    specs = [
        ("Case 1: fully separating (no gaps)",
         lambda a: 0.50 * np.asarray(a, dtype=float)**2, 0.10, 0.40 / 0.60),
        ("Case 2: top jump -- entry stops early, atom at alpha=1",
         lambda a: 0.48 * (3 * np.asarray(a, dtype=float)**2
                           - 2 * np.asarray(a, dtype=float)**3)
                   + 0.02 * np.asarray(a, dtype=float), 0.10, 0.25 / 0.75),
        ("Case 3: interior ironing -- a gap inside the schedule",
         lambda a: 0.55 * np.asarray(a, dtype=float)
                   + 0.02 * np.sin(4 * np.pi * np.asarray(a, dtype=float)), 0.05, 0.30 / 0.70),
    ]
    results = []
    try:
        for label, cost, Pi_v, bpg in specs:
            cfun = cost
            p = Parameters(Pi=Pi_v, beta=0.5, BperG=bpg)
            print(f"\n--- {label} ---")
            res = solve_iid(p)
            print(f"  case={res['case']}  alpha0={res['alpha0']:.4f}  "
                  f"alpha_bar={res['alpha_bar']:.4f}  "
                  f"gaps={[(round(a, 3), round(b, 3)) for (a, b, _, _) in res['gaps']]}  "
                  f"W_atom={res['W_atom']:.4f}")
            results.append((label, p, res))
    finally:
        cfun = old_cfun

    fig, axes = plt.subplots(3, 2, figsize=(11, 12))
    C_GAM, C_Q = '#3182bd', '#e6550d'
    for row, (label, p, res) in enumerate(results):
        axL, axR = axes[row]
        rK1 = res['r_perfect']

        # left: interest rate schedule
        axL.axhline(rK1, color='0.5', ls='--', lw=0.8)
        _plot_iid_curve(axL, res, res['r_eq'], 'r-', lw=2)
        _shade_iid_gaps(axL, res)
        if res.get('W_atom', 0) > 0:
            axL.plot(1, rK1, 'ro', markersize=7)
            axL.annotate('atom', (1, rK1), xytext=(0.97, rK1 - 0.05),
                         ha='right', fontsize=9)
        axL.annotate('$K(1)$', (0.02, rK1), xytext=(0.02, rK1 + 0.005),
                     fontsize=9, color='0.4')
        axL.set_ylabel(r'$r(\alpha)$')
        if show_titles:
            axL.set_title(label, loc='left', fontsize=10)
        axL.set_xlim([0, 1.03]); axL.grid(alpha=0.3)

        # right: pool quality z and repayment probability gamma
        _plot_iid_curve(axR, res, res['gamma_eq'], '-', lw=2)
        for line in axR.get_lines()[-len(_iid_segments(res)):]:
            line.set_color(C_GAM)
        _plot_iid_curve(axR, res, res['z_eq'], '--', lw=2)
        for line in axR.get_lines()[-len(_iid_segments(res)):]:
            line.set_color(C_Q)
        _shade_iid_gaps(axR, res)
        if res.get('W_atom', 0) > 0:
            axR.plot(1, 1.0, 'o', color=C_GAM, markersize=7)
        i0 = max(len(res['alpha_eq']) // 10, 0)
        axR.annotate(r'$\gamma(\alpha,\alpha)$',
                     (res['alpha_eq'][i0], res['gamma_eq'][i0] + 0.04),
                     fontsize=9, color='0.2', ha='center')
        axR.annotate(r'$q(\alpha)$',
                     (res['alpha_eq'][i0], res['z_eq'][i0] - 0.06),
                     fontsize=9, color='0.2', ha='center')
        axR.set_ylabel(r'$q(\alpha)$, $\gamma(\alpha,\alpha)$')
        axR.set_xlim([0, 1.03]); axR.set_ylim([0, 1.08]); axR.grid(alpha=0.3)
        if row == 2:
            axL.set_xlabel(r'$\alpha$'); axR.set_xlabel(r'$\alpha$')

    fig.suptitle('IID equilibrium: the three configurations '
                 '(Notes/ironing_iid.tex)', fontsize=12)
    fig.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(save_dir or script_dir, 'credit_model_iid_cases.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nIID case-demo plot saved to {out}")
    return results


# =============================================================================
# DENSITY-EVOLUTION FIGURE  (panels (e,f) of fig:density in the draft)
# =============================================================================
#
# Reproduces the MATLAB `densities` figure (secondversion/main.m): the pool of
# good / bad borrowers a lender of skill alpha faces, after every lower-skill
# lender in the Region-I queue has progressively depleted it.  The depletion
# recursion is identical to the one inside solve_nested (line ~219); here we
# additionally snapshot the full densities g(omega), b(omega) at a set of alpha
# values so they can be plotted.

def nested_density_snapshots(params, n_curves=6, delom=None):
    """Run the nested Region-I depletion recursion, storing snapshots of the
    good/bad densities at n_curves alpha values spanning [alpha0, alpha1].

    A snapshot is taken *after* the lenders of skill alpha have acted and is
    labelled by that alpha, matching the MATLAB convention (first curve =
    alpha_0, last curve = alpha_1).

    delom must be fine enough to resolve the per-step slivers of width
    Delta*min(beta, 1-beta); too coarse a grid makes the solver over-deplete
    (curves collapse to 0/1).  Defaults to params.delom (same as solve_nested).
    """
    global _current_params
    _current_params = params
    Pi, beta, BperG = params.Pi, params.beta, params.BperG
    Delta = params.Delta
    if delom is None:
        delom = params.delom

    # alpha0, rp, alpha1 -- identical to solve_nested
    res = minimize_scalar(
        lambda a: (Pi + cfun(a) + 1) / gam0(a, params) if 0 < a < 1 else 1e10,
        bounds=(0.01, 0.99), method='bounded')
    alpha0 = res.x
    rp = (Pi + cfun(alpha0) + 1) / gam0(alpha0, params) - 1
    if rp - cfun(1) > Pi:
        alpha1 = 1.0
    else:
        alpha1 = brentq(lambda a: cfun(a) - (rp - Pi), 0.01, 0.99)

    n_om = int(round(1 / delom))
    omvec = np.linspace(0, 1, n_om)
    g, b = gpriorfun(omvec), bpriorfun(omvec, BperG)
    D_rp = dfun(rp)

    targets = np.linspace(alpha0, alpha1, n_curves)
    g_snaps, b_snaps, snap_alphas = [], [], []
    ti = 0

    alpha = alpha0
    while alpha + Delta <= alpha1 + 1e-8:
        acting = alpha
        omega_g_alp = beta + alpha * (1 - beta)
        omega_b_alp = 1 - beta + alpha * beta
        mask_g, mask_b = omvec <= omega_g_alp, omvec >= omega_b_alp
        G_alp = np.sum(delom * g[mask_g])
        B_alp = np.sum(delom * b[mask_b])
        T_alp = G_alp + B_alp
        if T_alp < 1e-10:
            break

        alpha_next = min(alpha + Delta, alpha1)
        req_gamma = (1 + Pi + cfun(alpha_next)) / (1 + rp)
        omega_g_next = beta + alpha_next * (1 - beta)
        omega_b_next = 1 - beta + alpha_next * beta

        def gamma_after_w(w, mask_g=mask_g, mask_b=mask_b, T_alp=T_alp):
            scale = w / (T_alp * D_rp)
            if scale >= 1:
                return 1.0
            g_new = g * (1 - mask_g * scale)
            b_new = b * (1 - mask_b * scale)
            G_n = np.sum(delom * g_new[omvec <= omega_g_next])
            B_n = np.sum(delom * b_new[omvec >= omega_b_next])
            return G_n / (G_n + B_n) if G_n + B_n > 1e-10 else 1.0

        g0v, gmax = gamma_after_w(0), gamma_after_w(T_alp * D_rp * 0.999)
        if req_gamma <= g0v:
            w_opt = 0.0
        elif req_gamma >= gmax:
            w_opt = T_alp * D_rp * 0.999
        else:
            w_opt = brentq(lambda w: gamma_after_w(w) - req_gamma,
                           0, T_alp * D_rp * 0.999)

        scale = w_opt / (T_alp * D_rp)
        g = g * (1 - mask_g * scale)
        b = b * (1 - mask_b * scale)
        alpha = alpha_next

        # snapshot once the acting skill reaches the next target alpha
        while ti < n_curves and acting >= targets[ti] - 1e-9:
            g_snaps.append(g.copy())
            b_snaps.append(b.copy())
            snap_alphas.append(acting)
            ti += 1

    # ensure the final (alpha ~ alpha1) state is captured as the last curve
    while ti < n_curves:
        g_snaps.append(g.copy())
        b_snaps.append(b.copy())
        snap_alphas.append(alpha)
        ti += 1

    return dict(omvec=omvec, alphas=np.array(snap_alphas),
                g_snaps=g_snaps, b_snaps=b_snaps,
                alpha0=alpha0, alpha1=alpha1, rp=rp, beta=beta)


def _draw_density_panel(ax, omvec, snaps, alphas, beta, kind, thr0, thr1,
                        thr0_lab, thr1_lab, ylabel, ylim_top,
                        fs_label=16, fs_tick=13, fs_leg=12, fs_thr=12,
                        leg_loc='best'):
    """Draw one density panel (bad or good) onto ax, styled to match the
    TikZ panels of fig:density (red = alpha_0 .. green = alpha_1, line width
    increasing with skill, dotted threshold lines)."""
    n = len(alphas)
    c_lo = np.array([0.84, 0.10, 0.11])   # red  (alpha_0, least skilled)
    c_hi = np.array([0.00, 0.39, 0.00])   # green!60!black (alpha_1)
    lws = np.linspace(1.1, 2.8, n)

    labels = [r'$\alpha=\alpha_0$']
    labels += [rf'$\alpha={a:.2f}$' for a in alphas[1:-1]]
    labels += [r'$\alpha=\alpha_1$']

    for i in range(n):
        col = tuple(c_lo + (c_hi - c_lo) * (i / (n - 1) if n > 1 else 0))
        ax.plot(omvec, snaps[i], color=col, lw=lws[i], label=labels[i])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, ylim_top)
    ax.set_xlabel(r'$\omega$', fontsize=fs_label)
    # Axis name at the panel's top-left corner, to the left of the y-axis (as in
    # the TikZ panels above), instead of a rotated left label.
    ax.figure.text(0.01, 0.94, ylabel, ha='left', va='bottom', fontsize=fs_label)
    # Minimal unit ticks only (no 0.2/0.4/... ladder), matching the TikZ panels.
    ax.set_xticks([0, 1]); ax.set_xticklabels(['0', '1'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['0', '1'])
    ax.tick_params(labelsize=fs_tick)

    # threshold lines + labels (staggered top/bottom as in the MATLAB figure)
    for x, lab, where in ((thr0, thr0_lab, 'bottom' if kind == 'bad' else 'top'),
                          (thr1, thr1_lab, 'top' if kind == 'bad' else 'bottom')):
        ax.axvline(x, color='0.35', ls=':', lw=0.8)
        y = ylim_top * 0.96 if where == 'top' else ylim_top * 0.04
        va = 'top' if where == 'top' else 'bottom'
        ax.text(x, y, lab, fontsize=fs_thr, ha='center', va=va,
                bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none',
                          alpha=0.75))

    ax.legend(loc=leg_loc, fontsize=fs_leg, framealpha=0.85)


def plot_density_panels(params, save_dir=None, n_curves=6, figsize=(3.8, 4.0),
                        cost=None):
    """Generate the two density panels (bad pool, good pool) as separate vector
    PDFs, sized to drop into fig:density as panels (e) and (f).

    Saves densities_bad.pdf and densities_good.pdf into save_dir (defaults to
    the draft folder Peter-Pablo-Maryam/draft, falling back to the script dir).
    Returns (fig_bad, fig_good).

    cost, if given as (C2, P2, C1, P1), temporarily overrides the module cost
    constants for this figure only (the cost enters only via
    nested_density_snapshots).  This lets the density figure use the figs-8/9
    calibration (beta=0.5 etc.) while the other credit_model.py figures keep
    the Parameters() baseline.
    """
    import os
    global COST_C2, COST_P2, COST_C1, COST_P1
    if cost is not None:
        _saved_cost = (COST_C2, COST_P2, COST_C1, COST_P1)
        COST_C2, COST_P2, COST_C1, COST_P1 = cost
        try:
            snaps = nested_density_snapshots(params, n_curves=n_curves)
        finally:
            COST_C2, COST_P2, COST_C1, COST_P1 = _saved_cost
    else:
        snaps = nested_density_snapshots(params, n_curves=n_curves)
    omvec = snaps['omvec']
    alphas = snaps['alphas']
    alpha0, alpha1, beta = snaps['alpha0'], snaps['alpha1'], snaps['beta']
    BperG = params.BperG

    og0, og1 = beta + (1 - beta) * alpha0, beta + (1 - beta) * alpha1
    ob0, ob1 = (1 - beta) + beta * alpha0, (1 - beta) + beta * alpha1

    # --- Geometry: match the TikZ unit square of fig:density (rows a-d) ---
    # The TikZ panels draw the 1x1 prior pool as a square 1.0 wide x 0.9 tall
    # inside a 1.40 x 1.32 bbox, shown at \resizebox{!}{0.25\textheight}, i.e.
    # 0.1894 x 0.1705 \textheight.  Here we fix the axes box on the figure (no
    # tight bbox) so the [0,1]x[0,1] data square occupies f_h = AH/YMAX of the
    # image height and A*AW of its width (A = figure aspect).  With the values
    # below and the same \includegraphics[height=0.25\textheight] in the draft,
    # the continuous panels' unit square renders at the same size as the TikZ
    # ones:  height 0.25*f_h = 0.170\textheight, width 0.25*A*AW = 0.189\textheight.
    YMAX = 1.1
    # Mirror the TikZ bbox (1.40 x 1.32, unit square 1.0 x 0.9, left margin 0.10):
    # using the same margin fractions and aspect, the continuous panels line up
    # column-wise with the rows above when both sit at height 0.25\textheight.
    FIGSIZE = (5.3, 5.0)                    # aspect A = 1.06 = 1.40/1.32
    AXBOX = [0.0714, 0.1667, 0.714, 0.75]  # [L,B,AW,AH] = TikZ margin fractions
    FS = dict(fs_label=16, fs_tick=14, fs_leg=12, fs_thr=13)

    # --- Bad pool ---  (legend lower-left: the empty corner once omega<omega_b)
    figB = plt.figure(figsize=FIGSIZE); axB = figB.add_axes(AXBOX)
    _draw_density_panel(
        axB, omvec, snaps['b_snaps'], alphas, beta, kind='bad',
        thr0=ob0, thr1=ob1,
        thr0_lab=r'$\omega_b(\alpha_0)$', thr1_lab=r'$\omega_b(\alpha_1)$',
        ylabel=r'$b(\omega;r_p,1,\alpha)$', ylim_top=YMAX,
        leg_loc='lower left', **FS)

    # --- Good pool ---
    figG = plt.figure(figsize=FIGSIZE); axG = figG.add_axes(AXBOX)
    _draw_density_panel(
        axG, omvec, snaps['g_snaps'], alphas, beta, kind='good',
        thr0=og0, thr1=og1,
        thr0_lab=r'$\omega_g(\alpha_0)$', thr1_lab=r'$\omega_g(\alpha_1)$',
        ylabel=r'$g(\omega;r_p,1,\alpha)$', ylim_top=YMAX,
        leg_loc='upper left', **FS)

    # resolve output directory: draft folder if reachable, else script dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if save_dir is None:
        draft = os.path.normpath(os.path.join(
            script_dir, '..', '..', '..', '..', 'Peter-Pablo-Maryam', 'draft'))
        save_dir = draft if os.path.isdir(draft) else script_dir

    path_b = os.path.join(save_dir, 'densities_bad.pdf')
    path_g = os.path.join(save_dir, 'densities_good.pdf')
    figB.savefig(path_b)   # fixed-geometry figure: preserve the exact axes box
    figG.savefig(path_g)
    print(f"Density panels saved to:\n  {path_b}\n  {path_g}")
    print(f"  alpha0={alpha0:.4f}, alpha1={alpha1:.4f}, "
          f"snapshot alphas={np.round(alphas, 3)}")
    return figB, figG


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
    print(f"Total W (continuous part) = {iid['W_cumsum'][-1]:.4f}")
    print(f"case = {iid['case']}"
          + (f", ironing gaps = "
             f"{[(round(a, 3), round(b, 3)) for (a, b, _, _) in iid['gaps']]}"
             if iid['gaps'] else "")
          + f", atom at alpha=1: W_atom = {iid['W_atom']:.4f}")

    # Average interest rate for good borrowers
    avg_r_nested, avg_r_iid, W_good_nested, W_good_iid = compute_avg_good_rate(params, nested_a, iid)
    print("\n--- Average Interest Rate (good borrowers) ---")
    print(f"Nested: {avg_r_nested:.6f}  (capital to good = {W_good_nested:.4f})")
    print(f"IID:    {avg_r_iid:.6f}  (capital to good = {W_good_iid:.4f})")
    print(f"Diff (nested - iid): {avg_r_nested - avg_r_iid:+.6f}")

    # Plot using analytical solver as primary nested solution
    print("\n--- Creating Plot ---")
    fig = plot_comparison(params, nested_a, iid, nested_disc=nested)

    # Save in same folder as script
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'credit_model.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    # Cumulative lending by borrower type
    fig2 = plot_cumulative_lending(params, nested_a, iid)
    output_path2 = os.path.join(script_dir, 'credit_model_lending.png')
    fig2.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Lending plot saved to {output_path2}")

    # IID case demos: the three equilibrium configurations of the ironing note
    # (fully separating / top jump / interior ironing), each solved and plotted.
    print("\n--- IID Case Demos (Notes/ironing_iid.tex configurations) ---")
    run_iid_case_demos()

    # Density-evolution panels (e,f) of fig:density in the draft.
    # Uses the figs-8/9 calibration (Pi=0.235, beta=0.5, BperG=1.0,
    # cfun=9a^2+0.2a) -- beta=0.5 gives wide omega acceptance bands so the
    # cleansing gradient is legible, and matches the entry/spillover figures.
    print("\n--- Creating Density Panels (fig:density e,f) ---")
    plot_density_panels(Parameters(Pi=0.235, beta=0.5, BperG=1.0),
                        cost=(9.0, 2.0, 0.2, 1.0))

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
