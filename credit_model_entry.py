"""
Credit Market Entry Equilibrium: T(alpha) Construction

Computes the equilibrium when new entrants arrive into an incumbent
nested-information equilibrium.  Uses the algebraic T(alpha) approach
from entry_Talpha_construction.tex.

Usage:
    python credit_model_entry.py

Entrant parameters are set in EntrantParams and can be changed easily.
"""
import numpy as np
from scipy.optimize import minimize_scalar, brentq
from scipy.integrate import quad
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings
import os

warnings.filterwarnings('ignore')

# Import the incumbent solver
from credit_model import (
    Parameters, solve_nested_analytical,
    gpriorfun, bpriorfun, dfun, cfun, cfun_prime_exact,
    gam0, find_alpha2, _current_params
)
import credit_model as cm


# =============================================================================
# ENTRANT PARAMETERS (easily modifiable)
# =============================================================================

@dataclass
class EntrantParams:
    """Parameters for new entrants.

    Cost function: c^E(alpha) = CE_C * alpha^CE_P
    Total adjusted cost: K^E(alpha) = Pi_E + c^E(alpha)
    """
    Pi_E: float = 0.03        # Entrant cost of capital (lower than incumbent Pi=0.05)
    CE_C: float = 0.8         # Entrant cost coefficient
    CE_P: float = 2.0         # Entrant cost exponent (power function)


def cfun_E(alpha, ep):
    """Entrant screening cost function c^E(alpha) = CE_C * alpha^CE_P."""
    return ep.CE_C * alpha**ep.CE_P


def cfun_E_prime(alpha, ep):
    """First derivative of entrant cost: c^E'(alpha)."""
    return ep.CE_C * ep.CE_P * alpha**(ep.CE_P - 1)


def cfun_E_prime2(alpha, ep):
    """Second derivative of entrant cost: c^E''(alpha)."""
    if ep.CE_P <= 1:
        return 0.0
    return ep.CE_C * ep.CE_P * (ep.CE_P - 1) * alpha**(ep.CE_P - 2)


def K_E(alpha, ep):
    """Total adjusted cost for entrant: K^E(alpha) = Pi_E + c^E(alpha)."""
    return ep.Pi_E + cfun_E(alpha, ep)


def K_E_prime(alpha, ep):
    """Derivative of K^E: (K^E)'(alpha) = (c^E)'(alpha)."""
    return cfun_E_prime(alpha, ep)


# =============================================================================
# ENTRY EQUILIBRIUM SOLVER
# =============================================================================

def solve_entry(params, ep, incumbent=None):
    """
    Solve the entry equilibrium given incumbent equilibrium and entrant parameters.

    Parameters
    ----------
    params : Parameters
        Incumbent model parameters (Pi, beta, BperG, priors).
    ep : EntrantParams
        Entrant parameters (Pi_E, cost function).
    incumbent : dict or None
        Pre-computed incumbent equilibrium (from solve_nested_analytical).
        If None, computes it.

    Returns
    -------
    dict with entry equilibrium results.
    """
    global _current_params
    cm._current_params = params

    beta, BperG = params.beta, params.BperG

    # =========================================================================
    # Step 0: Compute incumbent equilibrium
    # =========================================================================
    if incumbent is None:
        incumbent = solve_nested_analytical(params)

    inc = incumbent
    alpha0_inc = inc['alpha0']
    alpha1_inc = inc['alpha1']
    alpha2_inc = inc['alpha2']
    rp_inc = inc['rp']
    r_NS_inc = inc['r_NS']

    # =========================================================================
    # Step 1: Find entrant alpha0^E and rp^E
    # =========================================================================
    # Entrant faces the same initial pool as incumbent (gamma_0 is the same)
    res = minimize_scalar(
        lambda a: (K_E(a, ep) + 1) / gam0(a, params) if 0 < a < 1 else 1e10,
        bounds=(0.01, 0.99), method='bounded'
    )
    alpha0_E = res.x
    rp_E = (K_E(alpha0_E, ep) + 1) / gam0(alpha0_E, params) - 1

    # Check: if entrant pooling rate >= incumbent pooling rate, no entry in Region I
    entry_in_R1 = rp_E < rp_inc

    if not entry_in_R1:
        print("  No entry in Region I (rp_E >= rp_inc)")
        print(f"  rp_E = {rp_E:.4f}, rp_inc = {rp_inc:.4f}")
        return _solve_entry_no_R1(params, ep, inc)

    # Also check: if alpha0_E >= alpha1_inc, the entrant's optimal alpha
    # is above incumbent Region I, so no entry in R1
    if alpha0_E >= alpha1_inc:
        print("  No entry in Region I (alpha0_E >= alpha1_inc)")
        print(f"  alpha0_E = {alpha0_E:.4f}, alpha1_inc = {alpha1_inc:.4f}")
        return _solve_entry_no_R1(params, ep, inc)

    # =========================================================================
    # Step 2: Find alpha1^E (upper boundary of entrant pooling region)
    # =========================================================================
    # Indifference boundary: rp_E - c^E(alpha1''^E) = Pi_E
    # i.e. c^E(alpha1''^E) = rp_E - Pi_E
    target_cE = rp_E - ep.Pi_E
    if target_cE <= 0:
        alpha1_indiff = 0.0
    elif cfun_E(1.0, ep) < target_cE:
        alpha1_indiff = 1.0
    else:
        alpha1_indiff = brentq(lambda a: cfun_E(a, ep) - target_cE, 0.01, 0.99)

    # CIM boundary: smallest alpha such that incumbent capital alone suffices
    # for all alpha' in [alpha, alpha1_inc]
    # This requires checking where the incumbent w(alpha) can sustain the
    # good-only lending at rate >= rp_E.
    alpha1_CIM = _find_alpha1_CIM(inc, rp_E, beta)

    alpha1_E = max(alpha1_indiff, alpha1_CIM)
    alpha1_E = min(alpha1_E, alpha1_inc)  # can't exceed incumbent alpha1

    # =========================================================================
    # Step 3: Region I — Combined T^E(alpha) construction
    # =========================================================================
    n_R1 = 2000
    alphas_R1 = np.linspace(alpha0_E, alpha1_E, n_R1)
    da_R1 = alphas_R1[1] - alphas_R1[0] if n_R1 > 1 else 1e-6

    omega_g_vals = beta + alphas_R1 * (1 - beta)
    omega_b_vals = 1 - beta + alphas_R1 * beta

    # Prior-dependent inputs
    g_tilde = np.array([gpriorfun(og) for og in omega_g_vals])
    b_tilde = np.array([bpriorfun(ob, BperG) for ob in omega_b_vals])
    B0_tilde = np.array([quad(lambda x: bpriorfun(x, BperG), ob, 1)[0]
                         for ob in omega_b_vals])

    # K^E(alpha) and Gamma^E(alpha) = (1 + K^E)/(1 + rp_E)
    KE_vals = np.array([K_E(a, ep) for a in alphas_R1])
    Gamma_E = (1 + KE_vals) / (1 + rp_E)
    Gamma_E_prime = np.array([K_E_prime(a, ep) for a in alphas_R1]) / (1 + rp_E)
    one_minus_Gamma_E = 1 - Gamma_E

    # T^E formula (same structure as incumbent, with K^E, rp_E)
    # T^E = (1-beta)*g_tilde*(rp_E - K^E)*B0_tilde*(1+rp_E)
    #       / [(K^E)'*B0_tilde*(1+rp_E) - beta*b_tilde*(1+K^E)*(rp_E - K^E)]
    rp_minus_KE = rp_E - KE_vals
    one_plus_KE = 1 + KE_vals

    numer = (1 - beta) * g_tilde * rp_minus_KE * B0_tilde * (1 + rp_E)
    denom_T = (Gamma_E_prime * (1 + rp_E) * B0_tilde
               - beta * b_tilde * one_plus_KE * rp_minus_KE)
    # Note: Gamma_E_prime * (1 + rp_E) = K_E_prime, so this is equivalent to:
    # denom_T = K_E_prime * B0_tilde - beta * b_tilde * (1+K^E)*(rp_E - K^E) / (1+rp_E)
    # But let's use the direct form from the formula:
    KE_prime_vals = np.array([K_E_prime(a, ep) for a in alphas_R1])
    denom_T = (KE_prime_vals * B0_tilde * (1 + rp_E)
               - beta * b_tilde * one_plus_KE * rp_minus_KE)
    denom_T_safe = np.where(np.abs(denom_T) < 1e-15, 1e-15, denom_T)

    TE_vals = numer / denom_T_safe

    # Good and bad in acceptance region
    GE_vals = Gamma_E * TE_vals
    BE_vals = one_minus_Gamma_E * TE_vals
    gamma_E_vals = np.where(TE_vals > 1e-15, GE_vals / TE_vals, 1.0)

    # Depletion factor
    EE_vals = BE_vals / np.where(np.abs(B0_tilde) < 1e-15, 1e-15, B0_tilde)

    # theta^E = -d(ln B^E)/dalpha + beta*b_tilde/B0_tilde
    # where B^E = (rp_E - K^E)*T^E/(1+rp_E)
    ln_BE = np.log(np.maximum(BE_vals, 1e-30))
    theta_E_vals = np.zeros_like(alphas_R1)
    theta_E_vals[1:-1] = -(ln_BE[2:] - ln_BE[:-2]) / (2 * da_R1)
    theta_E_vals[0] = -(ln_BE[1] - ln_BE[0]) / da_R1
    theta_E_vals[-1] = -(ln_BE[-1] - ln_BE[-2]) / da_R1
    # Add the bad-threshold contribution
    theta_E_vals += beta * b_tilde / np.where(np.abs(B0_tilde) < 1e-15, 1e-15, B0_tilde)

    # Total lending density needed in combined system
    D_rpE = dfun(rp_E)
    w_total_E = theta_E_vals * D_rpE * TE_vals
    w_total_E = np.maximum(w_total_E, 0)

    # Incumbent lending density in this region
    # Interpolate from incumbent solution
    w_inc_interp = np.interp(alphas_R1, inc['alphas_R1'], inc['ws_R1'],
                             left=0, right=0)

    # Entrant density = total needed - incumbent
    wE_R1 = w_total_E - w_inc_interp
    # Ironing: entrant density can't be negative
    wE_R1 = np.maximum(wE_R1, 0)

    # Cumulative entrant capital in Region I
    WE_cumsum_R1 = np.zeros_like(alphas_R1)
    WE_cumsum_R1[1:] = np.cumsum(wE_R1[:-1]) * da_R1

    # =========================================================================
    # Step 4: Remaining borrowers after Region I
    # =========================================================================
    # Good outside acceptance: integral of g from omega_g to 1
    G_outside_R1 = np.array([quad(gpriorfun, og, 1)[0] for og in omega_g_vals])
    G_leftover_R1 = GE_vals + G_outside_R1

    # Bad remaining
    omega_b_0 = 1 - beta + alpha0_E * beta
    B_below_0 = quad(lambda x: bpriorfun(x, BperG), 0, omega_b_0)[0]
    b_tilde_E_beta = b_tilde * EE_vals * beta
    B_dropouts = np.zeros_like(alphas_R1)
    B_dropouts[1:] = np.cumsum(b_tilde_E_beta[:-1]) * da_R1
    B_leftover_R1 = B_below_0 + B_dropouts + BE_vals

    G_end_R1 = G_leftover_R1[-1]
    B_end_R1 = B_leftover_R1[-1]
    badleftover_E = B_end_R1

    # =========================================================================
    # Step 5: Region II — CIM region
    # =========================================================================
    # In Region II, lenders serve only their own-slice good borrowers.
    # The interest rate is min(incumbent CIM rate, entrant CIM rate).
    # Entrant CIM rate: r = c^E(alpha) + Pi_E
    # Incumbent CIM rate: r = c(alpha) + Pi

    # Find alpha2^E: where non-selective entry becomes viable
    alpha2_E = _find_alpha2_E(alpha1_E, beta, badleftover_E, ep, params)
    if alpha2_E is None:
        alpha2_E = 1.0

    n_R2 = max(int((alpha2_E - alpha1_E) * 200), 10)
    alphas_R2 = np.linspace(alpha1_E, alpha2_E, n_R2)
    da_R2 = alphas_R2[1] - alphas_R2[0] if n_R2 > 1 else 0.01

    omega_g_R2 = beta + alphas_R2 * (1 - beta)
    g_tilde_R2 = np.array([gpriorfun(og) for og in omega_g_R2])

    # CIM rates: minimum of entrant and incumbent
    r_CIM_inc = np.array([cfun(a) + params.Pi for a in alphas_R2])
    r_CIM_ent = np.array([cfun_E(a, ep) + ep.Pi_E for a in alphas_R2])
    r_CIM = np.minimum(r_CIM_inc, r_CIM_ent)

    # Who serves: incumbent or entrant?
    entrant_serves_R2 = r_CIM_ent <= r_CIM_inc

    # w(alpha) = D(r) * g_tilde * (1-beta) for the serving lender
    ws_R2_total = np.array([dfun(r_CIM[i]) * (1 - beta) * g_tilde_R2[i]
                            for i in range(n_R2)])

    # Incumbent capital in this region
    w_inc_R2 = np.interp(alphas_R2, inc['alphas_R1'], inc['ws_R1'],
                         left=0, right=0)
    # Also check Region II of incumbent
    if len(inc['alphas_R2']) > 1:
        w_inc_R2_from_R2 = np.interp(alphas_R2, inc['alphas_R2'], inc['ws_R2'],
                                     left=0, right=0)
        w_inc_R2 = np.maximum(w_inc_R2, w_inc_R2_from_R2)

    # Entrant density in Region II
    wE_R2 = np.where(entrant_serves_R2,
                     np.maximum(ws_R2_total - w_inc_R2, 0),
                     0.0)

    # Cumulative entrant capital in Region II
    WE_cumsum_R2 = np.zeros(n_R2)
    WE_cumsum_R2[1:] = np.cumsum(wE_R2[:-1]) * da_R2
    WE_cumsum_R2 += WE_cumsum_R1[-1]

    # Update remaining good borrowers through Region II
    G_remaining_R2 = G_end_R1
    GLOs_R2 = []
    for i in range(n_R2):
        if i > 0:
            G_remaining_R2 -= (1 - beta) * g_tilde_R2[i] * da_R2
        GLOs_R2.append(max(G_remaining_R2, 0))
    GLOs_R2 = np.array(GLOs_R2)

    # =========================================================================
    # Step 6: Non-selective region
    # =========================================================================
    # Check if entrant non-selectives can enter
    omega_g_a2E = beta + alpha2_E * (1 - beta)
    goodleft_a2E, _ = quad(gpriorfun, omega_g_a2E, 1)
    total_left = goodleft_a2E + badleftover_E
    gamma_NS_E = goodleft_a2E / total_left if total_left > 1e-12 else 0.0

    # Rate for entrant NS: gamma_NS * (1 + r) = 1 + Pi_E
    r_NS_E = (1 + ep.Pi_E) / gamma_NS_E - 1 if gamma_NS_E > 1e-12 else 1e10

    # Rate for incumbent NS at alpha2_E
    r_NS_inc_a2E = cfun(alpha2_E) + params.Pi

    # Non-selective rate is the minimum
    r_NS_combined = min(r_NS_E, r_NS_inc_a2E)

    # Capital needed for NS
    WNS_E = dfun(r_NS_combined) * total_left if total_left > 1e-12 else 0.0

    # =========================================================================
    # Step 7: Total capital
    # =========================================================================
    W_total_entrant = WE_cumsum_R2[-1] + max(0, WNS_E - inc.get('WNS', 0))

    # =========================================================================
    # Assemble results
    # =========================================================================
    return {
        # Incumbent reference
        'incumbent': inc,
        # Entry thresholds
        'alpha0_E': alpha0_E, 'alpha1_E': alpha1_E, 'alpha2_E': alpha2_E,
        'rp_E': rp_E, 'r_NS_E': r_NS_E, 'r_NS_combined': r_NS_combined,
        'entry_in_R1': True,
        # Region I (entrant)
        'alphas_R1': alphas_R1,
        'wE_R1': wE_R1,
        'w_total_R1': w_total_E,
        'w_inc_R1': w_inc_interp,
        'TE_R1': TE_vals,
        'GE_R1': GE_vals,
        'BE_R1': BE_vals,
        'EE_R1': EE_vals,
        'gamma_E_R1': gamma_E_vals,
        'WE_cumsum_R1': WE_cumsum_R1,
        'G_leftover_R1': G_leftover_R1,
        'B_leftover_R1': B_leftover_R1,
        'da_R1': da_R1,
        # Region II
        'alphas_R2': alphas_R2,
        'wE_R2': wE_R2,
        'r_CIM': r_CIM,
        'entrant_serves_R2': entrant_serves_R2,
        'WE_cumsum_R2': WE_cumsum_R2,
        'GLOs_R2': GLOs_R2,
        'da_R2': da_R2,
        # Non-selective
        'WNS_E': WNS_E,
        'gamma_NS_E': gamma_NS_E,
        'badleftover_E': badleftover_E,
        'goodleft_a2E': goodleft_a2E,
        # Totals
        'W_total_entrant': W_total_entrant,
    }


def _find_alpha1_CIM(inc, rp_E, beta):
    """Find the CIM boundary: smallest alpha where incumbent capital suffices
    at rate >= rp_E for all alpha' in [alpha, alpha1]."""
    alphas = inc['alphas_R1']
    ws = inc['ws_R1']
    alpha1 = inc['alpha1']

    # For each alpha, check if incumbent w(alpha) can sustain good-only
    # lending at rate >= rp_E.  In Region II (good-only), the required
    # density is D(r) * g_tilde * (1-beta).  The CIM rate hat_r solves
    # w(alpha) = D(hat_r) * g_tilde * (1-beta), so hat_r = D^{-1}(w / (g*(1-beta))).
    # We need hat_r >= rp_E.

    # Start from alpha1 and work backwards
    # In Region II of incumbent, w is already set for CIM lending
    if 'alphas_R2' in inc and len(inc['alphas_R2']) > 1:
        # Incumbent Region II always has CIM rates >= rp_inc > rp_E
        # So alpha1_CIM <= alpha1_inc
        return inc['alpha1']

    return inc['alpha1']


def _find_alpha2_E(alpha1_E, beta, badleftover, ep, params):
    """Find alpha2^E: smallest alpha > alpha1_E where non-selective lenders
    (entrant or incumbent) can break even at the CIM rate.

    The condition is: gamma_NS(alpha) * (1 + r_CIM(alpha)) >= 1 + Pi_E,
    where r_CIM = min(c^E(alpha) + Pi_E, c(alpha) + Pi) and
    gamma_NS = goodleft / (goodleft + badleftover).
    """
    def NS_signed(al):
        omega_g = beta + al * (1 - beta)
        goodleft, _ = quad(gpriorfun, omega_g, 1)
        total = goodleft + badleftover
        if total < 1e-12:
            return -(1 + ep.Pi_E)
        gamma_NS = goodleft / total
        r_ent = cfun_E(al, ep) + ep.Pi_E
        r_inc = cfun(al) + params.Pi
        r_CIM = min(r_ent, r_inc)
        return gamma_NS * (1 + r_CIM) - (1 + ep.Pi_E)

    als = np.linspace(alpha1_E + 1e-4, 0.999, 500)
    vals = np.array([NS_signed(a) for a in als])
    for i in range(len(vals) - 1):
        if vals[i] * vals[i + 1] < 0:
            return brentq(NS_signed, als[i], als[i + 1])
    # If always positive, NS entry viable everywhere beyond alpha1_E
    if vals[-1] > 0:
        return alpha1_E + 1e-4
    return None


def _solve_entry_no_R1(params, ep, inc):
    """Handle the case where entrants don't enter Region I."""
    beta, BperG = params.beta, params.BperG
    alpha1_E = inc['alpha1']

    badleftover_E = inc['badleftover']
    G_end_R1 = inc['G_end_R1']
    B_end_R1 = inc['B_end_R1']

    # Region II: check if entrants have comparative advantage
    alpha2_E = _find_alpha2_E(alpha1_E, beta, badleftover_E, ep, params)
    if alpha2_E is None:
        alpha2_E = 1.0

    n_R2 = max(int((alpha2_E - alpha1_E) * 200), 10)
    alphas_R2 = np.linspace(alpha1_E, alpha2_E, n_R2)
    da_R2 = alphas_R2[1] - alphas_R2[0] if n_R2 > 1 else 0.01

    omega_g_R2 = beta + alphas_R2 * (1 - beta)
    g_tilde_R2 = np.array([gpriorfun(og) for og in omega_g_R2])

    r_CIM_inc = np.array([cfun(a) + params.Pi for a in alphas_R2])
    r_CIM_ent = np.array([cfun_E(a, ep) + ep.Pi_E for a in alphas_R2])
    r_CIM = np.minimum(r_CIM_inc, r_CIM_ent)
    entrant_serves_R2 = r_CIM_ent <= r_CIM_inc

    ws_R2_total = np.array([dfun(r_CIM[i]) * (1 - beta) * g_tilde_R2[i]
                            for i in range(n_R2)])
    w_inc_R2 = np.interp(alphas_R2, inc['alphas_R2'], inc['ws_R2'],
                         left=0, right=0)
    wE_R2 = np.where(entrant_serves_R2,
                     np.maximum(ws_R2_total - w_inc_R2, 0), 0.0)

    WE_cumsum_R2 = np.zeros(n_R2)
    WE_cumsum_R2[1:] = np.cumsum(wE_R2[:-1]) * da_R2

    # NS
    omega_g_a2E = beta + alpha2_E * (1 - beta)
    goodleft_a2E, _ = quad(gpriorfun, omega_g_a2E, 1)
    total_left = goodleft_a2E + badleftover_E
    gamma_NS_E = goodleft_a2E / total_left if total_left > 1e-12 else 0.0
    r_NS_E = (1 + ep.Pi_E) / gamma_NS_E - 1 if gamma_NS_E > 1e-12 else 1e10
    r_NS_inc = cfun(alpha2_E) + params.Pi
    r_NS_combined = min(r_NS_E, r_NS_inc)
    WNS_E = dfun(r_NS_combined) * total_left if total_left > 1e-12 else 0.0

    return {
        'incumbent': inc,
        'alpha0_E': inc['alpha0'], 'alpha1_E': alpha1_E, 'alpha2_E': alpha2_E,
        'rp_E': inc['rp'], 'r_NS_E': r_NS_E, 'r_NS_combined': r_NS_combined,
        'entry_in_R1': False,
        'alphas_R1': np.array([]), 'wE_R1': np.array([]),
        'w_total_R1': np.array([]), 'w_inc_R1': np.array([]),
        'TE_R1': np.array([]), 'GE_R1': np.array([]),
        'BE_R1': np.array([]), 'EE_R1': np.array([]),
        'gamma_E_R1': np.array([]),
        'WE_cumsum_R1': np.array([0.0]),
        'G_leftover_R1': np.array([G_end_R1]),
        'B_leftover_R1': np.array([B_end_R1]),
        'da_R1': 0.0,
        'alphas_R2': alphas_R2,
        'wE_R2': wE_R2, 'r_CIM': r_CIM,
        'entrant_serves_R2': entrant_serves_R2,
        'WE_cumsum_R2': WE_cumsum_R2,
        'GLOs_R2': np.array([G_end_R1] * n_R2),
        'da_R2': da_R2,
        'WNS_E': WNS_E, 'gamma_NS_E': gamma_NS_E,
        'badleftover_E': badleftover_E, 'goodleft_a2E': goodleft_a2E,
        'W_total_entrant': WE_cumsum_R2[-1],
    }


# =============================================================================
# PLOTTING
# =============================================================================

def plot_entry(params, ep, entry, save_path=None):
    """Plot the entry equilibrium against the incumbent."""
    inc = entry['incumbent']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    alpha0_inc = inc['alpha0']
    alpha1_inc = inc['alpha1']
    alpha2_inc = inc['alpha2']
    rp_inc = inc['rp']

    # =========================================================================
    # Panel 1: Interest Rate Schedule
    # =========================================================================
    ax = axes[0, 0]
    # Incumbent
    alpha_r1 = np.linspace(alpha0_inc, alpha1_inc, 100)
    ax.plot(alpha_r1, rp_inc * np.ones_like(alpha_r1), 'b-', lw=2, label='Incumbent')
    alpha_r2 = np.linspace(alpha1_inc, alpha2_inc, 100)
    ax.plot(alpha_r2, cfun(alpha_r2) + params.Pi, 'b-', lw=2)
    ax.plot(0, inc['r_NS'], 'bo', markersize=8)

    # Entry
    if entry['entry_in_R1']:
        alpha_e1 = np.linspace(entry['alpha0_E'], entry['alpha1_E'], 100)
        ax.plot(alpha_e1, entry['rp_E'] * np.ones_like(alpha_e1),
                'r-', lw=2, label=f'Entry (rp_E={entry["rp_E"]:.3f})')

    # Entry CIM
    ax.plot(entry['alphas_R2'], entry['r_CIM'], 'r--', lw=1.5)

    # Entrant cost schedule
    alpha_cost = np.linspace(0.01, 0.99, 100)
    ax.plot(alpha_cost, [cfun_E(a, ep) + ep.Pi_E for a in alpha_cost],
            'r:', lw=1, alpha=0.5, label=f'$K^E(\\alpha)$ = {ep.CE_C}$\\alpha^{{{ep.CE_P}}}$ + {ep.Pi_E}')

    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$r(\alpha)$')
    ax.set_title('Interest Rate Schedule')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])

    # =========================================================================
    # Panel 2: Pool Quality gamma
    # =========================================================================
    ax = axes[0, 1]
    # Incumbent
    ax.plot(inc['alphas_R1'], inc['gammas_R1'], 'b-', lw=2, label='Incumbent')
    ax.plot(inc['alphas_R2'], np.ones(len(inc['alphas_R2'])), 'b-', lw=2)
    # Entry
    if entry['entry_in_R1'] and len(entry['gamma_E_R1']) > 0:
        ax.plot(entry['alphas_R1'], entry['gamma_E_R1'], 'r-', lw=2,
                label='Entry combined')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\gamma(\alpha)$')
    ax.set_title('Pool Quality')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # =========================================================================
    # Panel 3: Lender Density w(alpha)
    # =========================================================================
    ax = axes[0, 2]
    # Incumbent
    ax.plot(inc['alphas_R1'], inc['ws_R1'], 'b-', lw=2, label='Incumbent w')
    if len(inc['alphas_R2']) > 1:
        ax.plot(inc['alphas_R2'], inc['ws_R2'], 'b-', lw=2)

    # Entry: total density and entrant density
    if entry['entry_in_R1'] and len(entry['w_total_R1']) > 0:
        ax.plot(entry['alphas_R1'], entry['w_total_R1'], 'g-', lw=1.5,
                label='Combined w (R1)')
        ax.plot(entry['alphas_R1'], entry['wE_R1'], 'r-', lw=2,
                label='Entrant w (R1)')

    if len(entry['wE_R2']) > 0:
        ax.plot(entry['alphas_R2'], entry['wE_R2'], 'r--', lw=1.5,
                label='Entrant w (R2)')

    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$w(\alpha)$')
    ax.set_title('Lender Density')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # =========================================================================
    # Panel 4: T(alpha) — acceptance pool
    # =========================================================================
    ax = axes[1, 0]
    if 'T_R1' in inc:
        ax.plot(inc['alphas_R1'], inc['T_R1'], 'b-', lw=2, label='Incumbent T')
    if entry['entry_in_R1'] and len(entry['TE_R1']) > 0:
        ax.plot(entry['alphas_R1'], entry['TE_R1'], 'r-', lw=2, label='Entry T')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$T(\alpha)$')
    ax.set_title('Acceptance Pool Mass')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # =========================================================================
    # Panel 5: Cumulative Entrant Capital
    # =========================================================================
    ax = axes[1, 1]
    if entry['entry_in_R1'] and len(entry['WE_cumsum_R1']) > 0:
        ax.plot(entry['alphas_R1'], entry['WE_cumsum_R1'], 'r-', lw=2,
                label='R1 entrant')
    if len(entry['WE_cumsum_R2']) > 0:
        ax.plot(entry['alphas_R2'], entry['WE_cumsum_R2'], 'r--', lw=2,
                label='R1+R2 entrant')
    ax.axhline(entry.get('W_total_entrant', 0), color='r', ls=':', alpha=0.5,
               label=f'Total entrant W = {entry.get("W_total_entrant", 0):.3f}')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$W^E(\alpha)$')
    ax.set_title('Cumulative Entrant Capital')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # =========================================================================
    # Panel 6: Remaining Borrowers
    # =========================================================================
    ax = axes[1, 2]
    # Incumbent
    ax.plot(inc['alphas_R1'], inc['GLOs_R1'], 'b-', lw=2, label='Inc. G remaining')
    ax.plot(inc['alphas_R1'], inc['BLOs_R1'], 'b--', lw=2, label='Inc. B remaining')
    # Entry
    if entry['entry_in_R1'] and len(entry['G_leftover_R1']) > 0:
        ax.plot(entry['alphas_R1'], entry['G_leftover_R1'], 'r-', lw=1.5,
                label='Entry G remaining')
        ax.plot(entry['alphas_R1'], entry['B_leftover_R1'], 'r--', lw=1.5,
                label='Entry B remaining')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Mass')
    ax.set_title('Remaining Borrowers')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f'Entry Equilibrium: $\\Pi$={params.Pi}, $\\Pi^E$={ep.Pi_E}, '
        f'$c^E$={ep.CE_C}$\\alpha^{{{ep.CE_P}}}$, '
        f'$\\beta$={params.beta}, B/G={params.BperG}',
        fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved to {save_path}")

    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the entry equilibrium computation."""
    # Incumbent parameters
    params = Parameters()

    # Entrant parameters (easily changeable)
    ep = EntrantParams(
        Pi_E=0.03,      # Lower cost of capital than incumbent (Pi=0.05)
        CE_C=0.8,       # Cost coefficient
        CE_P=2.0,       # Power function exponent
    )

    print("=" * 70)
    print("ENTRY EQUILIBRIUM: T(alpha) CONSTRUCTION")
    print("=" * 70)
    print(f"Incumbent: Pi={params.Pi}, beta={params.beta}, BperG={params.BperG}")
    print(f"Entrant:   Pi_E={ep.Pi_E}, c^E(alpha)={ep.CE_C}*alpha^{ep.CE_P}")
    print("=" * 70)

    # Solve incumbent
    print("\n--- Incumbent Equilibrium ---")
    inc = solve_nested_analytical(params)
    print(f"  alpha0 = {inc['alpha0']:.4f}")
    print(f"  alpha1 = {inc['alpha1']:.4f}")
    print(f"  alpha2 = {inc['alpha2']:.4f}")
    print(f"  rp     = {inc['rp']:.4f}")
    print(f"  r_NS   = {inc['r_NS']:.4f}")
    print(f"  WNS    = {inc['WNS']:.4f}")
    W_inc_total = inc['W_R2_cumsum'][-1] + inc['WNS']
    print(f"  Total W = {W_inc_total:.4f}")

    # Solve entry
    print("\n--- Entry Equilibrium ---")
    entry = solve_entry(params, ep, incumbent=inc)
    print(f"  Entry in R1:  {entry['entry_in_R1']}")
    print(f"  alpha0_E = {entry['alpha0_E']:.4f}")
    print(f"  alpha1_E = {entry['alpha1_E']:.4f}")
    print(f"  alpha2_E = {entry['alpha2_E']:.4f}")
    print(f"  rp_E     = {entry['rp_E']:.4f}")
    print(f"  r_NS_E   = {entry['r_NS_E']:.4f}")
    if entry['entry_in_R1']:
        print(f"  WE R1    = {entry['WE_cumsum_R1'][-1]:.4f}")
    print(f"  WE total = {entry['W_total_entrant']:.4f}")

    # Diagnostics
    if entry['entry_in_R1'] and len(entry['TE_R1']) > 0:
        print("\n--- Diagnostics ---")
        print(f"  T^E(alpha0_E)  = {entry['TE_R1'][0]:.6f}")
        print(f"  T^inc(alpha0)  = {inc['T_R1'][0]:.6f}")
        print(f"  E^E(alpha0_E)  = {entry['EE_R1'][0]:.6f} (should be ~1)")
        print(f"  gamma^E(alpha0) = {entry['gamma_E_R1'][0]:.6f}")
        print(f"  gamma^inc(a0)  = {inc['gammas_R1'][0]:.6f}")

    # Plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, 'credit_model_entry.png')
    plot_entry(params, ep, entry, save_path=save_path)

    # =========================================================================
    # Experiment: sweep entrant Pi_E
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT: Sweep entrant Pi_E")
    print("=" * 70)
    print(f"  {'Pi_E':>6}  {'rp_E':>8}  {'a0_E':>8}  {'a1_E':>8}  "
          f"{'WE_R1':>8}  {'WE_tot':>8}  {'entry_R1':>8}")
    for Pi_E_test in [0.01, 0.02, 0.03, 0.04, 0.045, 0.048, 0.05, 0.06]:
        ep_test = EntrantParams(Pi_E=Pi_E_test, CE_C=ep.CE_C, CE_P=ep.CE_P)
        try:
            e = solve_entry(params, ep_test, incumbent=inc)
            WE_R1 = e['WE_cumsum_R1'][-1] if len(e['WE_cumsum_R1']) > 0 else 0
            print(f"  {Pi_E_test:>6.3f}  {e['rp_E']:>8.4f}  {e['alpha0_E']:>8.4f}  "
                  f"{e['alpha1_E']:>8.4f}  {WE_R1:>8.4f}  "
                  f"{e['W_total_entrant']:>8.4f}  {e['entry_in_R1']!s:>8}")
        except Exception as ex:
            print(f"  {Pi_E_test:>6.3f}  FAILED: {ex}")

    # =========================================================================
    # Experiment: sweep entrant cost power
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT: Sweep entrant cost exponent CE_P")
    print("=" * 70)
    print(f"  {'CE_P':>6}  {'rp_E':>8}  {'a0_E':>8}  {'a1_E':>8}  "
          f"{'WE_R1':>8}  {'WE_tot':>8}  {'entry_R1':>8}")
    for CE_P_test in [1.5, 2.0, 2.5, 3.0, 4.0]:
        ep_test = EntrantParams(Pi_E=ep.Pi_E, CE_C=ep.CE_C, CE_P=CE_P_test)
        try:
            e = solve_entry(params, ep_test, incumbent=inc)
            WE_R1 = e['WE_cumsum_R1'][-1] if len(e['WE_cumsum_R1']) > 0 else 0
            print(f"  {CE_P_test:>6.2f}  {e['rp_E']:>8.4f}  {e['alpha0_E']:>8.4f}  "
                  f"{e['alpha1_E']:>8.4f}  {WE_R1:>8.4f}  "
                  f"{e['W_total_entrant']:>8.4f}  {e['entry_in_R1']!s:>8}")
        except Exception as ex:
            print(f"  {CE_P_test:>6.2f}  FAILED: {ex}")

    plt.show()
    return params, ep, inc, entry


if __name__ == "__main__":
    params, ep, inc, entry = main()
