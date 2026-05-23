"""
Python translation of mainE.m and all its dependencies.

Includes:
- All helper functions (gpriorfun, bpriorfun, dfun, dfuninv, cfun, cfunE, etc.)
- Baseline computation (main.m) → run_baseline()
- Entry computation (mainE.m) → run_mainE()

Usage:
    python mainE_python.py

Dependencies: numpy, scipy, matplotlib
"""

import os
import numpy as np
from scipy.optimize import minimize_scalar, fsolve, minimize, brentq
from scipy.integrate import quad
import matplotlib.pyplot as plt
from types import SimpleNamespace
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# Parameterization configs — one per panel of Figures 7-9 in the paper.
# Switch ACTIVE_CONFIG (or pass --config <name>) to change parameters and cost.
#
# Schema per config:
#   description : str
#   Pi, beta, BperG : incumbent baseline scalars
#   cfun            : incumbent cost function (callable α -> cost)
#   has_entry       : bool — if False, skip the entry-equilibrium solve
#   PiE             : entry capital cost (if has_entry)
#   cfunE_kind      : one of
#                       'selection_preserving' — SP form (γ-dependent), per Def. SPK
#                       'polynomial'           — closed-form polynomial
#                       'cost_reduction_top'   — C^E(α) = C(α) for α≤α̂, λ·C(α) for α>α̂
#                       'cost_reduction_low'   — C^E(α) = λ·C(α) for α<α̂, C(α) for α≥α̂
#   cfunE_params    : dict — kind-specific parameters
#                       SP:   {'kappa': 1.1, 'use_smoothing': True,
#                              'use_legacy_addons': False, 'kappa1_distortion': 0.0}
#                       poly: {'coeffs': lambda α -> cost}
#                       cost_reduction_*: {'alpha_hat': 0.3, 'lambda': 0.7}
# =============================================================================
_DEFAULT_INC_BASELINE = dict(
    Pi=0.2, beta=0.5, BperG=1.0,
    cfun=lambda alpha: 9.0 * alpha**2 + 0.2 * alpha,
)

PARAM_CONFIGS = {
    'FIG7_incumbent': {
        'description': 'Figure 7 / fig:SPT - incumbent only (green dashed curve).',
        **_DEFAULT_INC_BASELINE,
        'has_entry': False,
    },
    'FIG7_SP_highPiE': {
        'description': 'Figure 7 / fig:SPT - SP entry with PiE > PiE_bar (blue, no Region IIb).',
        **_DEFAULT_INC_BASELINE,
        'has_entry': True,
        'PiE': 0.15,
        'cfunE_kind': 'selection_preserving',
        'cfunE_params': {'kappa': 1.1, 'use_smoothing': True,
                         'use_legacy_addons': False, 'kappa1_distortion': 0.0},
    },
    'FIG7_SP_lowPiE': {
        'description': 'Figure 7 / fig:SPT - SP entry with PiE < PiE_bar (red, Region IIb present).',
        **_DEFAULT_INC_BASELINE,
        'has_entry': True,
        'PiE': 0.05,
        'cfunE_kind': 'selection_preserving',
        'cfunE_params': {'kappa': 1.1, 'use_smoothing': True,
                         'use_legacy_addons': False, 'kappa1_distortion': 0.0},
    },
    'FIG8_bigdata': {
        'description': 'Figure 8 / fig:AIintermediateInnovation - big data: C^E<C for alpha>alpha_hat.',
        # Setup chosen so K^E starts above K (Π^E > Π = 0.2), is flatter at high
        # α (λ < 1 multiplicative reduction), and crosses K only at a relatively
        # high α (around α ≈ 0.5, within the CIM region, before NS).
        # alpha_hat = 0.55 puts the transition near the NS boundary; the crossover
        # point K^E = K_inc then falls around α ≈ 0.5.
        **_DEFAULT_INC_BASELINE,
        'has_entry': True,
        'PiE': 0.5,
        'cfunE_kind': 'cost_reduction_top',
        'cfunE_params': {'alpha_hat': 0.55, 'lambda': 0.5, 'transition_width': 0.08},
        # transition_width = 0.08 keeps K^E monotone given C(α)=9α²+0.2α.
        # The bound is s ≥ (1-λ)·C(α_hat)/[(2+2λ)·C'(α_hat)] ≈ 0.047.
    },
    'FIG9_OB_limited': {
        'description': 'Figure 9a / fig:OB left - Open Banking, limited adoption (cost advantage at low alpha).',
        # Legacy SP + add-ons cost form (selection_preserving + legacy_addons).
        # K^E = γ(α)·(1 + D⁻¹((1/κ)·D((Π+1+C)/γ - 1))) - 1 - Π^E
        #         + κ₁·α·(α-α̅)                                  (distortion)
        #         + hyperbolic left addon (capped at 0.1·addon_cap, kink at α=α₁)
        #         + parallel right piece (K^E rises in lockstep with K_inc for α>α₁)
        #
        # Pi/Pi^E chosen so that Pi^E = K^E(0) (i.e. zero entrant rents at α=0
        # in the limit of the marginal lender breaking even).  K^E(0) for the SP
        # form depends only on Pi and γ(0), not on Pi^E, so this is a one-shot
        # set rather than a fixed point.  Pi=0.235 lifts the rate scale up by
        # ~16% but preserves α₀, α₁, α₀^E and Region IIb width (0.086) almost
        # exactly; r_NS drop shrinks from 0.36 to 0.26 (still clearly positive
        # spillover).
        'Pi': 0.235,
        'beta': 0.5,
        'BperG': 1.0,
        'cfun': lambda alpha: 9.0 * alpha**2 + 0.2 * alpha,
        'has_entry': True,
        'PiE': 0.168,
        'cfunE_kind': 'selection_preserving',
        'cfunE_params': {'kappa': 1.1, 'use_smoothing': True,
                         'use_legacy_addons': True, 'kappa1_distortion': -4.0,
                         # Lowered from legacy 100 → kink slides from α≈0.336 to
                         # α≈0.269 and K^E plateau drops from +10 to +1.
                         'addon_cap': 10.0},
    },
    'FIG9_OB_broad': {
        'description': 'Figure 9b / fig:OB right - Open Banking, broad adoption (cost advantage band at intermediate alpha).',
        # Target equilibrium branch: rprime is min AND rprime > K(α₂)
        # ("rNS goes up, no NS entry" branch from prop:EntEqHeterogC proof).
        # Uses cost_dip_multiplicative: C^E(α) = C(α)·[1+δ_baseline−δ_dip·bump(α)].
        # The multiplicative form ensures K^E(0) = Π^E automatically (since C(0)=0),
        # so the constraint K^E(0) = Π^E is a *property of the form* rather than
        # a tuning target that conflicts with the rprime branch.
        **_DEFAULT_INC_BASELINE,
        'has_entry': True,
        'PiE': 0.2,
        'cfunE_kind': 'cost_dip_multiplicative',
        'cfunE_params': {
            # α_c=0.25, σ=0.05: cream-skim band centered just below α₁=0.346.
            # δ_dip=0.40, δ_baseline=0.10:
            #   multiplier at α_c = 0.70 → C^E(α_c) = 0.70·C(α_c)
            #   multiplier at α₁  ≈ 1.04 → K^E > K_inc on (α₁,α₂) ⇒ no CIM entry
            #   multiplier at α=0 → C^E(0) = 0 ⇒ K^E(0) = Π^E exactly
            'alpha_center': 0.25,
            'sigma': 0.05,
            'delta_dip': 0.40,
            'delta_baseline': 0.10,
        },
    },
}

# Default config for direct execution; can be overridden by --config CLI flag.
ACTIVE_CONFIG = 'FIG9_OB_limited'


def _scalar(x):
    """Convert any numpy scalar/0-d/1-element array to a Python float."""
    return np.asarray(x, dtype=float).item()


def _precompute_cum_states(rate, matlab_gammaupdate2=False):
    """Pre-compute cumulative g/b distributions after applying each incumbent.

    Returns (cum_gd, cum_bd) where cum_gd[k] is the state after applying
    incumbents g.almass[0]..g.almass[k-1].  cum_gd[0] is the initial prior.

    matlab_gammaupdate2: if True, uses the MATLAB gammaupdate2.m formula
        (denom / dfun_r) instead of the inline formula (denom * dfun_r).
        These differ by r^2.  gammaupdate2.m uses the former; gam0E.m and
        main.m inline use the latter.
    """
    n_inc = len(g.almass)
    n_om = len(g.omvec)
    dfun_r = 1.0 / rate          # dfun(r) = 1/r
    beta1 = g.beta
    beta2 = 1 - beta1

    cum_gd = np.empty((n_inc + 1, n_om))
    cum_bd = np.empty((n_inc + 1, n_om))

    cum_gd[0] = gpriorfun(g.omvec)
    cum_bd[0] = bpriorfun(g.omvec)

    for k in range(n_inc):
        gd = cum_gd[k]
        bd = cum_bd[k]

        mask_g = g.omvec <= (beta1 + g.almass[k] * beta2)
        mask_b = g.omvec >= (beta2 + g.almass[k] * beta1)

        raw_denom = (np.sum(g.delom * gd * mask_g) +
                     np.sum(g.delom * bd * mask_b))

        if matlab_gammaupdate2:
            denom = raw_denom / dfun_r    # = raw_denom * rate  (MATLAB gammaupdate2.m)
        else:
            denom = raw_denom * dfun_r    # = raw_denom / rate  (gam0E.m / main.m inline)

        if denom > 0:
            factor = g.wmass[k] / denom
            cum_gd[k + 1] = gd * (1 - mask_g * factor)
            cum_bd[k + 1] = bd * (1 - mask_b * factor)
        else:
            cum_gd[k + 1] = gd
            cum_bd[k + 1] = bd

    return cum_gd, cum_bd


# =============================================================================
# Global state (equivalent to MATLAB global variables)
# =============================================================================
g = SimpleNamespace()


# =============================================================================
# Primitive functions
# =============================================================================

def gpriorfun(om):
    """Prior distribution for good borrowers (uniform). Matches gpriorfun.m"""
    return np.ones_like(np.asarray(om, dtype=float))


def gpriorfun_scalar(om):
    """Scalar version of gpriorfun for use with scipy.integrate.quad."""
    return 1.0


def bpriorfun(om):
    """Prior distribution for bad borrowers. Matches bpriorfun.m"""
    return g.BperG * np.ones_like(np.asarray(om, dtype=float))


def bpriorfun_scalar(om):
    """Scalar version of bpriorfun for use with scipy.integrate.quad."""
    return g.BperG


def dfun(r):
    """Demand function. Matches dfun.m"""
    return 1.0 / np.asarray(r, dtype=float)


def dfuninv(d):
    """Inverse demand function. Matches dfuninv.m"""
    return 1.0 / np.asarray(d, dtype=float)


def cfun(alpha):
    """Cost function for incumbents. Dispatches to ACTIVE_CONFIG."""
    alpha = np.asarray(alpha, dtype=float)
    result = PARAM_CONFIGS[ACTIVE_CONFIG]['cfun'](alpha)
    result = np.asarray(result, dtype=float)
    return result.item() if np.size(result) == 1 else result


def _resolve_cfunE_mode(cfg):
    """Map new cfunE_kind to the legacy cfunE_mode used internally.

    This is a temporary bridge while we keep the existing 'smooth'/'complex'
    code paths working. Once the cost_reduction_* and clean selection_preserving
    paths are validated end-to-end, the legacy mode strings can be retired.
    """
    kind = cfg.get('cfunE_kind')
    params = cfg.get('cfunE_params', {})

    if kind is None:
        # Legacy-style flat config; fall back to whatever cfunE_mode says.
        return cfg.get('cfunE_mode', 'complex'), params

    if kind == 'selection_preserving':
        # Routes through the SP form (γ-dependent). Smoothing optional.
        return ('smooth' if params.get('use_smoothing', True) else 'complex'), params

    if kind == 'polynomial':
        return 'simple', params

    if kind in ('cost_reduction_top', 'cost_reduction_low',
                'cost_offset_smooth', 'cost_dip_gaussian',
                'cost_dip_multiplicative', 'cost_exp_decay'):
        # New kinds — handled directly in cfunE, no legacy mode.
        return kind, params

    raise ValueError(f"Unknown cfunE_kind: {kind!r}")


def cfun_prime(alpha, eps=1e-6):
    """First derivative of cost function (numerical, central differences)."""
    return (cfun(alpha + eps) - cfun(alpha - eps)) / (2 * eps)


def cfun_prime2(alpha, eps=1e-5):
    """Second derivative of cost function (numerical, central differences)."""
    return (cfun(alpha + eps) - 2 * cfun(alpha) + cfun(alpha - eps)) / (eps ** 2)


# =============================================================================
# Baseline functions (used by main.m)
# =============================================================================

def gam0(alpha):
    """Initial gamma (fraction of good borrowers). Matches gam0.m"""
    alpha = _scalar(alpha)
    num = quad(gpriorfun_scalar, 0, g.beta + alpha * (1 - g.beta))[0]
    den_b = quad(bpriorfun_scalar, 1 - g.beta + alpha * g.beta, 1)[0]
    return num / (num + den_b)


def gamfun(w, al, alp):
    """
    Gamma function for incumbents in baseline pooling region. Matches gamfun.m
    Uses globals: gfunprev, bfunprev, rp, beta, delom, omvec
    """
    mask_g = g.omvec <= (g.beta + alp * (1 - g.beta))
    mask_b = g.omvec >= (1 - g.beta + alp * g.beta)
    denom = (np.sum(g.delom * g.gfunprev[mask_g]) +
             np.sum(g.delom * g.bfunprev[mask_b]))

    dfun_rp = _scalar(dfun(g.rp))
    gn = g.gfunprev - g.gfunprev * mask_g * w / denom / dfun_rp
    bn = g.bfunprev * (1 - mask_b * w / denom / dfun_rp)

    mask_g2 = g.omvec <= (g.beta + al * (1 - g.beta))
    mask_b2 = g.omvec >= (1 - g.beta + al * g.beta)

    num = np.sum(g.delom * gn[mask_g2])
    den = num + np.sum(g.delom * bn[mask_b2])
    return num / den


def profit(w, alpha, a):
    """Profit for incumbents. Matches profit.m"""
    return (1 + g.rp) * gamfun(w, alpha, a) - _scalar(cfun(alpha))


def gamcplx(w):
    """Complex gamma for incumbent optimization. Matches gamcplx.m
    Uses globals: Pi, alp, Delta, alpha1, rp"""
    res = minimize_scalar(
        lambda alpha: _scalar(cfun(alpha)) - (1 + g.rp) * gamfun(w, alpha, g.alp),
        bounds=(g.alp + g.Delta, g.alpha1), method='bounded'
    )
    alopt = res.x
    return (profit(w, alopt, g.alp) - 1 - g.Pi)**2


def NSfun(al):
    """Non-selective entry function for incumbents (squared residual). Matches NSfun.m"""
    al = _scalar(al)
    goodleftover = quad(gpriorfun_scalar, g.beta + al * (1 - g.beta), 1)[0]
    gammaNS = goodleftover / (goodleftover + g.badleftover)
    return (gammaNS * (1 + _scalar(cfun(al)) + g.Pi) - (1 + g.Pi))**2


def NSfun_signed(al):
    """Signed NS condition: gamma_NS*(1+K(alpha)) - (1+Pi). Zero at equilibrium."""
    al = _scalar(al)
    goodleftover = quad(gpriorfun_scalar, g.beta + al * (1 - g.beta), 1)[0]
    if goodleftover + g.badleftover < 1e-12:
        return -(1 + g.Pi)
    gammaNS = goodleftover / (goodleftover + g.badleftover)
    return gammaNS * (1 + _scalar(cfun(al)) + g.Pi) - (1 + g.Pi)


def find_alpha2_mainE(alpha1, n_scan=500):
    """Find smallest alpha2 > alpha1 satisfying the NS condition."""
    als = np.linspace(alpha1 + 1e-4, 0.999, n_scan)
    vals = np.array([NSfun_signed(a) for a in als])
    for i in range(len(vals) - 1):
        if vals[i] * vals[i + 1] < 0:
            return brentq(lambda a: NSfun_signed(a), als[i], als[i + 1])
    return None


def gammaNSfun(al):
    """Gamma NS for incumbents. Matches gammaNSfun.m"""
    al = np.atleast_1d(np.asarray(al, dtype=float))
    result = np.zeros(len(al))
    for i in range(len(al)):
        gl = quad(gpriorfun_scalar, g.beta + al[i] * (1 - g.beta), 1)[0]
        gammaNS_val = gl / (gl + g.badleftover)
        result[i] = gammaNS_val * (1 + _scalar(cfun(al[i])) + g.Pi) - (1 + g.Pi)
    return _scalar(result[0]) if len(result) == 1 else result


def wcim(al):
    """Wealth in the CIM region for incumbents. Matches wcim.m"""
    al = np.atleast_1d(np.asarray(al, dtype=float))
    if len(al) == 1:
        return _scalar(dfun(cfun(al[0]) + g.Pi)) * (1 - g.beta) * _scalar(gpriorfun(g.beta + al[0] * (1 - g.beta)))
    else:
        w = np.zeros(len(al))
        w[0] = _scalar(dfun(cfun(al[0]) + g.Pi)) * quad(
            gpriorfun_scalar, g.beta + al[0] * (1 - g.beta), g.beta + al[1] * (1 - g.beta))[0]
        for i in range(1, len(al)):
            w[i] = _scalar(dfun(cfun(al[i]) + g.Pi)) * quad(
                gpriorfun_scalar, g.beta + al[i-1] * (1 - g.beta), g.beta + al[i] * (1 - g.beta))[0]
        return w


def rfun(al, alpha0, alpha1, alpha2):
    """Interest rate function for incumbents. Matches rfun.m"""
    al = np.asarray(al, dtype=float)
    r1 = (((al >= alpha0) & (al < alpha1)).astype(float) * g.rp +
          ((al >= alpha1) & (al < alpha2)).astype(float) * (cfun(al) + g.Pi) +
          (al >= alpha2).astype(float) * _scalar(cfun(alpha2) + g.Pi))
    r1[r1 == 0] = np.nan
    return r1


# =============================================================================
# Entry functions (used by mainE.m)
# =============================================================================

def gammaupdate2(alvec, alstart, r):
    """
    Gamma update with incumbents. Matches gammaupdate2.m
    Uses pre-computed cache (g._cum_gd_rp) when rate == g.rp for O(1) lookup.
    Falls back to original loop otherwise.
    """
    alvec = np.atleast_1d(np.asarray(alvec, dtype=float))
    n_alvec = len(alvec)
    gamma = np.zeros(n_alvec)

    beta1 = g.beta
    beta2 = 1 - beta1

    # Fast path: use pre-computed cumulative states
    use_cache = (hasattr(g, '_cum_gd_rp') and
                 abs(r - g.rp) < 1e-12 and
                 abs(alstart - g.almass[0]) < 1e-12)

    if use_cache:
        for j in range(n_alvec):
            al = alvec[j]
            idx = np.searchsorted(g.almass, al, side='right')
            gd = g._cum_gd_rp[idx]
            bd = g._cum_bd_rp[idx]

            mask_gf = g.omvec <= (beta1 + al * beta2)
            mask_bf = g.omvec >= (beta2 + al * beta1)

            num = np.sum(g.delom * gd * mask_gf)
            den = num + np.sum(g.delom * bd * mask_bf)
            gamma[j] = num / den if den > 0 else 0.0
    else:
        # Original loop (fallback)
        g_init = gpriorfun(g.omvec)
        b_init = bpriorfun(g.omvec)
        dfun_r = _scalar(dfun(r))

        for j in range(n_alvec):
            al = alvec[j]
            gd = g_init.copy()
            bd = b_init.copy()

            valid = np.where((g.almass >= alstart) & (g.almass <= al))[0]

            for k in valid:
                alp_k = g.almass[k]
                wopt_k = g.wmass[k]

                threshold_g = beta1 + alp_k * beta2
                threshold_b = beta2 + alp_k * beta1

                mask_gk = g.omvec <= threshold_g
                mask_bk = g.omvec >= threshold_b

                denom_gk = np.sum(g.delom * gd * mask_gk)
                denom_bk = np.sum(g.delom * bd * mask_bk)
                total_denom = (denom_gk + denom_bk) / dfun_r  # matches MATLAB gammaupdate2.m

                if total_denom > 0:
                    update_factor = wopt_k / total_denom
                    gd = gd * (1 - mask_gk * update_factor)
                    bd = bd * (1 - mask_bk * update_factor)

            mask_gf = g.omvec <= (beta1 + al * beta2)
            mask_bf = g.omvec >= (beta2 + al * beta1)

            num = np.sum(g.delom * gd * mask_gf)
            den = num + np.sum(g.delom * bd * mask_bf)
            gamma[j] = num / den if den > 0 else 0.0

    return _scalar(gamma[0]) if n_alvec == 1 else gamma


def _build_cfunE_pchip():
    """Build a monotone PCHIP interpolant of the original (complex) cfunE.

    Evaluates the singular cfunE on a dense grid (values are finite due to the
    maxc cap).  The base_cost component depends on gammaupdate2 which has
    discrete omega-grid noise; this is smoothed with a Savitzky-Golay filter
    before combining with the smooth distortion and addon terms.
    The PCHIP interpolant is then monotone and smooth.
    Stored on g._cfunE_pchip for reuse.
    """
    from scipy.interpolate import PchipInterpolator
    from scipy.signal import savgol_filter

    # Evaluate original complex cfunE components on a dense grid
    # Avoid exactly a = maxa (use maxa - tiny offset instead)
    maxa = g.alpha1
    n_grid = 500
    grid_left = np.linspace(0.001, maxa - 0.0005, n_grid // 2)
    grid_right = np.linspace(maxa + 0.0005, 0.999, n_grid // 2)
    grid = np.concatenate([grid_left, [maxa - 1e-6], grid_right])
    grid.sort()

    cfg = PARAM_CONFIGS[ACTIVE_CONFIG]
    _, params = _resolve_cfunE_mode(cfg)
    kappa = params.get('kappa', 1.1)
    use_legacy_addons = params.get('use_legacy_addons', False)
    kappa1_dist = params.get('kappa1_distortion', 0.0)
    # spike_alpha = α at which the addon blows up.  Defaults to α₁ (legacy).
    spike_alpha = params.get('spike_alpha', g.alpha1)
    # addon_cap = the cap height (legacy maxc=100, → plateau at 0.1·100 = 10).
    # Lowering it: keeps the left hyperbolic 1/(s-α) anchored, but cuts it off
    # sooner (kink slides left) and lowers the right-side plateau (entrants no
    # longer locked out at high α).
    maxc = params.get('addon_cap', 100.0)

    alpha1 = g.alpha1
    alphabar = (g.alpha0 + g.alpha1) / 2.0

    # Pre-compute base_cost and distortion at α = spike_alpha for the
    # "parallel right-piece" extension.  For α > spike_alpha we freeze these
    # at their α=s values and add only cfun(α)-cfun(s), so K^E rises in
    # lockstep with K_inc on the right (parallel to the green curve).
    gu_s = _scalar(gammaupdate2(spike_alpha, g.alpha0, g.rp))
    inner_r_s = (g.Pi + 1 + _scalar(cfun(spike_alpha))) / gu_s - 1
    base_at_s = gu_s * (1 + _scalar(dfuninv(kappa * _scalar(dfun(inner_r_s))))) - (1 + g.PiE)
    distort_at_s = kappa1_dist * spike_alpha * (spike_alpha - alphabar)
    cfun_at_s = _scalar(cfun(spike_alpha))

    # Evaluate each component separately
    base_costs = np.zeros(len(grid))
    distortions = np.zeros(len(grid))
    addons = np.zeros(len(grid))

    for i, a in enumerate(grid):
        if use_legacy_addons and a > spike_alpha:
            # Parallel extension: freeze base+distortion at α=s, addon supplies
            # the cap height plus K_inc-parallel rise.
            base_costs[i] = base_at_s
            distortions[i] = distort_at_s
            addons[i] = 0.1 * maxc + (_scalar(cfun(a)) - cfun_at_s)
            continue

        gu = _scalar(gammaupdate2(a, g.alpha0, g.rp))
        inner_r = (g.Pi + 1 + _scalar(cfun(a))) / gu - 1
        base_costs[i] = gu * (1 + _scalar(dfuninv(kappa * _scalar(dfun(inner_r))))) - (1 + g.PiE)
        if use_legacy_addons:
            distortions[i] = kappa1_dist * a * (a - alphabar)
            if a < spike_alpha:
                addons[i] = 0.1 * min(0.1 * (1.0 / max(spike_alpha - a, 1e-6) - 1.0 / spike_alpha), maxc)
            else:  # a == spike_alpha
                addons[i] = 0.1 * maxc

    # Smooth base_cost: this is the noisy component (from discrete omega grid).
    # Savitzky-Golay: polynomial degree 3, window ~5% of grid points.
    win = min(len(grid) // 10, 51)
    if win % 2 == 0:
        win += 1  # must be odd
    base_costs_smooth = savgol_filter(base_costs, win, 3)

    raw_vals = base_costs + distortions + addons
    vals = base_costs_smooth + distortions + addons

    max_smooth_err = np.max(np.abs(base_costs - base_costs_smooth))
    print(f"  Smoothed base_cost: Savitzky-Golay window={win}, "
          f"max |raw - smooth| = {max_smooth_err:.6f}")

    g._cfunE_pchip = PchipInterpolator(grid, vals)
    g._cfunE_pchip_grid = grid
    g._cfunE_pchip_vals = vals
    g._cfunE_pchip_vals_raw = raw_vals
    print(f"  Built PCHIP interpolant for cfunE: {len(grid)} points, "
          f"range [{vals.min():.3f}, {vals.max():.3f}]")


def cfunE(alpha):
    """
    Entry cost function for new entrants. Dispatches by cfunE_kind.

    Kinds (new schema):
      'selection_preserving' — SP form per Def. SPK: γ(α)(1+D⁻¹((1/κ)D((1+K)/γ-1))) - 1 - Π^E.
                               Params: kappa, use_smoothing, use_legacy_addons, kappa1_distortion.
      'polynomial'           — polynomial closed form. Params: coeffs (callable).
      'cost_reduction_top'   — C^E(α)=C(α) for α≤α̂, λ·C(α) for α>α̂. Params: alpha_hat, lambda.
      'cost_reduction_low'   — C^E(α)=λ·C(α) for α<α̂, C(α) for α≥α̂. Params: alpha_hat, lambda.
    """
    cfg = PARAM_CONFIGS[ACTIVE_CONFIG]
    mode, params = _resolve_cfunE_mode(cfg)

    # --- cost-reduction kinds: smooth sigmoid-windowed C^E(α) = w(α)·C(α).
    # The sigmoid window gives a C¹ transition through α̂, eliminating the kink
    # that bare piecewise functions would introduce.  Steepness s controls how
    # sharp the transition is (s small ≈ piecewise; s large ≈ broad ramp).
    # cfunE returns C^E(α); downstream uses K^E = Π^E + cfunE(α).
    if mode in ('cost_reduction_top', 'cost_reduction_low'):
        alpha_arr = np.atleast_1d(np.asarray(alpha, dtype=float))
        alpha_hat = params['alpha_hat']
        lam = params['lambda']
        s = params.get('transition_width', 0.03)
        c_inc = np.array([_scalar(cfun(a)) for a in alpha_arr])
        sig = 1.0 / (1.0 + np.exp(-(alpha_arr - alpha_hat) / s))
        if mode == 'cost_reduction_top':
            # w: 1 (low α) → λ (high α).
            w = 1.0 - (1.0 - lam) * sig
        else:
            # w: λ (low α) → 1 (high α).
            w = lam + (1.0 - lam) * sig
        ca = w * c_inc
        return _scalar(ca[0]) if len(alpha_arr) == 1 else ca

    # --- cost_offset_smooth: additive offset that is negative at low α and
    # positive at high α, transitioning through α̂.
    # K^E(α) - K(α) = (Π^E - Π) + offset(α), where
    #   offset(α) = -δ_low + (δ_low + δ_high)·sigmoid((α - α̂)/s)
    # At low α: offset = -δ_low (entrant cheaper).
    # At high α: offset = +δ_high (entrant more expensive).
    if mode == 'cost_offset_smooth':
        alpha_arr = np.atleast_1d(np.asarray(alpha, dtype=float))
        alpha_hat = params['alpha_hat']
        delta_low = params['delta_low']
        delta_high = params['delta_high']
        s = params.get('transition_width', 0.03)
        c_inc = np.array([_scalar(cfun(a)) for a in alpha_arr])
        sig = 1.0 / (1.0 + np.exp(-(alpha_arr - alpha_hat) / s))
        offset = -delta_low + (delta_low + delta_high) * sig
        ca = c_inc + offset
        return _scalar(ca[0]) if len(alpha_arr) == 1 else ca

    # --- cost_exp_decay: exponential-decay offset.
    # C^E(α) = C(α) + delta_baseline - delta_low · exp(-α / tau).
    # At α = 0: offset = delta_baseline - delta_low (deeply negative if
    #   delta_low > delta_baseline).
    # At α → ∞: offset → delta_baseline (mild positive cushion).
    # The offset is monotonically *increasing* in α, so K^E = K_inc + offset
    # is automatically monotone regardless of how steep the discount is.
    # This decouples the deep discount needed at α=0 (to drive NS entry and
    # widen Region IIb) from the slope constraint that bites cost_offset_smooth
    # at low α.
    if mode == 'cost_exp_decay':
        alpha_arr = np.atleast_1d(np.asarray(alpha, dtype=float))
        delta_low = params['delta_low']
        tau = params['tau']
        delta_baseline = params.get('delta_baseline', 0.0)
        c_inc = np.array([_scalar(cfun(a)) for a in alpha_arr])
        offset = delta_baseline - delta_low * np.exp(-alpha_arr / tau)
        ca = c_inc + offset
        return _scalar(ca[0]) if len(alpha_arr) == 1 else ca

    # --- cost_dip_gaussian: a Gaussian-shaped cost ADVANTAGE band centered at
    # alpha_center, with a baseline disadvantage delta_baseline elsewhere.
    # C^E(α) = C(α) + delta_baseline - delta_dip · exp(-(α-α_c)²/(2σ²))
    # At α far from α_c: K^E ≈ K + delta_baseline (entrant slightly more expensive).
    # At α = α_c: K^E ≈ K + delta_baseline - delta_dip (entrant cheaper if dip>baseline).
    # This gives a "bump-down" shape: cost advantage in an intermediate band, no
    # advantage at very low/high α.  Used for OB-broad to make entrants prefer
    # higher α (mid-pooling) than α₀, which produces extra cream-skimming.
    if mode == 'cost_dip_gaussian':
        alpha_arr = np.atleast_1d(np.asarray(alpha, dtype=float))
        alpha_center = params['alpha_center']
        sigma = params['sigma']
        delta_dip = params['delta_dip']
        delta_baseline = params.get('delta_baseline', 0.0)
        c_inc = np.array([_scalar(cfun(a)) for a in alpha_arr])
        bump = np.exp(-(alpha_arr - alpha_center)**2 / (2.0 * sigma**2))
        offset = delta_baseline - delta_dip * bump
        ca = c_inc + offset
        return _scalar(ca[0]) if len(alpha_arr) == 1 else ca

    # --- cost_dip_multiplicative: same Gaussian shape as cost_dip_gaussian
    # but applied multiplicatively, so C^E(0) = 0·(anything) = 0 = C(0) and
    # therefore K^E(0) = Π^E by construction.
    # C^E(α) = C(α) · [1 + δ_baseline − δ_dip · exp(−(α−α_c)²/(2σ²))]
    # At α=0 (far from α_c): C^E ≈ C·(1+δ_baseline), but C(0)=0 ⇒ C^E(0)=0.
    # At α=α_c: C^E = C·(1+δ_baseline−δ_dip) (cream-skim band).
    # At α far from α_c on the right: C^E → C·(1+δ_baseline) (no CIM entry).
    if mode == 'cost_dip_multiplicative':
        alpha_arr = np.atleast_1d(np.asarray(alpha, dtype=float))
        alpha_center = params['alpha_center']
        sigma = params['sigma']
        delta_dip = params['delta_dip']
        delta_baseline = params.get('delta_baseline', 0.0)
        c_inc = np.array([_scalar(cfun(a)) for a in alpha_arr])
        bump = np.exp(-(alpha_arr - alpha_center)**2 / (2.0 * sigma**2))
        multiplier = 1.0 + delta_baseline - delta_dip * bump
        ca = c_inc * multiplier
        return _scalar(ca[0]) if len(alpha_arr) == 1 else ca

    # --- polynomial closed form ---
    if mode == 'simple':
        alpha_arr = np.atleast_1d(np.asarray(alpha, dtype=float))
        poly = params.get('coeffs') if params else cfg.get('cfunE_simple')
        if poly is None:
            raise ValueError("polynomial cfunE_kind requires 'coeffs' callable in cfunE_params")
        ca = poly(alpha_arr)
        return _scalar(ca[0]) if len(alpha_arr) == 1 else ca

    if mode == 'smooth':
        # Use PCHIP interpolant (built once, cached on g)
        if not hasattr(g, '_cfunE_pchip') or g._cfunE_pchip is None:
            _build_cfunE_pchip()
        alpha_arr = np.atleast_1d(np.asarray(alpha, dtype=float))
        ca = g._cfunE_pchip(alpha_arr)
        return _scalar(ca[0]) if len(alpha_arr) == 1 else ca

    # --- Complex mode (canonical SP form, optionally with legacy addons) ---
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    ca = np.zeros(len(alpha))

    kappa = params.get('kappa', 1.1)
    use_legacy_addons = params.get('use_legacy_addons', False)
    kappa1_dist = params.get('kappa1_distortion', 0.0)
    # spike_alpha = α at which the addon blows up.  Defaults to α₁ (legacy).
    maxa = params.get('spike_alpha', g.alpha1)
    # addon_cap = cap height — see _build_cfunE_smooth for explanation.
    maxc = params.get('addon_cap', 100.0)
    alphabar = (g.alpha0 + g.alpha1) / 2.0

    # Pre-compute base + distortion at α = maxa for the parallel right-piece.
    if use_legacy_addons:
        gu_s = _scalar(gammaupdate2(maxa, g.alpha0, g.rp))
        inner_r_s = (g.Pi + 1 + _scalar(cfun(maxa))) / gu_s - 1
        base_at_s = gu_s * (1 + _scalar(dfuninv(kappa * _scalar(dfun(inner_r_s))))) - (1 + g.PiE)
        distort_at_s = kappa1_dist * maxa * (maxa - alphabar)
        cfun_at_s = _scalar(cfun(maxa))

    for i in range(len(alpha)):
        a = alpha[i]

        if use_legacy_addons and a > maxa:
            # Parallel extension: K^E(α) = K^E(s) + (cfun(α) - cfun(s)).
            ca[i] = base_at_s + distort_at_s + 0.1 * maxc + (_scalar(cfun(a)) - cfun_at_s)
            continue

        gu = gammaupdate2(a, g.alpha0, g.rp)
        gu = _scalar(gu)

        # Canonical SP base cost: γ(α)(1+D⁻¹((1/κ)D((1+K)/γ-1))) - 1 - Π^E
        # NOTE the code uses `kappa * dfun(inner_r)` which equals (1/κ_paper)·D(·)
        # when the code's kappa equals 1/κ_paper.  Convention preserved from MATLAB.
        inner_r = (g.Pi + 1 + _scalar(cfun(a))) / gu - 1
        base_cost = gu * (1 + _scalar(dfuninv(kappa * _scalar(dfun(inner_r))))) - (1 + g.PiE)

        addon = 0.0
        distortion = 0.0
        if use_legacy_addons:
            distortion = kappa1_dist * a * (a - alphabar)
            if a < maxa:
                addon = 0.1 * min(0.1 * (1.0 / max(maxa - a, 1e-6) - 1.0 / maxa), maxc)
            else:  # a == maxa
                addon = 0.1 * maxc

        ca[i] = base_cost + distortion + addon

    return _scalar(ca[0]) if len(alpha) == 1 else ca


def cfunE_prime(alpha, eps=1e-5):
    """First derivative of entry cost function.
    Uses PCHIP analytical derivative for smooth (SP) mode, numerical otherwise."""
    cfg = PARAM_CONFIGS[ACTIVE_CONFIG]
    mode, _ = _resolve_cfunE_mode(cfg)
    if mode == 'smooth' and hasattr(g, '_cfunE_pchip') and g._cfunE_pchip is not None:
        return _scalar(g._cfunE_pchip(np.atleast_1d(alpha), 1)[0])
    return (_scalar(cfunE(alpha + eps)) - _scalar(cfunE(alpha - eps))) / (2 * eps)


def gam0E(al, rpE, cum_gd=None, cum_bd=None):
    """
    Gamma after first round of entry for new entrants. Matches gam0E.m
    If cum_gd/cum_bd are provided (pre-computed at rpE), uses O(1) lookup.
    """
    al = _scalar(al)

    if cum_gd is not None:
        # Fast path: lookup pre-computed state
        idx = np.searchsorted(g.almass, al, side='right')
        gfun_prev = cum_gd[idx]
        bfun_prev = cum_bd[idx]
    else:
        # Original loop
        gfun_prev = gpriorfun(g.omvec).copy()
        bfun_prev = bpriorfun(g.omvec).copy()
        dfun_rpE = _scalar(dfun(rpE))

        i = 0
        while i < len(g.almass) and g.almass[i] <= al:
            alp_i = g.almass[i]
            w_i = g.wmass[i]

            mask_g = g.omvec <= (g.beta + alp_i * (1 - g.beta))
            mask_b = g.omvec >= (1 - g.beta + alp_i * g.beta)
            denom = np.sum(g.delom * gfun_prev[mask_g]) + np.sum(g.delom * bfun_prev[mask_b])

            gn = gfun_prev - gfun_prev * mask_g * w_i / denom / dfun_rpE
            bn = bfun_prev * (1 - mask_b * w_i / denom / dfun_rpE)

            gfun_prev = gn
            bfun_prev = bn

            i += 1
            if i >= len(g.almass):
                break

    mask_g2 = g.omvec <= (g.beta + al * (1 - g.beta))
    mask_b2 = g.omvec >= (1 - g.beta + al * g.beta)
    num = np.sum(g.delom * gfun_prev[mask_g2])
    den = num + np.sum(g.delom * bfun_prev[mask_b2])
    return num / den


def gbupdate(al, alstart, r, gprev, bprev):
    """
    Update g and b distributions with incumbents between alstart and al.
    Matches gbupdate.m. Uses globals: beta, wmass, almass, omvec, delom
    """
    ind = np.where(g.almass > alstart)[0]
    if len(ind) == 0:
        return gprev.copy(), bprev.copy()

    i = ind[0]
    if g.almass[i] > al:
        return gprev.copy(), bprev.copy()

    gd = gprev.copy()
    bd = bprev.copy()
    dfun_r = _scalar(dfun(r))

    while i < len(g.almass) and g.almass[i] <= al:
        alp_i = g.almass[i]
        w_i = g.wmass[i]

        mask_g = g.omvec <= (g.beta + alp_i * (1 - g.beta))
        mask_b = g.omvec >= (1 - g.beta + alp_i * g.beta)
        denom = np.sum(g.delom * gd[mask_g]) + np.sum(g.delom * bd[mask_b])

        gn = gd - gd * mask_g * w_i / denom / dfun_r
        bn = bd * (1 - mask_b * w_i / denom / dfun_r)

        gd = gn
        bd = bn

        i += 1
        if i >= len(g.almass):
            break

    return gd, bd


def gamfunE(w, al, alp, r, gprev, bprev):
    """Gamma for new entrants. Matches gamfunE.m"""
    mask_g = g.omvec <= (g.beta + alp * (1 - g.beta))
    mask_b = g.omvec >= (1 - g.beta + alp * g.beta)
    denom = np.sum(g.delom * gprev[mask_g]) + np.sum(g.delom * bprev[mask_b])
    dfun_r = _scalar(dfun(r))

    gnext = gprev - gprev * mask_g * w / denom / dfun_r
    bnext = bprev * (1 - mask_b * w / denom / dfun_r)

    gn, bn = gbupdate(al, alp, r, gnext, bnext)

    mask_g2 = g.omvec <= (g.beta + al * (1 - g.beta))
    mask_b2 = g.omvec >= (1 - g.beta + al * g.beta)

    num = np.sum(g.delom * gn[mask_g2])
    den = num + np.sum(g.delom * bn[mask_b2])
    return num / den


def profitE(w, alpha, a, rpE, gfunprevE, bfunprevE):
    """Profit for new entrants. Matches profitE.m"""
    return (1 + rpE) * gamfunE(w, alpha, a, rpE, gfunprevE, bfunprevE) - _scalar(cfunE(alpha))


def gamcplxE(w, rpE, alp, gfunprevE, bfunprevE):
    """Complex gamma for new entrant optimization. Matches gamcplxE.m"""
    res = minimize_scalar(
        lambda alpha: _scalar(cfunE(alpha)) - (1 + rpE) * gamfunE(w, alpha, alp, rpE, gfunprevE, bfunprevE),
        bounds=(alp + g.Delta, g.alpha1E), method='bounded'
    )
    alopt = res.x
    return (profitE(w, alopt, alp, rpE, gfunprevE, bfunprevE) - 1 - g.PiE)**2


def mingamfunE(w, al, alp, r, gprev, bprev):
    """Minimum b function for new entrants. Matches mingamfunE.m"""
    mask_g = g.omvec <= (g.beta + alp * (1 - g.beta))
    mask_b = g.omvec >= (1 - g.beta + alp * g.beta)
    denom = np.sum(g.delom * gprev[mask_g]) + np.sum(g.delom * bprev[mask_b])
    dfun_r = _scalar(dfun(r))

    gnext = gprev - gprev * mask_g * w / denom / dfun_r
    bnext = bprev * (1 - mask_b * w / denom / dfun_r)

    gn, bn = gbupdate(al, alp, r, gnext, bnext)
    return np.min(bn)


def gammaNSfunE(al):
    """Gamma for non-selective entry (new entrants). Matches gammaNSfunE.m"""
    al = np.atleast_1d(np.asarray(al, dtype=float))
    goodleftover = np.array([
        quad(gpriorfun_scalar, g.beta + a * (1 - g.beta), 1)[0] for a in al
    ])
    gammaNS = goodleftover / (goodleftover + g.badleftoverE)
    return _scalar(gammaNS[0]) if len(al) == 1 else gammaNS


def NSfunE(al):
    """Non-selective entry profit for new entrants. Matches NSfunE.m"""
    al = _scalar(al)
    gNS = gammaNSfunE(al)
    return gNS * (1 + min(_scalar(cfunE(al)) + g.PiE, _scalar(cfun(al)) + g.Pi)) - (1 + g.PiE)


def rtildeafunE(al, al2):
    """Tilde-r calculation for new entrants. Matches rtildeafunE.m"""
    al2_scalar = _scalar(al2) if np.isscalar(al2) else _scalar(al2)

    gNS_al2 = gammaNSfunE(al2_scalar)
    gNS_al = gammaNSfunE(al)

    cost_al2 = min(_scalar(cfunE(al2_scalar)) + g.PiE, _scalar(cfun(al2_scalar)) + g.Pi)

    return gNS_al2 / gNS_al * (1 + cost_al2) - 1


def fifunE(al, al2):
    """Fraction function for NS entry. Matches fifunE.m"""
    al = np.atleast_1d(np.asarray(al, dtype=float))
    wcimvec = wcim(al)

    if len(al) == 1:
        rt = _scalar(rtildeafunE(al[0], al2))
        return _scalar(wcimvec) / (_scalar(dfun(rt)) * (1 - g.beta) *
                                 _scalar(gpriorfun(g.beta + al[0] * (1 - g.beta))))
    else:
        fi = np.zeros(len(al))
        rt0 = _scalar(rtildeafunE(al[0], al2))
        fi[0] = wcimvec[0] / (_scalar(dfun(rt0)) *
                               quad(gpriorfun_scalar, g.beta + al[0] * (1 - g.beta),
                                    g.beta + al[1] * (1 - g.beta))[0])
        for i in range(1, len(al)):
            rt_i = _scalar(rtildeafunE(al[i], al2))
            fi[i] = wcimvec[i] / (_scalar(dfun(rt_i)) *
                                   quad(gpriorfun_scalar, g.beta + al[i-1] * (1 - g.beta),
                                        g.beta + al[i] * (1 - g.beta))[0])
        return fi


def wcimE(al):
    """Wealth difference in CIM region for new entrants. Matches wcimE.m"""
    al = np.atleast_1d(np.asarray(al, dtype=float))
    if len(al) == 1:
        c_inc = _scalar(cfun(al[0])) + g.Pi
        c_ent = _scalar(cfunE(al[0])) + g.PiE
        if c_inc > c_ent:
            return ((_scalar(dfun(c_ent)) - _scalar(dfun(c_inc))) *
                    (1 - g.beta) * _scalar(gpriorfun(g.beta + al[0] * (1 - g.beta))))
        else:
            return 0.0
    else:
        w = np.zeros(len(al))
        for i in range(1, len(al)):
            c_inc = _scalar(cfun(al[i])) + g.Pi
            c_ent = _scalar(cfunE(al[i])) + g.PiE
            if c_inc > c_ent:
                w[i] = ((_scalar(dfun(c_ent)) - _scalar(dfun(c_inc))) *
                        quad(gpriorfun_scalar, g.beta + al[i-1] * (1 - g.beta),
                             g.beta + al[i] * (1 - g.beta))[0])
            else:
                w[i] = 0.0
        return w


def rfunE(al, al0, al1, al2):
    """Interest rate function for new entrants. Matches rfunE.m"""
    al = np.atleast_1d(np.asarray(al, dtype=float))
    r = np.zeros(len(al))

    for i in range(len(al)):
        if al[i] < al0:
            r[i] = np.nan
        elif al[i] <= al1:
            r[i] = g.rpE
        elif al[i] <= g.alpha1:
            diffs = (al[i] - g.almassshort)**2
            ind = np.argmin(diffs)
            r[i] = g.rhat[ind]
        elif al[i] <= min(al2, g.alpha2):
            r[i] = min(_scalar(cfun(al[i])) + g.Pi, _scalar(cfunE(al[i])) + g.PiE)
        elif al[i] <= g.alpha2:
            r[i] = _scalar(rtildeafunE(al[i], g.alpha2E))
        elif g.alpha2 < al[i] <= al2:
            r[i] = _scalar(cfunE(al[i])) + g.PiE
        else:
            r[i] = g.rnsE

    return r


def prof0Efun(r):
    """Entry profit for new entrants at pooling rate. Matches prof0Efun.m"""
    r = _scalar(r)
    # Pre-compute cumulative states at this trial rate (once per fsolve eval)
    cum_gd_r, cum_bd_r = _precompute_cum_states(r)
    res = minimize_scalar(
        lambda alpha: _scalar(cfunE(alpha)) - gam0E(alpha, r, cum_gd_r, cum_bd_r) * (1 + r),
        bounds=(0, 1), method='bounded'
    )
    al0E = res.x
    return gam0E(al0E, r, cum_gd_r, cum_bd_r) * (1 + r) - _scalar(cfunE(al0E)) - 1 - g.PiE


# =============================================================================
# Baseline computation (equivalent to main.m)
# =============================================================================

def run_baseline():
    """Run the baseline model using the analytical solver.

    Replaces the older discrete iterative algorithm with the closed-form
    analytical solution for Region I (same as credit_model.solve_nested_analytical).
    Reconstructs discrete arrays (almass, wmass, gfunprev, bfunprev) so that
    run_mainE() can consume them without changes.
    """
    print("=" * 60)
    print(f"Running baseline computation (analytical)  [config: {ACTIVE_CONFIG}]")
    print("=" * 60)

    # =====================================================================
    # Parameters from active config
    # =====================================================================
    cfg = PARAM_CONFIGS[ACTIVE_CONFIG]
    g.Pi = cfg['Pi']
    g.beta = cfg['beta']
    g.BperG = cfg['BperG']
    beta = g.beta

    # Discretization (needed for run_mainE compatibility)
    g.Delta = 0.001
    g.delom = 0.0001
    g.omvec = np.linspace(0, 1, int(round(1 / g.delom)))

    # =====================================================================
    # Find alpha0 and rp
    # =====================================================================
    res = minimize_scalar(
        lambda alpha: (g.Pi + _scalar(cfun(alpha)) + 1) / gam0(alpha),
        bounds=(0.01, 0.99), method='bounded'
    )
    g.alpha0 = res.x
    g.rp = (g.Pi + _scalar(cfun(g.alpha0)) + 1) / gam0(g.alpha0) - 1

    print(f"  alpha0 = {g.alpha0:.6f}")
    print(f"  rp     = {g.rp:.6f}")

    # =====================================================================
    # Find alpha1: where c(alpha1) + Pi = rp
    # =====================================================================
    if g.rp - _scalar(cfun(1.0)) > g.Pi:
        g.alpha1 = 1.0
    else:
        g.alpha1 = brentq(lambda a: _scalar(cfun(a)) - (g.rp - g.Pi),
                          0.01, 0.99)

    print(f"  alpha1 = {g.alpha1:.6f}")

    D_rp = _scalar(dfun(g.rp))

    # =====================================================================
    # Region I: Analytical solution on a fine alpha grid
    # =====================================================================
    n_R1 = 2000
    alphas = np.linspace(g.alpha0, g.alpha1, n_R1)
    da = alphas[1] - alphas[0] if n_R1 > 1 else 1e-6

    # Thresholds
    omega_g_vals = beta + alphas * (1 - beta)
    omega_b_vals = 1 - beta + alphas * beta

    # Prior-dependent inputs (uniform priors: g=1, b=BperG)
    g_tilde = np.array([gpriorfun_scalar(og) for og in omega_g_vals])
    b_tilde = np.array([bpriorfun_scalar(ob) for ob in omega_b_vals])
    B0_tilde = np.array([quad(bpriorfun_scalar, ob, 1)[0]
                         for ob in omega_b_vals])

    # Gamma(alpha) = (1 + K(alpha)) / (1 + rp) and its derivative
    K_vals = np.array([_scalar(cfun(a)) for a in alphas]) + g.Pi
    Gamma = (1 + K_vals) / (1 + g.rp)
    Gamma_p = np.array([_scalar(cfun_prime(a)) for a in alphas]) / (1 + g.rp)

    one_minus_Gamma = 1 - Gamma

    # General T formula
    denom_T = Gamma_p * B0_tilde - beta * b_tilde * Gamma * one_minus_Gamma
    denom_T_safe = np.where(np.abs(denom_T) < 1e-15, 1e-15, denom_T)
    T_vals = (1 - beta) * g_tilde * one_minus_Gamma * B0_tilde / denom_T_safe

    # Derived quantities
    G_vals = Gamma * T_vals
    B_vals = one_minus_Gamma * T_vals

    # Depletion factor E(alpha) = B(alpha) / B0_tilde(alpha)
    E_vals = B_vals / np.where(np.abs(B0_tilde) < 1e-15, 1e-15, B0_tilde)

    # theta(alpha) = -d(ln E)/dalpha
    ln_E = np.log(np.maximum(E_vals, 1e-30))
    theta_vals = np.zeros_like(alphas)
    theta_vals[1:-1] = -(ln_E[2:] - ln_E[:-2]) / (2 * da)
    theta_vals[0] = -(ln_E[1] - ln_E[0]) / da
    theta_vals[-1] = -(ln_E[-1] - ln_E[-2]) / da

    # w(alpha) = theta * D(rp) * T
    w_vals = theta_vals * D_rp * T_vals
    w_vals = np.maximum(w_vals, 0)

    # Cumulative capital in Region I
    W_cumsum_R1 = np.zeros_like(alphas)
    W_cumsum_R1[1:] = np.cumsum(w_vals[:-1]) * da

    print(f"  Region I capital = {W_cumsum_R1[-1]:.6f}")

    # Store fine-grid baseline for use by entry analytical solver and SP cost
    g.baseline_alphas_fine = alphas.copy()
    g.baseline_w_fine = w_vals.copy()
    g.gamma_inc_fine = Gamma.copy()  # γ(α, 1, r_p) on the Region-I fine grid

    # Build a PCHIP interpolant for γ_inc(α). Used by the canonical SP cost.
    from scipy.interpolate import PchipInterpolator
    g.gamma_inc_interp = PchipInterpolator(alphas, Gamma, extrapolate=True)

    # =====================================================================
    # Leftover borrowers at end of Region I
    # =====================================================================
    # Good outside acceptance: integral of g from omega_g to 1
    G_outside = np.array([quad(gpriorfun_scalar, og, 1)[0]
                          for og in omega_g_vals])
    G_leftover = G_vals + G_outside

    # Bad remaining
    omega_b_0 = 1 - beta + g.alpha0 * beta
    B_below_0 = quad(bpriorfun_scalar, 0, omega_b_0)[0]
    b_tilde_E = b_tilde * E_vals * beta
    B_dropouts = np.zeros_like(alphas)
    B_dropouts[1:] = np.cumsum(b_tilde_E[:-1]) * da
    B_leftover = B_below_0 + B_dropouts + B_vals

    g.badleftover = B_leftover[-1]

    # =====================================================================
    # Find alpha2 and WNS (non-selective lenders)
    # =====================================================================
    alpha2_result = find_alpha2_mainE(g.alpha1)
    if alpha2_result is not None:
        g.alpha2 = alpha2_result
        g.WNS = _scalar(dfun(cfun(g.alpha2) + g.Pi)) * (
            g.badleftover + quad(gpriorfun_scalar,
                                 beta + g.alpha2 * (1 - beta), 1)[0])
    else:
        g.WNS = 0.0
        g.alpha2 = 1.0

    # =====================================================================
    # Cumulative wealth (Region I + Region II + NS)
    # =====================================================================
    cim_alpha = np.linspace(g.alpha1, g.alpha2, 100)
    W_R2 = np.sum(wcim(cim_alpha))
    g.W_total = W_cumsum_R1[-1] + W_R2 + g.WNS

    # =====================================================================
    # Reconstruct discrete arrays for run_mainE() compatibility
    # =====================================================================
    # almass: evenly spaced at Delta from alpha0 through alpha1
    almass_pts = np.arange(g.alpha0, g.alpha1, g.Delta)
    if len(almass_pts) == 0:
        almass_pts = np.array([g.alpha0])

    # wmass: interpolate analytical w density at almass points, convert to
    # lending amount per step (density * Delta)
    w_at_almass = np.interp(almass_pts, alphas, w_vals)
    wmass_pts = w_at_almass * g.Delta

    g.almass = almass_pts
    g.wmass = wmass_pts

    # Reconstruct gfunprev/bfunprev by running forward depletion on omega grid
    gd = gpriorfun(g.omvec).copy()
    bd = bpriorfun(g.omvec).copy()
    dfun_rp = D_rp

    for k in range(len(g.almass)):
        mask_g_k = g.omvec <= (beta + g.almass[k] * (1 - beta))
        mask_b_k = g.omvec >= (1 - beta + g.almass[k] * beta)
        raw_denom = (np.sum(g.delom * gd * mask_g_k) +
                     np.sum(g.delom * bd * mask_b_k))
        denom_k = raw_denom * dfun_rp
        if denom_k > 0:
            factor = g.wmass[k] / denom_k
            gd = gd * (1 - mask_g_k * factor)
            bd = bd * (1 - mask_b_k * factor)

    g.gfunprev = gd
    g.bfunprev = bd

    # Also store cumulative W vector (WNS + pooling + CIM) for compatibility
    cim_alpha = np.linspace(g.alpha1, g.alpha2, 100)
    g.W = np.cumsum(np.concatenate([[g.WNS], g.wmass, wcim(cim_alpha)]))

    print(f"  alpha2       = {g.alpha2:.6f}")
    print(f"  WNS          = {g.WNS:.6f}")
    print(f"  badleftover  = {g.badleftover:.6f}")
    print(f"  W_total      = {g.W_total:.6f}")
    print("Baseline complete.\n")


# =============================================================================
# Polynomial approximation of cfunE for analytical entry solver
# =============================================================================

def fit_cfunE_smooth(alpha0E, alpha1E, n_fit=300, smoothing=0):
    """
    Build a smooth interpolant of cfunE(alpha) on [alpha0E, alpha1E].

    cfunE depends on the discrete omega-grid depletion, so its values (and
    especially its numerical derivative) are noisy.  A cubic spline gives
    a smooth, analytically differentiable approximation that handles the
    steep rise near alpha1 much better than a global polynomial.

    Parameters
    ----------
    alpha0E, alpha1E : float — domain
    n_fit : int — number of evaluation points
    smoothing : float — spline smoothing (0 = interpolating spline)

    Returns
    -------
    dict with keys:
        spline     : CubicSpline — interpolant for cfunE(alpha)
        alpha_fit  : array — grid used for fitting
        cfunE_fit  : array — raw cfunE values on the grid
        max_err    : float — max |cfunE - spline| on the grid
    """
    from scipy.interpolate import CubicSpline

    alpha_fit = np.linspace(alpha0E, alpha1E, n_fit)
    cfunE_vals = np.array([_scalar(cfunE(a)) for a in alpha_fit])

    # Build cubic spline (smooth, with analytical derivative)
    cs = CubicSpline(alpha_fit, cfunE_vals)

    cfunE_spline_vals = cs(alpha_fit)
    max_err = np.max(np.abs(cfunE_vals - cfunE_spline_vals))

    print(f"  Cubic spline fit to cfunE: n_fit={n_fit}")
    print(f"    max |cfunE - spline| = {max_err:.6e}")
    print(f"    cfunE range: [{cfunE_vals.min():.6f}, {cfunE_vals.max():.6f}]")

    return {
        'spline': cs,
        'alpha_fit': alpha_fit,
        'cfunE_fit': cfunE_vals,
        'max_err': max_err,
    }


# =============================================================================
# Analytical entry equilibrium — Region I (pooling)
# =============================================================================

def solve_entry_pooling_analytical(rpE, alpha0E, alpha1E, n_pts=500,
                                   cfunE_poly_info=None):
    """
    Analytical T^E construction for entry Region I (pooling).

    Uses the same T-formula as the baseline analytical solver, but with
    K^E(alpha) = PiE + cfunE(alpha) and entry rate rpE replacing K and rp.

    The incumbent w(alpha) from the baseline is fixed. The entry density is:
        w^E(alpha) = theta^E * D(rpE) * T^E(alpha) - w_incumbent(alpha)
    Entry occurs only where w^E > 0.

    Parameters
    ----------
    rpE : float — entry pooling rate
    alpha0E : float — marginal entrant skill
    alpha1E : float — upper boundary of entry pooling region
    n_pts : int — grid points (default 500)
    cfunE_poly_info : dict or None — if provided, use smooth spline
        approximation (output of fit_cfunE_smooth) for KE and KE'.

    Returns
    -------
    dict with keys: alphas, wE, w_total, w_incumbent, TE, GE, BE, gammaE, da
    """
    beta = g.beta

    alphas = np.linspace(alpha0E, alpha1E, n_pts)
    da = alphas[1] - alphas[0] if n_pts > 1 else 1e-6

    # Thresholds
    omega_g_vals = beta + alphas * (1 - beta)
    omega_b_vals = 1 - beta + alphas * beta

    # Prior-dependent inputs (uniform priors)
    g_tilde = np.array([gpriorfun_scalar(og) for og in omega_g_vals])
    b_tilde = np.array([bpriorfun_scalar(ob) for ob in omega_b_vals])
    B0_tilde = np.array([quad(bpriorfun_scalar, ob, 1)[0]
                         for ob in omega_b_vals])

    # K^E(alpha) = PiE + cfunE(alpha) and its derivative
    if cfunE_poly_info is not None:
        cs = cfunE_poly_info['spline']
        KE = g.PiE + cs(alphas)
        KE_prime = cs(alphas, 1)  # analytical first derivative of cubic spline
        print(f"  Using cubic spline cfunE")
    else:
        print("  Computing cfunE on grid (raw)...")
        KE = np.array([g.PiE + _scalar(cfunE(a)) for a in alphas])
        print("  Computing cfunE derivative (numerical)...")
        KE_prime = np.array([cfunE_prime(a) for a in alphas])

    # Effective range: only where KE < rpE (entry is profitable)
    # Beyond this, the T^E formula gives negative/nonsensical values.
    profitable = KE < rpE
    if not np.any(profitable):
        print("  WARNING: no profitable entry range (KE >= rpE everywhere)")
        return {
            'alphas': alphas, 'da': da, 'wE': np.zeros_like(alphas),
            'w_total': np.zeros_like(alphas),
            'w_incumbent': np.zeros_like(alphas),
            'TE': np.zeros_like(alphas), 'GE': np.zeros_like(alphas),
            'BE': np.zeros_like(alphas), 'EE': np.zeros_like(alphas),
            'gammaE': np.ones_like(alphas), 'theta': np.zeros_like(alphas),
            'KE': KE, 'KE_prime': KE_prime,
            'WE_cumsum': np.zeros_like(alphas),
        }

    # T^E formula (eq. 4 from entry_Talpha_construction.tex)
    num = (1 - beta) * g_tilde * (rpE - KE) * B0_tilde * (1 + rpE)
    den = (KE_prime * B0_tilde * (1 + rpE)
           - beta * b_tilde * (1 + KE) * (rpE - KE))
    den_safe = np.where(np.abs(den) < 1e-15, 1e-15, den)
    TE = num / den_safe

    # Zero out T^E where KE >= rpE (not profitable)
    TE = np.where(profitable, TE, 0.0)
    TE = np.maximum(TE, 0.0)  # T is a mass, can't be negative

    # Good/bad in acceptance region
    GE = (1 + KE) / (1 + rpE) * TE
    BE = (rpE - KE) / (1 + rpE) * TE
    gammaE = np.where(TE > 1e-15, GE / TE, 1.0)

    # Depletion factor
    EE = BE / np.where(np.abs(B0_tilde) < 1e-15, 1e-15, B0_tilde)

    # theta^E = -d(ln E^E)/dalpha  (same approach as baseline analytical solver)
    # Note: the notes eq. (7) has a sign error in the B-based formula;
    # using E = B/B0_tilde directly is equivalent and confirmed correct.
    ln_EE = np.log(np.maximum(EE, 1e-30))
    theta = np.zeros_like(alphas)
    theta[1:-1] = -(ln_EE[2:] - ln_EE[:-2]) / (2 * da)
    theta[0] = -(ln_EE[1] - ln_EE[0]) / da
    theta[-1] = -(ln_EE[-1] - ln_EE[-2]) / da

    # Total lending density in combined system
    D_rpE = _scalar(dfun(rpE))
    w_total = theta * D_rpE * TE
    w_total = np.maximum(w_total, 0)

    # Boundary fix: near alpha1E, theta→∞ and T→0 (0×∞ indeterminate form)
    # creates a spike in w_total at the last 1-2 grid points. Zero out where
    # rpE - KE < 0.005 (entry is negligible: T^E ≈ 0).
    boundary_mask = (rpE - KE) < 0.005
    w_total[boundary_mask] = 0.0

    # Incumbent w at entry alphas:
    #   - Region I (alpha <= alpha1): pooling density from baseline fine grid
    #   - Region II (alpha > alpha1): CIM density D(r_CIM) * g_tilde * (1-beta)
    w_inc_pooling = np.interp(alphas, g.baseline_alphas_fine,
                              g.baseline_w_fine, left=0, right=0)
    w_inc_cim = np.zeros_like(alphas)
    cim_mask = alphas > g.alpha1
    if np.any(cim_mask):
        for j in np.where(cim_mask)[0]:
            a_j = alphas[j]
            if a_j <= g.alpha2:
                r_cim_j = _scalar(cfun(a_j)) + g.Pi
                w_inc_cim[j] = (_scalar(dfun(r_cim_j)) *
                                (1 - beta) * gpriorfun_scalar(beta + a_j * (1 - beta)))
    w_incumbent = w_inc_pooling + w_inc_cim

    # Entry density (ironed: entrants only where total exceeds incumbent)
    wE = np.maximum(0, w_total - w_incumbent)

    # Cumulative entry capital
    WE_cumsum = np.zeros_like(alphas)
    WE_cumsum[1:] = np.cumsum(wE[:-1]) * da

    print(f"  Analytical entry: total W^E = {WE_cumsum[-1]:.6f}")

    # Diagnostics
    mid = n_pts // 2
    print(f"  --- Diagnostics at alpha={alphas[mid]:.4f} (midpoint) ---")
    print(f"    KE={KE[mid]:.4f}  KE'={KE_prime[mid]:.4f}  rpE-KE={rpE-KE[mid]:.4f}")
    print(f"    TE={TE[mid]:.4f}  BE={BE[mid]:.4f}  EE={EE[mid]:.4f}")
    print(f"    theta={theta[mid]:.4f}  D(rpE)={D_rpE:.4f}")
    print(f"    w_total={w_total[mid]:.4f}  w_incumbent={w_incumbent[mid]:.4f}  wE={wE[mid]:.4f}")
    print(f"  --- KE range: [{KE.min():.4f}, {KE.max():.4f}] ---")
    print(f"  --- KE' range: [{KE_prime.min():.4f}, {KE_prime.max():.4f}] ---")
    print(f"  --- TE range: [{TE.min():.4f}, {TE.max():.4f}] ---")
    print(f"  --- w_total range: [{w_total.min():.4f}, {w_total.max():.4f}] ---")
    print(f"  --- w_incumbent range: [{w_incumbent.min():.4f}, {w_incumbent.max():.4f}] ---")

    return {
        'alphas': alphas,
        'da': da,
        'wE': wE,
        'w_total': w_total,
        'w_incumbent': w_incumbent,
        'TE': TE,
        'GE': GE,
        'BE': BE,
        'EE': EE,
        'gammaE': gammaE,
        'theta': theta,
        'KE': KE,
        'KE_prime': KE_prime,
        'WE_cumsum': WE_cumsum,
    }


# =============================================================================
# Entry computation (equivalent to mainE.m)
# =============================================================================

def run_mainE():
    """Run the entry model. Equivalent to mainE.m"""
    print("=" * 60)
    print("Running entry computation (mainE.m)")
    print("=" * 60)

    # ---------- Settings ----------
    cfg = PARAM_CONFIGS[ACTIVE_CONFIG]
    _, _cfunE_params = _resolve_cfunE_mode(cfg)
    # kappa1 is the legacy distortion coefficient; only used by SP w/ legacy addons.
    # Defaults to 0 for any kind that doesn't use it; preserved as g.kappa1 because
    # filename templates downstream interpolate it.
    g.kappa1 = _cfunE_params.get('kappa1_distortion', cfg.get('kappa1', 0.0))
    g.PiE = cfg.get('PiE', 0.1)

    print(f"  kappa1 = {g.kappa1}")
    print(f"  PiE    = {g.PiE}")

    # Save baseline values
    g.almassshort = g.almass.copy()   # entry in the pooling region
    g.wmassshort = g.wmass.copy()     # corresponding mass

    # Extend almass and wmass to include CIM region
    n_cim = int(round((g.alpha2 - g.alpha1) / g.Delta))
    cim_alphas = np.linspace(g.alpha1, g.alpha2, max(n_cim, 2))
    g.almass = np.concatenate([g.almass[:-1], cim_alphas])
    g.wmass = np.concatenate([g.wmass, wcim(cim_alphas)])

    # Pre-compute cumulative incumbent states at baseline rate (speeds up cfunE)
    # Uses matlab_gammaupdate2=True to match MATLAB gammaupdate2.m formula
    print("  Pre-computing incumbent states...")
    g._cum_gd_rp, g._cum_bd_rp = _precompute_cum_states(g.rp, matlab_gammaupdate2=True)

    # ---------- Finding alpha0E and rpE, alpha1Emin ----------
    print("  Finding alpha0E, rpE...")

    # MATLAB uses fzero (bracket-based), not fsolve (Newton-based).
    # fzero starts at rp and expands outward to find a sign change,
    # then uses bisection/interpolation within the bracket.
    f_rp = prof0Efun(g.rp)
    rp0 = g.rp
    if f_rp > 0:
        # Search downward from rp for sign change (mimics fzero's bracket search)
        step = g.rp / 50.0
        for k in range(1, 100):
            r_try = g.rp - k * step
            if r_try <= 0:
                break
            f_try = prof0Efun(r_try)
            if f_try < 0:
                rp0 = brentq(prof0Efun, r_try, g.rp)
                break
    elif f_rp < 0:
        # Search upward
        step = g.rp / 50.0
        for k in range(1, 100):
            r_try = g.rp + k * step
            f_try = prof0Efun(r_try)
            if f_try > 0:
                rp0 = brentq(prof0Efun, g.rp, r_try)
                break

    if rp0 < g.rp:
        g.rpE = rp0
        # Pre-compute cumulative states at rpE for gam0E
        cum_gd_rpE, cum_bd_rpE = _precompute_cum_states(g.rpE)
        res = minimize_scalar(
            lambda alpha: _scalar(cfunE(alpha)) - gam0E(alpha, g.rpE, cum_gd_rpE, cum_bd_rpE) * (1 + g.rpE),
            bounds=(0, 1), method='bounded'
        )
        g.alpha0E = res.x
    else:
        g.rpE = g.rp  # no entry in the pooling region

    print(f"  rpE = {g.rpE:.6f}")

    if (1 + g.rpE) - 1 - _scalar(cfunE(1.0)) > g.PiE:
        # Even at max cost, profit is larger than PiE: no end of pooling
        g.alpha1E = 1.0
        alpha1Emin = 1.0
    else:
        alpha1Emin = fsolve(
            lambda alpha: _scalar(cfunE(alpha)) - (g.rpE - g.PiE), g.alpha0E
        )[0]

    if g.rpE == g.rp:
        g.alpha0E = alpha1Emin  # if no entry in pooling, jump over

    print(f"  alpha0E    = {g.alpha0E:.6f}")
    print(f"  alpha1Emin = {alpha1Emin:.6f}")

    # ---------- Calculate rhat ----------
    # rhat: the would-be-CIM-price in the pooling region
    g.rhat = np.zeros(len(g.almassshort))
    g.rhat[0] = (_scalar(dfuninv(g.wmassshort[0])) * (1 - g.beta) *
                 _scalar(gpriorfun(g.beta + g.alpha0 * (1 - g.beta))))
    for i in range(1, len(g.almassshort) - 1):
        integral_val = quad(gpriorfun_scalar,
                            g.beta + g.almassshort[i-1] * (1 - g.beta),
                            g.beta + g.almassshort[i] * (1 - g.beta))[0]
        if integral_val > 0:
            g.rhat[i] = _scalar(dfuninv(g.wmassshort[i] / integral_val))
        else:
            g.rhat[i] = g.rhat[i-1]

    g.rhat[-1] = g.rhat[-2]

    # ---------- Adjust alpha1E ----------
    # alpha1E can be below alpha1 only if rhat > rpE in between
    above = g.rhat > g.rpE
    indexvec = np.arange(1, len(above) + 1)  # 1-indexed like MATLAB
    # Reverse cumulative sum (intended behavior based on comments)
    rev_cumsum = np.cumsum(above[::-1])[::-1]
    x = np.where(rev_cumsum == (len(above) - indexvec))[0]

    almassshortpl1 = np.concatenate([g.almassshort, [g.alpha1]])
    if len(x) > 0:
        g.alpha1E = max(alpha1Emin, np.max(almassshortpl1[x + 1]))
    else:
        g.alpha1E = alpha1Emin

    print(f"  alpha1E = {g.alpha1E:.6f}")

    # ---------- Pooling region ----------
    print("  Computing pooling region for new entrants...")

    gfunprevE, bfunprevE = gbupdate(
        g.alpha0E, 0, g.rpE, gpriorfun(g.omvec), bpriorfun(g.omvec)
    )

    almassE_list = [g.alpha0E]
    wmassE_list = []
    gammaE_list = []

    # Initial gammaE
    mask_g0 = g.omvec <= (g.beta + g.alpha0E * (1 - g.beta))
    mask_b0 = g.omvec >= (1 - g.beta + g.alpha0E * g.beta)
    gammaE_0 = (np.sum(g.delom * gfunprevE[mask_g0]) /
                (np.sum(g.delom * gfunprevE[mask_g0]) +
                 np.sum(g.delom * bfunprevE[mask_b0])))
    gammaE_list.append(gammaE_0)

    n = 2
    gnext_E = None
    bnext_E = None

    while almassE_list[-1] + g.Delta <= g.alpha1E:
        alp = almassE_list[-1]

        if n >= 3:
            gfunprevE = gnext_E.copy()
            bfunprevE = bnext_E.copy()

            mask_g_n = g.omvec <= (g.beta + alp * (1 - g.beta))
            mask_b_n = g.omvec >= (1 - g.beta + alp * g.beta)
            gamE_n = (np.sum(g.delom * gfunprevE[mask_g_n]) /
                      (np.sum(g.delom * gfunprevE[mask_g_n]) +
                       np.sum(g.delom * bfunprevE[mask_b_n])))
            gammaE_list.append(gamE_n)

        # Compute wmax: max w at alp that keeps b non-negative through alpha1E
        wmax_val = fsolve(
            lambda w: mingamfunE(w, g.alpha1E, alp, g.rpE, gfunprevE, bfunprevE),
            1e-9
        )[0]
        wmax_val = max(wmax_val, 1e-12)  # guard against negative

        # Try no-hole solution
        alopt3E = alp + g.Delta
        res3 = minimize_scalar(
            lambda w: 1000 * (profitE(w, alopt3E, alp, g.rpE, gfunprevE, bfunprevE) - (1 + g.PiE))**2,
            bounds=(0, wmax_val), method='bounded'
        )
        wopt3E = res3.x

        res3b = minimize_scalar(
            lambda alpha: 1000 * (_scalar(cfunE(alpha)) -
                                  (1 + g.rpE) * gamfunE(wopt3E, alpha, alp, g.rpE, gfunprevE, bfunprevE)),
            bounds=(alp + g.Delta, g.alpha1E), method='bounded'
        )
        alopt3bE = res3b.x

        if abs(alopt3E - alopt3bE) < g.Delta:
            woptE = wopt3E
            aloptE = alopt3E
        else:
            # Search for holes in support
            res_w = minimize_scalar(
                lambda w: 1000 * gamcplxE(w, g.rpE, alp, gfunprevE, bfunprevE),
                bounds=(0, wmax_val), method='bounded'
            )
            woptE = res_w.x

            res_al = minimize_scalar(
                lambda alpha: 1000 * (_scalar(cfunE(alpha)) -
                                      (1 + g.rpE) * gamfunE(woptE, alpha, alp, g.rpE, gfunprevE, bfunprevE)),
                bounds=(alp + g.Delta, g.alpha1E), method='bounded'
            )
            aloptE = res_al.x

            # Try constrained optimization (fmincon equivalent)
            res2 = minimize(
                lambda w: 1000 * gamcplxE(w[0], g.rpE, alp, gfunprevE, bfunprevE),
                [woptE], bounds=[(0, wmax_val)], method='L-BFGS-B'
            )
            wopt2E = res2.x[0]

            res_al2 = minimize_scalar(
                lambda alpha: 1000 * (_scalar(cfunE(alpha)) -
                                      (1 + g.rpE) * gamfunE(wopt2E, alpha, alp, g.rpE, gfunprevE, bfunprevE)),
                bounds=(alp + g.Delta, g.alpha1E), method='bounded'
            )
            alopt2E = res_al2.x

            if (abs(profitE(woptE, aloptE, alp, g.rpE, gfunprevE, bfunprevE) - (1 + g.PiE)) >
                    abs(profitE(wopt2E, alopt2E, alp, g.rpE, gfunprevE, bfunprevE) - (1 + g.PiE))):
                woptE = wopt2E
                aloptE = alopt2E

        almassE_list.append(aloptE)

        # Update distributions
        mask_g = g.omvec <= (g.beta + alp * (1 - g.beta))
        mask_b = g.omvec >= (1 - g.beta + alp * g.beta)
        denom = (np.sum(g.delom * gfunprevE[mask_g]) +
                 np.sum(g.delom * bfunprevE[mask_b]))
        dfun_rpE = _scalar(dfun(g.rpE))

        gnext_E = gfunprevE - gfunprevE * mask_g * woptE / denom / dfun_rpE
        bnext_E = bfunprevE * (1 - mask_b * woptE / denom / dfun_rpE)
        gnext_E, bnext_E = gbupdate(aloptE, alp, g.rpE, gnext_E, bnext_E)

        wmassE_list.append(woptE)
        n += 1

        if n % 50 == 0:
            print(f"    n={n}, alp={alp:.4f}")

    print(f"  Pooling region: {n-1} iterations")

    # Guard: if loop didn't execute (degenerate/empty pooling region), keep the
    # initial distributions so downstream code has valid arrays to work with.
    # This case arises when K^E is high enough that entrants don't pool but may
    # still enter in CIM/NS regions (e.g., Big Data with Π^E > Π).
    if gnext_E is None:
        gnext_E = gfunprevE.copy()
        bnext_E = bfunprevE.copy()
        print("  [empty pooling region: skipping selective pooling entry]")

    # Final gammaE
    if gnext_E is not None:
        mask_gf = g.omvec <= (g.beta + g.alpha1E * (1 - g.beta))
        mask_bf = g.omvec >= (1 - g.beta + g.alpha1E * g.beta)
        gammaE_final = (np.sum(g.delom * gnext_E[mask_gf]) /
                        (np.sum(g.delom * gnext_E[mask_gf]) +
                         np.sum(g.delom * bnext_E[mask_bf])))
        gammaE_list.append(gammaE_final)

    almassE = np.array(almassE_list)
    wmassE = np.array(wmassE_list)
    gammaE = np.array(gammaE_list)

    # ---------- Calculate badleftoverE ----------
    if g.rpE < g.rp:
        # There is entry in pooling
        g.badleftoverE = np.sum(g.delom * bnext_E)
    else:
        # No entry in pooling
        g.alpha1E = g.alpha1
        g.badleftoverE = g.badleftover
        wmassE = np.array([0.0])
        almassE = np.array([g.alpha1E, g.alpha1E])

    print(f"  badleftoverE = {g.badleftoverE:.6f}")

    # ---------- Analytical entry comparison (Region I only) ----------
    # Skip if pooling region is degenerate (α₀^E ≥ α₁^E - Δ).
    pooling_is_valid = (g.rpE < g.rp) and (g.alpha1E - g.alpha0E > g.Delta * 2)
    if pooling_is_valid:
        print("\n  --- Analytical Entry (Region I) ---")
        cfg = PARAM_CONFIGS[ACTIVE_CONFIG]
        mode, _ = _resolve_cfunE_mode(cfg)

        if mode == 'simple':
            # cfunE is already a smooth polynomial — no spline needed.
            # Run analytical with raw cfunE directly (it IS smooth).
            print("  [Simple cfunE — no spline needed]")
            entry_ana = solve_entry_pooling_analytical(
                g.rpE, g.alpha0E, g.alpha1E, cfunE_poly_info=None)
            entry_ana_raw = entry_ana  # same thing
            g.cfunE_poly_info = None
        else:
            # Complex cfunE — fit spline for smooth analytical version
            poly_info = fit_cfunE_smooth(g.alpha0E, g.alpha1E, n_fit=300)
            g.cfunE_poly_info = poly_info

            print("  [Spline cfunE]")
            entry_ana = solve_entry_pooling_analytical(
                g.rpE, g.alpha0E, g.alpha1E, cfunE_poly_info=poly_info)

            print("  [Raw cfunE]")
            entry_ana_raw = solve_entry_pooling_analytical(
                g.rpE, g.alpha0E, g.alpha1E, cfunE_poly_info=None)

        # Compare: discrete vs analytical
        WE_disc = np.sum(wmassE)
        WE_ana = entry_ana['WE_cumsum'][-1]
        WE_raw = entry_ana_raw['WE_cumsum'][-1]
        print(f"\n  === Region I Entry Capital Comparison ===")
        print(f"  Discrete            W^E = {WE_disc:.6f}")
        print(f"  Analytical          W^E = {WE_ana:.6f}")
        if entry_ana is not entry_ana_raw:
            print(f"  Analytical (raw)    W^E = {WE_raw:.6f}")
        print(f"  Analytical - Discrete   = {WE_ana - WE_disc:.6f}")

        # Store for later plotting
        g.entry_analytical = entry_ana
        g.entry_analytical_raw = entry_ana_raw if entry_ana is not entry_ana_raw else None
    else:
        g.entry_analytical = None
        g.entry_analytical_raw = None

    # ---------- Calculate non-selective and CIM ----------
    print("  Computing non-selective and CIM regions...")

    rprime = _scalar(dfuninv(
        g.WNS / (g.badleftoverE +
                 quad(gpriorfun_scalar, g.beta + g.alpha2 * (1 - g.beta), 1)[0])
    ))
    rdprime = (1 + g.PiE) / gammaNSfunE(g.alpha2) - 1
    rtprime = _scalar(cfunE(g.alpha2)) + g.PiE

    min_r = min(rprime, rdprime, rtprime)
    WNSE = 0.0

    print(f"  rprime   = {rprime:.6f}")
    print(f"  rdprime  = {rdprime:.6f}")
    print(f"  rtprime  = {rtprime:.6f}")

    if rprime == min_r:
        if rprime > _scalar(cfun(g.alpha2)) + g.Pi:
            g.rnsE = rprime
            g.alpha2E = g.alpha2
            WNSE = 0.0
            print('  >> rNS goes up, no NS entry')
        else:
            # Bisection to find alpha2E
            err = 1.0
            alpha_low = g.alpha1E
            alpha_high = g.alpha2
            alit = (alpha_low + alpha_high) / 2.0

            while abs(err) > 0.0001:
                vec = np.linspace(alit, g.alpha2, 100)
                fivec = fifunE(vec, alit)

                upper_limits = g.beta + vec[:-1] * (1 - g.beta)
                Gvec = g.badleftoverE + np.array([
                    quad(gpriorfun_scalar, ul, 1)[0] for ul in upper_limits
                ])

                rtilde_vec = np.array([_scalar(rtildeafunE(v, alit)) for v in vec[:-1]])
                WNSvec = -np.diff(fivec) / np.diff(vec) * Gvec * dfun(rtilde_vec)
                WNSEattop = (fivec[-1] *
                             (g.badleftoverE + quad(gpriorfun_scalar,
                                                    g.beta + g.alpha2 * (1 - g.beta), 1)[0]) *
                             _scalar(dfun(rtildeafunE(g.alpha2, alit))))
                WNSE_iter = np.sum(WNSvec * np.diff(vec)) + WNSEattop

                err = g.WNS - WNSE_iter

                if err > 0:
                    alpha_high = alit
                else:
                    alpha_low = alit

                alit = (alpha_low + alpha_high) / 2.0

            g.alpha2E = alit
            WNSE = 0.0
            g.rnsE = _scalar(rtildeafunE(g.alpha2, g.alpha2E))
            print('  >> NS rate goes down, no NS entry before alpha2, incumbents enter before')

    elif rdprime == min_r:
        if rdprime > _scalar(cfun(g.alpha2)) + g.Pi:
            g.rnsE = rdprime
            g.alpha2E = g.alpha2
            print('  >> rNS goes up with NS entry')
        else:
            # Find alphaNSmax and alpha2E0
            res_ns = minimize_scalar(
                lambda al: -NSfunE(al),
                bounds=(g.alpha1E, 1), method='bounded'
            )
            alphaNSmax = res_ns.x

            if NSfunE(alphaNSmax) > 0:
                # MATLAB uses fzero (bracket-based); use brentq for reliability
                alpha2E0 = brentq(NSfunE, g.alpha1E, alphaNSmax)
            else:
                alpha2E0 = 1.0

            vec = np.linspace(alpha2E0, g.alpha2, 1000)
            fivec = fifunE(vec, alpha2E0)

            upper_limits = g.beta + vec[:-1] * (1 - g.beta)
            Gvec = g.badleftoverE + np.array([
                quad(gpriorfun_scalar, ul, 1)[0] for ul in upper_limits
            ])

            rtilde_vec = np.array([_scalar(rtildeafunE(v, alpha2E0)) for v in vec[:-1]])
            WNSvec = -np.diff(fivec) / np.diff(vec) * Gvec * dfun(rtilde_vec)
            # Note: MATLAB line 210 has rtildeafunE(vec, alpha2E0) which seems to be
            # a bug; using scalar alpha2 to match the pattern from the other branch
            WNSEattop = (fivec[-1] *
                         (g.badleftoverE + quad(gpriorfun_scalar,
                                                g.beta + g.alpha2 * (1 - g.beta), 1)[0]) *
                         _scalar(dfun(rtildeafunE(g.alpha2, alpha2E0))))
            WNSE_val = np.sum(WNSvec * np.diff(vec)) + WNSEattop

            if WNSE_val > g.WNS:
                g.alpha2E = alpha2E0
                g.rnsE = _scalar(rtildeafunE(g.alpha2, g.alpha2E))
                WNSE = WNSE_val - g.WNS  # atom of entrant NS capital at rtilde(alpha2)
                print('  >> NS rate goes down, with NS entry before alpha2')
            else:
                print('  >> we are on a branch which we guessed did not exist')
                g.alpha2E = g.alpha2
                g.rnsE = rprime

    else:  # rtprime == min_r
        # Find alpha2Eprime
        res_c = minimize_scalar(
            lambda al: ((1 + g.PiE) / gammaNSfunE(al) - 1 - _scalar(cfunE(al)) - g.PiE)**2,
            bounds=(g.alpha1E, 1), method='bounded'
        )
        alc = res_c.x
        if ((1 + g.PiE) / gammaNSfunE(alc) - 1 - _scalar(cfunE(alc)) - g.PiE)**2 < 0.001:
            alpha2Ep = alc
        else:
            alpha2Ep = 1.0

        # Find alpha2Edoubleprime
        res_c2 = minimize_scalar(
            lambda al: (_scalar(dfuninv(
                g.WNS / (g.badleftoverE +
                         quad(gpriorfun_scalar, g.beta + al * (1 - g.beta), 1)[0])
            )) - _scalar(cfunE(al)) - g.PiE)**2,
            bounds=(g.alpha1E, 1), method='bounded'
        )
        alc2 = res_c2.x
        val_c2 = _scalar(dfuninv(
            g.WNS / (g.badleftoverE +
                     quad(gpriorfun_scalar, g.beta + alc2 * (1 - g.beta), 1)[0])
        ))
        if (val_c2 - _scalar(cfunE(alc2)) - g.PiE)**2 < 0.001:
            alpha2Edp = alc2
        else:
            alpha2Edp = 1.0

        g.alpha2E = min(alpha2Edp, alpha2Ep)
        g.rnsE = _scalar(cfunE(g.alpha2E)) + g.PiE

        WNSE = max(0.0,
                    _scalar(dfun(_scalar(cfunE(g.alpha2E)) + g.PiE)) *
                    (g.badleftoverE + quad(gpriorfun_scalar,
                                           g.beta + g.alpha2E * (1 - g.beta), 1)[0])
                    - g.WNS)

        if WNSE > 0:
            print('  >> CIM extended, with NS entry')
        else:
            print('  >> CIM extended, without NS entry')

    print(f"  alpha2E = {g.alpha2E:.6f}")
    print(f"  rnsE    = {g.rnsE:.6f}")

    # ---------- Cumulative wealth ----------
    cim_alphasE = np.linspace(g.alpha1E, g.alpha2E, 100)
    # WE = selective entry only (pooling + CIM); WNSE added as jump at alpha=0
    WE = np.cumsum(np.concatenate([wmassE, wcimE(cim_alphasE)]))

    # ---------- Save info ----------
    if g.badleftover < g.badleftoverE:
        filename = f'equivalentBLOup_PiEnonlinear{int(100*g.PiE)}_{g.kappa1}'
    else:
        filename = f'equivalentBLOdown_PiEnonlinear{int(100*g.PiE)}_{g.kappa1}'
    print(f"  Filename: {filename}")

    # ---------- Figures ----------
    print("  Generating figures...")

    whereisentry = np.where(wmassE > g.Delta / 100)[0]
    maxentry = whereisentry[-1] if len(whereisentry) > 0 else 0

    # Analytical last entry: last alpha where w^E > 0
    if g.entry_analytical is not None:
        ea_wE = g.entry_analytical['wE']
        ea_al = g.entry_analytical['alphas']
        active = ea_wE > 1e-6
        last_entry_alpha = ea_al[active][-1] if np.any(active) else g.alpha0E
    else:
        last_entry_alpha = almassE[maxentry] if maxentry < len(almassE) else g.alpha0E

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # --- Subplot 1: Interest rate ---
    ax1 = axes[0, 0]
    al_vec = np.linspace(0, 1, 1000)
    r_new = rfunE(al_vec, min(g.alpha0, g.alpha0E), g.alpha1E, g.alpha2E)
    r_base = rfun(al_vec, g.alpha0, g.alpha1, g.alpha2)
    ax1.scatter(al_vec, r_new, s=3, label='new entrants', zorder=2)
    ax1.plot(al_vec, r_base, label='incumbent', zorder=1)
    ax1.set_xlim([0, 1])
    if g.rpE < g.rp:
        ax1.axvline(g.alpha0E, color='b', linestyle=':', linewidth=0.8)
        ax1.text(g.alpha0E, ax1.get_ylim()[1], r'$\alpha_0^E$', color='b', fontsize=8,
                 ha='center', va='bottom')
        ax1.axvline(g.alpha1E, color='b', linestyle=':', linewidth=0.8)
        ax1.text(g.alpha1E, ax1.get_ylim()[1], r'$\alpha_1^E$', color='b', fontsize=8,
                 ha='center', va='bottom')
        ax1.axvline(last_entry_alpha, color='b', linestyle=':', linewidth=0.8)
        ax1.text(last_entry_alpha, ax1.get_ylim()[1], 'last entry', color='b',
                 fontsize=8, ha='center', va='bottom')
    ax1.axvline(g.alpha2E, color='b', linestyle=':', linewidth=0.8)
    ax1.text(g.alpha2E, ax1.get_ylim()[1], r'$\alpha_2^E$', color='b', fontsize=8,
             ha='center', va='bottom')
    ax1.set_xlabel(r'$\alpha$')
    ax1.set_title('interest rate')
    ax1.legend(fontsize=8)

    # --- Subplot 2: Cumulative wealth ---
    ax2 = axes[0, 1]
    WE_total = WNSE + WE[-1]  # total entry capital including NS atom

    # Jump at alpha=0 for WNSE (non-selective entrants, alpha=0)
    if WNSE > 0:
        ax2.plot([0, 0], [0, WNSE], 'k-', lw=2)
        ax2.plot(0, 0, 'ko', markersize=6, markerfacecolor='white', markeredgewidth=1.5)
        ax2.plot(0, WNSE, 'ko', markersize=6, markerfacecolor='black',
                 label='NS entry (discrete)')
    # Flat from 0 to alpha0E at WNSE level
    alpha_flat_E = np.linspace(0, g.alpha0E, 20)
    ax2.plot(alpha_flat_E, WNSE * np.ones_like(alpha_flat_E), 'k-', lw=1)

    # Selective entry: pooling + CIM (offset by WNSE)
    welength = len(WE) - len(almassE[:-1])  # = len(cim_alphasE)
    x_new = np.concatenate([almassE[:-1],
                            np.linspace(g.alpha1E, g.alpha2E, max(welength, 1))])
    # Trim or extend to match WE length
    if len(x_new) < len(WE):
        x_new = np.concatenate([x_new, np.linspace(g.alpha2E, 1, len(WE) - len(x_new))])
    x_new = x_new[:len(WE)]

    ax2.scatter(x_new, WNSE + WE, s=5, c='k', zorder=3, label='discrete entry')
    # Extend flat to the right
    ax2.scatter(np.linspace(g.alpha2E, 1, 100), WE_total * np.ones(100), s=5, c='k')
    ax2.set_ylim([0, 1.5])

    # Analytical cumulative entry wealth (Region I + CIM + NS)
    if g.entry_analytical is not None:
        ea = g.entry_analytical
        WE_R1_total = ea['WE_cumsum'][-1]  # Region I total

        # Region II (CIM) entry: compute density and integrate
        if g.alpha2E > g.alpha1E:
            n_cim_ana = 200
            cim_ana_al = np.linspace(g.alpha1E, g.alpha2E, n_cim_ana)
            da_cim = cim_ana_al[1] - cim_ana_al[0] if n_cim_ana > 1 else 0
            wE_cim_ana = np.zeros(n_cim_ana)
            for j, a in enumerate(cim_ana_al):
                r_inc = _scalar(cfun(a)) + g.Pi
                r_ent = _scalar(cfunE(a)) + g.PiE
                g_tilde_j = gpriorfun_scalar(g.beta + a * (1 - g.beta))
                if r_inc > r_ent:
                    wE_cim_ana[j] = ((_scalar(dfun(r_ent)) - _scalar(dfun(r_inc))) *
                                     (1 - g.beta) * g_tilde_j)
            WE_cim_cumsum = WE_R1_total + np.cumsum(wE_cim_ana) * da_cim
        else:
            cim_ana_al = np.array([g.alpha1E])
            WE_cim_cumsum = np.array([WE_R1_total])

        # NS entry at alpha=0 (offset for analytical curve)
        WE_ana_total = WE_cim_cumsum[-1] + WNSE

        # Plot: WNSE offset + Region I + CIM as one continuous line
        ana_alphas = np.concatenate([ea['alphas'], cim_ana_al])
        ana_cumsum = WNSE + np.concatenate([ea['WE_cumsum'], WE_cim_cumsum])
        ax2.plot(ana_alphas, ana_cumsum, 'b-', lw=2, label='analytical entry')
        # Flat extension after alpha2E
        ax2.plot([g.alpha2E, 1.0], [WE_ana_total, WE_ana_total], 'b-', lw=2)

    # Baseline cumulative wealth
    x_base = np.concatenate([[0], g.almassshort,
                             np.linspace(g.alpha1, g.alpha2, 100)])
    # Trim or pad to match g.W length
    n_W = len(g.W)
    x_base = x_base[:n_W] if len(x_base) >= n_W else np.concatenate([x_base, np.full(n_W - len(x_base), x_base[-1])])
    ax2.scatter(x_base, g.W, s=3, c='r', label='incumbent')
    ax2.scatter(np.linspace(g.alpha2, 1, 10), g.W[-1] * np.ones(10), s=3, c='r')

    # Threshold lines
    ax2.axvline(g.alpha0E, color='blue', ls=':', lw=0.8, alpha=0.5)
    ax2.axvline(g.alpha1E, color='blue', ls='--', lw=0.8, alpha=0.5)
    ax2.axvline(g.alpha2E, color='blue', ls='-.', lw=0.8, alpha=0.5)
    ax2.axvline(g.alpha1, color='red', ls='--', lw=0.8, alpha=0.5)

    ax2.set_xlabel(r'$\alpha$')
    ax2.set_title('cumulative wealth')
    ax2.legend(fontsize=7)

    # --- Subplot 3: Cost + Pi ---
    ax3 = axes[1, 0]
    al_plot = np.linspace(0.01, 0.99, 100)  # avoid boundary issues
    cost_new = np.array([_scalar(cfunE(a)) for a in al_plot]) + g.PiE
    cost_base = cfun(al_plot) + g.Pi
    ax3.plot(al_plot, cost_new, label='new entrants')
    ax3.plot(al_plot, cost_base, label='incumbent')
    # Spline approximation of cfunE (if available)
    if hasattr(g, 'cfunE_poly_info') and g.cfunE_poly_info is not None:
        cs = g.cfunE_poly_info['spline']
        al_spline = np.linspace(g.alpha0E, g.alpha1E, 200)
        ax3.plot(al_spline, cs(al_spline) + g.PiE, 'g--', lw=1.5,
                 label='spline approx')
    ax3.legend()
    ax3.set_title(r'cost + $\Pi$')
    ax3.set_xlabel(r'$\alpha$')

    # --- Subplot 4: Discrete vs Analytical entry w^E ---
    ax4 = axes[1, 1]
    if g.entry_analytical is not None:
        ea = g.entry_analytical       # polynomial version
        ea_raw = g.entry_analytical_raw  # raw cfunE version

        # Analytical (polynomial): w^E density
        ax4.plot(ea['alphas'], ea['wE'], 'b-', lw=1.5,
                 label='analytical (poly) $w^E$')
        # Incumbent w: pooling region (from analytical) + CIM region extension
        ax4.plot(ea['alphas'], ea['w_incumbent'], 'r--', lw=1, alpha=0.6,
                 label='incumbent $w$')
        # Extend incumbent w into CIM region (alpha1 to alpha2)
        if g.alpha2 > g.alpha1:
            cim_ext = np.linspace(g.alpha1 + 0.001, g.alpha2, 100)
            w_cim_ext = np.array([
                _scalar(dfun(cfun(a) + g.Pi)) *
                (1 - g.beta) * gpriorfun_scalar(g.beta + a * (1 - g.beta))
                for a in cim_ext])
            ax4.plot(cim_ext, w_cim_ext, 'r--', lw=1, alpha=0.6)

        # Analytical CIM entry density (Region II: alpha1E to alpha2E)
        # Entry where entrant cost < incumbent cost: w^E = (D(r_ent) - D(r_inc)) * g * (1-beta)
        if g.alpha2E > g.alpha1E:
            cim_entry_al = np.linspace(g.alpha1E + 0.001, g.alpha2E, 200)
            wE_cim = np.zeros(len(cim_entry_al))
            for j, a in enumerate(cim_entry_al):
                r_inc = _scalar(cfun(a)) + g.Pi
                r_ent = _scalar(cfunE(a)) + g.PiE
                g_tilde_j = gpriorfun_scalar(g.beta + a * (1 - g.beta))
                if r_inc > r_ent:
                    # Entrant has cost advantage — total demand is D(r_ent),
                    # incumbent supplies D(r_inc), entrant fills the gap
                    wE_cim[j] = ((_scalar(dfun(r_ent)) - _scalar(dfun(r_inc))) *
                                 (1 - g.beta) * g_tilde_j)
            if np.any(wE_cim > 0):
                ax4.plot(cim_entry_al, wE_cim, 'b-', lw=1, alpha=0.7)

        # WNSE atom at alpha=0 (all non-selectives have precision 0;
        # the phi calculation determines how this capital is allocated
        # across markets, but the lenders themselves sit at alpha=0)
        if WNSE > 0:
            ax4.plot([0, 0], [0, WNSE], 'g-', lw=2)
            ax4.plot(0, WNSE, 'go', markersize=8,
                     markerfacecolor='green', markeredgewidth=2,
                     label=f'NS entry = {WNSE:.4f}')
            ax4.plot(0, 0, 'go', markersize=8,
                     markerfacecolor='white', markeredgewidth=2)

        # Discrete: wmassE / Delta (convert mass to density)
        disc_alphas = almassE[:-1] if len(almassE) > 1 else almassE
        disc_density = wmassE / g.Delta if g.Delta > 0 else wmassE
        if len(disc_alphas) == len(disc_density):
            ax4.scatter(disc_alphas, disc_density, s=8, c='k', zorder=3,
                        label='discrete $w^E/\\Delta$')
        # Threshold lines (no legend labels)
        ax4.axvline(g.alpha0, color='red', ls=':', lw=0.8, alpha=0.5)
        ax4.axvline(g.alpha1, color='red', ls=':', lw=0.8, alpha=0.5)
        ax4.axvline(g.alpha2, color='red', ls=':', lw=0.8, alpha=0.5)
        ax4.axvline(g.alpha0E, color='blue', ls=':', lw=0.8, alpha=0.5)
        ax4.axvline(g.alpha1E, color='blue', ls=':', lw=0.8, alpha=0.5)
        ax4.axvline(g.alpha2E, color='blue', ls=':', lw=0.8, alpha=0.5)

        ax4.set_xlabel(r'$\alpha$')
        ax4.set_title(r'Entry $w^E(\alpha)$: discrete vs analytical')
        ax4.legend(fontsize=6, ncol=2)
        ax4.grid(alpha=0.3)
        # Set ylim to exclude boundary spike (last 1-2 grid points artifact)
        all_wE = np.concatenate([ea['wE'], ea['w_incumbent']])
        ylim_top = np.percentile(all_wE[all_wE > 0], 98) * 1.3 if np.any(all_wE > 0) else 1.0
        ax4.set_ylim([0, max(ylim_top, 0.5)])

        # Threshold text labels at top of panel
        yl = ax4.get_ylim()
        yt = yl[1] - 0.02 * (yl[1] - yl[0])
        nudge = 0.008
        ax4.text(g.alpha0 + nudge, yt, r'$\alpha_0$', ha='left', va='top',
                 fontsize=8, color='red')
        ax4.text(g.alpha1 + nudge, yt, r'$\alpha_1$', ha='left', va='top',
                 fontsize=8, color='red')
        ax4.text(g.alpha2 + nudge, yt, r'$\alpha_2$', ha='left', va='top',
                 fontsize=8, color='red')
        ax4.text(g.alpha0E - nudge, yt, r'$\alpha_0^E$', ha='right', va='top',
                 fontsize=8, color='blue')
        ax4.text(g.alpha1E - nudge, yt, r'$\alpha_1^E$', ha='right', va='top',
                 fontsize=8, color='blue')
        ax4.text(g.alpha2E - nudge, yt, r'$\alpha_2^E$', ha='right', va='top',
                 fontsize=8, color='blue')
    else:
        ax4.set_visible(False)

    plt.tight_layout()
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _fig_path = os.path.join(_script_dir, 'mainE_results.png')
    plt.savefig(_fig_path, dpi=150)
    print(f"  Figure saved to {_fig_path}")
    plt.close(fig)

    print("\nEntry computation complete.")
    return almassE, wmassE, gammaE, WE


# =============================================================================
# Load baseline from MATLAB .mat file
# =============================================================================

def load_baseline_mat(mat_path=None):
    """Load baseline from MATLAB baseline.mat instead of recomputing.

    This ensures mainE starts from exactly the same data as the MATLAB code.
    """
    import scipy.io
    if mat_path is None:
        mat_path = (r"d:\Dropbox\projects-Dropbox\The-pablo-project"
                    r"\maryampabloprojectmynotes\matlab\secondversion\baseline.mat")

    print("=" * 60)
    print(f"Loading baseline from {mat_path}")
    print("=" * 60)

    d = scipy.io.loadmat(mat_path)

    g.Pi = d['Pi'].item()
    g.beta = d['beta'].item()
    g.BperG = d.get('BperG', np.array(1.0)).item()
    g.alpha0 = d['alpha0'].item()
    g.rp = d['rp'].item()
    g.alpha1 = d['alpha1'].item()
    g.alpha2 = d['alpha2'].item()
    g.Delta = d['Delta'].item()
    g.delom = d['delom'].item()
    g.omvec = d['omvec'].ravel()
    g.almass = d['almass'].ravel()
    g.wmass = d['wmass'].ravel()
    g.badleftover = d['badleftover'].item()
    g.WNS = d['WNS'].item()
    g.gfunprev = d['gfunprev'].ravel()
    g.bfunprev = d['bfunprev'].ravel()

    # Recompute W (cumulative wealth) to match run_baseline output
    cim_alpha = np.linspace(g.alpha1, g.alpha2, 100)
    g.W = np.cumsum(np.concatenate([[g.WNS], g.wmass, wcim(cim_alpha)]))

    print(f"  Pi={g.Pi}, beta={g.beta}")
    print(f"  alpha0 = {g.alpha0:.6f}")
    print(f"  rp     = {g.rp:.6f}")
    print(f"  alpha1 = {g.alpha1:.6f}")
    print(f"  alpha2 = {g.alpha2:.6f}")
    print(f"  WNS    = {g.WNS:.6f}")
    print(f"  badleftover = {g.badleftover:.6f}")
    print(f"  almass: {len(g.almass)} points, wmass: {len(g.wmass)} points")
    print("Baseline loaded.\n")


# =============================================================================
# Convergence test: discrete vs analytical at different Delta
# =============================================================================

def convergence_test(deltas=None):
    """Run discrete entry at several Delta values and compare to analytical.

    The analytical result is Delta-independent (continuous limit).
    If the discrete converges toward the analytical as Delta → 0,
    the T^E formula is confirmed correct.
    """
    if deltas is None:
        deltas = [0.004, 0.002, 0.001, 0.0005]

    results = []
    ana_WE = None

    for delta in deltas:
        print(f"\n{'='*60}")
        print(f"  Delta = {delta}")
        print(f"{'='*60}")

        # Reset global state
        g.__dict__.clear()

        # Override Delta in run_baseline
        run_baseline()
        g.Delta = delta
        # Rebuild almass/wmass at this Delta
        almass_pts = np.arange(g.alpha0, g.alpha1, g.Delta)
        if len(almass_pts) == 0:
            almass_pts = np.array([g.alpha0])
        w_at_almass = np.interp(almass_pts, g.baseline_alphas_fine,
                                g.baseline_w_fine, left=0, right=0)
        g.almass = almass_pts
        g.wmass = w_at_almass * g.Delta

        # Rebuild gfunprev/bfunprev
        beta = g.beta
        gd = gpriorfun(g.omvec).copy()
        bd = bpriorfun(g.omvec).copy()
        D_rp = _scalar(dfun(g.rp))
        for k in range(len(g.almass)):
            mask_g_k = g.omvec <= (beta + g.almass[k] * (1 - beta))
            mask_b_k = g.omvec >= (1 - beta + g.almass[k] * beta)
            raw_denom = (np.sum(g.delom * gd * mask_g_k) +
                         np.sum(g.delom * bd * mask_b_k))
            denom_k = raw_denom * D_rp
            if denom_k > 0:
                factor = g.wmass[k] / denom_k
                gd = gd * (1 - mask_g_k * factor)
                bd = bd * (1 - mask_b_k * factor)
        g.gfunprev = gd
        g.bfunprev = bd

        cim_alpha = np.linspace(g.alpha1, g.alpha2, 100)
        g.W = np.cumsum(np.concatenate([[g.WNS], g.wmass, wcim(cim_alpha)]))

        # Run entry
        almassE, wmassE, gammaE, WE = run_mainE()

        WE_disc = np.sum(wmassE)
        if g.entry_analytical is not None:
            WE_ana = g.entry_analytical['WE_cumsum'][-1]
            if ana_WE is None:
                ana_WE = WE_ana
        else:
            WE_ana = None

        results.append({
            'delta': delta,
            'WE_disc': WE_disc,
            'WE_ana': WE_ana,
            'rpE': g.rpE,
            'alpha0E': g.alpha0E,
            'alpha1E': g.alpha1E,
        })

    # Summary table
    print(f"\n{'='*60}")
    print("  CONVERGENCE TEST SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Delta':>8s}  {'W^E disc':>12s}  {'W^E ana':>12s}  {'Diff':>12s}  {'Rel %':>8s}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*8}")
    for r in results:
        diff = r['WE_ana'] - r['WE_disc'] if r['WE_ana'] is not None else float('nan')
        rel = 100 * diff / r['WE_ana'] if r['WE_ana'] and r['WE_ana'] > 0 else float('nan')
        print(f"  {r['delta']:>8.4f}  {r['WE_disc']:>12.6f}  "
              f"{r['WE_ana'] if r['WE_ana'] is not None else 'N/A':>12}  "
              f"{diff:>12.6f}  {rel:>7.2f}%")
    print(f"\n  Analytical W^E (Delta-independent) = {ana_WE:.6f}")

    # Convergence plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ds = [r['delta'] for r in results]
    we_d = [r['WE_disc'] for r in results]
    ax.plot(ds, we_d, 'ko-', lw=2, label='Discrete $W^E$')
    if ana_WE is not None:
        ax.axhline(ana_WE, color='b', ls='--', lw=1.5, label=f'Analytical $W^E$ = {ana_WE:.4f}')
    ax.set_xlabel(r'$\Delta$')
    ax.set_ylabel(r'$W^E$ (pooling region)')
    ax.set_title('Convergence: Discrete → Analytical as $\\Delta \\to 0$')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig('convergence_test.png', dpi=150)
    plt.close()

    return results


# =============================================================================
# Public API for figure-driver scripts
# =============================================================================

def _build_rate_curve_piecewise(rate_fn, alpha0, alpha1, alpha2,
                                extra_breakpoints=(), n=400):
    """Construct a rate curve r(α) with NaN sentinels at each discontinuity.

    Returns (alphas, rates) where matplotlib will draw separate segments
    separated by NaN gaps at every breakpoint.  Breakpoints are the union of
    (alpha0, alpha1, alpha2, 1.0) and any `extra_breakpoints` (e.g. the
    incumbent's alpha1/alpha2 when those differ from the entrant's).

    rate_fn(alpha_array, alpha0, alpha1, alpha2) — vectorized rate function
    matching the rfun / rfunE signature.
    """
    eps = 1e-4   # large enough to create a visible gap (~1% of α-range)
    # Collect all breakpoints, dedupe, sort, drop those outside [alpha0, 1].
    raw = [alpha0, alpha1, alpha2, 1.0] + [b for b in extra_breakpoints]
    raw = [b for b in raw if alpha0 - 1e-9 <= b <= 1.0 + 1e-9]
    bps = sorted(set(round(b, 8) for b in raw))
    # Build a segment between each consecutive pair, NaN-separated.
    # We sample the OPEN interval (bps[i], bps[i+1]) — i.e. shifted slightly past
    # the breakpoint on both sides — so that the segment carries the value of
    # the relevant region (not the boundary case which often belongs to the
    # *previous* region under rfun/rfunE's `<=` branching).
    segs_a, segs_r = [], []
    for i in range(len(bps) - 1):
        lo = bps[i] + (eps if i > 0 else 0.0)   # leftmost segment starts AT alpha0
        hi = bps[i + 1] - eps if i + 1 < len(bps) - 1 else bps[i + 1]
        if hi <= lo:
            continue
        n_pts = max(2, int(n * (hi - lo)))
        a = np.linspace(lo, hi, n_pts)
        r = rate_fn(a, alpha0, alpha1, alpha2)
        segs_a.append(a)
        segs_r.append(r)
    nan = np.array([np.nan])
    alphas = np.concatenate([s for pair in zip(segs_a, [nan] * len(segs_a))
                              for s in pair][:-1])  # drop trailing NaN
    rates = np.concatenate([s for pair in zip(segs_r, [nan] * len(segs_r))
                             for s in pair][:-1])
    return alphas, rates


def _build_omega_curve_from_alpha(r_alphas, r_rates, beta,
                                   alpha0_eff, r_pooling, alpha2_eff, r_NS,
                                   n_extra=80, eps=1e-4):
    """Map an α-axis rate curve to ω-axis via ω = ω_g(α) = β + α(1−β).

    Inputs:
        r_alphas, r_rates: existing α-axis curve (may contain NaN sentinels).
        alpha0_eff, alpha2_eff: skill thresholds used to extend the curve.
        r_pooling: rate for ω < ω_g(alpha0_eff)   (Region I extension).
        r_NS:      rate for ω > ω_g(alpha2_eff)   (Region III extension).
    Output:
        (omegas, rates) — r as a function of ω ∈ [0, 1].
            • flat at r_pooling on the left, until ω = ω_g(alpha0_eff);
            • mapped α-curve on the middle, with the original NaN sentinels
              preserved at α-discontinuities;
            • flat at r_NS on the right, from ω = ω_g(alpha2_eff) to 1.
        NaN sentinels are inserted at the join points so matplotlib breaks
        the line cleanly at any discontinuity.
    """
    omg_low = beta + alpha0_eff * (1 - beta)
    omg_high = beta + alpha2_eff * (1 - beta)

    # Left extension: ω in [0, ω_g(α₀_eff) - eps], flat at r_pooling.
    n_left = max(2, int(n_extra * omg_low))
    om_left = np.linspace(0.0, max(omg_low - eps, 0.0), n_left)
    r_left = np.full(len(om_left), r_pooling)

    # Right extension: ω in [ω_g(α₂_eff) + eps, 1], flat at r_NS.
    n_right = max(2, int(n_extra * (1 - omg_high)))
    om_right = np.linspace(min(omg_high + eps, 1.0), 1.0, n_right)
    r_right = np.full(len(om_right), r_NS)

    # Middle: map α-curve to ω-curve.
    om_mid = beta + r_alphas * (1 - beta)
    r_mid = np.asarray(r_rates, dtype=float).copy()

    nan = np.array([np.nan])
    omegas = np.concatenate([om_left, nan, om_mid, nan, om_right])
    rates = np.concatenate([r_left, nan, r_mid, nan, r_right])
    return omegas, rates


def solve_for_config(name):
    """Run incumbent (and entry, if configured) solve for a named config.

    Returns a dict with everything the figure-driver scripts need to plot:
      'config'       : the config dict
      'alpha0', 'alpha1', 'alpha2' : incumbent skill thresholds
      'rp'           : Region-I pooling rate
      'alphas_fine'  : fine α grid in Region I
      'w_inc_fine'   : incumbent entry density on the fine grid
      'gamma_inc'    : γ(α, 1, rp) on the fine grid
      'K_inc_fine'   : K(α) = Π + C(α) on the fine grid
      'has_entry'    : bool — whether the entry block was solved
      Entry-only (present when has_entry is True):
        'alpha0E', 'alpha1E', 'alpha2E', 'rpE', 'rnsE'
        'K_E_fine' : K^E(α) = Π^E + C^E(α) on the fine grid
        'wE'       : entry mass profile (from discrete solve)
        'almassE'  : α-grid for the discrete entry solve
    """
    global ACTIVE_CONFIG
    ACTIVE_CONFIG = name
    cfg = PARAM_CONFIGS[name]

    # Reset cached PCHIP interpolants between configs
    g._cfunE_pchip = None

    run_baseline()
    alphas_plot = np.linspace(0.0, 1.0, 1000)
    # Build the incumbent rate curve piecewise with NaN gaps at α₁ and α₂.
    r_inc_alphas, r_inc_plot = _build_rate_curve_piecewise(
        rfun, g.alpha0, g.alpha1, g.alpha2)
    # Same curve in ω-space: r(ω) over ω ∈ [0,1] with pooling-flat prefix
    # and NS-flat suffix.
    r_NS_inc = _scalar(cfun(g.alpha2)) + g.Pi  # baseline NS rate = K(α₂)
    r_inc_omegas, r_inc_omega_rates = _build_omega_curve_from_alpha(
        r_inc_alphas, r_inc_plot, g.beta, g.alpha0, g.rp, g.alpha2, r_NS_inc)
    K_inc_fine = np.array([_scalar(cfun(a)) for a in g.baseline_alphas_fine]) + g.Pi
    K_inc_plot = np.array([_scalar(cfun(a)) for a in alphas_plot]) + g.Pi

    result = {
        'config': cfg,
        'config_name': name,
        'alpha0': g.alpha0,
        'alpha1': g.alpha1,
        'alpha2': g.alpha2,
        'rp': g.rp,
        'rNS_inc': _scalar(cfun(g.alpha2)) + g.Pi if g.alpha2 < 1 else None,
        'alphas_fine': g.baseline_alphas_fine.copy(),
        'w_inc_fine': g.baseline_w_fine.copy(),
        'gamma_inc': g.gamma_inc_fine.copy(),
        'K_inc_fine': K_inc_fine,
        'alphas_plot': alphas_plot,
        'r_inc_alphas': r_inc_alphas,
        'r_inc_plot': r_inc_plot,
        'r_inc_omegas': r_inc_omegas,
        'r_inc_omega_rates': r_inc_omega_rates,
        'K_inc_plot': K_inc_plot,
        'has_entry': cfg.get('has_entry', True),
    }

    if cfg.get('has_entry', True):
        almassE, wmassE, gammaE, WE = run_mainE()
        K_E_fine = np.array([_scalar(cfunE(a)) for a in g.baseline_alphas_fine]) + g.PiE
        K_E_plot = np.array([_scalar(cfunE(a)) for a in alphas_plot]) + g.PiE
        # The entrant rate function has internal thresholds at the incumbent's
        # alpha1 and alpha2 (where the lower-envelope rule and the Region-IIb
        # rtildeafunE come in), in addition to alpha0E/alpha1E/alpha2E.
        # Pass these as extra breakpoints so every discontinuity gets a gap.
        r_E_alphas, r_E_plot = _build_rate_curve_piecewise(
            rfunE, max(g.alpha0E, g.alpha0), g.alpha1E, g.alpha2E,
            extra_breakpoints=(g.alpha1, g.alpha2))
        # ω-axis version of the entrant rate.
        # The "top" α for the entrant is max(alpha2E, alpha2) — covers both the
        # CIM-extended-right case (fig8) and the Region-IIb case (fig9a).
        alpha2_eff = max(g.alpha2E, g.alpha2)
        r_E_omegas, r_E_omega_rates = _build_omega_curve_from_alpha(
            r_E_alphas, r_E_plot, g.beta,
            max(g.alpha0E, g.alpha0), g.rpE,
            alpha2_eff, g.rnsE)
        result.update({
            'alpha0E': g.alpha0E,
            'alpha1E': g.alpha1E,
            'alpha2E': getattr(g, 'alpha2E', None),
            'rpE': g.rpE,
            'rnsE': getattr(g, 'rnsE', None),
            'K_E_fine': K_E_fine,
            'K_E_plot': K_E_plot,
            'r_E_alphas': r_E_alphas,
            'r_E_plot': r_E_plot,
            'r_E_omegas': r_E_omegas,
            'r_E_omega_rates': r_E_omega_rates,
            'almassE': almassE,
            'wE': wmassE,
            'gammaE': gammaE,
            'WE': WE,
            'PiE': g.PiE,
        })
    return result


# =============================================================================
# Main execution
# =============================================================================

if __name__ == '__main__':
    import sys
    if '--convergence' in sys.argv:
        convergence_test()
    elif '--load-mat' in sys.argv:
        load_baseline_mat()
        almassE, wmassE, gammaE, WE = run_mainE()
    else:
        # Allow --config <name> to override the default ACTIVE_CONFIG.
        if '--config' in sys.argv:
            idx = sys.argv.index('--config')
            if idx + 1 >= len(sys.argv):
                raise SystemExit("--config requires a config name")
            ACTIVE_CONFIG = sys.argv[idx + 1]
            if ACTIVE_CONFIG not in PARAM_CONFIGS:
                raise SystemExit(
                    f"Unknown config {ACTIVE_CONFIG!r}. "
                    f"Available: {list(PARAM_CONFIGS.keys())}")
            print(f"Using config: {ACTIVE_CONFIG}")
            print(f"  {PARAM_CONFIGS[ACTIVE_CONFIG]['description']}")
        run_baseline()
        if PARAM_CONFIGS[ACTIVE_CONFIG].get('has_entry', True):
            almassE, wmassE, gammaE, WE = run_mainE()
        else:
            print("[has_entry=False — skipping entry-equilibrium solve]")
