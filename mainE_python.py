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

import numpy as np
from scipy.optimize import minimize_scalar, fsolve, minimize, brentq
from scipy.integrate import quad
import matplotlib.pyplot as plt
from types import SimpleNamespace
import warnings

warnings.filterwarnings('ignore')


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
    """Cost function for incumbents. Matches cfun.m"""
    alpha = np.asarray(alpha, dtype=float)
    Ca = 9.0   # was 9, this moves alphas a lot
    Cb = 0.2
    Cpower = 2.0  # it was 2
    result = Ca * alpha**Cpower + Cb * alpha
    return result.item() if np.size(result) == 1 else result


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
    """Non-selective entry function for incumbents. Matches NSfun.m"""
    al = _scalar(al)
    goodleftover = quad(gpriorfun_scalar, g.beta + al * (1 - g.beta), 1)[0]
    gammaNS = goodleftover / (goodleftover + g.badleftover)
    return (gammaNS * (1 + _scalar(cfun(al)) + g.Pi) - (1 + g.Pi))**2


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


def cfunE(alpha):
    """
    Entry cost function for new entrants. Matches cfunE.m
    Uses globals: rp, Pi, PiE, alpha0, alpha1, kappa1
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    ca = np.zeros(len(alpha))

    maxa = g.alpha1
    maxc = 100.0
    kappa = 1.1
    alphabar = (g.alpha0 + g.alpha1) / 2.0

    for i in range(len(alpha)):
        a = alpha[i]
        gu = gammaupdate2(a, g.alpha0, g.rp)
        gu = _scalar(gu)

        # Base cost from incumbent equivalent
        inner_r = (g.Pi + 1 + _scalar(cfun(a))) / gu - 1
        base_cost = gu * (1 + _scalar(dfuninv(kappa * _scalar(dfun(inner_r))))) - (1 + g.PiE)

        # Distortion term
        distortion = g.kappa1 * a * (a - alphabar)

        # Nonlinear add-on (matches the boolean*min(...) pattern from MATLAB)
        if a < maxa:
            addon = 0.1 * min(0.1 * (1.0 / (maxa - a) - 1.0 / maxa), maxc)
        elif a > maxa:
            addon = 0.1 * (maxc + (a - maxa))
        else:
            addon = 0.0  # both boolean conditions false when a == maxa

        ca[i] = base_cost + distortion + addon

    return _scalar(ca[0]) if len(alpha) == 1 else ca


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
    """Run the baseline model. Equivalent to main.m"""
    print("=" * 60)
    print("Running baseline computation (main.m)")
    print("=" * 60)

    # Primitives — must match baseline.mat (Pi=0.2, beta=0.5)
    g.Pi = 0.2
    g.beta = 0.5
    g.BperG = 1.0  # proportion of bad to good

    # Finding alpha0 and rp
    res = minimize_scalar(
        lambda alpha: (g.Pi + _scalar(cfun(alpha)) + 1) / gam0(alpha),
        bounds=(0, 1), method='bounded'
    )
    g.alpha0 = res.x
    g.rp = (g.Pi + _scalar(cfun(g.alpha0)) + 1) / gam0(g.alpha0) - 1

    print(f"  alpha0 = {g.alpha0:.6f}")
    print(f"  rp     = {g.rp:.6f}")

    # Find alpha1
    if (1 + g.rp) - 1 - _scalar(cfun(1.0)) > g.Pi:
        g.alpha1 = 1.0
    else:
        g.alpha1 = fsolve(lambda alpha: _scalar(cfun(alpha)) - (g.rp - g.Pi), g.alpha0)[0]

    print(f"  alpha1 = {g.alpha1:.6f}")

    # Discretization
    g.Delta = 0.001
    g.delom = 0.0001
    g.omvec = np.linspace(0, 1, int(round(1 / g.delom)))

    # Initialize distributions
    g.gfunprev = gpriorfun(g.omvec).copy()
    g.bfunprev = bpriorfun(g.omvec).copy()

    # Initialize mass points
    almass_list = [g.alpha0]
    wmass_list = []

    wopt = 1.0
    n = 2

    # Store gnext, bnext
    gnext = None
    bnext = None

    print("  Iterating through pooling region...")

    while almass_list[-1] + g.Delta <= g.alpha1:
        if n >= 3:
            g.gfunprev = gnext.copy()
            g.bfunprev = bnext.copy()

        alp = almass_list[-1]
        g.alp = alp  # set global for gamcplx

        # Compute wmax: maximum w that keeps b non-negative
        mask_g_alp = g.omvec <= (g.beta + alp * (1 - g.beta))
        mask_b_alp = g.omvec >= (1 - g.beta + alp * g.beta)
        denom_alp = (np.sum(g.delom * g.gfunprev[mask_g_alp]) +
                     np.sum(g.delom * g.bfunprev[mask_b_alp]))
        dfun_rp = _scalar(dfun(g.rp))

        def bfunmin(w):
            return np.min(g.bfunprev * (1 - mask_b_alp * w / denom_alp / dfun_rp))

        wmax = fsolve(bfunmin, 1.0)[0]

        # Try no-hole solution first
        alopt3 = alp + g.Delta
        res3 = minimize_scalar(
            lambda w: 1000 * (profit(w, alopt3, alp) - (1 + g.Pi))**2,
            bounds=(0, wmax), method='bounded'
        )
        wopt3 = res3.x

        res3b = minimize_scalar(
            lambda alpha: 1000 * (_scalar(cfun(alpha)) - (1 + g.rp) * gamfun(wopt3, alpha, alp)),
            bounds=(alp + g.Delta, g.alpha1), method='bounded'
        )
        alopt3b = res3b.x

        if abs(alopt3 - alopt3b) < 10 * g.Delta:
            wopt = wopt3
            alopt = alopt3
        else:
            # Look for holes in support
            res_w = minimize_scalar(
                lambda w: 1000 * gamcplx(w),
                bounds=(0, wmax), method='bounded'
            )
            wopt = res_w.x

            res_al = minimize_scalar(
                lambda alpha: 1000 * (_scalar(cfun(alpha)) - (1 + g.rp) * gamfun(wopt, alpha, alp)),
                bounds=(alp + g.Delta, g.alpha1), method='bounded'
            )
            alopt = res_al.x

            # Try constrained optimization (fmincon equivalent)
            res2 = minimize(
                lambda w: 1000 * gamcplx(w[0]),
                [wopt], bounds=[(0, wmax)], method='L-BFGS-B'
            )
            wopt2 = res2.x[0]

            res_al2 = minimize_scalar(
                lambda alpha: 1000 * (_scalar(cfun(alpha)) - (1 + g.rp) * gamfun(wopt2, alpha, alp)),
                bounds=(alp + g.Delta, g.alpha1), method='bounded'
            )
            alopt2 = res_al2.x

            if abs(profit(wopt, alopt, alp) - (1 + g.Pi)) > abs(profit(wopt2, alopt2, alp) - (1 + g.Pi)):
                wopt = wopt2
                alopt = alopt2

        almass_list.append(alopt)

        # Update distributions
        gnext = g.gfunprev * (1 - mask_g_alp * wopt / denom_alp / dfun_rp)
        bnext = g.bfunprev * (1 - mask_b_alp * wopt / denom_alp / dfun_rp)

        wmass_list.append(wopt)
        n += 1

        if n % 50 == 0:
            print(f"    n={n}, alp={alp:.4f}")

    # Finalize arrays
    g.almass = np.array(almass_list)
    g.wmass = np.array(wmass_list)

    print(f"  Pooling region: {n-1} iterations")

    # CIM and NS segment
    g.badleftover = np.sum(g.delom * bnext)

    # Find alpha2
    res_a2 = minimize_scalar(lambda al: NSfun(al), bounds=(g.alpha1, 1), method='bounded')
    g.alpha2 = res_a2.x

    if NSfun(g.alpha2) < g.Delta:
        g.WNS = _scalar(dfun(cfun(g.alpha2) + g.Pi)) * (
            g.badleftover + quad(gpriorfun_scalar, g.beta + g.alpha2 * (1 - g.beta), 1)[0])
    else:
        g.WNS = 0.0
        g.alpha2 = 1.0

    # Cumulative wealth
    cim_alpha = np.linspace(g.alpha1, g.alpha2, 100)
    g.W = np.cumsum(np.concatenate([[g.WNS], g.wmass, wcim(cim_alpha)]))

    print(f"  alpha2       = {g.alpha2:.6f}")
    print(f"  WNS          = {g.WNS:.6f}")
    print(f"  badleftover  = {g.badleftover:.6f}")
    print("Baseline complete.\n")


# =============================================================================
# Entry computation (equivalent to mainE.m)
# =============================================================================

def run_mainE():
    """Run the entry model. Equivalent to mainE.m"""
    print("=" * 60)
    print("Running entry computation (mainE.m)")
    print("=" * 60)

    # ---------- Settings ----------
    g.kappa1 = -4.0   # positive: BLO up, negative: BLO goes down
    g.PiE = 0.1       # profit for new entrants

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
            g.rnsE = rprime
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
    WE = np.cumsum(np.concatenate([[WNSE], wmassE, wcimE(cim_alphasE)]))

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
        if maxentry < len(almassE):
            ax1.axvline(almassE[maxentry], color='b', linestyle=':', linewidth=0.8)
            ax1.text(almassE[maxentry], ax1.get_ylim()[1], 'last entry', color='b',
                     fontsize=8, ha='center', va='bottom')
    ax1.axvline(g.alpha2E, color='b', linestyle=':', linewidth=0.8)
    ax1.text(g.alpha2E, ax1.get_ylim()[1], r'$\alpha_2^E$', color='b', fontsize=8,
             ha='center', va='bottom')
    ax1.set_xlabel(r'$\alpha$')
    ax1.set_title('interest rate')
    ax1.legend(fontsize=8)

    # --- Subplot 2: Cumulative wealth ---
    ax2 = axes[0, 1]
    welength = len(WE) - len(almassE[:-1]) - 1
    x_new = np.concatenate([[0], almassE[:-1],
                            np.linspace(g.alpha1E, g.alpha2E, max(welength, 1))])
    # Trim or extend to match WE length
    if len(x_new) < len(WE):
        x_new = np.concatenate([x_new, np.linspace(g.alpha2E, 1, len(WE) - len(x_new))])
    x_new = x_new[:len(WE)]

    ax2.scatter(x_new, WE, s=3, label='new entrants')
    # Extend flat to the right
    ax2.scatter(np.linspace(g.alpha2E, 1, 100), WE[-1] * np.ones(100), s=3)
    ax2.set_ylim([0, 1.5])
    ax2.axvline(g.alpha0E, color='b', linestyle=':', linewidth=0.8)
    if maxentry < len(almassE):
        ax2.axvline(almassE[maxentry], color='b', linestyle=':', linewidth=0.8)
    ax2.set_xlabel(r'$\alpha$')
    ax2.set_title('cumulative wealth')

    # Baseline cumulative wealth
    x_base = np.concatenate([[0], g.almassshort[:-1],
                             np.linspace(g.alpha1, g.alpha2, 100)])
    x_base = x_base[:len(g.W)]
    ax2.scatter(x_base, g.W, s=3, c='r', label='incumbent')
    ax2.scatter(np.linspace(g.alpha2, 1, 10), g.W[-1] * np.ones(10), s=3, c='r')
    ax2.legend(fontsize=8)

    # --- Subplot 3: Cost + Pi ---
    ax3 = axes[1, 0]
    al_plot = np.linspace(0.01, 0.99, 100)  # avoid boundary issues
    cost_new = np.array([_scalar(cfunE(a)) for a in al_plot]) + g.PiE
    cost_base = cfun(al_plot) + g.Pi
    ax3.plot(al_plot, cost_new, label='new entrants')
    ax3.plot(al_plot, cost_base, label='incumbent')
    ax3.legend()
    ax3.set_title(r'cost + $\Pi$')
    ax3.set_xlabel(r'$\alpha$')

    # --- Subplot 4: (empty in original, was commented out) ---
    axes[1, 1].set_visible(False)

    plt.tight_layout()
    plt.savefig('mainE_results.png', dpi=150)
    plt.show()

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
# Main execution
# =============================================================================

if __name__ == '__main__':
    import sys
    if '--recompute-baseline' in sys.argv:
        run_baseline()
    else:
        load_baseline_mat()
    almassE, wmassE, gammaE, WE = run_mainE()
