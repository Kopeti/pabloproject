"""Shared plotting helpers used by the per-figure scripts.

Each `make_figureN.py` builds three PNGs by calling these helpers:

    fig_r_alpha_<figure>.png    r(alpha) on active support [alpha0, alpha2],
                                with an atom marker at alpha=0 for NS. No
                                flat extension above alpha2.
    fig_r_omega_<figure>.png    r(omega) over omega in [0, 1] with pooling
                                flat prefix and r_NS flat suffix.
    fig_K_alpha_<figure>.png    K(alpha) = Pi + C(alpha) over [0, 1].

The helpers accept a list of "curve specs" so a single panel can overlay
multiple parametrizations (e.g. Figure 7 shows incumbent + SP-highPiE +
SP-lowPiE on the same r(omega) panel).
"""
import os
import numpy as np
import matplotlib.pyplot as plt


_HERE = os.path.dirname(os.path.abspath(__file__))
# Save the PNGs to the paper's figures folder, computed relative to this
# script so the path stays correct on any machine.  From
# .../The-pablo-project/maryampabloprojectmynotes/matlab/integratewithiid/pythonversion
# the paper figures folder is four levels up + Peter-Pablo-Maryam/figures.
_OUTPUT_DIR = os.path.abspath(
    os.path.join(_HERE, '..', '..', '..', '..', 'Peter-Pablo-Maryam', 'figures'))


def _trim_to_support(alphas, rates, alpha_upper):
    a = np.asarray(alphas); r = np.asarray(rates)
    mask = (a <= alpha_upper + 1e-9) | np.isnan(a)
    return a[mask], r[mask]


def _save(fig, name):
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    out = os.path.join(_OUTPUT_DIR, name)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


def panel_r_alpha(specs, title, filename):
    """r(alpha) panel.

    Each spec is a dict with at least 'res' (the solve_for_config return),
    plus optional 'label', 'color', 'linestyle', 'kind' ('incumbent' or
    'entrant'). 'kind' defaults to 'entrant' for any spec that has entry
    data, otherwise 'incumbent'.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for spec in specs:
        res = spec['res']
        kind = spec.get('kind',
                        'entrant' if (spec.get('use_entrant') and res.get('has_entry'))
                        else 'incumbent')
        label = spec.get('label', f"{res['config_name']} ({kind})")
        color = spec.get('color', 'g' if kind == 'incumbent' else 'b')
        ls = spec.get('linestyle', '--' if kind == 'incumbent' else '-')

        if kind == 'incumbent':
            a, r = _trim_to_support(res['r_inc_alphas'], res['r_inc_plot'],
                                    res['alpha2'])
            ax.plot(a, r, color=color, linestyle=ls, lw=2, label=label)
            if res.get('rNS_inc') is not None:
                ax.plot(0.0, res['rNS_inc'], color=color, marker='o',
                        linestyle='', markersize=8)
        else:
            a2_eff = max(res['alpha2'], res['alpha2E'])
            a, r = _trim_to_support(res['r_E_alphas'], res['r_E_plot'], a2_eff)
            ax.plot(a, r, color=color, linestyle=ls, lw=2, label=label)
            if res.get('rnsE') is not None:
                ax.plot(0.0, res['rnsE'], color=color, marker='o',
                        linestyle='', markersize=8)

    ax.set_xlabel(r'$\alpha$'); ax.set_ylabel(r'$r(\alpha)$')
    ax.set_xlim(-0.02, 1.0)
    ax.set_title(title)
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='best')
    _save(fig, filename)


def panel_r_omega(specs, title, filename):
    """r(omega) panel — same spec format as panel_r_alpha."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for spec in specs:
        res = spec['res']
        kind = spec.get('kind',
                        'entrant' if (spec.get('use_entrant') and res.get('has_entry'))
                        else 'incumbent')
        label = spec.get('label', f"{res['config_name']} ({kind})")
        color = spec.get('color', 'g' if kind == 'incumbent' else 'b')
        ls = spec.get('linestyle', '--' if kind == 'incumbent' else '-')

        if kind == 'incumbent':
            ax.plot(res['r_inc_omegas'], res['r_inc_omega_rates'],
                    color=color, linestyle=ls, lw=2, label=label)
        else:
            ax.plot(res['r_E_omegas'], res['r_E_omega_rates'],
                    color=color, linestyle=ls, lw=2, label=label)

    ax.set_xlabel(r'$\omega$'); ax.set_ylabel(r'$r(\omega)$')
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='best')
    _save(fig, filename)


def panel_K_alpha(specs, title, filename):
    """K(alpha) panel — same spec format."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for spec in specs:
        res = spec['res']
        kind = spec.get('kind',
                        'entrant' if (spec.get('use_entrant') and res.get('has_entry'))
                        else 'incumbent')
        label = spec.get('label', f"{res['config_name']} ({kind})")
        color = spec.get('color', 'g' if kind == 'incumbent' else 'b')
        ls = spec.get('linestyle', '--' if kind == 'incumbent' else '-')

        if kind == 'incumbent':
            ax.plot(res['alphas_plot'], res['K_inc_plot'],
                    color=color, linestyle=ls, lw=2, label=label)
        else:
            ax.plot(res['alphas_plot'], res['K_E_plot'],
                    color=color, linestyle=ls, lw=2, label=label)

    ax.set_xlabel(r'$\alpha$'); ax.set_ylabel(r'$K(\alpha)$')
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='best')
    _save(fig, filename)
