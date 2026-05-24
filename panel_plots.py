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


def omega_g(alpha, beta):
    """ω_g(α) = β + α (1 − β) — opacity threshold for goods."""
    return beta + alpha * (1.0 - beta)


def find_omega_H(res):
    """Find ω_H = ω_g(α_H), where K^E first crosses K_inc inside the CIM
    region [α₁, α₂]. Returns None if there is no crossing in that interval.
    """
    if not res.get('has_entry'):
        return None
    alphas = res['alphas_plot']
    diff = res['K_E_plot'] - res['K_inc_plot']
    mask = (alphas >= res['alpha1']) & (alphas <= res['alpha2'])
    a_in = alphas[mask]
    d_in = diff[mask]
    # Locate the first sign change.
    for i in range(1, len(d_in)):
        if d_in[i - 1] * d_in[i] < 0:
            # Linear interp to the zero
            x = a_in[i - 1] - d_in[i - 1] * (a_in[i] - a_in[i - 1]) / (d_in[i] - d_in[i - 1])
            beta = res['config']['beta']
            return omega_g(x, beta)
    return None


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


def panel_r_omega(specs, title, filename, annotations=None):
    """r(omega) panel — same spec format as panel_r_alpha.

    annotations (optional) is a dict:
        'vlines':  list of (omega_value, label) tuples — vertical dashed lines.
        'regions': list of (omega_center, label) tuples — region labels along
                    the top of the panel.
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
            ax.plot(res['r_inc_omegas'], res['r_inc_omega_rates'],
                    color=color, linestyle=ls, lw=2, label=label)
        else:
            ax.plot(res['r_E_omegas'], res['r_E_omega_rates'],
                    color=color, linestyle=ls, lw=2, label=label)

    ax.set_xlabel(r'$\omega$'); ax.set_ylabel(r'$r(\omega)$')
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='best')

    if annotations:
        # Extend y-axis so the region labels live in a clear strip above
        # every rate curve (including the green dashed incumbent NS plateau
        # in region III).
        ymin, ymax = ax.get_ylim()
        yrange = ymax - ymin
        ymax_new = ymax + 0.14 * yrange
        ax.set_ylim(ymin, ymax_new)
        label_y = ymax + 0.06 * yrange  # above the curves' top, below new ymax

        for omega_value, _ in annotations.get('vlines', []):
            ax.axvline(omega_value, color='gray', ls=':', lw=0.8, alpha=0.7)
        # Replace the default x-axis ticks with the annotation positions
        # (plus 0 and 1), so the named ω labels do not collide with the
        # default 0.2 / 0.4 / ... labels.
        named_ticks = [(0.0, '0'), (1.0, '1')] + [
            (omega_value, label) for omega_value, label in annotations.get('vlines', [])]
        named_ticks.sort(key=lambda t: t[0])
        ax.set_xticks([t for t, _ in named_ticks])
        ax.set_xticklabels([l for _, l in named_ticks])
        for omega_center, label in annotations.get('regions', []):
            ax.text(omega_center, label_y, label,
                    ha='center', va='center', fontsize=11, color='dimgray',
                    fontweight='bold')

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
