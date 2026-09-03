"""Shared plotting helpers used by the per-figure scripts.

Four panels are available:

    r_alpha    r(alpha) on active support [alpha0, alpha2], with an atom
               marker at alpha=0 for NS.  No flat extension above alpha2.
    r_omega    r(omega) over omega in [0, 1] with pooling flat prefix and
               r_NS flat suffix.
    K_alpha    K(alpha) = Pi + C(alpha) over [0, 1].
    w_alpha    w(alpha), the capital density across the skill axis, plus the
               non-selective atom at alpha=0.

Each comes in two forms:

    panel_<name>(specs, title, filename)   one PNG, one panel.  Used for the
        zoom panels and for figures 7/9a/9b, which the paper still sets as
        separate images.  Pass title=None for an untitled panel.
    _draw_<name>(ax, specs, ...)           draw into an axis you own.

    panel_grid(specs, filename)            all four as one 2x2 PNG, laid out
        and headed as in Notes/four_panel_hockey_stick.pdf.  This is what
        figures 8/10/11 in the numerical appendix use.

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


def _spec_style(spec, labels=None):
    """Resolve (kind, label, color, linestyle) for one curve spec.

    `labels`, when given, is a {'incumbent': str, 'entrant': str} mapping that
    overrides the spec's own label — used by panel_grid so each panel can name
    the two curves in its own terms ("Incumbents"/"Entrants" on the cost panel,
    "Baseline"/"With entrants" on the rate panels).
    """
    res = spec['res']
    kind = spec.get('kind',
                    'entrant' if (spec.get('use_entrant') and res.get('has_entry'))
                    else 'incumbent')
    label = spec.get('label', f"{res['config_name']} ({kind})")
    if labels is not None and kind in labels:
        label = labels[kind]
    color = spec.get('color', 'g' if kind == 'incumbent' else 'b')
    ls = spec.get('linestyle', '--' if kind == 'incumbent' else '-')
    return res, kind, label, color, ls


def _draw_r_alpha(ax, specs, labels=None):
    """r(alpha) into an existing axis.

    Each spec is a dict with at least 'res' (the solve_for_config return),
    plus optional 'label', 'color', 'linestyle', 'kind' ('incumbent' or
    'entrant'). 'kind' defaults to 'entrant' for any spec that has entry
    data, otherwise 'incumbent'.
    """
    for spec in specs:
        res, kind, label, color, ls = _spec_style(spec, labels)

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
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='best')


def panel_r_alpha(specs, title, filename):
    """Standalone r(alpha) PNG.  `title` may be None for an untitled panel."""
    fig, ax = plt.subplots(figsize=(7, 5))
    _draw_r_alpha(ax, specs)
    if title:
        ax.set_title(title)
    _save(fig, filename)


def _draw_r_omega(ax, specs, annotations=None, xlim=None, ylim=None,
                  labels=None):
    """r(omega) into an existing axis — same spec format as _draw_r_alpha.

    xlim / ylim (optional) crop the axes for zoom panels; the annotation
    label strip adapts to the cropped y-range.

    annotations (optional) is a dict:
        'vlines':  list of (omega_value, label) tuples — vertical dashed lines.
        'regions': list of (omega_center, label) tuples — region labels along
                    the top of the panel.
        'callouts': list of (omega_target, label, omega_text) tuples — region
                    labels placed at omega_text with an arrow pointing at
                    omega_target.  For regions too narrow to label in place
                    (e.g. a sliver-thin Region IIb).
    """
    for spec in specs:
        res, kind, label, color, ls = _spec_style(spec, labels)

        if kind == 'incumbent':
            ax.plot(res['r_inc_omegas'], res['r_inc_omega_rates'],
                    color=color, linestyle=ls, lw=2, label=label)
        else:
            ax.plot(res['r_E_omegas'], res['r_E_omega_rates'],
                    color=color, linestyle=ls, lw=2, label=label)

    ax.set_xlabel(r'$\omega$'); ax.set_ylabel(r'$r(\omega)$')
    ax.set_xlim(*(xlim if xlim is not None else (0, 1)))
    if ylim is not None:
        ax.set_ylim(*ylim)
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
        if xlim is not None:
            # set_xticks with ticks outside the view resets the limits
            # (observed on matplotlib 3.11); re-apply the zoom crop.
            ax.set_xlim(*xlim)
        for omega_center, label in annotations.get('regions', []):
            ax.text(omega_center, label_y, label,
                    ha='center', va='center', fontsize=11, color='dimgray',
                    fontweight='bold')
        for omega_target, label, omega_text in annotations.get('callouts', []):
            ax.annotate(label,
                        xy=(omega_target, label_y - 0.055 * yrange),
                        xytext=(omega_text, label_y),
                        ha='center', va='center', fontsize=11, color='dimgray',
                        fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='dimgray', lw=1.0))


def panel_r_omega(specs, title, filename, annotations=None, xlim=None, ylim=None):
    """Standalone r(omega) PNG.  `title` may be None for an untitled panel."""
    fig, ax = plt.subplots(figsize=(7, 5))
    _draw_r_omega(ax, specs, annotations=annotations, xlim=xlim, ylim=ylim)
    if title:
        ax.set_title(title)
    _save(fig, filename)


def _draw_w_alpha(ax, specs, xlim=None, ylim=None, logy=True,
                  mark_thresholds=True, labels=None, legend_fontsize=8):
    """w(alpha) into an existing axis — the density of lending capital.

    Same spec format as _draw_r_alpha.  The companion to the r(alpha) panel:
    that one shows the price in each market, this one shows how much capital
    sits there, before entry (incumbent) and after (post-entry total).

    Two dimensionally different objects share the picture, so they get
    separate axes:

      • left axis  — the DENSITY w(alpha) on the selective range, capital per
        unit of skill.  It spans two orders of magnitude (the marginal lender
        at alpha_0 holds far more than the CIM markets at the top), so the
        default is a log scale; pass logy=False for a linear one.
      • right axis — the ATOM W^NS at alpha = 0, the capital of the
        non-selective lenders, a mass and not a density.  Drawn as a stem with
        a filled marker at its top, the incumbent's at alpha = 0 and the
        post-entry one nudged right so the two do not overlap.

    mark_thresholds draws thin dotted verticals at the incumbent's
    alpha_0/alpha_1/alpha_2 and the entrant's alpha_0^E/alpha_2^E, in each
    curve's own color, labelled along the top in two rows (incumbent above,
    entrant below) so a near-coincident pair does not overprint.
    """
    ax_atom = ax.twinx()

    atoms = []          # (x_offset, mass, color, label)
    thresholds = []     # (alpha, latex_label, color, ha_side, label_row)
    for spec in specs:
        res, kind, label, color, ls = _spec_style(spec, labels)

        # The two densities coincide over most of the range in the weak-effect
        # calibrations (10a), so the incumbent is drawn thick underneath and
        # the post-entry curve thin on top: where they agree, the green dashes
        # show through as a halo instead of vanishing under the blue.
        if kind == 'incumbent':
            ax.plot(res['w_inc_alphas'], res['w_inc_plot'],
                    color=color, linestyle=ls, lw=spec.get('lw', 3.2),
                    label=label)
            if res.get('WNS_inc') is not None:
                atoms.append((0.0, res['WNS_inc'], color, label))
            thresholds += [(res['alpha0'], r'$\alpha_0$', color, 'left', 0),
                           (res['alpha1'], r'$\alpha_1$', color, 'left', 0),
                           (res['alpha2'], r'$\alpha_2$', color, 'left', 0)]
        else:
            ax.plot(res['w_E_alphas'], res['w_E_plot'],
                    color=color, linestyle=ls, lw=spec.get('lw', 1.7),
                    label=label)
            if res.get('WNS_E') is not None:
                atoms.append((0.012, res['WNS_E'], color, label))
            # Row 1, so an α₂^E close to α₂ does not overprint it.
            thresholds += [(res['alpha0E'], r'$\alpha_0^E$', color, 'right', 1),
                           (res['alpha2E'], r'$\alpha_2^E$', color, 'right', 1)]

    if logy:
        ax.set_yscale('log')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$w(\alpha)$' + ('  (log scale)' if logy else ''))
    ax.set_xlim(*(xlim if xlim is not None else (-0.02, 1.0)))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)

    # --- the alpha = 0 atoms, on their own scale ------------------------
    atom_handles = []
    for x, mass, color, lbl in atoms:
        ax_atom.plot([x, x], [0.0, mass], color=color, lw=2, alpha=0.8)
        h, = ax_atom.plot(x, mass, color=color, marker='o', linestyle='',
                          markersize=8,
                          # 4 decimals: in the weak-effect calibrations the
                          # whole NS effect lives in the 4th decimal.
                          label=f'{lbl} $W^{{NS}}={mass:.4f}$')
        atom_handles.append(h)
    if atoms:
        ax_atom.set_ylim(0.0, 1.45 * max(m for _, m, _, _ in atoms))
        ax_atom.set_ylabel(r'$W^{NS}$  (atom at $\alpha=0$, right axis)',
                           fontsize=9)
    else:
        ax_atom.set_visible(False)

    # --- threshold markers ----------------------------------------------
    if mark_thresholds and thresholds:
        ymin, ymax = ax.get_ylim()
        # Log axes need a multiplicative headroom for the two label rows.
        if logy:
            ax.set_ylim(ymin, ymax * 3.0)
            label_y = [ymax * 2.3, ymax * 1.25]
        else:
            ax.set_ylim(ymin, ymax + 0.22 * (ymax - ymin))
            span = ymax - ymin
            label_y = [ymax + 0.16 * span, ymax + 0.06 * span]
        seen = {0: [], 1: []}
        for alpha, label, color, side, row in thresholds:
            ax.axvline(alpha, color=color, ls=':', lw=0.8, alpha=0.55)
            # Suppress a label that would land on one already placed in its row.
            if any(abs(alpha - a) < 0.02 for a in seen[row]):
                continue
            seen[row].append(alpha)
            nudge = 0.008 if side == 'left' else -0.008
            ax.text(alpha + nudge, label_y[row], label,
                    ha='left' if side == 'left' else 'right', va='top',
                    fontsize=8, color=color)

    # One legend for both axes: the density lines then the atom markers.
    line_handles, line_labels = ax.get_legend_handles_labels()
    # Centre right: w is supported on [alpha_0, alpha_2] and alpha_2 < 0.7 in
    # every calibration, so the right third of the panel is always empty,
    # while the top strip belongs to the α labels and the lower left to the
    # steeply falling curve itself.
    ax.legend(line_handles + atom_handles,
              line_labels + [h.get_label() for h in atom_handles],
              fontsize=legend_fontsize, loc='center right')


def panel_w_alpha(specs, title, filename, xlim=None, ylim=None,
                  logy=True, mark_thresholds=True):
    """Standalone w(alpha) PNG.  `title` may be None for an untitled panel."""
    fig, ax = plt.subplots(figsize=(7, 5))
    _draw_w_alpha(ax, specs, xlim=xlim, ylim=ylim, logy=logy,
                  mark_thresholds=mark_thresholds)
    if title:
        ax.set_title(title)
    _save(fig, filename)


def _draw_K_alpha(ax, specs, xlim=None, ylim=None, labels=None):
    """K(alpha) into an existing axis.  xlim/ylim crop for zoom panels."""
    for spec in specs:
        res, kind, label, color, ls = _spec_style(spec, labels)

        if kind == 'incumbent':
            ax.plot(res['alphas_plot'], res['K_inc_plot'],
                    color=color, linestyle=ls, lw=2, label=label)
        else:
            ax.plot(res['alphas_plot'], res['K_E_plot'],
                    color=color, linestyle=ls, lw=2, label=label)

    ax.set_xlabel(r'$\alpha$'); ax.set_ylabel(r'$K(\alpha)$')
    ax.set_xlim(*(xlim if xlim is not None else (0, 1)))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='best')


def panel_K_alpha(specs, title, filename, xlim=None, ylim=None):
    """Standalone K(alpha) PNG.  `title` may be None for an untitled panel."""
    fig, ax = plt.subplots(figsize=(7, 5))
    _draw_K_alpha(ax, specs, xlim=xlim, ylim=ylim)
    if title:
        ax.set_title(title)
    _save(fig, filename)


# Panel headings, in the wording and order of the schematic
# Notes/four_panel_hockey_stick.pdf.  Row-major: top row is the two primitives
# (what entry changes), bottom row the two equilibrium rate schedules.
GRID_TITLES = {
    'K': 'Cost function',
    'w': 'Density of lenders',
    'r_omega': 'Interest rate for good borrowers',
    'r_alpha': 'Interest rate chosen by lenders',
}

# The cost panel compares two cost FUNCTIONS, so its curves are the two lender
# types; the other three compare two EQUILIBRIA, so theirs are before/after.
GRID_LABELS = {
    'K': {'incumbent': 'Incumbents', 'entrant': 'Entrants'},
    'other': {'incumbent': 'Baseline', 'entrant': 'With entrants'},
}


def panel_grid(specs, filename, annotations=None, figsize=(13, 9),
               titles=None, w_kwargs=None):
    """The four main panels as one 2x2 figure, laid out as in the schematic.

        Cost function                    Density of lenders
        Interest rate for good borrowers Interest rate chosen by lenders

    The panels carry bold headings instead of per-figure titles, so the
    caption in the paper is the only place the example is named.  `titles`
    overrides GRID_TITLES for one-off wording; `w_kwargs` is passed through to
    the density panel (e.g. logy=False).
    """
    ttl = dict(GRID_TITLES, **(titles or {}))
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    _draw_K_alpha(axes[0, 0], specs, labels=GRID_LABELS['K'])
    _draw_w_alpha(axes[0, 1], specs, labels=GRID_LABELS['other'],
                  **(w_kwargs or {}))
    _draw_r_omega(axes[1, 0], specs, annotations=annotations,
                  labels=GRID_LABELS['other'])
    _draw_r_alpha(axes[1, 1], specs, labels=GRID_LABELS['other'])

    for ax, key in ((axes[0, 0], 'K'), (axes[0, 1], 'w'),
                    (axes[1, 0], 'r_omega'), (axes[1, 1], 'r_alpha')):
        ax.set_title(ttl[key], fontweight='bold', fontsize=13, pad=10)

    fig.tight_layout(w_pad=3.0, h_pad=3.0)
    _save(fig, filename)
