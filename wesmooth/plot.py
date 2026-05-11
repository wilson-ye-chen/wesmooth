import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.stats import gaussian_kde


def quantile_to_density(q_vals, p_grid, n_eval=2000, eps=1e-6):
    """
    Convert a discretised quantile function Q(p) into a density curve via
        f(Q(p)) = 1 / Q'(p).
    """
    # Monotone interpolation of the quantile function
    q_func = PchipInterpolator(p_grid, q_vals)

    p_fine = np.linspace(p_grid[0], p_grid[-1], n_eval)
    x = q_func(p_fine)
    dqdp = q_func.derivative()(p_fine)

    # Density: f(Q(p)) = 1 / Q'(p)
    f = 1.0 / np.maximum(dqdp, eps)
    return x, f


def kde_to_density(vals, n_eval=2000, bw_method=None, pad_frac=0.05):
    """
    Convert sample values into a kernel density estimate.
    """
    vals = np.asarray(vals, dtype=float)

    if np.isnan(vals).any():
        raise ValueError('Missing values detected.')

    if np.all(vals == vals[0]):
        raise ValueError('KDE is not well-defined for constant values.')

    kde = gaussian_kde(vals, bw_method=bw_method)

    x_min = np.min(vals)
    x_max = np.max(vals)
    pad = pad_frac * (x_max - x_min)

    x = np.linspace(x_min - pad, x_max + pad, n_eval)
    f = kde(x)

    return x, f


def plot_density_path(
    y_path,
    p_grid=None,
    density_method='quantile',
    width_scale=1.2,
    n_eval=2000,
    normalise_width=True,
    draw_baseline=False,
    baseline_width=1.0,
    kde_bw_method=None,
    kde_pad_frac=0.05,
    facecolor='#cfcfcf',
    edgecolor='black',
    alpha=0.95,
    linewidth=1.0,
    ref_lines=None,
    xlabel='t',
    ylabel='x',
    title=None,
    figsize=(10, 4),
    xtick_step=1,
):
    """
    Plot a sequence of realised distributions by displaying each density
    function sideways from its time index.

    Parameters
    ----------
    y_path : ndarray, shape (T, m)
        Each row represents one distribution. For density_method='quantile',
        each row is interpreted as a discretised quantile function. For
        density_method='kde', each row is treated as sample values.
    p_grid : ndarray or None
        Probability grid for the quantile values. Required when
        density_method='quantile'. Ignored when density_method='kde'.
    density_method : {'quantile', 'kde'}
        Method used to estimate each density curve.
    kde_bw_method : str, scalar, callable or None
        Bandwidth method passed to scipy.stats.gaussian_kde.
    kde_pad_frac : float
        Fractional padding added to the KDE plotting range.
    """
    if density_method not in {'quantile', 'kde'}:
        raise ValueError("density_method must be either 'quantile' or 'kde'.")

    if density_method == 'quantile' and p_grid is None:
        raise ValueError("p_grid is required when density_method='quantile'.")

    T = y_path.shape[0]
    fig, ax = plt.subplots(figsize=figsize)

    # Draw from right to left so left densities can cover right neighbours
    for t in range(T - 1, -1, -1):
        if density_method == 'quantile':
            yy, ff = quantile_to_density(
                y_path[t],
                p_grid,
                n_eval=n_eval
            )
        else:
            yy, ff = kde_to_density(
                y_path[t],
                n_eval=n_eval,
                bw_method=kde_bw_method,
                pad_frac=kde_pad_frac
            )

        if normalise_width:
            ff_scaled = width_scale * ff / np.max(ff)
        else:
            ff_scaled = width_scale * ff

        center = t + 1
        xx = center + ff_scaled

        # Fill only: no boundary lines
        ax.fill_betweenx(
            yy,
            center,
            xx,
            facecolor=facecolor,
            edgecolor=None,
            linewidth=0,
            alpha=alpha,
            zorder=T - t
        )

        # Outer density curve
        ax.plot(
            xx,
            yy,
            color=edgecolor,
            linewidth=linewidth,
            zorder=T - t + 0.05
        )

        # Draw bottom border if requested
        if draw_baseline:
            ax.plot(
                [center, xx[0]],
                [yy[0], yy[0]],
                color=edgecolor,
                linewidth=baseline_width,
                zorder=T - t + 0.1
            )

    if ref_lines is not None:
        for yref in ref_lines:
            ax.axhline(
                yref,
                linestyle='--',
                linewidth=0.8,
                color='black',
                alpha=0.7
            )

    ax.set_xlim(0.5, T + 1.5)
    ax.set_xticks(np.arange(1, T + 1, xtick_step))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()
    return fig, ax
