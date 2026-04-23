import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

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


def plot_density_path(
    y_path,
    p_grid,
    width_scale=1.2,
    n_eval=2000,
    normalise_width=True,
    draw_baseline=False,
    baseline_width=1.0,
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
    """
    T = y_path.shape[0]
    fig, ax = plt.subplots(figsize=figsize)

    # Draw from right to left so left densities can cover right neighbours
    for t in range(T - 1, -1, -1):
        yy, ff = quantile_to_density(
            y_path[t], p_grid, n_eval=n_eval
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
