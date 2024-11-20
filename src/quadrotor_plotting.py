import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.transforms

def confidence_ellipse(x, y, cov, ax, n_std=1.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Modified from: https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = matplotlib.patches.Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = matplotlib.transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)

def plot_quad_chart(xlim, ylim, graphs, x_trajs, title="", plot_obs=False, save_fig=False, save_fname = ""):
    fig, subplot_ax = plt.subplots(2, 2)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    })

    fig.set_size_inches(9, 7)
    axs = subplot_ax.flat#[ax1, ax2, ax3, ax4]
    for ax_i, ax in enumerate([axs[0], axs[1]]):
        x_trajs_mc = x_trajs[ax_i]
        graph = graphs[ax_i]
        for i in range(1, x_trajs_mc.shape[2]):
            ax.plot(x_trajs_mc[0, :, i], x_trajs_mc[1, :, i], color='black', alpha=0.2)

        for node in graph.nodes:
            cov = node.covariance[0:2, 0:2]
            if not node.is_goal and not node.is_start:
                pass
            elif node.is_goal:
                confidence_ellipse(node.mean[0], node.mean[1], cov, ax, facecolor='yellow', label='Goal', zorder=15)
            elif node.is_start:
                confidence_ellipse(node.mean[0], node.mean[1], cov, ax, facecolor='red', label='Start', zorder=10)

        if plot_obs:
            obstacle = matplotlib.patches.Rectangle((3, 3), 4, 4, facecolor='green', label='High-variance area', alpha=1)
            ax.add_patch(obstacle)
            obstacle = matplotlib.patches.Rectangle((0, 0), 10, 10, facecolor='green', label='Low-variance area', alpha=0.2)
            ax.add_patch(obstacle)


        ax.plot(x_trajs_mc[0, :, 0], x_trajs_mc[1, :, 0], color='black', alpha=0.2, label='Monte Carlo trajectory')

        ax.set_xlabel('X-position (m)')
        ax.set_ylabel('Y-position (m)')
        ax.set_xlim(xlim)
        ax.set_ylim([0, 9])
        if ax_i == 0:
            leg_handles, leg_labels = ax.get_legend_handles_labels()
            for lh, ll in zip(leg_handles, leg_labels):#legend_handles:
                if ll == 'Monte Carlo trajectory':
                    lh.set_alpha(1)
            ax.legend()
    for ax_i, ax in enumerate([axs[2], axs[3]]):
        x_trajs_mc = x_trajs[ax_i]
        ax.scatter(x_trajs_mc[0, -1, :], x_trajs_mc[1, -1, :], label='Empirical final state', alpha=0.5)
        ax.scatter([8], [8], label='Goal mean', marker='*', s=200)
        ax.set_xlabel('X-position (m)')
        ax.set_ylabel('Y-position (m)')
        ax.set_xlim([6.5, 9.5])
        ax.set_ylim([7, 9])
        if ax_i == 0:
            leg = ax.legend()
            for lh in leg.legend_handles:
                lh.set_alpha(1)
    if save_fig:
        plt.savefig(save_fname, dpi=1200, bbox_inches='tight')
    plt.show()

