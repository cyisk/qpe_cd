import numpy as np

from typing import Callable, Tuple
from numpy.typing import NDArray
from matplotlib.pylab import Axes
from scipy.stats import gaussian_kde


def visualize_scatter(
    ax: Axes,
    x: NDArray,
    y: NDArray,
    cmap: str = 'viridis',
    **kwargs,
):
    x, y = x[:, 0], y[:, 0]
    xy = np.vstack([x, y])
    sample_density = gaussian_kde(xy)(xy)
    ax.scatter(
        x,
        y,
        cmap=cmap,
        c=sample_density,
        edgecolors="none",
        **kwargs,
    )


def visualize_density(
    ax: Axes,
    x: NDArray,
    y: NDArray,
    x_coord: NDArray = np.linspace(-3, 3, 20),
    y_coord: NDArray = np.linspace(-3, 3, 20),
    cmap: str = 'viridis',
    **kwargs,
):
    x, y = x[:, 0], y[:, 0]
    X, Y = np.meshgrid(x_coord, y_coord)
    xy = np.vstack([x, y])
    XY = np.vstack([X.ravel(), Y.ravel()])
    grid_density = gaussian_kde(xy)(XY).reshape(X.shape)
    ax.contourf(X, Y, grid_density, levels=50)


def visualize_qpe_scatter(
    ax: Axes,
    qpe: Callable[[NDArray, NDArray], NDArray],
    x: NDArray,
    y: NDArray,
    cmap: str = 'viridis',
    qrange: Tuple[float, float] = (5, 95),
    **kwargs,
):
    q = qpe(x, y)[:, 0]
    q = q[~np.isnan(q)]
    lb, ub = np.percentile(q, qrange[0]), np.percentile(q, qrange[1])
    q[(q < lb) | (q > ub)] = np.nan
    ax.scatter(
        x[:, 0],
        y[:, 0],
        q,
        cmap=cmap,
        c=q,
        **kwargs,
    )


def visualize_qpe_surface(
    ax: Axes,
    qpe: Callable[[NDArray, NDArray], NDArray],
    x_coord: NDArray = np.linspace(-3, 3, 20),
    y_coord: NDArray = np.linspace(-3, 3, 20),
    cmap: str = 'viridis',
    qrange: Tuple[float, float] = (5, 95),
    **kwargs,
):
    XX, YY = np.meshgrid(x_coord, y_coord)
    XX_flat, YY_flat = XX.flatten(), YY.flatten()
    q = qpe(XX_flat[:, None], YY_flat[:, None]).reshape(XX.shape)
    q, qf = q[~np.isnan(q)], q.flatten()
    lb, ub = np.percentile(qf, qrange[0]), np.percentile(qf, qrange[1])
    q[(q < lb) | (q > ub)] = np.nan
    q = q.reshape(XX.shape)
    ax.plot_surface(
        XX,
        YY,
        q,
        cmap=cmap,
        linewidth=0,
        vmin=lb,
        vmax=ub,
        antialiased=True,
        **kwargs,
    )


def visualize_qpe_intersection_y(
    ax: Axes,
    qpe: Callable[[NDArray, NDArray], NDArray],
    x_coord: NDArray = np.linspace(-3, 3, 20),
    y_coord: NDArray = np.linspace(-3, 3, 20),
    qrange: Tuple[float, float] = (5, 95),
    **kwargs,
):
    XX, YY = np.meshgrid(x_coord, y_coord)
    XX_flat, YY_flat = XX.flatten(), YY.flatten()
    q = qpe(XX_flat[:, None], YY_flat[:, None]).reshape(XX.shape)
    q, qf = q[~np.isnan(q)], q.flatten()
    lb, ub = np.percentile(qf, qrange[0]), np.percentile(qf, qrange[1])
    q[(q < lb) | (q > ub)] = np.nan
    q = q.reshape(XX.shape)
    for idx in range(x_coord.shape[0]):
        ax.plot(
            XX[:, idx],
            YY[:, idx],
            q[:, idx],
            c='k',
            **kwargs,
        )
