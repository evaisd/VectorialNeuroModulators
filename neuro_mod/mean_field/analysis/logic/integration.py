"""Grid utilities for integrating mean-field force fields."""

import numpy as np


def find_point_on_grid(grid: np.ndarray, point: np.ndarray):
    """Find the closest grid index to a point.

    Args:
        grid: Array of grid coordinates of shape `(N, M, 2)`.
        point: Target point of shape `(2,)`.

    Returns:
        Index array `[i, j]` of the closest grid point.
    """
    from scipy.spatial.distance import cdist
    grid_size = grid.shape[:2]
    dists = cdist(grid.reshape(grid_size[0] * grid_size[1], 2),
                  point[np.newaxis, :]).reshape(grid_size)
    return np.array(np.unravel_index(dists.argmin(), dists.shape))


def get_path_of_min_res(force_field: np.ndarray):
    """Find a minimal-resistance path through a force field.

    Args:
        force_field: Vector field with shape `(N, M, 2)`.

    Returns:
        Path indices as an array of shape `(M, 2)`.
    """
    energy = np.sqrt(np.sum(force_field ** 2, axis=-1))
    rows_mins = energy.argmin(axis=0)
    path = np.stack([rows_mins, np.arange(energy.shape[0])]).T
    return path


def compute_line_integral_on_path(
        grid: np.ndarray,
        force_field: np.ndarray,
        path: np.ndarray
):
    """Compute line integral of a force field along a given path.

    Args:
        grid: Grid of coordinates with shape `(N, M, 2)`.
        force_field: Vector field with shape `(N, M, 2)`.
        path: Array of indices defining the path.

    Returns:
        Tuple `(line_integral, displacement)` where `line_integral` is the
        cumulative integral and `displacement` is the cumulative distance.
    """
    path = path.T
    coords = grid[*path]
    dr = coords[1:] - coords[:-1]
    fstep = force_field[*path][1:]
    nulls = (dr != 0.).any(axis=1)
    dr, fstep = dr[nulls], fstep[nulls]
    f_dot_dr = np.einsum('ij,ij->i', fstep, dr)
    line_integral = np.cumsum(-f_dot_dr, axis=-1)
    displacement = np.cumsum(np.linalg.norm(dr, axis=1))
    return line_integral, displacement
