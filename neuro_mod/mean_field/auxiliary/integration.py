
import numpy as np


def find_point_on_grid(grid: np.ndarray, point: np.ndarray):
    from scipy.spatial.distance import cdist
    grid_size = grid.shape[:2]
    dists = cdist(grid.reshape(grid_size[0] * grid_size[1], 2),
                  point[np.newaxis, :]).reshape(grid_size)
    return np.array(np.unravel_index(dists.argmin(), dists.shape))


def get_path_of_min_res(force_field: np.ndarray):
    energy = np.sqrt(np.sum(force_field ** 2, axis=-1))
    rows_mins = energy.argmin(axis=0)
    path = np.stack([rows_mins, np.arange(energy.shape[0])]).T
    return path


def compute_line_integral_on_path(
        grid: np.ndarray,
        force_field: np.ndarray,
        path: np.ndarray
):
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
