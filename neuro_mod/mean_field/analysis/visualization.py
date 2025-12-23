
import matplotlib.pyplot as plt
from neuro_mod.mean_field.analysis import integration as ing


def gen_phase_diagram_plot(
        sweep_param_name,
        sweep_param_vals,
        results,
        *args,
        **kwargs
):
    fig, ax = plt.subplots(1, 1, figsize=(18, 18))
    for i in range(results.shape[0]):
        ax.plot(
            sweep_param_vals, results[i],
            lw=12,
            label=f'Population {i + 1}',
            *args,
            **kwargs
        )
    ax.set_xlabel(sweep_param_name, fontsize=24)
    ax.set_ylabel('Firing rate', fontsize=24)
    fig.legend(fontsize=18)
    return fig, ax


def gen_rate_flow_map(
        nu_in_grid,
        force_field,
        points: list[dict] = None,
        *args,
        **kwargs
):
    path = ing.get_path_of_min_res(force_field)
    freq = round(nu_in_grid.shape[0] / 50), round(nu_in_grid.shape[1] / 50)
    _grid = nu_in_grid[::freq[0], ::freq[0], ...]
    _force_field = force_field[::freq[0], ::freq[0], ...]
    fig, ax = plt.subplots(1, 1, figsize=(18, 18))
    ax.quiver(_grid[..., 0],
              _grid[..., 1],
              _force_field[..., 0],
              _force_field[..., 1],
              label='Force Field',
              *args,
              **kwargs)
    # ax.scatter(*nu_in_grid[path[:, 0], path[:, 1]].T, label='Minimal Resistance path')
    ax.plot(*nu_in_grid[*path.T].T, '-o', lw=6, ms=12, label='Minimal Resistance path')
    if points is not None:
        _ = [ax.plot(*entry['point'], 'or', ms=24, label=entry['kind']) for entry in points]
    ax.set_xlabel(r'$\nu_{E,0}$', fontsize=24)
    ax.set_ylabel(r'$\nu_{E,1}$', fontsize=24)
    plt.legend(fontsize=18)
    return fig, ax

def gen_potential_plot(
        grid,
        force_field,
        path,
        *args,
        **kwargs
):

    potential, displacement = ing.compute_line_integral_on_path(grid, force_field, path)
    fig, ax = plt.subplots(figsize=(18, 18))
    ax.plot(displacement, potential, '-o', ms=18, lw=8, label='Potential')
    plt.legend(fontsize=18)
    ax.set_xlabel('Path Distance', fontsize=24)
    ax.set_ylabel('Potential', fontsize=24)
    return fig, ax
