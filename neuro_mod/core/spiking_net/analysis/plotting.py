"""Plotting utilities for spiking network analysis."""

import numpy as np
import matplotlib.pyplot as plt

def gen_raster_plot(spikes: np.ndarray,
                    delta_t: float,
                    duration_sec: float,
                    n_neurons: int,
                    *pops,
                    **kwargs):
    """Generate a spike raster plot.

    Args:
        spikes: Boolean spike matrix `(T, n_neurons)`.
        delta_t: Time step in seconds.
        duration_sec: Total duration in seconds.
        n_neurons: Number of neurons.
        *pops: Population boundary indices (exc, exc_bkg, inh_bkg).
        **kwargs: Ignored keyword arguments for compatibility.

    Returns:
        Matplotlib figure with the raster plot.
    """
    times, neurons = spikes.nonzero()
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.scatter(times * delta_t, neurons, s=0.1, color='black')  # each spike = one dot
    exc, exc_background, inh_background = pops
    population_markers = [
        (0, exc, 'skyblue', 'Excitatory'),
        (exc, n_neurons, 'salmon', 'Inhibitory'),
        (exc_background,
         exc, 'blue', 'background excitatory'),
        (inh_background,
         n_neurons, 'red', 'background inhibitory'),
    ]
    for y_min, y_max, color, label in population_markers:
        ax.axhspan(ymin=y_min, ymax=y_max, color=color, alpha=0.3,
                   label=label)

    ax.set_xlabel('Time [S]', fontsize=16)
    ax.set_ylabel('Neuron index', fontsize=16)
    ax.set_title('Spike Raster Plot', fontsize=24)
    ax.set_xlim(0, duration_sec)
    ax.set_ylim(0, n_neurons)
    # fig.gca().invert_yaxis()
    # fig.tight_layout()
    plt.close()
    return fig
