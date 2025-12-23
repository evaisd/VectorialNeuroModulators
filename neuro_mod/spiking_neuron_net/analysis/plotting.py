
import numpy as np
import matplotlib.pyplot as plt

def gen_raster_plot(spikes: np.ndarray,
                    delta_t: float,
                    duration_sec: float,
                    n_neurons: int,
                    *pops,
                    **kwargs):
    times, neurons = spikes.nonzero()
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.scatter(times * delta_t, neurons, s=0.1, color='black')  # each spike = one dot
    exc, exc_bkg, inh_bkg = pops
    population_markers = [
        (0, exc, 'skyblue', 'Excitatory'),
        (exc, n_neurons, 'salmon', 'Inhibitory'),
        (exc_bkg,
         exc, 'blue', 'background excitatory'),
        (inh_bkg,
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
    fig.tight_layout()
    plt.close()
    return fig