
import numpy as np


class ExternalCurrentsGenerator:

    def __init__(self,
                 n_neurons: int,
                 n_excitatory: int,
                 c_ext: np.ndarray[int] | list[int],
                 nu_ext_baseline: list[float] | list[float],
                 random_generator: np.random.Generator = None,
                 delta_nu_ext: list[float] | np.ndarray[float] = None,
                 ):
        self.c_ext = c_ext
        self.nu_ext_baseline = nu_ext_baseline
        self.total_neurons = n_neurons
        self.excitatory_neurons = n_excitatory
        self.inhibitory_neurons = self.total_neurons - n_excitatory
        self.rng = np.random.default_rng(256) if random_generator is None else random_generator
        self.delta_nu_ext = delta_nu_ext

    def _gen(self, rate, size):
        return self.rng.poisson(rate, size)

    def generate_external_currents(self,
                                   duration: float,
                                   delta_t: float,):
        total_steps = int(duration / delta_t)
        full_output = np.zeros((4, total_steps, self.total_neurons))
        poisson_e = []
        poisson_i = []
        for i, (nu, c) in enumerate(zip(self.nu_ext_baseline, self.c_ext)):
            from_excitatory = i < 2
            to_excitatory = i % 2 == 0
            base_rate = nu
            num_neurons_to = self.excitatory_neurons if i % 2 == 0 else self.inhibitory_neurons
            if to_excitatory:
                delta_rate = self.delta_nu_ext[:self.excitatory_neurons]
            else:
                delta_rate = self.delta_nu_ext[self.excitatory_neurons:]
            rate = (base_rate + delta_rate) * c * delta_t
            size = (total_steps, num_neurons_to)
            if from_excitatory:
                poisson_e.append(self._gen(rate, size))
            else:
                poisson_i.append(self._gen(rate, size))
        full_output[0] = np.concatenate(poisson_e, axis=1)
        full_output[1] = np.concatenate(poisson_i, axis=1)

        return full_output.swapaxes(0, 1)
