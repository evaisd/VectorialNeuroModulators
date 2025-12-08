
import numpy as np


class StimulusGenerator:

    def __init__(self):

        self.funcs = {
            "box": self._box,
            "linear": self._linear,
            "diff2exp": self._diff2exp,
        }

        self.enum_funcs = {
            1: self._box,
            2: self._linear,
            3: self._diff2exp,
        }

    @staticmethod
    def _box(sim_duration: int,
             onsets: int | list[int],
             offsets: int | list[int],
             amplitude: float,
             *args,
             **kwargs) -> np.ndarray:
        if isinstance(onsets, (int, float)):
            onsets = [onsets]
        stim_current = np.zeros(sim_duration)
        for onset, offset in zip(onsets, offsets):
            stim_current[onset:offset] += amplitude
        return stim_current

    @staticmethod
    def _linear(sim_duration: int,
                onsets: int | list[int],
                offsets: int | list[int],
                amplitude: float,
                *args,
                **kwargs
                ) -> np.ndarray:
        if isinstance(onsets, (int, float)):
            onsets = [onsets]
        stim_current = np.zeros(sim_duration)
        for onset, offset in zip(onsets, offsets):
            slope = amplitude/(offset - onset)
            intercept = -slope * onset
            stim_current[onset:offset] += (slope * np.arange(onset, offset) + intercept)
        return stim_current

    @staticmethod
    def _diff2exp(sim_duration: int,
                  onsets: int | list[int],
                  offsets: int | list[int],
                  amplitude: float,
                  tau_rise: float,
                  tau_decay: float,
                  delta_t: float,
                  *args,
                  **kwargs) -> np.ndarray:
        if isinstance(onsets, (int, float)):
            onsets = [onsets]
        stim_current = np.zeros(sim_duration)
        gamma_comp_1 = (tau_rise / tau_decay) ** (tau_rise / (tau_decay - tau_rise))
        gamma_comp_2 = (tau_rise / tau_decay) ** (tau_decay / (tau_decay - tau_rise))
        gamma = 1 / (gamma_comp_1 - gamma_comp_2)
        for onset in onsets:
            time = np.arange(onset, sim_duration) * delta_t
            exp_1 = np.exp((onset * delta_t - time) / tau_decay)
            exp_2 = np.exp((onset * delta_t - time) / tau_rise)

            stimulus = amplitude * gamma * (exp_1 - exp_2)
            stim_current[onset:] += stimulus
        return stim_current

    def generate_stimulus(self,
                          stimulus_shape: str | int,
                          total_duration: float,
                          n_neurons: int,
                          onsets: float,
                          duration: float,
                          amplitude: float,
                          stimulated_neurons: list[int] | np.ndarray[int] = None,
                          delta_t: float = 1e-3,
                          *args,
                          **kwargs
                          ) -> np.ndarray:

        if isinstance(stimulus_shape, str):
            try:
                shape = self.funcs[stimulus_shape]
            except KeyError:
                raise ValueError(f"Invalid external shape: {stimulus_shape}."
                                 f" Should be one of {list(self.funcs.keys())}")
        elif isinstance(stimulus_shape, int):
            try:
                shape = self.enum_funcs[stimulus_shape]
            except KeyError:
                raise ValueError(f"Invalid external shape: {stimulus_shape}."
                                 f"should be one of {list(self.enum_funcs.keys())}")
        else:
            raise ValueError("Invalid external shape.")

        total_steps = int(total_duration / delta_t)
        onsets_step = [int(onset / delta_t) for onset in onsets]
        duration_step = int(duration / delta_t)
        offsets_step = [onset_step + duration_step for onset_step in onsets_step]

        kwargs = {
            "sim_duration": total_steps,
            "onsets": onsets_step,
            "offsets": offsets_step,
            "amplitude": amplitude,
            "tau_rise": kwargs.get("tau_rise", None),
            "tau_decay": kwargs.get("tau_decay", None),
            "delta_t": delta_t,
        }
        stim_current = shape(**kwargs)
        if stimulated_neurons is None:
            stimulated_neurons = np.arange(n_neurons)

        stimulus = np.zeros((total_steps, n_neurons))
        stimulus[:, stimulated_neurons] = stim_current[:, None]

        return stimulus
