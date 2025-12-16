
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)


class LIFNet(nn.Module):
    """Recurrent LIF spiking network implemented in PyTorch.

    The network supports analytic or Euler integration of membrane and synaptic
    dynamics, heterogeneous neuron parameters, and optional external currents.

    Args:
        synaptic_weights: Recurrent synaptic weight matrix of shape
            `(n_neurons, n_neurons)`.
        j_ext: Vector of external synaptic efficacies.
        tau_membrane: Membrane time constants (scalar, list, or tensor).
        tau_synaptic: Synaptic time constants (scalar, list, or tensor).
        tau_refractory: Refractory periods per neuron.
        threshold: Firing threshold values per neuron.
        reset_voltage: Reset voltage values per neuron. Defaults to `0.`.
        delta_t: Time step used for numerical integration (seconds).
    """

    def __init__(
            self,
            synaptic_weights: torch.Tensor,
            j_ext: torch.Tensor | list[float],
            tau_membrane: float | list[float] | torch.Tensor,
            tau_synaptic: float | list[float] | torch.Tensor,
            tau_refractory: float | list[float] | torch.Tensor,
            threshold: float | list[float] | torch.Tensor,
            reset_voltage: float | list[float] | torch.Tensor = 0.,
            delta_t: float = 1e-3,
    ):
        super().__init__()

        self.delta_t = delta_t
        self.n_neurons = synaptic_weights.shape[0]
        self.j_ext = torch.tensor(j_ext, requires_grad=False)
        self.j_ext = self.j_ext / self.n_neurons ** .5
        self.register_buffer("tau_synaptic", self._broadcast_param(tau_synaptic))
        self.register_buffer("tau_membrane", self._broadcast_param(tau_membrane))
        self.register_buffer('synaptic_weights', synaptic_weights)
        self.register_buffer('threshold', self._broadcast_param(threshold))
        self.register_buffer('reset_voltage', self._broadcast_param(reset_voltage))
        self.register_buffer("tau_ref_vec", self._broadcast_param(tau_refractory))

        self.register_buffer('spikes_timer', torch.zeros(self.n_neurons,))
        self.register_buffer('propagators', self._get_propagators())

    def _broadcast_param(self, param) -> torch.Tensor:
        """Convert scalars/lists/tensors to a 1D tensor of length `n_neurons`.

        Args:
            param: Scalar, list/tuple, or tensor describing a neuron-wise
                parameter.

        Returns:
            Tensor of shape `(n_neurons,)` with the parameter broadcasted or
            validated.
        """
        if isinstance(param, (float, int)):
            # If scalar, repeat it N times
            return torch.full((self.n_neurons,), float(param),)

        elif isinstance(param, (list, tuple)):
            # If list, convert to tensor
            t = torch.tensor(param,)
            if len(t) != self.n_neurons:
                raise ValueError(f"Parameter length {len(t)} does not match neurons {self.n_neurons}")
            return t

        elif isinstance(param, torch.Tensor):
            if param.numel() == 1:
                return torch.full((self.n_neurons,), float(param),)
            if param.shape[0] != self.n_neurons:
                raise ValueError(f"Tensor shape {param.shape} does not match neurons {self.n_neurons}")
            return param.float()

        else:
            raise TypeError(f"Unsupported parameter type: {type(param)}")

    def _get_propagators(self):
        propagators = torch.zeros((self.n_neurons, 3, 3))
        propagators[:, 0, 0] = torch.exp(-self.delta_t / self.tau_membrane)
        propagators[:, 0, 1] = self._get_i_propagator()
        propagators[:, 0, 2] = self.tau_membrane * (1 - propagators[:, 0, 0])

        propagators[:, 1, 1] = torch.exp(-self.delta_t / self.tau_synaptic)

        propagators[:, 2, 2] = 1.

        return propagators

    def _get_i_propagator(self):
        diff = self.tau_synaptic - self.tau_membrane
        denom = (self.tau_synaptic * self.tau_membrane)
        mul_term = torch.where(diff == 0, 0., denom / diff)
        exp_diff = torch.exp(-self.delta_t / self.tau_synaptic) - torch.exp(-self.delta_t / self.tau_membrane)
        return mul_term * exp_diff

    @torch.no_grad()
    def forward(
            self,
            voltage: torch.Tensor,
            synaptic_current: torch.Tensor,
            stimulus: torch.Tensor,
            external_currents: torch.Tensor | None = None,
            mechanism: str | None = None,
    ):
        """Simulate network dynamics over time.

        Args:
            voltage: Initial membrane voltages, shape `(n_neurons,)`.
            synaptic_current: Initial synaptic currents, shape `(n_neurons,)`.
            stimulus: Time series of stimulus currents, shape
                `(T, n_neurons)`.
            external_currents: Optional time series of external input, same
                shape as `stimulus`. If ``None``, zeros are used.
            mechanism: Integration mechanism, either ``"analytic"`` or
                ``"euler"``. Defaults to analytic.

        Returns:
            Tuple `(v_hist, c_hist, s_hist)` where:

            * `v_hist`: Membrane voltages over time.
            * `c_hist`: Synaptic currents over time.
            * `s_hist`: Binary spike indicators over time.
        """
        if external_currents is None:
            external_currents = torch.zeros_like(stimulus)

        v_history, c_history, s_history = [], [], []
        for t in range(stimulus.shape[0]):
            voltage, synaptic_current, s = self._step(voltage,
                                                      synaptic_current,
                                                      stimulus[t],
                                                      external_currents[t],
                                                      mechanism)
            v_history.append(voltage.clone())
            c_history.append(synaptic_current.clone())
            s_history.append(s.clone())
        return (torch.stack(v_history),
                torch.stack(c_history),
                torch.stack(s_history)
                )

    @torch.no_grad()
    def _step(self, voltage,
              synaptic_current,
              stimulus,
              external_current,
              mechanism: str | None = None):
        """Advance the network state by a single time step."""

        is_refractory = ~(self.spikes_timer.round(decimals=6) == 0)

        if mechanism == 'euler':
            v_update, synaptic_current = self._euler_step(voltage, synaptic_current, stimulus)
        else:
            v_update, synaptic_current = self._analytic_step(voltage, synaptic_current, stimulus)

        voltage = torch.where(is_refractory, self.reset_voltage, v_update)

        # Threshold comparison uses the heterogeneous threshold vector
        spikes_vec = (voltage >= self.threshold).flatten()

        if spikes_vec.any():
            voltage[spikes_vec] = self.reset_voltage[spikes_vec]  # Use specific reset
            self.spikes_timer[spikes_vec] = self.tau_ref_vec[spikes_vec]
        synaptic_update = self.synaptic_weights @ spikes_vec.double() + self.j_ext @ external_current
        synaptic_current += synaptic_update / self.tau_synaptic
        self.spikes_timer[is_refractory] -= self.delta_t

        return voltage, synaptic_current, spikes_vec.float()

    @torch.no_grad()
    def _euler_step(self,
                    voltage,
                    synaptic_current,
                    stimulus,
                    ):

        dv = -voltage + self.tau_membrane * (synaptic_current + stimulus)
        v_update = voltage + self.delta_t / self.tau_membrane * dv
        synaptic_current = synaptic_current - self.delta_t / self.tau_synaptic * synaptic_current

        return v_update, synaptic_current

    @torch.no_grad()
    def _analytic_step(self,
                       voltage,
                       synaptic_current,
                       stimulus,
                       ):

        v_update = (voltage * self.propagators[:, 0, 0]
                    + synaptic_current * self.propagators[:, 0, 1]
                    + stimulus * self.propagators[:, 0, 2])

        synaptic_current = synaptic_current * self.propagators[:, 1, 1]
        return v_update, synaptic_current
