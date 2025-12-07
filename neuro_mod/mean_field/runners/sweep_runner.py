
import numpy as np
from neuro_mod.mean_field.runners import MainMFRunner
from neuro_mod.mean_field.runners._base import SweepRunner
from neuro_mod.mean_field.core import LIFMeanField


class MFSweepRunner(SweepRunner):

    def _params_tweaker(self, sweep_param_val: float | np.ndarray):
        if self.mode == 'new':
            self._get_params(sweep_param_val)
        elif self.mode == 'rotation':
            self._apply_transformation(self.sweep_param, sweep_param_val)
        elif self.mode == 'shift':
            self._shift_params(self.sweep_param, sweep_param_val)
        else:
            raise ValueError('Invalid mode')

    def _single_run(self, sweep_param_val: float | np.ndarray, nu_init: np.ndarray, *args, **kwargs):
        self._params_tweaker(sweep_param_val)
        lif = LIFMeanField(**self.params)
        base_runner = MainMFRunner(lif)
        res, _, evs = base_runner.run(nu_init)
        return res

    def run(self, sweep_params: np.ndarray | list[float], nu_init: np.ndarray, *args, **kwargs):
        sweep_params = np.array(sweep_params)
        res = []
        for sweep_param_val in sweep_params:
            val = self._single_run(sweep_param_val, nu_init)
            if val is not None:
                res.append(val)
            self._reset_params()
        res = np.stack(res)
        return res.T


class EffectiveMFSweepRunner(MFSweepRunner):

    def __init__(
            self,
            sweep_param: str,
            mode: str = 'new',
            focus_pops=None,
            *nu_vecs,
            **params
    ):

        super().__init__(sweep_param, mode, **params)
        if focus_pops is None:
            focus_pops = [0, 1]
        self.focus_pops = focus_pops
        self.nu_vecs = nu_vecs

    def _single_run(self, sweep_param_val: float | np.ndarray, nu_init: np.ndarray, *args, **kwargs):
        from scipy.signal import find_peaks
        from neuro_mod.mean_field.auxiliary import integration as ing
        self._params_tweaker(sweep_param_val)
        lif = LIFMeanField(**self.params)
        base_runner = MainMFRunner(lif)
        mesh, nu_outs = base_runner.run_effective_rates_on_grid(self.focus_pops, *self.nu_vecs)
        shape = (mesh.shape[0], mesh.shape[1], mesh.shape[-1])
        grid = mesh.reshape(shape)[..., :2]
        nu_out_grid = nu_outs.reshape(shape)[..., :2]
        force_field = nu_out_grid - grid
        path = ing.get_path_of_min_res(force_field)
        potential, displacement = ing.compute_line_integral_on_path(grid, force_field, path)
        peaks = find_peaks(-potential)[0]
        i = 0
        while len(peaks) < 2:
            return
            # peaks = np.append(peaks, i)
            # i += 1
            # peaks = np.append(peaks, 0)
        if len(peaks) > 2:
            pots_at_peaks = potential[peaks]
            peaks = np.sort(peaks[np.argsort(pots_at_peaks)][:2])
        return potential[peaks], displacement[peaks], np.array([lif.ext_mu[0], lif.ext_sigma[0]])
