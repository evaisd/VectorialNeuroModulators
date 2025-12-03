
from neuro_mod.mean_field.runners._base import *
from neuro_mod.mean_field.core import LIFMeanField

class MainMFRunner(SimRunner):

    def __init__(self, lif_mf: LIFMeanField):
        super().__init__(lif_mf)

    def run(self, nu_init, *args, **kwargs):
        res = self.lif.solve_rates(nu_init)
        stability = self.lif.determine_stability(res.x)
        return res.x, *stability

    def _gen_params(self, **params):
        pass

    def _mft_params(self, **params):
        pass

    def _settings(self):
        pass

    def run_effective_rates_on_grid(self, focus_pops: list[int], *focus_nu_vecs: np.ndarray):
        mesh = np.concatenate(np.meshgrid(*focus_nu_vecs), axis=-1)
        nu_outs = np.empty_like(mesh)
        for idx in np.ndindex(mesh.shape[:self.lif.n_populations - 1]):
            nu_outs[idx] = self.lif.effective_response_function(focus_pops, mesh[idx])
        return mesh, nu_outs
