"""Vector-based perturbation generator for clustered models."""

import numpy as np
from neuro_mod.core.perturbations._base import BasePerturbator


class VectorialPerturbation(BasePerturbator):
    """Generate perturbations as linear combinations of basis vectors."""

    def __init__(self, *vectors, **kwargs):
        """Initialize vectors and masks for perturbation generation.

        Args:
            *vectors: Optional basis vectors to use. If omitted, random or
                structured vectors are generated based on kwargs.
            **kwargs: Configuration for vector generation.
                n_vectors: Number of basis vectors.
                length: Length of each vector.
                structured: Whether to use structured +/-1 vectors.
                involved_clusters: Indices to include per vector.
                sigma: Std dev for unstructured random values.
                mean: Mean for unstructured random values.
                balance: Probability of +1 for structured vectors.
                rng: Optional NumPy random generator.
        """
        super().__init__(**kwargs)

        self.n_vectors = kwargs.get('n_vectors', len(vectors))
        self.length = kwargs.get('length') or len(vectors[0])
        self.structured = kwargs.get('structured', True)
        self.size = (self.n_vectors, self.length)

        involved = kwargs.get('involved_clusters', None)

        if involved is None:
            involved_entries = np.ones(self.size, dtype=bool)

        else:
            involved = list(involved)
            if isinstance(involved[0], (list, tuple, np.ndarray)):
                involved_lists = involved
            else:
                involved_lists = [involved] * self.n_vectors

            involved_entries = np.zeros(self.size, dtype=bool)
            rows = np.repeat(np.arange(self.n_vectors),
                             [len(v) for v in involved_lists])
            cols = np.concatenate(involved_lists)
            involved_entries[rows, cols] = True

        self.vectors = np.zeros(self.size, dtype=np.float64)
        if len(vectors) > 0:
            for i in range(self.n_vectors):
                self.vectors[i, involved_entries[i]] = vectors[i]
        else:
            if not self.structured:
                sigma = kwargs.get('sigma', 1.0)
                mean = kwargs.get('mean', 0.0)

                for i in range(self.n_vectors):
                    n_i = involved_entries[i].sum()
                    values = self.rng.normal(mean, sigma, size=n_i)
                    self.vectors[i, involved_entries[i]] = values

            else:
                balance = kwargs.get('balance', 0.5)

                # vectorizable only if all masks identical
                if np.all(involved_entries == involved_entries[0]):
                    gen_size = (self.n_vectors, involved_entries[0].sum())
                    values = self.rng.choice(
                        [1., -1.],
                        size=gen_size,
                        p=[balance, 1 - balance]
                    )
                    self.vectors[:, involved_entries[0]] = values
                else:
                    for i in range(self.n_vectors):
                        n_i = involved_entries[i].sum()
                        values = self.rng.choice(
                            [1., -1.],
                            size=n_i,
                            p=[balance, 1 - balance]
                        )
                        self.vectors[i, involved_entries[i]] = values

    def get_perturbation(self, *params: float, **kwargs):
        """Combine basis vectors using the provided coefficients.

        Args:
            *params: Coefficients for each vector (length must equal
                `n_vectors`).
            **kwargs: Ignored keyword arguments for compatibility.

        Returns:
            A perturbation vector of shape `(length,)`.
        """
        assert len(params) == self.n_vectors
        return np.dot(self.vectors.T, np.asarray(params))


if __name__ == '__main__':
    length = 6
    involved_entries = [[1,4], [3,5], [1,4], [2,0], [1,2,3,4]]
    n_vectors = 5
    vp = VectorialPerturbation(
        n_vectors=n_vectors,
        length=length,
        involved_clusters=involved_entries,
    )
    vp.get_perturbation(*np.random.rand(5))
