"""Positive nested Simpson quadrature for a separately controlled F5 pilot.

Each square compares a 3x3 rule with its 5x5 composite rule. Raw absolute
embedded differences remain conservative estimates, not rigorous error bounds.
"""
import numpy as np

from .quadrature import DyadicMesh


class SimpsonMesh(DyadicMesh):
    fine_weights = np.outer([1., 4., 2., 4., 1.], [1., 4., 2., 4., 1.]).ravel() / 144
    coarse_weights = np.outer([1., 4., 1.], [1., 4., 1.]).ravel() / 36
    coarse_indices = [i * 5 + j for i in (0, 2, 4) for j in (0, 2, 4)]

    def __init__(self, base=16, lattice=65536, cells=None):
        super().__init__(base, lattice, cells)
        if (self.cells[:, 2] < 4).any() or (self.cells[:, 2] % 4).any():
            raise ValueError('Simpson cells require integer quarter-point nodes')

    def stencil(self):
        nodes = np.stack([self.cells[:, :2] + self.cells[:, 2, None] * np.array([i, j]) // 4
                          for i in range(5) for j in range(5)], axis=1)
        unique, inverse = np.unique(nodes.reshape(-1, 2), axis=0, return_inverse=True)
        return unique, inverse.reshape(len(self.cells), 25)

    def assess(self, values, indices):
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('integrands must be finite and nonnegative')
        area = (self.cells[:, 2] / self.lattice) ** 2
        fine = sum(w * values[indices[:, k]] for k, w in enumerate(self.fine_weights))
        coarse = sum(w * values[indices[:, k]] for k, w in
                     zip(self.coarse_indices, self.coarse_weights))
        fine *= area[:, None]
        coarse *= area[:, None]
        error = np.abs(fine - coarse)
        total = fine.sum(0)
        if (total <= 0).any():
            raise ValueError('integrand has no sampled positive mass')
        return dict(integral=total, relative_error=error.sum(0) / total,
                    priority=(error / total).max(1))

    def weights(self, indices, n_nodes):
        area = (self.cells[:, 2] / self.lattice) ** 2
        out = np.zeros(n_nodes)
        for k, w in enumerate(self.fine_weights):
            np.add.at(out, indices[:, k], area * w)
        return out

    def refine(self, selected):
        if (self.cells[np.asarray(selected, dtype=int), 2] < 8).any():
            raise ValueError('minimum Simpson cell size reached')
        super().refine(selected)
