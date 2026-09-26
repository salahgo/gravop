"""Equal starting surveys and node caps on independent analytic integrands."""
import sys
from pathlib import Path

import numpy as np
from scipy.special import erf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gravop.quadrature import DyadicMesh
from gravop.simpson import SimpsonMesh
from forecast_validate import write_json, digest


def main():
    sigma = .008
    ridge_exact = 2 * (sigma * np.sqrt(np.pi / 2) * erf(1 / (np.sqrt(2) * sigma))
                       - sigma**2 * (1 - np.exp(-.5 / sigma**2)))
    s = .04
    center = np.array([.31, .72])
    peak_exact = np.prod(s * np.sqrt(np.pi / 2) *
        (erf((1 - center) / (np.sqrt(2) * s)) + erf(center / (np.sqrt(2) * s))))
    cases = {
        'diagonal_ridge': (lambda xy: np.exp(-.5 * ((xy[:, 0] - xy[:, 1]) / sigma)**2), ridge_exact),
        'smooth_peak': (lambda xy: np.exp(-.5 * np.square((xy - center) / s).sum(1)), peak_exact),
        'survival_triangle': (lambda xy: (xy.sum(1) <= .91).astype(float), .5 * .91**2),
    }
    out = dict(tolerance=.005, node_cap=20000, starting_nodes=4225, results={}, source_sha256=digest(__file__))
    for name, (fn, exact) in cases.items():
        out['results'][name] = {}
        for method in ('trapezoid', 'simpson'):
            mesh = DyadicMesh(base=32, lattice=65536) if method == 'trapezoid' else SimpsonMesh(base=16)
            history = []
            while True:
                nodes, ids = mesh.stencil()
                assessment = mesh.assess(fn(nodes / mesh.lattice), ids)
                error = float(assessment['relative_error'][0])
                actual = float(abs(assessment['integral'][0] / exact - 1))
                history.append(dict(nodes=len(nodes), embedded_error=error, actual_error=actual))
                if error <= out['tolerance']:
                    break
                per_split, minimum = (16, 4) if method == 'trapezoid' else (56, 8)
                eligible = np.flatnonzero(mesh.cells[:, 2] >= minimum)
                n = min(16, (out['node_cap'] - len(nodes)) // per_split, len(eligible))
                if not n:
                    break
                mesh.refine(eligible[np.argsort(assessment['priority'][eligible])[-n:]])
            out['results'][name][method] = dict(**history[-1], converged=error <= out['tolerance'], history=history)
            print(name, method, out['results'][name][method] | {'history': 'recorded'}, flush=True)
    write_json(ROOT / 'results/forecast_numerics/analytic_benchmark.json', out)


if __name__ == '__main__':
    main()
