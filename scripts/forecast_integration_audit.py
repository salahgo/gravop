"""Recompute F4 mesh diagnostics and verify cached experiment identities."""
import json
import platform

import numpy as np
import scipy
import torch

from forecast_integrate import (
    Bank, DATA, OUT, ROOT, TIMES, NOISE_DEX, DyadicMesh,
    digest, log_likelihood, write_json,
)


def main():
    bank = Bank()
    obs = np.load(DATA / 'observations.npz')
    state = json.loads((DATA / 'mesh.json').read_text())
    mesh = DyadicMesh(cells=state['cells'])
    keys, indices = mesh.stencil()
    ids = bank.ids(keys)
    lp = log_likelihood(bank.d['profiles'], bank.d['valid'], obs)
    values = np.exp(lp[ids] - lp.max(0))
    assessment = mesh.assess(values, indices)
    fine = np.zeros((len(mesh.cells), len(obs['row'])))
    coarse = np.zeros_like(fine)
    for k, w in enumerate(np.outer([1, 2, 1], [1, 2, 1]).ravel() / 16):
        fine += w * values[indices[:, k]]
    for k in (0, 2, 6, 8):
        coarse += .25 * values[indices[:, k]]
    area = (mesh.cells[:, 2] / mesh.lattice) ** 2
    error = np.abs(fine - coarse) * area[:, None] / assessment['integral']
    smallest = mesh.cells[:, 2] < 4
    result = dict(
        cell_step_counts={str(s): int((mesh.cells[:, 2] == s).sum())
                          for s in np.unique(mesh.cells[:, 2])},
        minimum_size_error_max=float(error[smallest].sum(0).max()), times={},
    )
    for j, t in enumerate(TIMES):
        ix = np.flatnonzero(obs['time_index'] == j)
        relative = assessment['relative_error'][ix]
        worst = ix[np.argmax(relative)]
        result['times'][str(int(t))] = dict(
            median=float(np.median(relative)), p95=float(np.quantile(relative, .95)),
            maximum=float(relative.max()),
            fraction_within_first_tolerance=float(np.mean(relative <= .02)),
            worst_observation=int(worst), worst_truth=obs['params'][worst].tolist(),
            worst_minimum_size_error=float(error[smallest, worst].sum()),
        )
    weights = np.zeros(len(bank.d['keys']))
    weights[ids] = mesh.weights(indices, len(keys))
    rule = np.load(DATA / 'latest_rule.npz')
    np.testing.assert_array_equal(weights, rule['weights'])
    np.testing.assert_array_equal(bank.d['keys'], rule['keys'])
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-14, rtol=0)
    assert (weights >= 0).all()

    # Check matrix likelihood arithmetic against explicit residuals, spanning
    # each time and the complete cache; also check invalid support exactly.
    sampled = np.linspace(0, len(bank.d['params']) - 1, 97, dtype=int)
    for j in range(3):
        case = np.flatnonzero(obs['time_index'] == j)[0]
        valid = bank.d['valid'][sampled, j]
        direct = -.5 * np.square((bank.d['profiles'][sampled[valid], j]
                                  - obs['observations'][case]) / NOISE_DEX).sum(1)
        np.testing.assert_allclose(lp[sampled[valid], case], direct, atol=1e-7, rtol=1e-10)
        assert np.isneginf(lp[sampled[~valid], case]).all()

    generation = json.loads((OUT / 'generation.json').read_text())
    prediction = json.loads((OUT / 'prediction_provenance.json').read_text())
    validation = json.loads((OUT / 'validation.json').read_text())
    for path, expected in (
        (ROOT / 'docs/FORECAST_INTEGRATION_PROTOCOL.md', generation['protocol_sha256']),
        (DATA / 'confirmation.npz', generation['data_sha256']),
        (DATA / 'observations.npz', generation['observations_sha256']),
        (DATA / 'bank.npz', prediction['bank_sha256']),
        (DATA / 'model_predictions.npz', prediction['predictions_sha256']),
    ):
        assert digest(path) == expected, path
    for group in prediction['checkpoint_hashes'].values():
        for path, expected in group.items():
            assert digest(ROOT / path) == expected, path
    for path, expected in validation['provenance']['source_sha256'].items():
        assert digest(ROOT / path) == expected, path
    np.testing.assert_allclose(assessment['relative_error'].max(),
                               validation['progress']['history'][-1]['worst_relative_error'])
    result['checks'] = dict(final_rule_matches_mesh=True, positive_full_prior_weights=True,
                           explicit_likelihood_agrees=True, recorded_hashes_unchanged=True)
    result['environment'] = dict(python=platform.python_version(), numpy=np.__version__,
                                 scipy=scipy.__version__, torch=torch.__version__,
                                 cuda=torch.version.cuda,
                                 gpus=[torch.cuda.get_device_name(i)
                                       for i in range(torch.cuda.device_count())])
    write_json(OUT / 'final_error_diagnostic.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
