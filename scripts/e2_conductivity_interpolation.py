#!/usr/bin/env python3
"""Conductivity interpolation form versus the isolated-halo calibration (2026-11-06).

Collects the direct C scans of scripts/e2_conductivity_alpha (solver copy with
K = (K_s^-alpha + K_l^-alpha)^(-1/alpha); alpha = 1 reproduces the production solver bit for bit
and the stored direct residual at C = 0.712) into results/e2/conductivity_interpolation.json, with
the source hashes. No solver call. The scans were run with no time rescaling on E2's frozen
calibration snapshots 010-045 and scorer.
"""
import hashlib, json, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SRC = {a: os.path.join(ROOT, 'results', 'e2', 'conductivity_alpha', f'alpha_{a}.json') for a in ('1.0', '0.5')}
out = {'note': __doc__.strip(), 'sources': {}, 'reference': 'van den Bosch & Tokayer 2026, arXiv:2610.28890 (alpha = 0.5)'}
for a, p in SRC.items():
    d = json.load(open(p))
    out['sources'][a] = {'path': os.path.relpath(p, ROOT), 'sha256': hashlib.sha256(open(p, 'rb').read()).hexdigest()}
    out[f'alpha_{a}'] = {'best_C': d['best']['C'], 'rms_dex': d['best']['rms_dex'], 'snapshot_051_dex': d['best']['snapshot_051']}
    if a == '1.0':
        out['alpha_1.0']['reproduces_production'] = d['reproduces_stored_production_run'] and d['reproduces_stored_direct_residual']
json.dump(out, open(os.path.join(ROOT, 'results', 'e2', 'conductivity_interpolation.json'), 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k.startswith('alpha')}))
