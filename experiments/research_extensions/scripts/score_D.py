#!/usr/bin/env python3
"""Score the D1/D2 particle results against the frozen predictions (written
before any D data were measured; commit history shows the order).

Observables per halo, from closure_design/nsphere/<run>/measures.npz:
  seed means of t_x10 (published marker) and t_x10_s (smoothed) in t_dyn at
  1e5 and 2e5; the dt/2 control against its same-seed 1e5 run.
Scores:
  1. Each preregistered closure: z = ln(T_obs / T_pred) / sigma_y, sigma_y^2 =
     tau^2/n + 0.03^2 (N-bias) with tau the observed seed scatter (floored at
     0.03); closures ranked by Gaussian log-likelihood summed over D1, D2.
  2. Frozen probabilistic forecast: log score of ln T_obs under N(mu, s^2),
     PIT, inside the 68%/95% intervals; for the 2e5 and 1e5 targets.
No parameter of any closure or forecast is changed here.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import multifidelity_analysis as MA  # noqa: E402
import nsphere_ext as NX  # noqa: E402
from scipy.stats import norm  # noqa: E402

PRE = json.load(open(os.path.join(EXT, 'closure_design', 'PREREGISTRATION_D.json')))
FC = json.load(open(os.path.join(EXT, 'collapse_probability', 'FORECAST_D.json')))


def runs_for(h):
    out = []
    for run, cfg in NX.RUNS.items():
        if cfg.get('halo') != h:
            continue
        p = os.path.join(NX.run_dir(run), 'measures.npz')
        if not os.path.exists(p):
            continue
        d = np.load(p)
        t_tdyn = d['t'] * float(d['t0_myr']) / float(d['tdyn_myr'])
        y = d['mean_rho'][:, 1]
        MA.TREF[h] = 70.0
        ric = None
        mk = {'t_x10': MA.IH.collapse_markers(t_tdyn, y)['t_x10']}
        ts, ys = MA.smooth(t_tdyn, y, 0.015 * 70.0)
        mk['t_x10_s'] = MA.IH.collapse_markers(ts, ys)['t_x10']
        out.append(dict(run=run, N=cfg['n'], dt2=run.split('_')[1].startswith('dt2'), seed=cfg['seed'], **mk))
    return out


def main():
    res = {'haloes': {}, 'closure_loglik': {}}
    ll = {}
    for h in ('D1', 'D2'):
        rows = runs_for(h)
        if not rows:
            continue
        R = {'runs': rows}
        for mk in ('t_x10', 't_x10_s'):
            lev = {}
            for N in (100_000, 200_000):
                v = [r[mk] for r in rows if r['N'] == N and not r['dt2'] and r[mk]]
                if v:
                    lev[N] = dict(n=len(v), mean=float(np.exp(np.mean(np.log(v)))),
                                  sd_ln=float(np.std(np.log(v), ddof=1)) if len(v) > 1 else None)
            R[mk] = lev
        dt2 = [r for r in rows if r['dt2']]
        if dt2:
            base = [r for r in rows if r['seed'] == dt2[0]['seed'] and not r['dt2'] and r['N'] == dt2[0]['N']]
            if base and base[0]['t_x10'] and dt2[0]['t_x10']:
                R['dt2_rel'] = dt2[0]['t_x10'] / base[0]['t_x10'] - 1
        # target: prefer 2e5 mean, else 1e5 mean (flagged)
        lev = R['t_x10']
        N_t = 200_000 if 200_000 in lev else 100_000
        obs = lev[N_t]
        sds = [l['sd_ln'] for l in lev.values() if l['sd_ln']]
        tau = max(float(np.sqrt(np.mean(np.square(sds)))) if sds else 0.05, 0.03)
        sig_y = float(np.sqrt(tau ** 2 / obs['n'] + 0.03 ** 2 + (0.05 ** 2 if N_t == 100_000 else 0)))
        off = 0.0 if N_t == 200_000 else -0.05
        R['target'] = dict(N=N_t, T_obs=obs['mean'], n=obs['n'], tau=tau, sigma_y=sig_y)
        R['closures'] = {}
        for name, p in PRE['haloes'][h]['predictions'].items():
            z = (np.log(obs['mean']) - (np.log(p['t_x10_tdyn']) + off)) / sig_y
            R['closures'][name] = dict(T_pred=p['t_x10_tdyn'], rel_err=obs['mean'] / p['t_x10_tdyn'] - 1, z=float(z))
            ll[name] = ll.get(name, 0.0) + float(norm.logpdf(z) - np.log(sig_y))
        tgt = 'N2e5_mean_of_2' if N_t == 200_000 else 'N1e5_mean_of_3'
        f = FC['haloes'][h]['targets'][tgt]
        m, s = np.log(f['median']), f['sd_ln']
        x = np.log(obs['mean'])
        R['forecast'] = dict(target=tgt, log_score=float(norm.logpdf(x, m, s)), pit=float(norm.cdf(x, m, s)),
                             in68=bool(f['ci68'][0] <= obs['mean'] <= f['ci68'][1]),
                             in95=bool(f['ci95'][0] <= obs['mean'] <= f['ci95'][1]))
        res['haloes'][h] = R
    res['closure_loglik'] = dict(sorted(ll.items(), key=lambda kv: -kv[1]))
    json.dump(res, open(os.path.join(EXT, 'closure_design', 'score_D.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1)[:6000])


if __name__ == '__main__':
    main()
