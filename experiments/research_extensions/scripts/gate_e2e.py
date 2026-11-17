#!/usr/bin/env python3
"""End-to-end test of the solver-fallback rule (manuscript Algorithm 3).

Baryon-aware emulator (three members, results/baryons_20261107/runs), served on
its own clock. Workload fixed in advance and independent of the true terminal
time: every halo is queried at the same 40 physical times, log-spaced from 10 to
1200 t0. A query is one density profile (128 radii); its error and its
uncertainty are RMS values over radii in dex.

Rule: defer to the solver if the predicted phase t / t_hat > 1; otherwise
serve mu +- h with h = q * s, and serve only if h <= budget b; else defer.
Scores s: ensemble spread, and spread with the clock error propagated
(Eq. 5 of the manuscript). Calibration on the validation haloes, split in two
fixed halves: sigma_c from one half, the conformal multiplier q from the other
(per-trajectory 90th percentile of error / s, quantile 0.9 over trajectories).

Truth: the solver trajectory interpolated to phase t / t_end. A query after the
true terminal time has no fluid answer (the solver stops there: collapse,
Knudsen floor or horizon) and must be deferred; serving it counts as unsafe.
Cost: emulator wall time measured here for the whole rule (three members,
three phases each), solver wall time of each halo from the dataset record; one
solver run answers every deferred query of its halo. All costs are estimates
from these two measured quantities; no fallback solve is executed here.

Single-query workload (one query per halo at a time drawn uniformly from the
40): `speedup_single_query` uses the mean deferred fraction and the mean
solver time; `speedup_single_query_weighted` weights each halo's deferred
fraction by its own solver time, sum_i w_i / sum_i (d_i w_i + c_emu), which
keeps the covariance between deferral and solver cost.

Writes experiments/research_extensions/clock_uq/gate_e2e.json and the per-halo
deferred fractions and solver times behind every cost ratio in gate_e2e.npz.
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(EXT, '..', '..'))
sys.path[:0] = [HERE, os.path.join(ROOT, 'scripts'), os.path.join(ROOT, 'src')]
import clock_uq as U  # noqa: E402

T_Q = np.logspace(1, np.log10(1200), 40)
BUDGETS = (0.01, 0.02, 0.05, 0.1)
EPS = 0.02


def evaluate_split(d, ids, members, q):
    import train_baryon_emulator as T
    n0 = members[0][1]
    _, _, prof, par, _, _ = T.tensors(d, n0, U.DEV, baryon_inputs=True)
    prof, par = prof[ids], par[ids]
    c_m = np.array([U.clocks(m, n, prof, par) for m, n in members])
    c_bar = np.exp(np.log(c_m).mean(0))
    c_true = 10 ** d['log_tc'][ids]
    tau_hat = T_Q[None, :] / c_bar[:, None]
    tau_true = T_Q[None, :] / c_true[:, None]
    P = np.array([U.predict(m, n, prof, par, tau_hat) for m, n in members])
    mu, s_base = P.mean(0), P.std(0)
    mp = np.array([U.predict(m, n, prof, par, tau_hat * np.exp(EPS)) for m, n in members]).mean(0)
    mm = np.array([U.predict(m, n, prof, par, tau_hat * np.exp(-EPS)) for m, n in members]).mean(0)
    dmu = (mp - mm) / (2 * EPS)
    lr = d['lrho'][ids]                                         # (B, 48, R) on the tau grid
    truth = np.empty_like(mu)
    for b in range(len(ids)):
        tt = np.clip(tau_true[b], 0, 1)
        for r in range(lr.shape[2]):
            truth[b, :, r] = np.interp(tt, q, lr[b, :, r])
    err = np.sqrt(np.mean((mu - truth) ** 2, axis=2))
    return dict(c_bar=c_bar, c_true=c_true, tau_hat=tau_hat, post_event=tau_true > 1.0, err=err,
                s_base_f=s_base, dmu_f=dmu, wall=d['wall'][ids])


def scores(S, sigma_c):
    return dict(s_base=np.sqrt(np.mean(S['s_base_f'] ** 2, axis=2)),
                s_prop=np.sqrt(np.mean(S['s_base_f'] ** 2 + (S['dmu_f'] * sigma_c) ** 2, axis=2)))


def main():
    import train_baryon_emulator as T
    from gravop import dataset as ds
    d = T.load()
    _, va, te, ho = T.splits(d)
    q = ds.tau_grid()
    members = [U.load_member(os.path.join(ROOT, f'results/baryons_20261107/runs/seed{s}/model.pt')) for s in range(3)]
    perm = np.random.default_rng(20261112).permutation(len(va))
    fit_ids, conf_ids = va[perm[: len(va) // 2]], va[perm[len(va) // 2:]]
    Sf = evaluate_split(d, fit_ids, members, q)
    sigma_c = float(np.sqrt(np.mean(np.log(Sf['c_bar'] / Sf['c_true']) ** 2)))
    Sc = evaluate_split(d, conf_ids, members, q)
    out = dict(workload='40 physical times log-spaced 10-1200 t0 per halo', sigma_c=sigma_c,
               n_fit=int(len(fit_ids)), n_conformal=int(len(conf_ids)), budgets_dex=BUDGETS, splits={})
    sc_c = scores(Sc, sigma_c)
    mult = {}
    for k, s in sc_c.items():
        ok = (Sc['tau_hat'] <= 1) & ~Sc['post_event']
        r = [np.percentile(Sc['err'][b][ok[b]] / np.maximum(s[b][ok[b]], 1e-6), 90) for b in range(len(conf_ids)) if ok[b].any()]
        r = np.sort(r); kq = int(np.ceil((len(r) + 1) * 0.9)) - 1
        mult[k] = float(r[min(kq, len(r) - 1)])
    out['multiplier'] = mult
    # timing of the full rule on the GPU (three members, three phases), per query
    import train_baryon_emulator as T2
    _, _, prof, par, _, _ = T2.tensors(d, members[0][1], U.DEV, baryon_inputs=True)
    tq = np.tile(T_Q, (64, 1)) / 300.0
    for _ in range(2):
        [U.predict(m, n, prof[:64], par[:64], tq * f) for m, n in members for f in (1, np.exp(EPS), np.exp(-EPS))]
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    [U.predict(m, n, prof[:64], par[:64], tq * f) for m, n in members for f in (1, np.exp(EPS), np.exp(-EPS))]
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    emu_per_query = (time.perf_counter() - t0) / (64 * len(T_Q))
    out['emulator_s_per_query'] = emu_per_query
    rng = np.random.default_rng(0)
    arrays = {}
    for name, ids in (('test', te), ('held_out', ho)):
        S = evaluate_split(d, ids, members, q)
        sc = scores(S, sigma_c)
        arrays[f'{name}_solver_wall_s'] = S['wall']
        res = {'n_haloes': int(len(ids)), 'post_event_frac': float(S['post_event'].mean()),
               'unsupported_frac': float((S['tau_hat'] > 1).mean()), 'solver_s_per_halo_median': float(np.median(S['wall']))}
        for k, s in sc.items():
            h = mult[k] * s
            for b in BUDGETS:
                serve = (S['tau_hat'] <= 1) & (h <= b)
                unsafe = serve & (S['post_event'] | (S['err'] > b))
                deferred_halo = (~serve).any(1)
                cost_h = emu_per_query * serve.shape[1] * len(ids) + float(S['wall'][deferred_halo].sum())
                cost_s = float(S['wall'].sum())

                def stat(idx):
                    sv = serve[idx].sum()
                    return unsafe[idx].sum() / max(sv, 1)
                boot = [stat(rng.integers(0, len(ids), len(ids))) for _ in range(500)]
                # variant: queries beyond the predicted event are answered 'collapsed' without a solver run
                inlife = S['tau_hat'] <= 1
                need_inlife = (inlife & ~serve).any(1)
                cost_h2 = emu_per_query * serve.shape[1] * len(ids) + float(S['wall'][need_inlife].sum())
                wrong_post = (~inlife) & ~S['post_event']          # 'collapsed' answered too early
                # single-query workload (one query per halo, e.g. one observed epoch): expected cost ratio
                f_def = float((~serve).mean())
                single = 1.0 / (f_def + emu_per_query / float(np.mean(S['wall'])))
                d_i = (~serve).mean(1)
                single_w = float(S['wall'].sum() / (d_i * S['wall'] + emu_per_query).sum())
                arrays[f'{name}_{k}@{b}_deferred_frac'] = d_i
                res[f'{k}@{b}'] = dict(served_frac=float(serve.mean()), unsafe_of_served=float(stat(np.arange(len(ids)))),
                                       unsafe_of_served_68=[float(np.percentile(boot, 16)), float(np.percentile(boot, 84))],
                                       unsafe_of_all=float(unsafe.mean()), haloes_needing_solver=float(deferred_halo.mean()),
                                       speedup=cost_s / cost_h, haloes_needing_solver_inlife=float(need_inlife.mean()),
                                       speedup_inlife=cost_s / cost_h2, premature_collapsed_frac=float(wrong_post.mean()),
                                       speedup_single_query=single,
                                       speedup_single_query_weighted=single_w)
                print(name, k, b, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in res[f'{k}@{b}'].items()})
        out['splits'][name] = res
    json.dump(out, open(os.path.join(EXT, 'clock_uq', 'gate_e2e.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(EXT, 'clock_uq', 'gate_e2e.npz'), **arrays)
    print('sigma_c', sigma_c, 'mult', mult, 'emu s/query', emu_per_query)


if __name__ == '__main__':
    main()
