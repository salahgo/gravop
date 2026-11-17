#!/usr/bin/env python3
"""Track B: disjoint-system transfer of the quadrature rule. Protocol: docs/DISJOINT_PREREGISTRATION.md.

Modes (each run directly, e.g. `d1_campaign.py score --out <dir>`; each writes <out>/summary.json):
  validate   reproduce results/statecond/decomposition.json for the five original FNO points with
             the per-trajectory decomposition used here, and agree with sc_decomposition.run
  generate   600 draws per new system (sc_gen.generate, unchanged), seed block 70_000_000 + 100_000 k
  system     one system: sc_experiment.py (unchanged; FNO, 3 seeds, 200 epochs) -> converged clock
             (sc_timehead.train_head, budgets 200/2000/20000, largest budget used) -> decomposition
  score      every criterion of the protocol, from stored per-trajectory sums only
  report     always runs; lists every job, gate and number

Nothing in src/statecond, sc_experiment.py, sc_timehead.py or sc_decomposition.py is modified.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / 'src'))

SMOKE = os.environ.get('D1_SMOKE') == '1'
PLUGIN = str(HERE / 'd1_systems.py')
ORDER = ['quench', 'pme', 'cgl', 'hj', 'corr', 'bimodal']
TIER1, TIER2 = ORDER[:4], ORDER[4:]
ORIGINAL = ['rd', 'ks', 'film', 'burgers', 'sidm']
N_DRAWS = 40 if SMOKE else 600
SEED0 = 70_000_000
SEEDS = 2 if SMOKE else 3
EPOCHS = 2 if SMOKE else 200
BUDGETS = [2, 4] if SMOKE else [200, 2000, 20000]
B_BOOT = 200 if SMOKE else 4000
MIN_KEPT = 20 if SMOKE else 300
TRAIN_FAIL_NRMSE = 0.25
RES = ROOT / 'results' / 'statecond'


def atomic_json(path, obj):
    path = Path(path); tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=1, default=float)); os.replace(tmp, path)


def summary(a, passed, **kw):
    atomic_json(a.out / 'summary.json', dict(valid_output=True, passed=bool(passed) or SMOKE, gate_value=bool(passed), smoke=SMOKE, **kw))


# ----------------------------------------------------------------------------- decomposition
def decompose_sums(name, data_path, ckpt_dir, dev, seeds):
    """sc_decomposition.run, with every norm kept as a per-test-trajectory sum of squares.

    Returns dict of arrays (seeds, n_test): D2 = sum Delta^2, Ed2 = sum e(v~)^2, EA2 = sum e(v)^2,
    Dep2 = sum (Ghat(v~) - G)^2, X = sum Delta * e(v~); plus n_el elements per trajectory."""
    import torch
    import statecond.data as scd
    from sc_experiment import Norm, tensors, load_checkpoint
    from sc_decomposition import fields_at
    d = scd.load(str(data_path))
    tr, _, te = scd.split(d['u'].shape[0])
    n = Norm(d, tr); T = tensors(d, n, dev); u, u0, par, phi, flag, lts = T
    taus = scd.tau_grid(); n_tau = len(taus); taus_t = torch.tensor(taus, dtype=torch.float32, device=dev)
    ri = np.asarray(te); G = u[ri].cpu().numpy(); un = u.cpu().numpy()
    out = {k: [] for k in ('D2', 'Ed2', 'EA2', 'Dep2', 'X')}; used = []
    for s in range(seeds):
        tau_ck = os.path.join(ckpt_dir, f'{name}_fno_tau_seed{s}.pt'); head_ck = os.path.join(ckpt_dir, f'{name}_fno_time_conv_seed{s}.pt')
        if not (os.path.exists(tau_ck) and os.path.exists(head_ck)):
            continue
        net, _ = load_checkpoint(tau_ck, dev); head, _ = load_checkpoint(head_ck, dev)
        with torch.no_grad():
            lts_hat = head(u0, par).squeeze(-1)
        ratio = torch.pow(10.0, (lts - lts_hat) * n.t_sd).cpu().numpy()
        tq = np.clip(taus[None, :] * ratio[ri][:, None], 0.0, 1.0)
        Gv = np.stack([np.stack([np.interp(tq[i], taus, un[ri[i], :, c, x]) for x in range(un.shape[3])], -1)
                       for i in range(len(ri)) for c in range(un.shape[2])])
        Gv = Gv.reshape(len(ri), un.shape[2], n_tau, un.shape[3]).transpose(0, 2, 1, 3)
        pos = {int(x): j for j, x in enumerate(ri)}

        def cond_true(r, t):
            return torch.cat([par[r], taus_t[t][:, None]], 1)

        def cond_dep(r, t):
            q = torch.tensor(tq[[pos[int(x)] for x in r], t], dtype=torch.float32, device=dev)
            return torch.cat([par[r], q[:, None]], 1)
        Ghat_v = fields_at(net, cond_true, u0, te, n_tau, dev); Ghat_dep = fields_at(net, cond_dep, u0, te, n_tau, dev)
        Delta = Gv - G; e_dep = Ghat_dep - Gv; e_true = Ghat_v - G; dep = Ghat_dep - G
        ax = (1, 2, 3)
        out['D2'].append((Delta ** 2).sum(ax)); out['Ed2'].append((e_dep ** 2).sum(ax)); out['EA2'].append((e_true ** 2).sum(ax))
        out['Dep2'].append((dep ** 2).sum(ax)); out['X'].append((Delta * e_dep).sum(ax)); used.append(s)
        del net, head
    res = {k: np.asarray(v, float) for k, v in out.items()}
    res['n_el'] = int(np.prod(G.shape[1:])); res['seeds_used'] = used; res['test_index'] = ri
    return res


def per_seed_from_sums(S, idx=None):
    """sc_decomposition's per-seed row (F, e_dep, E_A, E_dep, cross) from the sums, optionally on a trajectory resample."""
    sel = (lambda a: a) if idx is None else (lambda a: a[:, idx])
    m = sel(S['D2']).shape[1] * S['n_el']
    rows = []
    for s in range(S['D2'].shape[0]):
        f = lambda k: float(sel(S[k])[s].sum()) / m
        rows.append(dict(F=np.sqrt(f('D2')), e_dep=np.sqrt(f('Ed2')), E_A=np.sqrt(f('EA2')), E_dep=np.sqrt(f('Dep2')), cross=2 * f('X')))
    return rows


def aggregate(rows):
    """sc_decomposition.aggregate: RMS over seeds of each norm, mean of cross; identity must close."""
    rms = lambda k: float(np.sqrt(np.mean([r[k] ** 2 for r in rows])))
    F, Ee, EA, Ed = rms('F'), rms('e_dep'), rms('E_A'), rms('E_dep'); cross = float(np.mean([r['cross'] for r in rows]))
    closure = (F ** 2 + Ee ** 2 + cross) / max(Ed ** 2, 1e-30)
    return dict(F=F, e_dep=Ee, E_A=EA, E_dep=Ed, cross=cross, c=cross / Ed ** 2, k=Ee / EA, g=F / EA, p=Ed / EA,
                p_hat=float(np.hypot(EA, F) / EA), closure=closure)


def residuals(agg, log_p_const):
    r = float(np.log10(agg['p_hat'] / agg['p']))
    ident = 0.5 * np.log10((1 + agg['g'] ** 2) * (1 - agg['c']) / (agg['g'] ** 2 + agg['k'] ** 2))
    return dict(r_quadrature=r, r_identity=float(ident), r_constant=float(log_p_const - np.log10(agg['p'])),
                r_no_penalty=float(-np.log10(agg['p'])), r_bare=float(np.log10(agg['g'] / agg['p'])))


def per_trajectory_residuals(S):
    """Per-test-trajectory penalty and rule residual, norms aggregated over seeds as above."""
    n = S['n_el']; f = lambda k: S[k].mean(0) / n
    F, EA, Ed = np.sqrt(f('D2')), np.sqrt(f('EA2')), np.sqrt(f('Dep2'))
    return np.log10(np.hypot(EA, F) / Ed)


def save_sums(path, S):
    np.savez(path, **{k: np.asarray(v) for k, v in S.items()})


def load_sums(path):
    z = np.load(path); S = {k: z[k] for k in z.files}; S['n_el'] = int(S['n_el']); return S


# ----------------------------------------------------------------------------- modes
def validate(a):
    """Gate: the per-trajectory decomposition reproduces the archived FNO points and sc_decomposition.run."""
    import sc_decomposition as sd
    dec = json.loads((RES / 'decomposition.json').read_text()); rows = {}; ok = True
    for name in (['rd'] if SMOKE else ORIGINAL):
        S = decompose_sums(name, ROOT / 'data' / 'sc' / f'{name}.h5', str(RES / 'ckpt'), a.device, 3)
        mine = aggregate(per_seed_from_sums(S)); ref = dec[f'{name}_fno']
        theirs = sd.run(name, 'fno', 3, a.device)
        rel_arch = max(abs(mine[k] / ref[k] - 1) for k in ('F', 'e_dep', 'E_A', 'E_dep'))
        rel_run = max(abs(x[k] / y[k] - 1) for x, y in zip(per_seed_from_sums(S), theirs) for k in ('F', 'e_dep', 'E_A', 'E_dep'))
        cross = max(abs(x['cross'] - y['cross']) / y['E_dep'] ** 2 for x, y in zip(per_seed_from_sums(S), theirs))
        rows[name] = dict(rel_vs_archive=rel_arch, rel_vs_sc_decomposition=rel_run, cross_vs_sc_decomposition=cross, closure=mine['closure'])
        ok &= rel_arch < 1e-4 and rel_run < 1e-6 and cross < 1e-6 and abs(mine['closure'] - 1) < 1e-6
        print(name, rows[name], flush=True)
    atomic_json(a.out / 'validation.json', rows)
    summary(a, ok, scope='Per-trajectory decomposition reproduces decomposition.json (FNO, rel < 1e-4) and sc_decomposition.run (rel < 1e-6).')


def generate(a):
    import sc_gen
    import statecond.data as scd
    from statecond.pdes import SYSTEMS
    import d1_systems  # noqa: F401  (registers the systems in this process)
    info = {}
    for k, name in enumerate(ORDER):
        d, meta = sc_gen.generate(name, N_DRAWS, SEED0 + 100_000 * k, a.workers)
        scd.save(str(a.out / f'{name}.h5'), d)
        meta.update(seed0=SEED0 + 100_000 * k, log10_tsing_percentiles=np.percentile(d['log_tsing'], [0, 10, 50, 90, 100]).tolist(),
                    fields=int(d['u'].shape[2]), status_counts=np.bincount(d['status']).tolist())
        info[name] = meta; print(name, meta, flush=True)
    atomic_json(a.out / 'manifest.json', info)
    enough = {n: info[n]['kept'] >= MIN_KEPT for n in ORDER}
    summary(a, all(enough.values()), kept={n: info[n]['kept'] for n in ORDER}, enough=enough,
            scope=f'{N_DRAWS} draws per system; a system with fewer than {MIN_KEPT} singular runs is reported as insufficient.')


def system(a):
    import torch
    import statecond.data as scd
    from sc_experiment import Norm, tensors, save_checkpoint
    from sc_timehead import train_head, head_error
    name = a.system; data = a.out.parent / 'generate' / f'{name}.h5'; ck = a.out / 'ckpt'; ck.mkdir(exist_ok=True)
    t0 = time.time()
    cmd = [sys.executable, '-u', str(HERE / 'sc_experiment.py'), '--plugin', PLUGIN, '--system', name, '--seeds', str(SEEDS), '--epochs', str(EPOCHS),
           '--device', a.device, '--data', str(data.parent), '--ckpt-dir', str(ck), '--results-dir', str(a.out)]
    subprocess.run(cmd, check=True)
    arms = json.loads((a.out / f'{name}.json').read_text()); t_arms = time.time() - t0
    d = scd.load(str(data)); n_runs, n_tau, n_fields, n_x = d['u'].shape; n_par = d['par'].shape[1]
    tr, va, te = scd.split(n_runs); n = Norm(d, tr); T = tensors(d, n, a.device)
    curve = {}
    for ep in BUDGETS:
        errs = []
        for s in range(SEEDS):
            h = train_head(T, tr, s, a.device, ep, n_fields, n_par)
            e, _ = head_error(h, T, te, n, d['log_tsing'], a.device); errs.append(e)
            if ep == max(BUDGETS):
                save_checkpoint(str(ck / f'{name}_fno_time_conv_seed{s}.pt'), h, 'time', 'fno', n, s, ep, (n_fields, n_par, n_x))
            del h
        curve[str(ep)] = dict(mean=float(np.mean(errs)), per_seed=[float(x) for x in errs]); print(name, ep, curve[str(ep)], flush=True)
    S = decompose_sums(name, data, str(ck), a.device, SEEDS); save_sums(a.out / 'sums.npz', S)
    agg = aggregate(per_seed_from_sums(S))
    trained = arms['nrmse']['A_tau_oracle']['mean'] < TRAIN_FAIL_NRMSE
    atomic_json(a.out / 'system.json', dict(system=name, arms_nrmse={k: v['mean'] for k, v in arms['nrmse'].items()}, published_head_err_pct=arms['tsing_median_err_pct'],
                                             clock_curve=curve, converged_head_err_pct=curve[str(max(BUDGETS))]['mean'], aggregate=agg, trained=trained,
                                             n_runs=int(n_runs), n_test=int(len(te)), wall_seconds=time.time() - t0, arms_seconds=t_arms))
    summary(a, trained and abs(agg['closure'] - 1) < 1e-6, closure=agg['closure'], trained=trained, penalty=agg['p'],
            scope='Frozen recipe trains (oracle nRMSE < 0.25) and the exact decomposition closes (< 1e-6).')


def score(a):
    dec = json.loads((RES / 'decomposition.json').read_text())
    log_p_const = float(np.mean([np.log10(v['E_dep'] / v['E_A']) for v in dec.values()]))  # R0, fitted on the 15 original points
    tasks = a.out.parent
    orig = {}
    for nm in ORIGINAL:
        v = dec[f'{nm}_fno']; agg = dict(F=v['F'], e_dep=v['e_dep'], E_A=v['E_A'], E_dep=v['E_dep'], cross=v['cross'], c=v['chi'], k=v['rho'],
                                          g=v['F'] / v['E_A'], p=v['E_dep'] / v['E_A'], p_hat=v['quadrature'] / v['E_A'], closure=v['closure'])
        orig[nm] = dict(aggregate=agg, **residuals(agg, log_p_const))
    new, sums, status = {}, {}, {}
    for nm in ORDER:
        sj = tasks / f'system_{nm}' / 'summary.json'
        if not sj.exists() or not json.loads(sj.read_text()).get('passed'):
            status[nm] = 'missing or failed'; continue
        S = load_sums(tasks / f'system_{nm}' / 'sums.npz'); sums[nm] = S
        agg = aggregate(per_seed_from_sums(S)); rt = per_trajectory_residuals(S)
        new[nm] = dict(aggregate=agg, **residuals(agg, log_p_const), per_trajectory_rms=float(np.sqrt(np.mean(rt ** 2))),
                       per_trajectory_median_abs=float(np.median(np.abs(rt))), per_trajectory_median=float(np.median(rt)), n_test=int(len(rt)),
                       no_penalty_to_predict=bool(agg['p'] < 1.05))
        status[nm] = 'scored'
    t1 = [n for n in TIER1 if n in new]
    out = dict(protocol='docs/DISJOINT_PREREGISTRATION.md', log10_constant_penalty=log_p_const, original_fno=orig, new=new, status=status)
    crit = {}
    # identity (mechanistic): residual from (c, k, g) equals the measured residual
    allr = list(orig.values()) + list(new.values())
    crit['mechanistic'] = dict(max_abs_identity_minus_measured=float(max(abs(x['r_identity'] - x['r_quadrature']) for x in allr)))
    crit['mechanistic']['passed'] = crit['mechanistic']['max_abs_identity_minus_measured'] < 1e-6
    # primary: Tier-1 mean |r|, against the constant, trajectory bootstrap within system
    if len(t1) == 4:
        M = float(np.mean([abs(new[n]['r_quadrature']) for n in t1])); Mc = float(np.mean([abs(new[n]['r_constant']) for n in t1]))
        rng = np.random.default_rng(20261106); bq, bc = [], []
        for _ in range(B_BOOT):
            rq, rc = [], []
            for n in t1:
                S = sums[n]; idx = rng.integers(0, S['D2'].shape[1], S['D2'].shape[1])
                g = aggregate(per_seed_from_sums(S, idx)); rr = residuals(g, log_p_const)
                rq.append(abs(rr['r_quadrature'])); rc.append(abs(rr['r_constant']))
            bq.append(np.mean(rq)); bc.append(np.mean(rc))
        bq, bc = np.array(bq), np.array(bc); diff = bc - bq
        crit['primary'] = dict(mean_abs_residual=M, interval=np.percentile(bq, [2.5, 97.5]).tolist(), constant_mean_abs_residual=Mc,
                               advantage_over_constant=Mc - M, advantage_interval=np.percentile(diff, [2.5, 97.5]).tolist(), threshold=0.10)
        crit['primary']['passed'] = bool(M <= 0.10 and np.percentile(diff, 2.5) > 0)
        # permutation null: F of one Tier-1 system with E_A of another; the rule's advantage must vanish
        perm = []
        for i in t1:
            for j in t1:
                if i != j:
                    ai, aj = new[i]['aggregate'], new[j]['aggregate']
                    perm.append(abs(np.log10(np.hypot(aj['E_A'], ai['F']) / aj['E_A'] / ai['p'])))
        crit['permutation_null'] = dict(mean_abs_residual=float(np.mean(perm)), passed=bool(np.mean(perm) >= Mc))
    else:
        crit['primary'] = dict(passed=False, reason=f'only {len(t1)} Tier-1 systems scored')
    # concentration and envelopes
    c5 = [orig[n]['aggregate']['c'] for n in ORIGINAL]; k5 = [orig[n]['aggregate']['k'] for n in ORIGINAL]
    c9 = c5 + [new[n]['aggregate']['c'] for n in t1]; k9 = k5 + [new[n]['aggregate']['k'] for n in t1]
    rng_ = lambda v: max(v) - min(v)
    crit['concentration'] = dict(c_range_5=rng_(c5), c_range_9=rng_(c9), k_range_5=rng_(k5), k_range_9=rng_(k9), envelope_9=dict(c=[min(c9), max(c9)], k=[min(k9), max(k9)]))
    crit['concentration']['passed'] = bool(len(t1) == 4 and rng_(c9) <= 1.25 * rng_(c5) and rng_(k9) <= 1.25 * rng_(k5))

    def interval(g, c_lo, c_hi, k_lo, k_hi):
        r = lambda c, k: 0.5 * np.log10((1 + g ** 2) * (1 - c) / (g ** 2 + k ** 2))
        return [float(r(c_hi, k_hi)), float(r(c_lo, k_lo))]
    nft = {}
    for n in new:
        iv = interval(new[n]['aggregate']['g'], min(c5), max(c5), min(k5), max(k5)); r = new[n]['r_quadrature']
        nft[n] = dict(interval_from_g=iv, measured=r, inside=bool(iv[0] <= r <= iv[1]))
    crit['no_false_trust'] = dict(envelope_original_five=dict(c=[min(c5), max(c5)], k=[min(k5), max(k5)]), systems=nft,
                                  passed=bool(len(t1) == 4 and all(nft[n]['inside'] for n in t1)))
    env = crit['concentration']['envelope_9']; outside = lambda n: not (env['c'][0] <= new[n]['aggregate']['c'] <= env['c'][1] and env['k'][0] <= new[n]['aggregate']['k'] <= env['k'][1])
    adv = {}
    if 'corr' in new:
        x = new['corr']; adv['corr'] = dict(r=x['r_quadrature'], c=x['aggregate']['c'], outside_envelope=outside('corr'),
                                            predicted_direction=bool(x['r_quadrature'] < -0.2 and x['aggregate']['c'] > 0))
        adv['corr']['passed'] = adv['corr']['predicted_direction'] and adv['corr']['outside_envelope']
    if 'bimodal' in new:
        x = new['bimodal']; gap = bool(x['per_trajectory_rms'] > 0.2 and x['per_trajectory_median_abs'] < 0.5 * x['per_trajectory_rms'])
        adv['bimodal'] = dict(per_trajectory_rms=x['per_trajectory_rms'], per_trajectory_median_abs=x['per_trajectory_median_abs'], gap=gap, outside_envelope=outside('bimodal'))
        adv['bimodal']['passed'] = gap and adv['bimodal']['outside_envelope']
    crit['adversarial'] = dict(systems=adv, passed=bool(len(adv) == 2 and all(v['passed'] for v in adv.values())))
    out['criteria'] = crit
    atomic_json(a.out / 'track_b.json', out)
    summary(a, crit['primary']['passed'], criteria={k: v.get('passed') for k, v in crit.items()},
            scope='Primary: Tier-1 mean |r| <= 0.10 dex and better than the original-five constant with the bootstrap interval excluding parity.')


def report(a):
    tasks = a.out.parent; lines = ['# Track B: disjoint-system transfer, final report', '', f'Generated {time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}. Smoke: {SMOKE}.', '',
                                   '| Job | Summary |', '|---|---|']
    for d in sorted(tasks.iterdir()):
        s = d / 'summary.json'
        lines.append(f'| {d.name} | {json.dumps(json.loads(s.read_text())) if s.exists() else "missing"} |')
    tb = tasks / 'score' / 'track_b.json'
    if tb.exists():
        T = json.loads(tb.read_text()); lines += ['', '## Per system (converged clock, FNO, seed-aggregated)', '',
                                                   '| System | p | p_hat | r quad | r const | r none | c | k | g | per-traj RMS / median abs |', '|---|---|---|---|---|---|---|---|---|---|']
        for grp in ('original_fno', 'new'):
            for n, x in T[grp].items():
                g = x['aggregate']; pt = f"{x['per_trajectory_rms']:.3f} / {x['per_trajectory_median_abs']:.3f}" if 'per_trajectory_rms' in x else '-'
                lines.append(f"| {n} | {g['p']:.3f} | {g['p_hat']:.3f} | {x['r_quadrature']:+.3f} | {x['r_constant']:+.3f} | {x['r_no_penalty']:+.3f} | {g['c']:+.3f} | {g['k']:.3f} | {g['g']:.3f} | {pt} |")
        lines += ['', '## Criteria', '', '```', json.dumps(T['criteria'], indent=1), '```']
    (a.out / 'REPORT_FINAL.md').write_text('\n'.join(lines) + '\n')
    summary(a, True, scope='Final report; lists every job.')


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode'); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--system', default=None); p.add_argument('--device', default='cuda:0'); p.add_argument('--workers', type=int, default=6)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(1)
    {'validate': validate, 'generate': generate, 'system': system, 'score': score, 'report': report}[a.mode](a)


if __name__ == '__main__':
    main()
