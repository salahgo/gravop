"""P32 -- the branch flag, tested causally.

Trains arm C with the flag channel ZEROED rather than removed, so the network
keeps `cond_extra = 2` and an identical parameter count. Capacity is held fixed
and only the branch information is destroyed, which is the control
an earlier risk review asked for.

`scripts/sc_experiment.py` and `src/statecond/model.py` are pinned by the
running F9 calibration, so neither is modified: the flag array is zeroed in
memory before `tensors()`.

Pre-registration: docs/FLAG_ABLATION_PREREGISTRATION.md (committed first).
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from sc_experiment import Norm, tensors, train_operator, field_err
import statecond.data as scd

DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'sc')
OUT = os.path.join(os.path.dirname(__file__), '..', 'results', 'statecond')
SYSTEMS = ['rd', 'ks', 'film', 'burgers', 'sidm']


def run(name, seeds, dev, epochs, backbone='fno'):
    d = scd.load(os.path.join(DATA, f'{name}.h5'))
    tr, _, te = scd.split(d['u'].shape[0])
    n = Norm(d, tr)
    out = {}
    for tag, zero in (('with_flag', False), ('no_flag', True)):
        dd = dict(d)
        if zero:
            dd['flag'] = np.zeros_like(np.asarray(d['flag']))
        T = tensors(dd, n, dev)
        u, u0, par, phi, flag, _ = T
        n_fields, n_par = u.shape[2], par.shape[1]
        errs = []
        for s in range(seeds):
            net, cond, _ = train_operator('state', T, tr, te, s, dev, epochs,
                                          n_fields, n_par, backbone)
            errs.append(float(field_err(net, cond, u, u0, te).mean()))
            del net
            torch.cuda.empty_cache()
        out[tag] = {'per_seed': errs, 'mean': float(np.mean(errs))}
    out['ratio'] = out['no_flag']['mean'] / out['with_flag']['mean']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--systems', nargs='+', default=SYSTEMS)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    # F9 owns the CPU; this run must stay on the GPU.
    torch.set_num_threads(2)
    res = {}
    for name in a.systems:
        res[name] = run(name, a.seeds, a.device, a.epochs)
        r = res[name]
        print(f"{name:8s} with {r['with_flag']['mean']:.5f}  "
              f"no-flag {r['no_flag']['mean']:.5f}  ratio {r['ratio']:.2f}x",
              flush=True)
    json.dump(res, open(os.path.join(OUT, 'flag_ablation.json'), 'w'), indent=1)
    print('wrote flag_ablation.json')


if __name__ == '__main__':
    main()
