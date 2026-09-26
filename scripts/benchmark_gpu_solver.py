"""Bounded, reproducible CPU/GPU accuracy and throughput decision."""
import argparse
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
from contextlib import contextmanager
import hashlib,json,multiprocessing,os,signal,time
from pathlib import Path
import numpy as np
import torch
from gpu_solver import GPUSolver
from forecast_numerics import refined_row,LEVELS
from forecast_calibrate import LO,HI
from e9_inference import SOLVER_KW
from gravop import solver
ROOT=Path(__file__).resolve().parents[1]


def save(path,data):
    p=path.with_suffix('.tmp');p.write_text(json.dumps(data,indent=2)+'\n');p.replace(path)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def warm(_):
    torch.set_num_threads(1)
    kw=dict(SOLVER_KW,t_end=.001,max_steps=2,max_snaps=8)
    solver.run_halo(.3,2.,**kw)
    return os.getpid()


@contextmanager
def uncontended_cpu(pid):
    paused=False
    if pid and Path(f'/proc/{pid}/cmdline').exists():
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
        if b'forecast_final_validate.py' not in cmd or os.getpgid(pid)!=pid:
            raise RuntimeError('refusing to pause an unrecognized process group')
        if Path(f'/proc/{pid}/stat').read_text().split()[2] not in ('T','t'):
            os.killpg(pid,signal.SIGSTOP);paused=True
            print('Paused owned F8 group for uncontended CPU benchmark',pid,flush=True)
    try:yield
    finally:
        if paused:
            try:os.killpg(pid,signal.SIGCONT)
            except ProcessLookupError:pass
            print('Resumed F8 group',pid,flush=True)


def cpu_batch(params,level,pid):
    with uncontended_cpu(pid):
        with ProcessPoolExecutor(max_workers=24,mp_context=multiprocessing.get_context('spawn')) as pool:
            # Imports and JIT caches warmed before timed solves. Every worker
            # initializer is unnecessary here because each real task also uses
            # the same cached signatures; record the observed warm worker count.
            warmed=list(pool.map(warm,range(48)))
            start=time.perf_counter()
            rows=list(pool.map(refined_row,[(i,p,level) for i,p in enumerate(params)]))
            seconds=time.perf_counter()-start
    return dict(params=params,profiles=np.stack([r[1] for r in rows]),valid=np.stack([r[2] for r in rows]),
                status=np.array([r[3] for r in rows]),t_end=np.array([r[4] for r in rows]),
                errors=np.array([r[-1] or '' for r in rows]),seconds=seconds,warm_workers=len(set(warmed)))


def compare(cpu,gpu):
    mask=cpu['valid']&gpu['valid']
    delta=float(np.max(abs(cpu['profiles'][mask]-gpu['profiles'][mask]))) if mask.any() else 0.
    time_rel=float(np.max(abs(gpu['t_end']/cpu['t_end']-1)))
    status=np.flatnonzero(cpu['status']!=gpu['status']).tolist()
    support=np.argwhere(cpu['valid']!=gpu['valid']).tolist()
    finite=bool(np.isfinite(gpu['profiles'][mask]).all() and np.isfinite(gpu['t_end']).all())
    failed=np.flatnonzero(cpu['errors']!='').tolist()
    return dict(max_profile_dex=delta,max_terminal_time_relative=time_rel,status_disagreements=status,
        support_disagreements=support,reference_errors=failed,finite=finite,
        passed=bool(finite and not failed and not status and not support and delta<=1e-5 and time_rel<=1e-6))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/gpu_solver/pcr_trial');p.add_argument('--pause-pid',type=int,default=1527831);a=p.parse_args()
    out=ROOT/a.out;out.mkdir(parents=True,exist_ok=True)
    names=['scripts/cuda/halo.cu','scripts/gpu_solver.py','scripts/benchmark_gpu_solver.py','src/gravop/solver.py','docs/GPU_SOLVER_PROTOCOL.md']
    config={'sources':{n:digest(ROOT/n) for n in names},'devices':[0,1],'accuracy_profile_dex':1e-5,'terminal_time_relative':1e-6,'minimum_dual_speedup':2.,'torch':torch.__version__}
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=config:raise RuntimeError('trial source changed; preserve it and use a new trial directory')
    save(out/'config.json',config)
    truths=np.load(ROOT/'data/d1/forecast_numerics/pilot_observations.npz')['params']
    corners=np.array([[x,y] for x in [LO[0],HI[0]] for y in [LO[1],HI[1]]])
    validation=np.concatenate([truths,corners,np.random.default_rng(992000000).uniform(LO,HI,(22,2))])
    throughput=np.random.default_rng(992000001).uniform(LO,HI,(128,2))
    np.savez(out/'parameters.npz',validation=validation,throughput=throughput)
    torch.set_num_threads(1);gpus=[GPUSolver(i) for i in [0,1]]
    for gpu in gpus:
        for factor in [1.,1.001,.999]:
            initial=gpu.initial.copy();initial[3]*=factor
            r,m,rho,pressure=initial.copy();err=solver._hydro_step(r,m,rho,pressure)
            state,actual=gpu.probe_hydro(initial)
            np.testing.assert_allclose(state,np.stack([r,rho,pressure]),rtol=1e-10,atol=1e-14)
            np.testing.assert_allclose(actual,err,rtol=1e-10,atol=1e-14)
        gpu.run(truths[:1],max_steps=1)  # kernel/context warmup, excluded
    gates=[]
    for label,params,level in [('correctness',validation,1),('strict',truths,2)]:
        path=out/f'{label}_cpu.npz'
        if path.exists():
            with np.load(path) as z:cpu={k:z[k] for k in z.files}
            np.testing.assert_array_equal(cpu['params'],params)
        else:
            print('CPU',label,len(params),flush=True);cpu=cpu_batch(params,level,a.pause_pid);np.savez(path,**cpu)
        print('GPU',label,'on both devices',flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(lambda gpu:gpu.run(params,level),gpus))
        for gpu,result in zip(gpus,results):
            np.savez(out/f'{label}_gpu{gpu.device}.npz',**result)
            gate=compare(cpu,result);gates.append(dict(stage=label,device=gpu.device,**gate))
            print(label,gpu.device,'seconds',result['seconds'],'gate',gate,flush=True)
        save(out/'accuracy.json',{'gates':gates,'passed':all(g['passed'] for g in gates)})
        if not all(g['passed'] for g in gates):
            save(out/'decision.json',dict(complete=True,adopt=False,reason='CPU/GPU accuracy gate failed',gates=gates));return
    print('CPU throughput 128 trajectories',flush=True)
    cpu=cpu_batch(throughput,1,a.pause_pid);np.savez(out/'throughput_cpu.npz',**cpu)
    print('GPU throughput one device',flush=True)
    single=gpus[0].run(throughput);np.savez(out/'throughput_gpu0.npz',**single)
    print('GPU throughput two devices',flush=True)
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(gpu.run,params) for gpu,params in zip(gpus,np.array_split(throughput,2))]
        parts=[f.result() for f in futures]
    dual_seconds=time.perf_counter()-start
    merged={k:np.concatenate([r[k] for r in parts]) for k in ['params','profiles','valid','status','t_end']}
    np.savez(out/'throughput_dual.npz',**merged,seconds=dual_seconds)
    more=[compare(cpu,single),compare(cpu,merged)]
    speedup=float(cpu['seconds']/dual_seconds)
    decision=dict(complete=True,adopt=all(g['passed'] for g in more) and speedup>=2.,
        cpu24_seconds=float(cpu['seconds']),cpu_warmed_workers=int(cpu['warm_workers']),gpu0_kernel_seconds=single['seconds'],
        dual_end_to_end_seconds=dual_seconds,dual_speedup=speedup,throughput_accuracy=more,
        reason='Requires >=2x throughput and all accuracy gates before posterior integration validation',gates=gates)
    save(out/'decision.json',decision);print(json.dumps(decision,indent=2),flush=True)


if __name__=='__main__':main()
