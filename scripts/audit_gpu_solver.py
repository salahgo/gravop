"""Recompute the GPU prototype decision from saved CPU and GPU arrays."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/gpu_solver'
TRIAL=OUT/'pcr_trial'


def load(name):
    with np.load(TRIAL/name) as d:return {k:d[k] for k in d.files}


def compare(cpu,gpu):
    np.testing.assert_array_equal(cpu['params'],gpu['params'])
    mask=cpu['valid']&gpu['valid']
    difference=float(abs(cpu['profiles'][mask]-gpu['profiles'][mask]).max())
    relative_time=float(abs(gpu['t_end']/cpu['t_end']-1).max())
    status=np.flatnonzero(cpu['status']!=gpu['status']).tolist()
    support=np.argwhere(cpu['valid']!=gpu['valid']).tolist()
    errors=np.flatnonzero(cpu['errors']!='').tolist()
    finite=bool(np.isfinite(gpu['profiles'][mask]).all() and np.isfinite(gpu['t_end']).all())
    return dict(max_profile_dex=difference,max_terminal_time_relative=relative_time,status_disagreements=status,
        support_disagreements=support,reference_errors=errors,finite=finite,
        passed=bool(finite and not errors and not status and not support and difference<=1e-5 and relative_time<=1e-6))


def main():
    config=json.loads((TRIAL/'config.json').read_text())
    for name,h in config['sources'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    params=load('parameters.npz')
    with np.load(ROOT/'data/d1/forecast_numerics/pilot_observations.npz') as d:truths=d['params']
    lo=np.array([-1.5,-1.]);hi=np.array([1.,1.5])
    expected=np.concatenate([truths,np.array([[x,y] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]]]),np.random.default_rng(992000000).uniform(lo,hi,(22,2))])
    np.testing.assert_array_equal(params['validation'],expected)
    np.testing.assert_array_equal(params['throughput'],np.random.default_rng(992000001).uniform(lo,hi,(128,2)))
    decision=json.loads((TRIAL/'decision.json').read_text());accuracy=[]
    for stage,expected in [('correctness',params['validation']),('strict',truths)]:
        cpu=load(f'{stage}_cpu.npz');np.testing.assert_array_equal(cpu['params'],expected)
        for device in (0,1):accuracy.append(dict(stage=stage,device=device,**compare(cpu,load(f'{stage}_gpu{device}.npz'))))
    assert decision['gates']==accuracy
    cpu=load('throughput_cpu.npz');single=load('throughput_gpu0.npz');dual=load('throughput_dual.npz')
    np.testing.assert_array_equal(cpu['params'],params['throughput'])
    checks=[compare(cpu,single),compare(cpu,dual)]
    assert checks==decision['throughput_accuracy']
    assert int(cpu['warm_workers'])==24
    speedup=float(cpu['seconds']/dual['seconds']);assert speedup==decision['dual_speedup']
    adopted=all(x['passed'] for x in accuracy+checks) and speedup>=2
    assert adopted==decision['adopt']
    audit=dict(passed=True,source_identities_verified=True,parameter_selection_verified=True,
        gates_reproduced=True,decision_reproduced=True,adopt=adopted,
        max_profile_dex=max(x['max_profile_dex'] for x in accuracy+checks),
        max_terminal_time_relative=max(x['max_terminal_time_relative'] for x in accuracy+checks),
        cpu24_seconds=float(cpu['seconds']),gpu0_seconds=float(single['seconds']),dual_seconds=float(dual['seconds']),
        dual_speedup=speedup,reference_trajectories=32+6+128,gpu_trajectories=32*2+6*2+128+128,
        note='Benchmark/source audit passes; GPU adoption fails its performance gate. This is not a calibration outcome.')
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.suffix in {'.npz','.cu','.json'} and p.name!='data_manifest.json')
    (OUT/'data_manifest.json').write_text(json.dumps({'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,3.5),layout='constrained')
    values=[audit['cpu24_seconds'],audit['gpu0_seconds'],audit['dual_seconds']]
    bars=ax.bar(['24 CPU workers','One RTX 4090','Two RTX 4090s'],values,color=['#326b9d','#a45b45','#a45b45'])
    ax.bar_label(bars,labels=[f'{x:.1f} s' for x in values],padding=3)
    ax.set(ylabel='Seconds for 128 trajectories (lower is better)',ylim=(0,max(values)*1.2),title='FP64 solver prototype: accuracy passes, speed does not')
    fig.savefig(OUT/'throughput.png',dpi=160);fig.savefig(OUT/'throughput.pdf');plt.close(fig)
    text=f'''# GPU solver evaluation — 11 September 2026

**Retain the CPU solver.** The GPU prototype passed all CPU-agreement checks but
failed the prespecified throughput requirement. This is a negative performance
result, not a calibration failure and not a claim that no other GPU algorithm
could be faster.

| Implementation | 128 trajectories, seconds |
|---|---:|
| 24 warmed CPU workers | {values[0]:.3f} |
| One RTX 4090, kernel interval | {values[1]:.3f} |
| Two RTX 4090s, combined host-to-host interval | {values[2]:.3f} |

The two-GPU speedup was **{speedup:.3f}x**, below the fixed >=2x adoption gate;
it took {1/speedup:.2f} times the CPU wall time. Compilation/worker startup were
excluded. All 24 CPU workers were warmed, and the owned F8 process group was
suspended and resumed around the CPU measurements to avoid contention. GPU
measurements ran alongside the CPU diagnostic. This is one fixed batch and
hardware configuration, not an exhaustive tuning study. A fair deployment
comparison must include transfer costs; the one-GPU number excludes those costs
and is already slower than the CPU baseline.

![Measured throughput](../results/gpu_solver/throughput.png)

Accuracy checks used six F8 truths, four prior corners and 22 independent prior
parameters at level 1; all six truths were repeated at level 2 on both GPUs.
The additional 128-trajectory throughput batch also passed CPU agreement.
Maximum common observation-profile difference was **{audit['max_profile_dex']:.4g}
dex**, versus the 1e-5 dex gate. Maximum terminal-time relative difference was
**{audit['max_terminal_time_relative']:.4g}**, versus 1e-6. Terminal statuses and
survival masks agreed in every comparison. These checks include all three
observation times, snapshot interpolation and early physical termination.

The prototype uses one CUDA block per halo, FP64 without fast math/FMA and
parallel cyclic reduction for the implicit/hydrostatic linear systems. The
physical discretization, step controls and stopping conditions remain unchanged.
An initial hydrostatic transcription sign defect was caught by the one-step
CPU comparison and fixed before the full trial. The serial-Thomas prototype and
failed compilation/accuracy logs are preserved alongside the final passing
accuracy and negative throughput results.

The adapter is experimental and covers only the fixed 300-shell implicit
Yukawa-tu/index-5 inference configuration. It does not implement the complete
CPU solver's energy and snapshot-output API and is not used in calibration or
production. Larger batches or another parallel algorithm could change throughput;
that was not established here. No posterior-level GPU validation is claimed:
the performance prerequisite failed, so that adoption stage was not run.

Frozen protocol: [GPU_SOLVER_PROTOCOL.md](GPU_SOLVER_PROTOCOL.md).
Raw arrays, source identities and independently recomputed gates:
`results/gpu_solver/pcr_trial/`, `results/gpu_solver/audit.json`.
Reproduce the artifact audit with `python scripts/audit_gpu_solver.py`; repeating
the GPU experiment additionally needs CUDA-enabled PyTorch, NVIDIA cuda-bindings
and NVRTC (tested with PyTorch {config['torch']} and CUDA 13.0).

The fresh CPU confirmation excludes every GPU-development parameter at exact
and float32 precision and freezes that exclusion bank. Earlier F8 sources and
results are preserved. The population calibration continues under its original
numerical gates.
'''
    (ROOT/'docs/GPU_SOLVER_RESULTS_2026-09-11.md').write_text(text)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
