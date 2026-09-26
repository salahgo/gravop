"""Close a completed F8 phase without turning numerical failures into calibration."""
import json
from pathlib import Path
import numpy as np
from forecast_validate import digest, write_json
from forecast_final_case import arrays, BASE_OUT as OUT, BASE_DATA as DATA, ROOT


def main():
    path=OUT/'validation.json'
    if not path.exists(): raise RuntimeError('F8 has not reached its final stopping rule')
    result=json.loads(path.read_text())
    if not result['complete']: raise RuntimeError('F8 is incomplete')
    config=json.loads((OUT/'config.json').read_text())
    for group in ('sources','inputs'):
        for name,h in config[group].items():
            assert digest(ROOT/name)==h,f'changed {group}: {name}'
    observations=arrays(ROOT/'data/d1/forecast_numerics/pilot_observations.npz')
    for i in range(6):
        current=arrays(DATA/f'case{i}/observation.npz')
        for k,v in observations.items(): np.testing.assert_array_equal(current[k],v[i:i+1])
    truth=arrays(ROOT/'data/d1/forecast_numerics/pilot_truth.npz')
    strict=arrays(DATA/'truth_level2.npz')
    common=truth['valid']&strict['valid']
    maximum=float(np.abs(truth['profiles'][common]-strict['profiles'][common]).max())
    temporal=result['diagnostic_truth_temporal']
    assert maximum==temporal['max_profile_dex']
    assert np.argwhere(truth['valid']!=strict['valid']).tolist()==temporal['support_disagreements']
    generation=json.loads((OUT/'truth_generation.json').read_text())
    assert digest(DATA/'truth_level2.npz')==generation['sha256']
    reserved=completed=0
    new_strict=6
    cases=[]
    for r in result['diagnostics']:
        i=r['case']
        if r.get('reused')=='F7':
            cases.append(dict(case=i,time=[100,400,900][int(observations['time_index'][i])],passed=r['passed'],reused='F7'))
            continue
        directory=OUT/f'case{i}'
        v=json.loads((directory/'validation.json').read_text())
        assert v['passed']==r['passed']
        p=v['progress']
        count=sum(int(arrays(f)['new_count']) for f in (DATA/f'case{i}/columns').glob('*/nodes.npz'))
        assert count==v['completed_new_level1_evaluations']
        budget=json.loads((directory/'budget.json').read_text())
        assert p['submissions']==sum(budget['columns'].values())
        for meta in (DATA/f'case{i}/columns').glob('*/stage*.json'):
            assert digest(meta.with_suffix('.npz'))==json.loads(meta.read_text())['sha256']
        reserved+=p['submissions']; completed+=count
        if v['temporal']: new_strict+=v['temporal']['new_evaluations']
        errors=p['history'][-1]['relative_error'] if p['history'] else None
        cases.append(dict(case=i,time=[100,400,900][int(observations['time_index'][i])],
            passed=v['passed'],numerical_passed=p['numerical_passed'],reason=p['reason'],
            reserved=p['submissions'],completed=count,final_stage=p['final_stage'],
            maximum_error=np.max(errors,axis=1).tolist() if errors else None,
            temporal=v['temporal']))
    audit=dict(frozen_sources_inputs_models=True,observation_identity=True,truth_temporal_reproduced=True,
        reserved_level1=reserved,completed_level1=completed,new_level2=new_strict,cases=cases,
        population_status=result['population_status'],population_calibration_passed=result['population_calibration_passed'])
    write_json(OUT/'final_audit.json',audit)
    files=sorted(DATA.rglob('*.npz'))
    write_json(OUT/'data_manifest.json',dict(files={str(p.relative_to(ROOT)):digest(p) for p in files}))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plotted=[c for c in cases if not c.get('reused')]
    if plotted:
        fig,axes=plt.subplots(1,len(plotted),figsize=(5*len(plotted),3.6),squeeze=False,layout='constrained')
        for ax,c in zip(axes[0],plotted):
            history=json.loads((OUT/f"case{c['case']}/progress.json").read_text())['history']
            for k,label in enumerate(('solver','candidate')):
                if history: ax.semilogy([r['submissions'] for r in history],[np.max(r['relative_error'][k]) for r in history],'o-',label=label)
            for threshold in (.02,.005): ax.axhline(threshold,color='gray',ls='--',lw=.8)
            ax.set(title=f"case {c['case']}, time {c['time']}: {'pass' if c['passed'] else 'failed gate'}",xlabel='New level-1 submissions',ylabel='Maximum propagated relative error')
            ax.legend()
        fig.savefig(OUT/'convergence.png',dpi=160)
        fig.savefig(OUT/'convergence.pdf')
        plt.close(fig)
    lines=['# F8 — final numerical validation and research closure','',
        '**Continuous-prior population calibration is not established.**' if not result['population_calibration_passed'] else '**Population screens passed within the recorded scope.**','',
        'Protocol: [FORECAST_FINAL_PROTOCOL.md](FORECAST_FINAL_PROTOCOL.md). The frozen model, prior, likelihood and original numerical gates are retained. Earlier failures and the case-specific F7 success remain unchanged.','',
        f"All six diagnostic generating truths: maximum level-1/level-2 profile difference **{maximum:.8g} dex**, threshold 0.00434; support changes {temporal['support_disagreements']}; passed **{temporal['passed']}**.",'',
        '| Case | Time | Numerical result | Stop | Reserved / completed level-1 |','|---:|---:|---|---|---|']
    for c in cases:
        lines.append(f"| {c['case']} | {c['time']} | {c['passed']} | {c.get('reason','verified F7 reuse')} | {c.get('reserved',0)} / {c.get('completed',0)} |")
    lines+=['','![Diagnostic convergence](../results/forecast_final/convergence.png)','',f"Remaining diagnostic indices: {result['unrun_diagnostics']}. Population stage: **{result['population_status']}**.",'',
        f'Work: {reserved} reserved new level-1 submissions, {completed} recorded completed level-1 evaluations, {new_strict} new level-2 evaluations. Interrupted reservations remain charged.','',
        'The stopping rule was recorded before this experiment: a required diagnostic failure stops the final phase, blocks fresh calibration, and narrows the inference claim. A cap or incomplete rule is a failed numerical gate, not evidence that the candidate is statistically miscalibrated. No difficult cases were discarded to report calibration on an easier subset.','']
    for c in cases:
        if c.get('maximum_error'):
            lines += [f"Case {c['case']} last completed rule maximum propagated errors (solver/candidate): {c['maximum_error']}; final stage {c['final_stage']}. These refer to the last complete rule, not the later partial work.",'']
    interpretation = ('The fresh continuous-prior confirmation passed its prespecified numerical and population screens. '
        'This supports calibration within the tested prior, likelihood, survival conditions and finite test power; '
        'it does not establish universal calibration or automatic production acceptance.'
        if result['population_calibration_passed'] else
        'Continuous-prior population calibration remains unestablished. The final manuscript therefore retains '
        'the forward-model gains, mixed cross-system results and explicit inference limitation.')
    lines+=['## Interpretation','',
        'F7 established numerical convergence for one reused observation. F8 tests portability to other observations and times. The existing finite-grid population screens and improved forward predictions remain valid within their own scopes; neither alone establishes continuous-prior calibration. '+interpretation,'',
        'The experimental ordering follows [Stan SBC](https://mc-stan.org/docs/stan-users-guide/simulation-based-calibration.html) and [Modrak et al.](https://arxiv.org/abs/2211.02383): independent observations and likelihood-based test quantities matter, and a trustworthy numerical reference is a prerequisite.','',
        'The final audit rechecks frozen sources, inherited data, model identities, exact observation selection, stricter truth profiles and durable accounting. Case reports separately reconstruct quadrature and posterior summaries when a complete rule exists. Artifacts: `results/forecast_final/`; caches: `data/d1/forecast_final/`; logs: `logs/f8_*`.','']
    (ROOT/'docs/FORECAST_FINAL_RESULTS_2026-09-10.md').write_text('\n'.join(lines))
    print(json.dumps(audit,indent=2))


if __name__=='__main__': main()
