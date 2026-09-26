"""Release gates must fail on missing inputs, modified content and failed commands."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import reproduce
import sc_provenance


@pytest.mark.verification
def test_strict_provenance_rejects_missing_dependencies(tmp_path):
    (tmp_path/'contract.json').write_text(json.dumps({'_provenance':{'generated':'today','inputs':{}}}))
    assert sc_provenance.check(tmp_path)[0]
    ok, problems = sc_provenance.check(tmp_path, strict=True)
    assert not ok and any('required dependency direct.json is missing' in p for p in problems)
    script = Path(sc_provenance.__file__)
    assert subprocess.run([sys.executable,str(script),'--results',str(tmp_path)],capture_output=True).returncode != 0


@pytest.mark.verification
def test_release_hashes_reject_tampering_absence_and_escape(tmp_path):
    import hashlib
    p=tmp_path/'artifact'
    p.write_bytes(b'validated')
    m={'files':{'artifact':hashlib.sha256(b'validated').hexdigest()}}
    assert reproduce.verify_hashes(tmp_path,m)==1
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed artifact'): reproduce.verify_hashes(tmp_path,m)
    p.unlink()
    with pytest.raises(ValueError,match='missing artifact'): reproduce.verify_hashes(tmp_path,m)
    with pytest.raises(ValueError,match='empty'): reproduce.verify_hashes(tmp_path,{'files':{}})
    with pytest.raises(ValueError,match='outside'): reproduce.verify_hashes(tmp_path,{'files':{'../escape':'x'}})


@pytest.mark.verification
def test_runner_preflights_all_steps_and_propagates_failure(tmp_path):
    p=tmp_path/'reproduction.json'
    step={'command':['{python}','-c','raise SystemExit(7)'],'requires':['missing']}
    p.write_text(json.dumps({'targets':{'t':{'steps':[step]}}}))
    with pytest.raises(ValueError,match='missing prerequisites'): reproduce.run_target('t',tmp_path,execute=True)
    (tmp_path/'missing').touch()
    with pytest.raises(subprocess.CalledProcessError) as exc: reproduce.run_target('t',tmp_path,execute=True)
    assert exc.value.returncode==7


@pytest.mark.verification
def test_clean_smoke_accumulates_unexpected_step_failures(tmp_path):
    source=(Path(__file__).resolve().parents[1]/'scripts/clean_env_smoke.sh').read_text()
    function=source[source.index('FAILURES=0'):source.index('echo "clean-environment smoke test"')]
    harness='set -uo pipefail\nWORK="$1"\nREPORT="$WORK/report.txt"\n'+function+'\nstep succeeds true\nstep fails false\ntest "$FAILURES" -eq 1\n'
    result=subprocess.run(['bash','-c',harness,'smoke-test',str(tmp_path)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.verification
def test_clock_sweep_saves_the_prespecified_final_budget(tmp_path,monkeypatch):
    import numpy as np
    import sc_timehead as clock
    class StopAfterClockSweep(Exception): pass
    monkeypatch.setattr(clock,'OUT',str(tmp_path))
    monkeypatch.setattr(clock,'CKPT',str(tmp_path/'ckpt'))
    data={'u':np.zeros((3,2,1,8)),'par':np.zeros((3,1)),'log_tsing':np.zeros(3)}
    monkeypatch.setattr(clock.scd,'load',lambda path:data)
    monkeypatch.setattr(clock.scd,'split',lambda n:([0],[1],[2]))
    monkeypatch.setattr(clock,'Norm',lambda d,tr:object())
    monkeypatch.setattr(clock,'tensors',lambda *a:(data['u'],None,None,None,None,None))
    trained=[];saved=[]
    def train(*args,**kwargs):
        trained.append(args[4])
        return object()
    monkeypatch.setattr(clock,'train_head',train)
    monkeypatch.setattr(clock,'head_error',lambda *a:(1.,np.ones(1)))
    monkeypatch.setattr(clock,'save_checkpoint',lambda *a:saved.append(a[6]))
    def stop(*a): raise StopAfterClockSweep()
    monkeypatch.setattr(clock,'load_checkpoint',stop)
    monkeypatch.setattr(sys,'argv',['sc_timehead.py','--system','rd','--seeds','1','--device','cpu','--budgets','200','600','2000','6000','20000','--save-heads'])
    with pytest.raises(StopAfterClockSweep): clock.main()
    assert trained==[200,600,2000,6000,20000]
    assert saved==[20000]
