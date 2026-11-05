"""Adversarial scientific checks and real installed-wheel end-to-end smoke runs."""
import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import generate
import audit
from summarize import METRICS, describe, summarize
spec=importlib.util.spec_from_file_location("noisy_benchmark_runner",HERE / "run.py")
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_world_oracle_integration_positivity_independence_and_balance(tmp_path):
    hashes=[]
    for family in generate.FAMILIES:
        w=generate.parameters(family,1101)
        assert w==generate.parameters(family,1101)
        assert w!=generate.parameters(family,1102)
        for z in (0,1):
            probs=generate.assignment_probabilities(z)
            assert min(probs)>.026
            assert sum(probs)==pytest.approx(1)
        # Independent equation and exact averaging over symmetric latent/noise endpoints.
        for p in generate.EVAL_PRICES:
            for x in (-1,0,1):
                t=(p-40)/20
                expected=w['base']-w['slope']*t+w['context']*x-w['curvature']*t*t+w['interaction']*x*t-w['hinge']*max(t,0)
                outcomes=[expected+w['latent']*(z-.5)+w['noise']*e for z in (0,1) for e in (-1,1)]
                assert min(outcomes)>=10
                assert sum(outcomes)/4==pytest.approx(generate.mean_demand(p,x,w),abs=1e-12)
        for kind in ('observational','randomized','evaluation'):
            path=tmp_path/f'{family}-{kind}.csv'
            generate.generate(path,family,1101,kind)
            generated=audit.rows(path)
            assert not any('seed' in r or 'z' in r or 'family' in r for r in generated)
            if kind=='evaluation':
                hashes.append(path.read_bytes())
                train=audit.rows(tmp_path/f'{family}-randomized.csv')
                assert {r['x'] for r in train}.isdisjoint(r['x'] for r in generated)
                # Shared unit disturbance across all potential prices; no latent-aware oracle.
                group=generated[:9]
                residuals=[float(r['realized_demand'])-float(r['mean_demand']) for r in group]
                assert max(residuals)-min(residuals)<1e-12
            else:
                assert len(generated)==160
                assert set(generated[0])=={'id','x','price','demand'}
        a=audit.rows(tmp_path/f'{family}-observational.csv'); b=audit.rows(tmp_path/f'{family}-randomized.csv')
        assert [r['x'] for r in a]==[r['x'] for r in b]
        assert [r['price'] for r in a]!=[r['price'] for r in b]
        for left,right in zip(a,b):
            residual=lambda r:float(r['demand'])-generate.mean_demand(float(r['price']),float(r['x']),w)
            assert residual(left)==pytest.approx(residual(right),abs=1e-12)
    assert len(set(hashes))==4


def test_metrics_distinguish_forecast_error_from_correct_decision_and_ties():
    truth=[dict(id='a',context_id='x',price='30',mean_demand='60',expected_revenue='1800',realized_demand='80'),
           dict(id='b',context_id='x',price='40',mean_demand='40',expected_revenue='1600',realized_demand='60')]
    pred=[dict(id='a',predicted_mean_demand='64',predicted_revenue='1920'),
          dict(id='b',predicted_mean_demand='44',predicted_revenue='1760')]
    m=audit.independent_metrics(truth,pred,[30])
    assert m['mean_demand_rmse']==4 and m['regret']==0
    assert m['individual_demand_rmse']==16
    # Exact predicted-revenue tie selects the smaller price.
    pred[0].update(predicted_mean_demand='40',predicted_revenue='1200')
    pred[1].update(predicted_mean_demand='30',predicted_revenue='1200')
    assert audit.independent_metrics(truth,pred,[30])['regret']==0
    with pytest.raises(ValueError,match='complete unique'):
        audit.independent_metrics(truth,pred[:1],[30])
    with pytest.raises(ValueError,match='Duplicate truth'):
        audit.independent_metrics(truth+[truth[0]],pred,[30])


def test_campaign_statistics_pairing_and_failure_denominators():
    assert describe([1,3])==dict(n=2,mean=2,sd=math.sqrt(2),mcse=1,p10=1.2,median=2,p90=2.8)
    protocol=runner.allocation('smoke')
    outcomes=[]
    for f in protocol['families']:
        for seed in protocol['replicate_seeds']:
            for i,b in enumerate(protocol['baselines']):
                outcomes.append(dict(family=f,seed=seed,baseline=b['name'],status='completed',
                    metrics={m:100*(f=='threshold')+10*(seed-1101)+i for m in METRICS}))
    s=summarize(protocol,outcomes)
    assert s['planned']==s['completed']==12
    for p in s['paired_comparisons']:
        assert p['metrics']['regret']['mean']==-1
        assert p['metrics']['regret']['mcse']==0
    outcomes[0].update(status='execution_failed',error='test failure')
    outcomes[0].pop('metrics')
    s=summarize(protocol,outcomes)
    assert s['planned']==12 and s['completed']==11
    assert s['groups'][0]['completed']==1 and s['groups'][0]['planned']==2
    assert s['paired_comparisons'][0]['metrics']['regret']['n']==1
    assert s['paired_comparisons'][0]['metrics']['regret']['mcse'] is None


def test_slot_failure_is_retained_without_retry_and_later_slots_execute(tmp_path,monkeypatch):
    outcomes=[dict(slot='a',status='registered'),dict(slot='b',status='registered'),dict(slot='c',status='preparation_failed')]
    calls=[]
    def finish(q,e,slot):
        calls.append(slot['slot'])
        if slot['slot']=='a':
            raise RuntimeError('deliberate failure')
        slot.update(status='completed')
    monkeypatch.setattr(runner,'finish_slot',finish)
    monkeypatch.setattr(runner,'present_slot',lambda *args: (_ for _ in ()).throw(RuntimeError('dashboard failed')))
    runner.execute_slots(tmp_path,outcomes,{'a':(None,None),'b':(None,None)})
    assert calls==['a','b']
    assert outcomes[1]['status']=='completed' and outcomes[1]['reporting_error']['error']=='dashboard failed'
    assert [r['status'] for r in json.loads((tmp_path/'outcomes.json').read_text())]==['execution_failed','completed','preparation_failed']


def test_installed_wheel_reproduces_full_smoke_workflow_twice(tmp_path):
    import sciquest
    assert Path(sciquest.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), 'Use a wheel installation'
    runs=[]
    for i in range(2):
        output=tmp_path/f'run-{i}'
        env=dict(os.environ); env.pop('PYTHONPATH',None)
        p=subprocess.run([sys.executable,str(HERE/'run.py'),'--profile','smoke','--output',str(output)],
                         cwd=tmp_path,env=env,capture_output=True,text=True,timeout=600)
        assert p.returncode==0,p.stdout+'\n'+p.stderr
        checked=audit.audit(output)
        archived=subprocess.run([sys.executable,str(output/'owner_source/audit.py'),str(output)],
                                cwd=tmp_path,env=env,capture_output=True,text=True,timeout=120)
        assert archived.returncode==0,archived.stdout+'\n'+archived.stderr
        assert json.loads(archived.stdout)==checked
        assert not (output/'owner_source/__pycache__').exists()
        assert checked['planned']==checked['completed']==12
        assert checked['external_agent_calls']==0
        result=json.loads((output/'results.json').read_text())
        assert not result['genuine_external_agent_reasoning_demonstrated']
        assert all(not s['scientific_verified'] for s in result['outcomes'])
        assert all(s['supervisor_metric_names']==['individual_demand_rmse','mean_demand_rmse','policy_value','regret'] for s in result['outcomes'])
        for slot in result['outcomes']:
            q=output/'quests'/slot['slot']; exp=q/'experiments/exp_001'
            notebook=json.loads((exp/'executed_notebook.ipynb').read_text())
            assert all(c['execution_count'] is not None for c in notebook['cells'] if c['cell_type']=='code')
            assert 'owner source, labels, campaign history and supervisor excluded' in json.dumps(notebook)
            assert 'Supervisor evaluation recorded' in (q/'journal.md').read_text()
            assert (q/'reports/dashboard/index.html').is_file()
        assert (output/'comparison.png').read_bytes().startswith(b'\x89PNG')
        assert 'dots = campaign draws' in (output/'comparison.svg').read_text()
        runs.append(output)
    assert (runs[0]/'results.json').read_bytes()==(runs[1]/'results.json').read_bytes()
    retry=subprocess.run([sys.executable,str(HERE/'run.py'),'--profile','smoke','--output',str(runs[0])],capture_output=True)
    assert retry.returncode!=0
    # The independent audit rejects changed evidence, not just matching self-reported scores.
    prediction=next((runs[0]/'quests').rglob('worker_artifacts/predictions.csv'))
    prediction.write_text(prediction.read_text()+'\n')
    with pytest.raises(ValueError,match='Manifest hash mismatch'):
        audit.audit(runs[0])
    manifest_path=runs[0]/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    import hashlib
    manifest['sha256'][str(prediction.relative_to(runs[0]))]=hashlib.sha256(prediction.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Authoritative prediction artifact hash mismatch'):
        audit.audit(runs[0])


def test_audit_model_is_bound_to_allocation_and_missing_authority_is_read_only(tmp_path):
    protocol=runner.allocation('smoke')
    baseline=protocol['baselines'][2]
    valid=dict(model='quadratic',ridge_sum_loss=.01,n=160,coefficients=[0.]*6)
    audit.validate_model(valid,baseline,protocol)
    for changed in [dict(model='linear',coefficients=[0.]*3),dict(coefficients=[]),dict(coefficients=[0.]*7),dict(ridge_sum_loss=.02),dict(n=159),dict(coefficients=[float('nan')]*6)]:
        with pytest.raises(ValueError,match='allocated baseline'):
            audit.validate_model(dict(valid,**changed),baseline,protocol)
    quest=tmp_path/'partially-created-quest';quest.mkdir()
    before=list(quest.iterdir())
    assert audit.authority_record(quest,'registration') is None
    assert list(quest.iterdir())==before
    # Existing malformed DB must not be initialized by the authority reader.
    db=quest/'.sciquest/state.sqlite3'; db.parent.mkdir(); db.touch()
    from sciquest.readonly import database_reads_only
    import sqlite3
    with database_reads_only(), pytest.raises(sqlite3.OperationalError):
        audit.authority_record(quest,'registration')
    assert db.read_bytes()==b'' 


def test_exact_context_action_grid_cannot_be_replaced_by_same_row_count():
    protocol=dict(evaluation_contexts=2,evaluation_prices=[20,30])
    truth=[dict(context_id=c,price=str(p),x='0') for c in ('a','b') for p in (20,30)]
    audit.validate_evaluation_grid(truth,protocol)
    wrong=[dict(context_id='a',price=str(p),x='0') for p in (20,25,30,35)]
    assert len(wrong)==len(truth)
    with pytest.raises(ValueError,match='Wrong evaluation'):
        audit.validate_evaluation_grid(wrong,protocol)
