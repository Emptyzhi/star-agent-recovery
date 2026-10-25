"""Validate and aggregate recorded cumulative recovery-stage rewards and costs."""
import json
import math
from pathlib import Path

FIELDS=('elapsed_seconds','model_queue_seconds','input_tokens','output_tokens','model_requests','environment_calls')


def task_stages(summary, method, maximum):
    result=summary['result'] if method=='lats' else summary['method_result']
    stages=result['recovery_stages']
    assert stages and len(stages)<=maximum+1
    assert [s['recovery_count'] for s in stages]==list(range(len(stages)))
    previous={key:0 for key in FIELDS};prior_rewards=[]
    for stage in stages:
        cost=stage['cost'];rewards=stage['purchase_rewards']
        assert cost['token_accounting_complete'] and cost['timing_accounting_complete']
        assert rewards[:len(prior_rewards)]==prior_rewards
        assert all(type(r) in (int,float) and math.isfinite(r) and 0<=r<=1 for r in rewards)
        assert stage['best_purchase_reward']==max([0,*rewards])
        for key in FIELDS:
            value=cost[key]
            assert type(value) in (int,float) and math.isfinite(value) and value>=previous[key]
        assert cost['model_queue_seconds']<=cost['elapsed_seconds']
        previous=cost;prior_rewards=rewards
    if len(stages)<maximum+1:
        assert stages[-1]['best_purchase_reward']==1 or result['recovery_stop_reason']=='SEARCH_EXHAUSTED'
    full=summary['cost'];last=stages[-1]['cost']
    for key in ('input_tokens','output_tokens','model_requests','model_queue_seconds'):
        assert math.isclose(last[key],full[key],rel_tol=1e-9,abs_tol=1e-7)
    assert last['elapsed_seconds']<=full['elapsed_seconds']
    assert last['environment_calls']==result['environment_calls']
    return [stages[min(depth,len(stages)-1)] for depth in range(maximum+1)]


def aggregate(rows, method, maximum):
    curves=[task_stages(row,method,maximum) for row in rows]
    n=len(rows);assert n>0
    metrics=[]
    for depth in range(maximum+1):
        stages=[curve[depth] for curve in curves]
        successes=sum(s['best_purchase_reward']==1 for s in stages)
        metrics.append(dict(recovery_count=depth,R_percent=100*math.fsum(s['best_purchase_reward'] for s in stages)/n,
            successes=successes,SR_percent=100*successes/n,
            **{'mean_'+key:math.fsum(s['cost'][key] for s in stages)/n for key in FIELDS}))
    return metrics,curves


def publish(folder, ids, method, maximum):
    paths=sorted(Path(folder).glob('group_*/task_*/summary.json'))
    rows=[json.loads(p.read_text()) for p in paths]
    assert sorted(r['identity']['task_id'] for r in rows)==sorted(ids)
    assert all(r['complete'] and r['identity']['run_id']==folder.name for r in rows)
    assert not list(folder.glob('group_*/task_*/fatal.json'))
    metrics,curves=aggregate(rows,method,maximum)
    report=dict(run_id=folder.name,method=method,tasks=len(rows),fatal=0,complete=True,
                source_max_recoveries=maximum,scoring_policy='webshop_best_observed_purchase_by_recovery_count_v1',
                cost_policy='Recorded cumulative task-admission-to-stage-completion costs; carry ended task stages forward',
                token_accounting_complete=True,timing_accounting_complete=True,metrics=metrics,
                task_scores=[dict(task_id=r['identity']['task_id'],
                    reward_by_recovery_count=[s['best_purchase_reward'] for s in c],
                    cost_by_recovery_count=[s['cost'] for s in c]) for r,c in zip(rows,curves)],
                original_execution_records_modified=False)
    audit=dict(run_id=folder.name,complete=True,tasks=len(rows),fatal=0,
               task_ids_verified=True,all_stage_costs_recorded=True,
               final_stage_tokens_requests_queue_match_task_summary=True,
               stage_wall_times_are_recorded_not_reconstructed=True,metrics=metrics)
    for name,data in (('recovery_depth4_scores.json',report),('recovery_depth4_cost_audit.json',audit)):
        with (folder/name).open('x') as stream:json.dump(data,stream,indent=2,allow_nan=False)
    print('RECOVERY_STAGE_COSTS_VERIFIED',method,len(rows),flush=True)
