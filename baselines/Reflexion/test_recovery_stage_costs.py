import unittest
from recovery_stage_metrics import aggregate


def sample(method,rewards,stop='ROLLBACK_LIMIT_REACHED'):
    stages=[]
    for i,reward in enumerate(rewards):
        cost=dict(elapsed_seconds=10+30*i,model_queue_seconds=3+4*i,
                  input_tokens=100+500*i,output_tokens=10+70*i,
                  model_requests=2+3*i,environment_calls=2+4*i,
                  token_accounting_complete=True,timing_accounting_complete=True)
        stages.append(dict(recovery_count=i,purchase_rewards=rewards[:i+1],
                           best_purchase_reward=max(rewards[:i+1]),cost=cost))
    result=dict(recovery_stages=stages,recovery_stop_reason=stop,environment_calls=stages[-1]['cost']['environment_calls'])
    return {('result' if method=='lats' else 'method_result'):result,
            'cost':dict(stages[-1]['cost'],elapsed_seconds=stages[-1]['cost']['elapsed_seconds']+.01)}


class StageCostTests(unittest.TestCase):
    def test_five_stages_keep_distinct_cumulative_costs(self):
        for method in ('lats','reflexion'):
            metrics,_=aggregate([sample(method,[.2,.4,.3,.8,.6])],method,4)
            self.assertEqual([m['mean_input_tokens'] for m in metrics],[100,600,1100,1600,2100])
            self.assertEqual([m['mean_elapsed_seconds'] for m in metrics],[10,40,70,100,130])
            self.assertEqual([m['R_percent'] for m in metrics],[20,40,40,80,80])
    def test_early_success_carries_its_actual_cost_and_reward(self):
        metrics,_=aggregate([sample('reflexion',[.2,1],'TASK_SUCCESS')],'reflexion',4)
        self.assertEqual([m['mean_model_requests'] for m in metrics],[2,5,5,5,5])
        self.assertEqual([m['SR_percent'] for m in metrics],[0,100,100,100,100])
    def test_missing_stage_cost_is_rejected(self):
        row=sample('lats',[.2,.4,.3,.8,.6]);del row['result']['recovery_stages'][2]['cost']
        with self.assertRaises(KeyError):aggregate([row],'lats',4)
    def test_unexplained_missing_recovery_stages_are_rejected(self):
        with self.assertRaises(AssertionError):aggregate([sample('lats',[.2,.4])],'lats',4)
    def test_full_request_usage_must_match(self):
        row=sample('reflexion',[.2,.4,.3,.8,.6]);row['cost']['input_tokens']+=1
        with self.assertRaises(AssertionError):aggregate([row],'reflexion',4)


if __name__=='__main__':unittest.main()
