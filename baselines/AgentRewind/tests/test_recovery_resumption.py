"""User-authorized bounded recovery, strict full responses and real re-entry."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from test_adaptation import AdaptationTests, CounterEnvironment, ScriptedModel
from agentrewind_bench.controller import Controller
from agentrewind_bench.native_protocol import decode_line

BAD='backtrack_commit[rec_000001]\nThe task is not feasible.'
NOTE='backtrack_commit[rec_000001]\nInspect the counter and choose good after restoring.'
LIST='backtrack_candidates[observed failure]'


class RejectableEnvironment(CounterEnvironment):
    def execute(self,action):
        if action=='invalid':
            self.actions+=1
            return dict(self.observe(),accepted=False,executed_action=action)
        return super().execute(action)


class RecoveryResumptionTests(unittest.TestCase):
    def settings(self,**overrides):
        settings=AdaptationTests().config(extra=['--history-format','dialogue','--recovery-format','lines',
            '--prompt-format','plain','--candidate-history','once','--max-recoveries','unlimited',
            '--recovery-patience','2','--max-terminal-failures','5','--notebook-item-char-limit','4000',
            '--rejected-history-keep','2','--action-budget-policy','trajectory'])
        return replace(settings,**overrides)

    def run_case(self,responses,env=None,settings=None):
        env=env or CounterEnvironment()
        model=ScriptedModel(responses)
        with tempfile.TemporaryDirectory() as directory:
            result=Controller(env,model,settings or self.settings(),directory,'recovery_resume').run()
            events=[json.loads(line) for line in (Path(directory)/'events.jsonl').read_text().splitlines()]
        return model,result,events,env

    def test_full_candidate_reply_rejects_hallucinated_followup_instead_of_slicing(self):
        invalid=[LIST+'\nfailed_check: simulated reflection',LIST+'\n'+NOTE,
            LIST+' '+LIST,LIST+'\nObservation: invented',LIST+'\n[AGENTREWIND_MEMORY_SUMMARY]\n{}']
        for text in invalid:
            with self.subTest(text=text),self.assertRaises(ValueError):decode_line(text,'lines',whole_response=True)
        self.assertEqual(decode_line('backtrack_candidates[click[Buy Now] returned partial credit]','lines'),
            ('backtrack_candidates','click[Buy Now] returned partial credit'))
        self.assertEqual(decode_line('backtrack_candidates["quoted ] reason"]','lines'),
            ('backtrack_candidates','quoted ] reason'))
        with self.assertRaises(ValueError):decode_line('good\nObservation: invented','lines',whole_response=True)
        self.assertEqual(decode_line('good\nObservation: native first-line framing','lines'),('execute_action','good'))

    def test_two_rejections_pause_then_actor_can_succeed_without_restore_or_budget_reset(self):
        model,result,events,env=self.run_case(['bad',LIST,BAD,BAD,'good'])
        self.assertTrue(result['success'])
        self.assertEqual((result['executed_actions'],result['completed_recoveries']),(2,0))
        self.assertEqual(result['cost']['model_requests'],5)
        self.assertEqual(env.windows,[12])
        prepared=[e for e in events if e['event']=='model_input_prepared']
        self.assertEqual({t['name'] for t in prepared[-1]['tools']},{'execute_action'})
        self.assertEqual(prepared[-1]['recovery_policy_state']['no_progress'],2)
        self.assertIn('overrides earlier commit-only',json.dumps(prepared[-1]['messages']))
        self.assertEqual(sum(e['event']=='recovery_initial_listing_exempt' for e in events),1)
        self.assertEqual([e['consecutive'] for e in events if e['event']=='recovery_decision_without_progress'],[1,2])
        self.assertEqual(result['cost']['input_tokens'],5*31)

    def test_recovery_reopens_after_accepted_action_and_can_restore(self):
        _,result,events,_=self.run_case(['bad',LIST,BAD,BAD,'bad',LIST,NOTE,'good'])
        self.assertTrue(result['success'])
        self.assertEqual(result['completed_recoveries'],1)
        self.assertEqual(result['cost']['model_requests'],8)
        prepared=[e for e in events if e['event']=='model_input_prepared']
        self.assertEqual({t['name'] for t in prepared[4]['tools']},{'execute_action'})
        self.assertIn('backtrack_candidates',{t['name'] for t in prepared[5]['tools']})
        self.assertFalse(prepared[5]['recovery_policy_state']['paused'])
        self.assertEqual(prepared[5]['recovery_policy_state']['no_progress'],0)

    def test_rejected_native_action_does_not_unlock_or_refresh_budget(self):
        _,result,events,env=self.run_case(['bad',LIST,BAD,BAD,'invalid','good'],RejectableEnvironment())
        self.assertTrue(result['success'])
        prepared=[e for e in events if e['event']=='model_input_prepared']
        for step in prepared[-2:]:
            self.assertEqual({t['name'] for t in step['tools']},{'execute_action'})
            self.assertTrue(step['recovery_policy_state']['paused'])
        self.assertEqual(result['executed_actions'],3)
        self.assertEqual(env.windows,[12])
        self.assertEqual(result['cost']['model_requests'],6)

    def test_recovery_command_while_paused_is_rejected_without_unlock_or_execution(self):
        _,result,events,_=self.run_case(['bad',LIST,BAD,BAD,NOTE,'good'])
        self.assertTrue(result['success'])
        self.assertEqual(result['completed_recoveries'],0)
        self.assertEqual(len(result['protocol_rejections']),1)
        self.assertIn('Recovery is paused',result['protocol_rejections'][0]['error'])
        self.assertEqual(result['cost']['model_requests'],6)
        self.assertEqual(sum(e['event']=='author_tool_rejected' for e in events),2)

    def test_repeated_candidate_refreshes_count_and_cannot_restart_the_phase(self):
        _,result,events,_=self.run_case(['bad',LIST,LIST,LIST,'good'])
        self.assertTrue(result['success'])
        self.assertEqual(sum(e['event']=='recovery_initial_listing_exempt' for e in events),1)
        self.assertEqual([e['consecutive'] for e in events if e['event']=='recovery_decision_without_progress'],[1,2])
        self.assertEqual(result['cost']['model_requests'],5)

    def test_two_malformed_queries_also_pause_without_fake_candidate_results(self):
        raw=LIST+'\nModel reflection: fabricated'
        _,result,events,_=self.run_case(['bad',raw,raw,'good'])
        self.assertTrue(result['success'])
        self.assertEqual(len(result['protocol_rejections']),2)
        self.assertFalse(any(e['event']=='recovery_initial_listing_exempt' for e in events))
        self.assertEqual(sum(e['event']=='recovery_decision_without_progress' for e in events),2)
        self.assertEqual(result['cost']['model_requests'],4)

    def test_terminal_and_exhausted_attempts_end_without_phantom_action(self):
        class Partial(CounterEnvironment):
            def observe(self):
                return dict(super().observe(),done=self.value!=0,reward=.75 if self.value else None)
        _,result,events,env=self.run_case(['bad',BAD,BAD],Partial())
        self.assertEqual(result['status'],'RECOVERY_STALLED_NO_ACTION')
        self.assertEqual(result['benchmark_reward'],.75)
        self.assertEqual((result['executed_actions'],result['completed_recoveries']),(1,0))
        self.assertEqual(result['cost']['model_requests'],3)
        self.assertEqual(env.windows,[12])
        _,result,_,env=self.run_case(['bad',LIST,BAD,BAD],settings=self.settings(initial_actions=1,recovery_actions=1))
        self.assertEqual(result['status'],'RECOVERY_STALLED_NO_ACTION')
        self.assertEqual(result['cost']['model_requests'],4)
        self.assertEqual(env.windows,[1])

    def test_restore_resets_progress_and_preserves_retained_trajectory_budget(self):
        _,result,events,env=self.run_case(['bad',LIST,BAD,NOTE,'good'])
        self.assertTrue(result['success'])
        self.assertEqual(result['completed_recoveries'],1)
        self.assertEqual(result['cost']['model_requests'],5)
        resets=[e for e in events if e['event']=='recovery_progress_reset' and e['reason']=='checkpoint_restored']
        self.assertEqual(resets[0]['previous_no_progress'],1)
        self.assertEqual(result['consecutive_recovery_no_progress'],0)
        self.assertEqual(env.windows,[12,12])

    def test_cli_policy_is_explicit_and_budgets_are_unchanged(self):
        settings=self.settings()
        self.assertEqual(settings.recovery_patience,2)
        self.assertIsNone(settings.max_reflection_rejections)
        self.assertIsNone(settings.max_recoveries)
        self.assertIsNone(settings.as_dict()['model_request_limit'])
        self.assertEqual(settings.as_dict()['adaptation_protocol'],'native_act_only_v24_bounded_recovery_actor_resume_1200')
        with self.assertRaises(ValueError):replace(settings,max_reflection_rejections=5)


if __name__=='__main__':unittest.main()
