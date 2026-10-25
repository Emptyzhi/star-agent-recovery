import unittest
from dataclasses import replace

from test_game24_migration import controller, GOOD, BAD, SHORT_SETTINGS
from source.experiment_settings import Experiment
from test_game24.model_api import CallableModelClient,Generation
from test_game24.star_controller import Game24STARController


class NativeGame24ProtocolTests(unittest.TestCase):
    def test_actor_history_matches_author_examples(self):
        c, requests = controller(GOOD)
        result = c.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        green = [r['prompt'] for r in requests if r['role']=='green']
        self.assertTrue(green[0].endswith('Act 1>'))
        self.assertIn('Act 1> 1 - 1 = 0\nObs 1> numbers left: 4 6 0', green[1])
        self.assertTrue(green[1].endswith('Act 2>'))
        self.assertNotIn('Current public state:', green[1])

    def test_singleton_uses_model_written_final_action_format(self):
        c, requests = controller(['1 + 1 = 2', '4 * 6 = 24',
                                  '24 * 2 = 48', GOOD[-1]])
        result = c.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        singleton_prompt = [r['prompt'] for r in requests if r['role'] == 'green'
                            and 'FINAL-ACTION STATE' in r['prompt']]
        self.assertTrue(singleton_prompt)
        self.assertIn('Do not return Failure', singleton_prompt[0])

    def test_rejected_last_slot_enters_red_before_another_environment_action(self):
        actions = [
            '1 - 1 = 0', '4 * 6 = 24', '24 + 0 = 24',
            'answer: (1 - 1) + (4 * 6) = 25',
            'answer: (1 - 1) + (4 * 6) = 24',
        ]
        c, requests = controller(actions, settings=SHORT_SETTINGS)
        result = c.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        self.assertFalse(any(e['event'] == 'yellow_entered' for e in result.events))
        self.assertEqual(result.red_attempts, 1)
        self.assertEqual(result.forward_actions, 4)

    def test_unsubmitted_protocol_output_gets_bounded_yellow_feedback(self):
        c, requests = controller(['1 + 1 + 2 = 4', '1 + 1 + 2 = 4', *GOOD])
        result = c.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        self.assertEqual(result.red_attempts, 0)
        self.assertLess(result.model_requests, 64)
        self.assertTrue(any(e['event'] == 'yellow_entered' for e in result.events))
        self.assertTrue(any('Not executed (' in r['prompt'] for r in requests if r['role']=='green'))

    def test_explicit_four_step_window_requires_recovery_to_retry_final_answer(self):
        actions = [
            '1 + 1 = 2', '4 * 6 = 24', '24 * 2 = 48',
            'answer: (1 + 1) * 4 * 6 = 48',
            'answer: (1 - 1) + (4 * 6) = 24',
        ]
        c, requests = controller(actions, settings=SHORT_SETTINGS)
        result = c.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        self.assertEqual(result.forward_actions, 4)
        self.assertEqual(result.recovery_actions, 1)
        self.assertEqual(result.red_attempts, 1)
        self.assertTrue(result.history[-1]['done'])

    def test_default_twelve_step_windows_reach_both_budget_boundaries(self):
        cfg = replace(Experiment.for_dataset('game24'), yellow_entries=0)
        c, requests = controller(['1 + 1 = 99']*24, settings=cfg)
        result = c.run(0, '1 1 4 6')
        self.assertEqual(result.status, 'RECOVERY_WINDOW_EXHAUSTED')
        self.assertEqual((result.forward_actions,result.recovery_actions),(12,12))
        self.assertEqual(result.red_attempts,1)
        self.assertEqual([e['actions'] for e in result.events if e['event']=='window_started'],[12,12])
        self.assertEqual(len(requests),26)

    def test_actual_failed_final_answer_enters_red_with_full_public_evidence(self):
        c, requests = controller(BAD+GOOD)
        result = c.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        self.assertEqual(result.forward_actions,4)
        self.assertFalse(any(e['event']=='arithmetic_dead_end' for e in result.events))
        reflection = next(r for r in requests if r['role']=='red_reflection')
        self.assertEqual(reflection['stop'], ())
        self.assertEqual(reflection['max_new_tokens'],256)
        self.assertIn('pool before=2 24; after=12', reflection['prompt'])
        self.assertIn('action='+BAD[-1], reflection['prompt'])
        self.assertIn('Controller-observed failure: WINDOW_EXHAUSTED', reflection['prompt'])

    def test_initial_green_keeps_final_answer_opportunity_before_red(self):
        c, requests = controller(BAD[:-1]+[GOOD[-1]])
        result = c.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        self.assertEqual(result.forward_actions, 4)
        self.assertEqual(result.red_attempts, 0)
        self.assertEqual([r['role'] for r in requests], ['green']*4)
        self.assertEqual(result.history[-1]['action'], GOOD[-1])
        self.assertEqual(result.history[-1]['before']['numbers'], ['12'])

    def test_yellow_backtrack_text_is_recorded_without_shortening_the_window(self):
        c, requests = controller(['1 + 1 = 99','BACKTRACK',*GOOD])
        result = c.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        self.assertEqual(result.forward_actions,4)
        self.assertEqual(result.yellow_entries,1)
        self.assertEqual(result.red_attempts,1)
        self.assertFalse(any(n['receipt']['action']=='BACKTRACK' for n in result.ett['nodes']))
        action = next(r for r in requests if r['role']=='yellow_action')
        self.assertIn('Input: 1 1 4 6',action['prompt'])
        self.assertIn('Current numbers: 1 1 4 6',action['prompt'])
        self.assertTrue(action['prompt'].endswith('Act 2>'))

    def test_red_advice_persists_once_in_context_without_duplicate_subtree_copy(self):
        c, requests = controller(BAD+GOOD)
        result = c.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        advice='The earlier sum was valid but led to an unsuitable value. Reconsider it.'
        green=[r['prompt'] for r in requests if r['role']=='green']
        recovery_prompts = [r for r in green if 'Red recovery reflection' in r]
        self.assertTrue(recovery_prompts)
        self.assertTrue(all(r.count(advice)==1 for r in recovery_prompts))
        self.assertEqual(sum(e['event']=='event_subtree_delivered' for e in result.events),1)
        self.assertNotIn('Historical receipt node 0:',green[4])
        self.assertNotIn('Historical receipt node 2:',green[4])
        self.assertIn('HISTORICAL_REJECTION',green[4])
        self.assertEqual(c.nodes[3]['receipt']['detail'], 'final expression does not evaluate to 24')

    def test_equal_valued_distinct_operands_remain_valid(self):
        c, _ = controller(['6 + 6 = 12','6 + 6 = 12','12 + 12 = 24','answer: 6 + 6 + 6 + 6 = 24'])
        self.assertTrue(c.run(0,'6 6 6 6').success)

    def test_wrong_single_value_is_not_given_a_computed_replacement_answer(self):
        cfg=replace(SHORT_SETTINGS,ablation='without_red')
        c, requests=controller(BAD[:-1]+[GOOD[-1]],settings=cfg)
        result=c.run(0,'1 1 4 6')
        self.assertEqual(result.status,'SUCCESS')
        self.assertEqual(result.model_requests,4)
        self.assertEqual(result.history[-1]['after']['numbers'],['12'])
        self.assertEqual(result.history[-1]['action'],GOOD[-1])
        self.assertEqual(result.reward,1)
        self.assertTrue(any(e['event']=='non_target_prefix_final_answer_available' for e in result.events))

    def test_backtrack_before_first_environment_action_keeps_real_root(self):
        c, requests=controller(['','BACKTRACK',*GOOD])
        result=c.run(0,'1 1 4 6')
        self.assertTrue(result.success)
        self.assertEqual(result.forward_actions,0)
        self.assertFalse(any(r['role']=='red_checkpoint' for r in requests))
        reflection=next(r for r in requests if r['role']=='red_reflection')
        self.assertEqual(reflection['max_new_tokens'],256)
        self.assertIn('checking its actual value and using each original occurrence once', reflection['prompt'])
        self.assertNotIn('Yellow reported NEEDS_ROLLBACK',reflection['prompt'])

    def test_yellow_cannot_terminate_recovery_when_rollback_is_unavailable(self):
        c, requests=controller(BAD+['1 + 1 = 99','BACKTRACK',*GOOD])
        result=c.run(0,'1 1 4 6')
        self.assertFalse(result.success)
        self.assertEqual(result.red_attempts,1)
        self.assertEqual(result.recovery_actions,4)
        self.assertTrue(any(e['event']=='yellow_returned_green' and e['outcome']=='PREFIX_REVISION_ADVICE_RECORDED'
                            and not e['red_available'] for e in result.events))
        green=[r['prompt'] for r in requests if r['role']=='green']
        self.assertTrue(any('Actions remaining in this window: 3.' in p for p in green))
        self.assertFalse(any('LOCAL_RETURN_WITHOUT_ROLLBACK' in p for p in green))

    def test_checkpoint_review_reaches_green_without_controller_arithmetic(self):
        pending=iter(BAD+GOOD);seen=[]
        review='The first proposal reused an input. Reconsider the sum.\nOnly use the original four input occurrences.'
        def model(prompt,**kw):
            seen.append((kw['role'],prompt,kw))
            if kw['role']=='red_reflection':text='Use an additional 1 in the candidate.'
            elif kw['role']=='red_checkpoint':text='ACTION 0'
            else:text=next(pending)
            return Generation(text,input_tokens=10,output_tokens=2,queue_seconds=0)
        result=Game24STARController(CallableModelClient(model), settings=SHORT_SETTINGS).run(0,'1 1 4 6')
        self.assertTrue(result.success)
        self.assertEqual(result.red_attempts,1)
        restored=next(e for e in result.events if e['event']=='red_checkpoint_restored')
        self.assertEqual(restored['checkpoint_index'],0)
        self.assertNotIn('checkpoint_review',restored)
        self.assertEqual(next(kw['max_new_tokens'] for role,prompt,kw in seen if role=='red_checkpoint'),100)
        after=[prompt for role,prompt,kw in seen if role=='green'][4:]
        self.assertTrue(all('Red recovery reflection' in p for p in after))
        self.assertTrue(all('Use an additional 1 in the candidate.' in p for p in after))
        self.assertTrue(all(review not in p for p in after))
        self.assertTrue(all('Red checkpoint review' not in p for p in after))
        self.assertTrue(any('Use an additional 1 in the candidate.' in e['text']
                            for e in result.events if e['event']=='model_request' and e['role']=='red_reflection'))


if __name__=='__main__':unittest.main()
