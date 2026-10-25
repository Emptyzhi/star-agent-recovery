"""Repeated malformed proposals must not impersonate action-window exhaustion."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from experiment_settings import Experiment
from icr_core import ProtocolError
from web_v16_run import run_controller
from test_web_v16 import make_controller


BAD=' Proceed to checkout\n\n'


def exhaust_yellow():
    # Two actual Yellow entries return without executing a malformed proposal.
    return [BAD,'Use one native search or click.','invalid Yellow action',
            BAD,'Use one native search or click.','invalid Yellow action',BAD,BAD]


class GreenProtocolContinuationTests(unittest.TestCase):
    def test_first_window_continues_after_identical_protocol_failures(self):
        c,requests,events=make_controller(exhaust_yellow()+[
            'search[cup]','click[A]','click[red]','click[Buy Now]'])
        self.assertEqual(c.run()['status'],'SUCCESS')
        self.assertEqual(c.checkpoint_restore_count,0)
        self.assertEqual(len(c.occurrences),6)
        self.assertEqual(c.attempt_scope_entries['YELLOW'],2)
        submitted=[e for e in events if e['event']=='actor_protocol_failure_submitted']
        self.assertEqual(len(submitted),2)
        self.assertEqual(submitted[-1]['reason'],'YELLOW_ENTRY_LIMIT')
        self.assertIn('Invalid action!', requests[-4].prompt)
        self.assertTrue(any(r.prompt for r in requests))

    def test_recovery_uses_remaining_actions_instead_of_false_exhaustion(self):
        outputs=['search[cup]','click[A]','click[Buy Now]',
                 'Select the requested red color before buying.','ACTION 2']
        c,_,events=make_controller(outputs+exhaust_yellow()+['click[red]','click[Buy Now]'])
        state=c.run()
        self.assertEqual(state['status'],'SUCCESS')
        trial=state['forward_trials'][0]
        self.assertEqual(trial['forward_actions_used'],4)
        self.assertEqual(trial['forward_actions_granted'],15)
        self.assertEqual(trial['status'],'SUCCESS')
        self.assertEqual(c.attempt_scope_entries['YELLOW'],2)
        self.assertEqual(sum(o.action=='proceed to checkout' for o in c.occurrences.values()),2)
        self.assertTrue(any(e['event']=='actor_protocol_failure_submitted' for e in events))

    def test_persistent_bad_output_stops_at_real_model_budget(self):
        cfg=Experiment(model_request_limit=6,yellow_entries=0)
        c,requests,_=make_controller([BAD]*7,cfg)
        state=run_controller(c)
        self.assertEqual(state['status'],'RESOURCE_LIMIT')
        self.assertEqual(state['resource_limit'],'MODEL_RESOURCE_LIMIT')
        self.assertEqual(len(requests),6)
        self.assertEqual(len(c.occurrences),6)
        self.assertEqual(c.checkpoint_restore_count,0)
        self.assertIsNone(c.episode.snapshot()['public_terminal_score'])

    def test_recovery_resource_limit_keeps_unfinished_window_explicit(self):
        cfg=Experiment(model_request_limit=9,yellow_entries=0)
        c,requests,_=make_controller(['search[cup]','click[A]','click[Buy Now]',
            'Select red before buying.','ACTION 2']+[BAD]*5,cfg)
        state=run_controller(c)
        self.assertEqual(state['status'],'RESOURCE_LIMIT')
        self.assertEqual(len(requests),9)
        trial=state['forward_trials'][0]
        self.assertEqual(trial['status'],'RESOURCE_INTERRUPTED')
        self.assertEqual(trial['forward_actions_used'],4)
        self.assertIsNone(trial['score'])
        self.assertIsNone(c.episode.snapshot()['public_terminal_score'])

    def test_unrelated_protocol_errors_still_raise(self):
        c,_,_=make_controller(['search[cup]','click[A]','click[Buy Now]',
                              'Select red.','ACTION 99'])
        with self.assertRaisesRegex(ProtocolError,'unlisted executed action'):
            run_controller(c)


if __name__=='__main__':unittest.main()
