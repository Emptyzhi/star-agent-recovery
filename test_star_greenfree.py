from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
sys.path.insert(0,str(ROOT/'tests'))
from test_web_v16 import make_controller,make_episode
from icr_core import ActionSupport,SupportKind
from experiment_settings import Experiment
from web_v16_episode import WebV16Episode
from scoped_recovery import ScopeKind


class GreenFreeTests(unittest.TestCase):
    def test_default_episode_uses_unconstrained_first_line_on_finite_pages(self):
        self.assertFalse(WebV16Episode.ordinary_command_constrained)
        self.assertEqual(WebV16Episode.ordinary_stop_mode,'first_line')
        controller,requests,_=make_controller(['search[cup]','click[NOT_VISIBLE]'])
        controller.episode.step('search[cup]')
        controller._ordinary(controller.active_scope,'Continue')
        request=requests[-1]
        self.assertEqual(request.route,'normal')
        self.assertEqual(request.commands,())
        self.assertEqual(request.stop_mode,'first_line')

    def test_green_does_not_mask_unknown_product_choice(self):
        controller,requests,_=make_controller(['click[NOT_VISIBLE]'])
        controller.episode.step('search[cup]')
        parsed,_=controller._ordinary(controller.active_scope,'Continue')
        self.assertEqual(parsed.action,'click[NOT_VISIBLE]')
        self.assertNotIn(parsed.action,controller.episode.support().commands)
        self.assertEqual(requests[-1].route,'normal')

    def test_yellow_retains_finite_choice_after_actual_rejected_green_click(self):
        controller,requests,_=make_controller(['search[cup]','click[Next >]',
            'Inspect a visible item.','click[A]','click[red]','click[Buy Now]'])
        self.assertEqual(controller.run()['status'],'SUCCESS')
        choices=[r for r in requests if r.route=='command_trie']
        self.assertEqual(len(choices),1)
        self.assertIn('click[A]',choices[0].commands)
        self.assertNotIn('click[Next >]',choices[0].commands)
        self.assertTrue(all(r.stop_mode=='first_line' for r in requests if r.route=='normal'))

    def test_red_retains_checkpoint_choice_then_returns_to_free_green(self):
        controller,requests,_=make_controller(['search[cup]','click[A]','click[Buy Now]',
            'The selected product does not match the requested item.','ACTION 1',
            'click[A]','click[red]','click[Buy Now]'])
        self.assertEqual(controller.run()['status'],'SUCCESS')
        choice=next(r for r in requests if r.route=='command_trie')
        self.assertIn('ACTION 1',choice.commands)
        self.assertEqual(requests[-1].route,'normal')
        self.assertEqual(requests[-1].stop_mode,'first_line')

    def test_exhaustion_keeps_native_null_score_and_reports_zero_reward(self):
        from web_v16_run import benchmark_reward
        controller,_,_=make_controller(['search[cup]'],Experiment(ablation='without_red',initial_actions=1))
        self.assertEqual(controller.run()['status'],'INITIAL_WINDOW_EXHAUSTED')
        self.assertIsNone(controller.episode.snapshot()['public_terminal_score'])
        self.assertEqual(benchmark_reward(controller.episode.snapshot()['public_terminal_score']),0)


if __name__=='__main__':unittest.main()
