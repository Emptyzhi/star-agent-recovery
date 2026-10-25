"""Context exhaustion is a charged terminal resource outcome, never truncation."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from test_alfworld_ablations import execute
from alf_rebased_v64 import RebasedEpisode
from icr_core import BudgetExhausted,ProtocolError
from alf_episode import AlfEpisode
from alf_simple_v64 import SimpleEpisode as LegacyEpisode
from alf_native_v16 import SimpleEpisode as NativeEpisode
from alf_yellow_diagnosis import DiagnosingYellowV64Controller,DiscoveryV64Controller
from scoped_recovery import ScopeKind
from experiment_settings import Experiment


class RebasedResourceTests(unittest.TestCase):
    def test_initial_prompt_exhaustion_is_uncharged_and_terminal(self):
        original=RebasedEpisode.prompt
        def oversized(episode,memory):
            episode.prompt_token_counter=lambda text:40000
            return original(episode,memory)
        with patch.object(RebasedEpisode,'prompt',new=oversized):
            result,requests,events,episode=execute([])
        self.assertEqual(result['status'],'RESOURCE_LIMIT')
        self.assertEqual(requests,[])
        self.assertEqual(result['cost']['model_requests'],0)
        self.assertTrue(result['cost']['token_accounting_complete'])

    def test_complete_scope_guards_use_resource_errors(self):
        for cls in (LegacyEpisode,NativeEpisode):
            episode=object.__new__(cls)
            episode.history=[]
            episode._scope_kind,episode._scope_view='YELLOW','Actual selected evidence.'
            episode.prompt_context_limit,episode.prompt_service_limit=32768,32000
            episode.prompt_token_counter=lambda text:40000
            with patch.object(AlfEpisode,'prompt',return_value='Native history\n>'):
                with self.assertRaisesRegex(BudgetExhausted,'complete scope prompt'):
                    episode.prompt(())

    def test_yellow_diagnosis_handoff_guard_uses_resource_error(self):
        controller=object.__new__(DiagnosingYellowV64Controller)
        controller.episode=SimpleNamespace(prompt_token_counter=lambda text:40000,
            prompt_context_limit=32768,prompt_service_limit=32000)
        controller.yellow_diagnoses={5:{'text':'Actual diagnosis.'}}
        controller._local_feedback=lambda scope:{'actual_feedback':'fixture'}
        controller.settings=Experiment.for_dataset('alfworld')
        scope=SimpleNamespace(kind=ScopeKind.YELLOW,scope_id=5)
        with patch.object(DiscoveryV64Controller,'_bind_scope',return_value='Native history\n>'):
            with self.assertRaisesRegex(BudgetExhausted,'Yellow diagnosis handoff'):
                controller._bind_scope(scope,'fixture')

    def test_runtime_retains_usage_and_stops_before_oversized_recovery_generation(self):
        original = RebasedEpisode.prompt
        def oversized(episode,memory):
            if episode.recovery_prefix_length is not None:
                episode.prompt_token_counter = lambda text: 40000
            return original(episode,memory)
        with patch.object(RebasedEpisode,'prompt',new=oversized):
            result,requests,events,episode = execute(
                ['look','Take the visible apple.','ACTION 0'],initial_actions=1)
        self.assertTrue(result['complete'])
        self.assertEqual(result['status'],'RESOURCE_LIMIT')
        self.assertFalse(result['success'])
        self.assertEqual(len(requests),3)
        self.assertEqual(result['cost']['model_requests'],3)
        self.assertEqual(result['cost']['input_tokens'],30)
        self.assertEqual(result['cost']['output_tokens'],15)
        self.assertTrue(result['cost']['token_accounting_complete'])
        self.assertIn('rebased recovery input exceeds context',result['state']['resource_limit'])

    def test_invalid_recovery_reference_remains_a_protocol_error(self):
        episode = object.__new__(RebasedEpisode)
        episode.recovery_prefix_length,episode.history = -1,[]
        episode.red_repair_handoff = {'reflection':'fixture'}
        with self.assertRaisesRegex(ProtocolError,'actual retained prefix'):
            episode.prompt(())


if __name__=='__main__':
    unittest.main()
