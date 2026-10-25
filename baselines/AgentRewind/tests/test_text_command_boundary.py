from dataclasses import replace
from types import ModuleType
import sys
import unittest
from unittest.mock import patch

from agentrewind_bench.framing import COMMIT_END,decode_sectioned_reply
from agentrewind_bench.resident_text_frame import command_complete,install,STOP_MODE
from test_sectioned_interface import SectionedInterfaceTests


class TextCommandBoundaryTests(unittest.TestCase):
    def first_complete(self,text):
        for end in range(1,len(text)+1):
            if command_complete(text[:end]):return text[:end]
        return None

    def test_stop_at_literal_combined_commit_before_imagined_next_turn(self):
        call='backtrack_commit[rec_000001]\ncorrected_strategy: inspect\n'+COMMIT_END
        stream=call+' [CURRENT_TASK_BEGIN] imagined future input'
        stopped=self.first_complete(stream)
        self.assertEqual(stopped,call)
        self.assertEqual(decode_sectioned_reply(stopped,whole_response=True),
            ('backtrack_commit',dict(record_uid='rec_000001',memory_summary='corrected_strategy: inspect')))

    def test_query_boundary_respects_nested_action_brackets(self):
        call='backtrack_candidates[click[Buy Now] returned partial reward]'
        self.assertEqual(self.first_complete(call+' explanation'),call)
        self.assertIsNone(self.first_complete('backtrack_commit[rec_000001]\nmissing delimiter'))
        self.assertIsNone(self.first_complete('backtrack_candidates[unfinished'))
        self.assertFalse(command_complete('backtrack_commit[rec_000001] note '+COMMIT_END))

    def test_native_commands_still_require_sampled_line_boundary(self):
        for action in ('click[Buy Now]','1 + 1 = 2','go to kitchen'):
            self.assertEqual(self.first_complete(action+'\nObservation: invented'),action+'\n')
        self.assertFalse(command_complete('click[Buy Now]'))

    def test_no_posthoc_repair_of_token_containing_extra_content(self):
        complete='backtrack_commit[rec_000001]\ncorrection\n'+COMMIT_END
        self.assertFalse(command_complete(complete+' [CURRENT_TASK_BEGIN]'))
        with self.assertRaises(ValueError):decode_sectioned_reply(complete+' explanation',whole_response=True)

    def test_opt_in_installer_preserves_every_previous_mode(self):
        module=ModuleType('icr_output_framing')
        calls=[]
        def old(text,mode):calls.append((text,mode));return mode=='first_line' and '\n' in text
        module.STOP_MODES=('first_line','json_object');module.frame_complete=old
        with patch.dict(sys.modules,icr_output_framing=module):
            install()
            self.assertEqual(module.STOP_MODES,('first_line','json_object',STOP_MODE))
            self.assertTrue(module.frame_complete('x\n','first_line'))
            self.assertEqual(calls,[('x\n','first_line')])
            self.assertTrue(module.frame_complete('backtrack_candidates[failed]',STOP_MODE))
            with self.assertRaises(RuntimeError):install()

    def test_explicit_configuration_has_new_protocol_and_rejects_unsupported_backend(self):
        settings=replace(SectionedInterfaceTests().settings(),backend='resident',
            recovery_prompt_policy='staged-compact',text_output_boundary='command')
        self.assertEqual(settings.as_dict()['adaptation_protocol'],'native_act_only_v36_text_command_boundary_1200')
        for changes in (dict(backend='openai'),dict(interface_format='json-tools'),dict(text_output_boundary='guess')):
            with self.assertRaises(ValueError):replace(settings,**changes)


if __name__=='__main__':unittest.main()
