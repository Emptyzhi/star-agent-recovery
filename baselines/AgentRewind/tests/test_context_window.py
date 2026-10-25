"""Overflow projection preserves task/state/IDs without model summarization."""
from copy import deepcopy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch

from agentrewind_bench.context_window import fit_context,MARKER,ContextWindowTooSmall
from agentrewind_bench.candidate_context import candidate_receipts


def render(messages):
    return json.dumps(messages,ensure_ascii=False),None,'test'


def messages():
    result=[dict(role='system',content='UNCHANGED AUTHOR INSTRUCTION'),
            dict(role='system',content='UNCHANGED NOTEBOOK '+900*'N'),
            dict(role='user',content='ORIGINAL TASK AND EXAMPLES')]
    for index in range(30):
        result += [dict(role='assistant',content=f'action {index} '+200*'A'),
                   dict(role='user',content=f'actual observation {index} '+200*'O')]
    return result


class ContextWindowTests(unittest.TestCase):
    def test_root_tail_and_systems_survive_and_source_is_not_modified(self):
        source=messages();original=deepcopy(source)
        rendered,report=fit_context(source,render,len,6000,len(render(source)[0]))
        view=json.loads(rendered[0])
        self.assertEqual(source,original)
        self.assertTrue(all(item in view for item in source[:3]+source[-2:]))
        self.assertLessEqual(len(rendered[0]),6000)
        self.assertEqual(report['model_calls'],0)
        self.assertTrue(report['removed_message_indices'])
        self.assertEqual(sum(m['content'].startswith(MARKER) for m in view),1)
        dialogue=[m for m in view if m['role']!='system']
        self.assertEqual([m['role'] for m in dialogue],['user']+['assistant','user']*((len(dialogue)-1)//2))

    def test_latest_candidates_stay_selectable_even_outside_recent_tail(self):
        source=messages()
        entries=[dict(record_uid='rec_000001',decision_hint='literal decision')]
        source[12]['content']=json.dumps(dict(tool='backtrack_candidates',result=entries))
        rendered,_=fit_context(source,render,len,6000,len(render(source)[0]))
        actual=candidate_receipts(json.loads(rendered[0]))[-1]['value']['result']
        self.assertEqual(actual,entries)

    def test_large_preview_uses_literal_head_tail_and_preserves_uid_and_order(self):
        source=messages()[:3]
        entries=[dict(record_uid=f'rec_{i:06d}',decision_hint='START'+10000*'x'+'END',
                      tool_records_preview=[dict(record_uid='tool_uid',output_preview='HEAD'+10000*'y'+'TAIL')])
                 for i in range(1,4)]
        source += [dict(role='assistant',content='backtrack_candidates[actual failure]'),
                   dict(role='user',content=json.dumps(dict(tool='backtrack_candidates',result=entries)))]
        original=deepcopy(source)
        rendered,report=fit_context(source,render,len,5000,len(render(source)[0]))
        actual=candidate_receipts(json.loads(rendered[0]))[-1]['value']['result']
        self.assertEqual([e['record_uid'] for e in actual],[e['record_uid'] for e in entries])
        self.assertTrue(all(e['decision_hint'].startswith('START') and e['decision_hint'].endswith('END') for e in actual))
        self.assertIn('omitted',actual[0]['decision_hint'])
        self.assertIsNotNone(report['candidate_preview_characters'])
        self.assertEqual(source,original)

    def test_irreducible_task_is_explicit_error_instead_of_invented_or_partial_input(self):
        source=messages()[:3]
        with self.assertRaises(ContextWindowTooSmall):
            fit_context(source,render,len,100,len(render(source)[0]))

    def test_bad_dialogue_is_not_repaired(self):
        source=messages();source[4]['role']='assistant'
        with self.assertRaises(AssertionError):
            fit_context(source,render,len,5000,len(render(source)[0]))

    def test_in_budget_history_does_not_enter_overflow_projection(self):
        source=messages()[:3]
        with self.assertRaises(AssertionError):
            fit_context(source,render,len,5000,len(render(source)[0]))

    def model(self,policy='head-tail'):
        from agentrewind_bench.models import ResidentModel
        model=ResidentModel.__new__(ResidentModel)
        model.tokenizer=None;model.model_type='test';model.history_format='dialogue'
        model.prompt_format='plain';model.candidate_history='once';model.interface_format='sectioned'
        model.recovery_prompt_policy='staged-compact';model.text_history_format='native'
        model.text_output_boundary='command';model.context_overflow=policy
        model.prompt_limit=6000;model.context_limit=6200;model.count=0;model.response_grace_seconds=60
        def generate(request,**kwargs):
            return SimpleNamespace(text='look\n',input_tokens=len(request.prompt),output_tokens=2,
                raw_response=dict(queue_time_seconds=0,finish_reason='frame'))
        model.client=SimpleNamespace(generate=Mock(side_effect=generate))
        return model

    def test_resident_submits_one_fitted_request_without_summary_or_source_mutation(self):
        source=messages();original=deepcopy(source);model=self.model();submitted=Mock()
        with patch('agentrewind_bench.models.render_resident_request',side_effect=lambda view,*a:render(view)), \
             patch('model_backend.encode_prompt',side_effect=lambda t,p:dict(input_ids=list(p))):
            reply=model.generate(source,100,600,submitted,purpose='action')
        submitted.assert_called_once();model.client.generate.assert_called_once()
        self.assertLessEqual(reply.input_tokens,6000)
        self.assertIn('client_context_compression',reply.raw)
        self.assertEqual(reply.raw['client_messages'],original)
        self.assertEqual(source,original)

    def test_resident_under_limit_has_identical_prompt_and_no_compression_event(self):
        source=messages()[:3];model=self.model();submitted=Mock()
        with patch('agentrewind_bench.models.render_resident_request',side_effect=lambda view,*a:render(view)), \
             patch('model_backend.encode_prompt',side_effect=lambda t,p:dict(input_ids=list(p))):
            reply=model.generate(source,100,600,submitted,purpose='action')
        self.assertEqual(reply.raw['client_prompt'],render(source)[0])
        self.assertNotIn('client_context_compression',reply.raw)
        submitted.assert_called_once()

    def test_explicit_old_stop_policy_does_not_submit_overflow(self):
        from agentrewind_bench.models import ModelContextLimit
        model=self.model('stop');submitted=Mock()
        with patch('agentrewind_bench.models.render_resident_request',side_effect=lambda view,*a:render(view)), \
             patch('model_backend.encode_prompt',side_effect=lambda t,p:dict(input_ids=list(p))):
            with self.assertRaises(ModelContextLimit):model.generate(messages(),100,600,submitted,purpose='action')
        submitted.assert_not_called();model.client.generate.assert_not_called()


if __name__=='__main__':
    unittest.main()
