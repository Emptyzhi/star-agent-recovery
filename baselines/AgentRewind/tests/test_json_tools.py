from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from test_sectioned_interface import SectionedInterfaceTests
from test_adaptation import CounterEnvironment, ScriptedModel
from test_notebook_limits import LONG_NOTE
from agentrewind_bench import author
from agentrewind_bench.controller import Controller
from agentrewind_bench.json_tools import decode_tool_reply
from agentrewind_bench.json_tools import json_runtime_instruction
from agentrewind_bench.framing import sectioned_initial_prompt
from agentrewind_bench.models import render_resident_request


def call(name, **arguments):
    return json.dumps(dict(name=name, arguments=arguments))


class JsonToolTests(unittest.TestCase):
    def test_strict_single_call_and_literal_memory(self):
        text='failed_check: "bad"\ncorrected_strategy: use good.\n'
        reply=call('backtrack_commit',record_uid='rec_000001',memory_summary=text)
        self.assertEqual(decode_tool_reply(reply)[1]['memory_summary'],text)
        for invalid in (reply+' trailing',reply+reply,'```json\n'+reply+'\n```',
                '{"name":"execute_action","name":"backtrack_commit","arguments":{"action":"good"}}',
                call('backtrack_commit',record_uid='rec_000001'),call('execute_action',action=1),
                call('execute_action',action='good',extra='hidden')):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):decode_tool_reply(invalid)

    def test_native_action_query_combined_commit_restore_and_original_notebook(self):
        settings=replace(SectionedInterfaceTests().settings(),interface_format='json-tools',
            initial_actions=1,recovery_actions=1)
        model=ScriptedModel([call('execute_action',action='bad'),
            call('backtrack_candidates',reason='Observed failure'),
            call('backtrack_commit',record_uid='rec_000001',memory_summary=LONG_NOTE),
            call('execute_action',action='good')])
        with tempfile.TemporaryDirectory() as directory:
            result=Controller(CounterEnvironment(),model,settings,directory,'json_tools').run()
            events=[json.loads(line) for line in (Path(directory)/'events.jsonl').read_text().splitlines()]
            restore=next(e for e in events if e['event']=='checkpoint_restored')
            memory=author.load_backtrack_memory(Path(directory)/'author_traces',restore['trace'])
            self.assertEqual(memory[-1]['memory_summary'],LONG_NOTE)
            self.assertEqual(restore['notebook_render_config'],dict(items=20,item_char_limit=900,notebook_char_limit=8000))
        self.assertTrue(result['success'])
        self.assertEqual(result['cost']['model_requests'],4)
        self.assertEqual(result['completed_recoveries'],1)
        prepared=[e for e in events if e['event']=='model_input_prepared']
        self.assertEqual([e['output_limit'] for e in prepared],[100,100,1200,100])
        core=author.backtrack_tool_instructions(selection_policy='agent-direct',context_policy='rollback',workspace_policy='rollback')
        for request in model.requests:
            prompt,encoded,policy=render_resident_request(request,'action',None,None,'dialogue','plain','once','json-tools')
            self.assertIsNone(encoded)
            self.assertIn(core,prompt)
            self.assertTrue(prompt.endswith('Tool call JSON:'))
        self.assertEqual(settings.as_dict()['adaptation_protocol'],'native_act_only_v33_game24_native_observation_1200')

    def test_disabled_recovery_guidance_is_phase_scoped_without_losing_notebook_or_history(self):
        messages=[dict(role='system',content=author.instruction('lines',json_tools=True)),
            dict(role='user',content=sectioned_initial_prompt('webshop','ORIGINAL DEMO',dict(observation='CURRENT PAGE')))]
        runtime=json_runtime_instruction(tools=[author.tools()[0]],remaining=5,recoveries=None,
            blocked=False,candidates=False,paused=True,public_receipt=dict(done=False,reward=None),output_limit=1200)
        messages.insert(1,dict(role='system',content=runtime))
        prompt,encoded,policy=render_resident_request(messages,'action',None,None,'dialogue','plain','once','json-tools')
        self.assertNotIn(author.instruction('lines',json_tools=True),prompt)
        self.assertIn('ORIGINAL DEMO',prompt)
        self.assertIn('CURRENT PAGE',prompt)
        self.assertIn(runtime,prompt)
        self.assertEqual(policy,'native_act_only_plain_actor_only')
        self.assertTrue(prompt.endswith('Action:'))
        self.assertEqual(messages[0]['content'],author.instruction('lines',json_tools=True))
        messages.pop(1)
        prompt,_,policy=render_resident_request(messages,'action',None,None,'dialogue','plain','once','json-tools')
        self.assertIn(author.instruction('lines',json_tools=True),prompt)
        self.assertEqual(policy,'native_act_only_plain_json_tool_call')

    def test_author_optional_reason_is_preserved_and_native_actor_phase_is_separate(self):
        arguments=dict(record_uid='rec_000001',memory_summary='Inspect evidence.',reason='Original optional audit reason')
        self.assertEqual(decode_tool_reply(call('backtrack_commit',**arguments)),('backtrack_commit',arguments))
        from agentrewind_bench.json_tools import decode_interface_reply
        from agentrewind_bench.native_protocol import RUNTIME_PREFIX
        messages=[dict(role='system',content=RUNTIME_PREFIX+'Output tool set: execute_action only.')]
        self.assertEqual(decode_interface_reply('click[Buy Now]\n',messages),('execute_action','click[Buy Now]'))
        with self.assertRaises(ValueError):decode_interface_reply('click[Buy Now]\nObservation: invented',messages)

    def test_truncated_json_recovery_attempts_count_without_execution_or_repair(self):
        from agentrewind_bench.json_tools import declared_tool_name
        malformed=' {"name":"backtrack_commit","arguments":{"record_uid":"rec_000001","memory_summary":"unfinished'
        self.assertEqual(declared_tool_name(malformed),'backtrack_commit')
        with self.assertRaises(ValueError):decode_tool_reply(malformed)
        settings=replace(SectionedInterfaceTests().settings(),interface_format='json-tools')
        model=ScriptedModel([call('execute_action',action='bad'),malformed,malformed,'good'])
        with tempfile.TemporaryDirectory() as directory:
            result=Controller(CounterEnvironment(),model,settings,directory,'header_accounting').run()
            events=[json.loads(line) for line in (Path(directory)/'events.jsonl').read_text().splitlines()]
        self.assertTrue(result['success'])
        self.assertEqual(result['completed_recoveries'],0)
        self.assertEqual(result['executed_actions'],2)
        self.assertEqual(result['cost']['model_requests'],4)
        self.assertEqual([e['consecutive'] for e in events if e['event']=='recovery_decision_without_progress'],[1,2])
        self.assertEqual([e['raw_text'] for e in events if e['event']=='model_protocol_rejected'],[malformed,malformed])
        self.assertFalse(any(e['event']=='recovery_memory_generated' for e in events))
        self.assertEqual(sum(e['event']=='reflection_submission_rejected' for e in events),2)
        prepared=[e for e in events if e['event']=='model_input_prepared']
        self.assertTrue(prepared[-1]['recovery_policy_state']['paused'])
        self.assertEqual({t['name'] for t in prepared[-1]['tools']},{'execute_action'})

    def test_header_accounting_does_not_guess_missing_or_nested_tool_names(self):
        from agentrewind_bench.json_tools import declared_tool_name
        for text in ('{"name": "backtrack_comm',
                '{"arguments":{"name":"backtrack_commit","memory_summary":"unfinished',
                '{"arguments": {}, "name": 1}', 'unframed backtrack_commit'):
            with self.subTest(text=text):self.assertIsNone(declared_tool_name(text))
        self.assertEqual(declared_tool_name('{"arguments":{},"name":"backtrack_candidates","extra":'),
            'backtrack_candidates')


if __name__=='__main__':unittest.main()
