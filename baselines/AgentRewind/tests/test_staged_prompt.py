from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from test_adaptation import CounterEnvironment, ScriptedModel
from test_sectioned_interface import SectionedInterfaceTests
from agentrewind_bench import author
from agentrewind_bench.controller import Controller
from agentrewind_bench.framing import sectioned_initial_prompt
from agentrewind_bench.json_tools import json_runtime_instruction
from agentrewind_bench.models import render_resident_request
from agentrewind_bench.native_protocol import MEMORY_FIELDS
from agentrewind_bench.staged_prompt import ACTION_GUIDANCE, COMPACT_MEMORY_GUIDANCE, MEMORY_ORDER


def render(messages, purpose='action'):
    return render_resident_request(messages,purpose,None,None,'dialogue','plain',
                                   'once','json-tools','staged-compact')[0]


class StagedPromptTests(unittest.TestCase):
    def messages(self,dataset,*,recovery=False,paused=False):
        tools=[author.tools()[2]] if recovery else [author.tools()[0]] if paused else author.tools()[:2]
        allowance=json_runtime_instruction(tools=tools,remaining=3,recoveries=None,
            blocked=False,candidates=recovery,paused=paused,
            public_receipt=dict(done=False,reward=None,observation='PUBLIC FEEDBACK'),output_limit=1200)
        return [dict(role='system',content=author.instruction('lines',json_tools=True)),
            dict(role='user',content=sectioned_initial_prompt(dataset,'NATIVE ACT-ONLY DEMOS',
                dict(observation='CURRENT PUBLIC TASK'))),dict(role='system',content=allowance)]

    def test_action_and_recovery_phase_rendering_in_all_datasets(self):
        original=author.instruction('lines',json_tools=True)
        for dataset in ('webshop','alfworld','game24'):
            with self.subTest(dataset=dataset):
                messages=self.messages(dataset);before=deepcopy(messages)
                prompt=render(messages)
                self.assertIn(ACTION_GUIDANCE,prompt)
                self.assertNotIn(original,prompt)
                self.assertNotIn(COMPACT_MEMORY_GUIDANCE,prompt)
                self.assertIn('NATIVE ACT-ONLY DEMOS',prompt)
                self.assertIn('CURRENT PUBLIC TASK',prompt)
                self.assertIn(messages[-1]['content'],prompt)
                self.assertTrue(prompt.endswith('Tool call JSON:'))
                self.assertEqual(messages,before)
                recovery=self.messages(dataset,recovery=True)
                recovered=render(recovery,'recovery')
                self.assertIn(original,recovered)
                self.assertIn(COMPACT_MEMORY_GUIDANCE,recovered)
                self.assertEqual(set(MEMORY_ORDER),set(MEMORY_FIELDS))
                self.assertEqual(MEMORY_ORDER[:3],('corrected_strategy','do_not_repeat','first_3_tool_calls'))
                self.assertLess(len(prompt),len(recovered))

    def test_paused_actor_and_restored_notebook_remain_visible(self):
        messages=self.messages('webshop',paused=True)
        note=author.render_backtrack_memory_notebook([dict(memory_summary='corrected_strategy: use good')])
        messages.insert(1,dict(role='system',content=note))
        prompt=render(messages)
        self.assertIn(note,prompt)
        self.assertIn(ACTION_GUIDANCE,prompt)
        self.assertNotIn(COMPACT_MEMORY_GUIDANCE,prompt)
        self.assertTrue(prompt.endswith('Action:'))

    def test_cli_settings_preserve_stopping_sampling_and_original_note_boundary(self):
        base=replace(SectionedInterfaceTests().settings(),interface_format='json-tools')
        staged=replace(base,recovery_prompt_policy='staged-compact')
        self.assertEqual(staged.as_dict()['adaptation_protocol'],'native_act_only_v34_staged_compact_memory_1200')
        self.assertEqual(staged.as_dict()['notebook_limits'],base.as_dict()['notebook_limits'])
        for field in ('temperature','top_p','top_k','repetition_penalty','recovery_patience',
                      'initial_actions','recovery_actions','max_recoveries','max_output_tokens'):
            self.assertEqual(getattr(staged,field),getattr(base,field))
        for changes in (dict(interface_format='native'),dict(notebook_item_char_limit=4000),
                        dict(recovery_prompt_policy='guess')):
            with self.assertRaises(ValueError):replace(staged,**changes)

    def test_combined_commit_stays_verbatim_through_author_restore(self):
        settings=replace(SectionedInterfaceTests().settings(),interface_format='json-tools',
            recovery_prompt_policy='staged-compact',initial_actions=1,recovery_actions=1)
        values={name:'check public feedback' for name in MEMORY_ORDER}
        values.update(corrected_strategy='execute_action good',do_not_repeat='execute_action bad',
            first_3_tool_calls='execute_action good; inspect receipt; stop on success',
            evidence_observed='Counter value: 1',selected_candidate_safety_check='before bad action')
        note='; '.join(name+': '+values[name] for name in MEMORY_ORDER)
        self.assertLessEqual(len(note),900)
        def call(name,**arguments):return json.dumps(dict(name=name,arguments=arguments))
        model=ScriptedModel([call('execute_action',action='bad'),
            call('backtrack_candidates',reason='Observed failed branch'),
            call('backtrack_commit',record_uid='rec_000001',memory_summary=note),
            call('execute_action',action='good')])
        with tempfile.TemporaryDirectory() as directory:
            result=Controller(CounterEnvironment(),model,settings,directory,'staged').run()
            events=[json.loads(line) for line in (Path(directory)/'events.jsonl').read_text().splitlines()]
            restoration=next(e for e in events if e['event']=='checkpoint_restored')
            memory=author.load_backtrack_memory(Path(directory)/'author_traces',restoration['trace'])
            self.assertEqual(memory[-1]['memory_summary'],note)
            self.assertEqual(restoration['original_override_input_file'],restoration['effective_override_input_file'])
        self.assertTrue(result['success'])
        self.assertEqual(result['completed_recoveries'],1)
        self.assertEqual(result['cost']['model_requests'],4)
        prompts=[render(request,purpose) for request,purpose in zip(model.requests,model.purposes)]
        self.assertNotIn(COMPACT_MEMORY_GUIDANCE,prompts[0])
        self.assertIn(COMPACT_MEMORY_GUIDANCE,prompts[2])
        self.assertNotIn(COMPACT_MEMORY_GUIDANCE,prompts[3])
        self.assertIn('corrected_strategy: execute_action good',prompts[3])


if __name__=='__main__':unittest.main()
