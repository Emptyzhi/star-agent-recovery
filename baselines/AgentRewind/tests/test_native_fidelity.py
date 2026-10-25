from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import test_adaptation as fixtures
from test_adaptation import CounterEnvironment, NATIVE, ScriptedModel, commit
from agentrewind_bench import author
from agentrewind_bench.controller import Controller
from agentrewind_bench.env_worker import NativeEnvironment
from agentrewind_bench.models import render_resident_request
from agentrewind_bench.native_protocol import (MEMORY_FIELDS,
    initial_actor_prompt, append_actor_step, render_native_messages,
    without_runtime_allowance, RUNTIME_PREFIX, add_control_observation,
    available_command_instruction)


class NativeFidelityTests(unittest.TestCase):
    def test_game24_feedback_matches_native_actor_on_actions_rejections_and_restores(self):
        sys.path.insert(0, str(NATIVE))
        from test_game24.game24_env import Game24Environment
        env=NativeEnvironment.__new__(NativeEnvironment)
        env.dataset='game24';env.env=Game24Environment('1 2 3 4')
        env.current=None;env.actions=0;env.restores=0;env.checkpoints={}
        self.assertEqual(env.reset()['observation'],'Input: 1 2 3 4')
        checkpoint=env.checkpoint()
        first=env.execute('1 + 2 = 3')
        self.assertEqual(first['observation'],'numbers left: 3 4 3')
        self.assertEqual(first['native_receipt']['detail'],'numbers updated')
        rejected=env.execute('1 + 2')
        self.assertEqual(rejected['observation'],
                         'invalid arithmetic action; numbers left: 3 4 3')
        self.assertFalse(rejected['accepted'])
        env.restore(checkpoint)
        self.assertEqual(env.observe()['observation'],'Input: 1 2 3 4')
        self.assertEqual(env.observe()['state']['numbers'],['1','2','3','4'])
        final=env.execute('answer: (1 + 2 + 3) * 4 = 24')
        self.assertTrue(final['done'])
        self.assertEqual(final['reward'],1)
        self.assertEqual(final['observation'],
                         'Success!; numbers left: 1 2 3 4')

    def settings(self, *extra):
        return fixtures.AdaptationTests().config(extra=list(extra))

    def test_native_budget_receives_only_unoccupied_trajectory_positions(self):
        for dataset in ('game24','alfworld'):
            env=NativeEnvironment.__new__(NativeEnvironment)
            env.dataset=dataset
            env.env=SimpleNamespace(action_count=4,history=[None]*4)
            env.set_window(11)
            limit=env.env.max_actions if dataset=='game24' else env.env.action_limit
            self.assertEqual(limit,15)
            with self.assertRaises(ValueError):env.set_window(0)

    def test_webshop_plain_control_is_the_existing_act_only_prompt_function(self):
        sys.path.insert(0, str(NATIVE/'source'))
        from icr_prompts import web_prompt
        base = (NATIVE/'prompts/webshop/webshop_reflexion_actionly_prompt.txt').read_text()
        self.assertNotIn('think[', base)
        current = {'observation': 'Instruction: buy clips\n[Search]'}
        prompt = initial_actor_prompt('webshop', base, current)
        self.assertEqual(prompt, web_prompt(base, current['observation'], [], ()))
        action, observation = 'search[hair clips]', '[B123] Hair clips'
        prompt = append_actor_step('webshop', prompt, action, observation)
        self.assertEqual(prompt, web_prompt(base, current['observation'],
            [{'action': action, 'observation': observation}], ()))
        self.assertEqual(render_native_messages([{'role':'user','content':prompt}]), prompt)

    def test_other_native_transcripts_keep_demonstrations_and_action_boundaries(self):
        for dataset, suffix in [('alfworld', '\n>'), ('game24', '\nAct 1>')]:
            prompt = initial_actor_prompt(dataset, 'Literal Act-only examples', {'observation':'TASK'})
            self.assertEqual(prompt, 'Literal Act-only examples\nHere is the task:\nTASK'+suffix)
            prompt = append_actor_step(dataset, prompt, 'native action', 'native observation')
            expected = ('Literal Act-only examples\nHere is the task:\nTASK'+suffix+
                (' native action\nnative observation\n>' if dataset=='alfworld' else
                 ' native action\nObs 1> native observation\nAct 2>'))
            self.assertEqual(prompt, expected)

    def test_recorded_checkpoint_equals_every_actual_request_including_rejections_and_memory(self):
        model=ScriptedModel(['', 'bad', 'backtrack_candidates[observed failure]',
            commit('rec_000002','Use good instead.'), 'good'])
        with tempfile.TemporaryDirectory() as tmp:
            result=Controller(CounterEnvironment(),model,self.settings(),tmp,'fidelity').run()
            events=[json.loads(s) for s in (Path(tmp)/'events.jsonl').read_text().splitlines()]
            original=[json.loads(s) for s in (Path(tmp)/'author_traces/fidelity_r0.jsonl').read_text().splitlines()]
            fork=[json.loads(s) for s in (Path(tmp)/'author_traces/fidelity_r1.jsonl').read_text().splitlines()]
        records={(trace,r['record_uid']):r for trace,rows in [('fidelity_r0',original),('fidelity_r1',fork)]
                 for r in rows if r['kind']=='llm'}
        checkpointed=[e for e in events if e['event']=='model_request_checkpointed']
        self.assertEqual(len(checkpointed), len(model.requests))
        for event, actual in zip(checkpointed, model.requests):
            record=records[event['trace'],event['record_uid']]
            self.assertEqual(record['input']['messages'], actual)
            self.assertEqual(record['output']['usage'], {'input_tokens':31,'output_tokens':7})
        self.assertEqual(result['cost']['model_requests'],5)
        self.assertEqual(result['completed_recoveries'],1)
        restored=model.requests[-1]
        target=records['fidelity_r0','rec_000002']['input']['messages']
        notebook=[m for m in restored if m['content'].startswith('Backtrack memory notebook')]
        self.assertEqual(len(notebook),1)
        self.assertIn('Use good instead.',notebook[0]['content'])
        self.assertEqual([m for m in without_runtime_allowance(restored) if m not in notebook],
                         without_runtime_allowance(target))
        self.assertNotIn('Observed failure',next(m['content'] for m in restored if m['role']=='user'))

    def test_zero_recovery_control_has_no_added_instructions_or_chat_wrapper(self):
        model=ScriptedModel(['bad','good'])
        env=CounterEnvironment()
        with tempfile.TemporaryDirectory() as tmp:
            result=Controller(env,model,self.settings('--max-recoveries','0'),tmp,'plain').run()
        first=initial_actor_prompt('game24',env.initial_prompt(),{'observation':'Counter value: 0'})
        self.assertEqual(model.requests[0],[{'role':'user','content':first}])
        self.assertEqual(model.requests[1],[{'role':'user','content':append_actor_step(
            'game24',first,'bad','Counter value: 1')}])
        self.assertTrue(result['success'])

    def test_original_author_guidance_is_retained_and_all_operational_fields_survive(self):
        official=author.backtrack_tool_instructions(selection_policy='agent-direct',
            context_policy='rollback',workspace_policy='rollback')
        self.assertTrue(author.instruction().endswith(official))
        self.assertIn('Apply preconditions internally; output remains Act-only',author.instruction())
        for field in MEMORY_FIELDS:
            self.assertIn(field,official)

    def test_recovery_tool_phase_has_its_own_literal_output_boundary(self):
        transcript='Native demonstrations\nTASK\n\nAction:'
        allowance=RUNTIME_PREFIX+'Choose a returned checkpoint.\nOutput phase: recovery-command.'
        messages=[{'role':'system','content':allowance},{'role':'user','content':transcript}]
        rendered=render_native_messages(messages)
        self.assertEqual(rendered,'Native demonstrations\nTASK\n\n'+allowance+'\nRecovery command:')
        self.assertEqual(messages[-1]['content'],transcript)
        self.assertEqual(without_runtime_allowance(messages),[{'role':'user','content':transcript}])

    def test_ordinary_runtime_allowance_preserves_the_native_action_continuation(self):
        allowance=RUNTIME_PREFIX+'Native actions remaining: 3.\nOutput phase: native-action-or-rewind-command.'
        for suffix in ('\n\nAction:','\n>','\nAct 2>'):
            prompt='Unchanged examples and active history'+suffix
            messages=[{'role':'system','content':allowance},{'role':'user','content':prompt}]
            self.assertEqual(render_native_messages(messages),
                'Unchanged examples and active history\n\n'+allowance+suffix)
            self.assertEqual(messages[-1]['content'],prompt)

    def test_ordinary_actor_stays_plain_and_combined_commit_uses_native_chat_for_all_models(self):
        observed=[]
        def template(messages,**kwargs):
            observed.append((deepcopy(messages),kwargs))
            return 'native assistant generation boundary'
        tokenizer=SimpleNamespace(apply_chat_template=template)
        messages=[{'role':'system','content':'author instructions'},
                  {'role':'user','content':'Act-only examples\nTASK\n\nAction:'}]
        for model_type in ('qwen2','mistral','llama'):
            before=len(observed)
            prompt,encoded,policy=render_resident_request(messages,'action',tokenizer,model_type)
            self.assertEqual(len(observed),before)
            self.assertEqual(prompt,render_native_messages(messages))
            self.assertEqual(encoded,messages)
            recovery=[messages[0],dict(role='system',content=RUNTIME_PREFIX+'Output phase: recovery-command.'),messages[1]]
            prompt,encoded,policy=render_resident_request(recovery,'recovery',tokenizer,model_type)
            self.assertEqual(len(observed),before+1)
            self.assertEqual(policy,'native_model_chat_recovery_command')
            self.assertEqual(prompt,'native assistant generation boundary')
            self.assertEqual(observed[-1][1],{'tokenize':False,'add_generation_prompt':True})

    def test_recovery_commands_use_native_chat_with_current_format_after_the_task(self):
        observed=[]
        def template(messages,**kwargs):
            observed.append(deepcopy(messages))
            return 'assistant command boundary'
        tokenizer=SimpleNamespace(apply_chat_template=template)
        allowance=RUNTIME_PREFIX+'Only commit is available.\nOutput phase: recovery-command.'
        messages=[{'role':'system','content':'Author guidance'},
                  {'role':'system','content':allowance},
                  {'role':'user','content':'Native Act-only examples\nCURRENT_TASK\n\nAction:'}]
        saved=deepcopy(messages)
        for model_type in ('qwen2','mistral','llama'):
            prompt,encoded,policy=render_resident_request(messages,'recovery',tokenizer,model_type)
            self.assertEqual(prompt,'assistant command boundary')
            self.assertEqual(policy,'native_model_chat_recovery_command')
            self.assertEqual(encoded,observed[-1])
            self.assertEqual(encoded[0],{'role':'system','content':'Author guidance'})
            self.assertEqual(encoded[-1]['content'],
                'Native Act-only examples\nCURRENT_TASK\n\n'+allowance+'\nRecovery command:')
        self.assertEqual(messages,saved)

    def test_runtime_actor_uses_native_chat_without_changing_act_only_demonstrations(self):
        observed=[]
        tokenizer=SimpleNamespace(apply_chat_template=lambda messages,**kwargs:
            observed.append((deepcopy(messages),kwargs)) or 'native assistant boundary')
        transcript='Literal Act-only demonstrations\nCURRENT TASK\n\nAction:'
        allowance=RUNTIME_PREFIX+'Output phase: native-action-or-rewind-command.'
        messages=[dict(role='system',content='Author guidance'),
                  dict(role='system',content=allowance),dict(role='user',content=transcript)]
        for model_type in ('qwen2','mistral','llama'):
            prompt,encoded,policy=render_resident_request(messages,'action',tokenizer,model_type)
            self.assertEqual(policy,'native_model_chat_action_or_rewind')
            self.assertEqual(encoded[-1]['content'],
                'Literal Act-only demonstrations\nCURRENT TASK\n\n'+allowance+'\n\nAction:')
            self.assertEqual(prompt,'native assistant boundary')
            self.assertEqual(observed[-1][1],dict(tokenize=False,add_generation_prompt=True))
        self.assertEqual(messages[-1]['content'],transcript)

    def test_web_and_alf_actions_use_the_shared_native_parsers_without_extra_repairs(self):
        sys.path.insert(0,str(NATIVE/'source'))
        from icr_webshop import WebShopEpisode
        from alf_episode import parse_action
        for dataset, parser, raw, expected in [
            ('webshop',lambda text:WebShopEpisode.parse(None,text),'Act: Click[amber]','click[amber]'),
            ('webshop',lambda text:WebShopEpisode.parse(None,text),'color[amber]','color[amber]'),
            ('alfworld',parse_action,'> Take mug 1 from table 1','take mug 1 from table 1')]:
            sent=[]
            def step(action):
                sent.append(action)
                return dict(observation='native feedback',after={},done=False,
                    public_terminal_score=None,native_accepted=False)
            env=NativeEnvironment.__new__(NativeEnvironment)
            env.dataset,env.actions=dataset,0
            env.env=SimpleNamespace(parse=parser,step=step)
            receipt=env.execute(raw)
            self.assertEqual(sent,[expected])
            self.assertEqual(receipt['executed_action'],expected)
            self.assertFalse(receipt['accepted'])
            self.assertEqual(env.actions,1)

    def test_tool_receipts_are_chronological_and_do_not_mutate_saved_requests(self):
        for dataset in ('webshop','alfworld','game24'):
            original=[{'role':'system','content':'Author instructions'},
                      {'role':'user','content':initial_actor_prompt(dataset,'Examples',{'observation':'TASK'})}]
            saved=deepcopy(original)
            listed=add_control_observation(original,'CANDIDATES_RESULT')
            retried=add_control_observation(listed,'REPEATED_LIST_ERROR')
            self.assertEqual(original,saved)
            self.assertNotIn('REPEATED_LIST_ERROR',json.dumps(listed))
            user=next(m for m in retried if m['role']=='user')
            user['content']=append_actor_step(dataset,user['content'],'native action','NEXT_OBSERVATION')
            rendered=render_native_messages(retried)
            self.assertLess(rendered.index('TASK'),rendered.index('CANDIDATES_RESULT'))
            self.assertLess(rendered.index('CANDIDATES_RESULT'),rendered.index('REPEATED_LIST_ERROR'))
            self.assertLess(rendered.index('REPEATED_LIST_ERROR'),rendered.index('NEXT_OBSERVATION'))

    def test_candidate_phase_retains_native_and_listing_tools_without_free_requests(self):
        model=ScriptedModel(['bad','backtrack_candidates[failure]',
            'backtrack_candidates[repeat]',commit('invented','Check the observed state.'),
            commit('rec_000001','Choose good.'),'good'])
        with tempfile.TemporaryDirectory() as tmp:
            result=Controller(CounterEnvironment(),model,self.settings(),tmp,'phase').run()
        self.assertTrue(result['success'])
        self.assertEqual(result['cost']['model_requests'],6)
        self.assertEqual(result['completed_recoveries'],1)
        self.assertEqual(result['executed_actions'],2)
        for request in model.requests[2:5]:
            rendered=render_native_messages(request)
            current=rendered.split(RUNTIME_PREFIX)[-1]
            self.assertIn('- backtrack_commit[{"record_uid":',current)
            self.assertIn('- backtrack_candidates[reason]',current)
            self.assertIn('- one native action',current)
            self.assertTrue(rendered.endswith('Recovery command:'))
            self.assertLess(rendered.index('Counter value: 1'),rendered.index('"tool": "backtrack_candidates"'))
        rejected=render_native_messages(model.requests[3])
        self.assertNotIn('Candidate selection is pending',rejected)
        self.assertEqual(len(result['protocol_rejections']), 0)
        self.assertIn('backtrack target not found',render_native_messages(model.requests[4]))

    def test_command_format_instructions_describe_only_currently_available_commands(self):
        tools=author.tools()
        native=available_command_instruction([t for t in tools if t['name']=='execute_action'])
        for unavailable in ('backtrack','record_uid','memory_summary','bracket','For commit'):
            self.assertNotIn(unavailable,native)
        listing=available_command_instruction([t for t in tools if t['name']!='backtrack_commit'])
        self.assertIn('reason is a nonempty string',listing)
        self.assertNotIn('record_uid',listing)
        self.assertNotIn('memory_summary',listing)
        commit=available_command_instruction([t for t in tools if t['name']=='backtrack_commit'])
        self.assertIn('record_uid and memory_summary together',commit)
        self.assertNotIn('candidate-list request',commit)


if __name__=='__main__':unittest.main()
