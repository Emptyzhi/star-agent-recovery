"""Regression contracts for the author's combined commit and bounded notebook."""
import json
from pathlib import Path
import tempfile
import unittest

import test_adaptation as fixtures
from test_adaptation import CounterEnvironment, ScriptedModel, commit
from agentrewind_bench.controller import Controller
from agentrewind_bench.native_protocol import decode_line
from replay.backtrack import render_backtrack_memory_notebook


class CombinedCommitTests(unittest.TestCase):
    def run_case(self,responses,extra=()):
        model=ScriptedModel(responses)
        with tempfile.TemporaryDirectory() as tmp:
            result=Controller(CounterEnvironment(),model,fixtures.AdaptationTests().config(extra=extra),tmp,'combined').run()
            events=[json.loads(line) for line in (Path(tmp)/'events.jsonl').read_text().splitlines()]
        return model,result,events

    def test_one_commit_request_contains_target_and_memory_with_256_token_cap(self):
        model,result,events=self.run_case(['bad','backtrack_candidates[failure]',
            commit('rec_000001','Use good after checking the observed failure.'),'good'],
            extra=['--max-output-tokens','256'])
        self.assertEqual(result['cost']['model_requests'],4)
        prepared=[e for e in events if e['event']=='model_input_prepared']
        self.assertEqual([(e['purpose'],e['output_limit']) for e in prepared],
            [('action',100),('action',100),('recovery',256),('action',100)])
        self.assertNotIn('memory',model.purposes)
        memory=next(e for e in events if e['event']=='recovery_memory_generated')
        self.assertTrue(memory['combined_commit'])

    def test_multiline_json_commit_is_decoded_without_first_line_truncation(self):
        raw='backtrack_commit[\n{\n"record_uid": "rec_000001",\n"memory_summary": "Use good.\\nCheck feedback.",\n"reason": "Observed failure"\n}\n]'
        name,args=decode_line(raw)
        self.assertEqual(name,'backtrack_commit')
        self.assertEqual(args,dict(record_uid='rec_000001',memory_summary='Use good.\nCheck feedback.',reason='Observed failure'))

    def test_incomplete_or_extra_commit_fields_are_not_repaired(self):
        for raw in ['backtrack_commit[rec_000001]','backtrack_commit[{"record_uid":"rec_000001"}]',
                    'backtrack_commit[{"record_uid":"r","memory_summary":"m","extra":1}]',
                    'backtrack_commit[{"record_uid":1,"memory_summary":"m"}]',
                    'backtrack_commit[{"record_uid":"r","memory_summary":"m"}] trailing']:
            with self.assertRaises(ValueError):decode_line(raw)

    def test_tool_uid_uses_authors_mapping_to_its_owning_llm_checkpoint(self):
        _,result,events=self.run_case(['bad','backtrack_candidates[failure]',
            commit('rec_000002','Check the original decision and choose good.'),'good'])
        self.assertTrue(result['success'])
        restored=next(e for e in events if e['event']=='checkpoint_restored')
        self.assertEqual(restored['target'],'rec_000001')
        self.assertEqual(restored['trajectory_actions'],0)

    def test_direct_author_commit_does_not_impose_newest_candidate_membership(self):
        _,result,events=self.run_case(['bad','backtrack_candidates[failure]',
            commit('rec_000001','Choose good after the initial checkpoint.'),'good'],
            extra=['--candidate-limit','1','--recovery-actions','1'])
        self.assertTrue(result['success'])
        self.assertEqual(next(e for e in events if e['event']=='checkpoint_restored')['target'],'rec_000001')

    def test_unknown_target_is_rejected_without_restore_and_next_commit_is_charged(self):
        _,result,events=self.run_case(['bad','backtrack_candidates[failure]',
            commit('invented','Inspect the failure.'),commit('rec_000001','Choose good.'),'good'])
        rejected=next(e for e in events if e['event']=='author_tool_rejected')
        self.assertIn('backtrack target not found',rejected['error'])
        self.assertEqual(result['cost']['model_requests'],5)
        self.assertEqual(sum(e['event']=='checkpoint_restored' for e in events),1)
        self.assertEqual(len(result['recovery_stages']),2)

    def test_author_notebook_defaults_limit_items_and_item_characters(self):
        items=[dict(memory_summary=f'NOTE_{i:02d} '+('x'*1200)) for i in range(25)]
        rendered=render_backtrack_memory_notebook(items)
        self.assertNotIn('NOTE_00',rendered)
        self.assertIn('NOTE_05',rendered)
        self.assertNotIn('x'*901,rendered)
        self.assertTrue(rendered.endswith('...[backtrack memory truncated]'))
        self.assertLessEqual(len(rendered),8000+len('\n...[backtrack memory truncated]'))

    def test_repeated_first_checkpoint_matches_author_materialized_fork_numbering(self):
        model,result,events=self.run_case(['bad','backtrack_candidates[failure]',
            commit('rec_000001','Inspect the incorrect state before choosing an alternative.'),
            'bad','backtrack_candidates[repeat failure]',
            commit('rec_000001','Recheck the observed command outcome against the goal.'),'good'],
            extra=['--max-recoveries','unlimited'])
        self.assertTrue(result['success'])
        self.assertEqual(result['completed_recoveries'],2)
        self.assertFalse(any(e['event']=='author_tool_rejected' for e in events))
        first=[e for e in events if e['event']=='model_request_checkpointed' and e['record_uid']=='rec_000001']
        self.assertEqual([e['trace'] for e in first],['combined_r0','combined_r1','combined_r2'])
        self.assertEqual(result['cost']['model_requests'],7)

    def test_budget_instruction_uses_the_resolved_cli_limit(self):
        _,result,events=self.run_case(['bad','backtrack_candidates[failure]',
            commit('rec_000001','Choose good.'),'good'],extra=['--max-output-tokens','123'])
        prepared=next(e for e in events if e['event']=='model_input_prepared' and e['purpose']=='recovery')
        self.assertEqual(prepared['output_limit'],123)
        self.assertIn('123 tokens',json.dumps(prepared['messages']))

    def test_webshop_reports_best_purchase_across_multiple_restores_in_code(self):
        class Purchases(CounterEnvironment):
            def observe(self):
                score={0:None,1:.75,2:.5,3:1}[self.value]
                return dict(observation='Purchase score: '+str(score),accepted=True,
                    done=self.value!=0,reward=score,state={'value':self.value})
            def execute(self,action):
                self.actions+=1
                self.value={'first':1,'second':2,'full':3}[action]
                result=self.observe()
                result['executed_action']=action
                return result
        model=ScriptedModel(['first',commit('rec_000001','Inspect the missing product attribute.'),
            'second',commit('rec_000001','Recheck the original goal and choose the full match.'),'full'])
        with tempfile.TemporaryDirectory() as tmp:
            result=Controller(Purchases(),model,fixtures.AdaptationTests().config('webshop',
                extra=['--max-recoveries','unlimited']),tmp,'purchases').run()
        self.assertEqual([s['best_purchase_reward'] for s in result['recovery_stages']],[.75,.75,1])
        self.assertEqual(result['benchmark_reward'],1)
        self.assertTrue(result['success'])

    def test_webshop_last_lower_purchase_cannot_replace_an_earlier_higher_reward(self):
        class Purchases(CounterEnvironment):
            def observe(self):
                return dict(observation='Native purchase feedback',accepted=True,done=self.value!=0,
                    reward={0:None,1:.75,2:.5}[self.value],state={'value':self.value})
        model=ScriptedModel(['bad',commit('rec_000001','Inspect the unmet attribute before selecting again.'),'good'])
        with tempfile.TemporaryDirectory() as tmp:
            result=Controller(Purchases(),model,fixtures.AdaptationTests().config('webshop'),tmp,'purchases').run()
        self.assertEqual(result['final_public_reward'],.5)
        self.assertEqual(result['benchmark_reward'],.75)
        self.assertFalse(result['success'])


if __name__=='__main__':unittest.main()
