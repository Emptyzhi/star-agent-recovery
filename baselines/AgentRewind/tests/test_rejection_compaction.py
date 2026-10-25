"""Projection preserves native evidence, raw rejected records and actual checkpoints."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import test_adaptation as fixtures
from test_adaptation import CounterEnvironment, ScriptedModel
from agentrewind_bench import author
from agentrewind_bench.controller import Controller
from agentrewind_bench.native_protocol import without_runtime_allowance
from agentrewind_bench.rejection_context import LEDGER_PREFIX, RejectedHistory


class RejectionCompactionTests(unittest.TestCase):
    def settings(self):
        return fixtures.AdaptationTests().config(extra=[
            '--history-format','dialogue','--recovery-format','lines',
            '--notebook-item-char-limit','4000','--rejected-history-keep','2'])

    def test_config_is_explicit_and_rejects_incompatible_history(self):
        settings = self.settings()
        self.assertEqual(settings.as_dict()['adaptation_protocol'], 'native_act_only_v20_author_tools_1200')
        self.assertEqual(settings.max_output_tokens, 1200)
        for value in (-1, 1, 3, True, 2.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(settings, rejected_history_keep=value)
        with self.assertRaises(ValueError):
            replace(settings, history_format='flat')

    def test_fifty_rejections_preserve_base_recent_pairs_and_exact_errors(self):
        base = [dict(role='system',content='AUTHOR GUIDANCE'),
                dict(role='user',content='TASK; ORIGINAL OBSERVATION'),
                dict(role='assistant',content='native action'),
                dict(role='user',content='ACTUAL OBSERVATION; COMPLETE CANDIDATE LIST'),
                dict(role='system',content='Backtrack memory notebook ORIGINAL')]
        original = deepcopy(base)
        projection = RejectedHistory(2)
        projection.begin(base,trace='trace_r0',candidate_record_uid='rec_000004')
        messages = deepcopy(base)
        for i in range(50):
            raw = 'backtrack_commit[rec_000001]\nDifferent rejected diagnosis %d: ' % i + ('details ' * 150)
            messages += [dict(role='assistant',content=raw),dict(role='user',content='ORIGINAL ERROR RECEIPT')]
            projection.register(messages,trace='trace_r0',record_uid='rec_%06d' % (i+5),
                raw_reply=raw,tool='backtrack_commit',target='rec_000001',error_type='BacktrackError',error='EXACT DUPLICATE ERROR')
        unmodified = deepcopy(messages)
        result,report = projection.project(messages)
        self.assertEqual(messages,unmodified)
        self.assertEqual(base,original)
        ledger = next(m for m in result if m['content'].startswith(LEDGER_PREFIX))
        body = json.loads(ledger['content'][len(LEDGER_PREFIX):])
        self.assertTrue(body['controller_generated'])
        self.assertEqual(body['omitted_proposals'],48)
        self.assertEqual(body['groups'][0]['error'],'EXACT DUPLICATE ERROR')
        self.assertEqual(body['groups'][0]['count'],48)
        self.assertEqual(body['groups'][0]['first_source'],dict(trace='trace_r0',record_uid='rec_000005'))
        self.assertEqual(body['groups'][0]['last_source'],dict(trace='trace_r0',record_uid='rec_000052'))
        self.assertEqual(result[-4:],messages[-4:])
        self.assertEqual([m for m in result[:-4] if m is not ledger],base)
        self.assertEqual(report['omitted_proposals'],48)
        self.assertLess(sum(len(m['content']) for m in result),sum(len(m['content']) for m in messages)//3)
        turns=[m['role'] for m in result if m['role']!='system']
        self.assertEqual(turns,['user','assistant','user','assistant','user','assistant','user'])

    def test_repeated_error_provenance_does_not_grow_per_attempt_in_model_input(self):
        projector=RejectedHistory(2)
        messages=[dict(role='user',content='TASK AND COMPLETE CURRENT CANDIDATES')]
        projector.begin(messages,trace='trace_r0',candidate_record_uid='rec_000001')
        sizes=[]
        for i in range(2000):
            raw='backtrack_commit[rec_000001]\nRejected note '+str(i)
            messages=messages+[dict(role='assistant',content=raw),dict(role='user',content='ACTUAL ERROR')]
            projector.register(messages,trace='trace_r0',record_uid='rec_%06d' % (i+2),raw_reply=raw,
                tool='backtrack_commit',target='rec_000001',error_type='BacktrackError',error='SAME ORIGINAL ERROR')
            if i in (49,1999):
                rendered,report=projector.project(messages)
                sizes.append(sum(len(m['content']) for m in rendered))
                self.assertEqual(report['ledger']['groups'][0]['count'],i-1)
                self.assertNotIn('sources',report['ledger']['groups'][0])
        self.assertLess(sizes[1]-sizes[0],30)
        self.assertEqual(len(projector.entries),2000)

    def test_phase_boundaries_and_disabled_projection(self):
        base=[dict(role='user',content='TASK')]
        disabled=RejectedHistory(0)
        self.assertIsNone(disabled.begin(base,trace='r0',candidate_record_uid='rec_000001'))
        self.assertEqual(disabled.project(base),(base,None))
        active=RejectedHistory(2)
        active.begin(base,trace='r0',candidate_record_uid='rec_000001')
        with self.assertRaises(ValueError):
            active.begin(base,trace='r0',candidate_record_uid='rec_000002')
        with self.assertRaises(ValueError):
            active.project(base+[dict(role='assistant',content='unregistered action')])
        active.clear()
        restored=[dict(role='user',content='ACTUAL RESTORED PREFIX')]
        self.assertEqual(active.project(restored),(restored,None))
        active.begin(restored,trace='r1',candidate_record_uid='rec_000003')
        self.assertEqual(active.entries,[])

    def test_controller_raw_records_costs_and_actual_restore_remain_consistent(self):
        rejected=['backtrack_commit[rec_000001]\nThe task is not feasible. Attempt '+str(i) for i in range(6)]
        model=ScriptedModel(['bad','backtrack_candidates[Observed incorrect state]',*rejected,
            'backtrack_commit[rec_000001]\nInspect the counter and choose good to reach state two.','good'])
        with tempfile.TemporaryDirectory() as directory:
            result=Controller(CounterEnvironment(),model,self.settings(),directory,'compact_case').run()
            output=Path(directory)
            events=[json.loads(line) for line in (output/'events.jsonl').read_text().splitlines()]
            raw=[json.loads(line) for line in (output/'author_traces/compact_case_r0.jsonl').read_text().splitlines()]
            self.assertEqual([r['output']['content'] for r in raw if r['kind']=='llm'][2:8],rejected)
            self.assertEqual(sum(r['kind']=='tool' and r.get('error') is not None for r in raw),6)
            compact_request=model.requests[-2]
            marker=next(m for m in compact_request if m['content'].startswith(LEDGER_PREFIX))
            ledger=json.loads(marker['content'][len(LEDGER_PREFIX):])
            self.assertEqual(ledger['omitted_proposals'],4)
            self.assertEqual(compact_request[-4]['content'],rejected[-2])
            self.assertEqual(compact_request[-2]['content'],rejected[-1])
            self.assertNotIn(rejected[0],json.dumps(compact_request))
            last=next(r for r in reversed(raw) if r['kind']=='llm')
            self.assertEqual(last['input']['messages'],compact_request)
            restored=next(e for e in events if e['event']=='checkpoint_restored')
            effective=json.loads(Path(restored['effective_override_input_file']).read_text())
            self.assertEqual(without_runtime_allowance(model.requests[-1]),without_runtime_allowance(effective['messages']))
            self.assertFalse(any(m['content'].startswith(LEDGER_PREFIX) for m in model.requests[-1]))
            self.assertEqual(result['cost']['model_requests'],10)
            self.assertEqual((result['executed_actions'],result['completed_recoveries'],result['success']),(2,1,True))
            self.assertEqual(sum(e['event']=='rejected_history_registered' for e in events),6)
            self.assertEqual(sum(e['event']=='rejected_history_phase_started' for e in events),1)

    def test_malformed_pending_commands_are_recorded_without_guessing_target(self):
        bad=['backtrack_commit[rec_000001]' for _ in range(4)]
        model=ScriptedModel(['bad','backtrack_candidates[Observed failure]',*bad,
            'backtrack_commit[rec_000001]\nChoose good after the observed failure.','good'])
        with tempfile.TemporaryDirectory() as directory:
            result=Controller(CounterEnvironment(),model,self.settings(),directory,'protocol_case').run()
            marker=next(m for m in model.requests[-2] if m['content'].startswith(LEDGER_PREFIX))
            ledger=json.loads(marker['content'][len(LEDGER_PREFIX):])
            self.assertIsNone(ledger['groups'][0]['tool'])
            self.assertIsNone(ledger['groups'][0]['target'])
            self.assertEqual(ledger['groups'][0]['count'],2)
            self.assertEqual(len(result['protocol_rejections']),4)
            self.assertEqual(result['cost']['model_requests'],8)
            self.assertTrue(result['success'])


if __name__=='__main__':
    unittest.main()
