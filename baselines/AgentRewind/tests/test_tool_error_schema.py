"""Actual author exceptions and returned environment failures remain distinct."""
import json
from pathlib import Path
import tempfile
import unittest

import test_adaptation as fixtures
from test_adaptation import CounterEnvironment,ScriptedModel
from agentrewind_bench import author
from agentrewind_bench.controller import Controller
from replay.backtrack import deterministic_candidates


class RejectedActionEnvironment(CounterEnvironment):
    def execute(self,action):
        result=super().execute(action)
        if action=='bad':
            result['accepted']=False
            result['observation']='Invalid action!'
        return result


class ToolErrorSchemaTests(unittest.TestCase):
    def test_author_exception_uses_native_error_schema_and_preview_status(self):
        settings=fixtures.AdaptationTests().config(extra=[
            '--history-format','dialogue','--recovery-format','lines'])
        rejected='backtrack_commit[rec_000001]\nThe task is not feasible.'
        model=ScriptedModel(['bad','backtrack_candidates[Observed failure]',rejected,
            'backtrack_commit[rec_000001]\nChoose good after checking the failed transition.','good'])
        with tempfile.TemporaryDirectory() as tmp:
            result=Controller(CounterEnvironment(),model,settings,tmp,'schema').run()
            records=[json.loads(line) for line in (Path(tmp)/'author_traces/schema_r0.jsonl').read_text().splitlines()]
        error_record=next(row for row in records if row['kind']=='tool' and row['error'])
        self.assertIsNone(error_record['output'])
        self.assertEqual(set(error_record['error']),{'type','message','repr'})
        self.assertEqual(error_record['error']['type'],'BacktrackError')
        self.assertIn('memory_summary rejected:',error_record['error']['message'])
        previews=[preview for candidate in deterministic_candidates(records)
                  for preview in candidate['tool_records_preview']]
        preview=next(item for item in previews if item['status']=='error')
        self.assertEqual(preview['tool_name'],'backtrack_commit')
        self.assertIn('memory_summary rejected:',preview['error_preview'])
        receipt=json.loads(model.requests[3][-1]['content'].rsplit('\nAct ',1)[0])
        self.assertTrue(receipt['result']['rejected'])
        self.assertEqual(receipt['result']['error'],error_record['error']['message'])
        self.assertEqual(model.requests[3][-2]['content'],rejected)
        self.assertTrue(result['success'])
        self.assertEqual((result['cost']['model_requests'],result['executed_actions'],result['completed_recoveries']),(5,2,1))

    def test_returned_native_rejection_is_not_fabricated_as_tool_exception(self):
        settings=fixtures.AdaptationTests().config(extra=['--max-recoveries','0'])
        with tempfile.TemporaryDirectory() as tmp:
            result=Controller(RejectedActionEnvironment(),ScriptedModel(['bad','good']),settings,tmp,'native').run()
            records=[json.loads(line) for line in (Path(tmp)/'author_traces/native_r0.jsonl').read_text().splitlines()]
        tools=[row for row in records if row['kind']=='tool']
        self.assertTrue(all(row['error'] is None for row in tools))
        self.assertFalse(tools[0]['output']['value']['action_accepted'])
        previews=[preview for candidate in deterministic_candidates(records)
                  for preview in candidate['tool_records_preview']]
        self.assertTrue(all(item['status']=='ok' for item in previews))
        self.assertTrue(result['success'])
        self.assertEqual(result['cost']['model_requests'],2)


if __name__=='__main__':unittest.main()
