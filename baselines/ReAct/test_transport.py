import math
from pathlib import Path
import tempfile
import unittest
from types import ModuleType
from unittest.mock import Mock

from author_binding import bind_module
from resident_transport import GARollbackTransport, PipelineTransport, ResidentTransport


class FakeTransport:
    role = 'actor'
    def __init__(self, text):
        self.text, self.calls = text, []
    def generate(self, prompt, cap):
        self.calls.append((self.role, prompt, cap))
        return dict(raw_output=self.text, selected_probabilities=[.2,.8], output_tokens=2)


class Contracts(unittest.TestCase):
    def test_transport_does_not_add_a_search_horizon(self):
        class ReachedService(Exception):pass
        session=Mock()
        session.post.side_effect=ReachedService
        with tempfile.TemporaryDirectory() as folder:
            transport=ResidentTransport('endpoint','lease',{}, {},Path(folder),session=session)
            transport.started=-1000000
            with self.assertRaises(ReachedService):transport.generate('continue legitimate search',1000)
            self.assertEqual(session.post.call_args.kwargs['timeout'],1200)

    def test_ga_uses_all_generated_probabilities_before_first_line_parse(self):
        transport = FakeTransport('\nAnalysis A\nAnalysis B')
        result = GARollbackTransport(transport)._generate('P', stop='\n', max_new_tokens=500, gen_logits=True)
        self.assertEqual(transport.calls, [('analysis','P',500)])
        self.assertEqual(result['text'], 'Analysis A')
        self.assertEqual(result['average_prob'], .5)
        self.assertAlmostEqual(result['logsumexp_prob'], math.log(math.exp(.2)+math.exp(.8)))

    def test_empty_ga_output_retains_author_protocol_failure(self):
        with self.assertRaises(IndexError):
            GARollbackTransport(FakeTransport('\n\n'))._generate('P', stop='\n')

    def test_pipeline_leaves_multiline_output_for_author_parser(self):
        transport = FakeTransport('Action: first\nsecond')
        pipe = PipelineTransport(transport)
        self.assertEqual(pipe('P',max_new_tokens=100,return_full_text=False), [{'generated_text':'Action: first\nsecond'}])
        transport.role = 'reflection'
        pipe('R',max_new_tokens=100,return_full_text=False)
        self.assertEqual(transport.calls[-1], ('reflection','R',256))

    def test_task_binding_rebinds_nested_helpers_without_global_mutation(self):
        module = ModuleType('author_test')
        exec('value="original"\ndef helper(): return value\ndef run(): return helper()', module.__dict__)
        one = bind_module(module, value='one')
        two = bind_module(module, value='two')
        self.assertEqual((one.run(),two.run(),module.run()), ('one','two','original'))

    def test_queue_exclusion_keeps_inference_time(self):
        transport = ResidentTransport('unused','lease',{}, {},Path('.'),session=object())
        transport.records = [dict(response=dict(input_tokens=10,output_tokens=2,queue_time_seconds=28,model_active_step_seconds=2))]
        self.assertEqual(transport.costs(32)['queue_excluded_elapsed_seconds'],4)

    def test_class_methods_and_global_assignments_stay_task_local(self):
        module=ModuleType('author_class_test')
        exec('value="original"\nclass Task:\n @staticmethod\n def run(): return value\ndef set_value(v):\n global value\n value=v',module.__dict__)
        one=bind_module(module,value='one');two=bind_module(module,value='two')
        one.set_value('changed')
        self.assertEqual(one.Task().run(),'changed')
        self.assertEqual(two.Task.run(),'two')
        self.assertEqual(module.Task.run(),'original')


if __name__ == '__main__':
    unittest.main()
