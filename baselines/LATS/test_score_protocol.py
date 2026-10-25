from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from source.score_protocol import parse_score, value_prompt
from source.resident_transport import ResidentTransport
from source.search_adaptations import Game24ActionOnlyTask


class ScoreProtocolTests(unittest.TestCase):
    def test_complete_integer_preserves_ten_and_author_first_sample(self):
        self.assertEqual(parse_score([' 10\n', '1'], 0, 10), 1)
        self.assertEqual(parse_score(['0'], 0, 10), 0)
        self.assertEqual(parse_score(['7'], 0, 10), .7)

    def test_unrelated_digits_explanations_and_out_of_range_are_rejected(self):
        for text in ['', '\n', '$40', 'This product costs 7 dollars', 'Score: 7', '7 because it matches',
                     'Thus the correctness score is 7', '7\n10', '11', '-1', '0.7']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_score([text], 0, 10)
        with self.assertRaises(ValueError):
            parse_score(['0'], 1, 10)

    def test_prompt_keeps_dataset_range_and_places_slot_after_trajectory(self):
        for dataset in ('game24', 'alfworld', 'webshop'):
            prompt = value_prompt(dataset, 'UNIQUE_CURRENT_TRAJECTORY')
            self.assertTrue(prompt.endswith('Score:'))
            self.assertLess(prompt.index('UNIQUE_CURRENT_TRAJECTORY'), prompt.rindex('Score:'))
            self.assertIn('Score: 10', prompt)
            self.assertIn('Score: 1' if dataset == 'webshop' else 'Score: 0', prompt)

    def test_format_failure_keeps_evidence_and_ranks_as_unscored(self):
        task = Game24ActionOnlyTask(SimpleNamespace(examples='', env=SimpleNamespace(puzzle='1 2 3 4')), None)
        self.assertEqual(task.value_outputs_unwrap(['An explanation containing 10']), -1)
        self.assertEqual(task.score_protocol_failures[0]['outputs'], ['An explanation containing 10'])
        self.assertEqual(task.value_outputs_unwrap(['7']), .7)
        with self.assertRaises(AttributeError):
            task.value_outputs_unwrap([None])

    def test_evaluation_wire_uses_first_line_and_no_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Mock()
            session.post.side_effect = RuntimeError('wire captured')
            transport = ResidentTransport('http://resident', 'lease', {}, {}, Path(directory), session=session)
            transport.role = 'evaluation'
            with self.assertRaisesRegex(RuntimeError, 'wire captured'):
                transport.generate('Score:', 100)
            payload = session.post.call_args.kwargs['json']
            self.assertEqual(payload['stop_mode'], 'first_line')
            self.assertNotIn('allowed_commands', payload)


if __name__ == '__main__':
    unittest.main()
