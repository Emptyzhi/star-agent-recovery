from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from source.score_protocol import parse_labels, value_prompt
from source.resident_transport import ResidentTransport


class ScoreProtocolTests(unittest.TestCase):
    def test_labels_keep_author_weights_and_protocol_whitespace(self):
        self.assertEqual(parse_labels([' impossible\n', 'likely', 'sure\n']), 21.001)

    def test_invalid_scores_are_rejected_instead_of_assigned_zero(self):
        for output in ['', '\n', 'Likely', 'Label: likely', 'likely because it works', 'likely\nsure']:
            with self.subTest(output=output), self.assertRaises(ValueError):
                parse_labels([output])

    def test_all_datasets_have_balanced_examples_and_a_final_score_slot(self):
        for dataset in ('game24', 'alfworld', 'webshop'):
            prompt = value_prompt(dataset, 'UNIQUE_CURRENT_TRAJECTORY')
            self.assertTrue(prompt.endswith('Evaluation:'))
            self.assertLess(prompt.index('UNIQUE_CURRENT_TRAJECTORY'), prompt.rindex('Evaluation:'))
            for label in ('impossible', 'likely', 'sure'):
                self.assertIn('Evaluation: ' + label, prompt)

    def test_wire_stops_evaluation_on_first_line_without_command_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Mock()
            session.post.side_effect = RuntimeError('wire captured')
            transport = ResidentTransport('http://resident', 'lease', {}, {}, Path(directory), session=session)
            transport.role = 'evaluation'
            with self.assertRaisesRegex(RuntimeError, 'wire captured'):
                transport.generate('Evaluation:', 100)
            payload = session.post.call_args.kwargs['json']
            self.assertEqual(payload['stop_mode'], 'tot_label')
            self.assertNotIn('allowed_commands', payload)


if __name__ == '__main__':
    unittest.main()
