import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
from web_v16_run import purchase_score_report
from icr_core import ProtocolError


class BestPurchaseTests(unittest.TestCase):
    def test_earlier_better_purchase_is_default_reported_reward(self):
        row = purchase_score_report(0.2, [0.75, 0.2])
        self.assertEqual(row['benchmark_reward'], 0.75)
        self.assertEqual(row['final_public_reward'], 0.2)
        self.assertFalse(row['success'])

    def test_prior_success_is_reported_as_success(self):
        row = purchase_score_report(0.5, [1, 0.5])
        self.assertEqual(row['benchmark_reward'], 1)
        self.assertTrue(row['success'])
        self.assertFalse(row['native_final_success'])

    def test_no_purchase_is_zero_and_malformed_reward_is_rejected(self):
        self.assertEqual(purchase_score_report(None, [])['benchmark_reward'], 0)
        with self.assertRaises(ProtocolError):
            purchase_score_report(0.5, [None])


if __name__ == '__main__':
    unittest.main()
