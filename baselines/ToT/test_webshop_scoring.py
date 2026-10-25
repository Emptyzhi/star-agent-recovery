import unittest
from types import SimpleNamespace
from source.webshop_scoring import purchase_report, world_purchase_report


class PurchaseScoringTests(unittest.TestCase):
    def test_best_purchase_and_no_purchase(self):
        self.assertEqual(purchase_report([0.8, 0.2])['reward'], 0.8)
        self.assertTrue(purchase_report([1, 0.2])['success'])
        self.assertEqual(purchase_report([])['reward'], 0)

    def test_unselected_partial_purchase_counts_but_rejected_buy_does_not(self):
        world = SimpleNamespace(cache={
            (): {'reward': 0},
            ('search[x]',): {'reward': 0},
            ('click[Buy Now]',): {'reward': 0.9, 'terminal': False},
            ('search[x]', 'click[Buy Now]'): {'reward': 0.2, 'terminal': True},
            ('bad', 'click[Buy Now]'): {'reward': 0, 'terminal': False}}, snapshots={
            ('click[Buy Now]',): {'session': {'page_type': 'end'}},
            ('search[x]', 'click[Buy Now]'): {'session': {'page_type': 'end'}},
            ('bad', 'click[Buy Now]'): {'session': {'page_type': 'init'}}})
        report = world_purchase_report(world)
        self.assertEqual(report['purchase_rewards'], [0.9, 0.2])
        self.assertEqual(report['reward'], 0.9)

    def test_invalid_rewards_are_rejected(self):
        for value in (float('nan'), -1, 2, '0.5', None):
            with self.assertRaises(ValueError):
                purchase_report([value])
