import unittest

from test_game24.game24_env import Game24Environment, validate_answer


class Game24EnvironmentTests(unittest.TestCase):
    def test_valid_operations_and_final_answer(self):
        env = Game24Environment("4 4 6 8")
        self.assertTrue(env.execute("4 + 8 = 12").accepted)
        self.assertTrue(env.execute("6 - 4 = 2").accepted)
        self.assertTrue(env.execute("2 * 12 = 24").accepted)
        final = env.execute("answer: (6 - 4) * (4 + 8) = 24")
        self.assertTrue(final.accepted)
        self.assertEqual(final.reward, 1.0)
        self.assertTrue(env.done)

    def test_rejects_reused_operand_and_wrong_result(self):
        env = Game24Environment("1 2 3 4")
        reused = env.execute("1 + 1 = 2")
        wrong = env.execute("1 + 2 = 4")
        self.assertFalse(reused.accepted)
        self.assertFalse(wrong.accepted)
        self.assertFalse(env.done)

    def test_checkpoint_restores_public_arithmetic_state(self):
        env = Game24Environment("1 1 4 6")
        env.execute("1 + 1 = 2")
        checkpoint = env.checkpoint()
        env.execute("4 * 6 = 24")
        env.restore(checkpoint)
        self.assertEqual(env.public_state()["numbers"], ["4", "6", "2"])
        self.assertEqual(len(env.history), 1)

    def test_answer_uses_original_numbers_exactly_once(self):
        ok, _, _ = validate_answer("answer: (1 - 1) + (4 * 6) = 24", tuple(map(int, (1, 1, 4, 6))))
        bad, reason, _ = validate_answer("answer: 4 * 6 = 24", tuple(map(int, (1, 1, 4, 6))))
        self.assertTrue(ok)
        self.assertFalse(bad)
        self.assertIn("exactly once", reason)


if __name__ == "__main__":
    unittest.main()
