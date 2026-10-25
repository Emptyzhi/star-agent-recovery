"""Yellow follows native public action types without evaluating proposed answers."""
from dataclasses import replace
import unittest

from test_game24.action_language import (YELLOW_FINAL_LANGUAGE, YELLOW_LANGUAGE,
                                        grammar_for, language_for_state)
from test_game24.game24_env import Game24Environment
from test_game24_migration import BAD, GOOD, SHORT_SETTINGS, controller


class NativeActionKindTests(unittest.TestCase):
    def test_initial_state_still_allows_binary_or_direct_final_answer(self):
        env = Game24Environment('1 1 4 6')
        self.assertEqual(language_for_state(env.public_state()), YELLOW_LANGUAGE)
        self.assertTrue(env.execute(GOOD[-1]).accepted)

    def test_final_only_depends_on_public_action_type_not_value(self):
        for actions, value in ((GOOD[:3], '24'), (BAD[:3], '12')):
            with self.subTest(pool=value):
                env = Game24Environment('1 1 4 6')
                for action in actions:
                    self.assertTrue(env.execute(action).accepted)
                self.assertEqual(env.public_state()['numbers'], [value])
                self.assertEqual(language_for_state(env.public_state()), YELLOW_FINAL_LANGUAGE)

    def test_rejected_final_uses_final_only_yellow_and_same_action_slot(self):
        cfg = replace(SHORT_SETTINGS, initial_actions=5, ablation='without_red')
        model, requests = controller([*GOOD[:3], 'answer: 24 = 24', GOOD[-1]], settings=cfg)
        result = model.run(0, '1 1 4 6')
        self.assertTrue(result.success)
        yellow = [r for r in requests if r['role'] == 'yellow_action']
        self.assertEqual(len(yellow), 1)
        self.assertEqual(yellow[0]['parameter_language'], YELLOW_FINAL_LANGUAGE)
        self.assertEqual(result.forward_actions, 5)
        self.assertEqual(result.recovery_actions, 0)
        self.assertTrue(all('parameter_language' not in r for r in requests if r['role'] == 'green'))

    def test_language_does_not_assess_answer_arithmetic_or_original_operands(self):
        from lark import Lark
        from lark.exceptions import UnexpectedInput
        parser = Lark(grammar_for(YELLOW_FINAL_LANGUAGE), parser='lalr')
        for text in ('answer: 1 + 1 = 99\n', 'answer: 24 = 24\n'):
            parser.parse(text)
        with self.assertRaises(UnexpectedInput):
            parser.parse('1 + 1 = 2\n')

    def test_unknown_or_terminal_public_action_set_is_explicitly_rejected(self):
        for actions in ([], ['invented action type']):
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                language_for_state(dict(available_actions=actions))


if __name__ == '__main__':
    unittest.main()
