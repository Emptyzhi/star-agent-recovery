from copy import deepcopy
import json
import unittest

from agentrewind_bench.feedback import model_feedback


class Game24FeedbackTests(unittest.TestCase):
    def test_grammar_placeholders_are_not_advertised_as_executable_actions(self):
        receipt = dict(observation='Input: 1 1 4 6', done=False, reward=None,
            accepted=True, state=dict(puzzle='1 1 4 6', numbers=['1', '1', '4', '6'],
                sources=['constant.1', 'constant.2', 'constant.3', 'constant.4'],
                action_count=0, max_actions=12, accepted_operations=0,
                done=False, reward=None, available_actions=[
                    '<available binary operation using two current numbers>',
                    'answer: <complete expression using the four original numbers> = 24']))
        original = deepcopy(receipt)
        visible = model_feedback(receipt, 'game24')
        self.assertNotIn('available_actions', visible['state'])
        self.assertNotIn('<available binary operation', json.dumps(visible))
        self.assertEqual(visible['state'], {
            key: value for key, value in original['state'].items() if key != 'available_actions'})
        self.assertEqual(receipt, original)
        self.assertEqual(visible['observation'], original['observation'])

    def test_concrete_alfworld_commands_remain_visible(self):
        receipt = dict(observation='Room', done=False, reward=None, accepted=True,
                       state=dict(available_actions=['look', 'inventory']))
        self.assertEqual(model_feedback(receipt, 'alfworld')['available_actions'],
                         ['look', 'inventory'])


if __name__ == '__main__':
    unittest.main()
