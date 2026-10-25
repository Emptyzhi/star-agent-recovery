"""Check dataset-specific actor slots at root and after a branch action."""
from types import SimpleNamespace
import unittest
from source.game24_adapter import Game24Task
from source.environments import InteractiveTask


class PromptBoundary(unittest.TestCase):
    def test_game24_keeps_the_action_slot_empty(self):
        world = SimpleNamespace(examples='examples', cache={(): {'observation': 'numbers left: 1 1 4 6'}},
                                at=lambda actions: {'observation': 'numbers left: 1 4 6'})
        task = Game24Task(world)
        self.assertTrue(task.cot_prompt_wrap('', '').endswith('\nAct 1>'))
        self.assertTrue(task.cot_prompt_wrap('', '1 + 1 = 2\n').endswith('\nAct 2>'))

    def test_alfworld_uses_bare_marker_and_webshop_keeps_action_marker(self):
        for dataset, marker in [('alfworld', '>'), ('webshop', 'Action:')]:
            world = SimpleNamespace(dataset=dataset, examples='examples',
                                    trajectory=lambda actions: 'public observation')
            task = InteractiveTask(world)
            for branch in ('', 'look\n'):
                self.assertTrue(task.cot_prompt_wrap('', branch).endswith('\n' + marker))


if __name__ == '__main__':
    unittest.main()
