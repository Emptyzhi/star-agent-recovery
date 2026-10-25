from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from source.round_budget import install_round_search


class Node:
    def __init__(self, state=None, question='', parent=None):
        self.state = state
        self.question = question
        self.parent = parent
        self.depth = 0 if parent is None else parent.depth + 1
        self.children = []
        self.reward = self.value = self.em = 0
        self.is_terminal = False
        self.env_state = {'prefix_depth': self.depth}


class RollbackDepthTests(unittest.TestCase):
    def test_five_search_rounds_restore_four_selected_nonroot_prefixes(self):
        selected, restored = [], []
        core = SimpleNamespace(Node=Node, env=SimpleNamespace(
            sessions={'prefix_depth': 0}, step=lambda idx, action: ('task', 0, False)))
        def select(root):
            node = root if not selected else selected[-1].children[0]
            selected.append(node)
            return node
        def expand(node, *args):
            restored.append(core.env.sessions.copy())
            node.children = [Node(dict(action='search[cup]'), node.question, node)]
        core.select_node = select
        core.expand_node = expand
        core.evaluate_node = lambda *args: None
        core.rollout = lambda node, *args, **kwargs: node
        core.backpropagate = lambda *args: None
        core.collect_all_nodes = lambda root: selected + [n.children[0] for n in selected]
        core.collect_trajectory = lambda node: str(node.depth)
        stages = []
        with tempfile.TemporaryDirectory() as folder:
            install_round_search(core, 15, Path(folder), lambda event, index, node: stages.append((event, index)))
            core.lats_search(None, None, 'task', iterations=5)
            events = (Path(folder) / 'rounds.jsonl').read_text()
        self.assertEqual(events.count('branch_rollback_selected'), 4)
        self.assertEqual([r['prefix_depth'] for r in restored], [0, 1, 2, 3, 4])
        self.assertEqual([index for event, index in stages if event == 'finished'], list(range(5)))


if __name__ == '__main__':
    unittest.main()
