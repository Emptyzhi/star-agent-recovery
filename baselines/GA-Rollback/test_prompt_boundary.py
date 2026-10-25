"""Exercise ordinary and post-rollback actor queries without loading models."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parent


class History:
    def __init__(self, *args):
        self.history_len = 0
        self.base_query = ''

    def gen_query(self, exp):
        return 'retained numbered history\n'

    def add(self, kind, value):
        if kind == 'action':
            self.history_len += 1

    def rollback_history(self, count):
        self.history_len -= count

    def get_actions(self):
        return ['look']


class Model:
    def __init__(self):
        self.prompts = []

    def _generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return dict(text='look')


def functions(dataset):
    filename = 'alf_rollback.py' if dataset == 'alfworld' else 'web_rollback.py'
    path = ROOT / 'vendor/ga' / ('test_' + dataset) / filename
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = {'rollback', 'alfworld_run' if dataset == 'alfworld' else 'webshop_run'}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == names
    scope = dict(List=list, IntraHistory=History, format_text=lambda text: text,
                 process_ob=lambda text: text)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), scope)
    return scope


class PromptBoundary(unittest.TestCase):
    def test_alfworld_ordinary_and_restored_actor_slots(self):
        native = functions('alfworld')
        model = Model()
        env = SimpleNamespace(step=lambda actions: (['success'], [1], [True], {'won': [True]}),
                              batch_env=SimpleNamespace(reset=lambda: None))
        self.assertEqual(native['alfworld_run'](env, 'examples', model, model, False,
                         to_print=False, ob='task', max_rollback_num=0)[0], 1)
        history = History()
        history.history_len = 3
        native['rollback'](env, history, model, 3, [])
        self.assertEqual(model.prompts, ['retained numbered history\nAct 1>',
                                        'retained numbered history\nAct 3>'])

    def test_webshop_ordinary_and_restored_actor_slots(self):
        native = functions('webshop')
        model = Model()
        env = SimpleNamespace(step=lambda idx, action: ('success', 1, True))
        self.assertEqual(native['webshop_run']('fixed_0', env, 'examples', model, model,
                         False, to_print=False, max_rollback_num=0)[0], 1)
        history = History()
        history.history_len = 3
        native['rollback']('fixed_0', env, history, model, 3, [])
        self.assertEqual(model.prompts, ['retained numbered history\nAction 1:',
                                        'retained numbered history\nAction 3:'])


if __name__ == '__main__':
    unittest.main()
