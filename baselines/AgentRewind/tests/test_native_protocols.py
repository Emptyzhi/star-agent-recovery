"""Native parser/checkpoint contracts with HTTP/TextWorld test transports."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentrewind_bench.env_worker import NativeEnvironment
from agentrewind_bench.config import resolve
import run_agentrewind as cli


class NativeProtocolTests(unittest.TestCase):
    def config(self, directory, dataset):
        return dict(dataset=dataset,task_id=0,native_adapter_root=str(ROOT.parents[1]),
            website='http://127.0.0.1:9999',session_id='native_fixture_fixed_0',initial_actions=50,
            environment_seed=17,split='test',environment_config=None,
            environment_trace=str(Path(directory)/'trace.jsonl'))

    def test_webshop_exact_option_wire_and_restore_without_refunding_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = cli.parser().parse_args(['--dataset','webshop','--model','test-model',
                '--backend','openai','--endpoint','http://127.0.0.1:8000/v1','--task-ids','0',
                '--output-dir',tmp,'--run-namespace','native_fixture',
                '--website','http://127.0.0.1:9999'])
            settings,_ = resolve(args)
            env = NativeEnvironment(cli.environment_config(args,settings,0,Path(tmp)))
            wire = []
            def get(url,timeout):
                wire.append(url)
                page = env.env.environment._session['page_type']
                if page == 'init':
                    html = '<div>WebShop</div><div>Instruction:</div><div>Buy a red cup.</div><button>Search</button>'
                elif page == 'search':
                    html = '<div>WebShop</div><button>Back to Search</button><a class="product-link">A</a><div>Cup</div><div>$10</div>'
                elif page == 'item':
                    html = '<div>color</div><label>red</label><label>blue</label><div>Cup</div><button>Buy Now</button><button>Attributes</button>'
                elif page == 'end':
                    color = env.env.environment._session['options']['color']
                    html = '<div>Your score (min 0.0, max 1.0)</div><div>'+('1' if color=='red' else '.5')+'</div>'
                else:
                    raise AssertionError(page)
                return SimpleNamespace(text=html,raise_for_status=lambda:None)
            env.env.environment._http.get = get
            try:
                env.reset()
                env.execute('search[red cup]')
                env.execute('click[A]')
                checkpoint = env.checkpoint()
                at_checkpoint = deepcopy(env.observe())
                env.execute('click[blue]')
                self.assertEqual(env.execute('click[Buy Now]')['reward'], .5)
                requests = len(wire)
                env.restore(checkpoint)
                self.assertEqual(env.observe(),at_checkpoint)
                self.assertEqual(len(wire),requests)
                env.set_window(15)
                env.execute('click[red]')
                self.assertEqual(env.execute('click[Buy Now]')['reward'],1)
                self.assertIn('red', wire[-1])
                self.assertEqual(env.accounting(),dict(executed_actions=6,restores=1,backend_requests=7))
            finally:
                env.close()

    def test_alfworld_native_replay_restore_and_recovery_window(self):
        game = '/fixture/pick_and_place-Apple/trial0/game.tw-pddl'
        class TextWorldFixture:
            def __init__(self):
                self.stage = 0
                self.batch_env = self
            def info(self):
                return {'extra.gamefile':[game],'admissible_commands':[['look',
                    'take apple 1 from table 1','put apple 1 in/on table 2']], 'won':[self.stage==2]}
            def reset(self):
                self.stage = 0
                return ['HEADER\n\nYour task is to: put a apple in/on table 2.'],self.info()
            def step(self,actions):
                action = actions[0]
                if action == 'take apple 1 from table 1':
                    self.stage = 1
                    text = 'You pick up the apple 1 from the table 1.'
                elif action == 'put apple 1 in/on table 2' and self.stage==1:
                    self.stage = 2
                    text = 'You put the apple 1 in/on the table 2.'
                else:
                    text = 'You see table 1.'
                return [text],[0],[self.stage==2],self.info()
            def close(self): pass
        class Provider:
            def init_env(self,batch_size):
                self_test.assertEqual(self.config['general']['random_seed'],17)
                self_test.assertEqual(self.game_files,[game])
                return TextWorldFixture()
        self_test = self
        modules = {name:ModuleType(name) for name in ('alfworld','alfworld.agents','alfworld.agents.environment')}
        modules['alfworld.agents.environment'].AlfredTWEnv = Provider
        yaml_fixture = ModuleType('yaml')
        yaml_fixture.safe_load = lambda text: {'general':{}}
        modules['yaml'] = yaml_fixture
        with tempfile.TemporaryDirectory() as tmp,patch.dict(sys.modules,modules), \
                patch('importlib.metadata.version',return_value='0.3.5'):
            manifest = Path(tmp)/'manifest.json'
            manifest.write_text(json.dumps({'test':[game]}),encoding='utf-8')
            config = self.config(tmp,'alfworld')
            config['manifest'] = str(manifest)
            env = NativeEnvironment(config)
            try:
                env.reset()
                env.set_window(1)
                env.execute('take apple 1 from table 1')
                checkpoint = env.checkpoint()
                at_checkpoint = env.observe()
                env.set_window(1)
                self.assertEqual(env.execute('put apple 1 in/on table 2')['reward'],1)
                env.restore(checkpoint)
                self.assertEqual(env.observe(),at_checkpoint)
                env.set_window(50)
                self.assertEqual(env.env.action_limit,51)  # One replayed prefix action plus 50 new actions.
                self.assertEqual(env.execute('put apple 1 in/on table 2')['reward'],1)
                accounting = env.accounting()
                self.assertEqual(accounting['executed_actions'],3)
                self.assertEqual(accounting['replay_actions'],1)
                self.assertEqual(accounting['replay_resets'],1)
                self.assertEqual(accounting['backend_requests'],6)
            finally:
                env.close()


if __name__ == '__main__':
    unittest.main()
