"""Public environment receipts and independent branches for interactive ToT."""
import copy
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
import multiprocessing
import re
from types import FunctionType, SimpleNamespace
from urllib.parse import quote

from .config import ROOT
from .resident_transport import append
from .game24_adapter import Game24Environment, Game24Task

PREFIXES = {'pick_and_place': 'put', 'pick_clean_then_place': 'clean',
            'pick_heat_then_place': 'heat', 'pick_cool_then_place': 'cool',
            'look_at_obj': 'examine', 'pick_two_obj': 'puttwo'}


def native_url(website, state):
    sid = quote(state['session'], safe='')
    query = quote(quote(state.get('query_string', ''), safe=''), safe='')
    number = quote(str(state.get('page_num', 1)), safe='')
    asin = quote(state.get('asin', ''), safe='')
    options = quote(str(state.get('options', {})), safe='')
    routes = {'init': sid, 'search': f'search_results/{sid}/{query}/{number}',
              'item': f'item_page/{sid}/{asin}/{query}/{number}/{options}',
              'item_sub': f'item_sub_page/{sid}/{asin}/{query}/{number}/{quote(state.get("subpage", ""), safe="")}/{options}',
              'end': f'done/{sid}/{asin}/{options}'}
    return website.rstrip('/') + '/' + routes[state['page_type']]


class WebShop:
    def __init__(self, native_class, website, task_id, directory, server_session=None):
        import requests
        self.raw = native_class(website)
        self.task_session = 'fixed_' + str(task_id)
        self.session = self.task_session if server_session is None else server_session
        self.directory = directory
        self.http = requests.Session()
        original = self.raw.webshop_text.__func__

        def render(**state):
            url = native_url(website, state)

            def get(_author_url):
                response = self.http.get(url, timeout=1200)
                append(self.directory/'http.jsonl', dict(url=url, status=response.status_code))
                response.raise_for_status()
                return response

            namespace = dict(original.__globals__, requests=SimpleNamespace(get=get))
            function = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__)
            return function(self.raw, **state)

        self.raw.webshop_text = render

    def step(self, action):
        before = copy.deepcopy(self.raw.sessions)
        try:
            observation, reward, done = self.raw.step(self.session, action)
        except AssertionError:
            if action == 'reset':
                raise
            # GA exposes environment assertions as this explicit invalid-action receipt.
            import traceback
            append(self.directory/'environment.jsonl', dict(event='rejected', action=action,
                   before=before, after=copy.deepcopy(self.raw.sessions), traceback=traceback.format_exc()))
            return dict(observation='Invalid action!', reward=0, terminal=False)
        receipt = dict(observation=observation, reward=reward, terminal=bool(done))
        append(self.directory/'environment.jsonl', dict(action=action, before=before,
               after=copy.deepcopy(self.raw.sessions), receipt=receipt))
        return receipt

    def reset(self):
        return self.step('reset')

    def snapshot(self):
        state = copy.deepcopy(self.raw.sessions[self.session])
        state['session'] = self.task_session
        return {self.task_session: state}

    def restore(self, state):
        restored = copy.deepcopy(state[self.task_session])
        restored['session'] = self.session
        self.raw.sessions = {self.session: restored}

    def close(self):
        self.http.close()


class ALFWorld:
    def __init__(self, gamefile, config, directory):
        if multiprocessing.current_process().name == 'MainProcess':
            raise RuntimeError('ALFWorld must be created in an isolated environment process')
        from alfworld.agents.environment import AlfredTWEnv
        self.gamefile, self.directory = gamefile, directory
        provider = object.__new__(AlfredTWEnv)
        provider.config = copy.deepcopy(config)
        provider.train_eval = 'eval_out_of_distribution'
        provider.game_files, provider.num_games = [gamefile], 1
        self.raw = provider.init_env(batch_size=1)

    def reset(self):
        obs, info = self.raw.reset()
        if info['extra.gamefile'][0] != self.gamefile:
            raise ValueError('ALFWorld reset changed the selected task')
        receipt = dict(observation='\n'.join(obs[0].split('\n\n')[1:]), reward=0, terminal=False)
        append(self.directory/'environment.jsonl', dict(event='reset', receipt=receipt))
        return receipt

    def step(self, action):
        obs, _, done, info = self.raw.step([action])
        receipt = dict(observation=obs[0], reward=int(info['won'][0]), terminal=bool(done[0]))
        append(self.directory/'environment.jsonl', dict(action=action, receipt=receipt))
        return receipt

    def close(self):
        self.raw.close()


def load_inputs(req, game24_class):
    """Load once before worker threads; no process-global cwd changes."""
    dataset = req['dataset']
    if dataset == 'game24':
        template = dict(csv=str(Path(req['csv']).resolve()),
                        examples=(ROOT/'prompts/game24/game24_base_actionly.txt').read_text(encoding='utf-8'))
        with Path(req['csv']).open(newline='', encoding='utf-8') as stream:
            count = sum(1 for _ in csv.DictReader(stream))
    elif dataset == 'webshop':
        path = ROOT/'vendor/ga/test_webshop/web_react.py'
        spec = importlib.util.spec_from_file_location('tot_native_webshop', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        template = dict(native_class=module.webshopEnv, examples=module.prompt1_actonly, website=req['website'])
        count = 500
    else:
        import yaml
        from importlib.metadata import version
        runtime = req.get('alfworld_runtime')
        if runtime and runtime not in sys.path:
            sys.path.insert(0, runtime)
        # TextWorld is the selected backend. Match LATS's import-only vision
        # compatibility surface; visual execution remains explicitly unavailable.
        compat = str(ROOT/'runtime_compat')
        if not Path(compat).is_dir():
            raise RuntimeError('ALFWorld text-only compatibility package is missing')
        if compat in sys.path:
            sys.path.remove(compat)
        sys.path.insert(0, compat)
        if version('alfworld') != '0.3.5':
            raise ValueError('this adapter requires ALFWorld 0.3.5')
        if 'alfworld_data' in req:
            os.environ['ALFWORLD_DATA'] = req['alfworld_data']
        games = json.loads(Path(req['manifest']).read_text(encoding='utf-8'))['test']
        if not games or len(games) != len(set(games)):
            raise ValueError('ALFWorld manifest must contain distinct task paths')
        config = yaml.safe_load(Path(req['alfworld_config']).read_text(encoding='utf-8'))
        config['general']['random_seed'] = req['environment_seed']
        section = {'dqn': 'rl', 'dagger': 'dagger'}[config['general']['training_method']]
        config[section]['training']['max_nb_steps_per_episode'] = req['experiment']['search']['max_depth'] + 1
        demos = json.loads((ROOT/'vendor/ga/prompts/alfworld/alfworld_base_prompts.json').read_text(encoding='utf-8'))
        template, count = dict(games=games, config=config, demos=demos), len(games)
    if any(i >= count for i in req['task_ids']):
        raise ValueError('task ID exceeds the selected dataset')
    return template


class BranchWorld:
    """Serial reference world for fixtures; production uses IsolatedBranchWorld."""
    def __init__(self, env, dataset, examples, horizon, directory):
        self.env, self.dataset, self.examples = env, dataset, examples
        self.horizon, self.directory = horizon, directory
        self.cache = {(): env.reset()}
        self.snapshots = {(): env.snapshot()} if dataset in ('webshop', 'game24') else {}
        self.replay_calls, self.transition_calls = 0, 0

    def restart_search(self):
        root = self.env.reset()
        if root != self.cache[()]:
            raise ValueError('BFS restart changed the task root')
        self.cache = {(): root}
        self.snapshots = {(): self.env.snapshot()} if self.dataset in ('webshop', 'game24') else {}
        append(self.directory/'branches.jsonl', dict(event='round_reset',
            transition_calls=self.transition_calls, replay_calls=self.replay_calls))

    def restore(self, actions):
        if self.dataset in ('webshop', 'game24'):
            self.env.restore(self.snapshots[actions])
        else:
            root = self.env.reset()
            if root != self.cache[()]:
                raise ValueError('ALFWorld replay root differs from original receipt')
            for i, action in enumerate(actions):
                receipt = self.env.step(action)
                self.replay_calls += 1
                expected = self.cache[actions[:i+1]]
                if any(receipt[k] != expected[k] for k in ('observation', 'reward', 'terminal')):
                    raise ValueError('ALFWorld replay differs from original receipt')
        append(self.directory/'branches.jsonl', dict(event='restore', actions=actions,
               replay_calls=self.replay_calls))

    def at(self, actions):
        actions = tuple(actions)
        if actions in self.cache:
            return self.cache[actions]
        parent = actions[:-1]
        previous = self.at(parent)
        if previous['terminal'] or len(parent) >= self.horizon:
            raise ValueError('cannot extend a completed branch')
        self.restore(parent)
        receipt = self.env.step(actions[-1])
        self.transition_calls += 1
        self.cache[actions] = receipt
        if self.dataset in ('webshop', 'game24'):
            self.snapshots[actions] = self.env.snapshot()
        append(self.directory/'branches.jsonl', dict(event='transition', actions=actions,
               receipt=receipt, horizon_reached=len(actions) == self.horizon,
               snapshot=self.snapshots.get(actions)))
        return receipt

    def trajectory(self, actions):
        text = self.cache[()]['observation']
        for i, action in enumerate(actions):
            text += '\nAction: ' + action + '\nObservation: ' + self.at(actions[:i+1])['observation']
        return text

    def submit(self, actions):
        from .executors import Future
        future = Future()
        future.set_result(self.at(actions))
        return future

    def commit(self, actions, result):
        return result


class InteractiveTask:
    def __init__(self, world):
        self.world = world
        self.value_cache = {}

    def get_input(self, idx):
        return self.world.cache[()]['observation']

    def actions(self, y):
        return tuple(y.splitlines())

    def receipt(self, y):
        return self.world.at(self.actions(y))

    def cot_prompt_wrap(self, x, y):
        action_slot = '>' if self.world.dataset == 'alfworld' else 'Action:'
        return (self.world.examples + '\n\nCurrent branch:\n' + self.world.trajectory(self.actions(y))
                + '\nReturn one next environment action on one line, without an Action prefix.\n' + action_slot)

    def propose_prompt_wrap(self, x, y):
        return (self.world.examples + '\n\nCurrent branch:\n' + self.world.trajectory(self.actions(y))
                + '\nPropose alternative next environment actions, one candidate per line. '
                  'Do not include numbering, prefixes, explanations, or a sequence of actions.')

    def value_prompt_wrap(self, x, y):
        from .score_protocol import value_prompt
        return value_prompt(self.world.dataset, self.world.trajectory(self.actions(y)))

    def value_outputs_unwrap(self, x, y, outputs):
        from .score_protocol import parse_labels
        return parse_labels(outputs)

    def test_output(self, idx, y):
        return {'r': self.receipt(y)['reward']}


class WebShopTask(InteractiveTask):
    @staticmethod
    def normalize_action(text):
        # Match LATS parse_webshop_action after its existing first-line framing.
        action = re.sub(r'^Action:\s*', '', text.strip(), count=1, flags=re.I)
        if not action or '\n' in action:
            raise ValueError('ToT WebShop actor returned no single-line action candidate')
        return action


def create_task(req, task_id, template, directory, environment_pool):
    from .isolated_environments import IsolatedBranchWorld
    spec = dict(dataset=req['dataset'], task_id=task_id, directory=str(directory))
    if req['dataset'] == 'game24':
        spec['csv'] = template['csv']
        examples = template['examples']
    elif req['dataset'] == 'webshop':
        spec.update(native_class=template['native_class'], website=template['website'])
        examples = template['examples']
    else:
        game = template['games'][task_id]
        family = [v for k, v in PREFIXES.items() if Path(game).parent.parent.name.startswith(k)]
        if len(family) != 1:
            raise ValueError('unknown ALFWorld task family')
        examples = ('Interact with a household to solve a task. Here are two examples.\n'
                    + template['demos']['act_'+family[0]+'_1'] + template['demos']['act_'+family[0]+'_0'])
        spec.update(gamefile=game, config=template['config'])
    world = IsolatedBranchWorld(spec, examples, req['experiment']['search']['max_depth'],
                                directory, environment_pool)
    task_class = {'game24': Game24Task, 'webshop': WebShopTask, 'alfworld': InteractiveTask}[req['dataset']]
    task = task_class(world)
    return task, world
