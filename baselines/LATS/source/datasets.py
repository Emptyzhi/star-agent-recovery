"""Dataset inputs and public environments; no model-dependent decisions."""
import copy
import csv
import importlib
import json
import os
from pathlib import Path
import sys
from threading import RLock
from types import SimpleNamespace
from urllib.parse import quote

from .config import ROOT
from .resident_transport import append

PREFIXES = {'pick_and_place': 'put', 'pick_clean_then_place': 'clean',
            'pick_heat_then_place': 'heat', 'pick_cool_then_place': 'cool',
            'look_at_obj': 'examine', 'pick_two_obj': 'puttwo'}

# TextWorld's grammar parser has process-global mutable state during import and
# game loading. ALFWorld reset/step must run in process-pool workers, not in
# task threads; this lock alone does not protect TextWorld's lazy reset parser.
ALFWORLD_INIT_LOCK = RLock()


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


def load_inputs(req):
    dataset = req['dataset']
    if dataset == 'game24':
        with Path(req['csv']).open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        puzzles = [row['Puzzles'].strip() for row in rows if row.get('Puzzles')]
        # The normal CLI resolves 500 task IDs. Keeping the template at the
        # manifest's actual size also permits small offline protocol fixtures;
        # the task-id bounds check below still rejects an undersized full run.
        template, count = dict(puzzles=puzzles), len(puzzles)
    elif dataset == 'alfworld':
        import yaml
        from importlib.metadata import version
        # ALFWorld 0.3.5 imports torchvision through its detector module even
        # for text-only evaluation.  The cluster's torchvision binary is
        # incompatible with its Torch build; keep the documented text-only
        # compatibility package ahead of site-packages before ALFWorld loads.
        runtime = req.get('alfworld_runtime')
        if runtime:
            if runtime in sys.path:
                sys.path.remove(runtime)
            sys.path.insert(0, runtime)
        compat = ROOT / 'runtime_compat'
        if not compat.is_dir():
            raise RuntimeError('ALFWorld text-only compatibility package is missing')
        compat = str(compat)
        if compat in sys.path:
            sys.path.remove(compat)
        # Keep the shim ahead of the ALFWorld runtime and site-packages.
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
        # Prefix replay must not consume the newly authorized recovery window.
        from .config import validate_request
        from .round_budget import ROUND_ACTIONS
        search = validate_request(req)
        episode_cap = search.iterations*(search.round_actions or ROUND_ACTIONS['alfworld'])+1
        section = {'dqn': 'rl', 'dagger': 'dagger'}[config['general']['training_method']]
        config[section]['training']['max_nb_steps_per_episode'] = episode_cap
        template, count = dict(games=games, config=config), len(games)
    else:
        template, count = dict(website=req['website']), 500
    if max(req['task_ids']) >= count:
        raise ValueError('task ID exceeds the selected dataset')
    return template


class Environment:
    """One independent ALFWorld binding owned by one task or candidate."""
    def __init__(self, gamefile, config, directory, transport):
        from alfworld.agents.environment import AlfredTWEnv
        self.gamefile, self.directory, self.transport = gamefile, directory, transport
        provider = object.__new__(AlfredTWEnv)
        provider.config = copy.deepcopy(config)
        provider.train_eval = 'eval_out_of_distribution'
        provider.game_files, provider.num_games = [gamefile], 1
        with ALFWORLD_INIT_LOCK:
            self.raw = provider.init_env(batch_size=1)

    def reset(self):
        result = self.raw.reset()
        if result[1]['extra.gamefile'][0] != self.gamefile:
            raise ValueError('reset changed task')
        append(self.directory/'events.jsonl', dict(event='environment_reset',
            gamefile=self.gamefile, observation=result[0]))
        return result

    def step(self, actions):
        result = self.raw.step(actions)
        append(self.directory/'events.jsonl', dict(event='environment_step',
            action=actions[0], observation=result[0][0], done=bool(result[2][0]),
            won=bool(result[3]['won'][0])))
        return result

    def close(self):
        self.raw.close()


def environment_for(template, task_id, directory, transport):
    env = copy.copy(template)
    env.idx, env.step_cnt, env.value_cache = task_id-1, 0, []
    native_step, native_get_case = env.step, env.get_case

    def step(action):
        before = dict(numbers=list(env.value_cache), step_count=env.step_cnt)
        result = native_step(action)
        append(directory/'events.jsonl', dict(event='environment_step', action=action,
            before=before, after=dict(numbers=list(env.value_cache), step_count=env.step_cnt), receipt=result))
        return result

    def get_case(reset=False):
        result = native_get_case(reset=reset)
        append(directory/'events.jsonl', dict(event='environment_reset', reset=reset,
            task_id=env.idx, puzzle=result))
        return result

    env.step, env.get_case = step, get_case
    return env


def create_world(req, task_id, template, transport, directory):
    from .search_environments import BranchWorld
    if req['dataset'] not in ('game24', 'alfworld'):
        raise ValueError('WebShop uses the native LATS environment')
    authors = {'react': SimpleNamespace(REACT_FILE='game24_base_react.txt')}
    return BranchWorld(req['dataset'], task_id, template, authors, transport, directory)
