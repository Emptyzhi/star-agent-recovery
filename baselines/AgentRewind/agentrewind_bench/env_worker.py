"""Native environments in an isolated interpreter; Python 3.10-compatible RPC."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
import traceback


class NativeEnvironment:
    def __init__(self, config):
        self.config = config
        self.root = Path(config['native_adapter_root'])
        sys.path.insert(0, str(self.root/'source'))
        sys.path.insert(0, str(self.root))
        self.dataset = config['dataset']
        self.task = config['task_id']
        self.checkpoints = {}
        self.actions = self.restores = 0
        self.initial = None
        self.current = None
        if self.dataset == 'game24':
            from test_game24.game24_env import Game24Environment
            with Path(config['csv']).open(encoding='utf-8', newline='') as stream:
                puzzles = list(csv.DictReader(stream))
            self.env = Game24Environment(puzzles[self.task]['Puzzles'])
            self.base_prompt = (self.root/'prompts/game24/game24_base_actionly.txt').read_text(encoding='utf-8')
        elif self.dataset == 'webshop':
            from icr_webshop import WebShopEpisode
            self.base_prompt = (self.root/'prompts/webshop/webshop_reflexion_actionly_prompt.txt').read_text(encoding='utf-8')
            self.env = WebShopEpisode(config['website'],config['session_id'],self.task,self.base_prompt)
        elif self.dataset == 'alfworld':
            import importlib.metadata
            import yaml
            from alfworld.agents.environment import AlfredTWEnv
            from alf_runtime import ALF_ENV_LOCK, ThreadSafeAlfEpisode
            if importlib.metadata.version('alfworld') != '0.3.5':
                raise ValueError('ALFWorld requires the existing 0.3.5 environment')
            games = json.loads(Path(config['manifest']).read_text(encoding='utf-8'))[config['split']]
            gamefile = games[self.task]
            config_path = Path(config['environment_config']) if config['environment_config'] is not None else self.root/'vendor/base_config.yaml'
            environment_config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
            environment_config['general']['random_seed'] = config['environment_seed']
            provider = object.__new__(AlfredTWEnv)
            provider.config = deepcopy(environment_config)
            provider.train_eval = 'eval_out_of_distribution'
            provider.game_files = [gamefile]
            provider.num_games = 1
            families = {'pick_and_place':'put','pick_clean_then_place':'clean','pick_heat_then_place':'heat',
                        'pick_cool_then_place':'cool','look_at_obj':'examine','pick_two_obj':'puttwo'}
            family = [v for k,v in families.items() if Path(gamefile).parent.parent.name.startswith(k)]
            if len(family) != 1:
                raise ValueError('task has no unique native demonstration family')
            demos = json.loads((self.root/'prompts/alfworld/alfworld_base_prompts.json').read_text(encoding='utf-8'))
            self.base_prompt = 'Interact with a household to solve a task. Here are two examples.\n'+demos['act_'+family[0]+'_1']+demos['act_'+family[0]+'_0']
            with ALF_ENV_LOCK:
                raw = provider.init_env(batch_size=1)
            self.env = ThreadSafeAlfEpisode(raw,gamefile,self.base_prompt,None,
                                          action_limit=config['initial_actions'],emit=self.emit)
        else:
            raise ValueError('unknown dataset')
        self.trace = Path(config['environment_trace']).open('x', encoding='utf-8', buffering=1)

    def emit(self,event,**data):
        self.trace.write(json.dumps(dict(event=event,**data),ensure_ascii=False)+'\n')

    def receipt(self,row):
        return dict(observation=row['observation'],state=deepcopy(row['after']),
                    done=row['done'],reward=row['public_terminal_score'],
                    accepted=row['native_accepted'],native_receipt=deepcopy(row))

    def reset(self):
        if self.current is not None:
            raise ValueError('reset is initial-only; use checkpoints for recovery')
        if self.dataset == 'game24':
            state = self.env.reset()
            self.current = dict(observation='Input: '+self.env.puzzle,state=state,
                                done=state['done'],reward=state['reward'],accepted=True)
        else:
            self.current = self.receipt(self.env.reset())
        self.initial = deepcopy(self.current)
        return deepcopy(self.current)

    def observe(self):
        if self.current is None:
            raise ValueError('environment is not initialized')
        return deepcopy(self.current)

    def execute(self,action):
        if self.dataset == 'game24':
            row = self.env.execute(action).to_dict()
            remaining='numbers left: '+' '.join(row['after']['numbers'])
            observation=(remaining if row['accepted'] and not row['done'] else
                         row['detail']+'; '+remaining)
            self.current = dict(observation=observation,state=row['after'],done=row['done'],
                                reward=row['reward'],accepted=row['accepted'],native_receipt=row)
        else:
            parsed = self.env.parse(action)
            self.current = self.receipt(self.env.step(parsed.action))
            self.current['action_parser'] = deepcopy(parsed.evidence)
            self.current['executed_action'] = parsed.action
        self.actions += 1
        return deepcopy(self.current)

    def checkpoint(self):
        key = 'checkpoint_'+str(len(self.checkpoints))
        self.checkpoints[key] = (self.env.checkpoint(),deepcopy(self.current))
        return key

    def restore(self,key):
        native,current = self.checkpoints[key]
        report = self.env.restore(deepcopy(native))
        self.current = deepcopy(current)
        self.restores += 1
        return dict(native_restoration=report,checkpoint=key)

    def set_window(self,actions):
        if type(actions) is not int or actions < 1:
            raise ValueError('invalid native action window')
        if self.dataset == 'game24':
            self.env.max_actions = self.env.action_count+actions
        elif self.dataset == 'alfworld':
            self.env.action_limit = len(self.env.history)+actions

    def initial_prompt(self):
        return self.base_prompt

    def accounting(self):
        row = dict(executed_actions=self.actions,restores=self.restores)
        if self.dataset == 'alfworld':
            row.update(replay_actions=self.env.replay_actions,replay_resets=self.env.replay_resets,
                       backend_requests=self.env.backend_requests)
        elif self.dataset == 'webshop':
            row['backend_requests'] = self.env.backend_requests
        return row

    def close(self):
        if self.dataset != 'game24':
            self.env.close()
        self.trace.close()


def main():
    config = json.loads(sys.stdin.readline())
    # Native ALFWorld/TextWorld may print during initialization; protocol uses stdout alone.
    protocol = sys.stdout
    sys.stdout = sys.stderr
    env = NativeEnvironment(config)
    try:
        for line in sys.stdin:
            request = json.loads(line)
            method = request['method']
            if method not in ('reset','observe','execute','checkpoint','restore','set_window','initial_prompt','accounting','close'):
                raise ValueError('unknown environment RPC method')
            value = getattr(env,method)(*request['arguments'])
            protocol.write(json.dumps(dict(result=value),ensure_ascii=False)+'\n')
            protocol.flush()
            if method == 'close':
                return
    except Exception as error:
        protocol.write(json.dumps(dict(error=type(error).__name__,message=str(error),traceback=traceback.format_exc()))+'\n')
        protocol.flush()
        raise


if __name__ == '__main__':
    main()
