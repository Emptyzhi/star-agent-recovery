from pathlib import Path
import tempfile
from types import ModuleType,SimpleNamespace
from unittest.mock import patch
import unittest

from alfworld_methods import Environment


class Raw:
    def __init__(self,path):
        self.path=path;self.batch_env=SimpleNamespace(reset=self.reset)
    def reset(self):return ['Header\n\nYour task is to: place a cup'],{'extra.gamefile':[self.path]}
    def step(self,actions):return ['Nothing happens.'],[0],[False],{'won':[False]}
    def close(self):pass


class Provider:
    def init_env(self,batch_size):
        assert self.config['general']['random_seed']==1
        assert self.game_files==['exact-task'] and self.num_games==batch_size==1
        return Raw(self.game_files[0])


class Binding(unittest.TestCase):
    def test_exact_task_reset_and_rollback_reset_are_real_environment_calls(self):
        module=ModuleType('alfworld.agents.environment');module.AlfredTWEnv=Provider
        config={'general':{'random_seed':1}}
        with tempfile.TemporaryDirectory() as folder,patch.dict('sys.modules',{'alfworld.agents.environment':module}):
            env=Environment('exact-task',config,Path(folder),SimpleNamespace(trial=0))
            self.assertEqual(env.reset()[1]['extra.gamefile'],['exact-task'])
            self.assertFalse(env.step(['look'])[3]['won'][0])
            env.replay_reset()
            self.assertEqual((env.calls,env.resets),(1,2))
            self.assertEqual(config,{'general':{'random_seed':1}})
            env.close()


if __name__=='__main__':unittest.main()
