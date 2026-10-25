from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from search_adaptations import execute_tot,execute_lats
from search_environments import BranchWorld,LatsEnvironment
from test_lats_contract import author_core
from tot_game24 import load_tot_core


class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):return messages[0]['content']


class World:
    horizon=3;task_id=0;examples='Task examples.'
    def __init__(self,directory):
        self.directory=directory;self.cache={():dict(observation='Solve the task.',reward=0,terminal=False)}
    def at(self,actions):
        actions=tuple(actions)
        if actions not in self.cache:
            if self.at(actions[:-1])['terminal']:raise AssertionError('action after terminal')
            self.cache[actions]=dict(observation='Receipt '+actions[-1],reward=int(actions[-1]=='finish'),
                terminal=actions[-1] in ('bad','finish') or len(actions)==self.horizon)
        return self.cache[actions]
    def trajectory(self,actions):
        return 'Solve the task.\n'+'\n'.join('Action: '+a+'\nObservation: '+self.at(actions[:i+1])['observation'] for i,a in enumerate(actions))


class Model:
    sampling={'temperature':.1}
    def __init__(self):self.prompts=[];self.actors=0
    def generate(self,prompt,cap):
        self.prompts.append(prompt)
        if 'This attempt failed.' in prompt:return dict(raw_output='Try finish after go.')
        if 'correctness score from 1 to 10' in prompt:return dict(raw_output='5')
        if 'finish with exactly one label' in prompt:return dict(raw_output='likely')
        self.actors+=1
        if 'Observation: Receipt go' in prompt:return dict(raw_output='finish\nextra ignored generation')
        return dict(raw_output='go\nextra ignored generation')


class SearchContracts(unittest.TestCase):
    def test_native_tot_does_not_concatenate_steps_or_act_after_terminal(self):
        with tempfile.TemporaryDirectory() as folder:
            world=World(Path(folder));model=Model()
            result=execute_tot(world,load_tot_core(),Tokenizer(),model)
            self.assertTrue(result['success'])
            self.assertTrue(all(y=='go\nfinish\n' for y in result['outputs']))
            # Native BFS retains five sampled candidates, including duplicates:
            # five first actions, then five expansions for each of those beams.
            self.assertEqual(model.actors,30)
            self.assertFalse(any('extra ignored' in p for p in model.prompts))

    def test_lats_zero_reward_failure_reaches_reflection_and_later_actor(self):
        class LatsModel(Model):
            def generate(self,prompt,cap):
                if 'This attempt failed.' in prompt or 'correctness score from 1 to 10' in prompt:
                    return super().generate(prompt,cap)
                self.prompts.append(prompt);self.actors+=1
                return dict(raw_output='bad' if self.actors==1 else 'finish' if 'Receipt go' in prompt else 'go')
        task=SimpleNamespace(WebShopTask=SimpleNamespace(value_outputs_unwrap=lambda outputs:.5))
        with tempfile.TemporaryDirectory() as folder:
            world=World(Path(folder));model=LatsModel()
            result=execute_lats(world,(author_core(),task),Tokenizer(),model)
            self.assertTrue(result['success'])
            self.assertTrue(result['reflection_map'])
            self.assertTrue(any('Try finish after go.' in p and 'Current branch:' in p for p in model.prompts))

    def test_lats_branch_restoration_isolates_siblings(self):
        with tempfile.TemporaryDirectory() as folder:
            world=World(Path(folder));env=LatsEnvironment(world)
            env.step('fixed_0','reset');root=env.clone_state()
            env.step('fixed_0','go');env.sessions=root
            self.assertEqual(env.step('fixed_0','bad')[1:],(0,True))
            self.assertIn(('bad',),world.cache)
            self.assertNotIn(('go','bad'),world.cache)

    def test_alfworld_branch_restoration_replays_real_prefix_and_checks_receipt(self):
        class Env:
            def reset(self):self.actions=[];return ['Header\n\nRoot'],{}
            def step(self,actions):
                self.actions+=actions
                return ['Receipt '+','.join(self.actions)],[0],[False],{'won':[False]}
        with tempfile.TemporaryDirectory() as folder:
            world=BranchWorld.__new__(BranchWorld)
            world.dataset='alfworld';world.directory=Path(folder);world.env=Env();world.horizon=49
            world.cache={():dict(observation='Root',reward=0,terminal=False)};world.snapshots={():None}
            world.replay_calls=world.transition_calls=0
            world.at(('a','b'));world.at(('a','c'))
            self.assertEqual(world.env.actions,['a','c'])
            self.assertEqual(world.replay_calls,2)
            self.assertEqual(world.cache[('a','c')]['observation'],'Receipt a,c')
            world.cache[('a',)]['observation']='corrupted receipt'
            with self.assertRaisesRegex(ValueError,'differs'):world.at(('a','d'))


if __name__=='__main__':unittest.main()
