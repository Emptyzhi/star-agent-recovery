"""Shard-owned environment workers and main-thread branch receipt commits."""
import copy
import multiprocessing
import os
from .executors import Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from multiprocessing.util import Finalize
from pathlib import Path
from tempfile import mkdtemp
from threading import Lock,local,get_ident
from uuid import uuid4

from .environments import ALFWorld, BranchWorld, Game24Environment, WebShop
from .resident_transport import append


def environment_executor(dataset, workers):
    if dataset == 'alfworld':
        return ProcessPoolExecutor(max_workers=workers,
                                   mp_context=multiprocessing.get_context('spawn'))
    return ThreadPoolExecutor(max_workers=workers, thread_name_prefix='tot_environment')


def make_environment(spec, directory):
    if spec['dataset'] == 'alfworld':
        return ALFWorld(spec['gamefile'], spec['config'], directory)
    if spec['dataset'] == 'game24':
        return Game24Environment(spec['csv'], spec['task_id'])
    session = 'tot_'+uuid4().hex+'_fixed_'+str(spec['task_id'])
    return WebShop(spec['native_class'], spec['website'], spec['task_id'], directory, session)


def execute_environment(env, dataset, actions, root, expected, snapshot):
    if not actions:
        receipt = env.reset()
        return dict(receipt=receipt, snapshot=env.snapshot() if dataset!='alfworld' else None,
                    replay_steps=0)
    if dataset == 'alfworld':
        if env.reset() != root:
            raise ValueError('isolated ALFWorld replay root differs from original receipt')
        for i, action in enumerate(actions):
            receipt = env.step(action)
            prefix = actions[:i+1]
            if prefix in expected and receipt != expected[prefix]:
                raise ValueError('isolated ALFWorld replay differs from original receipt')
            if receipt['terminal'] and i+1 < len(actions):
                raise ValueError('isolated replay extends a terminal branch')
        return dict(receipt=receipt, snapshot=None, replay_steps=len(actions)-1)
    env.restore(snapshot)
    receipt = env.step(actions[-1])
    return dict(receipt=receipt, snapshot=env.snapshot(), replay_steps=0)


_PROCESS_ENVS = {}


def process_environment_job(spec, directory, actions, root, expected, snapshot):
    """Spawned workers execute one TextWorld operation at a time, without a lock."""
    directory = Path(directory)
    key = spec['directory']
    if key not in _PROCESS_ENVS:
        env = make_environment(spec, directory)
        _PROCESS_ENVS[key] = env
        Finalize(env, env.close, exitpriority=10)
    env = _PROCESS_ENVS[key]
    env.directory = directory
    result = execute_environment(env, spec['dataset'], actions, root, expected, snapshot)
    append(directory/'receipt.jsonl', dict(actions=actions, result=result, pid=os.getpid(),
                                           thread_id=get_ident()))
    return result


class ThreadEnvironments:
    """Task-scoped handles, reused by persistent shard threads, never shared."""
    def __init__(self, spec):
        self.spec = spec
        self.local = local()
        self.lock = Lock()
        self.resources = []

    def execute(self, directory, actions, root, expected, snapshot):
        directory = Path(directory)
        env = getattr(self.local, 'env', None)
        if env is None:
            env = make_environment(self.spec, directory)
            self.local.env = env
            with self.lock:
                self.resources.append(env)
            if actions and env.reset() != root:
                raise ValueError('isolated worker reset changed original task receipt')
        env.directory = directory
        result = execute_environment(env, self.spec['dataset'], actions, root, expected, snapshot)
        append(directory/'receipt.jsonl', dict(actions=actions, result=result, pid=os.getpid(),
                                               thread_id=get_ident()))
        return result

    def close(self):
        for env in self.resources:
            env.close()


class IsolatedBranchWorld(BranchWorld):
    def __init__(self, spec, examples, horizon, directory, executor):
        self.spec, self.dataset, self.examples = spec, spec['dataset'], examples
        self.horizon, self.directory, self.executor = horizon, directory, executor
        self.job_root = self.directory/'candidate_envs'
        self.job_root.mkdir(parents=True, exist_ok=False)
        self.worker = ThreadEnvironments(spec) if self.dataset!='alfworld' else None
        self.replay_calls = self.transition_calls = self.ordinal = 0
        self.pending = {}
        try:
            root = self._submit_job((), None, {}, None).result()
        except Exception:
            if self.worker is not None:
                self.worker.close()
            raise
        self.cache = {(): root['receipt']}
        self.snapshots = {(): root['snapshot']}

    def _submit_job(self, actions, root, expected, snapshot):
        # The submitting task reserves a unique directory before a worker runs.
        directory = Path(mkdtemp(prefix=f'job_{self.ordinal:06d}_', dir=self.job_root))
        self.ordinal += 1
        if self.dataset == 'alfworld':
            return self.executor.submit(process_environment_job, self.spec, directory,
                                        actions, root, expected, snapshot)
        return self.executor.submit(self.worker.execute, directory, actions, root, expected, snapshot)

    def submit(self, actions):
        actions = tuple(actions)
        if actions in self.cache:
            future = Future()
            future.set_result(dict(receipt=self.cache[actions], snapshot=self.snapshots[actions], replay_steps=0))
            return future
        if actions in self.pending:
            return self.pending[actions]
        parent = actions[:-1]
        previous = self.at(parent)
        if previous['terminal'] or len(parent) >= self.horizon:
            raise ValueError('cannot extend a completed branch')
        expected = {actions[:i]: self.cache[actions[:i]] for i in range(1, len(actions))}
        future = self._submit_job(actions, self.cache[()], copy.deepcopy(expected),
                                  copy.deepcopy(self.snapshots[parent]))
        self.pending[actions] = future
        return future

    def commit(self, actions, result):
        actions = tuple(actions)
        receipt = result['receipt']
        if actions in self.cache:
            if self.cache[actions]!=receipt or self.snapshots[actions]!=result['snapshot']:
                raise ValueError('isolated candidate conflicts with cached branch')
            return self.cache[actions]
        self.cache[actions], self.snapshots[actions] = receipt, result['snapshot']
        self.pending.pop(actions)
        self.transition_calls += 1
        self.replay_calls += result['replay_steps']
        append(self.directory/'branches.jsonl', dict(event='isolated_transition', actions=actions,
            receipt=receipt, snapshot=result['snapshot'], replay_steps=result['replay_steps'],
            horizon_reached=len(actions)==self.horizon))
        return receipt

    def at(self, actions):
        actions = tuple(actions)
        if actions in self.cache:
            return self.cache[actions]
        return self.commit(actions, self.submit(actions).result())

    def restore(self, actions):
        raise RuntimeError('isolated branches must be restored inside their environment worker')

    def restart_search(self):
        if self.pending:
            raise RuntimeError('cannot restart BFS with pending branch jobs')
        root = self._submit_job((), None, {}, None).result()
        if root['receipt'] != self.cache[()]:
            raise ValueError('BFS restart changed the task root')
        self.cache, self.snapshots = {(): root['receipt']}, {(): root['snapshot']}
        append(self.directory/'branches.jsonl', dict(event='round_reset',
            transition_calls=self.transition_calls, replay_calls=self.replay_calls))

    def close(self):
        wait(list(self.pending.values()))
        if self.worker is not None:
            self.worker.close()
