"""Protect TextWorld's process-global registry and parsers, not model requests."""
from threading import RLock
from alf_rebased_v64 import RebasedEpisode

ALF_ENV_LOCK = RLock()


class ThreadSafeAlfEpisode(RebasedEpisode):
    # restore calls step internally, so the environment lock must be reentrant.
    def reset(self):
        with ALF_ENV_LOCK:
            return super().reset()

    def step(self, action):
        with ALF_ENV_LOCK:
            return super().step(action)

    def restore(self, checkpoint):
        with ALF_ENV_LOCK:
            return super().restore(checkpoint)

    def close(self):
        with ALF_ENV_LOCK:
            return super().close()
