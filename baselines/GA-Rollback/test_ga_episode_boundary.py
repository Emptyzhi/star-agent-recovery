from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from webshop_methods import execute


class EpisodeBoundary(unittest.TestCase):
    def test_native_episode_assertion_is_failure_and_keeps_error_evidence(self):
        class Env:
            def __init__(self,*args):
                self.session='fixed_0';self.calls=3;self.terminal_rewards=[]
                self.last_receipt=('Actual page',0,False)
            def close(self):pass
        def run(*args,**kwargs):raise AssertionError('actual native invalid action during replay')
        author=SimpleNamespace(webshopEnv=object,BASE_PROMPT='author prompt',webshop_run=run)
        with tempfile.TemporaryDirectory() as temp,patch('webshop_methods.RecordedEnvironment',Env):
            folder=Path(temp)
            result=execute('ga_rollback',0,{'website':'http://site'},
                           {'ga_rollback':author},None,object(),folder)
            self.assertFalse(result['success']);self.assertEqual(result['reward'],0)
            self.assertTrue(result['author_episode_assertion_failure'])
            self.assertIn('actual native invalid action',
                          (folder/'author_episode_assertion_failure.json').read_text())
            self.assertNotIn('rollbacks',result)

    def test_transport_and_other_runtime_errors_are_not_scored(self):
        class Env:
            session='fixed_0'
            def __init__(self,*args):pass
            def close(self):pass
        def run(*args,**kwargs):raise RuntimeError('model transport failed')
        author=SimpleNamespace(webshopEnv=object,BASE_PROMPT='author prompt',webshop_run=run)
        with tempfile.TemporaryDirectory() as temp,patch('webshop_methods.RecordedEnvironment',Env):
            with self.assertRaisesRegex(RuntimeError,'model transport failed'):
                execute('ga_rollback',0,{'website':'http://site'},{'ga_rollback':author},None,object(),Path(temp))


if __name__=='__main__':unittest.main()
