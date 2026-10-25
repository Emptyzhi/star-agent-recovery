import sys
from pathlib import Path
import unittest
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_agentrewind import parser
from agentrewind_bench.config import resolve


class SamplingConfigTests(unittest.TestCase):
    def settings(self,*extra):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'source').mkdir()
            args=parser().parse_args([
                '--dataset','webshop','--model','test-model',
                '--endpoint','http://127.0.0.1:41581','--task-ids','0',
                '--output-dir','unused','--run-namespace','sampling-test',
                '--native-adapter-root',directory,'--website','http://127.0.0.1:3001',
                '--parent-job-id','test-parent','--node','test-node',
                '--step-id','test-step','--gpu-index','0',*extra])
            settings,tasks=resolve(args)
            self.assertEqual(tasks,[0])
            return settings

    def test_shared_sampling_defaults_reach_resolved_settings(self):
        settings=self.settings()
        self.assertEqual((settings.temperature,settings.top_p,settings.top_k,
                          settings.repetition_penalty),(.1,1,20,1.05))

    def test_explicit_sampling_values_reach_resolved_settings(self):
        settings=self.settings('--temperature','.2','--top-p','.9',
                               '--top-k','30','--repetition-penalty','1.1')
        self.assertEqual((settings.temperature,settings.top_p,settings.top_k,
                          settings.repetition_penalty),(.2,.9,30,1.1))


if __name__=='__main__':unittest.main()
