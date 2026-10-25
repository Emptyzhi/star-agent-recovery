import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import dispatch_star

ROOT=Path(__file__).resolve().parent
METHOD='lats'


class DispatcherTests(unittest.TestCase):
    def test_moved_llama_lats_never_dispatches_on_gpu9(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package = root/'packages/lats'
            package.mkdir(parents=True)
            migration = root/'score_v7'
            migration.mkdir()
            (migration/'LATS_GPU13_MIGRATION.json').write_text(json.dumps({'lats_runs':['moved']}))
            request = root/'request.json'
            request.write_text(json.dumps({'run_id':'moved', 'package':str(package)}))
            argv = ['dispatcher', '--request', str(request), '--model-path', '/model',
                '--dtype', 'bfloat16', '--results-root', str(root/'results'), '--defaults', 'plan.json',
                '--parent-job-id', '3847563', '--node', 'gpu9']
            with patch.object(sys, 'argv', argv), patch.object(dispatch_star, 'dispatch') as dispatch, \
                    patch.object(dispatch_star.time, 'sleep', side_effect=RuntimeError('test boundary')):
                with self.assertRaisesRegex(RuntimeError, 'test boundary'):
                    dispatch_star.main()
                dispatch.assert_not_called()
            self.assertEqual(json.loads((migration/'GPU9_LATS_TRANSFER_BLOCKED.json').read_text())['run_id'], 'moved')

    def test_post_qwen_correction_does_not_repeat_star(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);output=root/'results';(output/'probe').mkdir(parents=True)
            validation=output/'_validation/lats_empty_skip_unscored_v8/model_gpu9/boundary'
            validation.mkdir(parents=True);(validation/'PASS.json').write_text('[]')
            request=dict(run_id='probe',dataset='webshop',package=str(ROOT),task_ids=[0,1],
                workers_per_gpu=40,endpoints=['http://127.0.0.1:41581','http://127.0.0.1:41582'],
                environment_seed=1,transport='http',prompt_format='plain',website='http://site',
                experiment={'method':METHOD,'mode':'act'},after_qwen_scoring_rerun=True,
                sampling=dict(do_sample=True,temperature=.1,top_p=1.,top_k=20,repetition_penalty=1.05))
            path=root/'request.json';path.write_text(json.dumps(request))
            argv=['dispatcher','--request',str(path),'--model-path','/model','--dtype','bfloat16',
                  '--results-root',str(output),'--defaults','plan.json','--parent-job-id','3847563','--node','gpu9']
            with patch.object(sys,'argv',argv),patch.object(dispatch_star,'dispatch',return_value=0),\
                 patch.object(dispatch_star.subprocess,'run') as gate:
                self.assertEqual(dispatch_star.main(),0)
                gate.assert_not_called()

    def test_resolved_http_request_and_extra_gate_follow_native_exit(self):
        for dataset in ('game24','alfworld','webshop'):
            for code in (0,1):
                with self.subTest(dataset=dataset,code=code),tempfile.TemporaryDirectory() as temp:
                    root=Path(temp);output=root/'results';(output/'probe').mkdir(parents=True)
                    validation=output/'_validation'/'lats_empty_skip_unscored_v8'/'model_gpu9/boundary'
                    validation.mkdir(parents=True);(validation/'PASS.json').write_text('[]')
                    request=dict(run_id='probe',dataset=dataset,package=str(ROOT),task_ids=[0,1],
                        workers_per_gpu=40,endpoints=['http://127.0.0.1:41581','http://127.0.0.1:41582'],
                        environment_seed=1,transport='http',prompt_format='plain',
                        experiment={'method':METHOD,'mode':'act'},sampling=dict(do_sample=True,
                            temperature=.1,top_p=1.,top_k=20,repetition_penalty=1.05))
                    key={'game24':'csv','alfworld':'manifest','webshop':'website'}[dataset]
                    request[key]='http://127.0.0.1:3001' if dataset=='webshop' else '/data/tasks'
                    path=root/'request.json';path.write_text(json.dumps(request))
                    events=[]
                    def native(actual,*args):
                        self.assertEqual(actual['task_ids'],[0,1])
                        self.assertEqual(actual['endpoints'],request['endpoints'])
                        self.assertEqual(actual['workers_per_gpu'],40)
                        self.assertEqual(actual['max_num_seqs'],40)
                        self.assertEqual(actual['experiment']['search']['max_output_tokens'],100)
                        self.assertEqual(actual['experiment']['search']['iterations'],2)
                        events.append('native');return code
                    def gate(*args,**kwargs):events.append('star_gate');return SimpleNamespace(returncode=0)
                    argv=['dispatcher','--request',str(path),'--model-path','/model','--dtype','bfloat16',
                        '--results-root',str(output),'--defaults','plan.json','--parent-job-id','3847563','--node','gpu9']
                    with patch.object(sys,'argv',argv),patch.object(dispatch_star,'dispatch',side_effect=native),\
                         patch.object(dispatch_star.subprocess,'run',side_effect=gate):
                        self.assertEqual(dispatch_star.main(),code)
                    expected=['native','star_gate'] if METHOD=='lats' and dataset=='webshop' and code==0 else ['native']
                    self.assertEqual(events,expected)


if __name__=='__main__':unittest.main()
