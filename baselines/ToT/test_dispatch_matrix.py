import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import dispatch_star

ROOT=Path(__file__).resolve().parent
METHOD='tot'


class DispatcherTests(unittest.TestCase):
    def test_resolved_http_request_and_extra_gate_follow_native_exit(self):
        for dataset in ('game24','alfworld','webshop'):
            for code in (0,1):
                with self.subTest(dataset=dataset,code=code),tempfile.TemporaryDirectory() as temp:
                    root=Path(temp);output=root/'results';(output/'probe').mkdir(parents=True)
                    validation=output/'_validation'/'tot_label_stop_v9_attempt2'/'model_gpu9/boundary'
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
                        self.assertEqual(actual['experiment']['search']['iterations'],1)
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
