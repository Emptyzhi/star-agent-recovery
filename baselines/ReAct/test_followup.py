import json
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch

from dispatch_star import submit_validated_full


class Followup(unittest.TestCase):
    def request(self):
        return dict(run_id='smoke',dataset='alfworld',task_ids=[0,1],experiment=dict(
            method='reflexion',followup_full=dict(run_id='full',model='qwen')))
    def test_failed_smoke_never_submits(self):
        with patch('dispatch_star.subprocess.run') as run:
            submit_validated_full(self.request(),Path('unused'),[0,1])
            run.assert_not_called()
    def test_exact_coverage_required_but_score_is_not_a_selection_gate(self):
        with tempfile.TemporaryDirectory() as folder,patch('dispatch_star.subprocess.run') as run:
            root=Path(folder)
            with self.assertRaisesRegex(ValueError,'coverage'):submit_validated_full(self.request(),root,[0,0])
            for i in (0,1):
                p=root/('group_'+str(i))/('task_%03d'%i);p.mkdir(parents=True)
                (p/'summary.json').write_text(json.dumps(dict(complete=True,success=False,identity=dict(task_id=i))))
            submit_validated_full(self.request(),root,[0,0])
            cmd=run.call_args.args[0]
            self.assertEqual(cmd[0],sys.executable)
            self.assertEqual(cmd[cmd.index('--count')+1],'134')
            self.assertEqual(cmd[cmd.index('--validated-by')+1],'smoke')

    def test_output_capacity_tracks_dataset_specific_author_budget(self):
        from experiment_contracts import required_output_cap
        from resident_transport import GARollbackTransport
        from unittest.mock import Mock
        transport=Mock()
        transport.generate.return_value=dict(raw_output='analysis',selected_probabilities=[.5],output_tokens=1)
        GARollbackTransport(transport)._generate('prompt',max_new_tokens=1200,gen_logits=True)
        transport.generate.assert_called_once_with('prompt',1200)
        self.assertEqual(required_output_cap('alfworld','ga_rollback'),1200)
        self.assertEqual(required_output_cap('game24','ga_rollback'),500)
        self.assertEqual(required_output_cap('webshop','ga_rollback'),500)
        for dataset in ('alfworld','game24','webshop'):
            self.assertEqual(required_output_cap(dataset,'reflexion'),256)


if __name__=='__main__':unittest.main()
