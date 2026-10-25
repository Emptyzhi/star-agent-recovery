from pathlib import Path
import subprocess
import sys
import unittest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'source'))
from scoped_recovery_batch import endpoint_port, select_shard, task_ids


class ResidentBatchContractTests(unittest.TestCase):
    def test_endpoint_port_uses_resident_service_port(self):
        self.assertEqual(endpoint_port('http://127.0.0.1:41382'), 41382)

    def test_endpoint_rejects_non_loopback_host(self):
        with self.assertRaises(RuntimeError):
            endpoint_port('http://example.com:41382')

    def test_task_manifest_preserves_exact_ids(self):
        self.assertEqual(task_ids('2,5,42'), [2, 5, 42])

    def test_gpu_shards_are_disjoint_and_cover_manifest(self):
        manifest = task_ids('2,5,6,9,14,15')
        gpu0 = select_shard(manifest, 0, 2)
        gpu1 = select_shard(manifest, 1, 2)
        self.assertEqual(gpu0, [2, 6, 14])
        self.assertEqual(gpu1, [5, 9, 15])
        self.assertEqual(sorted(gpu0 + gpu1), manifest)

    def test_resident_cli_has_no_conflicting_arguments(self):
        runner = Path(__file__).resolve().parents[1] / 'run_webshop.py'
        completed = subprocess.run([sys.executable, str(runner), '--help'],
                                   capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == '__main__':
    unittest.main()
