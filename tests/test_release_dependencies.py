import importlib.util
import hashlib
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('fetch_upstreams', Path(__file__).resolve().parents[1] / 'fetch_upstreams.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DependencyTests(unittest.TestCase):
    def test_patch_preserves_exact_result_and_rejects_wrong_base(self):
        before = b'a\nb\nc\n'
        after = b'a\nx\ny\nc\n'
        spec = dict(base_sha256=hashlib.sha256(before).hexdigest(),
                    result_sha256=hashlib.sha256(after).hexdigest(),
                    changes=[dict(start=1,end=2,lines=['x\n','y\n'])])
        self.assertEqual(MODULE.patch_text(before,spec),after)
        with self.assertRaises(ValueError):MODULE.patch_text(b'wrong\n',spec)

    def test_external_function_extraction_does_not_execute_module(self):
        raw=b'raise RuntimeError("side effect")\ndef chosen(x: List[str]):\n    return x\n'
        generated=MODULE.extract_functions(raw,['chosen'])
        namespace={}
        exec(generated,namespace)
        self.assertEqual(namespace['chosen'](['a']),['a'])
        self.assertNotIn('side effect',generated.decode())

    def test_target_escape_is_rejected(self):
        with self.assertRaises(ValueError):MODULE.safe_target('../outside')
