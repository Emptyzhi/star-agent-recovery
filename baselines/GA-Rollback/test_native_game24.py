import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from game24_methods import load_authors, execute, PACKAGE


class ScriptedTransport:
    def __init__(self, texts):
        self.texts=iter(texts)
        self.role='actor'
        self.trial=0
        self.calls=[]
    def generate(self,prompt,cap):
        text=next(self.texts)
        self.calls.append(dict(prompt=prompt,cap=cap,role=self.role,trial=self.trial))
        return dict(raw_output=text,selected_probabilities=[1.0],output_tokens=1)


class NativeGame24(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.authors=load_authors()
        cls.template=cls.authors['react'].Game24Task(str(PACKAGE/'vendor/ga/test_game24/24.csv'),1,500)
        cls.tokenizer=SimpleNamespace(pad_token_id=0,eos_token_id=0)

    def test_react_calls_author_loop_with_react_prompt(self):
        t=ScriptedTransport(['answer: 1 * 1 * 4 * 6 = 24\nignored'])
        with tempfile.TemporaryDirectory() as folder:
            result=execute('react',0,self.template,self.authors,self.tokenizer,t,Path(folder))
            self.assertTrue(result['success'])
            self.assertEqual(len(t.calls),1)
            self.assertIn('think:',t.calls[0]['prompt'].lower())
        self.assertEqual(self.template.idx,-1)

    def test_reflexion_delivers_memory_to_second_trial(self):
        # Five invalid final answers exhaust the author's four environment actions.
        t=ScriptedTransport(['answer: 1+1+4+6=24']*5+['Use multiplication.','answer: 1*1*4*6=24'])
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            result=execute('reflexion',0,self.template,self.authors,self.tokenizer,t,Path(folder))
        self.assertTrue(result['success'])
        reflections=[c for c in t.calls if c['role']=='reflection']
        self.assertEqual(len(reflections),1)
        self.assertEqual(reflections[0]['cap'],256)
        second=[c for c in t.calls if c['role']=='actor' and c['trial']==1]
        self.assertEqual(len(second),1)
        self.assertIn('use multiplication.',second[0]['prompt'])
        self.assertEqual(self.template.idx,-1)


if __name__=='__main__':
    unittest.main()
