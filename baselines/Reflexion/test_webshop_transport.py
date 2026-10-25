import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import unittest
from urllib.parse import unquote,urlsplit

from webshop_methods import native_url,RecordedEnvironment


class NativeFixture:
    def __init__(self,website):self.website=website;self.sessions={}
    def webshop_text(self,session,page_type,**kwargs):
        import_missing_symbol = requests.get(self.website+'/'+session)
        return import_missing_symbol.text, {}
    def step(self,session,action):
        self.sessions[session]={'session':session,'page_type':'search','query_string':action[7:-1],'page_num':1}
        return self.webshop_text(**self.sessions[session])[0],0,False


class FakeHTTP:
    def __init__(self):self.urls=[]
    def get(self,url,timeout):
        self.urls.append(url)
        return SimpleNamespace(status_code=200,text='REAL_RECEIPT',raise_for_status=lambda:None)
    def close(self):pass


class Encoding(unittest.TestCase):
    def test_query_roundtrip_matches_deployed_backend(self):
        query='1/2 inch #red & blue 50%'
        url=native_url('http://127.0.0.1:3001',dict(session='fixed_7',page_type='search',query_string=query,page_num=1))
        parsed=urlsplit(url)
        self.assertEqual(parsed.fragment,'')
        encoded=parsed.path.split('/')[3]
        self.assertEqual(unquote(unquote(encoded)),query)
        self.assertNotIn('/',unquote(encoded))

    def test_native_step_gets_original_action_and_real_receipt(self):
        http=FakeHTTP();action='search[1/2 inch #red]'
        with tempfile.TemporaryDirectory() as folder,patch('requests.Session',return_value=http):
            env=RecordedEnvironment(NativeFixture,'http://127.0.0.1:3001',7,Path(folder),SimpleNamespace(trial=0))
            self.assertEqual(env.step('fixed_0',action),('REAL_RECEIPT',0,False))
            self.assertEqual(env.raw.sessions['fixed_7']['query_string'],'1/2 inch #red')
            self.assertIn('/fixed_7/',http.urls[0])
            self.assertEqual(len(http.urls),1)


if __name__=='__main__':unittest.main()
