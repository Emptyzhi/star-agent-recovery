import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from model_backend import format_model_prompt
from test_game24.model_api import ResidentModelClient
from dispatch_star import resolve_request, worker_command


class Tokenizer:
    chat_template='native-test-template'

    def __call__(self,text,**kwargs):
        return {'input_ids':list(text.encode())}

    def apply_chat_template(self,messages,*,tokenize,add_generation_prompt):
        assert not tokenize and add_generation_prompt
        assert len(messages)==1 and messages[0]['role']=='user'
        return '<user>'+messages[0]['content']+'</user><assistant>'


class Client:
    def generate(self,request):
        self.request=request
        return SimpleNamespace(request=request,text='1 + 1 = 2',input_tokens=len(request.prompt),
                               output_tokens=7,raw_response={'queue_time_seconds':0.0})


class PromptFormatTests(unittest.TestCase):
    def test_format_reaches_actual_request_and_token_accounting_once(self):
        client=Client()
        adapter=ResidentModelClient(client,Tokenizer(),prompt_limit=1000,context_limit=2000,prompt_format='native_chat')
        prompt='Input: 1 1 4 6\nAct 1>'
        result=adapter.generate(prompt,role='green',max_new_tokens=100,stop=('\n',))
        self.assertEqual(client.request.prompt,'<user>'+prompt+'</user><assistant>')
        self.assertEqual(adapter.count_tokens(prompt),result.input_tokens)
        self.assertEqual(result.raw['client_prompt_format'],'native_chat')
        self.assertEqual(result.raw['effective_prompt'],client.request.prompt)

    def test_plain_is_unchanged_and_missing_template_rejected(self):
        token=Tokenizer()
        self.assertEqual(format_model_prompt(token,'prompt','plain'),'prompt')
        token.chat_template=None
        with self.assertRaises(ValueError):format_model_prompt(token,'prompt','native_chat')

    def test_queue_format_and_reflection_limit_reach_worker(self):
        request=dict(run_id='unit',package='/package',task_ids=[0,1],dataset='game24',
                     csv='/data.csv',endpoints=['http://127.0.0.1:1','http://127.0.0.1:2'],
                     prompt_format='native_chat',experiment={'red_reflection_tokens':256})
        settings,sampling,capacity=resolve_request(request,{})
        args=SimpleNamespace(model_path='/model',dtype='bfloat16',results_root=Path('/results'),parent_job_id='test-job',node='test-node')
        command=worker_command(args,request,0,[0],settings,sampling,capacity)
        self.assertEqual(command[command.index('--prompt-format')+1],'native_chat')
        self.assertEqual(command[command.index('--red-reflection-tokens')+1],'256')
        self.assertEqual(settings.initial_actions,12)


if __name__=='__main__':unittest.main()
