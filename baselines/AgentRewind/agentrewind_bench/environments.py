import json
from pathlib import Path
import subprocess


class EnvironmentProcess:
    """Isolate native environment dependencies and keep private checkpoints out of model input."""
    def __init__(self,python,config,output):
        self.log = (Path(output)/'environment_stderr.log').open('x',encoding='utf-8')
        self.process = subprocess.Popen([str(python),'-u',str(Path(__file__).with_name('env_worker.py'))],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,encoding='utf-8',bufsize=1)
        self.process.stdin.write(json.dumps(config,ensure_ascii=False)+'\n')
        self.process.stdin.flush()
        self.closed = False

    def call(self,method,*arguments):
        self.process.stdin.write(json.dumps(dict(method=method,arguments=arguments),ensure_ascii=False)+'\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('native environment process stopped; inspect environment_stderr.log')
        value = json.loads(line)
        if 'error' in value:
            raise RuntimeError(json.dumps(value,ensure_ascii=False))
        if set(value) != {'result'}:
            raise ValueError('invalid environment RPC response')
        return value['result']

    def reset(self): return self.call('reset')
    def observe(self): return self.call('observe')
    def execute(self,action): return self.call('execute',action)
    def checkpoint(self): return self.call('checkpoint')
    def restore(self,key): return self.call('restore',key)
    def set_window(self,actions): return self.call('set_window',actions)
    def initial_prompt(self): return self.call('initial_prompt')
    def accounting(self): return self.call('accounting')

    def close(self):
        if self.closed:
            return
        try:
            if self.process.poll() is None:
                self.call('close')
                self.process.stdin.close()
                if self.process.wait(timeout=30) != 0:
                    raise RuntimeError('native environment cleanup failed')
        finally:
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=30)
            self.process.stdout.close()
            if not self.process.stdin.closed:
                self.process.stdin.close()
            self.log.close()
            self.closed = True
