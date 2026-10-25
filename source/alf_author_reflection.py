"""Author reflection instruction and examples, applied to a real ETT branch."""
from pathlib import Path


AUTHOR_INSTRUCTION = '''You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need this later when you are solving the same task. Give your plan after "Plan". Here are two examples:'''


def reflection_query(scenario):
    examples = (Path(__file__).resolve().parents[1] / 'prompts' / 'alfworld' /
                'alfworld_reflexion_examples.txt').read_text(encoding='utf-8')
    return AUTHOR_INSTRUCTION + '\n\n' + examples + '\n\n' + scenario.strip() + '\n\nNew plan>'
