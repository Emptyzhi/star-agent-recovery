"""Render recorded text commands in the benchmark's native transcript syntax."""
from copy import deepcopy
import re

from . import author
from .framing import COMMIT_END
from .native_protocol import ACTION_BOUNDARY,RUNTIME_PREFIX,render_plain_continuation
from .staged_prompt import MEMORY_ORDER,recovery_phase,staged_guidance


PAUSED_ACTION_CUE = ('Recovery is paused. Return one native environment action using '
    'the demonstrated syntax, not backtrack_candidates or backtrack_commit.')

COMMIT_LAYOUT = ('If you choose backtrack_commit, return only the following layout. '
    'Replace angle-bracket placeholders with your selected returned ID and concise '
    'field values based on actual evidence. Do not repeat fields, simulate the '
    'restored run, or append a tool result. End immediately after the final delimiter. '
    'Other available commands retain their advertised one-line format.\n'
    'backtrack_commit[<returned_record_uid>]\n'+
    '\n'.join(field+': <value>' for field in MEMORY_ORDER)+'\n'+COMMIT_END)


def render_native_history(messages,recovery_prompt_policy):
    dialogue=[deepcopy(m) for m in messages if m['role']!='system']
    systems=[m for m in messages if m['role']=='system']
    allowances=[m['content'] for m in systems if m['content'].startswith(RUNTIME_PREFIX)]
    if len(allowances)!=1 or not dialogue:
        raise ValueError('native history requires recorded dialogue and one current controller')
    first=dialogue[0]['content']
    boundary=re.search(ACTION_BOUNDARY,first)
    if boundary is None:raise ValueError('initial task lost its native action cue')
    pattern=(re.escape('[REFERENCE_INSTRUCTIONS_AND_EXAMPLES_BEGIN]\n')+'(.*?)'+
        re.escape('\n[REFERENCE_INSTRUCTIONS_AND_EXAMPLES_END]\n\n[CURRENT_TASK_BEGIN]\n')+
        '(.*?)'+re.escape('\n[CURRENT_TASK_END]')+'(.*)')
    match=re.fullmatch(pattern,first[:boundary.start()],re.DOTALL)
    if match is None:raise ValueError('initial task does not match the recorded sectioned input')
    suffix=match[3]
    if suffix and not suffix.startswith('\n\n[AGENTREWIND_TOOL_FEEDBACK]\n'):
        raise ValueError('unknown content after the recorded initial task')
    dialogue[0]['content']=match[1]+match[2]+suffix+boundary[0]
    instructions=[]
    original=author.instruction('lines',commit_boundary=COMMIT_END)
    for message in systems:
        content=message['content']
        if content.startswith(RUNTIME_PREFIX):continue
        if content==original and recovery_prompt_policy=='staged-compact':
            content=staged_guidance(content,allowances,actor_only=False)
        instructions.append(content)
    if recovery_phase(allowances):
        systems=[dict(role='system',content=text) for text in [*instructions,allowances[0]]]
        text=render_plain_continuation([*systems,*dialogue],'dialogue')
        if '\n- backtrack_commit[<record_uid>]\n' in allowances[0]:
            cue='\nRecovery command:'
            if not text.endswith(cue):raise ValueError('recovery lost its command cue')
            text=text[:-len(cue)]+'\n\n'+COMMIT_LAYOUT+cue
        return text
    # Ordinary actions continue the demonstrated native transcript. Repeating the
    # whole controller between the observation and its action disrupts that format.
    transcript=render_plain_continuation(dialogue,'dialogue')
    if '\nRecovery is paused after two decisions' in allowances[0]:
        boundary=re.search(ACTION_BOUNDARY,transcript)
        if boundary is None:raise ValueError('paused Actor lost its native action cue')
        transcript=transcript[:boundary.start()]+'\n\n'+PAUSED_ACTION_CUE+boundary[0]
    # Supply the initial action separator. Later decisions retain their native
    # continuation cue, including the cue used to submit a complete answer.
    if transcript.endswith('\nAct 1>'):
        transcript+=' '
    return '\n\n'.join([*instructions,allowances[0],transcript])
