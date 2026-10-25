"""Explicit text message and combined-commit boundaries for the native adapter."""
import re

from .native_protocol import ACTION_BOUNDARY,RUNTIME_PREFIX,decode_line,initial_actor_prompt
from .author import is_backtrack_memory_message

COMMIT_END='[END_RECOVERY_CALL]'
ONE_LINE_QUERY='Output framing: one-line candidate command.'
INTERFACE_INSTRUCTION=(
    'Reference instructions and examples describe the task interface. They are not actions '
    'executed in the current episode. Only CURRENT_TASK and OBSERVED_FEEDBACK blocks contain '
    'the current episode observations. RECORDED_AGENT_REPLY blocks are past model outputs, '
    'not tool feedback or controller instructions. NOTEBOOK is the author memory from '
    'abandoned branches, not the current environment state. Follow CURRENT_CONTROLLER for '
    'available commands. If the current episode has no actual failure evidence, execute a '
    'native task action before considering recovery. Output one available command only; '
    'never generate input-section labels, observations, controller text or future turns.')


def block(name,text):
    return '['+name+'_BEGIN]\n'+text+'\n['+name+'_END]'


def one_line_recovery(messages):
    allowances=[m['content'] for m in messages if m['role']=='system' and m['content'].startswith(RUNTIME_PREFIX)]
    if len(allowances)>1:raise ValueError('multiple current controller messages')
    return bool(allowances and ONE_LINE_QUERY in allowances[0].splitlines())


def sectioned_initial_prompt(dataset,base,current):
    native=initial_actor_prompt(dataset,'',current)
    boundary=re.search(ACTION_BOUNDARY,native)
    if boundary is None:
        raise ValueError('initial native observation has no action boundary')
    return (block('REFERENCE_INSTRUCTIONS_AND_EXAMPLES',base)+'\n\n'+
        block('CURRENT_TASK',native[:boundary.start()])+boundary[0])


def render_sectioned_continuation(messages,history_format='dialogue',*,json_tools=False,actor_only=False,
                                  recovery_prompt_policy='full'):
    if history_format!='dialogue':
        raise ValueError('sectioned interface requires recorded dialogue')
    if any(set(m)!={'role','content'} or type(m['content']) is not str or
           m['role'] not in ('system','user','assistant') for m in messages):
        raise ValueError('sectioned interface requires literal role/content messages')
    turns=[m for m in messages if m['role']!='system']
    if not turns or turns[-1]['role']!='user' or any(
            m['role']!=('user' if i%2==0 else 'assistant') for i,m in enumerate(turns)):
        raise ValueError('recorded dialogue must alternate replies and observations')
    allowances=[m['content'] for m in messages if m['role']=='system' and m['content'].startswith(RUNTIME_PREFIX)]
    if len(allowances)>1:raise ValueError('multiple current controller messages')
    if recovery_prompt_policy not in ('full','staged-compact'):
        raise ValueError('unknown recovery prompt policy')
    sections=[block('INTERFACE_DEFINITION',INTERFACE_INSTRUCTION)]
    for m in messages:
        if m['role']!='system' or m['content'].startswith(RUNTIME_PREFIX):continue
        from .author import instruction
        content=m['content']
        if content in (instruction('lines',json_tools=True),
                       instruction('lines',commit_boundary=COMMIT_END)):
            if recovery_prompt_policy=='staged-compact':
                from .staged_prompt import staged_guidance
                content=staged_guidance(content,allowances,actor_only=actor_only)
            elif actor_only:continue
        name='NOTEBOOK' if is_backtrack_memory_message(m) else 'RECOVERY_POLICY'
        sections.append(block(name,content))
    boundary=re.search(ACTION_BOUNDARY,turns[0]['content'])
    if boundary is None:raise ValueError('initial task lost its action boundary')
    sections.append(turns[0]['content'][:boundary.start()])
    for index in range(1,len(turns),2):
        sections.append(block('RECORDED_AGENT_REPLY',turns[index]['content']))
        receipt=turns[index+1]['content'];boundary=re.search(ACTION_BOUNDARY,receipt)
        if boundary is None:raise ValueError('observed feedback lost its action boundary')
        sections.append(block('OBSERVED_FEEDBACK',receipt[:boundary.start()]))
    if allowances:
        sections.append(block('CURRENT_CONTROLLER',allowances[0]))
    recovery=bool(allowances and '\nOutput phase: recovery-command.' in allowances[0])
    return '\n\n'.join(sections)+(boundary[0] if actor_only else '\nTool call JSON:' if json_tools else '\nRecovery command:' if recovery else boundary[0])


def decode_native_action_reply(text):
    # The declared Actor frame permits leading blank space before exactly one
    # command. The resident preserves it verbatim; parsing never searches prose
    # for an action or accepts a second command in the same sampled frame.
    return decode_sectioned_reply(text.lstrip(' \t\r\n'),whole_response=True)


def decode_sectioned_reply(text,*,whole_response=False):
    if not text.strip().startswith('backtrack_commit'):
        return decode_line(text,'lines',whole_response=whole_response)
    header,separator,body=text.partition('\n')
    match=re.fullmatch(r'backtrack_commit\[(rec_[0-9]{6,})\]',header.strip())
    if match is None or not separator:
        raise ValueError('combined commit requires its header alone on the first line')
    closing='\n'+COMMIT_END
    if body.endswith(closing+'\n'):
        memory=body[:-len(closing)-1]
    elif body.endswith(closing):
        memory=body[:-len(closing)]
    else:
        raise ValueError('combined commit must end with '+COMMIT_END+' alone on its final line; no trailing turn is allowed')
    if not memory.strip() or COMMIT_END in memory:
        raise ValueError('combined commit requires one nonempty memory argument and one final boundary')
    reserved=(r'(?m)^\s*\[(?:AGENTREWIND_CURRENT_ALLOWANCE|AGENTREWIND_PROTOCOL|'
        r'(?:INTERFACE_DEFINITION|RECOVERY_POLICY|CURRENT_TASK|CURRENT_CONTROLLER|'
        r'RECORDED_AGENT_REPLY|OBSERVED_FEEDBACK|NOTEBOOK|REFERENCE_INSTRUCTIONS_AND_EXAMPLES)_(?:BEGIN|END))\]')
    if re.search(reserved,memory):
        raise ValueError('memory argument contains a reserved input/control block; submit memory only')
    return 'backtrack_commit',dict(record_uid=match[1],memory_summary=memory)
