"""Explicit text transport for native actions and the author's two rewind tools."""
import re
import json
import ast
from copy import deepcopy


MEMORY_FIELDS = ('failed_check', 'observed_loop', 'why_local_repair_failed',
    'root_cause', 'evidence_observed', 'assumptions_to_recheck', 'rewind_target_reason',
    'selected_candidate_safety_check', 'do_not_repeat', 'corrected_strategy',
    'do_not_call_before', 'must_satisfy_before_mutation', 'mutation_preflight',
    'first_3_tool_calls', 'success_invariant')

RUNTIME_PREFIX = '[AGENTREWIND_CURRENT_ALLOWANCE]\n'
ACTION_BOUNDARY = r'(\n\nAction:|\n>|\nAct [1-9][0-9]*>)$'

COMMIT_FORM = 'backtrack_commit[{"record_uid":"<id>","memory_summary":"<recovery note>"}]'


def initial_actor_prompt(dataset, base, current):
    observation = current['observation']
    if dataset == 'webshop':
        return base + observation + '\n\nAction:'
    if dataset == 'alfworld':
        return base + '\nHere is the task:\n' + observation + '\n>'
    if dataset == 'game24':
        return base + '\nHere is the task:\n' + observation + '\nAct 1>'
    raise ValueError('unknown native Actor dataset')


def append_actor_step(dataset, prompt, action, observation):
    if dataset == 'webshop' and prompt.endswith('\n\nAction:'):
        return prompt + ' ' + action + '\nObservation: ' + observation + '\n\nAction:'
    if dataset == 'alfworld' and prompt.endswith('\n>'):
        return prompt + ' ' + action + '\n' + observation + '\n>'
    if dataset == 'game24':
        match = re.search(r'\nAct ([1-9][0-9]*)>$', prompt)
        if match is not None:
            turn = int(match[1])
            return prompt + ' ' + action + f'\nObs {turn}> ' + observation + f'\nAct {turn+1}>'
    raise ValueError('native Actor prompt has lost its action boundary')


def add_control_observation(messages, text):
    result = [dict(message) for message in messages]
    users = [message for message in result if message['role'] == 'user']
    if len(users) != 1:
        raise ValueError('tool feedback requires one complete task transcript')
    transcript = users[0]
    boundary = re.search(ACTION_BOUNDARY, transcript['content'])
    if boundary is None:
        raise ValueError('tool feedback requires a native action decision boundary')
    # Tool receipts belong after the latest observation, at the decision that
    # consumes them. Keep the literal native action cue for the next action.
    transcript['content'] = (transcript['content'][:boundary.start()] +
        '\n\n[AGENTREWIND_TOOL_FEEDBACK]\n' + text + boundary[0])
    return result


def append_dialogue_feedback(messages,raw_reply,text,dataset,next_action):
    result=deepcopy(messages)
    dialogue=[(i,m) for i,m in enumerate(result) if m['role']!='system']
    if not dialogue or dialogue[-1][1]['role']!='user':
        raise ValueError('a decision receipt must follow the current user observation')
    if raw_reply is None:
        # The terminal harness attaches its deterministic candidate listing to
        # the actual environment receipt; it does not invent an assistant call.
        index,last=dialogue[-1]
        patched=add_control_observation([last],text)
        result[index]=patched[0]
        return result
    if type(raw_reply) is not str:
        raise ValueError('dialogue must retain the literal model response')
    if dataset=='webshop':
        content=text+'\n\nAction:'
    elif dataset=='alfworld':
        content=text+'\n>'
    elif dataset=='game24':
        content=text+f'\nAct {next_action}>'
    else:
        raise ValueError('unknown native dialogue dataset')
    return result+[dict(role='assistant',content=raw_reply),dict(role='user',content=content)]


def without_runtime_allowance(messages):
    return [message for message in messages if not (
        message['role'] == 'system' and message['content'].startswith(RUNTIME_PREFIX))]


def render_native_messages(messages):
    # Logical messages retain the author's notebook API; resident Actors use the
    # same plain-text continuation transport as the benchmark implementations.
    if any(set(m) != {'role', 'content'} or type(m['content']) is not str for m in messages):
        raise ValueError('native text transport requires literal role/content messages')
    systems = [m['content'] for m in messages if m['role'] == 'system'
               and not m['content'].startswith(RUNTIME_PREFIX)]
    allowances = [m['content'] for m in messages if m['role'] == 'system'
                  and m['content'].startswith(RUNTIME_PREFIX)]
    users = [m['content'] for m in messages if m['role'] == 'user']
    if len(users) != 1 or any(m['role'] not in ('system', 'user') for m in messages):
        raise ValueError('native Actor transport requires one complete task transcript')
    if len(allowances)>1:
        raise ValueError('multiple current recovery allowances')
    continuation = users[0]
    if allowances:
        boundary = re.search(ACTION_BOUNDARY, continuation)
        if boundary is None:
            raise ValueError('allowance must accompany a native action decision boundary')
        if '\nOutput phase: recovery-command.' in allowances[0]:
            continuation = (continuation[:boundary.start()]+'\n\n'+
                            allowances[0]+'\nRecovery command:')
        else:
            continuation = (continuation[:boundary.start()]+'\n\n'+allowances[0]+
                            boundary[0])
    return '\n\n'.join([*systems, continuation])


def render_plain_continuation(messages, history_format='flat'):
    if history_format == 'flat':
        return render_native_messages(messages)
    if history_format != 'dialogue':
        raise ValueError('unknown plain-text history format')
    if any(set(m) != {'role','content'} or type(m['content']) is not str for m in messages):
        raise ValueError('plain continuation requires literal role/content records')
    if any(m['role'] not in ('system','user','assistant') for m in messages):
        raise ValueError('unknown role in recorded native history')
    turns = [m for m in messages if m['role'] != 'system']
    if not turns or turns[-1]['role'] != 'user' or any(
            m['role'] != ('user' if i % 2 == 0 else 'assistant') for i,m in enumerate(turns)):
        raise ValueError('recorded native history must alternate actual replies and receipts')
    transcript = turns[0]['content']
    for index in range(1, len(turns), 2):
        boundary = re.search(ACTION_BOUNDARY, transcript)
        if boundary is None:
            raise ValueError('recorded observation has lost its native action boundary')
        cue = boundary[0]
        if cue == '\n\nAction:':
            receipt_prefix = '\nObservation: '
        elif cue == '\n>':
            receipt_prefix = '\n'
        else:
            step = re.fullmatch(r'\nAct ([1-9][0-9]*)>', cue)
            if step is None:
                raise ValueError('unknown native action boundary')
            receipt_prefix = '\nObs '+step[1]+'> '
        transcript += ' '+turns[index]['content']+receipt_prefix+turns[index+1]['content']
    systems = [dict(m) for m in messages if m['role'] == 'system']
    return render_native_messages(systems+[dict(role='user',content=transcript)])


def available_command_instruction(tools,output_limit=1200,recovery_format='bracket-json'):
    forms = {'execute_action': 'one native action, using the demonstrated syntax',
             'backtrack_candidates': 'backtrack_candidates[reason]',
             'backtrack_commit': COMMIT_FORM}
    if recovery_format=='python':
        forms.update(backtrack_candidates='backtrack_candidates(reason="<observed failure>")',
            backtrack_commit='backtrack_commit(record_uid="<id>", memory_summary=r"""<recovery note>""")')
    elif recovery_format=='lines':
        forms['backtrack_commit']='backtrack_commit[<record_uid>]\n<complete recovery note>'
    elif recovery_format!='bracket-json':
        raise ValueError('unknown recovery argument transport')
    names = [tool['name'] for tool in tools]
    if not names or len(set(names)) != len(names):
        raise ValueError('a decision requires a nonempty unique command set')
    instruction=('Available text commands for this decision only:\n' +
            '\n'.join('- '+forms[name] for name in names) +
            '\nReturn exactly one tool call per response. Never combine candidate listing '
            'and commit in one response. Do not simulate a tool result or a later turn.')
    if 'backtrack_candidates' in names:
        instruction+=(' For a candidate-list request, reason is a nonempty string '
            +('inside brackets.' if recovery_format in ('bracket-json','lines') else 'argument.')+
            ' End the response immediately after this one candidate command. '
            'Wait for the real candidate result before choosing an ID or writing a commit. '
            'Serialization example only (not task evidence):\n'+
            forms['backtrack_candidates'].replace('[reason]','[observed failure]')+'\n'+
            'The example above is one complete response: no repetition, commit, or reflection follows it.')
    if 'backtrack_commit' in names:
        if recovery_format=='lines':
            instruction+=(' To commit, use this exact two-part layout:\n'
                'backtrack_commit[<record_uid>]\n<complete recovery note>\n'
                'Replace <record_uid> with the exact top-level record_uid from the latest '
                'real candidate result. Do not use a list position such as 1. '
                'Submit both arguments in this single response. The first line '
                'contains only backtrack_commit[<record_uid>]; every character '
                'after its newline is the memory_summary argument, passed verbatim. '
                'A real newline after the closing bracket is required, not a space or literal \\n. '
                'Use the selected returned ID, not the example ID. Do not wrap the note '
                'in JSON, Python quotes or Markdown fences. Follow the unchanged author '
                'reflection guidance. The optional author reason argument is omitted by '
                'this text transport. Write the required memory fields concisely; '
                'do not append another tool call or an imagined observation.'+
                f'\nThe response output limit is {output_limit} tokens, including the first line.')
        elif recovery_format=='python':
            instruction+=(' Use the exact Python function-call syntax shown above, with named '
                'string arguments record_uid and memory_summary together; reason is optional. '
                'Use a raw triple-quoted string for memory_summary so its contents do not require '
                'an extra layer of JSON string escaping. No Markdown fences, assignments, '
                'imports, helper expressions or executable code are supported. '
                'Follow the unchanged author reflection guidance. Author argument schema: '+
                json.dumps(next(t['parameters'] for t in tools if t['name']=='backtrack_commit'))+
                f'\nThe response output limit is {output_limit} tokens, including the tool-call wrapper.')
        else:
            instruction+=(' Commit takes one JSON object containing record_uid and '
            'memory_summary together, with optional reason, exactly as the author tool '
            'schema below. Follow the unchanged author reflection guidance. JSON may span lines. '
            'Do not put raw newlines inside JSON strings. Tool schema: '+
            json.dumps(next(t['parameters'] for t in tools if t['name']=='backtrack_commit'))+
            f'\nThe response output limit is {output_limit} tokens, including the tool-call wrapper.')
    return instruction


def decision_chat_messages(messages,history_format='flat'):
    allowances=[m for m in messages if m['role']=='system'
                and m['content'].startswith(RUNTIME_PREFIX)]
    if len(allowances)!=1:
        raise ValueError('decision chat requires exactly one current allowance')
    if history_format=='dialogue':
        result=deepcopy(without_runtime_allowance(messages))
        dialogue=[(i,m) for i,m in enumerate(result) if m['role']!='system']
        if not dialogue or dialogue[-1][1]['role']!='user':
            raise ValueError('native dialogue must end at an observed decision boundary')
        if any(m['role']!=('user' if i%2==0 else 'assistant') for i,(_,m) in enumerate(dialogue)):
            raise ValueError('native dialogue must alternate actual model responses and observations')
        index,last=dialogue[-1]
        result[index]=dict(role='user',content=render_native_messages([allowances[0],last]))
        return result
    if history_format!='flat':
        raise ValueError('unknown history format')
    result=[]
    for message in without_runtime_allowance(messages):
        if message['role']=='user':
            result.append(dict(role='user',content=render_native_messages([allowances[0],message])))
        else:
            result.append(dict(message))
    return result


def recovery_command_messages(messages,history_format='flat'):
    allowances=[m for m in messages if m['role']=='system' and m['content'].startswith(RUNTIME_PREFIX)]
    if len(allowances)!=1 or '\nOutput phase: recovery-command.' not in allowances[0]['content']:
        raise ValueError('recovery command chat requires exactly one current command phase')
    return decision_chat_messages(messages,history_format)


def unavailable_command_error(name, *, recoveries, blocked, candidates,recovery_format='bracket-json'):
    text=_unavailable_command_error(name,recoveries=recoveries,blocked=blocked,candidates=candidates)
    if recovery_format=='python':
        return text.replace('backtrack_candidates[reason]','backtrack_candidates(reason="<observed failure>")')
    if recovery_format not in ('bracket-json','lines'):
        raise ValueError('unknown recovery argument transport')
    return text


def _unavailable_command_error(name, *, recoveries, blocked, candidates):
    if candidates:
        return ('Candidate selection is pending. The candidate list is already returned. '
                'Only backtrack_commit with record_uid and memory_summary is available; select a top-level '
                'record_uid from that list. Do not call backtrack_candidates again '
                'or execute a native action before committing.')
    if recoveries == 0:
        return ('Recovery allowance is exhausted. Neither backtrack_candidates nor '
                'backtrack_commit is available. Return one native action from the '
                'current observation.')
    if blocked:
        return ('The native attempt has ended. Only backtrack_candidates[reason] '
                'is available before selecting a checkpoint; ordinary actions and '
                'backtrack_commit are not available yet.')
    return ('No candidate list is pending. '+name+' is not available. Return one '
            'native action or backtrack_candidates[reason] to request checkpoints first.')


def boundary_instruction(*, remaining, recoveries, blocked, candidates,recovery_format='bracket-json'):
    text=_boundary_instruction(remaining=remaining,recoveries=recoveries,blocked=blocked,candidates=candidates)
    if recovery_format=='python':
        return text.replace('backtrack_candidates[reason]','backtrack_candidates(reason="<observed failure>")')
    if recovery_format not in ('bracket-json','lines'):
        raise ValueError('unknown recovery argument transport')
    return text


def _boundary_instruction(*, remaining, recoveries, blocked, candidates):
    allowance = 'unlimited' if recoveries is None else str(recoveries)
    budget = f'Native actions remaining: {remaining}. Recoveries remaining: {allowance}. '
    if candidates:
        return (budget + 'A checkpoint list has been returned. Select one of its top-level '
                'record_uid values and submit it with memory_summary in one '
                'backtrack_commit call. Follow the author checkpoint and memory guidance. '
                'Do not list candidates again.')
    if recoveries == 0:
        return (budget + 'Recovery is unavailable. Continue with one native task action '
                'from the current observation, using the retained recovery note if present. '
                'Do not call backtrack_candidates or backtrack_commit. Return only the '
                'action itself, without an Action: label.')
    recovery = ('To abandon a failed branch, output backtrack_candidates[reason] with '
                'the observed failure as reason. Then select a returned checkpoint. ')
    if blocked:
        return (budget + 'The current attempt has ended without full task success. '
                'done=true ends an attempt, not the recovery allowance. A positive reward '
                'below 1 is partial credit, and action_accepted is command validity only. '
                'No ordinary action can run in this phase. Return '
                'backtrack_candidates[reason] to inspect recovery checkpoints. End this '
                'response there. After the real tool result arrives, a later response '
                'can submit backtrack_commit with record_uid and memory_summary together. '
                'Return the tool command itself without prose. ')
    return (budget + 'Return one native task action, as in the demonstrations, without '
            'an Action: label. ' + recovery + 'Use recovery only after concrete failure '
            'feedback or a repeated failed attempt, not as an opening diagnostic.')


def decode_python_recovery(text):
    try:
        call=ast.parse(text.strip(),mode='eval').body
    except SyntaxError as error:
        raise ValueError('incomplete/invalid literal recovery call: '+error.msg) from error
    if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or call.args:
        raise ValueError('recovery must be one named function call with named string arguments')
    name=call.func.id
    required={'backtrack_candidates':{'reason'},'backtrack_commit':{'record_uid','memory_summary'}}
    if name not in required or any(k.arg is None for k in call.keywords):
        raise ValueError('unknown recovery call or argument unpacking')
    keys=[k.arg for k in call.keywords]
    allowed=required[name]|({'reason'} if name=='backtrack_commit' else set())
    if len(keys)!=len(set(keys)) or not required[name]<=set(keys) or set(keys)-allowed:
        raise ValueError('recovery argument fields must match the author signature')
    if any(not isinstance(k.value,ast.Constant) or type(k.value.value) is not str for k in call.keywords):
        raise ValueError('recovery arguments must be literal strings; expressions are not executed')
    arguments={k.arg:k.value.value for k in call.keywords}
    if name=='backtrack_candidates' and not arguments['reason'].strip():
        raise ValueError('candidate reason must be nonempty')
    return name,arguments if name=='backtrack_commit' else arguments['reason']


def decode_line(text,recovery_format='bracket-json',*,whole_response=False):
    if recovery_format=='lines' and text.strip().startswith('backtrack_commit'):
        header,separator,memory=text.partition('\n')
        match=re.fullmatch(r'backtrack_commit\[(rec_[0-9]{6,})\]',header.strip())
        if match is None or not separator or not memory.strip():
            raise ValueError('combined commit requires backtrack_commit[record_uid] on the first line and a nonempty memory_summary body')
        return 'backtrack_commit',dict(record_uid=match[1],memory_summary=memory)
    if recovery_format=='python' and text.strip().startswith(('backtrack_commit','backtrack_candidates')):
        return decode_python_recovery(text)
    if recovery_format not in ('bracket-json','python','lines'):
        raise ValueError('unknown recovery argument transport')
    if text.strip().startswith('backtrack_commit'):
        match = re.fullmatch(r'backtrack_commit\[(.*)\]', text.strip(), re.DOTALL)
        if match is None:
            raise ValueError('commit requires backtrack_commit[JSON object]')
        arguments = json.loads(match[1])
        if type(arguments) is not dict or not {'record_uid','memory_summary'} <= arguments.keys():
            raise ValueError('commit requires record_uid and memory_summary together')
        if arguments.keys()-{'record_uid','memory_summary','reason'}:
            raise ValueError('unknown commit argument')
        if any(type(arguments[k]) is not str for k in arguments):
            raise ValueError('commit arguments must be strings')
        return 'backtrack_commit', arguments
    # Native Actor framing stays first-line; recovery tools consume the full reply.
    complete = text.strip()
    line = complete if whole_response else text.split('\n', 1)[0].strip()
    for tool in ('backtrack_candidates',):
        if complete.startswith(tool):
            match = re.fullmatch(re.escape(tool) + r'\[([^\r\n]+)\]', complete)
            if match is None or not match[1].strip():
                raise ValueError('candidate response must contain exactly one complete '+tool+'[argument], without trailing text')
            argument = match[1]
            if argument.startswith('"'):
                argument = json.loads(argument)
                if type(argument) is not str or not argument:
                    raise ValueError('recovery command argument must be a nonempty string')
            else:
                depth = 0
                for character in argument:
                    depth += (character == '[') - (character == ']')
                    if depth < 0:
                        raise ValueError('candidate response contains text after its closing bracket')
                if depth:
                    raise ValueError('candidate reason has unbalanced brackets')
            return tool, argument
    if whole_response and ('\n' in line or '\r' in line):
        raise ValueError('a native action in recovery must be one complete line without trailing text')
    if not line:
        raise ValueError('empty action line')
    return 'execute_action', line
