"""Strict single-call transport; native actions and author arguments stay literal."""
import json

ACTOR_ONLY='Output tool set: execute_action only.'


def actor_only_request(messages):
    from .native_protocol import RUNTIME_PREFIX
    controllers=[m['content'] for m in messages if m['role']=='system' and m['content'].startswith(RUNTIME_PREFIX)]
    if len(controllers)>1:
        raise ValueError('multiple current controller messages')
    return bool(controllers and ACTOR_ONLY in controllers[0].splitlines())


def declared_tool_name(text):
    """Read a complete top-level name field, never repair or execute the body.

    An incomplete argument string still represents an attempted recovery when
    its literal tool name was already emitted. This preserves the pre-JSON
    no-progress accounting for malformed recovery proposals.
    """
    decoder=json.JSONDecoder()
    position=len(text)-len(text.lstrip())
    if text[position:position+1]!='{':return None
    position+=1
    while True:
        position+=len(text[position:])-len(text[position:].lstrip())
        try:key,position=decoder.raw_decode(text,position)
        except json.JSONDecodeError:return None
        if type(key) is not str:return None
        position+=len(text[position:])-len(text[position:].lstrip())
        if text[position:position+1]!=':':return None
        position+=1
        position+=len(text[position:])-len(text[position:].lstrip())
        try:value,position=decoder.raw_decode(text,position)
        except json.JSONDecodeError:return None
        if key=='name':return value if type(value) is str else None
        position+=len(text[position:])-len(text[position:].lstrip())
        if text[position:position+1]!=',':return None
        position+=1


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate tool JSON key: '+key)
        result[key] = value
    return result


def decode_tool_reply(text):
    value = json.loads(text, object_pairs_hook=unique_object)
    if type(value) is not dict or set(value) != {'name', 'arguments'}:
        raise ValueError('return one JSON object with name and arguments only')
    name, arguments = value['name'], value['arguments']
    from .author import tools
    schemas = {tool['name']:tool['parameters'] for tool in tools()}
    if type(name) is not str or name not in schemas:
        raise ValueError('unknown tool name')
    schema=schemas[name]
    if (type(arguments) is not dict or not set(schema['required'])<=set(arguments)
            or set(arguments)-set(schema['properties'])):
        raise ValueError('tool arguments must exactly match its declared schema')
    if any(type(v) is not str for v in arguments.values()):
        raise ValueError('tool arguments must be literal strings')
    if name == 'backtrack_commit':
        return name, arguments
    return name, arguments['action' if name == 'execute_action' else 'reason']


def decode_interface_reply(text,messages):
    if actor_only_request(messages):
        from .native_protocol import decode_line
        return decode_line(text,'lines',whole_response=True)
    return decode_tool_reply(text)


def tool_instruction(tools):
    return ('Output one JSON tool call, with exactly the keys "name" and "arguments". '
        'Native action syntax belongs in the action string, unchanged. '
        'For recovery submit record_uid and memory_summary together. JSON string escapes '
        'represent literal memory text; do not include Markdown fences or future turns. '
        'Only these tools are available at this decision:\n'+json.dumps(tools, ensure_ascii=False))


def json_runtime_instruction(**kwargs):
    from .runtime_prompt import compact_runtime_instruction
    from .native_protocol import RUNTIME_PREFIX
    # Retain state/allowance semantics; replace only the line-oriented transport.
    from .runtime_prompt import compact_command_instruction
    tools = kwargs['tools']
    text = compact_runtime_instruction(**kwargs)
    old = compact_command_instruction(tools, kwargs['output_limit'])
    if text.count(old) != 1 or not text.startswith(RUNTIME_PREFIX):
        raise ValueError('runtime transport mapping changed unexpectedly')
    native_only={tool['name'] for tool in tools}=={'execute_action'}
    instruction=('Output one native environment action using the demonstrated syntax. '
        'No JSON wrapper, recovery command, reflection, or simulated feedback is allowed.'
        if native_only else tool_instruction(tools))
    text=text.replace(old,instruction)
    if native_only:
        text+='\n'+ACTOR_ONLY
    return text
