"""A Yellow command ID selects one literal action from its bound public menu."""
import json

from icr_core import ProtocolError
from scoped_prompt_efficiency import YellowChoiceError


class YellowActionIdRangeError(YellowChoiceError):
    """A model-selected ID has no command in this unchanged request's menu."""


def command_choices(commands):
    if (not isinstance(commands,(list,tuple)) or not commands
            or any(type(c) is not str or not c or '\n' in c or '\r' in c for c in commands)
            or len(set(commands))!=len(commands)):
        raise ProtocolError('Yellow action IDs require a nonempty distinct native menu')
    return [{'action_id':i,'command':c} for i,c in enumerate(commands)]


def parse_selection(text,commands):
    command_choices(commands)
    def fields(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise YellowChoiceError('Yellow choice contains duplicate JSON fields')
            result[key]=value
        return result
    if type(text) is not str:raise YellowChoiceError('Yellow choice must be JSON text')
    try:value=json.loads(text,object_pairs_hook=fields)
    except json.JSONDecodeError as error:
        raise YellowChoiceError(f'Yellow choice requires one JSON object: {error.msg} at line {error.lineno}, '
                                f'column {error.colno}. JSON string values require double quotes.') from error
    if type(value) is not dict or set(value)!={'role','reason','action_id'}:
        raise YellowChoiceError('Yellow requires exactly role, reason and action_id; do not write an action string')
    if type(value['role']) is not str or value['role'] not in ('PREPARE','CORRECT'):
        raise YellowChoiceError('Yellow role must be exactly PREPARE or CORRECT')
    if type(value['reason']) is not str or not value['reason'].strip():
        raise YellowChoiceError('Yellow choice requires a nonempty reason')
    index=value['action_id']
    if type(index) is not int:
        raise YellowChoiceError('action_id must be an integer from the current menu, not text or a boolean')
    if not 0<=index<len(commands):
        raise YellowActionIdRangeError(f'action_id {index} is outside the current menu 0..{len(commands)-1}')
    return {'diagnosis':value['reason'],'step_role':value['role'],'current_action':commands[index]}
