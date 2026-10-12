"""Literal public-task evidence for event selection, never action filtering."""
import re
import json
from pathlib import Path
from icr_core import ProtocolError

ENTITY=r'[a-z][a-z0-9]*(?: [1-9][0-9]*)?'
TARGET_ARTICLE=r'(?:(?P<target_determiner>a|an|some|the|two) )?'
TOOL_ARTICLE=r'(?:(?P<tool_determiner>a|an|some|the|two) )?'
DEST_ARTICLE=r'(?:(?P<destination_determiner>a|an|some|the|two) )?'
PROCESS={'hot':'heat','cold':'cool','clean':'clean','sliced':'slice'}
PUBLIC_ENTITY_CATALOG=json.loads((Path(__file__).with_name('alf_public_entity_names.json')).read_text(encoding='utf-8'))
PUBLIC_ENTITY_NAMES=frozenset(PUBLIC_ENTITY_CATALOG['names'])


def task_evidence(initial_observation):
    lines=[line for line in initial_observation.splitlines() if line.startswith('Your task is to: ')]
    if len(lines)!=1:
        raise ProtocolError('goal evidence requires exactly one literal public task line')
    line=lines[0]
    task=line[len('Your task is to: '):]
    patterns=(
        ('examine',rf'examine {TARGET_ARTICLE}(?P<object>{ENTITY}) with {TOOL_ARTICLE}(?P<tool>{ENTITY})\.'),
        ('place',rf'put {TARGET_ARTICLE}(?:(?P<condition>hot|cold|clean|sliced) )?(?P<object>{ENTITY}) (?:in|on) {DEST_ARTICLE}(?P<destination>{ENTITY})\.'),
        ('process_place',rf'(?P<process>heat|cool|clean|slice|find) {TARGET_ARTICLE}(?P<object>{ENTITY}) and put (?:it|them) (?:in|on) {DEST_ARTICLE}(?P<destination>{ENTITY})\.'),
    )
    for form,pattern in patterns:
        match=re.fullmatch(pattern,task)
        if match is None:
            continue
        fields=match.groupdict()
        entities=[fields[role] for role in ('object','tool','destination')
                  if fields.get(role) is not None]
        if any(entity.split(' ')[0] not in PUBLIC_ENTITY_NAMES for entity in entities):
            return dict(status='UNKNOWN_TASK_EXPRESSION',source_text=line,
                        reason='Entity name is not in the declared public ALFWorld object vocabulary; no referent or singular form inferred.')
        condition=fields.get('condition')
        process=fields.get('process')
        operations=['take','put'] if form!='examine' else ['take','examine']
        if condition is not None:operations.append(PROCESS[condition])
        if process not in (None,'find'):operations.append(process)
        return dict(status='EXPLICIT_PUBLIC_TASK',source_text=line,form=form,
            entity_vocabulary=PUBLIC_ENTITY_CATALOG['source'],
            target=fields['object'],tool=fields.get('tool'),destination=fields.get('destination'),
            condition=condition,required_processing=None if process=='find' else process,
            quantity='two' if fields['target_determiner']=='two' else 'UNSPECIFIED',
            role_determiners={role:fields.get(role+'_determiner') for role in ('target','tool','destination')},
            supported_target_operations=operations)
    # Unsupported language is an explicit epistemic state, not a guessed goal.
    return dict(status='UNKNOWN_TASK_EXPRESSION',source_text=line,
                reason='No exact supported public-task construction matched.')


def entity_matches(instance,mentioned):
    if mentioned is None:return False
    if instance==mentioned:return True
    return re.fullmatch(re.escape(mentioned)+r' [1-9][0-9]*',instance) is not None


def native_operation(action):
    for kind,pattern in (
        ('take',rf'take (?P<object>{ENTITY}) from (?P<other>{ENTITY})'),
        ('put',rf'put (?P<object>{ENTITY}) in/on (?P<other>{ENTITY})'),
        ('process',rf'(?P<verb>heat|cool|clean|slice) (?P<object>{ENTITY}) with (?P<other>{ENTITY})'),
        ('unary',rf'(?P<verb>use|open|close|examine) (?P<object>{ENTITY})'),
        ('navigate',rf'go to (?P<object>{ENTITY})'),
    ):
        match=re.fullmatch(pattern,action)
        if match:
            fields=match.groupdict()
            return dict(verb=fields.get('verb',kind),object=fields['object'],other=fields.get('other'))
    return None


def object_evidence(obj,goal,commands):
    if goal['status']!='EXPLICIT_PUBLIC_TASK':
        return dict(status='UNKNOWN',reason='Public task expression is unsupported.',source=goal['source_text'])
    if entity_matches(obj,goal['target']):
        return dict(status='TARGET',source=goal['source_text'],mentioned=goal['target'])
    for role in ('tool','destination'):
        if entity_matches(obj,goal[role]):
            return dict(status='SUPPORTED_PREREQUISITE',source=goal['source_text'],role=role,
                        semantics='Explicit task participant, not proof that acquiring it is necessary.')
    for command in commands:
        op=native_operation(command)
        if (op is not None and op['verb'] in goal['supported_target_operations']
                and entity_matches(op['object'],goal['target']) and op['other']==obj):
            if op['verb']=='put' and not entity_matches(op['other'],goal['destination']):
                continue
            return dict(status='SUPPORTED_PREREQUISITE',source=goal['source_text'],
                public_command=command,role='CURRENT_PUBLIC_TARGET_OPERATION_ARGUMENT',
                semantics='Candidate prerequisite supported by a current command, not proof of necessity.')
    return dict(status='UNKNOWN',source=goal['source_text'],
        reason='No direct target match or explicit/current-command prerequisite evidence. Not a claim of irrelevance.')


def action_evidence(action,goal,commands):
    op=native_operation(action)
    if op is None or goal['status']!='EXPLICIT_PUBLIC_TASK':
        return dict(status='UNKNOWN',source=goal['source_text'],reason='Unrecognized operation or task expression.')
    evidence=object_evidence(op['object'],goal,commands)
    if evidence['status']=='TARGET':
        if op['verb'] not in goal['supported_target_operations']:
            return dict(status='UNKNOWN',source=goal['source_text'],reason='Operation not supported by the expressed goal.')
        if op['verb']=='put' and not entity_matches(op['other'],goal['destination']):
            return dict(status='UNKNOWN',source=goal['source_text'],reason='Placement is not the expressed destination.')
        return evidence
    if evidence['status']=='SUPPORTED_PREREQUISITE':
        if op['verb'] in ('take','open','use'):
            return evidence
    return dict(status='UNKNOWN',source=goal['source_text'],reason='No supported role for this operation.')
