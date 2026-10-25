"""Overflow-only, deterministic head/tail projection of literal recorded history."""
from copy import deepcopy
import json

from .candidate_context import candidate_receipts


MARKER = '[AGENTREWIND_CONTEXT_WINDOW]\n'
POLICY = 'head_tail_on_overflow_v1'


class ContextWindowTooSmall(ValueError):
    pass


def _excerpt(value, limit):
    if len(value) <= limit:
        return value
    half = limit//2
    return value[:half]+'\n[... context preview middle omitted ...]\n'+value[-(limit-half):]


def _previews(value, limit, *, preview=False):
    if isinstance(value, str):
        return _excerpt(value, limit) if preview else value
    if isinstance(value, list):
        return [_previews(item, limit, preview=preview) for item in value]
    if isinstance(value, dict):
        return {key:_previews(item,limit,preview=(False if key in
                    ('record_uid','record_kind','created_at','source_mode','tool_name') else
                    preview or key.endswith('_preview') or key=='decision_hint'))
                for key,item in value.items()}
    return value


def _view(messages, retained, preview_limit, removed):
    view = deepcopy(messages)
    # Parse controller JSON receipts only; model-generated command text is literal.
    if preview_limit is not None:
        for receipt in reversed(candidate_receipts(view)):
            value = deepcopy(receipt['value'])
            assert isinstance(value['result'], list)
            for index, candidate in enumerate(value['result']):
                uid = candidate['record_uid']
                value['result'][index] = _previews(candidate,preview_limit)
                assert value['result'][index]['record_uid'] == uid
            index = receipt['message']
            text = view[index]['content']
            view[index]['content'] = (text[:receipt['start']]+json.dumps(value,ensure_ascii=False)+
                                      text[receipt['end']:])
    view = [message for index,message in enumerate(view) if index in retained]
    first_user = next(index for index,message in enumerate(view) if message['role']=='user')
    notice = (MARKER+'This request uses a deterministic context window. The original task, '
              'current controller, notebook, latest observation and latest candidate IDs remain. '
              'Older middle dialogue turns are omitted; preview strings may retain only their '
              'literal beginning and end, explicitly marked. Omitted text is not an observation '
              'or a successful action. Full records and checkpoint history remain in storage.\n'+
              json.dumps(dict(removed_message_indices=removed,candidate_preview_characters=preview_limit)))
    view.insert(first_user,dict(role='system',content=notice))
    return view


def fit_context(messages, render, count_tokens, budget, original_tokens):
    """Keep root + recent complete turns; shorten only candidate previews if needed."""
    assert original_tokens > budget and budget > 0
    dialogue = [index for index,message in enumerate(messages) if message['role']!='system']
    assert dialogue and len(dialogue)%2 == 1
    assert all(messages[index]['role']==('user' if pos%2==0 else 'assistant')
               for pos,index in enumerate(dialogue))
    pairs = [dialogue[pos:pos+2] for pos in range(1,len(dialogue),2)]
    receipts = candidate_receipts(messages)
    latest = receipts[-1]['message'] if receipts else None
    protected = {index for index,message in enumerate(messages) if message['role']=='system'}
    protected.add(dialogue[0])
    protected.add(dialogue[-1])
    if len(dialogue)>1:
        protected.add(dialogue[-2])
    if latest is not None:
        protected.add(latest)
        position = dialogue.index(latest)
        if position:
            protected.add(dialogue[position-1])
    for recent in (4,2,1,0):
        retained = protected.union(index for pair in pairs[-recent:] for index in pair) if recent else set(protected)
        removed = [index for index in dialogue if index not in retained]
        view = _view(messages,retained,None,removed)
        rendered = render(view)
        length = count_tokens(rendered[0])
        if length <= budget:
            return rendered, dict(policy=POLICY,before_tokens=original_tokens,after_tokens=length,
                token_budget=budget,removed_message_indices=removed,recent_turn_pairs=recent,
                candidate_preview_characters=None,model_calls=0)
    # Candidate IDs and ordering stay exact; only oversized display previews change.
    for preview_limit in (512,256,128,64):
        view = _view(messages,retained,preview_limit,removed)
        rendered = render(view)
        length = count_tokens(rendered[0])
        if length <= budget:
            return rendered, dict(policy=POLICY,before_tokens=original_tokens,after_tokens=length,
                token_budget=budget,removed_message_indices=removed,recent_turn_pairs=0,
                candidate_preview_characters=preview_limit,model_calls=0)
    raise ContextWindowTooSmall('Protected task, controller, notebook and latest observation cannot fit the context budget')
