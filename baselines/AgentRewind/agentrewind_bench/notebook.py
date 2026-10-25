"""Explicit notebook rendering variant; preserve the author's original override."""
from copy import deepcopy
import json

from . import author


def notebook_override(prepared, log_dir, *, item_char_limit, memory_limit):
    if type(item_char_limit) is not int or item_char_limit not in (900, 4000):
        raise ValueError('notebook item character limit must be 900 or 4000')
    config = dict(items=memory_limit, item_char_limit=item_char_limit, notebook_char_limit=8000)
    original_path = prepared['override_input_file']
    if item_char_limit == 900:
        return original_path, config
    if (prepared['backtrack_memory_policy'] != 'cumulative'
            or prepared['backtrack_context_policy'] != 'rollback'
            or prepared['backtrack_workspace_policy'] != 'rollback'
            or prepared['backtrack_memory_limit'] != memory_limit):
        raise ValueError('notebook variant requires the declared cumulative rollback configuration')
    original = json.loads(original_path.read_text(encoding='utf-8'))
    messages = original['messages']
    if type(messages) is not list:
        raise ValueError('author override messages must be a list')
    indices = [i for i, message in enumerate(messages) if author.is_backtrack_memory_message(message)]
    if len(indices) != 1:
        raise ValueError('author override must contain exactly one notebook system message')
    memory = author.load_backtrack_memory(log_dir, prepared['fork_run'])
    if not memory:
        raise ValueError('cumulative fork memory is empty after an accepted commit')
    render_args = dict(limit=memory_limit,
                       context_policy=prepared['backtrack_context_policy'],
                       workspace_policy=prepared['backtrack_workspace_policy'],
                       target_record_uid=prepared['breakpoint_record_uid'],
                       context_source_record_uid=prepared['backtrack_context_source_record_uid'],
                       notebook_char_limit=8000)
    original_notebook = author.render_backtrack_memory_notebook(memory, item_char_limit=900, **render_args)
    index = indices[0]
    if messages[index]['content'] != original_notebook:
        raise ValueError('original override notebook differs from the complete fork memory rendered at 900')
    effective = deepcopy(original)
    effective['messages'][index]['content'] = author.render_backtrack_memory_notebook(
        memory, item_char_limit=item_char_limit, **render_args)
    effective_path = original_path.with_name(original_path.stem + '.notebook4000.effective.json')
    with effective_path.open('x', encoding='utf-8') as stream:
        json.dump(effective, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return effective_path, config
