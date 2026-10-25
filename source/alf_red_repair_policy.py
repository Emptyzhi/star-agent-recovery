"""Pure task-related public-evidence predicate for bounded Red repair."""
import re


def task_related_progress(task, receipt, action):
    if receipt['done']:
        return True
    if (not receipt['native_accepted'] or receipt['before'] == receipt['after']
            or action.startswith(('go to ', 'examine ', 'open ', 'close '))
            or action in {'look', 'inventory'}):
        return False
    task = task.lower()
    action_lower = action.lower()
    if action_lower.startswith('put '):
        match = re.search(r'put (?:two|one|a|an) ([a-z]+) in(?:/on|) ([a-z]+)', task)
        return bool(match and match.group(1) in action_lower and match.group(2) in action_lower)
    for verb in ('heat', 'cool', 'clean'):
        if action_lower.startswith(verb + ' ') and verb in task:
            object_match = re.match(verb + r' ([a-z]+)', action_lower)
            return bool(object_match and object_match.group(1) in task)
    if action_lower.startswith('examine ') and 'examine' in task:
        action_match = re.match(r'examine ([a-z]+) with (.+)', action_lower)
        if action_match is None:
            return False
        normalize = lambda value: re.sub(r'[^a-z0-9]', '', value)
        normalized_task = normalize(task)
        return all(normalize(part) in normalized_task for part in action_match.groups())
    return False
