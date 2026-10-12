"""Project prior local-recovery events for a new Yellow or Red decision."""
from alf_event_memory import recovery_events
from icr_core import ProtocolError


def recovery_subtree(controller):
    active = [o.occurrence_id for o in controller.lineage]
    events, anchors, detail = recovery_events(controller, [None, *active])
    selected = set(anchors) | detail
    nodes = []
    for key in sorted(selected, key=lambda k: -1 if k is None else k):
        if key is None:
            nodes.append({'id': 'root', 'parent_id': None, 'retained_history_index': None,
                          'observation': controller.episode.initial['observation']})
            continue
        if key not in controller.occurrences:
            raise ProtocolError('recovery event refers to a missing ETT occurrence')
        occurrence = controller.occurrences[key]
        parent = occurrence.execution_parent_occurrence_id
        nodes.append({'id': 'o'+str(key), 'parent_id': 'root' if parent is None else 'o'+str(parent),
                      'retained_history_index': active.index(key) if key in active else None,
                      'path_relation': 'ACTIVE_PREFIX' if key in active else 'HISTORICAL_BRANCH',
                      'action_executed': occurrence.action,
                      'observation': occurrence.receipt['observation'],
                      'native_accepted': occurrence.receipt['native_accepted']})
    return {'events': events, 'nodes': nodes,
            'persistent_cycle_constraints': (controller.persistent_cycle_constraints()
                if hasattr(controller, 'persistent_cycle_constraints') else []),
            'semantics': 'Prior local-recovery decisions and their actual ETT receipts. '
                'Conclusions are fallible model advice, not facts or pending commands. '
                'A retained_history_index is a zero-based executed receipt, not a checkpoint. '
                'For checkpoint k, only retained indices less than k have happened in that state; '
                'later and historical-branch nodes are evidence of past attempts only. '
                'Unselected nodes and snapshots remain in the controller.'}
