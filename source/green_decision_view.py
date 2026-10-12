"""Read-only, event-scoped projection of all actual execution occurrences."""
from collections import defaultdict

from icr_core import ProtocolError
from tree_history import canonical


def occurrence_ref(occurrence_id):
    return 'root' if occurrence_id is None else 'o' + str(occurrence_id)


def render_green_view(controller, scope):
    position = {'scope_id': scope.scope_id, 'branch_id': controller.cache_branch_id,
                'current_node': scope.current_node_id,
                'executed_prefix_length': len(controller.lineage)}
    view = '[EXECUTION_POSITION]\n' + canonical(position)
    view += '\n[FORWARD_WINDOW] ' + canonical(controller._forward_window())
    if getattr(controller.episode, 'native_actor_history', False):
        # ALFWorld's native Actor receives these fields only inside event evidence.
        view = ''
    mode = controller.episode.green_tree_mode
    if mode not in ('none', 'event', 'full'):
        raise ProtocolError('unknown Green tree visibility mode')
    if mode == 'none':
        return view
    occurrences = controller.occurrences
    active = [None, *(o.occurrence_id for o in controller.lineage)]
    current = active[-1]
    children = defaultdict(list)
    for key, item in sorted(occurrences.items()):
        parent = item.execution_parent_occurrence_id
        if key != item.occurrence_id or (parent is not None and (parent not in occurrences or parent >= key)):
            raise ProtocolError('execution history has an invalid occurrence parent')
        children[parent].append(key)

    anchors, reasons, detail = [], [], set()
    event_records = []
    if children[current]:
        anchors.append(current)
        reasons.append('PRIOR_BRANCH_AT_CURRENT')
    handoff = controller.green_history_handoff
    if handoff is not None:
        anchors.append(handoff['anchor_occurrence_id'])
        reasons.append('YELLOW_HANDOFF')
        detail.update(handoff['executed_occurrence_ids'])
    if controller.active_trial_id is not None:
        trial = controller.forward_trials[controller.active_trial_id]
        anchor = trial.prefix_occurrence_ids[-1] if trial.prefix_occurrence_ids else None
        anchors.append(anchor)
        reasons.append('RED_TRIAL_HISTORY')
        if controller.alfworld_event_memory:
            # Retain the current continuation's intermediate successful receipts.
            detail.update(trial.actual_suffix_occurrence_ids)
    if (controller.alfworld_event_memory and controller.lineage
            and controller.lineage[-1].receipt['invalid_action']):
        anchors.append(occurrences[current].execution_parent_occurrence_id)
        detail.add(current)
        reasons.append('ENVIRONMENT_REJECTION')
    if not anchors and mode == 'event':
        return view
    if controller.alfworld_event_memory and (
            mode == 'full' or set(reasons) & {'PRIOR_BRANCH_AT_CURRENT','YELLOW_HANDOFF','RED_TRIAL_HISTORY'}):
        from alf_event_memory import recovery_events
        event_records, event_anchors, event_details = recovery_events(controller, active)
        anchors.extend(event_anchors)
        detail.update(event_details)
        if event_records:
            reasons.append('YELLOW_EVENTS_ON_RETAINED_PATH')
    anchors = list(dict.fromkeys(anchors))
    if (any(anchor not in active for anchor in anchors)
            or not detail <= (set(occurrences) if controller.alfworld_event_memory else set(active))):
        raise ProtocolError('Green history context is outside the actual active prefix')
    selected = {current, *anchors, *detail}
    if mode == 'full':
        reasons = ['FULL_TREE']
        anchors = list(dict.fromkeys([None, *anchors,
                       *(parent for parent in sorted(occurrences) if len(children[parent]) > 1)]))
        selected = {None, *occurrences}
    paths, branches = [], []

    def ancestry(node):
        path = []
        while node is not None:
            path.append(node)
            node = occurrences[node].execution_parent_occurrence_id
        return [None, *reversed(path)]

    def descendants(node):
        result, pending = [], [node]
        while pending:
            child = pending.pop()
            result.append(child)
            pending.extend(reversed(children[child]))
        return result

    # Outcomes describe actual receipts or review-entry reasons, not root causes.
    for anchor in anchors:
        for child in children[anchor]:
            selected.add(child)
            outcomes = []
            if child not in active:
                subtree = set(descendants(child))
                for key in sorted(subtree):
                    receipt = occurrences[key].receipt
                    if receipt['done']:
                        outcomes.append({'at': occurrence_ref(key), 'kind': 'TERMINAL',
                                         'public_score': receipt['public_terminal_score']})
                        selected.add(key)
                    elif receipt['invalid_action']:
                        outcomes.append({'at': occurrence_ref(key), 'kind': 'ACTION_REJECTED'})
                        selected.add(key)
                for session in controller.review_sessions.values():
                    frozen = session.frozen_path_occurrence_ids
                    if frozen[-1] in subtree and not occurrences[frozen[-1]].receipt['done']:
                        outcomes.append({'at': occurrence_ref(frozen[-1]), 'kind': 'REVIEW_ENTRY',
                                         'reason': controller.scopes[session.red_scope_id].entry_reason})
                        selected.add(frozen[-1])
            branches.append({'anchor': occurrence_ref(anchor), 'entry': occurrence_ref(child),
                             'on_current_path': child in active, 'observed_outcomes': outcomes})

    # An omitted path is explicit; it must not become an invented direct edge.
    for node in sorted(n for n in selected if n is not None):
        path = ancestry(node)
        prior = next((p for p in reversed(path[:-1]) if p in selected), None)
        if prior is None and None not in selected:
            continue
        middle = path[path.index(prior) + 1:-1]
        if middle:
            paths.append({'from': occurrence_ref(prior), 'to': occurrence_ref(node),
                          'omitted_occurrences': [occurrence_ref(k) for k in middle],
                          'relation': 'COLLAPSED_EXECUTED_PATH_NOT_DIRECT_EDGE'})

    observations, observation_ids, nodes = [], {}, []
    live_observation = (controller.episode.history[-1]['observation'] if controller.episode.history
                        else controller.episode.initial['observation'])
    for key in sorted(selected, key=lambda value: -1 if value is None else value):
        row = {'id': occurrence_ref(key),
               'position': 'CURRENT' if key == current else 'ACTIVE_PATH' if key in active else 'HISTORICAL_BRANCH'}
        if key is not None:
            item = occurrences[key]
            receipt = item.receipt
            parent = item.execution_parent_occurrence_id
            row.update(parent_id=occurrence_ref(parent), parent_outside_fragment=parent not in selected,
                       action_executed=item.action, scope_kind=item.scope_kind.value,
                       native_accepted=receipt['native_accepted'])
            row.update(terminal=receipt['done'], rejected=receipt['invalid_action'],
                       public_terminal_score=receipt['public_terminal_score'])
            state = receipt['after']
            observation = receipt['observation']
        else:
            state = controller.episode.initial['after']
            observation = controller.episode.initial['observation']
        # Both tree arms use exactly this node representation. Only membership differs.
        row['state_after'] = {field: value for field, value in state.items() if field != 'observation'}
        if observation == live_observation:
            row['observation_ref'] = 'CURRENT_OBSERVATION'
        else:
            if observation not in observation_ids:
                observation_ids[observation] = len(observations)
                observations.append(observation)
            row['observation_id'] = observation_ids[observation]
        nodes.append(row)
    fragment = {'namespace': 'actual_execution_occurrences', 'triggers': reasons,
                'current_occurrence': occurrence_ref(current),
                'anchors': [occurrence_ref(a) for a in anchors], 'nodes': nodes,
                'edges': [[occurrence_ref(occurrences[k].execution_parent_occurrence_id), occurrence_ref(k)]
                          for k in sorted(n for n in selected if n is not None)
                          if occurrences[k].execution_parent_occurrence_id in selected],
                'collapsed_paths': paths, 'branches': branches, 'observations': observations}
    if controller.alfworld_event_memory:
        fragment['recovery_events'] = event_records
        if controller.active_trial_id is not None:
            from alf_trial_evidence import active_trial_evidence, factor_node_interfaces
            fragment['public_interfaces']=factor_node_interfaces(nodes)
            fragment['interface_encoding']='Each node public_interface_id indexes its exact available_actions.'
            fragment['active_recovery']=active_trial_evidence(controller)
    return view + '\n[GREEN_TREE_FRAGMENT]\n' + canonical(fragment) + '\n[GREEN_TREE_FRAGMENT_END]'
