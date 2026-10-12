"""Public WebShop evidence only; no product database or hidden task labels."""
from copy import deepcopy
import json
import re

from icr_core import ProtocolError
from tree_history import canonical
from webshop_semantics import checkpoint_snapshot


def interface_key(state):
    return canonical({key: state[key] for key in ('page_type', 'query', 'page_num',
        'asin', 'selected_options', 'subpage', 'visible_asins', 'option_types', 'available_actions')})


def receipt_evidence(receipt):
    if not receipt['native_accepted'] or receipt['action'].startswith('think['):
        return None
    return canonical([interface_key(receipt['after']), receipt['observation']])


def known_evidence(history, initial):
    values = {canonical([interface_key(initial['after']), initial['observation']])}
    for receipt in history:
        evidence = receipt_evidence(receipt)
        if evidence is not None:
            values.add(evidence)
    return values


def decision_key(history, state, initial):
    # Unique receipts, not an ever-increasing step/effect counter: returning to
    # the same page after an A-B-A cycle cannot create fictitious progress.
    return canonical([interface_key(state), sorted(known_evidence(history, initial))])


def new_evidence(receipt, earlier, initial):
    evidence = receipt_evidence(receipt)
    return evidence is not None and evidence not in known_evidence(earlier, initial)


def node(occurrence, active_ids):
    return dict(occurrence_id=occurrence.occurrence_id,
        execution_parent_occurrence_id=occurrence.execution_parent_occurrence_id,
        path_relation='ACTIVE_PREFIX' if occurrence.occurrence_id in active_ids else 'HISTORICAL_BRANCH',
        action=occurrence.action, observation=occurrence.receipt['observation'],
        accepted=occurrence.receipt['native_accepted'])


def option_resume_event(controller):
    path = controller.lineage
    if len(path) <= controller.branch_start_ordinal or controller.episode.snapshot()['done']:
        return None
    transition = path[-1]
    receipt = transition.receipt
    if not receipt['native_accepted'] or receipt['action'].startswith('think['):
        return None
    before, after = receipt['before'], receipt['after']
    added = set(after['available_actions']) - set(before['available_actions'])
    seen = {transition.action}
    for previous in reversed(path[:-1]):
        action = previous.action
        if action in seen:
            continue
        seen.add(action)
        if action not in added or previous.receipt['native_accepted']:
            continue
        match = re.fullmatch(r'click\[(.*)\]', action)
        if match is None or after['page_type'] != 'item':
            continue
        label = match[1]
        # Only a literal requested option on this same product can be a resume
        # reminder. Product/navigation labels are not certified goal operations.
        if (label not in after['option_types'] or not label
                or previous.receipt['before']['asin'] != after['asin']
                or not re.search(r'(?<!\w)' + re.escape(label) + r'(?!\w)',
                                 controller.episode.instruction, re.IGNORECASE)):
            continue
        key = canonical(['REQUESTED_OPTION_AVAILABLE', controller.cache_branch_id, previous.occurrence_id])
        if key in controller.delivered_green_events:
            return None
        return dict(event_key=key, kind='REQUESTED_OPTION_AVAILABLE',
            nodes=[previous.occurrence_id, transition.occurrence_id], action=action,
            evidence=dict(literal_instruction=controller.episode.instruction,
                          option_label=label, product=after['asin']),
            semantics='A literal requested option is now callable, not proof this product satisfies the task.')
    return None


def product_review_event(controller):
    path = controller.lineage
    if len(path) <= controller.branch_start_ordinal:
        return None
    current = path[-1]
    before, after = current.receipt['before'], current.receipt['after']
    if not (current.receipt['native_accepted'] and before['page_type'] == 'search'
            and after['page_type'] == 'item'):
        return None
    key = canonical(['PRODUCT_REQUIREMENT_REVIEW', controller.cache_branch_id, after['asin']])
    import json
    used = [json.loads(k) for k in controller.delivered_green_events]
    used = [k for k in used if k[0] == 'PRODUCT_REQUIREMENT_REVIEW' and k[1] == controller.cache_branch_id]
    if len(used) >= 2 or key in controller.delivered_green_events:
        return None
    return dict(event_key=key, kind='PRODUCT_REQUIREMENT_REVIEW', nodes=[current.occurrence_id],
        original_instruction=controller.episode.instruction, relevance='NOT_CERTIFIED_BY_CONTROLLER',
        semantics='Opening a product is not evidence that it meets the shopping instruction. '
            'Compare literal requirements with its actual page and selected options. '
            'Missing information is unknown, not a mismatch. Decide the next action yourself.')


def green_events(controller):
    c = controller
    active = {o.occurrence_id for o in c.lineage}
    events, ids = [], set()
    current = c.lineage[-1].occurrence_id if c.lineage else None
    # During a Red forward trial the restored page and Red's compact evidence
    # are the relevant public event. Re-emitting prior branch/product nodes
    # duplicates the failed suffix and competes with the current interface.
    recovery_trial = c.active_trial_id is not None
    if not recovery_trial:
        children = [o.occurrence_id for o in c.occurrences.values()
                    if o.execution_parent_occurrence_id == current]
        branch_key = canonical(['PRIOR_BRANCH_AT_CURRENT', current, children])
        if children and branch_key not in c.delivered_green_events:
            events.append(dict(event_key=branch_key, kind='PRIOR_BRANCH_AT_CURRENT',
                anchor_occurrence_id=current, historical_children=children,
                semantics='Earlier actions taken from this ETT node are history, not actions to repeat.'))
            ids.update(children)
            if current is not None:
                ids.add(current)
        for event in (option_resume_event(c), product_review_event(c)):
            if event is not None:
                events.append(event)
                ids.update(event['nodes'])
    if c.green_history_handoff is not None and not c.green_history_handoff['shown_request_ids']:
        record = c.latest_yellow_correction
        events.append(dict(event_key=canonical(['YELLOW_HANDOFF', record['scope_id']]),
            kind='YELLOW_HANDOFF', outcome=record['conclusion']['outcome'],
            model_advice_not_verified=deepcopy(record['conclusion']),
            semantics='Use the receipts; Yellow diagnosis is not an established cause or a new task.'))
        ids.update(c.green_history_handoff['executed_occurrence_ids'])
        navigation = getattr(c, 'web_yellow_navigation_handoff', None)
        if (navigation is not None
                and navigation['event_key'] not in c.delivered_green_events):
            events.append(deepcopy(navigation))
    if c.active_trial_id is not None:
        trial = c.forward_trials[c.active_trial_id]
        key = canonical(['RED_HANDOFF', trial.trial_id])
        if key not in c.delivered_green_events:
            path = c.archived_lineages[-1]
            advice = json.loads(trial.advice)
            restored = c.episode.snapshot()
            event = dict(event_key=key, kind='RED_HANDOFF',
                checkpoint_index=trial.checkpoint_index,
                original_task=c.episode.instruction,
                red_reflection_unverified=advice['reflection'],
                restored_public_page=checkpoint_snapshot(restored, observation_limit=500),
                current_legal_actions=deepcopy(restored['available_actions']),
                recovery_objective=(
                    'Continue the original task from the restored public page. '
                    'The prior terminal purchase is evidence about what failed, '
                    'not a command or a reason to replay its suffix.'),
                semantics='The reflection is fallible. Use the public receipts to assess it. '
                    'The controller restored before the selected action; that action is history, '
                    'not a required next action. Choose freely from the current public interface.')
            if trial.checkpoint_index < len(path):
                occurrence = path[trial.checkpoint_index]
                event['restored_before_action'] = dict(
                    action_index=trial.checkpoint_index,
                    occurrence_id=occurrence.occurrence_id,
                    action=occurrence.action)
            elif trial.checkpoint_index == len(path):
                failure = advice['unexecuted_model_failure']
                if (failure['action_executed'] is not False
                        or failure['executed_prefix_length'] != len(path)):
                    raise ProtocolError('empty Actor handoff differs from its real checkpoint')
                event['unexecuted_model_failure'] = deepcopy(failure)
                event['anchor_occurrence_id'] = current
            else:
                raise ProtocolError('Red checkpoint is outside the archived action path')
            terminal = next((o for o in reversed(path) if o.receipt['done']), None)
            if terminal is not None:
                receipt = terminal.receipt
                event['terminal_purchase_evidence'] = dict(
                    action=terminal.action,
                    product_page=checkpoint_snapshot(receipt['before'], observation_limit=500),
                    public_terminal_score=receipt['public_terminal_score'])
            events.append(event)
            if trial.checkpoint_index < len(path):
                ids.add(path[trial.checkpoint_index].occurrence_id)
            elif current is not None:
                ids.add(current)
    c.pending_green_events = [event['event_key'] for event in events]
    if not events:
        return ''
    return '[EVENT_SUBTREE]\n' + canonical(dict(events=events,
        nodes=[node(c.occurrences[i], active) for i in sorted(ids)],
        semantics='Only selected ETT event nodes. The full tree stays in the controller. '
            'Node parent IDs link to stored ETT nodes; omitted parents are not invented observations.'))
