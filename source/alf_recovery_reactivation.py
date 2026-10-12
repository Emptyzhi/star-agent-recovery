"""Select existing recovery evidence when a new public execution incident occurs."""
from copy import deepcopy
import json

from icr_core import ProtocolError
from scoped_repetition import operation_key
from tree_history import canonical


def relevant_recovery(controller):
    """Active Red plus the latest retained Yellow sharing an action or public state."""
    c = controller
    active = {None, *(o.occurrence_id for o in c.lineage)}
    yellows = []
    for scope in c.scopes.values():
        if scope.kind.value != 'YELLOW' or scope.event_anchor_occurrence_id not in active:
            continue
        if not scope.correction_records:
            continue
        record = scope.correction_records[-1]
        executed = record['executed_occurrence_id']
        if executed is not None and executed not in active:
            continue
        if not c.lineage:
            continue
        current = c.lineage[-1]
        anchor = scope.event_anchor_occurrence_id
        same_action = current.action == record['selected_action'] or (
            anchor is not None and current.action == c.occurrences[anchor].action)
        same_state = operation_key(scope.entry_state) in (
            operation_key(current.receipt['before']), operation_key(current.receipt['after']))
        if not (same_action or same_state):
            continue
        yellows.append((scope, record))
    yellow = max(yellows, key=lambda pair: pair[0].scope_id) if yellows else None
    trial = None if c.active_trial_id is None else c.forward_trials[c.active_trial_id]
    return trial, yellow


def repeated_receipt_event(controller, recovery):
    """A recurrence is observed evidence, never a controller claim of task failure."""
    c = controller
    trial, yellow = recovery
    # A restored prefix tail is historical, not a newly executed recovery edge.
    # Keep its original revision; sparse RED_HANDOFF still delivers its evidence.
    if trial is not None and not trial.actual_suffix_occurrence_ids:
        return None
    if (trial is None and yellow is None) or len(c.lineage) < 2:
        return None
    latest = c.lineage[-1]
    records = {t['occurrence_id']: t for t in c.repetition.transitions}
    if latest.occurrence_id not in records:
        raise ProtocolError('current occurrence lacks a public repetition record')
    current = records[latest.occurrence_id]
    if current['done'] or current['revision'] != c.repetition.revision:
        raise ProtocolError('recurrence projection requires the latest nonterminal receipt')
    for occurrence in reversed(c.lineage[:-1]):
        previous = records[occurrence.occurrence_id]
        if (previous['revision'] == current['revision'] and not previous['done']
                and all(previous[k] == current[k] for k in ('before', 'action', 'after', 'accepted'))):
            context = [None if trial is None else trial.trial_id,
                       None if yellow is None else yellow[0].scope_id]
            key = canonical(['RECOVERY_TRANSITION_RECURRED', c.cache_branch_id, context,
                             current['before'], current['action'], current['after'],
                             current['accepted'], current['revision']])
            return {'key': key, 'prior_occurrence_id': occurrence.occurrence_id,
                    'current_occurrence_id': latest.occurrence_id,
                    'public_evidence_revision': current['revision'],
                    'semantics': 'The same public before/action/after transition recurred on the '
                        'retained path without new public evidence. This does not prove hidden '
                        'state equality, goal failure, or the cause of any rejection.'}
    return None


def reactivated_recovery(controller, recovery):
    """Return small anchors and literal model advice; plans are not state facts."""
    c = controller
    trial, yellow = recovery
    anchors, selected, advice = [], set(), []
    if trial is not None:
        anchor = trial.prefix_occurrence_ids[-1] if trial.prefix_occurrence_ids else None
        anchors.append(anchor)
        selected.add(trial.source_checkpoint_occurrence_id)
        if trial.actual_suffix_occurrence_ids:
            selected.add(trial.actual_suffix_occurrence_ids[0])
        first = (None if not trial.actual_suffix_occurrence_ids else
                 c.occurrences[trial.actual_suffix_occurrence_ids[0]])
        advice.append({'kind': 'RED_REACTIVATED_MODEL_ADVICE_NOT_VERIFIED',
                       'trial_id': trial.trial_id, 'checkpoint_index': trial.checkpoint_index,
                       'review': json.loads(trial.advice),
                       'actual_first_occurrence_id': None if first is None else first.occurrence_id,
                       'actual_first_action': None if first is None else first.action,
                       'continuation_actions_executed': len(trial.actual_suffix_occurrence_ids),
                       'remaining_actions': c.episode.action_limit-len(c.lineage),
                       'semantics': 'Review is historical model output, not a pending command. '
                           'Any rejected_proposal was not executed; its rationale is not a '
                           'validated rationale for the corrected replacement.'})
    if yellow is not None:
        scope, record = yellow
        anchors.append(scope.event_anchor_occurrence_id)
        selected.update(scope.occurrence_ids)
        advice.append({'kind': 'YELLOW_REACTIVATED_MODEL_ADVICE_NOT_VERIFIED',
                       'scope_id': scope.scope_id, 'conclusion': deepcopy(record['conclusion']),
                       'selected_action': record['selected_action'],
                       'executed_occurrence_id': record['executed_occurrence_id'],
                       'correction_completed': record['correction_completed'],
                       'semantics': 'Read the actual receipt; a local correction label is not '
                           'proof that the obstruction or household goal was resolved.'})
    return anchors, selected, advice
