"""Small evidence subtrees at explicit events; full ETT stays in the controller."""
from copy import deepcopy
import json

from tree_history import canonical
from alf_public_receipts import event_state_evidence
from icr_core import ProtocolError
from alf_recovery_reactivation import relevant_recovery, repeated_receipt_event, reactivated_recovery


def render_sparse_green(controller, *, preproposal_incidents=True, include_discarded_discoveries=False,
                        include_available_operations=False,include_transfer_goal_review=False,
                        goal_grounded_events=False):
    c=controller
    occurrences=c.occurrences
    active=[None,*(o.occurrence_id for o in c.lineage)]
    current=active[-1]
    children={}
    for key,o in occurrences.items():
        if key!=o.occurrence_id or (o.execution_parent_occurrence_id is not None and o.execution_parent_occurrence_id not in occurrences):
            raise ProtocolError('event subtree has an invalid execution parent')
        children.setdefault(o.execution_parent_occurrence_id,[]).append(key)
    triggers=[];keys=[];anchors=[];selected={current};advice=[]
    discoveries=()
    available_operations=[]
    transfer_review=None
    if include_transfer_goal_review:
        from alf_transfer_goal_review import transfer_goal_review
        if goal_grounded_events:
            from alf_goal_grounded_events import unknown_transfer_review
            transfer_review=unknown_transfer_review(c)
        else:
            transfer_review=transfer_goal_review(c)
        if transfer_review is not None:
            triggers.append('TRANSFER_GOAL_REVIEW')
            keys.append(transfer_review['event_key'])
            selected.update(transfer_review['transfer_occurrence_ids'])
            first=occurrences[transfer_review['transfer_occurrence_ids'][0]]
            anchors.append(first.execution_parent_occurrence_id)
    if include_available_operations:
        from alf_operation_available import newly_available_operations
        if goal_grounded_events:
            from alf_goal_grounded_events import goal_available_operations
            available_operations=goal_available_operations(c)
        else:
            available_operations=newly_available_operations(c)
        if available_operations:
            triggers.append('FAILED_OPERATION_AVAILABLE')
            for event in available_operations:
                keys.append(event['event_key'])
                failure=occurrences[event['failed_occurrence_id']]
                anchors.append(failure.execution_parent_occurrence_id)
                selected.update((failure.occurrence_id,event['transition_occurrence_id']))
    if children.get(current):
        key=canonical(['PRIOR_BRANCH_AT_CURRENT',current,children[current]])
        if key not in c.delivered_green_events:
            keys.append(key);triggers.append('PRIOR_BRANCH_AT_CURRENT');anchors.append(current)
    handoff=c.green_history_handoff
    if handoff is not None and not handoff['shown_request_ids']:
        key=canonical(['YELLOW_HANDOFF',c.latest_yellow_correction['scope_id']])
        if key not in c.delivered_green_events:
            keys.append(key);triggers.append('YELLOW_HANDOFF')
            anchors.append(handoff['anchor_occurrence_id']);selected.update(handoff['executed_occurrence_ids'])
            record=c.latest_yellow_correction
            advice.append({'kind':'YELLOW_MODEL_ADVICE_NOT_VERIFIED',
                'scope_id':record['scope_id'],'conclusion':deepcopy(record['conclusion']),
                'selected_action':record['selected_action'],'executed_occurrence_id':record['executed_occurrence_id'],
                'correction_completed':record['correction_completed']})
    if c.active_trial_id is not None:
        trial=c.forward_trials[c.active_trial_id]
        key=canonical(['RED_HANDOFF',trial.trial_id])
        if trial.handoff_request_id is None and key not in c.delivered_green_events:
            keys.append(key);triggers.append('RED_HANDOFF')
            anchors.append(trial.prefix_occurrence_ids[-1] if trial.prefix_occurrence_ids else None)
            selected.update(trial.actual_suffix_occurrence_ids)
            advice.append({'kind':'RED_MODEL_ADVICE_NOT_VERIFIED','trial_id':trial.trial_id,
                           'checkpoint_index':trial.checkpoint_index,'review':json.loads(trial.advice),
                           'already_executed_occurrence_ids':list(trial.actual_suffix_occurrence_ids)})
            session=c.review_sessions[trial.owner_review_session_id]
            # Empty Actor output can fail before the first environment action.
            # Root is then the actual event anchor, not a fabricated occurrence.
            selected.add(session.frozen_path_occurrence_ids[-1] if session.frozen_path_occurrence_ids else None)
            if include_discarded_discoveries and trial.checkpoint_index < len(session.frozen_path_occurrence_ids):
                from alf_discovery_handoff import discarded_discoveries
                discoveries=discarded_discoveries(
                    [occurrences[i] for i in session.frozen_path_occurrence_ids],trial.checkpoint_index)
    if preproposal_incidents and c.lineage and c.lineage[-1].receipt['invalid_action']:
        o=c.lineage[-1]
        key=canonical(['ENVIRONMENT_REJECTION',c.cache_branch_id,o.action,o.receipt['after'],c.repetition.revision])
        if key not in c.delivered_green_events:
            keys.append(key);triggers.append('ENVIRONMENT_REJECTION')
            anchors.append(o.execution_parent_occurrence_id)
    recovery=relevant_recovery(c)
    recurrence=repeated_receipt_event(c,recovery) if preproposal_incidents else None
    if recurrence is not None and recurrence['key'] not in c.delivered_green_events:
        keys.append(recurrence['key']);triggers.append('RECOVERY_TRANSITION_RECURRED')
        selected.update((recurrence['prior_occurrence_id'],recurrence['current_occurrence_id']))
        anchors.append(occurrences[recurrence['prior_occurrence_id']].execution_parent_occurrence_id)
    reactivated=False
    if set(triggers)&{'ENVIRONMENT_REJECTION','RECOVERY_TRANSITION_RECURRED'}:
        recall_anchors,recall_nodes,recall_advice=reactivated_recovery(c,recovery)
        if recall_advice:
            anchors.extend(recall_anchors);selected.update(recall_nodes);advice.extend(recall_advice)
            reactivated=True
    persistent = c.persistent_cycle_constraints() if hasattr(c, 'persistent_cycle_constraints') else []
    if persistent:
        constraint_key = canonical(['PERSISTENT_CYCLE_CONSTRAINTS', persistent])
        if constraint_key not in c.delivered_green_events:
            keys.append(constraint_key)
            triggers.append('PERSISTENT_CYCLE_CONSTRAINT')
            selected.add(current)
    c.pending_green_events=keys
    if not triggers:return ''
    anchors=list(dict.fromkeys(anchors))
    if not set(anchors)<=set(active):raise ProtocolError('event anchor is outside the retained prefix')
    selected.update(anchors)
    short_event_only=set(triggers)<={'FAILED_OPERATION_AVAILABLE','TRANSFER_GOAL_REVIEW'}
    if not short_event_only:
        for anchor in anchors:selected.update(children.get(anchor,[]))
    if set(triggers)&{'RED_HANDOFF','PRIOR_BRANCH_AT_CURRENT'}:
        for scope in c.scopes.values():
            if scope.kind.value=='YELLOW' and set(scope.occurrence_ids)&selected:
                for record in scope.correction_records:
                    advice.append({'kind':'HISTORICAL_YELLOW_MODEL_ADVICE_NOT_VERIFIED',
                        'scope_id':scope.scope_id,'conclusion':deepcopy(record['conclusion']),
                        'executed_occurrence_id':record['executed_occurrence_id']})
    # Adding view nodes must not also recall unrelated old Yellow instructions.
    selected.update(discoveries)
    def ref(node):return 'root' if node is None else 'o'+str(node)
    def ancestors(node):
        path=[]
        while node is not None:
            path.append(node);node=occurrences[node].execution_parent_occurrence_id
        return [None,*reversed(path)]
    paths=[];nodes=[]
    for key in sorted(selected,key=lambda k:-1 if k is None else k):
        if key is None:
            nodes.append({'id':'root','position':'CURRENT' if current is None else 'ACTIVE_PATH',
                          'observation':c.episode.initial['observation']})
            continue
        o=occurrences[key];p=o.execution_parent_occurrence_id
        nodes.append({'id':ref(key),'parent_id':ref(p),'position':'CURRENT' if key==current else 'ACTIVE_PATH' if key in active else 'HISTORICAL_BRANCH',
                      'action_executed':o.action,'observation':o.receipt['observation'],
                      'native_accepted':o.receipt['native_accepted'],'done':o.receipt['done'],
                      'public_terminal_score':o.receipt['public_terminal_score']})
        lineage=ancestors(key)
        ancestor=next((a for a in reversed(lineage[:-1]) if a in selected),None)
        if ancestor is None and None not in selected:continue
        middle=lineage[lineage.index(ancestor)+1:-1]
        if middle:paths.append({'from':ref(ancestor),'to':ref(key),'omitted_occurrences':[ref(k) for k in middle],
                                'relation':'RECORDED_PATH_NOT_DIRECT_EDGE'})
    if short_event_only:
        from alf_transfer_goal_review import compact_event_state
        current_evidence=compact_event_state(c)
    else:
        current_evidence=event_state_evidence(c.episode.initial,c.episode.history,c.episode.snapshot(),c.episode.action_limit)
    fragment={'triggers':triggers,'current_occurrence':ref(current),'anchors':[ref(k) for k in anchors],'nodes':nodes,
              'edges':[[ref(occurrences[k].execution_parent_occurrence_id),ref(k)] for k in sorted(n for n in selected if n is not None) if occurrences[k].execution_parent_occurrence_id in selected],
              'collapsed_paths':paths,'recovery_advice':advice,
              'current_evidence':current_evidence,
              'persistent_cycle_constraints':persistent,
              'semantics':'Selected actual event nodes only. Unshown nodes and checkpoints remain in controller ETT. '
                  'Advice is historical model output; already executed actions are not pending commands. '
                  'Other retained actions and receipts remain in the ordinary history above.'}
    if discoveries:
        fragment['historical_discovery_nodes']=[ref(i) for i in discoveries]
        fragment['historical_discovery_semantics']=(
            'These accepted views were observed on the abandoned branch, after the restored prefix. '
            'Use them as search clues, not as claims about current possession, open containers or '
            'object positions after rollback. Verify against current receipts. Their recorded actions '
            'are past events, not pending commands. The current_evidence describes the actual restored state.')
    if transfer_review is not None:
        fragment['transfer_goal_review']=transfer_review
        fragment['transfer_goal_instruction']=(
            'Compare the actual recent take/put receipts and currently held objects with '
            'the original task before choosing the next action. Does this acquisition '
            'advance the requested goal or obtain a necessary tool, container, or other '
            'prerequisite? Acceptance alone does not show task relevance. A different '
            'object name alone does not show irrelevance. If it serves the task, continue; '
            'otherwise choose a local correction or further evidence using the native '
            'menu. Do not keep relocating objects merely because those transfers work. '
            'Previously placed objects are at the locations in their actual receipts, '
            'not at imagined locations. Relevance has not been decided by the controller. '
            'This is a review at a transfer boundary, not a required put or action veto.')
        if goal_grounded_events:
            fragment['transfer_goal_instruction']=(
                'The acquired object has no established goal or prerequisite role in the '
                'quoted task/current command evidence. UNKNOWN does not mean irrelevant. '
                'Before the next action, reconsider whether it is the requested object or '
                'has a supported tool/container/precondition role. Preserve necessary '
                'prerequisites. Do not substitute another object for the requested one '
                'merely because it can be acquired or processed. Select the next native '
                'action using actual facts; no put is required and no action is filtered.')
    if available_operations:
        fragment['newly_available_failed_operations']=available_operations
        fragment['operation_resume_task']=c.episode.initial['observation']
        fragment['operation_resume_instruction']=(
            'An earlier attempted operation failed. After the recorded later action, '
            'it changed from unavailable to available in the public native menu. '
            'The earlier operation has not thereby been executed or completed. '
            'Reconsider it against the original task and current facts before choosing '
            'your next action. If it still serves the task, its prerequisite may now '
            'be satisfied. This is an observed availability change, not proof of its '
            'cause, a required retry, or proof that the earlier plan was appropriate. '
            'No action is forced or removed.')
    if 'RECOVERY_TRANSITION_RECURRED' in triggers:
        fragment['observed_recurrence']={k:v for k,v in recurrence.items() if k!='key'}
    if reactivated:
        fragment['recovery_reassessment']=(
            'Reassess the earlier recovery advice against the original goal and all accepted '
            'receipts since its anchor. A later rejection does not undo an earlier successful '
            'transfer or operation. Do not assume that the old diagnosis or an already executed '
            'repair is still appropriate. Decide what prerequisite is actually supported or '
            'unknown before selecting one current action. Repeating an inspection with the '
            'same result supplies no new evidence by itself. No action is automatically banned '
            'and no extra recovery budget is granted.')
    return '[EVENT_SUBTREE]\n'+canonical(fragment)+'\n[EVENT_SUBTREE_END]'
