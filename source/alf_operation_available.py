"""Recall failed operations when actual forward execution restores availability."""
from alf_yellow_diagnosis import DiagnosingYellowV64Controller
from alf_sparse_events import render_sparse_green
from scoped_recovery import ScopeKind
from tree_history import canonical
from alf_local_evidence import is_query
from alf_transfer_goal_review import TRANSFER_GOAL_REVIEW_POLICY

VERSION='v64_scoped_operation_resume_and_transfer_goal_review_v15'
OPERATION_RESUME_POLICY='LATEST_NONQUERY_FAILURE_AFTER_ACCEPTED_NONQUERY_CHANGE_ONCE_PER_BRANCH'


def newly_available_operations(controller, *, relevance=None):
    path=controller.lineage
    # Replaying a checkpoint is not a new forward repair event.
    if len(path)<=controller.branch_start_ordinal or controller.episode.snapshot()['done']:
        return []
    transition=path[-1]
    receipt=transition.receipt
    if not receipt['native_accepted'] or is_query(transition.action):
        return []
    before=receipt['before']['available_actions']
    after=receipt['after']['available_actions']
    added={action for action in after if action not in before}
    # The transition itself may have successfully executed a previously failed action.
    seen={transition.action}
    events=[]
    for previous in reversed(path[:-1]):
        action=previous.action
        if action not in added or action in seen or is_query(action):
            continue
        seen.add(action)
        # An accepted later execution retires an earlier rejection of this action.
        if previous.receipt['native_accepted']:
            continue
        evidence=None if relevance is None else relevance(action)
        if relevance is not None and evidence['status'] not in ('TARGET','SUPPORTED_PREREQUISITE'):
            continue
        key=canonical(['FAILED_OPERATION_AVAILABLE',controller.cache_branch_id,
                       previous.occurrence_id])
        if key in controller.delivered_green_events:
            return []
        events.append(dict(event_key=key,action=action,
            failed_occurrence_id=previous.occurrence_id,
            transition_occurrence_id=transition.occurrence_id,
            available_before_transition=False,available_now=True))
        if evidence is not None:events[-1]['goal_evidence']=evidence
        # Only the most recent restored operation is relevant to this reminder.
        break
    return events


class OperationAvailableV64Controller(DiagnosingYellowV64Controller):
    def _bind_scope(self,scope,directive):
        if scope.kind is not ScopeKind.GREEN:
            return super()._bind_scope(scope,directive)
        view=render_sparse_green(self,preproposal_incidents=False,
            include_discarded_discoveries=True,include_available_operations=True,
            include_transfer_goal_review=True)
        self.episode.set_scope_prompt(view,scope.tree.tree_version,
            kind=scope.kind.value,directive=directive,incident_kind=scope.incident_kind)
        return self.episode.prompt(())

    def to_dict(self):
        return dict(super().to_dict(),controller_version=VERSION,
                    operation_resume_policy=OPERATION_RESUME_POLICY,
                    transfer_goal_review_policy=TRANSFER_GOAL_REVIEW_POLICY)
