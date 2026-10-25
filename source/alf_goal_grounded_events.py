"""Select sparse events using source-grounded goal relevance."""
from alf_operation_available import OperationAvailableV64Controller, newly_available_operations
from alf_sparse_events import render_sparse_green
from alf_goal_evidence import task_evidence, native_operation, object_evidence, action_evidence
from scoped_recovery import ScopeKind
from tree_history import canonical
from icr_core import ProtocolError
import json
from copy import deepcopy

VERSION='v16_empty_actor_output_red_recovery'
OPERATION_RESUME_POLICY='PUBLIC_MENU_CHANGE_AND_SUPPORTED_TASK_OPERATION'
TRANSFER_GOAL_REVIEW_POLICY='UNKNOWN_PURPOSE_ACQUISITION_MAX_TWO_DISTINCT_OBJECTS_PER_WINDOW'


def goal_available_operations(controller):
    goal=task_evidence(controller.episode.initial['observation'])
    commands=controller.episode.snapshot()['available_actions']
    return newly_available_operations(controller,
        relevance=lambda action:action_evidence(action,goal,commands))


def unknown_transfer_review(controller):
    path=controller.lineage
    if len(path)<=controller.branch_start_ordinal or controller.episode.snapshot()['done']:
        return None
    current=path[-1]
    if not current.receipt['native_accepted'] or not current.action.startswith('take '):return None
    op=native_operation(current.action)
    if op is None:raise ProtocolError('accepted take has no literal native object')
    goal=task_evidence(controller.episode.initial['observation'])
    evidence=object_evidence(op['object'],goal,controller.episode.snapshot()['available_actions'])
    if evidence['status']!='UNKNOWN':return None
    delivered=[json.loads(key) for key in controller.delivered_green_events]
    delivered=[key for key in delivered if key[0]=='UNKNOWN_TRANSFER_PURPOSE' and key[1]==controller.cache_branch_id]
    key=canonical(['UNKNOWN_TRANSFER_PURPOSE',controller.cache_branch_id,op['object']])
    if len(delivered)>=2 or key in controller.delivered_green_events:return None
    return dict(event_key=key,slot='UNKNOWN_PURPOSE',transfer_occurrence_ids=[current.occurrence_id],
        current_take_occurrence_id=current.occurrence_id,relevance_status='UNKNOWN',
        goal_evidence=evidence,public_goal=goal,acquired_object=op['object'])


class GoalGroundedV64Controller(OperationAvailableV64Controller):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.public_goal_evidence=None

    def start(self):
        initial=super().start()
        self.public_goal_evidence=task_evidence(self.episode.initial['observation'])
        return initial

    def _bind_scope(self,scope,directive):
        if scope.kind is not ScopeKind.GREEN:return super()._bind_scope(scope,directive)
        view=render_sparse_green(self,preproposal_incidents=False,
            include_discarded_discoveries=True,include_available_operations=True,
            include_transfer_goal_review=True,goal_grounded_events=True)
        self.episode.set_scope_prompt(view,scope.tree.tree_version,kind=scope.kind.value,
            directive=directive,incident_kind=scope.incident_kind)
        return self.episode.prompt(())

    def to_dict(self):
        return dict(super().to_dict(),controller_version=VERSION,
            operation_resume_policy=OPERATION_RESUME_POLICY,
            transfer_goal_review_policy=TRANSFER_GOAL_REVIEW_POLICY,
            public_goal_evidence=deepcopy(self.public_goal_evidence))
