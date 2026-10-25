"""Bounded Yellow diagnosis before choosing a local repair action."""
from copy import deepcopy
from alf_discovery_controller import DiscoveryV64Controller
from alf_local_evidence import is_query, new_accepted_evidence
from alf_restored_facts import restored_facts
from alf_simple_v64 import YELLOW_ACTIONS
from icr_core import BudgetExhausted,ProtocolError,Role
from scoped_recovery import ScopeKind
from tree_history import canonical

VERSION='v64_native_history_yellow_receipt_feedback_v13'
DIAGNOSIS_TOKENS=100


def state_change_evidence(occurrence):
    receipt=occurrence.receipt
    before=receipt['before']['available_actions']
    after=receipt['after']['available_actions']
    return dict(occurrence_id=occurrence.occurrence_id,action=occurrence.action,
                observation=receipt['observation'],
                commands_removed=[action for action in before if action not in after],
                commands_added=[action for action in after if action not in before])


class DiagnosingYellowV64Controller(DiscoveryV64Controller):
    DIAGNOSIS_OUTPUT_TOKENS = DIAGNOSIS_TOKENS
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.yellow_diagnoses={}
        self.yellow_entry_states={}

    def _local_feedback(self, scope):
        state=self.episode.snapshot()
        key=self._current_key()
        attempts=[]
        known_checks=[]
        for ordinal, occurrence in enumerate(self.lineage):
            receipt=occurrence.receipt
            earlier=[o.receipt for o in self.lineage[:ordinal]]
            novel=new_accepted_evidence(receipt,earlier,self.episode.initial)
            row=dict(state_change_evidence(occurrence),
                accepted=receipt['native_accepted'],new_public_evidence=novel)
            if occurrence.occurrence_id in scope.occurrence_ids:
                attempts.append(row)
            if (self._edge_key(ordinal)==key and
                    (not receipt['native_accepted'] or (is_query(occurrence.action) and not novel))):
                known_checks.append(row)
        return dict(pending_operation=self.simple_yellow_proposal,
            attempts=attempts,previous_ineffective_actions_at_current_state=known_checks,
            current_facts=restored_facts(self.episode.history,state),
            pending_operation_available=self.simple_yellow_proposal in state['available_actions'],
            actions_remaining=self.YELLOW_ACTION_BUDGET-scope.action_serial,
            semantics='These are actual receipts, not predicted outcomes. A rejected action '
                'or check without new evidence did not verify the proposed repair. That alone '
                'does not disprove every possible cause. Earlier advice remains unverified. '
                'The complete native menu remains available; no semantic action filter is applied.')

    def _yellow_receipt_outcome(self, incident, receipt, earlier_receipts):
        if not receipt['native_accepted']:
            return None
        if receipt['action']==incident['action'] and incident['kind']=='ENVIRONMENT_REJECTION':
            return 'ORIGINAL_OPERATION_ACCEPTED'
        entry=self.yellow_entry_states[self.active_scope_id]
        target=self.simple_yellow_proposal
        if target not in entry['available_actions'] and target in receipt['after']['available_actions']:
            return 'PENDING_OPERATION_BECAME_AVAILABLE'
        novel=new_accepted_evidence(receipt,earlier_receipts,self.episode.initial)
        if novel and not is_query(receipt['action']):
            return 'ACCEPTED_STATE_CHANGE_REPLAN'
        if (novel and receipt['action'].startswith('go to ')
                and incident['kind']=='REPEATED_DECISION_WITHOUT_NEW_PUBLIC_EVIDENCE'):
            return 'NEW_NAVIGATION_EVIDENCE_REPLAN'
        # New query wording alone does not establish removal of this obstruction.
        return None

    def _diagnose_local_event(self,scope):
        if (scope.kind is not ScopeKind.YELLOW or self.active_scope_id!=scope.scope_id
                or scope.action_serial!=0 or self.simple_yellow_event is None
                or scope.scope_id in self.yellow_diagnoses):
            raise ProtocolError('local diagnosis requires a fresh active Yellow event')
        state=deepcopy(self.episode.snapshot())
        self.yellow_entry_states[scope.scope_id]=state
        effects=[state_change_evidence(o) for o in self.lineage
                 if o.receipt['native_accepted'] and not is_query(o.action)]
        evidence=dict(original_task=self.episode.initial['observation'],
            incident=deepcopy(self.simple_yellow_event),pending_green_action=self.simple_yellow_proposal,
            current_state=state,current_transfer_facts=restored_facts(self.episode.history,state),
            accepted_state_change_nodes=effects,
            local_feedback=self._local_feedback(scope),
            semantics='These are actual public receipts on the active execution path. '
                'Accepted state changes remain in effect unless later receipts change them. '
                'Command changes are exact before/after public menu differences, not inferred rules. '
                'A rejected operation does not change possession. No old model diagnosis '
                'or predicted environment observation is included as evidence.')
        prompt=('Diagnose this ONE local failure using its public receipts. Do not summarize '
            'the failure or invent a cause from "Nothing happens." Devise a concise local '
            'plan accounting for the attempted actions and their observed effects. Start with '
            'one current native operation, then its supporting fact and the receipt or command '
            'change to check afterward. Use at most two short sentences; no special format. '
            'Compare the requested object, established possession and current menu before '
            'explaining accessibility. When a visible target cannot be taken while another '
            'object is held, consider an available put and then verify pickup availability. '
            'Remember the observed placement if the object is needed later. This is a testable '
            'candidate, not an asserted capacity rule or mandatory put. A visible object is '
            'not necessarily held. A check already made without new evidence is not a new '
            'repair. For a search loop, use actual prior observations rather than assuming '
            'where the target should be. Do not repeat an unsupported explanation about '
            'blocked objects, full containers or hidden rules. If the cause is unknown, say '
            'so briefly and choose a check that can add evidence. Plan only the next local '
            'test; Green remains responsible for the whole task.\n\n'
            '[LOCAL_EVENT_SUBTREE]\n'+canonical(evidence)+'\n\nLocal diagnosis:')
        address=self._address(scope)
        self.oracle.bind(address,self.episode,prompt,())
        generation=self.oracle.request(address,Role.REFLECTION,0,prompt,self.ledger,
                                       output_limit=self.DIAGNOSIS_OUTPUT_TOKENS,stop_mode='first_nonempty_line')
        if self.episode.snapshot()!=state:
            raise ProtocolError('local diagnosis changed the bound environment state')
        if not generation.text.strip():
            raise ProtocolError('empty local diagnosis; no substituted advice')
        record=dict(request_id=generation.request.request_id,text=generation.text,
                    kind='LOCAL_DIAGNOSIS_MODEL_ADVICE_NOT_VERIFIED')
        self.yellow_diagnoses[scope.scope_id]=record
        self.emit('yellow_local_diagnosis',scope_id=scope.scope_id,diagnosis=deepcopy(record),
                  evidence=evidence,prompt=prompt,output_limit=self.DIAGNOSIS_OUTPUT_TOKENS,
                  environment_actions_used=0)

    def _ordinary(self,scope,directive):
        if scope.kind is ScopeKind.YELLOW and scope.scope_id not in self.yellow_diagnoses:
            self._diagnose_local_event(scope)
        return super()._ordinary(scope,directive)

    def _bind_scope(self,scope,directive):
        prompt=super()._bind_scope(scope,directive)
        if scope.kind is not ScopeKind.YELLOW:
            return prompt
        if scope.scope_id not in self.yellow_diagnoses or not prompt.endswith('>'):
            raise ProtocolError('Yellow selection requires its recorded local diagnosis')
        prompt=(prompt[:-1]+'\n[LOCAL_DIAGNOSIS_MODEL_ADVICE_NOT_VERIFIED]\n'
                +self.yellow_diagnoses[scope.scope_id]['text']
                +'\n[LOCAL_REPAIR_FEEDBACK]\n'+canonical(self._local_feedback(scope))
                +'\nChoose one current native action using the latest actual feedback. '
                 'The initial plan is a hypothesis, not an instruction to repeat. If its '
                 'check failed or added no evidence, revise the next operation using current '
                 'facts instead of prescribing the same check again. Do not treat a new '
                 'observation phrase as a solved obstruction. Never invent the result of '
                 'an unexecuted action. Return only the next native action.\n>')
        if self.episode.prompt_token_counter(prompt)>min(self.episode.prompt_context_limit-100,
                                                         self.episode.prompt_service_limit):
            raise BudgetExhausted('Yellow diagnosis handoff exceeds context; no truncation')
        return prompt

    def _return_to_green(self,scope,reason,handoff_directive=None):
        if scope.kind is not ScopeKind.YELLOW or scope.scope_id not in self.yellow_diagnoses:
            raise ProtocolError('Yellow return requires its original diagnosis')
        records=(scope.correction_record,scope.correction_records[-1],self.latest_yellow_correction)
        for record in records:
            if record['scope_id']!=scope.scope_id:
                raise ProtocolError('Yellow diagnosis handoff crossed scope ownership')
            record['conclusion']['local_diagnosis']=deepcopy(self.yellow_diagnoses[scope.scope_id])
            record['conclusion']['local_repair_feedback']=self._local_feedback(scope)
        return super()._return_to_green(scope,reason,handoff_directive)

    def to_dict(self):
        return dict(super().to_dict(),controller_version=VERSION,
                    yellow_diagnoses=deepcopy(self.yellow_diagnoses),
                    yellow_entry_states=deepcopy(self.yellow_entry_states),
                    yellow_diagnosis_policy='ONCE_PER_ENTRY_100_TOKENS_FIRST_LINE',
                    yellow_evidence_policy='ACCEPTED_STATE_CHANGES_WITH_EXACT_PUBLIC_COMMAND_DELTAS',
                    yellow_prompt_policy='FACT_GROUNDED_LOCAL_PLAN_WITH_ACTUAL_RECEIPT_FEEDBACK',
                    yellow_return_policy='OPERATION_ACCEPTED_CAPABILITY_RESTORED_STATE_CHANGE_OR_NEW_NAVIGATION',
                    max_yellow_diagnosis_calls=4)
