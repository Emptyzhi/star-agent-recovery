"""Simplified recovery using the archived v64 ETT/scopes and sparse Green events."""
from copy import deepcopy
import re

from alf_episode import AlfEpisode
from alf_restored_facts import restored_facts
from icr_core import ProtocolError, Role, environment_call
from scoped_recovery import ScopedRecoveryController, ScopeKind
from scoped_review_session import ForwardTrial, ReviewSession
from tree_history import canonical
from alf_author_reflection import reflection_query
from alf_local_evidence import decision_key, new_accepted_evidence, trace_text


VERSION = 'v64_simple_author_reflection_bounded_yellow_v2'
WINDOW = 49
YELLOW_ACTIONS = 4


class SimpleEpisode(AlfEpisode):
    def prompt(self, memory):
        text = super().prompt(memory)
        self.last_history_window = dict(truncated=False,
            omitted_records=0,retained_records=len(self.history),
            policy='complete_active_action_observation_history',event_subtree_separate=True)
        if self._scope_kind == 'YELLOW':
            text = text[:-1] + '\n[YELLOW_EVENT_SUBTREE]\n' + self._scope_view + '\n'
            text += ('Correct the specific rejected operation or break the recorded local loop. '
                     'Check the original task object, actual possession, location and native menu. '
                     'Use one action to check a missing prerequisite or correct it; do not start '
                     'a separate whole-room search. Do not take a substitute object merely because '
                     'it is available. Historical Red advice is fallible; actual receipts take priority. '
                     'Return one native action.\n>')
        # The base adapter counts before adding the Yellow subtree; count the
        # complete request too, without silently truncating either component.
        if self.prompt_token_counter(text) > min(self.prompt_context_limit-100, self.prompt_service_limit):
            raise ProtocolError('complete scope prompt exceeds declared context')
        return text


class ShortRepairMemory:
    """One immutable Red lesson; execution receipts, not memory edits, track effects."""
    def __init__(self, text, occurrence_id):
        self.text, self.occurrence_id = text, occurrence_id

    def actor_context(self):
        return dict(reflection=self.text, diagnosed_fault_occurrence=self.occurrence_id,
            semantics='Fallible failed-branch diagnosis. Read current receipts before acting; '
                      'discarded actions are not completed in this restored state.')

    def observe_action(self, action, occurrence_id, accepted):
        # The short lesson remains available throughout this single recovery window.
        return None


class SimpleV64Controller(ScopedRecoveryController):
    """Reuse v64 execution trees, scope calls/merges, snapshots and event projection."""
    RECOVERY_WINDOW = WINDOW
    YELLOW_ACTION_BUDGET = YELLOW_ACTIONS
    RED_ENABLED = True
    EVENT_SUBTREE_ENABLED = True
    def __init__(self, episode, oracle, ledger, *, task, emit):
        super().__init__(episode, oracle, ledger, task=task, emit=emit,
            yellow_entry_limit=2, red_entry_limit=1, max_root_recoveries=0,
            output_framing=True, yellow_action_ids=True, alfworld_event_memory=True,
            single_reflection_red=True)
        self.simple_yellow_event = None
        self.red_diagnosis = None
        self.selected_fault_index = None
        self.local_outcomes = []
        self.branch_start_ordinal = 0
        self.empty_actor_failure = None
        self.empty_actor_events = []

    def _edge_key(self, ordinal):
        receipt = self.lineage[ordinal].receipt
        return decision_key([o.receipt for o in self.lineage[:ordinal]],
                            receipt['before'], self.episode.initial['observation'])

    def _current_key(self):
        return decision_key(self.episode.history, self.episode.snapshot(),
                            self.episode.initial['observation'])

    def _proposal_incident(self, proposal):
        key = self._current_key()
        for ordinal in range(len(self.lineage)-1, self.branch_start_ordinal-1, -1):
            occurrence = self.lineage[ordinal]
            if occurrence.action != proposal or self._edge_key(ordinal) != key:
                continue
            receipt = occurrence.receipt
            kind = ('REPEATED_DECISION_WITHOUT_NEW_PUBLIC_EVIDENCE'
                    if receipt['native_accepted'] else 'ENVIRONMENT_REJECTION')
            return dict(kind=kind, action=proposal,
                incident_id=canonical([self.cache_branch_id, key, proposal]),
                occurrence_id=occurrence.occurrence_id, receipt=deepcopy(receipt),
                observed_occurrence_ids=[o.occurrence_id for o in self.lineage[ordinal:]],
                observed_actions=[o.action for o in self.lineage[ordinal:]], region=[key])
        if (len(self.lineage) > self.branch_start_ordinal and self.lineage[-1].receipt['invalid_action']
                and proposal not in self.episode.snapshot()['available_actions']):
            last = self.lineage[-1]
            return dict(kind='ENVIRONMENT_REJECTION', action=last.action,
                incident_id=canonical([self.cache_branch_id, key, last.action]),
                occurrence_id=last.occurrence_id, receipt=deepcopy(last.receipt))
        return None

    def _bind_scope(self, scope, directive):
        if scope.kind is ScopeKind.GREEN:
            # This is the original v64 sparse event renderer, including actual ETT
            # branch nodes, Yellow handoff, Red handoff and recurrence reactivation.
            return super()._bind_scope(scope, directive)
        if scope.kind is not ScopeKind.YELLOW or self.simple_yellow_event is None:
            raise ProtocolError('ordinary scope binding requires Green or active Yellow')
        event = dict(trigger=self.simple_yellow_event,
            original_task=self.episode.initial['observation'],
            pending_green_proposal=self.simple_yellow_proposal,
            local_actions=[dict(occurrence_id=i, action=self.occurrences[i].action,
                               observation=self.occurrences[i].receipt['observation'],
                               accepted=self.occurrences[i].receipt['native_accepted'])
                           for i in scope.occurrence_ids],
            current_facts=restored_facts(self.episode.history, self.episode.snapshot()),
            current_admissible_commands=self.episode.snapshot()['available_actions'],
            previous_recovery_events=self.episode.recovery_event_subtree,
            action_limit=self.YELLOW_ACTION_BUDGET,
            semantics='Actual local event nodes, not the full ETT. New text alone does not '
                      'resolve a rejected operation. Unresolved repair returns to Green.')
        if self.active_red_memory is not None:
            event['red_reflection'] = self.active_red_memory.actor_context()
        self.episode.set_scope_prompt(canonical(event), scope.tree.tree_version,
            kind='YELLOW', directive=directive, incident_kind=scope.incident_kind)
        return self.episode.prompt(())

    def _yellow_receipt_outcome(self, incident, receipt, earlier_receipts):
        if (incident['kind']=='ENVIRONMENT_REJECTION' and receipt['native_accepted']
                and receipt['action']==incident['action']):
            return 'ORIGINAL_OPERATION_ACCEPTED'
        if not receipt['native_accepted']:
            return 'UNRESOLVED_REJECTED_REPAIR'
        if new_accepted_evidence(receipt, earlier_receipts, self.episode.initial):
            return 'EVIDENCE_OBTAINED_REPLAN'
        return None

    def _local_repair(self, incident, proposal):
        from alf_recovery_projection import recovery_subtree
        self.episode.recovery_event_subtree = (recovery_subtree(self)
            if self.EVENT_SUBTREE_ENABLED else {'events': [], 'nodes': []})
        scope = self._enter_yellow(incident['kind'], proposal, incident_id=incident['incident_id'])
        scope.incident_kind = incident['kind']
        scope.exit_rules = ('LOCAL_REPAIR_OR_FOUR_ACTIONS_TO_GREEN', 'REAL_TERMINAL', 'SHARED_WINDOW_END')
        self.simple_yellow_event = deepcopy(incident)
        self.simple_yellow_proposal = proposal
        outcome = 'UNRESOLVED'
        local_edges = set()
        unexecuted = None
        while scope.action_serial < self.YELLOW_ACTION_BUDGET and self._forward_window()['new_actions_remaining']:
            parsed, request_id = self._ordinary(scope, 'Address the local obstruction using public receipts.')
            if not parsed.grammar_valid:
                raise ProtocolError('Yellow returned no native action')
            edge = (self._current_key(), parsed.action)
            if edge in local_edges:
                outcome = 'UNRESOLVED_LOCAL_REPEAT'
                unexecuted = dict(action=parsed.action, request_id=request_id)
                self.emit('yellow_repeated_proposal_returned', scope_id=scope.scope_id,
                          state_key=edge[0], **unexecuted)
                break
            local_edges.add(edge)
            earlier_receipts = list(self.episode.history)
            receipt, occurrence = self._execute(scope, parsed, request_id)
            if receipt['done']:
                outcome = 'ENVIRONMENT_TERMINAL'
                break
            receipt_outcome = self._yellow_receipt_outcome(incident, receipt, earlier_receipts)
            if receipt_outcome is not None:
                outcome = receipt_outcome
                break
        if not scope.occurrence_ids:
            raise ProtocolError('Yellow entered without an available action')
        last = self.occurrences[scope.occurrence_ids[-1]]
        record = dict(request_id=last.request_id, scope_id=scope.scope_id,
            conclusion=dict(outcome=outcome, original_obstruction=deepcopy(incident),
                pending_green_proposal=proposal, unexecuted_repeat=unexecuted,
                attempted_actions=[dict(occurrence_id=i, action=self.occurrences[i].action,
                    observation=self.occurrences[i].receipt['observation'],
                    accepted=self.occurrences[i].receipt['native_accepted']) for i in scope.occurrence_ids],
                current_facts=restored_facts(self.episode.history,self.episode.snapshot())),
            step_role='LOCAL_REPAIR', selected_action=last.action,
            executed_occurrence_id=last.occurrence_id,
            correction_completed=outcome=='ORIGINAL_OPERATION_ACCEPTED')
        scope.correction_record = deepcopy(record)
        scope.correction_records.append(deepcopy(record))
        self.latest_yellow_correction = deepcopy(record)
        self.local_outcomes.append(dict(scope_id=scope.scope_id,outcome=outcome,
                                       actions=scope.action_serial,unexecuted_repeat=unexecuted))
        self._return_to_green(scope, outcome,
            handoff_directive='Use the actual Yellow receipts and attempted actions. Evidence obtained '
                'is not proof that the task or obstruction is solved. Resume planning for the original '
                'task; an unresolved repair does not end the task.')
        self.simple_yellow_event = None

    def _run_window(self):
        seen_incidents = set()
        while not self.episode.snapshot()['done'] and self._forward_window()['new_actions_remaining']:
            green = self._green_scope()
            if self.active_scope_id != green.scope_id:
                raise ProtocolError('ordinary execution must resume the existing Green tree')
            parsed, request_id = self._ordinary(green, 'Continue the original task.')
            if not parsed.grammar_valid:
                raw_text = parsed.evidence['raw_text']
                if isinstance(raw_text, str) and not raw_text.strip():
                    failure = dict(kind='EMPTY_ACTOR_OUTPUT', request_id=request_id,
                        scope_id=green.scope_id, scope_kind='GREEN', raw_text=raw_text,
                        executed_prefix_length=len(self.lineage),
                        state=deepcopy(self.episode.snapshot()), action_executed=False,
                        recovery_trial_id=self.active_trial_id)
                    self.empty_actor_failure = failure
                    self.empty_actor_events.append(deepcopy(failure))
                    self.controller_events.append(dict(node_kind='MODEL_FAILURE', **failure))
                    self.emit('actor_empty_output_failure', **failure)
                    return
                raise ProtocolError('Green returned no native action')
            incident = self._proposal_incident(parsed.action)
            if (incident is None and self.lineage and self.lineage[-1].receipt['invalid_action']
                    and parsed.action in self.episode.snapshot()['available_actions']):
                self.emit('green_alternative_preserved', request_id=request_id, action=parsed.action,
                          prior_rejection_occurrence_id=self.lineage[-1].occurrence_id)
            if (incident is not None and incident['incident_id'] not in seen_incidents
                    and self.attempt_scope_entries['YELLOW'] < self.YELLOW_ENTRY_LIMIT):
                seen_incidents.add(incident['incident_id'])
                self._local_repair(incident,parsed.action)
            else:
                # No veto or hidden replacement. After Yellow is exhausted Green
                # keeps its remaining window, including the chance to reach Red.
                self._execute(green,parsed,request_id)

    def _red_request(self, scope, prompt, *, commands=()):
        address = self._address(scope)
        self.oracle.bind(address,self.episode,prompt,())
        return self.oracle.request(address, Role.ALTERNATIVE if commands else Role.REFLECTION,
            0,prompt,self.ledger,commands=commands,output_limit=100 if commands else 256)

    def _reflect_and_restore(self):
        path = list(self.lineage)
        empty_failure = deepcopy(self.empty_actor_failure)
        if not path and empty_failure is None:
            raise ProtocolError('failed attempt has no executed action to diagnose')
        green = self._green_scope()
        reason = ('EMPTY_ACTOR_OUTPUT' if empty_failure is not None else
                  'FAILED_ENVIRONMENT_TERMINAL' if self.episode.snapshot()['done'] else 'FORWARD_WINDOW_EXHAUSTED')
        red = self._new_scope(ScopeKind.RED,reason,caller=green)
        red.exit_rules = ('REFLECT_IDENTIFY_FAULT_RESTORE_THEN_GREEN',)
        self.attempt_scope_entries['RED'] += 1
        nodes = [dict(action_index=i,occurrence_id=o.occurrence_id,
            parent_occurrence_id=o.execution_parent_occurrence_id,scope_kind=o.scope_kind.value,
            action=o.action,observation=o.receipt['observation'],accepted=o.receipt['native_accepted'])
            for i,o in enumerate(path)]
        evidence = dict(task=self.episode.initial['observation'],failure_reason=reason,
            failed_branch_event_subtree=nodes,
            final_facts=restored_facts(self.episode.history,self.episode.snapshot()),
            current_admissible_commands=self.episode.snapshot()['available_actions'],
            yellow_outcomes=self.local_outcomes,
            semantics='This is the actual failed ETT branch. Actions can be accepted but wrong '
                      'for the task. The first rejection need not be the first wrong decision.')
        transcript = trace_text(self.episode.initial['observation'], path)
        if empty_failure is not None:
            evidence['unexecuted_model_failure'] = empty_failure
            transcript += ('\n\nController-observed failure: the next Actor response was empty. '
                'No action was submitted and no new environment observation was produced. '
                'The retained prefix above is the actual executed history.\n'
                + canonical(empty_failure))
        prompt = reflection_query(transcript + '\n\nSTATUS: FAIL')
        reflection = self._red_request(red,prompt)
        if not reflection.text.strip():
            raise ProtocolError('Red reflection is empty')
        self.red_diagnosis = reflection.text
        self.emit('red_failure_reflected',scope_id=red.scope_id,request_id=reflection.request.request_id,
                  reflection=reflection.text,evidence=evidence)
        selection_request_id = None
        if empty_failure is not None:
            # A missing action has no environment edge to undo. Its exact
            # pre-decision checkpoint is the current prefix, including root.
            k = len(path)
            checkpoint = deepcopy(self.tip_checkpoint)
            fault_occurrence_id = None
            self.emit('empty_actor_checkpoint_selected',checkpoint_index=k,
                failed_request_id=empty_failure['request_id'],
                policy='EXACT_PRE_GENERATION_CHECKPOINT',action_executed=False)
        else:
            choices = tuple('ACTION '+str(i) for i in range(len(path)))
            choose = self.episode.review_formatter([
                dict(role='system',content='Locate the executed action that should be changed according '
                    'to this reflection. Choose its actual action index, not a success score. '
                    'The controller will restore BEFORE that action. Do not automatically choose '
                    'the first rejected command: an earlier accepted wrong-object decision may '
                    'be the cause. There is no default root. Choose one listed ACTION id.'),
                dict(role='user',content=trace_text(self.episode.initial['observation'],path,indexed=True)
                    + '\n\nReflection:\n' + reflection.text
                    + '\n\nSelect the action to change. Retaining an earlier wrong take retains '
                      'that held object; undo the causing action rather than only its later symptom.\n'
                    + '\n'.join(choices))])
            selection = self._red_request(red,choose,commands=choices)
            if selection.text not in choices:
                raise ProtocolError('Red selected an unlisted executed action')
            k = int(re.fullmatch(r'ACTION ([0-9]+)',selection.text)[1])
            checkpoint = self.checkpoint_cache[path[k].occurrence_id]
            fault_occurrence_id = path[k].occurrence_id
            selection_request_id = selection.request.request_id
        if checkpoint['history'] != [o.receipt for o in path[:k]]:
            raise ProtocolError('selected ETT node does not match its retained prefix')
        start_cost = self.ledger.to_dict()
        start_restore = self.checkpoint_restore_count
        environment_call(self.episode,self.ledger,self.episode.restore,checkpoint)
        self.checkpoint_restore_count += 1
        self.episode.action_limit = k+self.RECOVERY_WINDOW
        self.tip_checkpoint = self.episode.checkpoint()
        self.selected_fault_index = k
        self.archived_lineages.append(path)
        self.lineage = path[:k]
        self.branch_start_ordinal = k
        self.cache_branch_id += 1
        self.green_history_handoff = None
        self.latest_yellow_correction = None
        self.rejected_public_actions.clear()
        self.attempt_scope_entries['YELLOW'] = 0
        self._set_green_cursor_from_lineage(self.lineage)
        advice = dict(reflection=reflection.text,fault_action_index=k,fault_occurrence_id=fault_occurrence_id,
                      selection_request_id=selection_request_id)
        if empty_failure is not None:
            advice['unexecuted_model_failure'] = empty_failure
        self.review_sessions[0] = ReviewSession(0,red.scope_id,reason,
            tuple(o.occurrence_id for o in path),k,current_checkpoint_index=k,trial_ids=[0])
        self.forward_trials[0] = ForwardTrial(0,0,k,tuple(o.occurrence_id for o in self.lineage),
            fault_occurrence_id,k+self.RECOVERY_WINDOW,self.RECOVERY_WINDOW,reflection.request.request_id,canonical(advice),
            start_cost,start_restore)
        self.active_trial_id = 0
        self.active_red_memory = ShortRepairMemory(reflection.text,fault_occurrence_id)
        self.empty_actor_failure = None
        self.green_handoff_directive = 'Recover using this short Red lesson and the actual restored state.'
        red.correction_record = deepcopy(advice)
        red.archived = True
        red.exit_reason = 'VERIFIED_RESTORE_TO_GREEN'
        self.active_scope_id = green.scope_id
        self.controller_events.append(dict(node_kind='RETURN',from_scope_id=red.scope_id,
            to_scope_id=green.scope_id,reason=red.exit_reason))
        self.emit('red_checkpoint_restored',checkpoint_index=k,fault_occurrence_id=fault_occurrence_id,
                  actual_state=self.episode.snapshot(),new_actions=self.RECOVERY_WINDOW,actor_owner='GREEN',
                  reflection_request_id=reflection.request.request_id,selection_request_id=selection_request_id)

    def run(self):
        if self.status != 'ACTIVE':
            raise ProtocolError('controller must be started exactly once')
        self._run_window()
        if self.episode.snapshot()['public_terminal_score'] == 1:
            self.status='SUCCESS'
            return self.to_dict()
        if (self.empty_actor_failure is None and not self.episode.snapshot()['done']
                and self._forward_window()['new_actions_remaining']):
            raise ProtocolError('Red cannot enter before failure or budget exhaustion')
        if not self.RED_ENABLED:
            self.status = ('INITIAL_EMPTY_ACTOR_OUTPUT' if self.empty_actor_failure is not None else
                           'TERMINAL_FAILURE' if self.episode.snapshot()['done'] else
                           'INITIAL_WINDOW_EXHAUSTED')
            self.emit('red_disabled_initial_attempt_finished', status=self.status)
            return self.to_dict()
        self._reflect_and_restore()
        self._run_window()
        if self.empty_actor_failure is not None:
            session=self._finish_forward_trial('REPAIR_UNRESOLVED','EMPTY_ACTOR_OUTPUT_AFTER_RED')
            session.status='COMPLETE_EMPTY_ACTOR_OUTPUT'
            self.status='RECOVERY_EMPTY_ACTOR_OUTPUT'
            return self.to_dict()
        success=self.episode.snapshot()['public_terminal_score']==1
        outcome='SUCCESS' if success else 'TERMINAL_FAILURE' if self.episode.snapshot()['done'] else 'WINDOW_EXHAUSTED'
        session=self._finish_forward_trial(outcome,'SINGLE_RECOVERY_FINISHED')
        session.status='COMPLETE_'+outcome
        self.status='SUCCESS' if success else 'TERMINAL_FAILURE' if self.episode.snapshot()['done'] else 'RECOVERY_WINDOW_EXHAUSTED'
        return self.to_dict()

    def to_dict(self):
        return dict(super().to_dict(),schema_version='V64_SIMPLE_REFLECTION_LOCAL_ROLLBACK',
            controller_version=VERSION,red_diagnosis=self.red_diagnosis,
            selected_fault_index=self.selected_fault_index,local_outcomes=deepcopy(self.local_outcomes),
            checkpoint_policy='REFLECTION_IDENTIFIES_EXECUTED_ERROR_ACTION',
            branch_start_ordinal=self.branch_start_ordinal,
            empty_actor_events=deepcopy(self.empty_actor_events),
            empty_actor_policy='ONE_RED_REFLECTION_RESTORE_PRE_GENERATION_CHECKPOINT',
            max_red_model_calls=2,max_recovery_actions=self.RECOVERY_WINDOW)
