"""WebShop specialization of the v16 SimpleV64 ETT recovery lifecycle."""
from copy import deepcopy

from alf_simple_v64 import SimpleV64Controller, YELLOW_ACTIONS
from icr_core import ProtocolError, Role, SupportKind
from scoped_recovery import ScopeKind
from tree_history import canonical
from experiment_settings import Experiment, EXECUTION_PROTOCOL
from web_v16_episode import author_reflection
from web_v16_evidence import decision_key, green_events, interface_key, new_evidence
from webshop_semantics import (checkpoint_kind, checkpoint_snapshot,
                               progress_kind, public_fault_action_index)

VERSION = 'webshop_v16_native_invalid_after_yellow_unlimited_requests_20261018'
NAVIGATION_ACTIONS = frozenset((
    'click[Search]', 'click[Back to Search]', 'click[< Prev]', 'click[Next >]'))


def is_navigation_action(action):
    return action in NAVIGATION_ACTIONS or action.startswith('search[')


class WebV16Controller(SimpleV64Controller):
    EVENT_MEMORY = False
    RECOVERY_WINDOW = 15

    def __init__(self, *args, settings=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.settings = Experiment() if settings is None else settings
        self.RECOVERY_WINDOW = self.settings.recovery_actions
        self.YELLOW_ENTRY_LIMIT = (0 if self.settings.ablation == 'without_yellow'
                                   else self.settings.yellow_entries)
        self.episode.action_limit = self.settings.initial_actions
        self.episode.actor_output_tokens = self.settings.actor_output_tokens
        self.episode.yellow_output_tokens = self.settings.yellow_output_tokens
        self.oracle.ordinary_output_limit = self.settings.actor_output_tokens
        self.yellow_diagnoses = {}
        self.yellow_protocol_failures = []
        self.red_phase1_evidence = None
        self.simple_yellow_event = None
        self.simple_yellow_proposal = None
        self.local_outcomes = []
        self.web_yellow_navigation_handoff = None
        self.recovery_stages = []
        self.recovery_index = 0
        if self.settings.max_recoveries != 1:
            self.RED_ENTRY_LIMIT = self.settings.max_recoveries
            original_emit = self.emit
            self.emit = lambda event, **data: original_emit(event, recovery_index=self.recovery_index, **data)

    def run(self):
        if self.settings.ablation == 'full' and self.settings.max_recoveries != 1:
            return self._run_recovery_depth()
        if self.settings.ablation != 'without_red':
            return super().run()
        if self.status != 'ACTIVE':
            raise ProtocolError('controller must be started exactly once')
        self._run_window()
        state = self.episode.snapshot()
        if state['public_terminal_score'] == 1:
            self.status = 'SUCCESS'
        elif self.empty_actor_failure is not None:
            self.status = 'INITIAL_EMPTY_ACTOR_OUTPUT'
        elif state['done']:
            self.status = 'TERMINAL_FAILURE'
        elif not self._forward_window()['new_actions_remaining']:
            self.status = 'INITIAL_WINDOW_EXHAUSTED'
        else:
            raise ProtocolError('initial window returned before its actual boundary')
        self.emit('red_disabled_trial_finished', status=self.status,
                  unused_recovery_actions=self.RECOVERY_WINDOW)
        return self.to_dict()

    def _record_recovery_stage(self):
        rewards = [o.receipt['public_terminal_score'] for o in self.occurrences.values() if o.receipt['done']]
        stage = dict(recovery_count=self.recovery_index, purchase_rewards=rewards,
                     best_purchase_reward=max([0, *rewards]), cost=self.ledger.to_dict())
        self.recovery_stages.append(stage)
        self.emit('recovery_stage_finished', **stage)

    def to_dict(self):
        result = super().to_dict()
        result['execution_protocol'] = EXECUTION_PROTOCOL
        if self.settings.max_recoveries != 1:
            result.update(recovery_stages=deepcopy(self.recovery_stages),
                          max_recoveries=self.settings.max_recoveries,
                          completed_recoveries=self.recovery_index)
        return result

    def _run_recovery_depth(self):
        if self.status != 'ACTIVE':
            raise ProtocolError('controller must be started exactly once')
        for index in range(self.settings.max_recoveries + 1):
            self.recovery_index = index
            if index:
                self._reflect_and_restore()
            self._run_window()
            self._record_recovery_stage()
            success = self.recovery_stages[-1]['best_purchase_reward'] == 1
            state = self.episode.snapshot()
            if index:
                outcome = ('SUCCESS' if success else 'REPAIR_UNRESOLVED' if self.empty_actor_failure is not None
                           else 'TERMINAL_FAILURE' if state['done'] else 'WINDOW_EXHAUSTED')
                session = self._finish_forward_trial(outcome, 'BOUNDED_RECOVERY_STAGE_FINISHED')
                session.status = 'COMPLETE_' + outcome
            if success:
                self.status = 'SUCCESS'
                break
            if self.empty_actor_failure is None and not state['done'] and self._forward_window()['new_actions_remaining']:
                raise ProtocolError('recovery stage returned before a real boundary')
        if self.status != 'SUCCESS':
            self.status = 'RECOVERY_LIMIT_REACHED'
        result = self.to_dict()
        result.update(recovery_stages=deepcopy(self.recovery_stages),
                      max_recoveries=self.settings.max_recoveries,
                      completed_recoveries=self.recovery_index,
                      max_red_model_calls=2*self.settings.max_recoveries)
        return result

    def _plain_recovery_notes(self):
        # Keep component-generated advice when ablating ETT evidence projection.
        notes, keys = [], []
        if self.green_history_handoff is not None and not self.green_history_handoff['shown_request_ids']:
            record = self.latest_yellow_correction
            conclusion = record['conclusion']
            notes.append('Yellow outcome: ' + conclusion['outcome'] + '\nYellow advice: '
                         + conclusion['local_diagnosis']['text'])
            keys.append(canonical(['YELLOW_HANDOFF', record['scope_id']]))
        if self.active_trial_id is not None:
            trial = self.forward_trials[self.active_trial_id]
            key = canonical(['RED_HANDOFF', trial.trial_id])
            if key not in self.delivered_green_events:
                notes.append('Red reflection about the failed attempt:\n' + self.active_red_memory.text)
                keys.append(key)
        self.pending_green_events = keys
        if not notes:
            return ''
        return ('[RECOVERY_NOTE]\n' + '\n'.join(notes)
                + '\nThis advice is fallible history; use the current public page and native actions.')

    def _red_request(self, scope, prompt, *, commands=()):
        address = self._address(scope)
        self.oracle.bind(address, self.episode, prompt, ())
        return self.oracle.request(address, Role.ALTERNATIVE if commands else Role.REFLECTION,
            0, prompt, self.ledger, commands=commands,
            output_limit=self.settings.red_selection_tokens if commands else self.settings.red_reflection_tokens)

    def _record_yellow_protocol_failure(self, scope, *, phase, request_id,
                                        raw_output, error, parser=None):
        """Persist an unexecuted Yellow model-output failure before returning Green."""
        record = {
            'phase': phase,
            'scope_id': scope.scope_id,
            'request_id': request_id,
            'raw_output': raw_output,
            'error_type': type(error).__name__ if isinstance(error, Exception) else 'ProtocolError',
            'error': str(error),
            'parser': deepcopy(parser),
            'executed': False,
            'lineage_length': len(self.lineage),
            'public_checkpoint': checkpoint_snapshot(self.episode.snapshot(), observation_limit=260),
        }
        self.yellow_protocol_failures.append(deepcopy(record))
        self.controller_events.append(dict(node_kind='MODEL_FAILURE', failure_kind='YELLOW_PROTOCOL', **record))
        self.emit('yellow_protocol_failure', **record)
        return record

    def _current_key(self):
        return decision_key(self.episode.history, self.episode.snapshot(), self.episode.initial)

    def _edge_key(self, ordinal):
        return decision_key([o.receipt for o in self.lineage[:ordinal]],
                            self.lineage[ordinal].receipt['before'], self.episode.initial)

    def _proposal_incident(self, proposal):
        duplicate = self._pending_duplicate(proposal)
        if duplicate is not None:
            entries = self.terminal_evidence[duplicate.result_key]
            failed = next(entry for entry in reversed(entries) if entry['score'] != 1.0)
            occurrence = self.occurrences[failed['occurrence_id']]
            return dict(kind='REPEATED_TERMINAL', action=proposal,
                occurrence_id=occurrence.occurrence_id,
                incident_id=duplicate.incident_id, receipt=deepcopy(occurrence.receipt),
                prior_score=failed['score'], result_projection=deepcopy(failed['metadata']))

        current = interface_key(self.episode.snapshot())
        for i in range(len(self.lineage)-1, self.branch_start_ordinal-1, -1):
            occurrence = self.lineage[i]
            receipt = occurrence.receipt
            before, after = receipt['before'], receipt['after']
            if (occurrence.action == proposal and receipt['native_accepted'] and not receipt['done']
                    and not proposal.startswith('think[')
                    and interface_key(before) == current):
                return dict(kind='REPEATED_DECISION_WITHOUT_STABLE_INTERFACE_PROGRESS',
                    action=proposal, occurrence_id=occurrence.occurrence_id,
                    incident_id=canonical([self.cache_branch_id, current, proposal,
                                           occurrence.occurrence_id]),
                    receipt=deepcopy(receipt), interface_state=deepcopy(self.episode.snapshot()))
        return None

    def _latest_green_rejection(self):
        if len(self.lineage) <= self.branch_start_ordinal:
            return None
        occurrence = self.lineage[-1]
        receipt = occurrence.receipt
        if occurrence.scope_kind is not ScopeKind.GREEN or not receipt['invalid_action']:
            return None
        return dict(kind='ENVIRONMENT_REJECTION', action=occurrence.action,
            incident_id=canonical(['ENVIRONMENT_REJECTION', occurrence.occurrence_id,
                                   receipt['invalid_detail']]),
            occurrence_id=occurrence.occurrence_id, receipt=deepcopy(receipt),
            invalid_detail=receipt['invalid_detail'])

    def _run_window(self):
        seen_incidents = set()
        while not self.episode.snapshot()['done'] and self._forward_window()['new_actions_remaining']:
            rejected = self._latest_green_rejection()
            if (rejected is not None and rejected['incident_id'] not in seen_incidents
                    and self.attempt_scope_entries['YELLOW'] < self.YELLOW_ENTRY_LIMIT):
                seen_incidents.add(rejected['incident_id'])
                self.emit('yellow_incident_detected', incident_id=rejected['incident_id'],
                          kind='ENVIRONMENT_REJECTION', action=rejected['action'],
                          occurrence_id=rejected['occurrence_id'],
                          invalid_detail=rejected['invalid_detail'], trigger_timing='AFTER_REAL_RECEIPT')
                self._local_repair(rejected, rejected['action'])
                continue

            green = self._green_scope()
            if self.active_scope_id != green.scope_id:
                raise ProtocolError('ordinary execution must resume the existing Green tree')
            parsed, request_id = self._ordinary(green, 'Continue the original task.')
            if not parsed.grammar_valid:
                if self.attempt_scope_entries['YELLOW'] >= self.YELLOW_ENTRY_LIMIT:
                    self.emit('actor_protocol_failure_submitted', request_id=request_id,
                              raw_text=parsed.evidence['raw_text'], action=parsed.action,
                              reason='YELLOW_ENTRY_LIMIT')
                    self._execute(green, parsed, request_id)
                    continue
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
                # A non-empty malformed proposal is a local model/interface
                # incident. Let Yellow inspect the current public page once;
                # the next Green request remains the owner of the task plan.
                incident = dict(
                    kind='MODEL_ACTION_PROTOCOL_FAILURE',
                    action=raw_text.strip() if isinstance(raw_text, str) else str(raw_text),
                    incident_id=canonical([
                        'MODEL_ACTION_PROTOCOL_FAILURE', len(self.lineage), raw_text,
                    ]),
                    occurrence_id=None,
                    receipt=None,
                    invalid_detail=parsed.evidence.get('protocol_failure'),
                    interface_state=deepcopy(self.episode.snapshot()),
                )
                self.emit('yellow_incident_detected',
                          incident_id=incident['incident_id'],
                          kind=incident['kind'], action=incident['action'],
                          occurrence_id=None, trigger_timing='AFTER_MODEL_PROTOCOL_FAILURE',
                          invalid_detail=incident['invalid_detail'])
                self._local_repair(incident, incident['action'])
                continue

            incident = self._proposal_incident(parsed.action)
            if (incident is not None and incident['incident_id'] not in seen_incidents
                    and self.attempt_scope_entries['YELLOW'] < self.YELLOW_ENTRY_LIMIT):
                seen_incidents.add(incident['incident_id'])
                self.emit('yellow_incident_detected', incident_id=incident['incident_id'],
                          kind=incident['kind'], action=parsed.action,
                          occurrence_id=incident['occurrence_id'], trigger_timing='BEFORE_EXECUTION')
                self._local_repair(incident, parsed.action)
            else:
                self._execute(green, parsed, request_id)

    def _red_facts(self):
        return dict(original_instruction=self.episode.instruction, current_public_page=self.episode.snapshot())

    @staticmethod
    def _red_public_state(state):
        """Keep Red's public mismatch evidence without replaying page dumps."""
        compact = checkpoint_snapshot(state, observation_limit=700)
        compact.pop('option_domains', None)
        return compact

    def _red_trace(self, path, *, indexed=False):
        # Red needs product/title/price/options and receipts, not the entire
        # HTML-like observation history. This keeps the author reflection
        # protocol while bounding the WebShop-specific evidence transcript.
        text = self.episode.initial['observation'][:900] + '\n'
        for i, occurrence in enumerate(path):
            prefix = 'ACTION %d\n' % i if indexed else ''
            receipt = occurrence.receipt
            before = self._red_public_state(receipt['before'])
            after = self._red_public_state(receipt['after'])
            text += (prefix + 'Action: ' + occurrence.action + '\n'
                     + 'Observation: ' + after['visible_observation'] + '\n'
                     + '[PUBLIC_RECEIPT]\n' + canonical({
                         'action_index': i,
                         'accepted': receipt['native_accepted'],
                         'done': receipt['done'],
                         'score': receipt.get('public_terminal_score'),
                         'before': before,
                         'after': after,
                     }) + '\n')
        return text

    def _red_reflection_prompt(self, transcript):
        prompt = author_reflection(transcript)
        marker = '\n\nNew plan:'
        prefix, separator, suffix = prompt.rpartition(marker)
        if not separator:
            raise ProtocolError('author Reflexion prompt must expose a final New plan boundary')
        audit = (
            '\n\nBefore writing the new plan, audit the purchased product against every condition '
            'in the original shopping instruction. List all confirmed mismatches, including product '
            'identity/category, color or style, size/quantity, and price when requested. For each '
            'mismatch, cite the public product title, displayed price, selected option, or receipt '
            'that supports it. Distinguish MATCH, MISMATCH, and UNKNOWN; missing evidence is UNKNOWN, '
            'not a guessed mismatch. A partial score alone does not identify which condition failed. '
            'Keep the audit concise enough to name every mismatch, then give a short corrective plan. '
            'Do not choose a checkpoint in this reflection.'
        )
        return prefix + audit + marker + suffix

    def _red_selection_hint(self):
        return ('Use the mismatch conditions listed in the reflection to locate the action that first '
                'selected each conflicting value. The controller restores BEFORE the ACTION index you '
                'choose, so select the product-click that introduced a mismatching product, the '
                'option-click that selected a mismatching variant, or the search action only when the '
                'query itself introduced the mismatch. If several conditions conflict, choose the '
                'earliest action that selected any of them, so the restored state no longer retains '
                'those choices. Do not choose an action after the mismatch was selected (such as Buy Now) '
                'or merely the later rejected command; those checkpoints preserve the bad choice. Use '
                'the public trace and receipts, not the partial score alone.')

    def _red_action_index_mapping(self, path, reflection):
        return public_fault_action_index(path, reflection)

    def _red_phase1_evidence(self, path, reflection, evidence):
        """Freeze phase-1 public evidence for the second-stage locator."""
        transitions = []
        for index, occurrence in enumerate(path):
            receipt = occurrence.receipt
            transitions.append({
                'action_index': index,
                'occurrence_id': occurrence.occurrence_id,
                'checkpoint_kind': checkpoint_kind(occurrence),
                'action': occurrence.action,
                'accepted': receipt['native_accepted'],
                'done': receipt['done'],
                'invalid_detail': receipt.get('invalid_detail'),
                'before': checkpoint_snapshot(receipt['before'], observation_limit=120),
                'after': checkpoint_snapshot(receipt['after'], observation_limit=120),
            })
        return {
            'schema': 'webshop_red_phase1_public_evidence_v1',
            'reflection': reflection,
            'task_instruction': self.episode.instruction,
            'failure_reason': evidence['failure_reason'],
            'public_transitions': transitions,
            'rule': 'Phase 2 may select only an ACTION index from this executed path. '
                    'The controller restores BEFORE that action; the reflection is advice, '
                    'not a verified fact.',
        }

    def _red_memory_facts(self, path, fault_index, reflection):
        fault = path[fault_index] if fault_index < len(path) else None
        fault_receipt = None if fault is None else fault.receipt
        suffix = []
        for occurrence in path[fault_index:]:
            receipt = occurrence.receipt
            suffix.append({
                'action': occurrence.action,
                'before': checkpoint_snapshot(receipt['before'], observation_limit=180),
                'after': checkpoint_snapshot(receipt['after'], observation_limit=180),
                'accepted': receipt['native_accepted'],
                'done': receipt['done'],
            })
        return {
            'schema': 'webshop_red_recovery_memory_v1',
            'restored_checkpoint_index': fault_index,
            'fault_action_index': fault_index,
            'fault_action': None if fault is None else fault.action,
            'fault_transition': (None if fault_receipt is None else {
                'before': checkpoint_snapshot(fault_receipt['before'], observation_limit=220),
                'after': checkpoint_snapshot(fault_receipt['after'], observation_limit=220),
                'accepted': fault_receipt['native_accepted'],
                'done': fault_receipt['done'],
                'invalid_detail': fault_receipt.get('invalid_detail'),
            }),
            'restored_public_state': checkpoint_snapshot(self.episode.snapshot(), observation_limit=260),
            'discarded_failed_suffix': suffix[:8],
            'reflection_is_fallible': True,
            'do_not_repeat_without_new_evidence': [
                {'action': row['action'], 'before': row['before']}
                for row in suffix if row['accepted'] and not row['done']
            ][:4],
            'reflection_evidence_source': 'Red author reflection plus public receipts',
            'reflection_digest': reflection[:600],
        }

    def _red_actor_context(self):
        """Keep one stable, public recovery note across stateless Actor calls."""
        raw = self.active_red_memory.actor_context()
        facts = raw.get('controller_recovery_facts', raw.get('recovery_facts', {}))
        lesson = raw.get('historical_failure_lesson', raw.get('reflection', ''))
        checkpoint = raw.get('lesson_origin_checkpoint',
                             raw.get('restored_checkpoint_index', 0))
        pending = raw.get('pending_instruction')
        state = facts.get('current_public_state', {})
        public_state = {
            key: state[key] for key in (
                'page_type', 'query', 'page_num', 'asin', 'selected_options',
                'option_types', 'available_actions', 'public_terminal_score')
            if key in state
        }
        failed_edges = facts.get('failed_action_edges', [])
        return {
            'schema': 'webshop_compact_recovery_context_v1',
            'lesson': lesson[:480],
            'checkpoint_index': checkpoint,
            'pending_instruction': pending,
            'restored_public_state': public_state,
            'failed_actions': [row['action'] for row in failed_edges[:4]],
            'required_next_evidence': facts.get('required_next_evidence'),
            'semantics': ('This is one recovery handoff for the current trial. '
                          'Use the current page and native action menu as authority. '
                          'Failed actions are history, not commands.'),
        }

    def _reflect_and_restore(self):
        super()._reflect_and_restore()
        self.episode.recovery_prefix_length = self.selected_fault_index
        self.emit('actor_recovery_context_rebased', checkpoint_index=self.selected_fault_index,
                  actual_state=self.episode.snapshot(), ett_discarded=False)

    def _yellow_feedback(self, scope):
        if self.settings.ablation == 'without_event_subtree':
            incident = self.simple_yellow_event
            return dict(original_instruction=self.episode.instruction,
                current_public_page=self.episode.snapshot(),
                incident_kind=incident['kind'], pending_green_proposal=self.simple_yellow_proposal,
                invalid_detail=incident.get('invalid_detail'),
                local_receipts=[dict(action=self.occurrences[i].action,
                    observation=self.occurrences[i].receipt['observation'],
                    accepted=self.occurrences[i].receipt['native_accepted']) for i in scope.occurrence_ids],
                actions_remaining=min(self.settings.yellow_actions-scope.action_serial,
                                      self._forward_window()['new_actions_remaining']))
        return dict(original_instruction=self.episode.instruction,
            incident=self.simple_yellow_event, pending_green_proposal=self.simple_yellow_proposal,
            current_public_page=self.episode.snapshot(),
            local_receipts=[dict(occurrence_id=i, action=self.occurrences[i].action,
                observation=self.occurrences[i].receipt['observation'],
                accepted=self.occurrences[i].receipt['native_accepted']) for i in scope.occurrence_ids],
            actions_remaining=min(self.settings.yellow_actions-scope.action_serial,
                                  self._forward_window()['new_actions_remaining']),
            semantics='Literal page evidence only. Invalid action does not prove a semantic mismatch. '
                'An accepted click does not prove shopping success. Historical advice is unverified.')

    def _bind_scope(self, scope, directive):
        if scope.kind is ScopeKind.GREEN:
            view = (self._plain_recovery_notes() if self.settings.ablation == 'without_event_subtree'
                    else green_events(self))
        elif scope.kind is ScopeKind.YELLOW:
            heading = ('[LOCAL_FEEDBACK]' if self.settings.ablation == 'without_event_subtree'
                       else '[YELLOW_EVENT_SUBTREE]')
            view = heading + '\n' + canonical(self._yellow_feedback(scope))
            diagnosis = self.yellow_diagnoses[scope.scope_id]
            diagnosis_text = diagnosis['text'] or '[NO_DIAGNOSIS_AVAILABLE; use receipts and native menu only]'
            view += '\n[LOCAL_DIAGNOSIS_MODEL_ADVICE_NOT_VERIFIED]\n' + diagnosis_text
            page = self.episode.snapshot()['page_type']
            if page == 'init':
                view += ('\n[CURRENT_NATIVE_ACTION_INTERFACE]\n'
                    'This is the initial search form. Its native action is search[query]. '
                    'Return one concrete search[query] action. Search is a visible UI label, '
                    'not click[Search]; click[Back to Search], click[Next >], and other click[...] '
                    'actions are not executable on this page.')
            else:
                view += ('\n[CURRENT_NATIVE_ACTION_INTERFACE]\n'
                    'Choose exactly one command from the current page native action menu.')
            view += ('\nRepair this local page/command obstruction with exactly one testable native action. '
                'The receipt will decide whether the local repair made progress; do not plan a search sequence. '
                'Do not repeat an ineffective check. '
                'Do not invent product attributes or treat a partial score as a per-attribute explanation. '
                'Green owns the shopping plan. Return exactly one native action.')
            if directive:
                view += '\n[CURRENT_YELLOW_FEEDBACK]\n' + directive
        else:
            raise ProtocolError('ordinary prompt in Red')
        self.episode.set_scope_prompt(view, scope.tree.tree_version, kind=scope.kind.value, directive=directive)
        return self.episode.prompt(())

    def _model_request_submitted(self, request):
        if self.active_scope.kind is ScopeKind.GREEN and request.route != 'reflection':
            if self.pending_green_events:
                self.delivered_green_events.update(self.pending_green_events)
                self.emit('plain_recovery_note_delivered' if self.settings.ablation == 'without_event_subtree'
                          else 'web_event_subtree_delivered', request_id=request.request_id,
                          event_keys=list(self.pending_green_events))
            self.pending_green_events = []
        super()._model_request_submitted(request)

    def _ordinary(self, scope, directive):
        if scope.kind is ScopeKind.GREEN and self.active_red_memory is not None:
            address = self._address(scope)
            if self.red_handoff_directive is not None:
                self.emit('red_green_handoff_delivered', directive=self.red_handoff_directive)
                self.red_handoff_directive = None
            self.episode.red_repair_handoff = None
            prompt = self._bind_scope(scope, directive)
            try:
                parsed, request_id = self.oracle.ordinary(
                    self.episode, address, (), self.ledger)
            finally:
                self.episode.red_repair_handoff = None
            self.emit('scoped_actor_action_selected', scope_id=scope.scope_id,
                      scope_kind=scope.kind.value, request_id=request_id,
                      action=parsed.action, prompt=prompt,
                      history_window=getattr(self.episode, 'last_history_window', None))
            return parsed, request_id
        if scope.kind is not ScopeKind.YELLOW:
            return super()._ordinary(scope, directive)
        state = self.episode.checkpoint()
        if scope.scope_id not in self.yellow_diagnoses:
            prompt = ('Diagnose this ONE WebShop local obstruction from public page receipts. '
                'Propose a concise testable next operation, its supporting fact and what receipt '
                'would confirm progress. At most two short sentences, no fixed schema. '
                'Do not invent hidden product attributes, unavailable controls or failure causes. '
                'Use the literal shopping instruction. Repeating a known page without new evidence '
                'does not repair a loop. Green remains responsible for the whole shopping task.\n\n'
                + ('[LOCAL_FEEDBACK]\n' if self.settings.ablation == 'without_event_subtree'
                   else '[LOCAL_EVENT_SUBTREE]\n')
                + canonical(self._yellow_feedback(scope)) + '\nLocal diagnosis:')
            address = self._address(scope)
            self.oracle.bind(address, self.episode, prompt, ())
            generation = self.oracle.request(address, Role.REFLECTION, 0, prompt, self.ledger,
                                             output_limit=self.settings.yellow_diagnosis_tokens, stop_mode='first_nonempty_line')
            if not generation.text.strip():
                self._record_yellow_protocol_failure(
                    scope, phase='DIAGNOSIS', request_id=generation.request.request_id,
                    raw_output=generation.text, error=ProtocolError('empty Yellow diagnosis'))
                self.yellow_diagnoses[scope.scope_id] = dict(
                    text='', request_id=generation.request.request_id,
                    status='EMPTY_DIAGNOSIS_PROTOCOL_EVENT')
            else:
                self.yellow_diagnoses[scope.scope_id] = dict(
                    text=generation.text, request_id=generation.request.request_id,
                    status='MODEL_DIAGNOSIS')
            self.emit('yellow_local_diagnosis', scope_id=scope.scope_id, prompt=prompt,
                      diagnosis=self.yellow_diagnoses[scope.scope_id])
        # Yellow receives only its local public incident. Red's recovery lesson
        # is delivered by the one-shot RED_HANDOFF event to the next Green call.
        self.episode.red_repair_handoff = None
        prompt = self._bind_scope(scope, directive)
        address = self._address(scope)
        self.oracle.bind(address, self.episode, prompt, ())
        support = self.episode.support()
        if support.kind is SupportKind.FINITE:
            if not support.commands:
                raise ProtocolError('Yellow has no visible callable command')
            generation = self.oracle.request(address, Role.ALTERNATIVE, 0, prompt, self.ledger,
                                             commands=support.commands, output_limit=self.settings.yellow_output_tokens)
        else:
            generation = self.oracle.request(address, Role.ALTERNATIVE, 0, prompt, self.ledger,
                                             output_limit=self.settings.yellow_output_tokens, stop_mode='first_line')
        try:
            parsed = self.episode.parse(generation.text)
            if not self.episode.public_physical(parsed, support):
                raise ProtocolError('Yellow action is not callable on the current public interface')
        except ProtocolError as error:
            self._record_yellow_protocol_failure(
                scope, phase='ACTION_SELECTION', request_id=generation.request.request_id,
                raw_output=generation.text, error=error,
                parser=getattr(generation, 'raw_response', None))
            self.emit('yellow_unresolved_protocol', scope_id=scope.scope_id,
                      last_output=generation.text, attempts=1,
                      disposition='RETURN_TO_GREEN_WITHOUT_ENVIRONMENT_ACTION')
            self.episode.red_repair_handoff = None
            return None, generation.request.request_id

        if self.episode.checkpoint() != state:
            raise ProtocolError('Yellow model request mutated environment')
        self.episode.red_repair_handoff = None
        self.emit('scoped_actor_action_selected', scope_id=scope.scope_id, scope_kind='YELLOW',
                  request_id=generation.request.request_id, action=parsed.action, prompt=prompt)
        return parsed, generation.request.request_id

    def _yellow_receipt_outcome(self, incident, receipt, earlier_receipts):
        if not receipt['native_accepted']:
            return None
        if incident['kind'] == 'ENVIRONMENT_REJECTION' and receipt['action'] == incident['action']:
            return 'ORIGINAL_OPERATION_ACCEPTED'
        if not receipt['native_accepted']:
            return 'UNRESOLVED_REJECTED_REPAIR'
        before, after = receipt['before'], receipt['after']
        kind = progress_kind(before, after, receipt['action'],
                             accepted=True, done=receipt['done'])
        # Navigation alone is not a local repair. The next Green proposal may
        # navigate when needed, but Yellow must not turn it into a search loop.
        if receipt['action'] in {'click[Back to Search]', 'click[< Prev]'}:
            return 'UNRESOLVED_NAVIGATION_ONLY'
        if kind in {'NO_PUBLIC_CHANGE', 'TEXT_CHANGED', 'SUBPAGE_VIEWED'}:
            return 'UNRESOLVED_NO_PUBLIC_PROGRESS'
        if new_evidence(receipt, earlier_receipts, self.episode.initial):
            return 'NEW_PUBLIC_EVIDENCE_REPLAN'
        return 'UNRESOLVED_NO_PUBLIC_PROGRESS'

    def _local_repair(self, incident, proposal):
        """Default: one action/receipt; larger explicit budgets retain feedback."""
        scope = self._enter_yellow(incident['kind'], proposal,
                                   incident_id=incident['incident_id'])
        scope.incident_kind = incident['kind']
        scope.exit_rules = ('ONE_CALLABLE_ACTION_TO_GREEN' if self.settings.yellow_actions == 1
                           else 'CONFIGURED_LOCAL_ACTION_BUDGET_TO_GREEN', 'REAL_TERMINAL',
                            'SHARED_WINDOW_END')
        self.simple_yellow_event = deepcopy(incident)
        self.simple_yellow_proposal = proposal
        self.episode.red_repair_handoff = None
        outcome = 'UNRESOLVED_YELLOW_PROTOCOL'
        request_id = None
        occurrence = None
        attempted_actions = []
        try:
            for _ in range(self.settings.yellow_actions):
                if not self._forward_window()['new_actions_remaining']:
                    break
                parsed, request_id = self._ordinary(
                    scope, 'Address the local obstruction with one callable action.')
                if parsed is None:
                    return
                if not parsed.grammar_valid:
                    raise ProtocolError('Yellow returned no native action')
                if (incident['kind'].startswith('REPEATED')
                        and parsed.action == incident.get('action')):
                    outcome = 'UNRESOLVED_REPEATED_INCIDENT_EDGE'
                    break
                earlier_receipts = list(self.episode.history)
                receipt, occurrence = self._execute(scope, parsed, request_id)
                attempted_actions.append(occurrence.action)
                outcome = ('ENVIRONMENT_TERMINAL' if receipt['done']
                           else self._yellow_receipt_outcome(
                               incident, receipt, earlier_receipts))
                if receipt['done'] or outcome in ('NEW_PUBLIC_EVIDENCE_REPLAN','ORIGINAL_OPERATION_ACCEPTED'):
                    break
        finally:
            record = dict(request_id=request_id, scope_id=scope.scope_id,
                conclusion={'outcome': outcome, 'original_obstruction': deepcopy(incident),
                            'pending_green_proposal': proposal,
                            'attempted_actions': attempted_actions,
                            'local_diagnosis': deepcopy(self.yellow_diagnoses.get(scope.scope_id))},
                step_role='RETURN_GREEN_UNRESOLVED' if occurrence is None else 'LOCAL_REPAIR',
                selected_action=None if occurrence is None else occurrence.action,
                executed_occurrence_id=None if occurrence is None else occurrence.occurrence_id,
                correction_completed=outcome == 'NEW_PUBLIC_EVIDENCE_REPLAN')
            scope.correction_record = deepcopy(record)
            scope.correction_records.append(deepcopy(record))
            self.latest_yellow_correction = deepcopy(record)
            self.local_outcomes.append(dict(scope_id=scope.scope_id,
                outcome=outcome, actions=len(attempted_actions)))
            self._return_to_green(scope, outcome,
                handoff_directive=('Use the actual Yellow receipt. Yellow performed at most '
                    + ('one local action' if self.settings.yellow_actions == 1 else
                       str(self.settings.yellow_actions)+' local actions')
                    + '; it did not certify the whole shopping task. Continue the '
                    'original task and do not replay the same state-action edge.'))
            self.simple_yellow_event = None
            self.simple_yellow_proposal = None

    def _return_to_green(self, scope, reason, handoff_directive=None):
        diagnosis = deepcopy(self.yellow_diagnoses.get(scope.scope_id,
            dict(text='', status='NO_DIAGNOSIS_RECORD')))
        records = [scope.correction_record, *scope.correction_records,
                   self.latest_yellow_correction]
        seen = set()
        for record in records:
            if record is None or id(record) in seen:
                continue
            seen.add(id(record))
            record['conclusion']['local_diagnosis'] = diagnosis
        incident = self.simple_yellow_event
        if (incident is not None and is_navigation_action(incident.get('action', ''))
                and not self.episode.snapshot()['done']):
            receipt = incident.get('receipt') or {}
            before = receipt.get('before') or incident.get('interface_state') or {}
            support = self.episode.support()
            legal_actions = (list(support.commands) if support.kind is SupportKind.FINITE
                             else ['search[<query>]'] if self.episode.snapshot().get('page_type') == 'init'
                             else [])
            self.web_yellow_navigation_handoff = {
                'event_key': canonical(['YELLOW_NAVIGATION_HANDOFF', scope.scope_id,
                                        incident.get('incident_id')]),
                'kind': 'YELLOW_NAVIGATION_HANDOFF',
                'repeated_navigation_edges': [{
                    'action': incident['action'],
                    'before': {key: before[key] for key in (
                        'page_type', 'query', 'page_num', 'asin', 'selected_options',
                        'subpage', 'available_actions') if key in before},
                    'accepted': receipt.get('native_accepted'),
                    'invalid_detail': receipt.get('invalid_detail'),
                }],
                'current_legal_actions': legal_actions,
                'current_action_kind': support.kind.value,
                'actions_remaining': self._forward_window()['new_actions_remaining'],
                'semantics': ('Yellow handled '+('one local action' if scope.action_serial == 1
                              else str(scope.action_serial)+' local actions')+' and returned to Green. '
                              'The listed navigation edge is historical evidence, not a '
                              'command. Choose from the current public interface.'),
            }
        super()._return_to_green(scope, reason, handoff_directive)

    def to_dict(self):
        return dict(super().to_dict(), controller_version=VERSION,
            benchmark='webshop', first_window_actions=self.settings.initial_actions,
            recovery_window_actions=self.settings.recovery_actions,
            yellow_actions_per_entry=self.settings.yellow_actions, yellow_entries_per_window=self.YELLOW_ENTRY_LIMIT,
            event_subtree_enabled=self.settings.ablation != 'without_event_subtree',
            max_red_model_calls=0 if self.settings.ablation == 'without_red' else 2,
            max_recovery_actions=0 if self.settings.ablation == 'without_red' else self.RECOVERY_WINDOW,
            experiment_settings=self.settings.as_dict(),
            yellow_diagnoses=deepcopy(self.yellow_diagnoses),
            yellow_protocol_failures=deepcopy(self.yellow_protocol_failures),
            red_phase1_evidence=deepcopy(self.red_phase1_evidence),
            recovery_prefix_length=self.episode.recovery_prefix_length,
            red_prompt_source='vendor/webshop_reflection_prompt.py:_generate_reflection_query')
