"""Bounded GREEN/YELLOW/RED recovery scopes over real WebShop execution."""

from __future__ import annotations

from copy import deepcopy
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import re
from typing import Any, Callable

from icr_core import (BudgetExhausted, Ledger, ProtocolError, REFLECTION_OUTPUT_TOKEN_LIMIT,
                      Role, SupportKind, environment_call)
from icr_identity import NodeAddress, Phase
from tree_history import ExecutionHistoryTree, canonical, columnar_graph
from scoped_review_session import ForwardTrial, ReviewSession
from scoped_repetition import RepetitionEvidence, operation_key
from scoped_prompt_efficiency import parse_red_choice, red_batch_prompt, parse_yellow_choice, YellowChoiceError
from scoped_prompt_efficiency import parse_red_reason_first, red_candidate_evidence
from scoped_prompt_efficiency import parse_red_selection, red_selection_prompt, RedSelectionError
from icr_output_framing import frame_complete
from green_decision_view import render_green_view
from alf_action_selection import YellowActionIdRangeError
from alf_red_repair_policy import task_related_progress


class YellowCallabilityError(YellowChoiceError):
    """A proposal violates a complete public interface; never execute it."""


class ScopeKind(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


class NodeKind(str, Enum):
    ENTRY = "ENTRY"
    ACTION = "ACTION"
    RECEIPT = "RECEIPT"
    CALL = "CALL"
    RETURN = "RETURN"
    ESCALATE = "ESCALATE"
    TERMINAL = "TERMINAL"


class ReviewDecision(str, Enum):
    RESUME = "RESUME"
    BACK = "BACK"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ExecutionOccurrence:
    occurrence_id: int
    execution_parent_occurrence_id: int | None
    ordinal: int
    action: str
    receipt: dict[str, Any]
    scope_id: int
    scope_kind: ScopeKind
    scope_node_id: int | None
    request_id: str

    def to_dict(self):
        return {
            "occurrence_id": self.occurrence_id,
            "execution_parent_occurrence_id": self.execution_parent_occurrence_id,
            "ordinal": self.ordinal,
            "action": self.action,
            "receipt": deepcopy(self.receipt),
            "scope_id": self.scope_id,
            "scope_kind": self.scope_kind.value,
            "scope_node_id": self.scope_node_id,
            "request_id": self.request_id,
        }


@dataclass
class ScopeTree:
    scope_id: int
    kind: ScopeKind
    entry_reason: str
    entry_state: dict[str, Any]
    caller_scope_id: int | None
    caller_node_id: int | None
    exit_rules: tuple[str, ...]
    tree: ExecutionHistoryTree = field(init=False)
    request_serial: int = 0
    action_serial: int = 0
    archived: bool = False
    incident_kind: str | None = None
    occurrence_ids: list[int] = field(default_factory=list)
    correction_record: dict[str, Any] | None = None
    correction_records: list[dict[str, Any]] = field(default_factory=list)
    event_anchor_occurrence_id: int | None = None
    exit_reason: str | None = None

    def __post_init__(self):
        self.tree = ExecutionHistoryTree(self.entry_state)
        self.tree.begin_attempt(f"scope-{self.scope_id}", restored=False)

    @property
    def current_node_id(self):
        return self.tree.current_id

    def add_receipt(self, receipt: dict[str, Any], *, attempt: int,
                    request_id: str | None) -> int:
        child_id = self.tree.add_edge(
            self.tree.current_id, receipt["action"], receipt,
            visit_kind="SCOPED_EXECUTION", attempt=attempt,
            request_id=request_id)
        self.action_serial += 1
        return child_id

    def view(self, *, compact=False, columnar=False) -> str:
        rules = ",".join(self.exit_rules)
        return (f"[SCOPE_TREE id={self.scope_id} kind={self.kind.value}]\n"
                f"[ENTRY_REASON] {self.entry_reason}\n"
                f"[CALLER] scope={self.caller_scope_id} node={self.caller_node_id}\n"
                f"[EXIT_RULES] {rules}\n"
                + self.tree.render_graph(scope_id=self.scope_id, scope_kind=self.kind.value, compact=compact, columnar=columnar))


@dataclass
class ReviewRecord:
    scope_id: int
    checkpoint_index: int
    decision: ReviewDecision
    request_id: str
    prompt: str
    output: str


@dataclass
class CallFrame:
    caller_scope_id: int
    caller_node_id: int | None
    callee_scope_id: int
    incident_id: str
    entry_state: dict[str, Any]
    return_policy: str
    remaining_budget: dict[str, int]


@dataclass(frozen=True)
class PendingIncident:
    incident_id: str
    result_key: str
    action: str


class ScopedRecoveryController:
    """Controller for the three logical scopes; all states come from real receipts."""

    YELLOW_REQUEST_LIMIT = None
    YELLOW_ENTRY_LIMIT = 3
    RED_ENTRY_LIMIT = 3
    RED_REVIEW_LIMIT = None
    def __init__(self, episode, oracle, ledger: Ledger, *, task: int,
                 emit: Callable[..., None], yellow_request_limit=YELLOW_REQUEST_LIMIT,
                 red_review_limit=RED_REVIEW_LIMIT, yellow_entry_limit=3,
                 red_entry_limit=3, max_root_recoveries=2,
                 compact_context=False, batched_red=False, enable_yellow=True,
                 enable_red=True, terminal_guidance=False, yellow_action_ids=False, output_framing=False,
                 progress_recovery=False, yellow_action_ban=False, joint_red_selection=False, typed_scope_choices=False,
                 efficient_recovery=False, tree_internal_repair=False, alfworld_event_memory=False,
                 single_reflection_red=False):
        if type(task) is not int or not 0 <= task < 500:
            raise ProtocolError("scoped recovery task is outside the fixed task set")
        if yellow_request_limit is not None:
            raise ProtocolError("Yellow shares the action window and has no local request limit")
        if red_review_limit is not None:
            raise ProtocolError("Red uses finite path cursors, not an arbitrary review limit")
        if type(max_root_recoveries) is not int or not 0 <= max_root_recoveries <= 2:
            raise ProtocolError("invalid root recovery limit")
        if type(single_reflection_red) is not bool or single_reflection_red and batched_red:
            raise ProtocolError('single-reflection and batched Red are distinct protocols')
        if enable_red and not batched_red and not single_reflection_red:
            raise ProtocolError("Red requires the structured batched repair protocol")
        if (type(yellow_entry_limit) is not int or not 0 <= yellow_entry_limit <= 3
                or type(red_entry_limit) is not int or not 0 <= red_entry_limit <= 3):
            raise ProtocolError("Yellow and Red entry limits must be integers in 0..3")
        if any(type(v) is not bool for v in (compact_context, batched_red, enable_yellow,
                                             enable_red, terminal_guidance, yellow_action_ids, output_framing, progress_recovery,
                                             yellow_action_ban, joint_red_selection, typed_scope_choices)):
            raise ProtocolError("controller switches must be explicit booleans")
        if progress_recovery and not yellow_action_ids:
            raise ProtocolError("progress recovery requires the single-action JSON Yellow protocol")
        if (yellow_action_ban or joint_red_selection) and not progress_recovery:
            raise ProtocolError("V37 switches require the explicit progress recovery contract")
        if joint_red_selection and batched_red:
            raise ProtocolError("joint single selection and legacy batched verdicts are distinct modes")
        if typed_scope_choices and not (progress_recovery and yellow_action_ban and joint_red_selection):
            raise ProtocolError('typed scope choices require the explicit ban/joint policy')
        self.episode = episode
        if type(alfworld_event_memory) is not bool:
            raise ProtocolError('ALFWorld event memory must be an explicit boolean')
        if alfworld_event_memory and (progress_recovery or tree_internal_repair or not yellow_action_ids):
            raise ProtocolError('ALFWorld event memory requires the r5.2 JSON Yellow protocol')
        self.alfworld_event_memory = alfworld_event_memory
        self.oracle = oracle
        self.ledger = ledger
        self.task = task
        self.emit = emit
        self.episode.prompt_token_counter = oracle.token_counter
        self.episode.prompt_context_limit = oracle.context_limit
        self.episode.prompt_service_limit = oracle.prompt_limit
        self.yellow_request_limit = yellow_request_limit
        self.red_review_limit = red_review_limit
        self.max_root_recoveries = 0
        self.YELLOW_ENTRY_LIMIT = yellow_entry_limit
        self.RED_ENTRY_LIMIT = red_entry_limit
        self.compact_context = compact_context
        self.episode.compact_context = compact_context
        self.batched_red = batched_red
        self.enable_yellow, self.enable_red = enable_yellow, enable_red
        self.terminal_guidance = terminal_guidance
        self.yellow_action_ids = yellow_action_ids
        self.output_framing = output_framing
        self.progress_recovery = progress_recovery
        self.episode.progress_recovery = progress_recovery
        self.yellow_action_ban = yellow_action_ban
        self.joint_red_selection = joint_red_selection
        self.typed_scope_choices = typed_scope_choices
        if type(efficient_recovery) is not bool or (efficient_recovery and not typed_scope_choices):
            raise ProtocolError('efficient recovery requires typed scopes')
        self.efficient_recovery = efficient_recovery
        self.episode.efficient_recovery = efficient_recovery
        if type(tree_internal_repair) is not bool or (tree_internal_repair and not efficient_recovery):
            raise ProtocolError('internal tree repair requires efficient typed recovery')
        self.tree_internal_repair = tree_internal_repair
        self.episode.tree_internal_repair = tree_internal_repair
        self.yellow_local_decisions = {}
        self.pending_recovery_yellow = None
        self.episode.yellow_action_ban = yellow_action_ban
        self.episode.yellow_forbidden_actions = ()
        self.yellow_bans = {}
        self.red_joint_choices = {}
        self.yellow_suppressed = set()
        self.yellow_invalid_proposal = False
        self.yellow_callability_refund_used = False
        self.reviewed_failure_paths = set()
        self.yellow_local_action_limit = None
        self.episode.ordinary_stop_mode = "first_line" if output_framing else None
        self.episode.terminal_guidance = terminal_guidance
        self.episode.yellow_action_ids = yellow_action_ids
        self.red_batches = {}
        self.scopes: dict[int, ScopeTree] = {}
        self.active_scope_id: int | None = None
        self.next_scope_id = 0
        self.lineage: list[ExecutionOccurrence] = []
        self.archived_lineages: list[list[ExecutionOccurrence]] = []
        self.call_frames: list[CallFrame] = []
        self.reviews: list[ReviewRecord] = []
        self.controller_events: list[dict[str, Any]] = []
        self.terminal_evidence: dict[str, list[dict[str, Any]]] = {}
        self.root_checkpoint: dict[str, Any] | None = None
        self.root_recoveries = 0
        self.incident_serial = 0
        self.yellow_entry_counts: dict[str, int] = {}
        self.attempt_scope_entries = {"RED": 0, "YELLOW": 0}
        self.occurrences: dict[int, ExecutionOccurrence] = {}
        self.checkpoint_cache: dict[int, dict[str, Any]] = {}
        self.tip_checkpoint: dict[str, Any] | None = None
        self.checkpoint_inspection_count = 0
        self.cache_branch_id = 0
        self.repetition = RepetitionEvidence()
        self.review_sessions: dict[int, ReviewSession] = {}
        self.review_queue = deque()
        self.failure_serial = 0
        self.active_review_session_id: int | None = None
        self.forward_trials: dict[int, ForwardTrial] = {}
        self.active_trial_id: int | None = None
        self.address_serial = 0
        self.global_action_serial = 0
        self.checkpoint_restore_count = 0
        self.green_handoff_directive: str | None = None
        self.green_history_handoff = None
        self.delivered_green_events = set()
        self.delivered_yellow_corrections = set()
        self.pending_green_events = []
        self.latest_yellow_correction: dict[str, Any] | None = None
        self.pending_yellow_action: dict[str, Any] | None = None
        self.yellow_continuation: dict[str, Any] | None = None
        self.pending_red_action: dict[str, Any] | None = None
        self.active_red_repair: dict[str, Any] | None = None
        self.last_yellow_unresolved = False
        self.red_handoff_directive: str | None = None
        self.active_red_memory = None
        # Public negative evidence is branch-local: a rejected command is not
        # a permanent ban, but it must not be proposed again while the same
        # public state and target facts remain unchanged.
        self.rejected_public_actions = {}
        self.status = "NOT_STARTED"
        self.resource_limit: str | None = None
        if self.oracle.on_request_submitted is not None:
            raise ProtocolError("scoped controller requires its own proposal oracle")
        self.oracle.on_request_submitted = self._model_request_submitted

    @property
    def active_scope(self) -> ScopeTree:
        if self.active_scope_id is None or self.active_scope_id not in self.scopes:
            raise ProtocolError("no active scoped recovery tree")
        return self.scopes[self.active_scope_id]

    def _new_scope(self, kind: ScopeKind, reason: str, *, caller=None) -> ScopeTree:
        if kind is ScopeKind.GREEN:
            rules = (("TERMINAL_SUCCESS",)
                     + (("LOCAL_INCIDENT_TO_YELLOW",) if self.enable_yellow else ())
                     + (("WINDOW_EXHAUSTED_TO_RED",) if self.enable_red else ("WINDOW_EXHAUSTED_TO_ROOT",))
                     + (("NON_SUCCESS_TERMINAL_TO_RED",) if self.enable_red else ("NON_SUCCESS_TERMINAL_TO_ROOT",)))
        elif kind is ScopeKind.YELLOW:
            rules = ("VALID_CORRECT_ACTION_TO_GREEN", "PREPARE_ACTION_STAYS_YELLOW",
                     ("NON_SUCCESS_TERMINAL_TO_RED" if self.enable_red else "NON_SUCCESS_TERMINAL_TO_ROOT"),
                     "ACTION_WINDOW_EXHAUSTED_TO_RED" if self.enable_red else "ACTION_WINDOW_EXHAUSTED_TO_ROOT")
            if self.progress_recovery:
                rules = ("LOCAL_REPAIR_TO_GREEN", "UNRESOLVED_OR_LOCAL_BUDGET_TO_GREEN",
                         "REAL_TERMINAL_TO_TERMINAL_HANDLER")
            if self.yellow_action_ban:
                rules = ("DIFFERENT_CALLABLE_ACCEPTED_ACTION_TO_GREEN", "REAL_TERMINAL_TO_TERMINAL_HANDLER")
        else:
            rules = ("RESUME_TO_GREEN", "BACK_TO_EARLIER_CHECKPOINT",
                     "NO_POSITION_TO_ROOT_OR_STOP")
        if self.alfworld_event_memory and kind is ScopeKind.YELLOW:
            rules = ('ONE_LOCAL_ACTION_THEN_GREEN_WITH_EVENT_EVIDENCE',
                     'REAL_TERMINAL_TO_TERMINAL_HANDLER',
                     'ACTION_WINDOW_EXHAUSTED_TO_RED')
        scope = ScopeTree(
            scope_id=self.next_scope_id, kind=kind, entry_reason=reason,
            entry_state=self.episode.snapshot(),
            caller_scope_id=None if caller is None else caller.scope_id,
            caller_node_id=None if caller is None else caller.current_node_id,
            exit_rules=rules)
        self.scopes[scope.scope_id] = scope
        self.next_scope_id += 1
        self.active_scope_id = scope.scope_id
        self.emit("scope_entered", scope_id=scope.scope_id, kind=kind.value,
                  reason=reason, caller_scope_id=scope.caller_scope_id,
                  caller_node_id=scope.caller_node_id)
        return scope

    def start(self) -> dict[str, Any]:
        initial = environment_call(self.episode, self.ledger, self.episode.reset)
        if self.alfworld_event_memory and self.episode.snapshot()['benchmark'] != 'alfworld':
            raise ProtocolError('ALFWorld event-memory policy used with a different environment')
        if initial["done"]:
            raise ProtocolError("scoped recovery cannot start from terminal reset")
        if self.alfworld_event_memory:
            self.repetition.record_initial(initial)
        self.root_checkpoint = self.episode.checkpoint()
        self.tip_checkpoint = deepcopy(self.root_checkpoint)
        self._new_scope(ScopeKind.GREEN, "TASK_START")
        self.status = "ACTIVE"
        self.emit("scoped_recovery_started", initial_receipt=initial)
        return initial

    def _address(self, scope: ScopeTree) -> NodeAddress:
        serial = self.address_serial
        if serial >= 8192:
            raise ProtocolError("model request address space exhausted")
        self.address_serial += 1
        scope.request_serial += 1
        # Keep model request addresses independent of logical scope lifetime.
        # The existing mixed-radix contract has 16 prefixes * 8 replicas * 64
        # native ordinals, enough for this fixed pilot without reusing a bound
        # address after a Red resume.
        native_ordinal = serial % 64
        lane = serial // 64
        replica = lane % 8
        group = lane // 8
        bits = tuple((group >> shift) & 1 for shift in (3, 2, 1, 0))
        return NodeAddress(0, self.task, Phase.TREE, replica, bits,
                           native_ordinal)

    def _bind_scope(self, scope: ScopeTree, directive: str) -> str:
        if self.alfworld_event_memory:
            if scope.kind is ScopeKind.GREEN:
                from alf_sparse_events import render_sparse_green
                view=render_sparse_green(self)
            else:
                from alf_recovery_projection import recovery_subtree
                self.episode.recovery_event_subtree=recovery_subtree(self)
                view=canonical({'scope_kind':scope.kind.value,'entry_reason':scope.entry_reason,
                                'incident_kind':scope.incident_kind})
            self.episode.set_scope_prompt(view,scope.tree.tree_version,kind=scope.kind.value,
                                          directive=directive,incident_kind=scope.incident_kind)
            return self.episode.prompt(())
        root_recovery_tree = (scope.kind is ScopeKind.GREEN and self.root_recoveries > 0
                              and self.active_trial_id is None)
        recovery_context = (scope.kind is not ScopeKind.GREEN or self.root_recoveries > 0
                            or self.latest_yellow_correction is not None or self.active_trial_id is not None)
        compact = self.compact_context and recovery_context
        self.episode.compact_context = compact
        self.episode.terminal_guidance = self.terminal_guidance
        view = (render_green_view(self, scope) if scope.kind is ScopeKind.GREEN else
                self._clean_root_recovery_tree() if root_recovery_tree
                else scope.view(compact=compact, columnar=self.efficient_recovery) + "\n[FORWARD_WINDOW] " + canonical(self._forward_window()))
        if self.efficient_recovery and root_recovery_tree:
            view += '\n[FORWARD_WINDOW] ' + canonical(self._forward_window())
            view += '\n[COMPLETION_BUDGET] ' + canonical({
                'actions_remaining': self._forward_window()['new_actions_remaining'],
                'root_retries_remaining': self.max_root_recoveries - self.root_recoveries})
        if self.episode.terminal_guidance and not root_recovery_tree:
            view += "\n[COMPLETION_BUDGET] " + canonical({
                "actions_remaining": self._forward_window()["new_actions_remaining"],
                "root_retries_remaining": self.max_root_recoveries - self.root_recoveries})
        if self.compact_context and scope.kind is ScopeKind.YELLOW and not root_recovery_tree:
            view += "\n[YELLOW_PROGRESS] " + canonical({
                "actions_taken": scope.action_serial,
                "prepare_decisions": sum(r["step_role"] == "PREPARE" for r in scope.correction_records),
                "correct_decisions": sum(r["step_role"] == "CORRECT" for r in scope.correction_records)})
        if self.progress_recovery and scope.kind is ScopeKind.YELLOW:
            view += "\n[YELLOW_LOCAL_BUDGET] " + canonical({
                "action_limit": self.yellow_local_action_limit, "actions_used": scope.action_serial,
                "remaining_shared_actions": self.episode.action_limit - len(self.lineage),
                "on_limit": "RETURN_GREEN_UNRESOLVED", "callability_resubmissions": 1})
        if self.yellow_action_ban and scope.kind is ScopeKind.YELLOW:
            view += "\n[YELLOW_FORBIDDEN_ACTIONS] " + canonical(self.yellow_bans[scope.scope_id])
        correction_ready = (
            self.latest_yellow_correction is not None
            and self.latest_yellow_correction["request_id"]
            not in self.delivered_yellow_corrections)
        if (correction_ready and not root_recovery_tree
                and scope.kind is ScopeKind.GREEN
                and self.green_history_handoff is not None):
            record = self.latest_yellow_correction
            memory = {key: record[key] for key in ("scope_id", "incident_id", "request_id", "conclusion",
                                                  "step_role", "executed_occurrence_id", "correction_completed")}
            memory["at_state"] = deepcopy(record["decision_state"])
            view += ("\n[YELLOW_CORRECTION_MEMORY model_advice_not_verified_fact]\n"
                     + json.dumps(memory, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                     + "\nReassess this historical model conclusion against the current public state. "
                     "Use its evidence and proposed change to guide the original task; "
                     "do not replay advice blindly or treat it as a forced action. "
                     "Executed actions are historical evidence, not pending commands. "
                     "Choose from the current observation; no earlier target string is binding.")
        if (self.active_trial_id is not None and scope.kind in {ScopeKind.GREEN, ScopeKind.YELLOW}
                and not (self.alfworld_event_memory and scope.kind is ScopeKind.GREEN)):
            trial = self.forward_trials[self.active_trial_id]
            view += ("\n[ACTIVE_RECOVERY_INTENT model_advice_not_verified_fact]\n"
                     + canonical({"trial_id": trial.trial_id,
                                  "checkpoint_index": trial.checkpoint_index,
                                  "advice": trial.advice})
                     + "\nThe checkpoint has already been restored. Any executed replacement is history, "
                       "not a pending command. Use the proposed change only where supported by the "
                       "current public evidence; continue within the shared FORWARD_WINDOW.")
        if hasattr(self, 'persistent_cycle_constraints'):
            constraints = self.persistent_cycle_constraints()
            if constraints:
                view += ('\n[PERSISTENT_CYCLE_CONSTRAINTS]\n' + canonical(constraints)
                         + '\nThese state-action edges remain unavailable across Green and Yellow. '
                           'Accepted navigation does not clear them; only task-related public '
                           'progress creates a new recovery state.')
        self.episode.set_scope_prompt(view, scope.tree.tree_version,
                                      kind=scope.kind.value, directive=directive,
                                      incident_kind=scope.incident_kind)
        return self.episode.prompt(())

    def _clean_root_recovery_tree(self) -> str:
        """Serialize the complete execution history, with Red reviews as references."""
        def ref(occurrence_id):
            return "root" if occurrence_id is None else f"o{occurrence_id}"

        nodes = [{"id": "root", "parent_id": None, "scope_kind": "ROOT", "action": None}]
        for occurrence in sorted(self.occurrences.values(), key=lambda row: row.occurrence_id):
            receipt = occurrence.receipt
            nodes.append({"id": ref(occurrence.occurrence_id),
                          "parent_id": ref(occurrence.execution_parent_occurrence_id),
                          "scope_id": occurrence.scope_id, "scope_kind": occurrence.scope_kind.value,
                          "action": occurrence.action, "terminal": receipt["done"],
                          "terminal_score": receipt["public_terminal_score"], "rejected": receipt["invalid_action"]})
        review_scopes = []
        for session in self.review_sessions.values():
            path = session.frozen_path_occurrence_ids
            reviews = []
            for record in self.reviews:
                if record.scope_id == session.red_scope_id:
                    occurrence = None if record.checkpoint_index is None else self.occurrences[path[record.checkpoint_index]]
                    reviews.append({"checkpoint_node": None if occurrence is None else ref(occurrence.execution_parent_occurrence_id),
                                    "decision": record.decision.value})
            review_scopes.append({"scope_id": session.red_scope_id, "scope_kind": "RED",
                                  "entry_node": ref(path[-1]), "status": session.status, "reviews": reviews})
        current_path = ["root", *(ref(row.occurrence_id) for row in self.lineage)]
        graph = {"schema": "execution_graph_v1", "root_id": "root", "current_node": current_path[-1],
                 "current_marker": "[YOU ARE HERE]", "current_path": current_path,
                 "nodes": nodes, "review_scopes": review_scopes}
        if self.progress_recovery:
            graph["scopes"] = [{"scope_id": scope.scope_id, "kind": scope.kind.value,
                "entry_reason": scope.entry_reason, "caller_scope_id": scope.caller_scope_id,
                "occurrence_ids": scope.occurrence_ids, "latest_correction": scope.correction_record}
                for scope in self.scopes.values()]
        if self.efficient_recovery:
            # Keep every executed node and review. Do not repeatedly embed full
            # request records and duplicate decision-state snapshots in the graph.
            for row in graph.get('scopes', []):
                record = row['latest_correction']
                if record is not None:
                    row['latest_correction'] = {k: record[k] for k in (
                        'conclusion', 'selected_action', 'executed_occurrence_id', 'correction_completed')}
        encoded = (json.dumps(columnar_graph(graph), ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
                   if self.efficient_recovery else canonical(graph))
        return "[ROOT_RECOVERY_TREE]\n" + encoded + "\n[ROOT_RECOVERY_TREE_END]"

    def _forward_window(self):
        if self.active_trial_id is None:
            prefix_depth, granted, used = 0, self.episode.action_limit, len(self.lineage)
        else:
            trial = self.forward_trials[self.active_trial_id]
            prefix_depth = len(trial.prefix_occurrence_ids)
            granted, used = trial.forward_actions_granted, trial.forward_actions_used
            expected = (*trial.prefix_occurrence_ids, *trial.actual_suffix_occurrence_ids)
            if trial.status != "ACTIVE" or tuple(o.occurrence_id for o in self.lineage) != expected:
                raise ProtocolError("forward trial differs from actual execution lineage")
        if granted != self.episode.action_limit - prefix_depth or not 0 <= used <= granted:
            raise ProtocolError("forward trial exceeded its declared path window")
        return {"path_limit": self.episode.action_limit, "retained_prefix_depth": prefix_depth,
                "new_actions_granted": granted, "new_actions_used": used,
                "new_actions_remaining": granted - used, "submission_uses_one_action": True,
                "trial_id": self.active_trial_id}

    def _model_request_submitted(self, request):
        if self.alfworld_event_memory and self.active_scope.kind is ScopeKind.GREEN and request.route!='reflection':
            if self.pending_green_events:
                self.delivered_green_events.update(self.pending_green_events)
                self.emit('alf_event_subtree_delivered',request_id=request.request_id,
                          event_keys=list(self.pending_green_events),visible_prompt_is_self_contained=True)
            self.pending_green_events=[]
        if self.active_scope.kind not in {ScopeKind.GREEN, ScopeKind.YELLOW}:
            return
        if (self.active_scope.kind is ScopeKind.GREEN and request.route != "reflection"
                and self.green_history_handoff is not None):
            self.green_history_handoff["shown_request_ids"].append(request.request_id)
        if (not self.alfworld_event_memory and request.route != "reflection"
                and self.active_scope.kind is ScopeKind.GREEN
                and self.latest_yellow_correction is not None):
            correction_id = self.latest_yellow_correction["request_id"]
            if correction_id not in self.delivered_yellow_corrections:
                self.delivered_yellow_corrections.add(correction_id)
                self.emit("yellow_correction_delivered", request_id=request.request_id,
                          consumer_scope_id=self.active_scope_id, consumer_kind="GREEN",
                          correction_request_id=correction_id,
                          origin_scope_id=self.latest_yellow_correction["scope_id"])
        if self.active_trial_id is not None:
            self.forward_trials[self.active_trial_id].forward_request_ids.append(request.request_id)
        if self.active_scope.kind is ScopeKind.GREEN and self.green_handoff_directive is not None:
            if self.active_trial_id is not None:
                trial = self.forward_trials[self.active_trial_id]
                if trial.handoff_request_id is None:
                    trial.handoff_request_id = request.request_id
            self.emit("green_handoff_submitted", request_id=request.request_id,
                      trial_id=self.active_trial_id, directive=self.green_handoff_directive)
            self.green_handoff_directive = None

    def close(self):
        self.oracle.on_request_submitted = None

    def _ordinary(self, scope: ScopeTree, directive: str):
        address = self._address(scope)
        handoff = (None if self.active_red_memory is None
                   else self._red_actor_context())
        self.episode.red_repair_handoff = handoff
        if self.red_handoff_directive is not None:
            # The old first-action plan remains audit history, not a new instruction.
            self.emit('red_green_handoff_delivered', directive=self.red_handoff_directive)
            self.red_handoff_directive = None
        prompt = self._bind_scope(scope, directive)
        parsed, request_id = self.oracle.ordinary(self.episode, address, (), self.ledger)
        if self.active_red_memory is not None:
            self.emit('red_memory_delivered', request_id=request_id,
                      trial_id=self.active_trial_id, memory=handoff)
        self.episode.red_repair_handoff = None
        self.emit("scoped_actor_action_selected", scope_id=scope.scope_id,
                  scope_kind=scope.kind.value, request_id=request_id,
                  action=parsed.action, prompt=prompt,
                  history_window=getattr(self.episode, "last_history_window", None))
        return parsed, request_id

    def _red_actor_context(self):
        """Return the model-visible recovery handoff for the active trial."""
        return self.active_red_memory.actor_context()

    def _parse_review(self, text: str) -> ReviewDecision:
        if type(text) is not str or not text.strip():
            raise ProtocolError("red review returned no decision")
        lines = text.strip().splitlines()
        first = lines[0].strip()
        try:
            decision = ReviewDecision(first)
        except ValueError as error:
            raise ProtocolError("red review first line is not RESUME/BACK/UNKNOWN") from error
        if len(lines) != 2 or not lines[1].strip():
            raise ProtocolError("red review requires a decision line and a nonempty reason line")
        return decision

    def _review_prompt(self, scope: ScopeTree, checkpoint_index: int) -> str:
        session = self.review_sessions[scope.scope_id]
        frozen_path = [self.occurrences[key] for key in session.frozen_path_occurrence_ids]
        if not frozen_path:
            raise ProtocolError("red review requires an actual execution lineage")
        if not 0 <= checkpoint_index < len(frozen_path):
            raise ProtocolError("red review checkpoint is outside the lineage")
        occurrence = frozen_path[checkpoint_index]
        path = []
        for item in frozen_path:
            path.append({"occurrence_id": item.occurrence_id,
                         "execution_parent_occurrence_id": item.execution_parent_occurrence_id,
                         "ordinal": item.ordinal, "action": item.action,
                         "scope_id": item.scope_id,
                         "scope_kind": item.scope_kind.value,
                         "outcome": {
                             "done": item.receipt["done"],
                             "score": item.receipt["public_terminal_score"],
                             "invalid": item.receipt["invalid_action"]}})
        graph = {"schema": "execution_graph_v1", "root_id": "root", "scope_kind": "RED",
                 "current_node": "root" if checkpoint_index == 0 else f"o{frozen_path[checkpoint_index - 1].occurrence_id}",
                 "current_path": ["root", *(f"o{item.occurrence_id}" for item in frozen_path[:checkpoint_index])],
                 "view": "cached_checkpoint", "environment_restored": False,
                 "live_current_node": "root" if not self.lineage else f"o{self.lineage[-1].occurrence_id}",
                 "nodes": [{"id": "root", "parent_id": None, "action": None}] + [
                     {"id": f"o{row['occurrence_id']}",
                      "parent_id": "root" if row["execution_parent_occurrence_id"] is None else f"o{row['execution_parent_occurrence_id']}",
                      "scope_id": row["scope_id"], "scope_kind": row["scope_kind"],
                      "action": row["action"], "outcome": row["outcome"]} for row in path]}
        view = (f"[RED_REVIEW_TREE scope={scope.scope_id} kind=RED]\n"
                "[ENTRY_REASON] " + scope.entry_reason + "\n"
                "[HISTORY_IS_READ_ONLY_EVIDENCE]\n"
                "[EXECUTION_GRAPH] " + canonical(graph) + "\n"
                "[PREVIOUS_REVIEWS] " + canonical([
                    {"checkpoint_index": record.checkpoint_index,
                     "decision": record.decision.value, "output": record.output}
                    for record in self.reviews if record.scope_id == scope.scope_id]) + "\n"
                "[TRIAL_OUTCOMES_THIS_REVIEW] " + canonical([
                    self._trial_evidence(self.forward_trials[key]) for key in session.trial_ids]) + "\n"
                "[REVIEW_LIFECYCLE] This review closes when a checkpoint is selected. "
                "A later exhausted continuation is reviewed as its own actual path.\n"
                "[YOU ARE HERE] cached candidate checkpoint; live environment is unchanged\n"
                "[CHECKPOINT_REVIEW_MODE] SNAPSHOT_ONLY\n"
                f"[CHECKPOINT_INDEX] {checkpoint_index}\n"
                f"[FORWARD_ACTIONS_REMAINING] {self.episode.action_limit - checkpoint_index}\n"
                "[CURRENT_CHECKPOINT_STATE] " + canonical(occurrence.receipt["before"]) + "\n"
                "[HISTORICAL_NEXT_ACTION] " + occurrence.action + "\n"
                "[HISTORICAL_RECEIPT] " + canonical(occurrence.receipt) + "\n"
                "[RED_SCOPE_DIRECTIVE] Review this cached checkpoint without executing or restoring anything. "
                "Only a selected RESUME checkpoint will be restored for actual execution. "
                f"选择此结点恢复后只剩 {self.episode.action_limit - checkpoint_index} 步，"
                f"预估在这 {self.episode.action_limit - checkpoint_index} 次真实行动内，"
                "通过公开动作链产生更符合原任务目标要求的真实终局！"
                "RESUME 不会重置步数。必须把打开商品、选择必要选项和最终购买都计入剩余行动。"
                "商品相关或价格合适不等于剩余步数足够。请根据当前缓存页面和公开动作判断："
                "只有公开证据支持预估能在剩余步数内产生更符合目标要求的终局才选 RESUME；"
                "应从更早节点改变决策或步数不足时选 BACK；无法依据公开信息作出判断时选 UNKNOWN。"
                "理由必须使用格式：还剩n步行动机会；当前位置：缓存节点的page_type及商品ID（如有）；"
                "预计终局公开动作链：从该节点出发的公开动作，以 -> 分隔，"
                "最后到达更符合目标要求的真实终局，并在括号内说明目标依据及仍不确定之处。"
                "说明预计满足或改善了原任务中的哪些明确要求，依据来自哪些公开商品或选项信息。"
                "仅能购买、换了选项或改变了页面，不构成更符合目标的证据；不要求保证满分，"
                "也不能从低分臆断具体失败属性或声称知道隐藏奖励。"
                "当前位置是历史下一动作执行之前的状态，不是HISTORICAL_RECEIPT中的after状态。"
                "每一个search[...]或click[...]都单独计一步，打开商品与购买必须分开计数。"
                "第一步必须在当前节点公开可用。search[query]的开放参数允许根据目标改写新查询，"
                "不要求重复旧搜索词；商品或选项点击仍必须有公开页面依据。后续动作只能依据公开信息推估，"
                "不能捏造尚未见到的商品选项或合并多次环境调用。"
                f"每轮最多进入红树 {self.RED_ENTRY_LIMIT} 次，不能依赖反复进入红树延长执行。"
                "Selection closes this review; a later exhausted continuation is reviewed as a new path. "
                "BACK or UNKNOWN continues this review without excluding future paths.")
        if self.progress_recovery:
            view = view.split("[RED_SCOPE_DIRECTIVE]", 1)[0] + (
                "[RED_SCOPE_DIRECTIVE] Review only; restore exactly once after RESUME. "
                "选择RESUME不需要证明一定成功或优于旧结果。第一步必须当前公开可执行，"
                "说明它如何针对已观察的问题，并预估剩余n步内完成原任务的路线。"
                "未来页面未知时可用条件步骤描述，不得捏造商品ID、选项或观察；"
                "未知搜索结果、未知最终分数本身不是拒绝恢复的理由。"
                "说明还剩n步、当前位置、预计行动链及不确定性；每次环境调用都计一步。"
                "若有具体公开障碍或预算不足且更早位置更有利，选择BACK。"
                "无法提出有依据的下一步时选择UNKNOWN，不把它伪装成RESUME。"
                "checkpoint0没有更早结点，只能RESUME或UNKNOWN，禁止BACK。"
                "UNKNOWN在checkpoint0结束本次审查，交由既有Root预算处理。"
                "理由为自由文本，不要求固定中文字段或标点。恢复后由Green重新选动作。")
        return view

    def _red_review(self, scope: ScopeTree, checkpoint_index: int):
        if self.batched_red:
            return self._batched_red_review(scope, checkpoint_index)
        address = self._address(scope)
        view = self._review_prompt(scope, checkpoint_index)
        self.episode.set_scope_prompt(view, scope.tree.tree_version,
                                      kind=ScopeKind.RED.value,
                                      directive="Review only; output RESUME, BACK, or UNKNOWN.")
        occurrence_id = self.review_sessions[scope.scope_id].frozen_path_occurrence_ids[checkpoint_index]
        cached = self.checkpoint_cache[occurrence_id]
        prompt = self.episode.review_prompt((), checkpoint=cached)
        review_view = self.episode.checkpoint_review_view(cached)
        self.oracle.bind(address, review_view, prompt, ())
        for attempt in range(2 if self.progress_recovery and checkpoint_index == 0 else 1):
            generation = self.oracle.request(address, Role.REFLECTION, 0, prompt,
                                              self.ledger,
                                              output_limit=self._reflection_output_limit(prompt),
                                              stop_mode="lines_2" if self.output_framing else None)
            self._require_reflection_eos(generation)
            decision = self._parse_review(generation.text)
            if not (self.progress_recovery and checkpoint_index == 0 and decision is ReviewDecision.BACK):
                break
            self.emit("red_root_back_rejected", scope_id=scope.scope_id,
                      request_id=generation.request.request_id, output=generation.text,
                      resubmission_allowed=attempt == 0, environment_restored=False)
            if attempt == 1:
                raise ProtocolError("checkpoint0 cannot BACK; explicit resubmission also violated the protocol")
            prompt += "\n[PROTOCOL_FEEDBACK] checkpoint0 has no parent. Previous output was not applied: " + canonical(generation.text) + "\nReturn RESUME or UNKNOWN with a reason."
            address = self._address(scope)
            self.oracle.bind(address, review_view, prompt, ())
        record = ReviewRecord(scope.scope_id, checkpoint_index, decision,
                              generation.request.request_id, prompt, generation.text)
        self.reviews.append(record)
        self.emit("red_review_completed", scope_id=scope.scope_id,
                  checkpoint_index=checkpoint_index, decision=decision.value,
                  request_id=record.request_id, output=generation.text)
        return decision, record.request_id

    @staticmethod
    def _parse_red_checkpoint_only(text, indices):
        lines=[line.strip() for line in text.strip().splitlines()]
        # An explicit DECLINE is a terminal phase-1 decision. Some model
        # samples repeat checkpoint fields before that terminal token; the
        # controller must preserve the explicit decline instead of treating
        # those stale fields as a second control decision.
        if 'DECLINE' in lines:
            if not lines or not lines[0].startswith('EVIDENCE '):
                raise ProtocolError('Red decline requires nonempty EVIDENCE')
            if not lines[0][len('EVIDENCE '):].strip():
                raise ProtocolError('Red decline evidence must be nonempty')
            if 'DECISION' not in lines:
                raise ProtocolError('Red decline requires DECISION')
            return None
        if len(lines)==3 and lines[0].startswith('EVIDENCE ') and lines[1]=='DECISION' and lines[2]=='DECLINE':
            if not lines[0][len('EVIDENCE '):].strip():
                raise ProtocolError('Red checkpoint evidence must be nonempty')
            return None
        if len(lines)!=4 or not lines[0].startswith('EVIDENCE ') or lines[1]!='DECISION':
            raise ProtocolError('Red checkpoint phase requires EVIDENCE, DECISION, CHECKPOINT and RESTORE_STEP')
        try:
            index=int(lines[2].split(None,1)[1]); restore=int(lines[3].split(None,1)[1])
        except (IndexError,ValueError) as error:
            raise ProtocolError('Red checkpoint phase has invalid indices') from error
        if index not in set(indices) or restore!=index:
            raise ProtocolError('Red checkpoint and restore step must identify one supplied candidate')
        evidence=lines[0][len('EVIDENCE '):].strip()
        if not evidence: raise ProtocolError('Red checkpoint evidence must be nonempty')
        return {'checkpoint_index':index,'restore_step':restore,'reason':evidence,'evidence':evidence}

    @staticmethod
    def _parse_red_action_only(text):
        lines=[line.strip() for line in text.strip().splitlines()]
        if len(lines)!=2 or not lines[0].startswith('REPLACEMENT_PLAN ') or not lines[1].startswith('EVIDENCE '):
            raise ProtocolError('Red action phase requires REPLACEMENT_PLAN and EVIDENCE')
        try:
            plan=json.loads(lines[0][len('REPLACEMENT_PLAN '):])
        except json.JSONDecodeError as error:
            raise ProtocolError('Red replacement plan must be a valid JSON array') from error
        evidence=lines[1][len('EVIDENCE '):].strip()
        if (not isinstance(plan,list) or not 1<=len(plan)<=4
                or any(type(item) is not str or not item.strip() or item!=item.strip()
                       or '\n' in item or '\r' in item for item in plan) or not evidence):
            raise ProtocolError('Red replacement plan must contain one to four nonempty actions')
        return plan,evidence

    def _red_checkpoint_indices(self, path):
        return list(range(len(path) - 1, -1, -1))

    def _batched_red_review(self, scope, checkpoint_index):
        if scope.scope_id not in self.red_batches:
            self.episode.compact_context = self.compact_context
            session = self.review_sessions[scope.scope_id]
            path = [self.occurrences[k] for k in session.frozen_path_occurrence_ids]
            indices = self._red_checkpoint_indices(path)
            if not indices:
                raise ProtocolError('required Red recovery has no supplied checkpoint')
            candidates = [(i, self.checkpoint_cache[path[i].occurrence_id]) for i in indices]
            public_path = [{"index": i, "action": o.action,
                            "accepted": o.receipt["native_accepted"],
                            "done": o.receipt["done"], "score": o.receipt["public_terminal_score"]}
                           for i, o in enumerate(path)]
            if self.alfworld_event_memory:
                from alf_recovery_projection import recovery_subtree
                self.episode.recovery_event_subtree=recovery_subtree(self)
            automatic_selector = getattr(self, '_automatic_checkpoint_from_reflection', None)
            if callable(automatic_selector) and hasattr(self.episode, 'red_causal_assessment_prompt'):
                prompt = self.episode.red_causal_assessment_prompt({
                    'candidates': [{'checkpoint_index': i} for i, _ in candidates],
                    'path': public_path})
            else:
                prompt = red_batch_prompt(self.episode, candidates, public_path, self.terminal_evidence)
            address = self._address(scope)
            live = self.episode.checkpoint()
            self.oracle.bind(address, self.episode, prompt, ())
            generation = self.oracle.request(address, Role.REFLECTION, 0, prompt, self.ledger,
                                             output_limit=self._reflection_output_limit(prompt))
            if self.episode.checkpoint() != live:
                raise ProtocolError("batched Red review mutated the environment")
            # Bounded free advice is not an executable protocol. Preserve its
            # exact text, including a token-limited ending, for the choice call.
            automatic_selector = getattr(self, '_automatic_checkpoint_from_reflection', None)
            if callable(automatic_selector):
                selected_index, checkpoint_choice_request_id, locator = automatic_selector(
                    scope, path, generation)
                if selected_index not in dict(candidates):
                    raise ProtocolError('automatic checkpoint selector returned an unavailable index')
                selected_checkpoint={
                    'checkpoint_index': selected_index,
                    'restore_step': selected_index,
                    'reason': generation.text,
                    'evidence': generation.text,
                    'selection_source': locator['locator'],
                    'transition_certificate': locator['certificate'],
                    'choice_request_id': checkpoint_choice_request_id,
                }
                checkpoint_choice = None
            else:
                choices={f'CHECKPOINT {i}':i for i in indices}
                options=tuple(choices)
                checkpoint_choice=self._typed_request(scope,
                    self.episode.red_selection_prompt(1,generation.text,options),commands=options)
                selected_checkpoint={
                    'checkpoint_index':choices[checkpoint_choice.text],
                    'restore_step':choices[checkpoint_choice.text],
                    'reason':generation.text,'evidence':generation.text}
            correction_request_id = None
            checkpoint = dict(candidates)[selected_checkpoint['checkpoint_index']]
            action_prompt=self.episode.red_action_prompt(
                checkpoint,selected_checkpoint['checkpoint_index'],selected_checkpoint['evidence'])
            action_address=self._address(scope)
            self.oracle.bind(action_address,self.episode,action_prompt,())
            action_generation=self.oracle.request(action_address,Role.REFLECTION,0,action_prompt,self.ledger,
                                                  output_limit=self._reflection_output_limit(action_prompt))
            if self.episode.checkpoint()!=live:
                raise ProtocolError('Red action phase mutated the environment')
            commands=tuple(checkpoint['environment']['public_state']['available_actions'])
            if not commands:
                raise ProtocolError('required Red recovery has no native action at checkpoint')
            action_choice=self._typed_request(scope,
                self.episode.red_selection_prompt(2,action_generation.text,commands),commands=commands)
            proposal_request_id=action_choice.request.request_id
            selected_prompt=action_choice.request.prompt
            selected={**selected_checkpoint,'replacement_action':action_choice.text,
                      'replacement_plan':[action_choice.text],
                      'continuation_advice':action_generation.text,
                      'reason':action_generation.text,'evidence':action_generation.text,
                      'failed_suffix':[o.action for o in path[selected_checkpoint['checkpoint_index']:]][:12]}
            if not self.episode.checkpoint_action_callable(checkpoint,action_choice.text):
                raise ProtocolError('Red choice escaped the cached native interface')
            rows = [selected]
            self.red_batches[scope.scope_id] = {"rows": {r["checkpoint_index"]: r for r in rows},
                                               "request_id": correction_request_id or proposal_request_id,
                                               "prompt": selected_prompt}
            self.emit("red_batch_completed", scope_id=scope.scope_id,
                      request_id=generation.request.request_id, candidate_indices=indices,
                      output=generation.text+'\n'+action_generation.text,
                      selected_checkpoint=selected["checkpoint_index"],
                      correction_request_id=correction_request_id,
                      checkpoint_choice=(None if checkpoint_choice is None else checkpoint_choice.text),
                      checkpoint_choice_request_id=(checkpoint_choice_request_id
                        if checkpoint_choice is None else checkpoint_choice.request.request_id),
                      environment_restores=0)
        batch = self.red_batches[scope.scope_id]
        if len(batch['rows']) != 1:
            raise ProtocolError('required Red recovery must bind exactly one checkpoint and action')
        selected = next(iter(batch["rows"].values()))
        selected_index = selected["checkpoint_index"]
        decision = ReviewDecision.RESUME
        record = ReviewRecord(scope.scope_id, selected_index, decision,
                              batch["request_id"], batch["prompt"], canonical(selected))
        self.reviews.append(record)
        self.emit("red_review_completed", scope_id=scope.scope_id, checkpoint_index=selected_index,
                  decision=decision.value, request_id=record.request_id, output=record.output,
                  from_batch=True)
        return decision, record.request_id

    def _correct_red_callability(self, scope, checkpoint, selected, request_id, prompt,
                                 *, review_evidence=None, raw_proposal=None):
        """One charged, model-selected correction at the same cached position."""
        review_view = self.episode.checkpoint_review_view(checkpoint)
        support = review_view.support()
        if "DECLINE" in support.commands or len(support.commands) >= 1024:
            raise ProtocolError("Red correction action language cannot represent explicit decline")
        commands = (*support.commands, "DECLINE") if support.kind is SupportKind.FINITE else ()
        interface = ({"allowed_commands": list(commands)} if commands else
                     {"open_action_contract": review_view.open_action_contract(),
                      "allowed_control": "DECLINE"})
        self.emit("red_callability_feedback", scope_id=scope.scope_id,
                  checkpoint_index=selected["checkpoint_index"], rejected_proposal=deepcopy(selected),
                  proposal_request_id=request_id, **interface,
                  environment_restored=False, max_correction_calls=1)
        feedback = (
            "\n[CALLABILITY_CORRECTION]\n" + canonical({
                "restore_step": selected["restore_step"],
                "rejected_proposal": selected,
                "error": "The proposed action is absent from this checkpoint's complete public interface.",
                **interface})
            + "\nThe checkpoint is fixed. The environment has not changed. Select exactly one complete "
              "native command permitted by the supplied public interface, or DECLINE if no repair is justified. "
              "Return only that command; no CHECKPOINT, RESTORE_STEP, evidence or replacement fields. "
              "This is one protocol correction, not another environment attempt."
        )
        if self.episode.snapshot().get('benchmark')=='alfworld':
            if review_evidence is None or type(raw_proposal) is not str:
                raise ProtocolError('ALFWorld Red correction requires its original evidence and raw proposal')
            feedback+= ('\n[FIXED_CACHED_STATE]\n'+canonical({
                'checkpoint_index':selected['checkpoint_index'],
                'remaining_actions':self.episode.action_limit-selected['checkpoint_index'],
                'public_state':review_view.snapshot()})
                +'\nChoose for this cached state before its historical next action, not for the live '
                  'failed endpoint. The previous proposal was not executed. Return one allowed command '
                  'or DECLINE; no evidence header or explanation.')
            correction_prompt=self.episode.red_callability_prompt(
                review_evidence,raw_proposal,feedback,checkpoint=checkpoint)
        else:
            correction_prompt=prompt+feedback
        try:
            corrected = self._typed_request(scope, correction_prompt, commands=commands)
        except ProtocolError as error:
            self.emit("red_callability_correction_rejected", scope_id=scope.scope_id,
                      checkpoint_index=selected["checkpoint_index"],
                      proposal_request_id=request_id,
                      correction_request_id=None,
                      raw_output=None,
                      reason="CORRECTION_ESCAPED_DECLARED_ACTION_LANGUAGE",
                      detail=str(error), environment_restored=False,
                      terminal_inferred=False)
            return None, request_id, prompt
        self.emit("red_callability_corrected", scope_id=scope.scope_id,
                  checkpoint_index=selected["checkpoint_index"], proposal_request_id=request_id,
                  correction_request_id=corrected.request.request_id,
                  raw_output=corrected.text, environment_restored=False)
        if corrected.text == "DECLINE":
            return None, corrected.request.request_id, corrected.request.prompt
        if not self.episode.checkpoint_action_callable(checkpoint, corrected.text):
            self.emit("red_callability_correction_rejected", scope_id=scope.scope_id,
                      checkpoint_index=selected["checkpoint_index"],
                      proposal_request_id=request_id,
                      correction_request_id=corrected.request.request_id,
                      raw_output=corrected.text,
                      reason="CORRECTION_NOT_CALLABLE_IN_CACHED_PUBLIC_STATE",
                      environment_restored=False, terminal_inferred=False)
            return None, corrected.request.request_id, corrected.request.prompt
        # The correction protocol selects one action, not a revised continuation.
        # A suffix conditioned on the rejected first action is no longer a plan.
        if "replacement_plan" in selected:
            self.emit("red_continuation_cancelled", scope_id=scope.scope_id,
                      reason="FIRST_ACTION_REPLACED_BY_EXPLICIT_CORRECTION",
                      rejected_plan=selected["replacement_plan"],
                      correction_request_id=corrected.request.request_id)
        return {**selected, "replacement_action": corrected.text,
                **({"replacement_plan": [corrected.text]} if "replacement_plan" in selected else {}),
                "evidence": "", "reason": "",
                "rejected_proposal": deepcopy(selected), "proposal_request_id": request_id,
                "correction_request_id": corrected.request.request_id}, corrected.request.request_id, corrected.request.prompt

    def _result_key_from_metadata(self, metadata: dict[str, Any]) -> str:
        if not isinstance(metadata, dict) or not isinstance(metadata.get("complete_projection"), dict):
            raise ProtocolError("terminal result lacks complete public projection")
        return canonical({
            "environment_contract": metadata.get("environment_contract"),
            "complete_projection": metadata["complete_projection"],
        })

    @staticmethod
    def _checkpoint_public_state(checkpoint: dict[str, Any]) -> dict[str, Any]:
        environment = checkpoint.get("environment")
        if not isinstance(environment, dict):
            raise ProtocolError("checkpoint lacks environment state")
        public_state = environment.get("public_state")
        if not isinstance(public_state, dict):
            raise ProtocolError("checkpoint lacks public state")
        return public_state

    def _checkpoint_can_reach_terminal(self, checkpoint: dict[str, Any], index: int) -> bool:
        """Do not resume a checkpoint whose public state cannot finish in-window."""
        lower_bound = getattr(self.episode, "minimum_completion_actions", None)
        if not callable(lower_bound):
            return True
        required = lower_bound(self._checkpoint_public_state(checkpoint))
        if required is None:
            return True
        if type(required) is not int or required < 0:
            raise ProtocolError("completion lower bound must be a nonnegative integer or None")
        return self.episode.action_limit - index >= required

    def _trial_evidence(self, trial: ForwardTrial):
        suffix = [self.occurrences[key] for key in trial.actual_suffix_occurrence_ids]
        return {"trial_id": trial.trial_id, "checkpoint_index": trial.checkpoint_index,
                "status": trial.status, "reason": trial.reason, "score": trial.score,
                "granted": trial.forward_actions_granted, "used": trial.forward_actions_used,
                "remaining": trial.forward_actions_granted - trial.forward_actions_used,
                "advice": trial.advice, "handoff_request_id": trial.handoff_request_id,
                "actual_suffix": [{"occurrence_id": o.occurrence_id, "action": o.action,
                    "accepted": o.receipt["native_accepted"], "done": o.receipt["done"],
                    "score": o.receipt["public_terminal_score"],
                    "public_operation_changed":
                        {k: v for k, v in o.receipt["before"].items() if k != "observation"}
                        != {k: v for k, v in o.receipt["after"].items() if k != "observation"}}
                    for o in suffix]}

    def _finish_forward_trial(self, outcome: str, reason: str):
        if self.active_trial_id is None:
            raise ProtocolError("trial completion requires an active forward trial")
        self._forward_window()
        trial = self.forward_trials[self.active_trial_id]
        completed = outcome in {"SUCCESS", "TERMINAL_FAILURE", "WINDOW_EXHAUSTED"}
        if outcome not in {"SUCCESS", "TERMINAL_FAILURE", "WINDOW_EXHAUSTED",
                           "RESOURCE_INTERRUPTED", "TECHNICAL_INTERRUPTED", "SCOPE_ENTRY_LIMIT", "REPAIR_UNRESOLVED"}:
            raise ProtocolError("unknown forward trial outcome")
        if outcome in {"SUCCESS", "TERMINAL_FAILURE"}:
            if not trial.actual_suffix_occurrence_ids or not self.lineage[-1].receipt["done"]:
                raise ProtocolError("terminal trial outcome lacks a real terminal receipt")
            trial.score = self.lineage[-1].receipt["public_terminal_score"]
            if (trial.score == 1.0) != (outcome == "SUCCESS"):
                raise ProtocolError("trial success differs from actual terminal score")
        elif outcome == "WINDOW_EXHAUSTED":
            if (trial.forward_actions_used != trial.forward_actions_granted
                    or self.lineage[-1].receipt["done"]):
                raise ProtocolError("trial window outcome differs from actual boundary")
        trial.status, trial.reason = outcome, reason
        trial.end_cost = self.ledger.to_dict()
        trial.end_restore_count = self.checkpoint_restore_count
        session = self.review_sessions[trial.owner_review_session_id]
        original = [self.occurrences[key] for key in
                    session.frozen_path_occurrence_ids[trial.checkpoint_index:]]
        actual = [self.occurrences[key] for key in trial.actual_suffix_occurrence_ids]
        # Observed behavior equality is diagnostic only; request identity is separate.
        same_suffix = [(o.action, o.receipt) for o in original] == [(o.action, o.receipt) for o in actual]
        self.emit("resume_trial_finished", **self._trial_evidence(trial),
                  owner_review_session_id=session.session_id, completed_failure=
                  outcome in {"TERMINAL_FAILURE", "WINDOW_EXHAUSTED"},
                  complete_observed_suffix_equal=same_suffix if completed else None,
                  actual_suffix_occurrence_ids=list(trial.actual_suffix_occurrence_ids),
                  start_cost=trial.start_cost, end_cost=trial.end_cost,
                  cost_delta={k: trial.end_cost[k] - trial.start_cost[k] for k in (
                      "model_requests", "physical_model_requests", "environment_calls",
                      "backend_requests", "input_tokens", "output_tokens", "elapsed_seconds")})
        self.emit('red_memory_released', trial_id=self.active_trial_id, reason=outcome)
        self.active_red_memory = None
        self.active_red_repair = None
        self.active_trial_id = None
        if outcome == "SUCCESS":
            session.status = "TASK_SUCCESS"
            self.scopes[session.red_scope_id].archived = True
        return session

    def interrupt(self, status: str, reason: str):
        if status not in {"RESOURCE_LIMIT", "TECHNICAL_ERROR"}:
            raise ProtocolError("invalid controller interruption")
        if self.pending_yellow_action is not None:
            self.emit("yellow_action_binding_cancelled", **self.pending_yellow_action, reason=reason)
        self.pending_yellow_action = None
        self.yellow_continuation = None
        if self.active_trial_id is not None:
            self._finish_forward_trial(
                "RESOURCE_INTERRUPTED" if status == "RESOURCE_LIMIT" else "TECHNICAL_INTERRUPTED",
                reason)
        if self.active_review_session_id is not None:
            self.review_sessions[self.active_review_session_id].status = status
        self.status = status
        if status == "RESOURCE_LIMIT":
            self.resource_limit = reason
        self.emit("scoped_controller_interrupted", status=status, reason=reason,
                  green_handoff_pending=self.green_handoff_directive is not None)
        self._release_checkpoints(final=True)

    def _release_checkpoints(self, *, final=False):
        needed = set()
        if not final:
            if self.active_scope.kind in {ScopeKind.GREEN, ScopeKind.YELLOW}:
                needed.update(o.occurrence_id for o in self.lineage)
            for session in self.review_sessions.values():
                if session.status in {"ACTIVE", "SUSPENDED_FOR_FORWARD_TRIAL"}:
                    needed.update(session.frozen_path_occurrence_ids[:session.next_checkpoint_index + 1])
                    if session.current_checkpoint_index is not None:
                        needed.add(session.frozen_path_occurrence_ids[session.current_checkpoint_index])
            for failure in self.review_queue:
                needed.update(failure["path_occurrence_ids"])
        released = sorted(set(self.checkpoint_cache) - needed)
        for key in released:
            del self.checkpoint_cache[key]
        if released:
            self.emit("checkpoint_cache_released", occurrence_ids=released,
                      live_count=len(self.checkpoint_cache), final=final)
        if final:
            self.root_checkpoint = None
            self.tip_checkpoint = None

    def _invalid_incident(self, receipt: dict[str, Any]) -> PendingIncident:
        if not receipt["invalid_action"]:
            raise ProtocolError("invalid incident requires an invalid receipt")
        payload = canonical({
            "public_state": {key: value for key, value in receipt["before"].items()
                             if key != "observation"},
            "action": receipt["action"],
            "invalid_detail": receipt["invalid_detail"],
        })
        incident_id = "incident-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return PendingIncident(
            incident_id=incident_id,
            result_key=canonical({"kind": "INVALID_ACTION",
                                   "detail": receipt["invalid_detail"]}),
            action=receipt["action"])

    def _pending_duplicate(self, action: str) -> PendingIncident | None:
        if action != "click[Buy Now]" or self.episode.snapshot()["page_type"] != "item":
            return None
        metadata = self.episode.result_metadata()
        key = self._result_key_from_metadata(metadata)
        entries = self.terminal_evidence.get(key, [])
        if any(entry["score"] != 1.0 for entry in entries):
            payload = canonical({
                "action": action,
                "result_key": key,
            })
            incident_id = "incident-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
            return PendingIncident(incident_id, key, action)
        return None

    def _green_scope(self) -> ScopeTree:
        greens = [scope for scope in self.scopes.values()
                  if scope.kind is ScopeKind.GREEN]
        if len(greens) != 1:
            raise ProtocolError("scoped recovery requires one reusable green tree")
        return greens[0]

    def _set_green_cursor_from_lineage(self, prefix: list[ExecutionOccurrence]) -> None:
        green = self._green_scope()
        cursor = green.tree.nodes[0].node_id
        for occurrence in prefix:
            existing = [green.tree.node(child) for child in green.tree.node(cursor).children
                        if green.tree.node(child).receipt == occurrence.receipt]
            if existing:
                cursor = existing[0].node_id
            else:
                cursor = green.tree.add_edge(
                    cursor, occurrence.action, occurrence.receipt,
                    visit_kind="RED_RESUME_PREFIX", attempt=occurrence.ordinal,
                    request_id=occurrence.request_id)
        green.tree.current_id = cursor
        if prefix:
            if prefix[-1].receipt["after"] != self.episode.snapshot():
                raise ProtocolError("green continuation cursor differs from restored checkpoint")
        elif green.tree.nodes[cursor].after != self.episode.snapshot():
            raise ProtocolError("green root cursor differs from restored checkpoint")

    def _execute(self, scope: ScopeTree, parsed, request_id: str):
        if self._forward_window()["new_actions_remaining"] == 0:
            raise ProtocolError("cannot execute beyond the shared forward window")
        before_checkpoint = self.episode.checkpoint()
        receipt = environment_call(self.episode, self.ledger, self.episode.step, parsed.action)
        if receipt["action"] != parsed.action:
            raise ProtocolError("native environment changed the selected action")
        node_id = scope.add_receipt(receipt, attempt=self.global_action_serial,
                                    request_id=request_id)
        occurrence = ExecutionOccurrence(
            occurrence_id=self.global_action_serial,
            execution_parent_occurrence_id=(self.lineage[-1].occurrence_id if self.lineage else None),
            ordinal=len(self.lineage), action=parsed.action,
            receipt=deepcopy(receipt), scope_id=scope.scope_id,
            scope_kind=scope.kind, scope_node_id=node_id,
            request_id=request_id)
        self.lineage.append(occurrence)
        self.occurrences[occurrence.occurrence_id] = occurrence
        self.checkpoint_cache[occurrence.occurrence_id] = before_checkpoint
        self.tip_checkpoint = self.episode.checkpoint()
        self.repetition.record(occurrence)
        certificate = getattr(self, 'transition_certificates', {}).get(occurrence.occurrence_id)
        if certificate is not None and certificate.get('task_progress') is True:
            clear_count = sum(len(bucket) for bucket in self.persistent_cycle_edges.values()) \
                if hasattr(self, 'persistent_cycle_edges') else 0
            if clear_count:
                self.persistent_cycle_edges.clear()
                self.emit('persistent_cycle_constraints_cleared', occurrence_id=occurrence.occurrence_id,
                          reason='TASK_RELATED_PUBLIC_PROGRESS', cleared_count=clear_count)
        state_key = self._public_state_key(receipt["before"])
        if receipt["native_accepted"]:
            self.rejected_public_actions.pop(state_key, None)
        elif receipt["invalid_action"]:
            self.rejected_public_actions.setdefault(state_key, set()).add(parsed.action)
        scope.occurrence_ids.append(occurrence.occurrence_id)
        if self.active_trial_id is not None:
            self.forward_trials[self.active_trial_id].actual_suffix_occurrence_ids.append(occurrence.occurrence_id)
        self._forward_window()
        self.global_action_serial += 1
        if self.active_red_memory is not None:
            retired = self.active_red_memory.observe_action(
                parsed.action, occurrence.occurrence_id, receipt['native_accepted'])
            if retired is not None:
                self.emit('red_instruction_retired', trial_id=self.active_trial_id, **retired)
        self.emit("scoped_action_executed", scope_id=scope.scope_id,
                  scope_kind=scope.kind.value, node_id=node_id,
                  occurrence_id=occurrence.occurrence_id,
                  execution_parent_occurrence_id=occurrence.execution_parent_occurrence_id,
                  ordinal=occurrence.ordinal, action=parsed.action,
                  receipt=receipt)
        if receipt["done"]:
            metadata = self.episode.result_metadata()
            key = self._result_key_from_metadata(metadata)
            self.terminal_evidence.setdefault(key, []).append({
                "occurrence_id": occurrence.occurrence_id,
                "ordinal": occurrence.ordinal,
                "score": receipt["public_terminal_score"],
                "metadata": metadata,
            })
        if (scope.kind is ScopeKind.GREEN and self.green_history_handoff is not None
                and request_id in self.green_history_handoff["shown_request_ids"]
                and receipt["native_accepted"] and not parsed.is_think):
            self.emit("green_history_handoff_consumed", request_id=request_id,
                      occurrence_id=occurrence.occurrence_id)
            self.green_history_handoff = None
        return receipt, occurrence

    @staticmethod
    def _public_state_key(state):
        return canonical({key: value for key, value in state.items()
                          if key not in {"observation", "last_public_observation"}})

    def _rejected_action_in_current_state(self, action):
        state_key = self._public_state_key(self.episode.snapshot())
        return action in self.rejected_public_actions.get(state_key, set())

    def _currently_rejected_actions(self):
        state_key = self._public_state_key(self.episode.snapshot())
        return tuple(sorted(self.rejected_public_actions.get(state_key, set())))

    def _suppress_rejected_proposal(self, scope, action):
        """Re-open Yellow with the same incident, without spending an env step."""
        self.emit("public_rejection_action_suppressed", action=action,
                  state_key=self._public_state_key(self.episode.snapshot()),
                  reason="same_public_state_after_rejection")
        # The old proposal is not executed.  Yellow must choose a different
        # callable command from the unchanged interface.
        result = self._run_yellow(
            "previous action was rejected without a public state change; choose a different callable action",
            action, "local-rejection-" + str(self.global_action_serial),
            incident_kind="REJECTED_ACTION_SAME_STATE")
        if result[0] == "STOP":
            return result[0]
        return result[0]

    def _enter_yellow(self, reason: str, pending_action: str,
                      incident_id: str | None = None):
        if self.attempt_scope_entries["YELLOW"] >= self.YELLOW_ENTRY_LIMIT:
            raise ProtocolError("Yellow entry exceeds the configured task allowance")
        caller = self.active_scope
        if incident_id is None:
            self.incident_serial += 1
            incident_id = f"incident-{self.incident_serial}"
        entries = self.yellow_entry_counts.get(incident_id, 0)
        self.yellow_entry_counts[incident_id] = entries + 1
        self.attempt_scope_entries["YELLOW"] += 1
        scope = self._new_scope(ScopeKind.YELLOW,
                               reason + "; incident_id=" + incident_id
                               + "; pending_action=" + pending_action,
                               caller=caller)
        scope.event_anchor_occurrence_id = self.lineage[-1].occurrence_id if self.lineage else None
        self.call_frames.append(CallFrame(
            caller_scope_id=caller.scope_id, caller_node_id=caller.current_node_id,
            callee_scope_id=scope.scope_id, incident_id=incident_id,
            entry_state=self.episode.snapshot(),
            return_policy=("BOUNDED_LOCAL_SEGMENT_TASK_PROGRESS_OR_LIMIT" if self.alfworld_event_memory else
                           "VALID_CORRECT_ACTION_TO_GREEN; PREPARE_STAYS_YELLOW; terminal_or_window_boundary"),
            remaining_budget={"shared_actions": self.episode.action_limit - len(self.lineage)}))
        return scope

    def _incident(self, reason: str) -> str:
        self.incident_serial += 1
        incident = f"incident-{self.incident_serial}"
        self.controller_events.append({"node_kind": NodeKind.CALL.value,
                                       "incident_id": incident, "reason": reason})
        return incident

    def _reflection_output_limit(self, prompt):
        if (not callable(self.oracle.token_counter) or type(self.oracle.context_limit) is not int
                or self.oracle.context_limit < 1):
            raise ProtocolError("uncapped scoped generation requires the actual tokenizer and context limit")
        input_tokens = self.oracle.token_counter(prompt)
        if type(input_tokens) is not int or input_tokens < 1:
            raise ProtocolError("invalid scoped prompt token count")
        context_remaining = self.oracle.context_limit - input_tokens
        if context_remaining < 1:
            raise BudgetExhausted("MODEL_CONTEXT_LIMIT: no generation space remains")
        task_remaining = self.ledger.limits.output_tokens - self.ledger.output_tokens
        if task_remaining < 1:
            raise BudgetExhausted("OUTPUT_TOKEN_RESOURCE_LIMIT")
        return min(REFLECTION_OUTPUT_TOKEN_LIMIT, context_remaining, task_remaining)

    def _require_reflection_eos(self, generation):
        finish = generation.raw_response["finish_reason"]
        if (finish == "frame" and self.output_framing and generation.request.stop_mode is not None
                and frame_complete(generation.text, generation.request.stop_mode)):
            return
        if finish in {"eos", "eos_at_limit"}:
            return
        if finish == "length":
            if self.ledger.output_tokens >= self.ledger.limits.output_tokens:
                raise BudgetExhausted("OUTPUT_TOKEN_RESOURCE_LIMIT")
            if generation.input_tokens + generation.request.output_limit >= self.oracle.context_limit:
                raise BudgetExhausted("MODEL_CONTEXT_OUTPUT_LIMIT")
        raise ProtocolError("scoped generation did not complete before its output limit")

    @staticmethod
    def _parse_yellow_conclusion(text):
        if type(text) is not str:
            raise ProtocolError("Yellow conclusion must be text")
        lines = text.strip().splitlines()
        fields = ("problem", "evidence", "next_change", "step_role", "current_action")
        if len(lines) != len(fields) or any(not line.strip() for line in lines):
            raise ProtocolError("Yellow decision requires exactly five nonempty positional lines")
        conclusion = dict(zip(fields, lines))
        # V30 grammar: horizontal ASCII whitespace delimits the role token.
        # The original five-line response remains in the correction record.
        role = re.fullmatch(r"[ \t]*(PREPARE|CORRECT)[ \t]*", conclusion["step_role"])
        if role is None:
            raise ProtocolError("Yellow step role must be PREPARE or CORRECT")
        conclusion["step_role"] = role.group(1)
        return conclusion

    def _yellow_conclusion(self, scope, incident_id):
        if (scope.kind is not ScopeKind.YELLOW or self.active_scope_id != scope.scope_id
                or self.pending_yellow_action is not None):
            raise ProtocolError("Yellow decision requires an active scope with no pending action")
        self.yellow_invalid_proposal = False
        self._bind_scope(scope, "Reassess the latest observation; select one action and declare PREPARE or CORRECT.")
        path = [{"occurrence_id": o.occurrence_id, "action": o.action,
                 "observation": o.receipt["observation"],
                 "before": deepcopy(o.receipt["before"]),
                 "after": deepcopy(o.receipt["after"]),
                 "accepted": o.receipt["native_accepted"], "done": o.receipt["done"],
                 "score": o.receipt["public_terminal_score"]} for o in self.lineage]
        if self.typed_scope_choices:
            return self._yellow_typed_choice(scope, incident_id, path)
        checkpoint = self.episode.checkpoint()
        feedback = None
        # One explicit resubmission resolves protocol ambiguity without guessing an action.
        for attempt in range(2 if self.yellow_action_ids else 1):
            prompt = (self.episode.yellow_conclusion_prompt(path, self.terminal_evidence)
                      if feedback is None else self.episode.yellow_conclusion_prompt(
                          path, self.terminal_evidence, protocol_feedback=feedback))
            address = self._address(scope)
            self.oracle.bind(address, self.episode, prompt, ())
            generation = self.oracle.request(address, Role.REFLECTION, 0, prompt, self.ledger,
                                             output_limit=self._reflection_output_limit(prompt),
                                             stop_mode=("json_object" if self.yellow_action_ids else "lines_5")
                                             if self.output_framing else None)
            if self.episode.checkpoint() != checkpoint:
                raise ProtocolError("Yellow diagnosis mutated the live environment")
            self._require_reflection_eos(generation)
            try:
                if self.episode.snapshot().get('benchmark')=='alfworld':
                    conclusion=self.episode.parse_yellow_selection(generation.text)
                else:
                    conclusion = (parse_yellow_choice(generation.text, self.episode.support(), progress_recovery=self.progress_recovery) if self.yellow_action_ids
                                  else self._parse_yellow_conclusion(generation.text))
                rejected_now = self._currently_rejected_actions()
                if self.yellow_action_ban:
                    if conclusion['current_action'] is None:
                        raise YellowChoiceError("This Yellow must choose a non-banned callable action; action=null cannot complete this repair")
                    if conclusion['current_action'] in self.yellow_bans[scope.scope_id]:
                        raise YellowChoiceError("This action is explicitly forbidden in the current Yellow scope; choose another current action")
                if conclusion["current_action"] in rejected_now:
                    raise YellowChoiceError("This action was rejected in the unchanged public state; choose another currently admissible action")
                if conclusion["current_action"] is not None:
                    support = self.episode.support()
                    candidate = self.episode.parse(conclusion["current_action"])
                    if support.kind in {SupportKind.FINITE, SupportKind.OPEN} and not self.episode.public_physical(candidate, support):
                        raise YellowCallabilityError("Action violates CURRENT_INTERFACE; choose a currently callable action, not an action from a future page")
            except YellowChoiceError as error:
                feedback = {"request_id": generation.request.request_id,
                            "unexecuted_output": generation.text, "error": str(error)}
                self.emit("yellow_protocol_rejected", scope_id=scope.scope_id, attempt=attempt,
                          resubmission_allowed=attempt == 0, **feedback)
                if attempt == 1:
                    if not self.yellow_action_ban and isinstance(error, (YellowCallabilityError,YellowActionIdRangeError)):
                        self.emit("yellow_callability_resubmissions_exhausted", scope_id=scope.scope_id,
                                  request_id=generation.request.request_id, output=generation.text,
                                  environment_actions_used=0)
                        self.yellow_invalid_proposal = True
                        self._record_yellow_unexecuted(scope, incident_id, generation, {
                            "diagnosis": "No callable proposal after one resubmission", "rejection": feedback})
                        return None, generation.request.request_id
                    raise
            else:
                if feedback is not None:
                    self.emit("yellow_protocol_resubmitted", scope_id=scope.scope_id,
                              rejected_request_id=feedback["request_id"],
                              accepted_request_id=generation.request.request_id)
                break
        step_role = conclusion.pop("step_role")
        current_action = conclusion.pop("current_action")
        if self.progress_recovery and step_role == "RETURN_GREEN_UNRESOLVED":
            self.emit("yellow_model_yielded", scope_id=scope.scope_id,
                      request_id=generation.request.request_id, output=generation.text, conclusion=conclusion,
                      environment_actions_used=0)
            self._record_yellow_unexecuted(scope, incident_id, generation, conclusion)
            return None, generation.request.request_id
        parsed = self.episode.parse(current_action)
        if parsed.action != current_action or not parsed.grammar_valid or parsed.is_think:
            raise ProtocolError("Yellow decision contains a non-native current action")
        # Callability errors are real native rejected actions, just as in Green.
        # They consume one forward slot and supply actual feedback to Yellow.
        if not self.episode.public_physical(parsed, self.episode.support()):
            self.emit("yellow_native_rejection_expected", scope_id=scope.scope_id,
                      request_id=generation.request.request_id, action=current_action,
                      public_state=self.episode.snapshot())
        record = {"scope_id": scope.scope_id, "incident_id": incident_id,
                  "request_id": generation.request.request_id,
                  "decision_state": deepcopy(self.episode.snapshot()),
                  "conclusion": conclusion, "output": generation.text,
                  "step_role": step_role, "selected_action": current_action,
                  "executed_occurrence_id": None, "correction_completed": None}
        scope.correction_record = deepcopy(record)
        scope.correction_records.append(deepcopy(record))
        self.latest_yellow_correction = record
        self.pending_yellow_action = {"scope_id": scope.scope_id, "request_id": record["request_id"],
                                      "action": current_action, "step_role": step_role}
        self.emit("yellow_conclusion_completed", **deepcopy(record),
                  environment_actions_used=0)
        self.emit("scoped_actor_action_selected", scope_id=scope.scope_id,
                  scope_kind=scope.kind.value, request_id=generation.request.request_id,
                  action=current_action, prompt=prompt, route="reflection_joint_action",
                  step_role=step_role)
        return parsed, generation.request.request_id

    def _typed_request(self, scope, prompt, *, commands=(), parameter_language=None):
        address = self._address(scope)
        self.oracle.bind(address, self.episode, prompt, ())
        live = self.episode.checkpoint()
        limit = 100 if commands else self._reflection_output_limit(prompt)
        if commands:
            if not callable(self.oracle.command_language_validator):
                raise ProtocolError('typed finite selection requires the actual tokenizer validator')
            self.oracle.command_language_validator(commands, limit)
        generation = self.oracle.request(address, Role.ALTERNATIVE if commands else Role.REFLECTION,
            0, prompt, self.ledger, commands=commands, output_limit=limit, parameter_language=parameter_language)
        if self.episode.checkpoint() != live:
            raise ProtocolError('typed scope selection mutated the live environment')
        self._require_reflection_eos(generation)
        if commands and generation.text not in commands:
            raise ProtocolError('constrained selection escaped its declared command language')
        if parameter_language is not None:
            prefix, suffix = parameter_language['prefix'], parameter_language['suffix']
            text = generation.text
            if not text.startswith(prefix) or not text.endswith(suffix):
                raise ProtocolError('parameter generation escaped its declared frame')
            value = text[len(prefix):-len(suffix)]
            if (not value.strip() or any(c in value for c in '\r\n' + suffix + prefix[-1:])
                    or text in parameter_language['forbidden']):
                raise ProtocolError('parameter generation escaped its declared exclusions')
        if not generation.text.strip():
            raise ProtocolError('empty typed scope response')
        return generation

    def _yellow_typed_choice(self, scope, incident_id, path):
        if self.tree_internal_repair:
            support = self.episode.support()
            commands = tuple(a for a in support.commands
                             if a not in self.yellow_bans[scope.scope_id]
                             and a not in self._currently_rejected_actions())
            if not commands:
                raise ProtocolError('Yellow has no callable action after excluding public rejections')
            failed_suffix = []
            if self.active_trial_id is not None:
                trial = self.forward_trials[self.active_trial_id]
                session = self.review_sessions[trial.owner_review_session_id]
                failed_suffix = [{'action':self.occurrences[k].action,
                    'state':self.occurrences[k].receipt['before'],
                    'accepted':self.occurrences[k].receipt['native_accepted']}
                    for k in session.frozen_path_occurrence_ids[trial.checkpoint_index:]]
            prompt = self.episode.single_yellow_prompt(scope.entry_reason, self._forward_window(), path, failed_suffix)
            language = self.episode.open_parameter_language(self.yellow_bans[scope.scope_id]) if support.kind is SupportKind.OPEN else None
            generation = self._typed_request(scope, prompt, commands=commands, parameter_language=language)
            parsed = self.episode.parse(generation.text)
            if (parsed.action != generation.text or not parsed.grammar_valid or parsed.is_think
                    or not self.episode.public_physical(parsed, support)):
                raise ProtocolError('Yellow action escaped its declared public interface')
            record = {'scope_id':scope.scope_id,'incident_id':incident_id,'request_id':generation.request.request_id,
                'assessment_request_id':None,'decision_state':deepcopy(self.episode.snapshot()),
                'conclusion':{'controller_entry_reason':scope.entry_reason}, 'output':generation.text,
                'step_role':'ALTERNATIVE','selected_action':parsed.action,
                'executed_occurrence_id':None,'correction_completed':None}
            scope.correction_record = deepcopy(record)
            scope.correction_records.append(deepcopy(record))
            self.latest_yellow_correction = record
            self.pending_yellow_action = {'scope_id':scope.scope_id,'request_id':record['request_id'],
                'action':parsed.action,'step_role':'ALTERNATIVE'}
            self.emit('yellow_conclusion_completed', **deepcopy(record), environment_actions_used=0)
            self.emit('scoped_actor_action_selected',scope_id=scope.scope_id,scope_kind='YELLOW',
                request_id=record['request_id'],action=parsed.action,prompt=prompt,route='v41_single_selection')
            return parsed,record['request_id']
        assessment = self._typed_request(scope, self.episode.yellow_typed_prompt(
            path, self.terminal_evidence, phase='assess'))
        support = self.episode.support()
        commands = tuple(a for a in support.commands
                         if a not in self.yellow_bans[scope.scope_id]
                         and a not in self._currently_rejected_actions())
        if not commands and support.kind is SupportKind.FINITE:
            raise ProtocolError('Yellow has no callable action after excluding public rejections')
        if self.efficient_recovery:
            prompt = self.episode.efficient_yellow_selection(assessment.text)
            language = self.episode.open_parameter_language(self.yellow_bans[scope.scope_id]) if support.kind is SupportKind.OPEN else None
            generation = self._typed_request(scope, prompt, commands=commands, parameter_language=language)
            action = generation.text
            parsed = self.episode.parse(action)
            if (parsed.action != action or not parsed.grammar_valid or parsed.is_think
                    or action in self.yellow_bans[scope.scope_id] or not self.episode.public_physical(parsed, support)):
                raise ProtocolError('typed Yellow escaped the declared public action language')
        feedback = None
        for attempt in range(0 if self.efficient_recovery else 1 if support.kind is SupportKind.FINITE else 2):
            prompt = self.episode.yellow_typed_prompt(path, self.terminal_evidence, phase='choose',
                assessment=assessment.text, feedback=feedback)
            generation = self._typed_request(scope, prompt, commands=commands)
            try:
                action = generation.text if support.kind is SupportKind.FINITE else self.episode.open_action_from_parameters(generation.text)
                parsed = self.episode.parse(action)
                if (action in self.yellow_bans[scope.scope_id]
                        or action in self._currently_rejected_actions()
                        or not self.episode.public_physical(parsed, support)):
                    raise YellowChoiceError('parameter choice violates the current public interface or exact local ban')
            except YellowChoiceError as error:
                feedback = {'request_id': generation.request.request_id, 'unexecuted_output': generation.text, 'error': str(error)}
                self.emit('yellow_protocol_rejected', scope_id=scope.scope_id, attempt=attempt,
                          resubmission_allowed=attempt == 0, **feedback)
                if attempt == 1:
                    raise
            else:
                break
        record = {'scope_id':scope.scope_id, 'incident_id':incident_id, 'request_id':generation.request.request_id,
            'assessment_request_id':assessment.request.request_id, 'decision_state':deepcopy(self.episode.snapshot()),
            'conclusion':{'diagnosis':assessment.text}, 'output':generation.text, 'step_role':'ALTERNATIVE',
            'selected_action':action, 'executed_occurrence_id':None, 'correction_completed':None}
        scope.correction_record = deepcopy(record)
        scope.correction_records.append(deepcopy(record))
        self.latest_yellow_correction = record
        self.pending_yellow_action = {'scope_id':scope.scope_id,'request_id':record['request_id'],
                                     'action':action,'step_role':'ALTERNATIVE'}
        self.emit('yellow_conclusion_completed', **deepcopy(record), environment_actions_used=0)
        self.emit('scoped_actor_action_selected', scope_id=scope.scope_id, scope_kind='YELLOW',
                  request_id=record['request_id'], action=action, prompt=prompt,
                  route='typed_finite_choice' if commands else 'typed_open_parameters', step_role='ALTERNATIVE')
        return parsed, record['request_id']

    def _record_yellow_unexecuted(self, scope, incident_id, generation, conclusion):
        scope.correction_record = {"scope_id": scope.scope_id, "incident_id": incident_id,
            "request_id": generation.request.request_id, "decision_state": deepcopy(self.episode.snapshot()),
            "conclusion": conclusion, "output": generation.text, "step_role": "RETURN_GREEN_UNRESOLVED",
            "selected_action": None, "executed_occurrence_id": None, "correction_completed": False}
        self.latest_yellow_correction = deepcopy(scope.correction_record)

    def _yellow_incident_key(self, kind, action):
        return canonical({"kind": kind, "action": action, "state": self.episode.snapshot(),
                          "evidence_revision": self.repetition.revision})

    def _yellow_eligible(self, kind, action):
        if self.attempt_scope_entries["YELLOW"] >= self.YELLOW_ENTRY_LIMIT:
            self.emit("yellow_entry_suppressed", kind=kind, action=action,
                      reason="YELLOW_ENTRY_LIMIT", destination="GREEN",
                      entries=self.attempt_scope_entries["YELLOW"], limit=self.YELLOW_ENTRY_LIMIT,
                      execution_policy="ALLOW_CURRENT_GREEN_PROPOSAL")
            return False
        if not self.progress_recovery:
            return self._yellow_incident_key(kind, action) not in self.yellow_suppressed
        reason = None
        if self.episode.action_limit - len(self.lineage) < (1 if self.yellow_action_ban else 2):
            reason = "RESERVE_FORWARD_ACTION"
        elif self._yellow_incident_key(kind, action) in self.yellow_suppressed:
            reason = "SAME_INCIDENT_WITHOUT_NEW_PUBLIC_EVIDENCE"
        if reason is not None:
            self.emit("yellow_entry_suppressed", kind=kind, action=action, reason=reason,
                      destination="GREEN", remaining_actions=self.episode.action_limit - len(self.lineage))
        return reason is None

    def _yield_yellow(self, scope, pending_action, reason, *, suppress_incident=True):
        self.last_yellow_unresolved = True
        if suppress_incident:
            self.yellow_suppressed.add(self._yellow_incident_key(scope.incident_kind, pending_action))
        else:
            self.emit("yellow_incident_not_suppressed", scope_id=scope.scope_id,
                      reason=reason, pending_action=pending_action,
                      because="UNEXECUTED_INVALID_PROPOSAL")
        self._return_to_green(scope, "RETURN_GREEN_UNRESOLVED", handoff_directive=(
            "Yellow ended without certifying a repair: " + reason + ". Original incident: " + scope.entry_reason
            + ". Continue the original task using the current public interface and remaining actions. "
            "Do not replay rejected proposals. No action is bound and no terminal has been inferred."))
        self.emit("yellow_unresolved_return", scope_id=scope.scope_id, reason=reason,
                  actions_used=scope.action_serial, remaining_actions=self.episode.action_limit - len(self.lineage))
        if self.tree_internal_repair and self.active_trial_id is not None:
            self._finish_forward_trial('REPAIR_UNRESOLVED', reason)
            return self._root_fallback(reason='YELLOW_REPAIR_UNRESOLVED'), scope.action_serial
        return "RETURN_GREEN", scope.action_serial

    def _continue_yellow_segment(self, scope, pending_action, incident_id, incident_kind):
        """Continue one bounded Yellow segment from its live scope.

        Yellow may make a few local observations/actions to escape the current
        obstruction. It never becomes a second Green planner: navigation alone
        is retained as context and cannot complete the segment.
        """
        limit = self.yellow_local_action_limit
        if limit is None:
            limit = min(4, max(1, self._forward_window()['new_actions_remaining'] + scope.action_serial))
        while (scope.action_serial < limit
               and len(self.lineage) < self.episode.action_limit):
            before = self.episode.snapshot()
            parsed, request_id = self._yellow_conclusion(scope, incident_id)
            if parsed is None:
                self.yellow_invalid_proposal = False
                return self._yield_yellow(
                    scope, pending_action, 'YELLOW_UNRESOLVED_NO_CALLABLE_LOCAL_ACTION')
            if (self.pending_yellow_action is None
                    or self.pending_yellow_action['request_id'] != request_id
                    or self.pending_yellow_action['action'] != parsed.action):
                raise ProtocolError('Yellow continuation differs from its single-step binding')
            repeated = self.repetition.repeated(before, parsed.action)
            if repeated is not None:
                self.pending_yellow_action = None
                self.emit('yellow_local_step_unresolved', scope_id=scope.scope_id,
                          action=parsed.action, step_role='CONTINUATION',
                          accepted=False,
                          shared_actions_remaining=self.episode.action_limit-len(self.lineage))
                return self._yield_yellow(
                    scope, pending_action, 'REPEATED_LOCAL_REPAIR_WITHOUT_NEW_EVIDENCE')
            receipt, occurrence = self._execute(scope, parsed, request_id)
            self.pending_yellow_action = None
            certificate = getattr(self, 'transition_certificates', {}).get(
                occurrence.occurrence_id, {})
            task_progress = certificate.get('task_progress') is True
            self.emit('yellow_continuation_action_executed',
                      scope_id=scope.scope_id, action=parsed.action,
                      occurrence_id=occurrence.occurrence_id,
                      task_progress=task_progress,
                      classification=certificate.get('classification'),
                      actions_used=scope.action_serial,
                      actions_limit=limit)
            if receipt['done']:
                if receipt['public_terminal_score'] == 1.0:
                    return 'SUCCESS', scope.action_serial
                return 'TERMINAL_FAILURE', scope.action_serial
            if task_progress:
                self._return_to_green(
                    scope, 'YELLOW_TASK_PROGRESS',
                    handoff_directive=(
                        'Yellow produced new task-related public evidence: '
                        + parsed.action + '. Continue the original task from the updated state.'))
                return 'RETURN_GREEN', scope.action_serial
        return self._yield_yellow(scope, pending_action, 'YELLOW_UNRESOLVED_LOCAL_ACTION_LIMIT')

    def _run_yellow(self, reason: str, pending_action: str, incident_id: str, *,
                    incident_kind="CALLABILITY"):
        self.last_yellow_unresolved = False
        if not self.enable_yellow:
            raise ProtocolError("disabled Yellow cannot be entered")
        if self.attempt_scope_entries["YELLOW"] >= self.YELLOW_ENTRY_LIMIT:
            self._yellow_eligible(incident_kind, pending_action)
            return "BYPASS_YELLOW", 0
        if (incident_kind == "REJECTED_ACTION_SAME_STATE"
                and self._yellow_incident_key(incident_kind, pending_action) in self.yellow_suppressed):
            self.emit("public_rejection_action_suppressed", action=pending_action,
                      reason="same_state_rejection_already_recovered")
            self.green_handoff_directive = (
                "The proposed action was already rejected in this unchanged public state. "
                "Do not retry it; choose a different currently admissible action.")
            return "RETURN_GREEN", 0
        if self.progress_recovery:
            if not self._yellow_eligible(incident_kind, pending_action):
                if self.tree_internal_repair and self.active_trial_id is not None:
                    self._finish_forward_trial('REPAIR_UNRESOLVED','YELLOW_ENTRY_UNAVAILABLE')
                    return self._root_fallback(reason='YELLOW_ENTRY_UNAVAILABLE'), 0
                return "RETURN_GREEN", 0
            if not self.yellow_action_ban:
                self.yellow_suppressed.add(self._yellow_incident_key(incident_kind, pending_action))
            self.yellow_local_action_limit = max(1, (self.episode.action_limit - len(self.lineage)) // 2)
        # Each incident gets at most one intervention within the task allowance.
        if self.yellow_entry_counts.get(incident_id, 0) >= 1:
            self.emit("yellow_entry_suppressed", kind=incident_kind,
                      action=pending_action, incident_id=incident_id,
                      reason="INCIDENT_CORRECTION_ALREADY_ATTEMPTED",
                      execution_policy="RETURN_GREEN_WITH_SAME_BUDGET")
            self.yellow_suppressed.add(self._yellow_incident_key(incident_kind, pending_action))
            self.green_handoff_directive = (
                'Yellow has already attempted this incident without establishing a repair. '
                'Continue the original task from the actual receipts. No action has executed '
                'and no new action budget has been granted. Incident: ' + reason)
            self.emit('yellow_unresolved_return', incident_id=incident_id,
                      actions_used=0, remaining_actions=self._forward_window()['new_actions_remaining'])
            return "RETURN_GREEN", 0
        scope = self._enter_yellow(reason, pending_action, incident_id)
        scope.incident_kind = incident_kind
        if self.yellow_local_action_limit is None:
            self.yellow_local_action_limit = min(
                4, max(1, self._forward_window()['new_actions_remaining']))
        entry_revision = self.repetition.revision
        recovery_segment = self.tree_internal_repair and self.active_trial_id is not None
        segment_states = {operation_key(self.episode.snapshot())}
        if self.tree_internal_repair:
            self.yellow_local_decisions[scope.scope_id] = {
                operation_key(self.episode.snapshot()): [] if recovery_segment else [pending_action]}
        if self.yellow_action_ban:
            self.yellow_bans[scope.scope_id] = [] if recovery_segment else [pending_action]
            self.episode.yellow_forbidden_actions = tuple(self.yellow_bans[scope.scope_id])
            self.emit("yellow_action_forbidden", scope_id=scope.scope_id, incident_id=incident_id,
                      action=pending_action, lifetime="THIS_YELLOW_SCOPE_ONLY", incident_kind=incident_kind)
            support = self.episode.support()
            if not recovery_segment and support.kind is SupportKind.FINITE and not any(a != pending_action for a in support.commands):
                self.emit("yellow_action_ban_no_alternative", scope_id=scope.scope_id,
                          action=pending_action, destination="GREEN", environment_actions_used=0)
                self._return_to_green(scope, "NO_PUBLIC_ALTERNATIVE_AFTER_BAN",
                                      handoff_directive=("Yellow could not enter action-ban mode because the current "
                                                         "public interface had no alternative callable action. Continue "
                                                         "using the original Green policy."))
                return "RETURN_GREEN", 0
        self.call_frames[-1].return_policy = (
            "BOUNDED_LOCAL_SEGMENT_TASK_PROGRESS_OR_LIMIT" if self.alfworld_event_memory
            else "VALID_CORRECT_ACTION_TO_GREEN")
        # ALFWorld PREPARE remains in this scope within the existing shared window.
        # The legacy policy keeps its original single-action behavior.
        if len(self.lineage) < self.episode.action_limit:
            if self.progress_recovery and scope.action_serial >= self.yellow_local_action_limit:
                if self.yellow_action_ban and not self.tree_internal_repair:
                    raise ProtocolError("Yellow local action budget exhausted without an accepted alternative")
                return self._yield_yellow(scope, pending_action, "LOCAL_ACTION_BUDGET")
            before = self.episode.snapshot()
            if self.tree_internal_repair:
                local = self.yellow_local_decisions[scope.scope_id]
                self.yellow_bans[scope.scope_id] = list(local.get(operation_key(before), []))
                self.episode.yellow_forbidden_actions = tuple(self.yellow_bans[scope.scope_id])
                support = self.episode.support()
                if support.kind is SupportKind.FINITE and not any(a not in self.yellow_bans[scope.scope_id] for a in support.commands):
                    return self._yield_yellow(scope, pending_action, 'LOCAL_ALTERNATIVES_EXHAUSTED')
            parsed, request_id = self._yellow_conclusion(scope, incident_id)
            if parsed is None:
                invalid_proposal = self.yellow_invalid_proposal
                self.yellow_invalid_proposal = False
                if invalid_proposal and not self.yellow_callability_refund_used and not self.alfworld_event_memory:
                    self.attempt_scope_entries["YELLOW"] = max(0, self.attempt_scope_entries["YELLOW"] - 1)
                    self.yellow_entry_counts[incident_id] = max(0, self.yellow_entry_counts.get(incident_id, 0) - 1)
                    self.yellow_callability_refund_used = True
                    self.emit("yellow_entry_refunded", scope_id=scope.scope_id,
                              incident_id=incident_id, reason="UNEXECUTED_INVALID_PROPOSAL",
                              environment_actions_used=0)
                return self._yield_yellow(
                    scope, pending_action,
                    "MODEL_YIELD_OR_CALLABILITY_RESUBMISSIONS_EXHAUSTED",
                    suppress_incident=not invalid_proposal)
            step_role = scope.correction_record["step_role"]
            repeated_proposal = self.repetition.repeated(before, parsed.action)
            duplicate_terminal = self._pending_duplicate(parsed.action)
            if parsed.action == pending_action:
                self.emit("yellow_original_action_selected", scope_id=scope.scope_id,
                          action=pending_action, request_id=request_id,
                          incident_id=incident_id)
            if (self.pending_yellow_action is None or self.pending_yellow_action["request_id"] != request_id
                    or self.pending_yellow_action["action"] != parsed.action):
                raise ProtocolError("Yellow execution differs from its single-step binding")
            receipt, occurrence = self._execute(scope, parsed, request_id)
            self.pending_yellow_action = None
            certificate = getattr(self, 'transition_certificates', {}).get(occurrence.occurrence_id, {})
            task_progress = certificate.get('task_progress') is True
            recovered = (step_role == "CORRECT" and repeated_proposal is None and duplicate_terminal is None
                         and task_progress)
            if self.progress_recovery and incident_kind == "INVALID_ACTION":
                # The public invocation obstacle, not the whole task strategy, is the repair target.
                recovered = repeated_proposal is None and duplicate_terminal is None and task_progress
            if self.yellow_action_ban:
                recovered = receipt['native_accepted'] and parsed.action not in self.yellow_bans[scope.scope_id]
            if self.tree_internal_repair:
                recovered = receipt['native_accepted'] and (receipt['done'] or
                    self.repetition.revision > entry_revision or (incident_kind == 'INVALID_ACTION'
                    and operation_key(before) != operation_key(receipt['after'])))
                self.yellow_local_decisions[scope.scope_id].setdefault(operation_key(before), []).append(parsed.action)
                self.emit('yellow_progress_assessed', scope_id=scope.scope_id,
                    entry_evidence_revision=entry_revision, current_evidence_revision=self.repetition.revision,
                    correction_completed=recovered, actions_used=scope.action_serial)
            for record in (scope.correction_record, scope.correction_records[-1], self.latest_yellow_correction):
                record["executed_occurrence_id"] = occurrence.occurrence_id
                record["correction_completed"] = recovered
            self.emit("yellow_action_binding_released", scope_id=scope.scope_id, action=parsed.action,
                      occurrence_id=occurrence.occurrence_id, request_id=request_id,
                      step_role=step_role, accepted=receipt["native_accepted"],
                      correction_completed=recovered, terminal=receipt["done"])
            if receipt["done"]:
                if receipt["public_terminal_score"] == 1.0:
                    return "SUCCESS", scope.action_serial
                return "TERMINAL_FAILURE", scope.action_serial
            state_changed = (receipt["native_accepted"]
                             and not receipt["done"]
                             and operation_key(before) != operation_key(receipt["after"]))
            return_to_green = recovered or incident_id.endswith("#continuation")
            if return_to_green:
                self._return_to_green(
                    scope, "VALID_CORRECT_ACTION_EXECUTED" if recovered else "ACCEPTED_LOCAL_PREPARATION",
                    handoff_directive=(
                        "The local Yellow recovery segment has returned control to "
                        "Green after executing a local action: " + parsed.action
                        + ". This single-step binding has been released; the action is historical. The original "
                        "Green action that triggered this incident was "
                        + pending_action
                        + ". Make a fresh ordinary decision from the current state. "
                        "The triggering proposal is not automatically executed. "
                        "This state change is not evidence of task success."
                    ))
                return "RETURN_GREEN", scope.action_serial
            if recovery_segment and operation_key(receipt['after']) in segment_states:
                return self._yield_yellow(scope,pending_action,'KNOWN_STATE_CYCLE_WITHOUT_NEW_EVIDENCE')
            if self.alfworld_event_memory:
                self.emit('yellow_local_step_unresolved', scope_id=scope.scope_id,
                          action=parsed.action, step_role=step_role,
                          accepted=receipt['native_accepted'], shared_actions_remaining=self.episode.action_limit-len(self.lineage))
                if repeated_proposal is not None or duplicate_terminal is not None:
                    return self._yield_yellow(scope, pending_action, 'REPEATED_LOCAL_REPAIR_WITHOUT_NEW_EVIDENCE')
                return self._continue_yellow_segment(
                    scope, pending_action, incident_id, incident_kind)
            segment_states.add(operation_key(receipt['after']))
            if self.progress_recovery and not self.yellow_action_ban and (not receipt["native_accepted"] or repeated_proposal is not None
                                           or before == receipt["after"]):
                return self._yield_yellow(scope, pending_action, "REJECTED_OR_REPEATED_WITHOUT_PROGRESS")
        if self._forward_window()["new_actions_remaining"] > 0:
            return self._yield_yellow(scope, pending_action, "LOCAL_STEP_UNRESOLVED")
        self.controller_events.append({"node_kind": NodeKind.ESCALATE.value,
                                       "scope_id": scope.scope_id,
                                       "reason": "ACTION_WINDOW_EXHAUSTED",
                                       "incident_id": incident_id})
        return "ESCALATE_RED", scope.action_serial

    def _return_to_green(self, scope: ScopeTree, reason: str,
                         handoff_directive: str | None = None):
        if scope.kind is not ScopeKind.YELLOW or scope.caller_scope_id is None:
            raise ProtocolError("only a yellow action segment can merge into green")
        caller = self.scopes[scope.caller_scope_id]
        scope.exit_reason = reason
        if self.yellow_action_ban:
            self.episode.yellow_forbidden_actions = ()
            self.emit("yellow_action_prohibition_released", scope_id=scope.scope_id,
                      actions=self.yellow_bans[scope.scope_id], reason=reason)
        if self.tree_internal_repair:
            del self.yellow_local_decisions[scope.scope_id]
        # Carry only public rejection evidence into Green.  The evidence is
        # keyed by the exact pre-action public interface/state, so it expires
        # naturally after a real state transition and cannot ban a command in
        # an unrelated state or restored branch.
        if getattr(self, "local_rejections", None):
            self.emit("public_rejection_evidence_carried",
                      scope_id=scope.scope_id,
                      entries=len(self.local_rejections),
                      semantics="same public state only; expires on accepted state change")
        if caller.kind is not ScopeKind.GREEN:
            raise ProtocolError("yellow scope caller is not a green scope")
        cursor = caller.current_node_id
        merged = []
        for occurrence_id in scope.occurrence_ids:
            occurrence = self.occurrences[occurrence_id]
            if self.lineage[occurrence.ordinal].occurrence_id != occurrence_id:
                raise ProtocolError("Yellow merge occurrence is not on the active lineage")
            cursor = caller.tree.add_edge(
                cursor, occurrence.action, occurrence.receipt,
                visit_kind="YELLOW_MERGED", attempt=occurrence.ordinal,
                request_id=occurrence.request_id)
            merged.append(occurrence.ordinal)
        caller.tree.current_id = cursor
        scope.archived = True
        if not self.call_frames or self.call_frames[-1].callee_scope_id != scope.scope_id:
            raise ProtocolError("Yellow return has no matching call frame")
        self.call_frames.pop()
        self.active_scope_id = caller.scope_id
        self.green_history_handoff = {
            "anchor_occurrence_id": (
                self.occurrences[scope.occurrence_ids[0]].execution_parent_occurrence_id
                if scope.occurrence_ids else self.lineage[-1].occurrence_id if self.lineage else None),
            "executed_occurrence_ids": tuple(scope.occurrence_ids),
            "shown_request_ids": [],
        }
        if handoff_directive is not None:
            if (type(handoff_directive) is not str or not handoff_directive.strip()):
                raise ProtocolError("green handoff directive must be nonempty text")
            self.green_handoff_directive = handoff_directive
        self.controller_events.append({
            "node_kind": NodeKind.RETURN.value,
            "from_scope_id": scope.scope_id,
            "to_scope_id": caller.scope_id,
            "reason": reason,
            "merged_occurrences": merged,
            "handoff_directive": handoff_directive,
        })
        self.emit("scope_returned", from_scope_id=scope.scope_id,
                  to_scope_id=caller.scope_id, to_kind=ScopeKind.GREEN.value,
                  reason=reason, merged_occurrences=merged,
                  actual_state=self.episode.snapshot())

    def _yellow_normal_state(self, receipt: dict[str, Any]) -> bool:
        """A yellow scope is a real action segment, not a one-request probe."""
        before = {key: value for key, value in receipt["before"].items() if key != "observation"}
        after = {key: value for key, value in receipt["after"].items() if key != "observation"}
        if not receipt["native_accepted"] or before == after:
            return False
        if self.progress_recovery:
            return not receipt["done"]
        # Information subpages are still a local detour. Continue inside yellow
        # until a normal init/search/item decision surface is reached.
        if receipt["after"].get("benchmark") == "alfworld":
            return not receipt["done"]
        return receipt["after"]["page_type"] in {"init", "search", "item"}

    def _pending_action_is_callable(self, action: str) -> bool:
        support = self.episode.support()
        parsed = self.episode.parse(action)
        return (parsed.action == action and parsed.grammar_valid
                and not parsed.is_think and self.episode.public_physical(parsed, support))

    def _red_repair_action(self, state: dict[str, Any]):
        raise ProtocolError("Red continuation is not defined for this controller")

    def _red_forbidden_actions(self, commands):
        current = self.episode.snapshot()
        state_key = self._public_state_key(current)
        persistent = getattr(self, 'persistent_cycle_edges', {}).get(state_key, {})
        forbidden = set(self.rejected_public_actions.get(state_key, set()))
        forbidden.update(persistent)
        navigation_forbidden = getattr(self, 'navigation_forbidden', set())
        for action in commands:
            if self.repetition.repeated(current, action) is not None:
                forbidden.add(action)
            if (action.startswith('go to ')
                    and action[6:].strip() in navigation_forbidden):
                forbidden.add(action)
        return tuple(action for action in commands if action not in forbidden)

    def _red_repair_evidence(self, receipt: dict[str, Any], action: str) -> bool:
        """Use only a public task-effect receipt as the Red/Green boundary."""
        return task_related_progress(self.episode.initial['observation'], receipt, action)

    def _run_red(self, reason: str):
        self.green_history_handoff = None
        if not self.lineage:
            raise ProtocolError("red scope requires a nonempty actual execution lineage")
        last = self.lineage[-1].receipt
        terminal_failure = last["done"] and last["public_terminal_score"] != 1.0
        if last["done"] and not terminal_failure:
            raise ProtocolError("successful terminals cannot enter Red")
        if not terminal_failure and self._forward_window()['new_actions_remaining'] != 0:
            raise ProtocolError("red requires an exhausted action window")
        if not terminal_failure:
            reason = 'GREEN_ACTION_LIMIT'
        self.emit('red_entry_gate_verified', reason=reason, terminal_failure=terminal_failure,
                  path_length=len(self.lineage), window=self._forward_window())
        self.green_handoff_directive = None
        caller = self.active_scope
        if caller.kind is ScopeKind.YELLOW:
            caller.archived = True
            caller.exit_reason = reason
            if not self.call_frames or self.call_frames[-1].callee_scope_id != caller.scope_id:
                raise ProtocolError("Yellow escalation has no matching call frame")
            self.call_frames.pop()
        if self.active_trial_id is not None:
            outcome = "TERMINAL_FAILURE" if terminal_failure else "WINDOW_EXHAUSTED"
            self._finish_forward_trial(outcome, reason)
        if self.active_review_session_id is not None:
            raise ProtocolError("new failure cannot preempt an active snapshot review")
        if not self.enable_red:
            self.emit("red_disabled_window_to_root", path_length=len(self.lineage))
            return self._root_fallback(reason="ACTION_WINDOW_EXHAUSTED_RED_DISABLED")
        signature = canonical({"path": [{"action": o.action, "before": o.receipt["before"],
                "after": o.receipt["after"], "accepted": o.receipt["native_accepted"],
                "observation": o.receipt["observation"], "done": o.receipt["done"]} for o in self.lineage]})
        if signature in self.reviewed_failure_paths:
            self.emit("red_duplicate_failure_skipped", path_occurrence_ids=[o.occurrence_id for o in self.lineage],
                      reason="SAME_PUBLIC_PATH_WITHOUT_NEW_EVIDENCE", destination="STOP_POLICY")
            return self._root_fallback(reason="REPEATED_FAILURE_PATH")
        if self.attempt_scope_entries["RED"] >= self.RED_ENTRY_LIMIT:
            return self._scope_limit_to_root("RED")
        self.reviewed_failure_paths.add(signature)
        self.attempt_scope_entries["RED"] += 1
        self._enqueue_failure(reason)
        self._activate_next_review(caller)
        return self._advance_review()

    def _handle_terminal(self):
        if not self.lineage or not self.lineage[-1].receipt["done"]:
            raise ProtocolError("terminal routing requires a real terminal receipt")
        success = self.lineage[-1].receipt["public_terminal_score"] == 1.0
        if self.active_trial_id is not None:
            self._finish_forward_trial("SUCCESS" if success else "TERMINAL_FAILURE", "REAL_TERMINAL")
        if not success:
            # A real non-success terminal is a failed trajectory and must be
            # diagnosed by Red before the optional Root fallback is used.
            return self._run_red("NON_SUCCESS_TERMINAL")
        for session in self.review_sessions.values():
            if session.status in {"ACTIVE", "SUSPENDED_FOR_FORWARD_TRIAL"}:
                session.status = "TASK_SUCCESS" if success else "CLOSED_FOR_TERMINAL_ROOT_RETRY"
                self.scopes[session.red_scope_id].archived = True
        self.emit("terminal_routed", score=self.lineage[-1].receipt["public_terminal_score"],
                  destination="SUCCESS" if success else "ROOT_RETRY",
                  cancelled_failure_ids=[row["source_failure_event_id"] for row in self.review_queue])
        self.review_queue.clear()
        self.active_review_session_id = None
        if self.active_scope.kind is ScopeKind.YELLOW:
            if self.yellow_action_ban:
                self.episode.yellow_forbidden_actions = ()
                self.emit("yellow_action_prohibition_released", scope_id=self.active_scope.scope_id,
                          actions=self.yellow_bans[self.active_scope.scope_id], reason="REAL_TERMINAL")
            self.active_scope.archived = True
            self.call_frames.pop()
        if success:
            self.status = "SUCCESS"
            self._release_checkpoints(final=True)
            return "STOP"
        return self._root_fallback(reason="NON_SUCCESS_TERMINAL")

    def _enqueue_failure(self, reason: str):
        if not self.lineage:
            raise ProtocolError("cannot queue an empty failure path")
        record = {"source_failure_event_id": "failure-" + str(self.failure_serial),
                  "path_occurrence_ids": tuple(o.occurrence_id for o in self.lineage),
                  "reason": reason}
        self.failure_serial += 1
        self.review_queue.append(record)
        self.emit("review_failure_queued", **record, queue_length=len(self.review_queue))

    def _activate_next_review(self, caller: ScopeTree):
        if self.active_trial_id is not None or not self.review_queue:
            raise ProtocolError("next review requires a queued path and no active trial")
        failure = self.review_queue.popleft()
        scope = self._new_scope(ScopeKind.RED, failure["reason"], caller=caller)
        session = ReviewSession(scope.scope_id, scope.scope_id,
                                failure["source_failure_event_id"], failure["path_occurrence_ids"],
                                len(failure["path_occurrence_ids"]) - 1)
        self.review_sessions[session.session_id] = session
        self.active_review_session_id = session.session_id
        self.emit("red_review_session_started", **asdict(session))
        return scope

    def _joint_red_choice(self, scope, session):
        path = [self.occurrences[k] for k in session.frozen_path_occurrence_ids]
        candidates = [(i, self.checkpoint_cache[o.occurrence_id]) for i, o in enumerate(path)]
        candidates.append((len(path), self.tip_checkpoint))
        remaining = {i: self.episode.action_limit - i for i, _ in candidates}
        public_path = [{"occurrence_id": o.occurrence_id,
                        "execution_parent_occurrence_id": o.execution_parent_occurrence_id,
                        "action": o.action, "scope_kind": o.scope_kind.value,
                        "accepted": o.receipt['native_accepted'], "done": o.receipt['done']}
                       for o in path]
        if self.typed_scope_choices:
            assessment = None if self.tree_internal_repair else self._typed_request(scope, red_selection_prompt(self.episode, candidates,
                public_path, self.terminal_evidence, phase='assess'))
            assessment_text = None if assessment is None else assessment.text
            mapping = {f'checkpoint[{i}]':i for i,cached in candidates
                       if remaining[i]>0 and self._checkpoint_can_reach_terminal(cached,i)}
            mapping['UNKNOWN'] = None
            prompt = red_selection_prompt(self.episode, candidates, public_path, self.terminal_evidence,
                phase='choose', assessment=assessment_text, choices=tuple(mapping))
            generation = self._typed_request(scope, prompt, commands=tuple(mapping))
            index = mapping[generation.text]
            self.checkpoint_inspection_count += len(candidates)
            self.red_joint_choices[scope.scope_id] = {'checkpoint_index':index, 'reason':assessment_text,
                'action_plan':None, 'request_id':generation.request.request_id,
                'assessment_request_id':None if assessment is None else assessment.request.request_id, 'candidate_indices':list(remaining),
                'choice':generation.text, 'protocol':'TYPED_FINITE_CHECKPOINT_CHOICE'}
            decision = ReviewDecision.UNKNOWN if index is None else ReviewDecision.RESUME
            output = canonical({'model_choice':generation.text,'model_assessment':assessment_text})
            if self.efficient_recovery:
                output = json.dumps({'model_choice':generation.text,'model_assessment':assessment_text},
                                    ensure_ascii=False, separators=(',', ':'))
            self.reviews.append(ReviewRecord(scope.scope_id,index,decision,generation.request.request_id,prompt,output))
            self.emit('red_joint_selection_completed',scope_id=scope.scope_id,**self.red_joint_choices[scope.scope_id],
                      environment_restores=0,output=output)
            self.emit('red_review_completed',scope_id=scope.scope_id,checkpoint_index=index,decision=decision.value,
                      request_id=generation.request.request_id,output=output,joint_selection=True)
            return index,generation.request.request_id
        live = self.episode.checkpoint()
        feedback = None
        for attempt in range(2):
            prompt = red_selection_prompt(self.episode, candidates, public_path, self.terminal_evidence, feedback=feedback)
            address = self._address(scope)
            self.oracle.bind(address, self.episode, prompt, ())
            generation = self.oracle.request(address, Role.REFLECTION, 0, prompt, self.ledger,
                output_limit=self._reflection_output_limit(prompt), stop_mode='json_object' if self.output_framing else None)
            if self.episode.checkpoint() != live:
                raise ProtocolError("joint Red selection mutated the live environment")
            self._require_reflection_eos(generation)
            try:
                row = parse_red_selection(generation.text, remaining)
                index = row['checkpoint_index']
                if index is not None:
                    cached = candidates[index][1]
                    if not self._checkpoint_can_reach_terminal(cached, index):
                        raise RedSelectionError("selected state cannot meet the public minimum completion budget")
                    if not self.episode.checkpoint_action_callable(cached, row['action_plan'][0]):
                        raise RedSelectionError("the plan's first action is not callable in the selected snapshot")
            except RedSelectionError as error:
                feedback = {"request_id": generation.request.request_id, "unexecuted_output": generation.text,
                            "error": str(error)}
                self.emit('red_selection_rejected', scope_id=scope.scope_id, attempt=attempt,
                          resubmission_allowed=attempt == 0, **feedback)
                if attempt == 1:
                    raise
            else:
                break
        self.checkpoint_inspection_count += len(candidates)
        self.red_joint_choices[scope.scope_id] = {**row, "request_id": generation.request.request_id,
                                                 "candidate_indices": list(remaining)}
        decision = ReviewDecision.UNKNOWN if index is None else ReviewDecision.RESUME
        record = ReviewRecord(scope.scope_id, index, decision, generation.request.request_id, prompt, generation.text)
        self.reviews.append(record)
        self.emit('red_joint_selection_completed', scope_id=scope.scope_id, **self.red_joint_choices[scope.scope_id],
                  environment_restores=0, output=generation.text)
        self.emit('red_review_completed', scope_id=scope.scope_id, checkpoint_index=index,
                  decision=decision.value, request_id=record.request_id, output=record.output, joint_selection=True)
        return index, record.request_id

    def _advance_review(self):
        while self.active_review_session_id is not None:
            session = self.review_sessions[self.active_review_session_id]
            scope = self.scopes[session.red_scope_id]
            if session.status != "ACTIVE" or self.active_scope_id != scope.scope_id:
                raise ProtocolError("review cursor is not owned by the active Red scope")
            if self.joint_red_selection and scope.scope_id not in self.red_joint_choices:
                selected_index, selected_request_id = self._joint_red_choice(scope, session)
                session.next_checkpoint_index = -1 if selected_index is None else selected_index
            if self.batched_red and scope.scope_id not in self.red_batches:
                self._batched_red_review(scope, session.next_checkpoint_index)
                selected_rows = self.red_batches[scope.scope_id]["rows"]
                session.next_checkpoint_index = next(iter(selected_rows)) if selected_rows else -1
            if session.next_checkpoint_index < 0:
                session.status = "COMPLETED"
                scope.archived = True
                self.active_review_session_id = None
                self.emit("red_review_session_completed", session_id=session.session_id,
                          queued_failure_paths=len(self.review_queue))
                return self._root_fallback()
            cursor = session.next_checkpoint_index
            session.next_checkpoint_index = -1 if self.joint_red_selection or self.batched_red else cursor - 1
            session.current_checkpoint_index = cursor
            path = [self.occurrences[key] for key in session.frozen_path_occurrence_ids]
            occurrence = path[cursor]
            cached = self.checkpoint_cache[occurrence.occurrence_id]
            live_before = self.episode.checkpoint()
            if not self.joint_red_selection:
                self.checkpoint_inspection_count += 1
            self.emit("red_checkpoint_inspected", scope_id=scope.scope_id,
                      checkpoint_index=cursor, cached_state=self._checkpoint_public_state(cached),
                      live_environment_changed=False, cache_branch_id=self.cache_branch_id)
            if self.joint_red_selection:
                decision, request_id = ReviewDecision.RESUME, self.red_joint_choices[scope.scope_id]['request_id']
            elif self.batched_red:
                row = self.red_batches[scope.scope_id]["rows"][cursor]
                decision, request_id = ReviewDecision.RESUME, self.red_batches[scope.scope_id]["request_id"]
            else:
                decision, request_id = self._red_review(scope, cursor)
            if self.episode.checkpoint() != live_before:
                raise ProtocolError("snapshot review mutated the live environment")
            session.current_checkpoint_index = None
            if decision is not ReviewDecision.RESUME:
                continue
            if self.batched_red and self.red_batches[scope.scope_id]["rows"][cursor]["restore_step"] is None:
                self.emit("red_replacement_rejected", scope_id=scope.scope_id,
                          checkpoint_index=cursor, reason="RESTORE_STEP_MISMATCH",
                          reported_restore_step=self.red_batches[scope.scope_id]["rows"][cursor]
                          ["reported_restore_step"])
                continue
            if not self._checkpoint_can_reach_terminal(cached, cursor):
                self.emit("red_resume_rejected_insufficient_budget", scope_id=scope.scope_id,
                          checkpoint_index=cursor,
                          remaining_actions=self.episode.action_limit - cursor)
                continue
            if self.batched_red:
                row = self.red_batches[scope.scope_id]["rows"][cursor]
                if not self.episode.checkpoint_action_callable(cached, row["replacement_action"]):
                    self.emit("red_replacement_rejected", scope_id=scope.scope_id,
                              checkpoint_index=cursor, action=row["replacement_action"],
                              reason="NOT_CALLABLE_IN_CACHED_PUBLIC_STATE",
                              environment_restored=False, terminal_inferred=False,
                              disposition="CONTROLLED_RED_DECLINE")
                    continue
            environment_call(self.episode, self.ledger, self.episode.restore, cached)
            self.tip_checkpoint = deepcopy(cached)
            self.checkpoint_restore_count += 1
            if self.episode.checkpoint() != cached:
                raise ProtocolError("selected checkpoint restore differs from its saved environment/history")
            self.emit("red_checkpoint_restored", scope_id=scope.scope_id,
                      checkpoint_index=cursor, actual_state=self.episode.snapshot(),
                      review_request_id=request_id,
                      restore_guarantee="LOCAL_SESSION_AND_ACTOR_HISTORY_RESTORED")
            self.archived_lineages.append(list(self.lineage))
            self.lineage = list(path[:cursor])
            self._replace_branch_cache()
            # The restored branch must not forget public failures observed on
            # its retained prefix.  Rebuild the rejection ledger from that
            # prefix; discarded suffix failures are intentionally excluded.
            self.rejected_public_actions = {}
            for retained in self.lineage:
                if retained.receipt.get("invalid_action"):
                    key = self._public_state_key(retained.receipt["before"])
                    self.rejected_public_actions.setdefault(key, set()).add(retained.action)
            green = self._green_scope()
            self._set_green_cursor_from_lineage(self.lineage)
            if self.batched_red:
                row = self.red_batches[scope.scope_id]["rows"][cursor]
                candidate = self.episode.parse(row["replacement_action"])
                self.pending_red_action = {**deepcopy(row), "action": candidate.action, "request_id": request_id,
                                          "restore_step": row["restore_step"], "evidence": row["evidence"]}
            trial = ForwardTrial(
                trial_id=len(self.forward_trials), owner_review_session_id=session.session_id,
                checkpoint_index=cursor, prefix_occurrence_ids=tuple(o.occurrence_id for o in self.lineage),
                source_checkpoint_occurrence_id=occurrence.occurrence_id,
                path_window_limit=self.episode.action_limit,
                forward_actions_granted=self.episode.action_limit - cursor,
                review_request_id=request_id, advice=canonical(row) if self.batched_red else self.reviews[-1].output,
                start_cost=self.ledger.to_dict(), start_restore_count=self.checkpoint_restore_count)
            self.forward_trials[trial.trial_id] = trial
            session.trial_ids.append(trial.trial_id)
            self.active_trial_id = trial.trial_id
            from alf_repair_memory import RepairMemory
            self.active_red_memory = RepairMemory(
                row['continuation_advice'] if self.batched_red else trial.advice,
                cursor, row['replacement_action'] if self.batched_red else None,
                facts={
                    'restored_checkpoint_index': cursor,
                    'current_public_state': self.episode.snapshot(),
                    'failed_action_edges': [
                        {'before': o.receipt['before'], 'action': o.action,
                         'after': o.receipt['after'], 'accepted': o.receipt['native_accepted']}
                        for o in path[cursor:]],
                    'non_reproducible_state_action_edges': (
                        self.persistent_cycle_constraints()
                        if hasattr(self, 'persistent_cycle_constraints') else []),
                    'required_next_evidence': 'new task-related public evidence',
                })
            self.emit('red_memory_activated', trial_id=trial.trial_id,
                      checkpoint_index=cursor, memory=self.active_red_memory.actor_context(),
                      lifetime='THIS_FORWARD_TRIAL')
            session.status = "SELECTED_AND_CLOSED"
            scope.archived = True
            self.active_review_session_id = None
            self.active_scope_id = green.scope_id
            if self.tree_internal_repair:
                self.pending_recovery_yellow = {
                    'reason': 'RED_RESUME_FAILED_SUFFIX_REPLACEMENT',
                    'pending_action': occurrence.action,
                    'incident_id': 'red-resume-' + str(session.session_id),
                    'incident_kind': 'REPEATED_DECISION'}
            self.green_handoff_directive = (
                "Red selected this checkpoint for normal continuation, not a verified repair. "
                "Make a fresh ordinary decision within FORWARD_WINDOW.\n"
                "[RED_REVIEW_ADVICE]\n" + trial.advice)
            self.controller_events.append({"node_kind": NodeKind.RETURN.value,
                "from_scope_id": scope.scope_id, "to_scope_id": green.scope_id,
                "reason": "RED_RESUME", "checkpoint_index": cursor, "trial_id": trial.trial_id})
            self.emit("resume_trial_started", **asdict(trial))
            self.emit("scope_returned", from_scope_id=scope.scope_id, to_scope_id=green.scope_id,
                      to_kind="GREEN", reason="RED_RESUME", checkpoint_index=cursor,
                      review_request_id=request_id, trial_id=trial.trial_id,
                      actual_state=self.episode.snapshot())
            return "RETURN_GREEN"
        raise ProtocolError("review advancement requires an active review session")

    def _replace_branch_cache(self):
        # Restoration ownership is branch-local; public repetition evidence is task-wide.
        old_branch = self.cache_branch_id
        old_ids = set(self.checkpoint_cache)
        prefix_ids = [o.occurrence_id for o in self.lineage]
        self.checkpoint_cache = {key: deepcopy(self.checkpoint_cache[key]) for key in prefix_ids}
        self.cache_branch_id += 1
        self.emit("branch_cache_replaced", previous_branch_id=old_branch,
                  cache_branch_id=self.cache_branch_id, inherited_prefix_occurrence_ids=prefix_ids,
                  discarded_suffix_occurrence_ids=sorted(old_ids - set(prefix_ids)),
                  retained_repetition_transitions=len(self.repetition.transitions),
                  repetition_evidence_revision=self.repetition.revision)

    def _scope_limit_to_root(self, kind):
        reason = kind + "_ENTRY_LIMIT"
        self.emit("scope_entry_limit_reached", kind=kind,
                  root_attempt=self.root_recoveries,
                  entries=self.attempt_scope_entries[kind], destination="ROOT_RETRY")
        if self.active_trial_id is not None:
            self._finish_forward_trial("SCOPE_ENTRY_LIMIT", reason)
        self.green_handoff_directive = None
        return self._root_fallback(reason=reason)

    def _root_fallback(self, *, reason="RED_REVIEW_EXHAUSTED"):
        if self.review_queue or self.active_trial_id is not None or self.active_review_session_id is not None:
            raise ProtocolError("Red exhausted with pending work")
        self.status = "RED_REPAIR_FAILED" if self.enable_red else "TERMINAL_FAILURE"
        self.emit("red_recovery_stopped", reason=reason, root_rollback_removed=True)
        self._release_checkpoints(final=True)
        return "STOP"

    def _local_rejection_for_proposal(self, action):
        return None

    def _local_obstruction_reason(self):
        return ('two consecutive rejected actions left the public state unchanged; latest action='
                if self.alfworld_event_memory else
                'the previous native action was rejected by the environment interface: action=')

    def run(self) -> dict[str, Any]:
        if self.status != "ACTIVE":
            raise ProtocolError("scoped recovery must be started before run")
        segment_actions = len(self.lineage)
        while self.status == "ACTIVE":
            scope = self.active_scope
            if scope.kind is not ScopeKind.GREEN:
                raise ProtocolError("ordinary loop entered with a non-green scope")
            if self._forward_window()["new_actions_remaining"] == 0:
                # Red owns an unproductive repair branch until it observes a
                # task-related public effect; never leak it to free Green.
                self.active_red_repair = None
                result = self._run_red("GREEN_ACTION_LIMIT")
                if result == "STOP":
                    break
                segment_actions = len(self.lineage)
                continue
            directive = self.green_handoff_directive
            if directive is None:
                directive = "Continue the original task with one native action."
            if self.pending_recovery_yellow is not None:
                request = self.pending_recovery_yellow
                self.pending_recovery_yellow = None
                yellow_result, _ = self._run_yellow(**request)
                if yellow_result in {'SUCCESS', 'TERMINAL_FAILURE'}:
                    if self._handle_terminal() == 'STOP':
                        break
                if yellow_result == 'STOP':
                    break
                segment_actions = len(self.lineage)
                continue
            if self.yellow_continuation is not None:
                continuation = self.yellow_continuation
                self.yellow_continuation = None
                yellow_result, _ = self._run_yellow(**continuation)
                if yellow_result in {'SUCCESS', 'TERMINAL_FAILURE'}:
                    if self._handle_terminal() == 'STOP':
                        break
                if yellow_result == 'STOP':
                    break
                segment_actions = len(self.lineage)
                continue
            red_proposal = self.pending_red_action
            if red_proposal is not None:
                proposal = self.pending_red_action
                self.pending_red_action = None
                parsed = self.episode.parse(proposal["action"])
                if (parsed.action != proposal["action"] or not parsed.grammar_valid
                        or parsed.is_think or not self.episode.public_physical(parsed, self.episode.support())):
                    raise ProtocolError("validated Red replacement_action became non-callable before execution")
                request_id = proposal["request_id"]
                self.emit("red_replacement_submitted", action=parsed.action, request_id=request_id,
                          trial_id=self.active_trial_id, restore_step=proposal["restore_step"],
                          routing="SHARED_LOCAL_INCIDENT_CHECK")
            elif self.active_red_repair is not None:
                parsed, request_id = self._red_repair_action(self.active_red_repair)
            else:
                parsed, request_id = self._ordinary(scope, directive)
            if (self._rejected_action_in_current_state(parsed.action)
                    and red_proposal is None):
                result = self._suppress_rejected_proposal(scope, parsed.action)
                if result == "STOP":
                    break
                continue
            duplicate = self._pending_duplicate(parsed.action)
            repeated = (self._local_rejection_for_proposal(parsed.action)
                        or self.repetition.repeated(self.episode.snapshot(), parsed.action))
            incident_id = duplicate.incident_id if duplicate is not None else (
                repeated["incident_id"] if repeated is not None else None)
            if incident_id is not None:
                entries = self.yellow_entry_counts.get(incident_id, 0)
                self.emit("yellow_incident_detected", incident_id=incident_id,
                          kind="REPEATED_TERMINAL" if duplicate is not None else repeated["kind"],
                          action=parsed.action, prior_yellow_entries=entries,
                          observed_occurrence_ids=[] if repeated is None else repeated["observed_occurrence_ids"])
            red_owned = red_proposal is not None or self.active_red_repair is not None
            if (incident_id is not None and self.enable_yellow and not red_owned
                    and self._yellow_eligible("REPEATED_TERMINAL" if duplicate is not None else "REPEATED_DECISION", parsed.action)):
                reason = ("the proposed complete result matches an observed non-success result "
                          + duplicate.result_key if duplicate is not None else
                          "a previously executed decision is being repeated without new public evidence: "
                          + canonical({k: v for k, v in repeated.items() if k != "region"}))
                result = self._run_yellow(
                    reason, parsed.action, incident_id,
                    incident_kind="REPEATED_TERMINAL" if duplicate is not None else "REPEATED_DECISION")
                yellow_result, yellow_actions = result
                if red_proposal is not None and yellow_result != "BYPASS_YELLOW":
                    self.emit("red_replacement_reconsidered", action=parsed.action,
                              request_id=request_id, incident_id=incident_id,
                              yellow_result=yellow_result, yellow_actions=yellow_actions,
                              original_proposal_executed=False)
                if yellow_result in {"SUCCESS", "TERMINAL_FAILURE"}:
                    if self._handle_terminal() == "STOP":
                        break
                    segment_actions = len(self.lineage)
                    continue
                if yellow_result == "RETURN_GREEN":
                    segment_actions = len(self.lineage)
                    continue
                if yellow_result == "BYPASS_YELLOW":
                    segment_actions = len(self.lineage)
                    # Fall through to the ordinary execution path below.
                    pass
                elif yellow_result == "STOP":
                    break
                else:
                    # Yellow is a local repair scope.  An unresolved local
                    # diagnosis does not consume the Green action window and
                    # cannot authorize Red before the window is exhausted.
                    # Red is entered only by the explicit gate below.
                    if yellow_result != "ESCALATE_RED":
                        raise ProtocolError(
                            "Yellow returned an unknown control state: " + str(yellow_result))
                    if self._forward_window()['new_actions_remaining'] != 0:
                        raise ProtocolError(
                            "Yellow requested Red before the Green action window was exhausted")
                    result = self._run_red("GREEN_ACTION_LIMIT")
                    if result == "STOP":
                        break
                    segment_actions = len(self.lineage)
                    continue
            receipt, _ = self._execute(scope, parsed, request_id)
            if red_proposal is not None:
                self.emit("red_replacement_executed", action=parsed.action,
                          restore_step=red_proposal["restore_step"], evidence=red_proposal["evidence"],
                          terminal=receipt["done"], accepted=receipt["native_accepted"])
                self.red_handoff_directive = canonical({
                    'checkpoint_index': red_proposal['restore_step'],
                    'replacement_action': parsed.action,
                    'continuation_plan': red_proposal['replacement_plan'][1:],
                    'continuation_advice': red_proposal['continuation_advice'],
                    'evidence': red_proposal['evidence'],
                    'failed_suffix_to_avoid': red_proposal.get('failed_suffix', []),
                    'accepted': receipt['native_accepted'],
                    'observation': receipt['observation']})
                self.active_red_repair = {
                    'checkpoint_index': red_proposal['restore_step'],
                    'advice': red_proposal['continuation_advice'],
                    'evidence': red_proposal['evidence'],
                    'actions_used': 1,
                    'actions_limit': 4,
                    'segments_completed': 0,
                }
                self.emit('red_repair_segment_started',
                          checkpoint_index=red_proposal['restore_step'],
                          first_action=parsed.action, actions_limit=4,
                          accepted=receipt['native_accepted'])
            elif self.active_red_repair is not None:
                self.active_red_repair['actions_used'] += 1
                self.emit('red_repair_segment_action_executed',
                          action=parsed.action,
                          checkpoint_index=self.active_red_repair['checkpoint_index'],
                          actions_used=self.active_red_repair['actions_used'],
                          actions_limit=self.active_red_repair['actions_limit'],
                          accepted=receipt['native_accepted'],
                          observation=receipt['observation'])
                if self._red_repair_evidence(receipt, parsed.action):
                    self.emit('red_repair_evidence_observed',
                              action=parsed.action,
                              checkpoint_index=self.active_red_repair['checkpoint_index'],
                              evidence_kind='PUBLIC_TASK_EFFECT')
                    self.active_red_repair = None
                elif self.active_red_repair['actions_used'] >= self.active_red_repair['actions_limit']:
                    self.active_red_repair['segments_completed'] += 1
                    self.active_red_repair['actions_used'] = 0
                    self.emit('red_repair_segment_reopened',
                              checkpoint_index=self.active_red_repair['checkpoint_index'],
                              reason='NO_TASK_PROGRESS_PUBLIC_EVIDENCE',
                              segment=self.active_red_repair['segments_completed'],
                              actions_limit=self.active_red_repair['actions_limit'],
                              remaining_actions=self._forward_window()['new_actions_remaining'])
            segment_actions = len(self.lineage)
            if receipt["done"]:
                result = self._handle_terminal()
                if result == "STOP":
                    break
                segment_actions = len(self.lineage)
                continue
            local_obstruction=receipt["invalid_action"]
            if self.alfworld_event_memory and local_obstruction:
                local_obstruction=self.episode.local_obstruction(receipt)
            if (local_obstruction and self.enable_yellow and not red_owned
                    and self._yellow_eligible("INVALID_ACTION", parsed.action)):
                if segment_actions >= self.episode.action_limit:
                    continue
                incident = self._invalid_incident(receipt)
                result = self._run_yellow(
                    self._local_obstruction_reason() + incident.action
                    + "; detail=" + incident.result_key
                    + "; incident=" + incident.incident_id,
                    incident.action, incident.incident_id, incident_kind="INVALID_ACTION")
                yellow_result, yellow_actions = result
                if yellow_result in {"SUCCESS", "TERMINAL_FAILURE"}:
                    if self._handle_terminal() == "STOP":
                        break
                    segment_actions = len(self.lineage)
                    continue
                if yellow_result in {"RETURN_GREEN", "BYPASS_YELLOW"}:
                    segment_actions = len(self.lineage)
                    continue
                if yellow_result == "STOP":
                    break
                if yellow_result != "ESCALATE_RED":
                    raise ProtocolError(
                        "Yellow returned an unknown control state: " + str(yellow_result))
                if self._forward_window()['new_actions_remaining'] != 0:
                    raise ProtocolError(
                        "Yellow requested Red before the Green action window was exhausted")
                result = self._run_red("GREEN_ACTION_LIMIT")
                if result == "STOP":
                    break
                segment_actions = len(self.lineage)
        if self.status == "ACTIVE":
            self.status = "FAILED"
        if self.status == "SUCCESS" and self.active_trial_id is not None:
            self._finish_forward_trial("SUCCESS", "REAL_TERMINAL_SUCCESS")
        return self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            'sparse_green_event_keys':sorted(self.delivered_green_events),
            'alfworld_event_memory': self.alfworld_event_memory,
            "schema_version": "TREE_INTERNAL_REPAIR_V41" if self.tree_internal_repair else "EFFICIENT_RECOVERY_V39" if self.efficient_recovery else "TYPED_SCOPES_V38" if self.typed_scope_choices else "SCOPED_BAN_JOINT_V37" if self.yellow_action_ban or self.joint_red_selection else "SCOPED_PROGRESS_GRAPH_V36" if self.progress_recovery else "SCOPED_SNAPSHOT_REVIEW_V31",
            "pending_recovery_yellow": deepcopy(self.pending_recovery_yellow),
            "efficient_recovery": self.efficient_recovery,
            "tree_internal_repair": self.tree_internal_repair,
            "yellow_action_ban": self.yellow_action_ban,
            "joint_red_selection": self.joint_red_selection,
            "typed_scope_choices": self.typed_scope_choices,
            "yellow_scope_prohibitions": deepcopy(self.yellow_bans),
            "red_joint_choices": deepcopy(self.red_joint_choices),
            "progress_recovery": self.progress_recovery,
            "progress_recovery_policy": {
                "yellow_local_actions": "floor(remaining_actions_at_entry / 2)",
                "yellow_invalid_proposal_resubmissions": 1,
                "yellow_unresolved_exit": "GREEN_WITH_EVIDENCE",
                "yellow_reentry": "requires_new_public_evidence_for_same_state_incident",
                "red_checkpoint0": "RESUME_OR_UNKNOWN", "red_path_dedup": "public_path_and_evidence_revision"} if self.progress_recovery else None,
            "compact_context": self.compact_context,
            "batched_red": self.batched_red,
            "enable_yellow": self.enable_yellow,
            "enable_red": self.enable_red,
            "terminal_guidance": self.terminal_guidance,
            "yellow_action_ids": self.yellow_action_ids,
            "output_framing": self.output_framing,
            "completed_red_batches": {str(k): {"request_id": v["request_id"], "rows": v["rows"]}
                                      for k, v in self.red_batches.items()},
            "status": self.status,
            "resource_limit": self.resource_limit,
            "active_scope_id": self.active_scope_id,
            "root_recoveries": self.root_recoveries,
            "scope_entry_limits_per_root_attempt": {"RED": self.RED_ENTRY_LIMIT,
                                                   "YELLOW": self.YELLOW_ENTRY_LIMIT},
            "attempt_scope_entries": dict(self.attempt_scope_entries),
            "repetition_memory_scope": "TASK_PUBLIC_EVIDENCE",
            "repetition_transition_count": len(self.repetition.transitions),
            "repetition_root_attempt_start": self.repetition.root_attempt_start,
            "latest_yellow_correction": deepcopy(self.latest_yellow_correction),
            "pending_yellow_action": deepcopy(self.pending_yellow_action),
            "active_red_repair": deepcopy(self.active_red_repair),
            "last_yellow_unresolved": self.last_yellow_unresolved,
            "checkpoint_restore_count": self.checkpoint_restore_count,
            "checkpoint_inspection_count": self.checkpoint_inspection_count,
            "cache_branch_id": self.cache_branch_id,
            "checkpoint_cache_live_ids": sorted(self.checkpoint_cache),
            "tip_checkpoint_cached": self.tip_checkpoint is not None,
            "scopes": [{
                "scope_id": scope.scope_id,
                "kind": scope.kind.value,
                "entry_reason": scope.entry_reason,
                "caller_scope_id": scope.caller_scope_id,
                "caller_node_id": scope.caller_node_id,
                "exit_rules": list(scope.exit_rules),
                "archived": scope.archived,
                "incident_kind": scope.incident_kind,
                "occurrence_ids": list(scope.occurrence_ids),
                "correction_record": deepcopy(scope.correction_record),
                "correction_records": deepcopy(scope.correction_records),
                "event_anchor_occurrence_id": scope.event_anchor_occurrence_id,
                "exit_reason": scope.exit_reason,
                "tree": scope.tree.to_dict(),
            } for scope in self.scopes.values()],
            "lineage": [item.to_dict() for item in self.lineage],
            "occurrences": {str(key): value.to_dict() for key, value in self.occurrences.items()},
            "archived_lineage_occurrence_ids": [[item.occurrence_id for item in path]
                                               for path in self.archived_lineages],
            "review_sessions": [asdict(session) for session in self.review_sessions.values()],
            "review_queue": list(self.review_queue),
            "active_review_session_id": self.active_review_session_id,
            "forward_trials": [{**asdict(trial), "forward_actions_used": trial.forward_actions_used}
                               for trial in self.forward_trials.values()],
            "active_trial_id": self.active_trial_id,
            "call_frames": [asdict(frame) for frame in self.call_frames],
            "reviews": [asdict(review) | {"decision": review.decision.value}
                        for review in self.reviews],
            "controller_events": deepcopy(self.controller_events),
            "terminal_evidence": deepcopy(self.terminal_evidence),
            "yellow_entry_counts": dict(self.yellow_entry_counts),
            "yellow_callability_refund_used": self.yellow_callability_refund_used,
            "green_handoff_pending": self.green_handoff_directive is not None,
            "green_handoff_directive": self.green_handoff_directive,
        }
