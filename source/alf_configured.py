"""Per-task configuration of the preserved ALFWorld v16 controller."""
from alf_goal_grounded_events import GoalGroundedV64Controller
from icr_core import Role
from scoped_recovery import ScopeKind
from experiment_settings import EXECUTION_PROTOCOL
import plain_history


class ConfiguredAlfController(GoalGroundedV64Controller):
    def __init__(self, *args, settings, **kwargs):
        super().__init__(*args, **kwargs)
        self.settings = settings
        self.RED_ENABLED = settings.ablation != 'without_red'
        self.EVENT_SUBTREE_ENABLED = settings.ablation != 'without_event_subtree'
        self.RECOVERY_WINDOW = settings.recovery_actions
        self.YELLOW_ENTRY_LIMIT = 0 if settings.ablation == 'without_yellow' else settings.yellow_entries
        self.YELLOW_ACTION_BUDGET = settings.yellow_actions
        self.DIAGNOSIS_OUTPUT_TOKENS = settings.yellow_diagnosis_tokens
        self.YELLOW_OUTPUT_TOKENS = settings.yellow_output_tokens
        self.episode.action_limit = settings.initial_actions
        if not self.EVENT_SUBTREE_ENABLED:
            plain_history.install(self)

    def _bind_scope(self, scope, directive):
        if self.EVENT_SUBTREE_ENABLED:
            return super()._bind_scope(scope, directive)
        self.pending_green_events = []
        return plain_history.bind(self, scope)

    def _ordinary(self, scope, directive):
        self.oracle.ordinary_output_limit = (self.settings.yellow_output_tokens
            if scope.kind is ScopeKind.YELLOW else self.settings.actor_output_tokens)
        return super()._ordinary(scope, directive)

    def _red_request(self, scope, prompt, *, commands=()):
        address = self._address(scope)
        self.oracle.bind(address, self.episode, prompt, ())
        return self.oracle.request(address, Role.ALTERNATIVE if commands else Role.REFLECTION,
            0, prompt, self.ledger, commands=commands,
            output_limit=self.settings.red_selection_tokens if commands else self.settings.red_reflection_tokens)

    def to_dict(self):
        return dict(super().to_dict(), experiment_settings=self.settings.as_dict(),
                    execution_protocol=EXECUTION_PROTOCOL,
                    ablation=self.settings.ablation, red_enabled=self.RED_ENABLED,
                    event_subtree_enabled=self.EVENT_SUBTREE_ENABLED,
                    ablation_input_policy=(None if self.EVENT_SUBTREE_ENABLED else plain_history.POLICY),
                    yellow_enabled=self.YELLOW_ENTRY_LIMIT > 0,
                    max_red_model_calls=2 if self.RED_ENABLED else 0,
                    max_recovery_actions=self.RECOVERY_WINDOW if self.RED_ENABLED else 0,
                    effective_yellow_action_budget=self.YELLOW_ACTION_BUDGET,
                    effective_yellow_entries=self.YELLOW_ENTRY_LIMIT,
                    max_yellow_diagnosis_calls=2*self.YELLOW_ENTRY_LIMIT,
                    configuration_adapter='instance_scoped_no_global_mutation')
