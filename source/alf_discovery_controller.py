"""v64 ETT recovery retaining discoveries from the failed branch."""
from alf_feasible_yellow import FeasibleYellowV64Controller
from alf_sparse_events import render_sparse_green
from scoped_recovery import ScopeKind

VERSION='v64_native_history_bounded_yellow_discovery_handoff_v9'


class DiscoveryV64Controller(FeasibleYellowV64Controller):
    def _bind_scope(self,scope,directive):
        if scope.kind is not ScopeKind.GREEN:
            return super()._bind_scope(scope,directive)
        view=render_sparse_green(self,preproposal_incidents=False,include_discarded_discoveries=True)
        self.episode.set_scope_prompt(view,scope.tree.tree_version,
            kind=scope.kind.value,directive=directive,incident_kind=scope.incident_kind)
        return self.episode.prompt(())

    def to_dict(self):
        return dict(super().to_dict(),controller_version=VERSION,
            first_window_history='COMPLETE_NATIVE_ACTIVE_HISTORY',
            discovery_policy='DISTINCT_ACCEPTED_ABANDONED_BRANCH_VIEWS_AT_RED_HANDOFF')
