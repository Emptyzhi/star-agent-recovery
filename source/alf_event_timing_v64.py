"""Defer local incident prompting until Green proposes a problematic action."""
from alf_rebased_v64 import RebasedV64Controller
from alf_sparse_events import render_sparse_green
from scoped_recovery import ScopeKind

VERSION = 'v64_rebased_proposal_before_local_incident_v5'


class EventTimingV64Controller(RebasedV64Controller):
    def _bind_scope(self, scope, directive):
        if scope.kind is not ScopeKind.GREEN:
            return super()._bind_scope(scope, directive)
        # Handoffs remain visible. A rejection alone must not preempt Green's
        # next proposal; _proposal_incident routes an actual repeat to Yellow.
        view = render_sparse_green(self, preproposal_incidents=False)
        self.episode.set_scope_prompt(view, scope.tree.tree_version,
            kind=scope.kind.value, directive=directive, incident_kind=scope.incident_kind)
        return self.episode.prompt(())

    def to_dict(self):
        return dict(super().to_dict(), controller_version=VERSION,
                    local_event_policy='PROPOSE_THEN_DIAGNOSE_IN_YELLOW')
