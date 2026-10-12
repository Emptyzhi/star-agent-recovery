"""Restore native-menu feasibility for bounded v64 Yellow actions."""
from alf_event_timing_v64 import EventTimingV64Controller
from icr_core import ProtocolError, Role
from scoped_recovery import ScopeKind

VERSION='v64_bounded_native_menu_yellow_v7'


class FeasibleYellowV64Controller(EventTimingV64Controller):
    YELLOW_OUTPUT_TOKENS = 100
    def _ordinary(self,scope,directive):
        if scope.kind is not ScopeKind.YELLOW:
            return super()._ordinary(scope,directive)
        state=self.episode.snapshot()
        commands=tuple(state['available_actions'])
        if not commands:
            raise ProtocolError('Yellow has no native command catalog')
        address=self._address(scope)
        self.episode.red_repair_handoff=(None if self.active_red_memory is None
            else self.active_red_memory.actor_context())
        prompt=self._bind_scope(scope,directive)
        self.oracle.bind(address,self.episode,prompt,())
        self.emit('yellow_native_menu_bound',scope_id=scope.scope_id,state=state,
                  commands=list(commands),semantic_filter=False,output_limit=self.YELLOW_OUTPUT_TOKENS)
        generation=self.oracle.request(address,Role.ALTERNATIVE,0,prompt,self.ledger,
            commands=commands,output_limit=self.YELLOW_OUTPUT_TOKENS)
        self.episode.red_repair_handoff=None
        if self.episode.snapshot()!=state:
            raise ProtocolError('Yellow selection state changed during generation')
        if generation.text not in commands:
            self.emit('yellow_native_menu_violation',request_id=generation.request.request_id,
                      raw_output=generation.text,commands=list(commands),action_executed=False)
            raise ProtocolError('Yellow output is outside the bound native menu')
        parsed=self.episode.parse(generation.text)
        if not parsed.grammar_valid or parsed.is_think or parsed.action!=generation.text:
            raise ProtocolError('native Yellow command does not roundtrip through the action parser')
        if self.active_red_memory is not None:
            self.emit('red_memory_delivered',request_id=generation.request.request_id,
                      trial_id=self.active_trial_id,memory=self.active_red_memory.actor_context())
        self.emit('scoped_actor_action_selected',scope_id=scope.scope_id,scope_kind=scope.kind.value,
            request_id=generation.request.request_id,action=parsed.action,prompt=prompt,
            history_window=self.episode.last_history_window,selection_route='complete_native_menu')
        return parsed,generation.request.request_id

    def to_dict(self):
        return dict(super().to_dict(),controller_version=VERSION,
                    yellow_selection_policy='COMPLETE_CURRENT_NATIVE_MENU_NO_GOAL_FILTER')
