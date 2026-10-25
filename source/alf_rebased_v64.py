"""Resume v64 ETT execution with a fresh Actor context at the actual checkpoint."""
from alf_native_v16 import SimpleEpisode, SimpleV64Controller
from alf_restored_facts import restored_facts
from icr_core import BudgetExhausted, ProtocolError

VERSION='v64_author_reflection_rebased_recovery_v3'


def recovery_input(episode,reflection):
    k=episode.recovery_prefix_length
    if type(k) is not int or not 0 <= k <= len(episode.history):
        raise ProtocolError('recovery context must reference an actual retained prefix')
    prompt=episode.base_prompt+'\n\nYour memory for the task below:\nTrial 0:\n'+reflection
    prompt+='\nOriginal task and initial room description (not current position):\n'
    prompt+=episode.initial['observation']+'\n'
    prompt+='\nRetained historical observations; later transfers may change object positions:\n'
    for i,row in enumerate(episode.history[:k]):
        prompt+=f'Receipt {i}: '+row['observation']+'\n'
    facts=restored_facts(episode.history,episode.snapshot())
    prompt+='\nHere is the task at the restored execution state. Plan from this state.\n'
    prompt+='Current observation: '+episode.snapshot()['observation']+'\n'
    prompt+='Objects currently established as held by accepted transfer receipts: '
    held=[row['object'] for row in facts['known_held_objects']]
    prompt+=(', '.join(held) if held else 'none established; this is not an inventory observation')+'\n'
    prompt+='Current available commands:\n'+'\n'.join(episode.snapshot()['available_actions'])+'\n'
    prompt+='\nActions taken in this recovery attempt:\n'
    prompt+=''.join('> '+row['action']+'\n'+row['observation']+'\n' for row in episode.history[k:])
    return prompt


class RebasedEpisode(SimpleEpisode):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.recovery_prefix_length=None

    def prompt(self,memory):
        if getattr(self, 'plain_history_prompt', None) is not None:
            return self.plain_history_prompt
        if self.recovery_prefix_length is None:
            return super().prompt(memory)
        if memory!=() or self.red_repair_handoff is None:
            raise ProtocolError('rebased recovery requires the bound Red memory')
        text=recovery_input(self,self.red_repair_handoff['reflection'])
        if self._scope_kind=='GREEN':
            if self._scope_view:
                text+='\n'+self._scope_view+'\nReturn one next action using the current task and actual receipts.\n'
        elif self._scope_kind=='YELLOW':
            text+='\n[YELLOW_EVENT_SUBTREE]\n'+self._scope_view+'\n'
            text+=('Correct the specific rejected operation or break the recorded local loop. '
                   'Check the original task object, actual possession, location and native menu. '
                   'Use one action to check a missing prerequisite or correct it; do not start '
                   'a separate whole-room search. Do not take a substitute object merely because '
                   'it is available. Historical Red advice is fallible; actual receipts take priority. '
                   'Return one native action.\n')
        else:
            raise ProtocolError('rebased Actor must be Green or Yellow')
        text+='>'
        if self.prompt_token_counter(text)>min(self.prompt_context_limit-100,self.prompt_service_limit):
            raise BudgetExhausted('rebased recovery input exceeds context; no truncation')
        self.last_history_window=dict(truncated=False,omitted_records=0,retained_records=len(self.history),
            policy='retained_observations_plus_complete_recovery_history',
            prefix_receipts=self.recovery_prefix_length,
            recovery_actions=len(self.history)-self.recovery_prefix_length,event_subtree_separate=True)
        return text


class RebasedV64Controller(SimpleV64Controller):
    def _reflect_and_restore(self):
        super()._reflect_and_restore()
        self.episode.recovery_prefix_length=self.selected_fault_index
        self.emit('actor_recovery_context_rebased',checkpoint_index=self.selected_fault_index,
                  actual_state=self.episode.snapshot(),ett_discarded=False,
                  current_branch_history_policy='retained_observations_plus_new_action_history')

    def to_dict(self):
        return dict(super().to_dict(),controller_version=VERSION,
                    recovery_prefix_length=self.episode.recovery_prefix_length)
