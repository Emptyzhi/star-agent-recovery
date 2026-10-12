"""WebShop public adapter for v16 native history and bounded local Yellow context."""
from copy import deepcopy
from pathlib import Path

from icr_core import ProtocolError
from icr_prompts import reflection_functions
from icr_webshop import WebShopEpisode
from tree_history import canonical
from webshop_semantics import semantic_snapshot

PACKAGE = Path(__file__).resolve().parents[1]


def instruction_from_reset(observation):
    if observation.count('Instruction:') != 1 or observation.count('[Search]') != 1:
        raise ProtocolError('reset must expose exactly one instruction and search interface')
    instruction = observation.split('Instruction:', 1)[1].split('[Search]', 1)[0].strip()
    if not instruction:
        raise ProtocolError('empty public shopping instruction')
    return instruction


def trace_text(initial, path, *, indexed=False):
    text = initial + '\n'
    for i, occurrence in enumerate(path):
        prefix = 'ACTION %d\n' % i if indexed else ''
        text += prefix + 'Action: ' + occurrence.action + '\nObservation: ' + occurrence.receipt['observation'] + '\n'
    return text


def author_reflection(transcript):
    query = reflection_functions(PACKAGE/'vendor/web_reflexion.py',
                                 PACKAGE/'vendor/webshop_reflexion_examples.txt')
    return query(transcript + '\n\nSTATUS: FAIL', [])


class WebV16Episode(WebShopEpisode):
    ordinary_command_constrained = False
    ordinary_stop_mode = 'first_line'
    native_actor_history = True
    unused_final_proposal = False

    def __init__(self, website, session_id, task_id, base_prompt, formatter):
        super().__init__(website, session_id, task_id, base_prompt)
        self.review_formatter = formatter
        self.action_limit = 15
        self.actor_output_tokens = 100
        self.yellow_output_tokens = 100
        self.recovery_prefix_length = None
        self.red_repair_handoff = None
        self._scope_view = ''
        self._scope_kind = None
        self._score = None
        self.last_history_window = {}
        self.replay_actions = 0
        self.replay_resets = 0

    @staticmethod
    def _decorate(state, score):
        done = state['page_type'] == 'end'
        if done != (score is not None):
            raise ProtocolError('terminal page and public score disagree')
        return semantic_snapshot(dict(deepcopy(state), benchmark='webshop', done=done,
                                      public_terminal_score=score))

    def _receipt(self, raw):
        row = super()._receipt(raw)
        row['before'] = None if raw.before is None else self._decorate(raw.before, self._score)
        row['after'] = self._decorate(raw.after, raw.public_terminal_score)
        self._score = raw.public_terminal_score
        return row

    def reset(self):
        receipt = super().reset()
        self.instruction = instruction_from_reset(receipt['observation'])
        return receipt

    def snapshot(self):
        return self._decorate(self.environment.snapshot(), self._score)

    def restore(self, checkpoint):
        super().restore(checkpoint)
        self._score = None
        if self.checkpoint() != checkpoint:
            raise ProtocolError('restored native session or history differs from checkpoint')
        return dict(backend_requests=0, restoration='EXACT_LOCAL_SESSION_AND_HISTORY')

    def parse(self, text):
        # Preserve the declared author adapter, including raw parser evidence.
        # No additional action repair or guessed missing bracket is introduced.
        return super().parse(text)

    def set_scope_prompt(self, view, version, *, kind, directive, incident_kind=None):
        if kind not in ('GREEN', 'YELLOW'):
            raise ProtocolError('ordinary WebShop prompt requires Green or Yellow')
        self._scope_view, self._scope_kind = view, kind

    def prompt(self, memory):
        if memory != () or self.initial is None or self._scope_kind is None:
            raise ProtocolError('WebShop v16 requires reset and bound scope, no external memory')
        text = self.base_prompt + '\n\n'
        k = self.recovery_prefix_length
        if k is None:
            text += self.initial['observation'] + '\n'
            rows = self.history
        else:
            if not 0 <= k <= len(self.history):
                raise ProtocolError('recovery prefix is outside retained WebShop history')
            if self._scope_kind == 'GREEN':
                text += ('Original task: ' + self.instruction +
                         '\nRetained historical observations:\n')
            else:
                if self.red_repair_handoff is not None and self.red_repair_handoff.get('schema') == 'webshop_compact_recovery_context_v1':
                    text += 'Recovery handoff for this trial:\n' + canonical(self.red_repair_handoff)
                    text += ('\nThis handoff is fallible history. The current page and native action '
                             'menu are authoritative.\nOriginal task: ' + self.instruction +
                             '\nRetained historical observations:\n')
                elif self.red_repair_handoff is not None:
                    text += 'Your memory for the task below:\nTrial 0:\n' + self.red_repair_handoff['reflection']
                    text += '\nStructured recovery memory (public controller facts):\n' + canonical(
                        self.red_repair_handoff.get('recovery_facts', {}))
                    text += ('\nThis is fallible advice about a failed branch, not current page state.\n'
                             'Original task: ' + self.instruction + '\nRetained historical observations:\n')
                else:
                    text += ('Original task: ' + self.instruction +
                             '\nRetained historical observations:\n')
            if self._scope_kind == 'YELLOW':
                # Yellow diagnoses one current public obstruction. Replaying
                # the entire recovery branch here duplicates the Green/Red
                # history and can exceed the model context on long pages.
                text += ('Yellow sees the current public page and local event '
                         'receipts below; earlier branch history is retained '
                         'by the controller, not repeated in this prompt.\n')
                rows = ()
            else:
                for i, row in enumerate(self.history[:k]):
                    text += 'Receipt %d: %s\n' % (i, row['observation'])
                rows = self.history[k:]
            text += '\nRestored page, selected options and visible controls:\n' + canonical(self.snapshot())
            text += '\nActions in this recovery attempt:\n'
        for row in rows:
            text += 'Action: ' + row['action'] + '\nObservation: ' + row['observation'] + '\n'
        if self._scope_view:
            text += '\n' + self._scope_view + '\n'
        text += '\nAction:'
        output_reserve = self.yellow_output_tokens if self._scope_kind == 'YELLOW' else self.actor_output_tokens
        if self.prompt_token_counter(text) > min(self.prompt_context_limit-output_reserve, self.prompt_service_limit):
            raise ProtocolError('complete WebShop Actor prompt exceeds context; no truncation')
        self.last_history_window = dict(truncated=False, retained_records=len(self.history),
            prefix_receipts=k, policy='native_history_or_rebased_retained_observations',
            event_subtree_separate=True)
        return text
