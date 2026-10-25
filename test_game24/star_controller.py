"""Game24 adaptation of bounded Green/Yellow/two-stage Red recovery."""
from copy import deepcopy
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
import re
import time

from source.experiment_settings import Experiment, EXECUTION_PROTOCOL
from .game24_env import Game24Environment, OPERATION_RE, ANSWER_RE, parse_number, format_number
from .model_api import Generation
from .action_language import language_for_state
from .public_evidence import executed_operand_lineage
from .prompts import (GREEN_PROMPT, RED_PROMPT, RED_CHECKPOINT_PROMPT,
                      YELLOW_ACTION_PROMPT, YELLOW_DIAGNOSIS_PROMPT, render, author_reflection)

GAME24_ADAPTATION_VERSION = 'game24_receipt_finalization_v12'


class ProtocolError(ValueError):
    pass


class ResourceLimit(RuntimeError):
    pass


@dataclass
class TaskResult:
    task_id: int | str
    puzzle: str
    status: str
    reward: float | None
    success: bool
    forward_actions: int
    recovery_actions: int
    model_requests: int
    red_attempts: int
    yellow_entries: int
    history: list
    events: list
    ett: dict
    cost: dict
    settings: dict
    error: str | None = None
    execution_protocol: str = EXECUTION_PROTOCOL

    def to_dict(self):
        return asdict(self)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class Game24STARController:
    def __init__(self, model, *, settings=None, emit=None):
        self.model = model
        self.settings = Experiment.for_dataset('game24') if settings is None else settings
        self.emit_sink = emit
        self.env = None

    def _emit(self, event, **data):
        self.events.append(dict(event=event, time=time.time(), **deepcopy(data)))
        if self.emit_sink is not None:
            self.emit_sink(event, **data)

    @staticmethod
    def _action(text):
        if not isinstance(text, str):
            raise ProtocolError('model output must be text')
        lines = text.strip().splitlines()
        if len(lines) > 1:
            raise ProtocolError('action response exceeded the first-line boundary')
        line = lines[0].strip() if lines else ''
        # The documented Act/Action label is framing, not an action rewrite.
        return re.sub(r'^(?:Act|Action)\s*\d*\s*[>:]\s*', '', line, count=1, flags=re.I)

    @staticmethod
    def _grammar(action):
        return OPERATION_RE.fullmatch(action) is not None or ANSWER_RE.fullmatch(action) is not None

    def _key(self):
        state = self.env.public_state()
        return encoded({k: state[k] for k in ('numbers', 'sources', 'done')})

    def _request(self, prompt, role, limit, *, commands=()):
        cfg = self.settings
        if cfg.model_request_limit is not None and self.model_requests >= cfg.model_request_limit:
            raise ResourceLimit('MODEL_RESOURCE_LIMIT')
        if time.monotonic()-self.started >= cfg.task_timeout:
            raise ResourceLimit('TASK_TIMEOUT')
        count = self.model.count_tokens(prompt) if hasattr(self.model, 'count_tokens') else None
        if cfg.total_input_tokens is not None and (count is None or self.input_tokens+count > cfg.total_input_tokens):
            raise ResourceLimit('INPUT_TOKEN_LIMIT_OR_UNAVAILABLE_ACCOUNTING')
        if cfg.total_output_tokens is not None and self.output_tokens+limit > cfg.total_output_tokens:
            raise ResourceLimit('OUTPUT_TOKEN_LIMIT')
        if not self.token_accounting_complete and (cfg.total_input_tokens is not None or cfg.total_output_tokens is not None):
            raise ResourceLimit('TOKEN_ACCOUNTING_UNAVAILABLE')
        self.model_requests += 1
        stop = ('\n',) if role in ('green', 'yellow_action', 'yellow_diagnosis') else ()
        language = ({'parameter_language': language_for_state(self.env.public_state())}
                    if role == 'yellow_action' else {})
        generation = self.model.generate(prompt, role=role, max_new_tokens=limit,
                                         stop=stop, commands=commands, **language)
        if not isinstance(generation, Generation) or not isinstance(generation.text, str):
            raise ProtocolError('ModelClient must return Generation with raw text')
        self.generations.append(generation)
        for name in ('input_tokens', 'output_tokens'):
            value = getattr(generation, name)
            if value is None:
                self.token_accounting_complete = False
            elif type(value) is not int or value < 0:
                raise ProtocolError('invalid token count')
            else:
                setattr(self, name, getattr(self, name)+value)
        if generation.output_tokens is not None and generation.output_tokens > limit:
            raise ProtocolError('model exceeded output cap')
        if generation.queue_seconds is None:
            self.timing_accounting_complete = False
        else:
            if not math.isfinite(generation.queue_seconds) or generation.queue_seconds < 0:
                raise ProtocolError('invalid measured queue time')
            self.queue_seconds += generation.queue_seconds
        self._emit('model_request', role=role, request_id=generation.request_id,
                   prompt=prompt, text=generation.text, input_tokens=generation.input_tokens,
                   output_tokens=generation.output_tokens, output_limit=limit,
                   queue_seconds=generation.queue_seconds, raw=generation.raw)
        if role == 'green' and self.current_event_record is not None:
            self.event_memory.append(deepcopy(self.current_event_record))
            self.delivered_facts.update(self.current_event_record['fact_keys'])
            self._emit('event_subtree_recorded', **self.current_event_record)
            self.current_event_record = None
        if role == 'green' and self.pending_events:
            if cfg.ablation != 'without_event_subtree':
                self._emit('event_subtree_delivered', event_payload=self.pending_events)
            else:
                self._emit('event_subtree_suppressed', event_payload=self.pending_events)
            self.pending_events = []
        return generation

    def _record_protocol_feedback(self, raw, role):
        self.protocol_feedback = ('Not executed ('+role+'): '+repr(raw)+
            '. Expected one binary operation a op b = c, or answer: <complete expression> = 24. '
            'The intermediate grammar accepts two numeric literals and one operator; '
            'a fraction literal is written a/b without parentheses or internal spaces. '
            'Parenthesized or compound intermediates do not match this grammar. Current numbers: '+
            ' '.join(self.env.public_state()['numbers'])+'. Use exact fractions, not rounded decimals.')
        self.protocol_incidents.append(dict(role=role, raw=raw, current_state=self.env.public_state()))

    @staticmethod
    def _observation(receipt):
        if receipt.accepted and not receipt.done:
            return 'numbers left: '+' '.join(receipt.after['numbers'])
        return receipt.detail+'; numbers left: '+' '.join(receipt.after['numbers'])

    def _history(self):
        lines = []
        for i, receipt in enumerate(self.env.history, 1):
            lines.append('Act %d> %s\nObs %d> %s' %
                         (i, receipt.action, i, self._observation(receipt)))
        return '\n'.join(lines) + ('\n' if lines else '')

    def _operand_record(self, *, checkpoint=None, lineage=True):
        """Render executed provenance only; never expand a final expression."""
        state = self.env.public_state() if checkpoint is None else self.env.public_checkpoint(checkpoint)
        history = self.env.history if checkpoint is None else checkpoint['history']
        lines = ['Original input occurrences: '+', '.join(
            value+' [constant.'+str(i)+']' for i, value in enumerate(self.env.puzzle.split(), 1))]
        repetitions = {1: 'once', 2: 'twice', 3: 'three times', 4: 'four times'}
        lines.append('Required ORIGINAL occurrences in a final expression: '+', '.join(
            format_number(value)+' '+repetitions[count]
            for value, count in Counter(self.env.original).items())+'.')
        label = 'Current operand occurrences: ' if checkpoint is None else 'Saved checkpoint operand occurrences: '
        lines.append(label+', '.join(
            value+' ['+source+']' for value, source in zip(state['numbers'], state['sources'])))
        for step, receipt in enumerate(history, 1):
            if receipt.accepted and receipt.produced is not None:
                lines.append('Recorded operation %d: %s; %s -> %s' %
                             (step, receipt.action, ', '.join(receipt.consumed), receipt.produced))
        if lineage:
            lines.append(executed_operand_lineage(self.env.puzzle, history, state))
        return '\n'.join(lines)

    @staticmethod
    def _dependency_text(receipt):
        if receipt is None:
            return 'no next recorded action'
        return ('consumed=%s; produced=%s; parents=%s' %
                (','.join(receipt['consumed']) or 'none',
                 receipt['produced'] or 'none',
                 ','.join(str(parent) for parent in receipt['parents']) or 'none'))

    def _branch_state_key(self):
        # Reconstruct only executed operand provenance. Operation IDs can be
        # reused after rollback, so IDs alone cannot identify an arithmetic state.
        origins = {'constant.'+str(i): ('input', i, value)
                   for i, value in enumerate(self.env.puzzle.split(), 1)}
        for receipt in self.env.history:
            if receipt.accepted and receipt.produced is not None:
                operation = OPERATION_RE.fullmatch(receipt.action)
                if operation is None:
                    raise ProtocolError('accepted operation lacks the environment operation grammar')
                result = receipt.after['numbers'][receipt.after['sources'].index(receipt.produced)]
                origins[receipt.produced] = ('executed', operation.group(2), result,
                    tuple(origins[source] for source in receipt.consumed))
        state = self.env.public_state()
        return encoded(dict(puzzle=self.env.puzzle, done=state['done'],
            operands=[(value, origins[source]) for value, source in zip(state['numbers'], state['sources'])]))

    @staticmethod
    def _operands_present(receipt, state):
        # Only inspect occurrences from an already rejected operation. Do not
        # evaluate its operator, declared result, or any candidate continuation.
        match = OPERATION_RE.fullmatch(receipt['action'])
        if match is None:
            raise ProtocolError('operand rejection lacks operation grammar')
        required = Counter(parse_number(match.group(i)) for i in (1, 3))
        present = Counter(parse_number(value) for value in state['numbers'])
        return not (required-present)

    def _remember_fact(self, kind, node_ids, anchor, trigger):
        key = encoded((kind, node_ids, trigger))
        self.event_facts[key] = dict(kind=kind, node_ids=node_ids,
                                    anchor=anchor, trigger=trigger)

    def _event_receipt(self, node_id):
        # Active receipts already have one complete source in Act/Obs and the
        # operand record. Resolve references on every prompt, never cache labels.
        if node_id in self.active_path:
            return 'Receipt node %d: see Act/Obs %d and its operand record.' % (
                node_id, self.active_path.index(node_id)+1)
        receipt = self.nodes[node_id]['receipt']
        return ('Historical receipt node %d: %s -> %s; accepted=%s\n'
                'Recorded pool before: %s; after: %s; %s' %
                (node_id, receipt['action'], receipt['detail'], receipt['accepted'],
                 ' '.join(receipt['before']['numbers']),
                 ' '.join(receipt['after']['numbers']), self._dependency_text(receipt)))

    def _compact_receipts(self, node_ids):
        """Group identical rejections without changing stored execution nodes."""
        groups = {}
        for node_id in node_ids:
            node = self.nodes[node_id]
            receipt = node['receipt']
            key = ('accepted', node_id) if receipt['accepted'] else (
                'rejected', node['before_state_key'], receipt['action'], receipt['detail'])
            groups.setdefault(key, []).append(node_id)
        lines = []
        for ids in groups.values():
            receipt = self.nodes[ids[0]]['receipt']
            lines.append('Receipt nodes %s; count=%d; scope=%s; action=%s; accepted=%s; feedback=%s; '
                         'pool before=%s; after=%s; %s' % (
                ','.join(map(str, ids)), len(ids),
                ','.join(dict.fromkeys(self.nodes[i]['scope'] for i in ids)),
                receipt['action'], receipt['accepted'], receipt['detail'],
                ' '.join(receipt['before']['numbers']), ' '.join(receipt['after']['numbers']),
                self._dependency_text(receipt)))
        return '\n'.join(lines)

    def _local_event_projection(self, event):
        incident = event['incident']
        lines = [
            'LOCAL_REPAIR_HANDOFF: '+event['event_key'],
            'Trigger: '+incident['kind']+'; proposal='+encoded(incident.get('action', incident.get('raw', ''))),
            'Observed issue: '+(incident['receipt']['detail'] if 'receipt' in incident else incident['kind']),
            'Public local repair outcome: '+event['outcome'],
            'Local repair evidence status: '+event['progress_evidence'],
            'Recorded pool after local repair: '+' '.join(event['current_public_state']['numbers']),
            'Relevant executed dependency nodes: '+','.join(map(str, event['dependency_node_ids']))]
        if event['node_ids']:
            lines.append(self._compact_receipts(event['node_ids']))
        else:
            lines.append('No environment action was executed by this Yellow incident.')
        for node_id in event['node_ids']:
            node = self.nodes[node_id]
            receipt = node['receipt']
            later = [i for i, candidate in self.nodes.items()
                     if i > node_id and candidate['before_state_key'] == node['before_state_key']
                     and candidate['receipt']['action'] == receipt['action']]
            if later:
                lines.append('Later executions of the same repair in the same provenance state: '
                             + ','.join(map(str, later)) + '.')
                lines.append(self._compact_receipts(later))
        lines.append('Yellow model advice (unverified): '+event['advice'])
        lines.append('Interpret the advice with its executed repair receipts above. '
                     'A rejected repair is not evidence of successful recovery; an accepted intermediate is not final success.')
        lines.append('Yellow advice cannot trigger Red. Only the ordinary failed-attempt boundary can enter Red.')
        return '\n'.join(lines)

    def _receipt_event_view(self):
        if self.settings.ablation == 'without_event_subtree':
            return ''
        self.current_event_record = None
        anchor = self._branch_state_key()
        state = self.env.public_state()
        last_id = self.active_path[-1] if self.active_path else None
        last = self.nodes[last_id]['receipt'] if last_id is not None else None
        red = next((e for e in self.pending_events if e['kind']=='RED_HANDOFF'), None)

        # Transition notices are delivered once. Lost executed chains remain
        # contextual memory only along their actual provenance-compatible path.
        self.event_facts = {key: fact for key, fact in self.event_facts.items()
            if (anchor in fact['anchor'] if fact['kind']=='HISTORICAL_POOL_24'
                else anchor == fact['anchor'])}

        # Track the latest actual missing-operand receipt per action. A new
        # accepted receipt or restore is the only source of a changed obstacle.
        unavailable = {}
        for i, node in self.nodes.items():
            receipt = node['receipt']
            if receipt['detail'].startswith('operand occurrence unavailable'):
                unavailable[receipt['action']] = i
            else:
                # Any newer executed outcome supersedes the old obstacle,
                # including an arithmetic rejection after operands appeared.
                unavailable.pop(receipt['action'], None)
        for i in unavailable.values():
            receipt = self.nodes[i]['receipt']
            if not self._operands_present(receipt, state):
                continue
            if red is not None:
                if not self._operands_present(receipt, red['failed_public_state']):
                    self._remember_fact('OPERAND_OBSTACLE_CHANGED', [i], anchor,
                                        ('restore', red['event_key'], red['checkpoint_index']))
            elif (last is not None and last['accepted'] and i < last_id
                  and not self._operands_present(receipt, last['before'])):
                self._remember_fact('OPERAND_OBSTACLE_CHANGED', [i, last_id], anchor, last_id)

        if red is not None:
            path = self.archived_paths[-1]
            reached = [i for i in path if self.nodes[i]['receipt']['accepted']
                       and self.nodes[i]['receipt']['after']['numbers']==['24']
                       and not self.nodes[i]['receipt']['done']]
            if reached:
                prefix = path[:path.index(reached[-1])+1]
                chain = [i for i in prefix if self.nodes[i]['receipt']['accepted']
                         and self.nodes[i]['receipt']['produced'] is not None]
                scopes = [self.nodes[i]['before_state_key'] for i in chain]
                if anchor in scopes:
                    self._remember_fact('HISTORICAL_POOL_24', chain, scopes, red['event_key'])

        visible = {key: fact for key, fact in self.event_facts.items()
            if fact['kind']=='HISTORICAL_POOL_24'
            or (key not in self.delivered_facts and fact['node_ids'][0] in unavailable.values())}
        if not visible:
            return ''
        lines = ['[EVENT_SUBTREE]', 'Relevant public evidence for the current decision. '
                 'These receipts do not prescribe or forbid an action.']
        shown = set()
        for fact in visible.values():
            kind = fact['kind']
            lines.append(kind+': '+{
                'OPERAND_OBSTACLE_CHANGED': 'Previously missing operand occurrences are now present. This does not certify the old operation or its result.',
                'HISTORICAL_POOL_24': 'This abandoned chain actually reached a single 24 without a successful final submission. Only the unreplayed suffix is shown; historical IDs belong to that chain.'
            }[kind])
            node_ids = fact['node_ids']
            if kind == 'HISTORICAL_POOL_24':
                # Earlier operations already live in the ordinary operand
                # record, including provenance-equivalent reexecuted nodes.
                offset = fact['anchor'].index(anchor)
                node_ids = node_ids[offset:]
                # The matched branch key includes ordered executed provenance,
                # not just values. Rejections can shift operation IDs, so bridge
                # the two namespaces without rewriting historical receipts.
                boundary = self.nodes[node_ids[0]]['receipt']['before']
                correspondence = [
                    'historical '+old+' -> current '+new+' (value '+value+')'
                    for old, new, value in zip(boundary['sources'], state['sources'], state['numbers'])
                    if old != new]
                if correspondence:
                    lines.append('Matched executed-provenance boundary: '+'; '.join(correspondence)+'.')
            if kind == 'OPERAND_OBSTACLE_CHANGED':
                if isinstance(fact['trigger'], tuple) and fact['trigger'][0]=='restore':
                    lines.append('Observed transition: restored CHECKPOINT '+str(fact['trigger'][2])+'.')
            for i in node_ids:
                if i not in shown:
                    lines.append(self._event_receipt(i))
                    shown.add(i)
        text = '\n'.join(lines)+'\n'
        fresh = [key for key in visible if key not in self.delivered_facts]
        if fresh:
            self.current_event_record = dict(text=text, fact_keys=fresh,
                facts=[deepcopy(self.event_facts[key]) for key in fresh])
        return text

    def _event_view(self):
        if self.settings.ablation == 'without_event_subtree':
            return ''
        text = self._receipt_event_view()
        record = self.current_event_record
        facts = []
        lines = []
        shown_local_events = set()
        for event in self.pending_events:
            if event['kind'] != 'YELLOW_HANDOFF':
                continue
            lines.append(self._local_event_projection(event))
            shown_local_events.add(event['event_key'])
            lines.append('Green must decide the next action from the actual current pool. '
                         'Local acceptance alone does not establish a solution; advice may be wrong.')
            facts.append(dict(kind='LOCAL_REPAIR_HANDOFF', event_key=event['event_key'],
                              event=deepcopy(event)))

        anchor = self._branch_state_key()
        rejections = {}
        active = set(self.active_path)
        for node_id, node in self.nodes.items():
            receipt = node['receipt']
            final = ANSWER_RE.fullmatch(receipt['action']) is not None
            if receipt['accepted'] or (not final and node['before_state_key'] != anchor):
                continue
            key = (receipt['action'], receipt['detail'])
            rejections.setdefault(key, []).append(node_id)
        # Present the most recently observed outcome first, keeping every group.
        for (action, detail), node_ids in sorted(rejections.items(), key=lambda row: row[1][-1], reverse=True):
            historical = [i for i in node_ids if i not in active]
            if not historical and len(node_ids) < 2:
                continue
            kind = 'HISTORICAL_REJECTION' if historical else 'REPEATED_REJECTION'
            lines.append(kind+': action='+encoded(action)+'; observed feedback='+detail+
                         '; recorded occurrences='+str(len(node_ids)))
            lines.append('Receipt node references: '+','.join(map(str, node_ids)))
            linked = [event for event in self.yellow_incidents
                      if set(node_ids).intersection(event['node_ids'])]
            for event in linked:
                lines.append('Linked local repair '+event['event_key']+': '+event['outcome']+
                             '; evidence='+event['progress_evidence']+
                             '. The following receipt records the attempted repair, not a successful recommendation.')
                if event['event_key'] not in shown_local_events:
                    lines.append(self._local_event_projection(event))
                    shown_local_events.add(event['event_key'])
            lines.append(self._event_receipt(node_ids[0]))
            final = ANSWER_RE.fullmatch(action) is not None
            scope = 'this task and final expression, independently of the intermediate pool' if final else 'the same executed-provenance state'
            lines.append('These are past outcomes for '+scope+', not a ban or an evaluated alternative action.')
            facts.append(dict(kind=kind, node_ids=node_ids,
                              anchor=self.env.puzzle if final else anchor,
                              evidence_scope='task_expression' if final else 'operand_state'))
        if not lines:
            return text
        if not text:
            text = ('[EVENT_SUBTREE]\nRelevant public evidence for the current decision. '
                    'These receipts do not prescribe or forbid an action.\n')
        text += '\n'.join(lines)+'\n'
        keys = [encoded(fact) for fact in facts]
        fresh = [(key, fact) for key, fact in zip(keys, facts) if key not in self.delivered_facts]
        if record is not None or fresh:
            self.current_event_record = dict(text=text,
                fact_keys=(record['fact_keys'] if record is not None else [])+[key for key, _ in fresh],
                facts=(record['facts'] if record is not None else [])+[fact for _, fact in fresh])
        return text

    def _recovery_context(self):
        if not self.recovery_memory:
            return ''
        continuation = self.active_path[self.branch_start_length:]
        if not continuation:
            return self.recovery_memory
        return (self.recovery_memory+'\n[EXECUTED_CONTINUATION_AFTER_RESTORE]\n'+
                self._compact_receipts(continuation)+
                '\nThese are actual continuation outcomes after the note, not proof that its suggested plan was followed. '
                'A rejected continuation is not a verified repair. Use current receipts when reconsidering this fallible note.')

    def _green_prompt(self):
        memory = ('Your memory for this attempt (fallible advice):\n'+self._recovery_context()+'\n'
                  if self.recovery_memory else '')
        numbers = self.env.public_state()['numbers']
        if len(numbers) == 1:
            state_instruction = (
                'FINAL-ACTION STATE: only one value remains. Return a complete '
                'model-written action in the form answer: <expression using all '
                'four ORIGINAL input occurrences> = 24. Do not return Failure, '
                'prose, or another binary operation. The current prefix may be '
                'wrong, so construct a fresh expression from the original inputs.'
                if numbers != ['24'] else
                'FINAL-ACTION STATE: the current pool is 24. Its executed expression is '
                'shown in EXECUTED_OPERAND_LINEAGE. Write the final answer from '
                'that recorded expression, preserving its operands, operators and '
                'parentheses. Earlier unverified repair plans must not replace '
                'these accepted operations. The pending step is a model-written '
                'answer: <the recorded expression> = 24 action. It still uses one '
                'action and needs environment acceptance. Return no prose or '
                'another binary operation.')
        else:
            state_instruction = (
                'OPERATION STATE: return one binary operation over two separate '
                'occurrences in the CURRENT pool (their values may be equal), or submit a complete answer only '
                'when you intentionally revise the prefix.')
        action_format = (
            'ACTION FORMAT: a binary step is exactly <number> <+|-|*|/> <number> = <number>. '
            'A complete answer is exactly answer: <expression using each ORIGINAL input '
            'occurrence once> = 24. The answer prefix is mandatory; never emit a bare '
            'expression. Check the arithmetic yourself before emitting it and do not '
            'write = 24 when the left side is not 24. Recovery notes are unverified: '
            'never copy a compound or parenthesized intermediate as one action; '
            'emit one atomic binary step or a complete answer.')
        action_format += ('\nActions remaining in this window: '+str(self.window_budget-self.used)+
                          '. Intermediate operations and final answer submissions each cost one action.')
        action_format += '\n'+self._operand_record(lineage=len(numbers)==1)
        feedback = ('Controller feedback on an unexecuted proposal:\n'+self.protocol_feedback+'\n'
                    if self.protocol_feedback else '')
        return render(GREEN_PROMPT, task=self.env.puzzle, memory=memory,
                      history=self._history(), events=self._event_view()+feedback,
                      state_instruction=state_instruction, action_format=action_format,
                      step=len(self.env.history)+1)

    def _execute(self, action, scope):
        if self.used >= self.window_budget:
            raise ProtocolError('local action exceeded shared execution window')
        if self.settings.environment_call_limit is not None and self.environment_calls >= self.settings.environment_call_limit:
            raise ResourceLimit('ENVIRONMENT_CALL_LIMIT')
        parent = self.active_path[-1] if self.active_path else None
        self.seen_edges.add((self._key(), action))
        before_checkpoint = self.env.checkpoint()
        before_state_key = self._branch_state_key()
        self.environment_calls += 1
        receipt = self.env.execute(action)
        self.protocol_feedback = ''
        self.used += 1
        if self.recovering:
            self.recovery_actions += 1
        else:
            self.forward_actions += 1
        node_id = len(self.nodes)
        self.nodes[node_id] = dict(node_id=node_id, parent_id=parent, scope=scope,
                                   receipt=receipt.to_dict(), window=self.red_attempts,
                                   before_state_key=before_state_key)
        self.node_checkpoints[node_id] = before_checkpoint
        self.active_path.append(node_id)
        self.checkpoints.append(self.env.checkpoint())
        self._emit('action_executed', node=self.nodes[node_id], actions_remaining=self.window_budget-self.used)
        return receipt, node_id

    def _yellow_state_instruction(self, rollback_available):
        instruction = (
            'The current pool has one value. Return a complete final action in '
            'the exact form answer: <expression using all four ORIGINAL inputs> = 24.'
            if len(self.env.public_state()['numbers']) == 1 else
            'The current pool has multiple values. Return one exact binary operation '
            'using two current occurrences (equal values are allowed), or a complete final answer.')
        if self.env.public_state()['numbers'] == ['24']:
            instruction += (
                ' The current executed pool is 24. Use the recorded executed '
                'expression in EXECUTED_OPERAND_LINEAGE for the final action, '
                'preserving its operands, operators and parentheses. Repair the '
                'final transcription rather than replacing accepted operations '
                'with an earlier unverified plan. Environment acceptance of the '
                'model-written final action is still required.')
        return instruction+'\nActions remaining in this window: '+str(self.window_budget-self.used)+(
            '. A final answer also uses one action. Yellow only proposes local environment actions. '
            'Any judgment about revising earlier operations is recorded in ETT for Red at the '
            'ordinary failed-attempt boundary; do not output BACKTRACK.')

    def _yellow(self, incident):
        self.window_yellow += 1
        self.yellow_entries += 1
        nodes = []
        self._emit('yellow_entered', incident=incident, entry=self.yellow_entries)
        attempt = incident.get('action', incident.get('raw', ''))
        issue = incident['receipt']['detail'] if 'receipt' in incident else incident['kind']
        repair_focus = 'Use the exact observed issue; distinguish local validity from overall success.'
        if issue == 'INVALID_ACTION_FORMAT':
            repair_focus = 'Repair the action grammar; no environment execution or operand consumption occurred.'
        elif issue.startswith('operand occurrence unavailable'):
            repair_focus = 'Use the actual current operand occurrences; an already consumed occurrence is unavailable.'
        elif issue == 'final expression must use every original number exactly once':
            repair_focus = 'Repair the final expression using each ORIGINAL input occurrence exactly once; inspect the recorded operation sources.'
        elif issue in ('declared result does not match the expression', 'declared result does not match the operation'):
            repair_focus = 'Repair the incorrect arithmetic in the submitted action; appending = 24 is not a correction.'
        elif issue == 'REPEATED_STATE_ACTION':
            repair_focus = 'Inspect the previous outcome of this exact state-action pair before proposing a correction.'
        context = dict(task=self.env.puzzle, history=self._history(),
                       numbers=' '.join(self.env.public_state()['numbers']),
                       attempt=attempt, issue=issue, step=len(self.env.history)+1,
                       rollback_available=self.red_attempts==0 and self.settings.ablation!='without_red',
                       protocol_feedback=self.protocol_feedback or 'No unexecuted protocol proposal.',
                       recovery_memory=self._recovery_context() or 'No Red recovery note in this window.',
                       remaining_actions=self.window_budget-self.used, repair_focus=repair_focus)
        context['state_instruction'] = self._yellow_state_instruction(context['rollback_available'])
        context['history'] += self._operand_record()+'\n'
        diagnosis = self._request(render(YELLOW_DIAGNOSIS_PROMPT, **context),
            'yellow_diagnosis', self.settings.yellow_diagnosis_tokens)
        if not diagnosis.text.strip():
            self._emit('yellow_protocol_failure', phase='diagnosis', raw=diagnosis.text)
        outcome = 'UNRESOLVED'
        progress_evidence = 'NONE'
        for _ in range(min(self.settings.yellow_actions, self.window_budget-self.used)):
            context.update(history=self._history()+self._operand_record()+'\n', numbers=' '.join(self.env.public_state()['numbers']),
                           step=len(self.env.history)+1,
                           state_instruction=self._yellow_state_instruction(context['rollback_available']),
                           remaining_actions=self.window_budget-self.used,
                           protocol_feedback=self.protocol_feedback or 'No unexecuted protocol proposal.')
            if nodes:
                context.update(attempt=self.nodes[nodes[-1]]['receipt']['action'],
                               issue=self.nodes[nodes[-1]]['receipt']['detail'])
            action = self._request(render(YELLOW_ACTION_PROMPT, diagnosis=diagnosis.text, **context),
                'yellow_action', self.settings.yellow_output_tokens)
            try:
                text = self._action(action.text)
            except ProtocolError as error:
                self._record_protocol_feedback(action.text, 'yellow_action')
                self._emit('yellow_protocol_failure', phase='action', raw=action.text, error=str(error))
                break
            if text == 'BACKTRACK':
                outcome = 'PREFIX_REVISION_ADVICE_RECORDED'
                self._record_protocol_feedback(action.text, 'yellow_action')
                self._emit('yellow_prefix_revision_advice_recorded', raw=action.text,
                           incident=incident, diagnosis=diagnosis.text,
                           disposition='RECORD_ETT_RETURN_GREEN_NO_RED_TRANSITION')
                break
            if not self._grammar(text):
                self._record_protocol_feedback(action.text, 'yellow_action')
                self._emit('yellow_protocol_failure', phase='action', raw=action.text)
                break
            receipt, node_id = self._execute(text, 'YELLOW')
            nodes.append(node_id)
            if receipt.done or receipt.accepted:
                outcome = 'TERMINAL' if receipt.done else 'ACCEPTED_PUBLIC_STATE_CHANGE'
                progress_evidence = ('FINAL_SUCCESS' if receipt.reward == 1 else
                    'CURRENT_POOL_24_WITHOUT_FINAL_SUBMISSION' if receipt.after['numbers'] == ['24'] else
                    'LOCAL_ACCEPTANCE_ONLY_NOT_VERIFIED_SOLUTION_PROGRESS')
                break
        event = dict(kind='YELLOW_HANDOFF', event_key='yellow:'+str(self.yellow_entries),
            node_ids=nodes, outcome=outcome, advice=diagnosis.text,
            incident=incident, current_public_state=self.env.public_state(),
            dependency_node_ids=[i for i in self.active_path if self.nodes[i]['receipt']['produced'] is not None],
            red_available=self.red_attempts==0 and self.settings.ablation!='without_red',
            progress_evidence=progress_evidence)
        self.pending_events.append(event)
        self.yellow_incidents.append(event)
        self._emit('yellow_returned_green', **event)
        return outcome

    def _window(self, recovery):
        self.recovering = recovery
        self.used = 0
        self.window_budget = self.settings.recovery_actions if recovery else self.settings.initial_actions
        self.window_yellow = 0
        seen_incidents = set()
        noted_prefixes = set()
        self._emit('window_started', recovery=recovery, actions=self.window_budget)
        while not self.env.done and self.used < self.window_budget:
            numbers = self.env.public_state()['numbers']
            if len(numbers) == 1 and numbers != ['24']:
                # Native final answers use original inputs, so an unsuccessful
                # intermediate value does not end either execution window.
                if self._key() not in noted_prefixes:
                    noted_prefixes.add(self._key())
                    self._emit('non_target_prefix_final_answer_available', remaining_numbers=numbers)
            last = self.env.history[-1] if self.env.history else None
            if (last is not None and not last.accepted
                    and len(self.active_path) > self.branch_start_length
                    and self.nodes[self.active_path[-1]]['scope'] == 'GREEN'):
                incident = dict(kind='ENVIRONMENT_REJECTION', action=last.action, receipt=last.to_dict())
                key = ('receipt', self.active_path[-1])
                if self._can_yellow() and key not in seen_incidents:
                    seen_incidents.add(key)
                    self._yellow(incident)
                    continue
            generation = self._request(self._green_prompt(), 'green', self.settings.actor_output_tokens)
            if self.current_branch_key is not None:
                self.shown_branches.add(self.current_branch_key)
                self.current_branch_key = None
            if not generation.text.strip() and self._can_yellow():
                self._emit('actor_empty_output_failure', raw=generation.text, action_executed=False)
                return 'EMPTY_ACTOR_OUTPUT'
            try:
                action = self._action(generation.text)
            except ProtocolError:
                action = generation.text
                grammar = False
            else:
                grammar = self._grammar(action)
            edge = (self._key(), action)
            if not grammar:
                protocol_key = self._key()
                incident = dict(kind='INVALID_ACTION_FORMAT', raw=generation.text,
                                current_state=self.env.public_state())
                key = ('protocol', protocol_key)
                if self._can_yellow() and key not in seen_incidents:
                    self._record_protocol_feedback(generation.text, 'green')
                    self._emit('green_protocol_failure', raw=generation.text, destination='YELLOW')
                    seen_incidents.add(key)
                    self._yellow(incident)
                    continue
                # Unrepaired proposals are real native attempts, not free retries.
                self._emit('actor_protocol_failure_submitted', raw=generation.text,
                           action=action, yellow_available=self._can_yellow())
                self._execute(action, 'GREEN')
                continue
            if edge in self.seen_edges:
                incident = dict(kind='REPEATED_STATE_ACTION', action=action, current_state=self.env.public_state())
                key = ('repeat', edge)
            else:
                incident = None
            if incident is not None and self._can_yellow() and key not in seen_incidents:
                seen_incidents.add(key)
                self._yellow(incident)
                continue
            self._execute(action, 'GREEN')
        return 'SUCCESS' if self.env.reward == 1 else 'TERMINAL_FAILURE' if self.env.done else 'WINDOW_EXHAUSTED'

    def _can_yellow(self):
        return self.settings.ablation != 'without_yellow' and self.window_yellow < self.settings.yellow_entries

    def _red(self, failure):
        if self.red_attempts:
            raise ProtocolError('only one Red recovery is allowed')
        self._emit('red_preparation_started')
        self.red_attempts += 1
        failed_path = self.active_path[:]
        heading = ('[FAILED_BRANCH_PUBLIC_HISTORY]' if self.settings.ablation == 'without_event_subtree'
                   else '[FAILED_BRANCH_EVENT_SUBTREE]')
        transcript = ('Here is the task:\nInput: '+self.env.puzzle+
                      '\n'+heading+'\n'+self._compact_receipts(failed_path))
        transcript += '\n'+self._operand_record()+'\n'
        for event in self.yellow_incidents:
            transcript += '\n'+self._local_event_projection(event)
        for event in self.protocol_incidents:
            transcript += '\nUnexecuted '+event['role']+' proposal: '+repr(event['raw'])
        transcript += '\nController-observed failure: '+failure+'\nSTATUS: FAIL'
        if self.protocol_feedback:
            transcript += '\n'+self.protocol_feedback
        reflection = self._request(author_reflection(transcript, task=self.env.puzzle),
            'red_reflection', self.settings.red_reflection_tokens)
        if not reflection.text.strip():
            raise ProtocolError('Red reflection is empty')
        # ACTION n restores the state before that actual executed action.
        indices = range(len(failed_path))
        candidates = [dict(checkpoint=k, public_state=self.env.public_checkpoint(self.checkpoints[k]),
            next_action=self.nodes[failed_path[k]]['receipt']['action'] if k < len(failed_path) else None,
            next_receipt=self.nodes[failed_path[k]]['receipt'] if k < len(failed_path) else None)
            for k in indices if not self.checkpoints[k]['done']]
        if not candidates and failed_path:
            raise ProtocolError('no actual restorable checkpoint')
        candidate_text = '\n'.join(('ACTION %d: after %d recorded actions; numbers left: %s; next recorded action: %s; %s' %
                                  (c['checkpoint'], c['checkpoint'], ' '.join(c['public_state']['numbers']),
                                   c['next_action'] if c['next_action'] is not None else '(none)',
                                   self._dependency_text(c['next_receipt'])))+'\n'+
                                  self._operand_record(checkpoint=self.checkpoints[c['checkpoint']])
                                  for c in candidates)
        choices = tuple('ACTION '+str(c['checkpoint']) for c in candidates)
        if choices:
            selection = self._request(render(RED_CHECKPOINT_PROMPT, task=self.env.puzzle,
                failure=failure, reflection=reflection.text, candidates=candidate_text),
                'red_checkpoint', self.settings.red_selection_tokens, commands=choices)
            choice = selection.text
            if choice not in choices:
                self._emit('red_checkpoint_selection_failed', raw=choice, choices=choices)
                raise ProtocolError('Red selection violated the declared ACTION trie')
            k = int(choice.split()[1])
            selection_reason = 'RED_ACTION_TRIE_SELECTION'
        else:
            # No environment action exists after an empty Actor response. Match
            # the other datasets: restore its pre-generation checkpoint directly.
            choice, k = None, 0
            selection_reason = 'NO_EXECUTED_ACTION_PRE_GENERATION_RESTORE'
        failed_state = self.env.public_state()
        if self.settings.environment_call_limit is not None and self.environment_calls >= self.settings.environment_call_limit:
            raise ResourceLimit('ENVIRONMENT_CALL_LIMIT')
        self.environment_calls += 1
        self.env.restore(self.checkpoints[k])
        self.restore_count += 1
        self.archived_paths.append(failed_path)
        self.archived_failures.append(failure)
        self.active_path = failed_path[:k]
        self.branch_start_length = k
        self.checkpoints = self.checkpoints[:k+1]
        self.seen_edges = set()
        self.protocol_feedback = ''
        self.recovery_memory = ('Red recovery reflection (unverified model advice):\n'+reflection.text+
            '\nActual restored CHECKPOINT '+str(k)+':\n'
            'Saved-state snapshot at recovery entry (historical after further actions):\n'+
            self._operand_record(checkpoint=self.checkpoints[k])+
            '\nThe actual current pool and environment receipts take precedence over this note.')
        self.pending_events = [dict(kind='RED_HANDOFF', event_key='red:1',
            node_ids=failed_path[k:], original_task=self.env.puzzle, advice=reflection.text,
            checkpoint_index=k, failed_public_state=failed_state, failure=failure,
            restored_public_state=self.env.public_state(),
            semantics='Failure belongs to the abandoned branch. Choose the next action freely from the restored state.')]
        self._emit('red_checkpoint_restored', checkpoint_index=k,
                   checkpoint_selection_response=choice,
                   reason=selection_reason, archived_path=failed_path, active_path=self.active_path,
                   reflection=reflection.text,
                   new_action_budget=self.settings.recovery_actions)

    def run(self, task_id, puzzle):
        self.env = Game24Environment(puzzle)
        self.started = time.monotonic()
        self.nodes, self.node_checkpoints = {}, {}
        self.active_path, self.archived_paths = [], []
        self.archived_failures = []
        self.branch_start_length = 0
        self.checkpoints = [self.env.checkpoint()]
        self.events, self.pending_events, self.generations = [], [], []
        self.event_memory, self.current_event_record = [], None
        self.event_facts, self.delivered_facts = {}, set()
        self.yellow_incidents, self.protocol_incidents = [], []
        self.recovery_memory = ''
        self.protocol_feedback = ''
        self.seen_edges, self.shown_branches = set(), set()
        self.current_branch_key = None
        self.model_requests = self.red_attempts = self.yellow_entries = 0
        self.forward_actions = self.recovery_actions = self.restore_count = 0
        self.environment_calls = 1
        self.input_tokens = self.output_tokens = 0
        self.queue_seconds = 0.0
        self.token_accounting_complete = self.timing_accounting_complete = True
        error = None
        try:
            status = self._window(False)
            if status != 'SUCCESS' and self.settings.ablation != 'without_red':
                self._red(status)
                status = self._window(True)
                if status in ('WINDOW_EXHAUSTED', 'EMPTY_ACTOR_OUTPUT', 'ARITHMETIC_DEAD_END',
                              'YELLOW_REQUESTED_BACKTRACK', 'PROTOCOL_STALL'):
                    status = 'RECOVERY_'+status
            elif status in ('WINDOW_EXHAUSTED', 'EMPTY_ACTOR_OUTPUT', 'ARITHMETIC_DEAD_END',
                            'YELLOW_REQUESTED_BACKTRACK', 'PROTOCOL_STALL'):
                status = 'INITIAL_'+status
        except ResourceLimit as exc:
            status, error = 'RESOURCE_LIMIT', str(exc)
            self._emit('resource_limit', reason=error)
        except ProtocolError as exc:
            status, error = 'PROTOCOL_ERROR', str(exc)
            self._emit('fatal', error=error)
        elapsed = time.monotonic()-self.started
        cost = dict(model_requests=self.model_requests, environment_calls=self.environment_calls,
            input_tokens=self.input_tokens if self.token_accounting_complete else None,
            output_tokens=self.output_tokens if self.token_accounting_complete else None,
            wall_elapsed_seconds=elapsed,
            model_queue_seconds=self.queue_seconds if self.timing_accounting_complete else None,
            queue_excluded_elapsed_seconds=elapsed-self.queue_seconds if self.timing_accounting_complete else None,
            token_accounting_complete=self.token_accounting_complete,
            timing_accounting_complete=self.timing_accounting_complete,
            checkpoint_restores=self.restore_count, replay_actions=0)
        return TaskResult(task_id, puzzle, status, self.env.reward,
            status == 'SUCCESS' and self.env.reward == 1,
            self.forward_actions, self.recovery_actions, self.model_requests,
            self.red_attempts, self.yellow_entries, [r.to_dict() for r in self.env.history],
            self.events, dict(nodes=list(self.nodes.values()), active_path=self.active_path,
                              archived_paths=self.archived_paths,
                              local_events=deepcopy(self.yellow_incidents),
                              adaptation_version=GAME24_ADAPTATION_VERSION), cost, self.settings.as_dict(), error)
