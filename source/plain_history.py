"""Chronological model input for the all-role event-subtree ablation."""
from icr_core import BudgetExhausted

POLICY = 'all_roles_native_chronological_history_v3'


def install(controller):
    controller.plain_history = []
    original_emit = controller.emit

    def emit(event, **data):
        if event == 'scoped_action_executed':
            if controller.episode.initial['after'].get('benchmark') == 'alfworld':
                record = '> ' + data['action'] + '\n' + data['receipt']['observation']
            else:
                record = 'Action: ' + data['action'] + '\nObservation: ' + data['receipt']['observation']
            controller.plain_history.append(record)
        elif event == 'yellow_local_diagnosis':
            controller.plain_history.append('Local diagnosis (model advice): ' + data['diagnosis']['text'])
        elif event == 'red_failure_reflected':
            controller.plain_history.append('Failure reflection (model advice): ' + data['reflection'])
        elif event == 'red_checkpoint_restored':
            controller.plain_history.append(
                'Restored the environment to before ACTION ' + str(data['checkpoint_index'])
                + '. The actions above remain historical records, not the current environment state.\n'
                + 'Current observation: ' + controller.episode.snapshot()['observation'])
        elif event == 'actor_empty_output_failure':
            controller.plain_history.append('Actor returned an empty response; no action was executed.')
        elif event == 'yellow_protocol_failure':
            controller.plain_history.append('Unexecuted local proposal: ' + repr(data['raw_output']))
        original_emit(event, **data)

    controller.emit = emit
    controller.episode.plain_history_prompt = None


def transcript(controller):
    records = '\n'.join(controller.plain_history)
    return controller.episode.initial['observation'] + '\n' + records + ('\n' if records else '')


def indexed_history(controller, path):
    # Indices address existing checkpoints, not event nodes or branch relations.
    return (transcript(controller) + '\n\nCurrent restorable actions; restore BEFORE the selected action:\n'
            + '\n'.join('ACTION %d: %s' % (i, row.action) for i, row in enumerate(path)))


def local_context(controller):
    state = controller.episode.snapshot()
    text = transcript(controller)
    text += '\nCurrent available actions:\n' + '\n'.join(state['available_actions'])
    text += '\nProposed action: ' + controller.simple_yellow_proposal
    return text


def bind(controller, scope):
    state = controller.episode.snapshot()
    alfworld = state.get('benchmark') == 'alfworld'
    if alfworld:
        from alf_actor_context import memory_base
        text = memory_base(controller.episode.base_prompt, ()) + '\nHere is the task:\n'
    else:
        text = controller.episode.base_prompt + '\n\n'
    text += transcript(controller)
    if scope.kind.value == 'GREEN' and controller.episode.recovery_prefix_length is not None:
        if alfworld:
            text += '\nCurrent available commands:\n' + '\n'.join(state['available_actions']) + '\n'
        else:
            text += '\nCurrent available actions:\n' + '\n'.join(state['available_actions']) + '\n'
    if scope.kind.value == 'YELLOW':
        # Keep the native role instructions and current public interface, but no
        # controller-selected past records, diagnosed facts, or repeated advice.
        text += '\nCurrent available actions:\n' + '\n'.join(state['available_actions'])
        text += '\nProposed action: ' + controller.simple_yellow_proposal + '\n'
        if alfworld:
            text += ('Correct the specific rejected operation or break the recorded local loop. '
                     'Check the original task object, actual possession, location and native menu. '
                     'Use one action to check a missing prerequisite or correct it; do not start '
                     'a separate whole-room search. Do not take a substitute object merely because '
                     'it is available. Historical Red advice is fallible; actual receipts take priority. '
                     'Return one native action.\n')
            text += ('Choose one current native action using the latest actual feedback. '
                     'The initial plan is a hypothesis, not an instruction to repeat. If its '
                     'check failed or added no evidence, revise the next operation using current '
                     'facts instead of prescribing the same check again. Do not treat a new '
                     'observation phrase as a solved obstruction. Never invent the result of '
                     'an unexecuted action. Return only the next native action.\n')
        else:
            if state['page_type'] == 'init':
                text += ('This is the initial search form. Its native action is search[query]. '
                         'Return one concrete search[query] action. Search is a visible UI label, '
                         'not click[Search]; click[Back to Search], click[Next >], and other click[...] '
                         'actions are not executable on this page.\n')
            else:
                text += 'Choose exactly one command from the current page native action menu.\n'
            text += ('Repair this local page/command obstruction with exactly one testable native action. '
                     'The receipt will decide whether the local repair made progress; do not plan a search sequence. '
                     'Do not repeat an ineffective check. '
                     'Do not invent product attributes or treat a partial score as a per-attribute explanation. '
                     'Green owns the shopping plan. Return exactly one native action.\n')
    text += '>' if alfworld else '\nAction:'
    reserve = (controller.settings.yellow_output_tokens if scope.kind.value == 'YELLOW'
               else controller.settings.actor_output_tokens)
    if controller.episode.prompt_token_counter(text) > min(
            controller.episode.prompt_context_limit - reserve, controller.episode.prompt_service_limit):
        raise BudgetExhausted('complete chronological history exceeds context; no truncation')
    controller.episode.plain_history_prompt = text
    controller.episode.last_history_window = dict(
        truncated=False, policy=POLICY, retained_records=len(controller.plain_history),
        event_subtree_separate=False)
    return text
