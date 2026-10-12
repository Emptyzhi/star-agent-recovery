"""ALFWorld public adapter for the cluster's explicit-memory Reflexion control."""
from comparison_native import memory_base
from icr_core import ProtocolError, Role, environment_call
from icr_identity import NodeAddress, Phase


def _actor_history(episode):
    # The active action/observation transcript is distinct from the controller's
    # full ETT. Preserve accepted discoveries between the prefix and recent tail.
    return ''.join('> ' + row['action'] + '\n' + row['observation'] + '\n'
                   for row in episode.history)


def actor_input(episode, memory):
    text = memory_base(episode.base_prompt, memory)
    text += '\nHere is the task:\n' + episode.initial['observation'] + '\n'
    text += _actor_history(episode)
    return text


def execute_native(native, episode, oracle, ledger, *, task, attempt, memory,
                   delivered, emit, actor_stop_mode=None):
    if delivered is not True or actor_stop_mode != 'first_line':
        raise ProtocolError('Reflexion requires explicit full-memory delivery and native first-line actions')
    environment_call(episode, ledger, episode.reset)
    for ordinal in range(episode.action_limit):
        prompt = actor_input(episode, memory) + '>'
        address = NodeAddress(0, task, Phase.TREE, attempt, (), ordinal)
        oracle.bind(address, episode, prompt, memory)
        emit('reflexion_actor_prompt', attempt=attempt, ordinal=ordinal,
             memory=list(memory), prompt=prompt)
        generation = oracle.request(address, Role.ORDINARY, 0, prompt, ledger,
                                    stop_mode=actor_stop_mode)
        parsed = episode.parse(generation.text)
        if not parsed.grammar_valid:
            raise ProtocolError('Reflexion actor returned an empty or invalid native action')
        receipt = environment_call(episode, ledger, episode.step, parsed.action)
        emit('native_environment_receipt', receipt=receipt, attempt=attempt,
             cumulative_cost=ledger.to_dict())
        if receipt['done']:
            break
    snapshot = episode.snapshot()
    return {'status': 'TERMINAL' if snapshot['done'] else 'EPISODE_ACTION_LIMIT',
            'public_terminal_score': snapshot['public_terminal_score'],
            'success': snapshot['public_terminal_score'] == 1,
            'failed_log': actor_input(episode, memory), 'history': list(episode.history),
            'reset_receipt': episode.initial, 'result_metadata': episode.result_metadata(),
            'executed_actions': [row['action'] for row in episode.history],
            'cumulative_cost': ledger.to_dict()}
