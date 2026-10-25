"""Public executed continuation at an existing Red event; no inferred task state."""
import json
from icr_core import ProtocolError


def active_trial_evidence(controller):
    trial=controller.forward_trials[controller.active_trial_id]
    prefix=list(trial.prefix_occurrence_ids)
    lineage=[o.occurrence_id for o in controller.lineage]
    if lineage[:len(prefix)]!=prefix:
        raise ProtocolError('active trial does not retain its declared checkpoint prefix')
    suffix=lineage[len(prefix):]
    if suffix!=list(trial.actual_suffix_occurrence_ids):
        raise ProtocolError('active trial receipts differ from the executed lineage')
    remaining=trial.path_window_limit-len(lineage)
    if remaining<0:
        raise ProtocolError('active trial exceeded its action window')
    selection=json.loads(trial.advice)
    if selection['checkpoint_index']!=trial.checkpoint_index:
        raise ProtocolError('Red selection and active trial checkpoint differ')
    executions=[]
    for key in suffix:
        occurrence=controller.occurrences[key]
        receipt=occurrence.receipt
        executions.append({'occurrence_id':key,'parent_occurrence_id':occurrence.execution_parent_occurrence_id,
            'action':occurrence.action,'observation':receipt['observation'],
            'accepted':receipt['native_accepted'],'done':receipt['done'],
            'public_terminal_score':receipt['public_terminal_score']})
    return {'trial_id':trial.trial_id,'checkpoint_index':trial.checkpoint_index,
            'retained_prefix_occurrence_ids':prefix,
            'executed_continuation':executions,'remaining_actions':remaining,
            'review_record':selection,
            'review_record_status':'MODEL_ADVICE_AT_RESTORE_NOT_CURRENT_STATE_OR_PENDING_ACTIONS',
            'interpretation':'Every listed action already executed on the current continuation. '
                'Use the actual receipts to assess the original task. Rejected proposals were not '
                'executed as proposals; later executed occurrences are separately recorded. '
                'An accepted command establishes its receipt, not completion of the entire task.'}


def factor_node_interfaces(nodes):
    interfaces=[]
    indices={}
    for node in nodes:
        state=node['state_after']
        commands=state.pop('available_actions')
        key=tuple(commands)
        if key not in indices:
            indices[key]=len(interfaces)
            interfaces.append({'available_actions':commands})
        node['public_interface_id']=indices[key]
    return interfaces
