"""Index literal public receipts on one retained prefix; infer no hidden state."""
from copy import deepcopy

from icr_core import ProtocolError


def receipt_references(history):
    """References mean observed events, never current inventory/goal certificates.

    None explicitly means no such accepted receipt exists on this prefix.
    Transfer events are retained after the last inventory query so an old
    inventory answer is never presented as an up-to-date inventory snapshot.
    """
    latest_navigation=None
    latest_inventory=None
    transfers=[]
    processing=[]
    containers=[]
    for index,receipt in enumerate(history):
        action=receipt['action']
        if type(action) is not str or type(receipt['native_accepted']) is not bool:
            raise ProtocolError('public receipt index requires an actual action and boolean acceptance')
        if not receipt['native_accepted']:
            continue
        if action.startswith('go to '):
            latest_navigation=index
        elif action=='inventory':
            latest_inventory=index
            transfers=[]
        elif action.startswith(('take ','put ')):
            transfers.append(index)
        elif action.startswith(('clean ','heat ','cool ','slice ')):
            processing.append(index)
        elif action.startswith(('open ','close ')):
            containers.append(index)
    return {'retained_action_count':len(history),
            'latest_accepted_navigation':latest_navigation,
            'latest_inventory_query':latest_inventory,
            'accepted_transfers_after_inventory':transfers,
            'accepted_processing_receipts':processing,
            'accepted_container_changes':containers,
            'latest_receipt':len(history)-1 if history else None}


def public_receipt_table(history,indices):
    result=[]
    for index in sorted(set(indices)):
        if type(index) is not int or not 0<=index<len(history):
            raise ProtocolError('public receipt reference is outside its retained prefix')
        receipt=history[index]
        result.append({'history_index':index,'action':receipt['action'],
                       'observation':receipt['observation'],
                       'native_accepted':receipt['native_accepted'],
                       'done':receipt['done'],'public_terminal_score':receipt['public_terminal_score']})
    return result


def referenced_indices(view):
    values=[]
    for key in ('latest_accepted_navigation','latest_inventory_query','latest_receipt'):
        if view[key] is not None:values.append(view[key])
    for key in ('accepted_transfers_after_inventory','accepted_processing_receipts','accepted_container_changes'):
        values.extend(view[key])
    return values


RECEIPT_SEMANTICS=(
    'Indices are zero-based positions on this retained execution prefix, not ETT occurrence IDs. '
    'Every reference points to an exact previously observed action and receipt. '
    'null means no such receipt on this prefix, not an empty inventory or absent object. '
    'Read transfer receipts after an inventory answer before judging possession. '
    'Processing receipts establish only the reported executed operation, not task success. '
    'A rejected command does not identify its cause. The command catalog is the current public interface, '
    'not a recommended plan or proof of a prerequisite. No additional environment call was made.')


def event_state_evidence(initial,history,state,action_limit):
    if history:
        if history[-1]['after']!=state:
            raise ProtocolError('event state differs from its last retained receipt')
    elif initial['after']!=state:
        raise ProtocolError('root event state differs from reset')
    view=receipt_references(history)
    remaining=action_limit-len(history)
    if remaining<0:raise ProtocolError('negative remaining action budget')
    return {'original_task_observation':initial['observation'],'receipt_view':view,
            'receipts':public_receipt_table(history,referenced_indices(view)),
            'current_admissible_commands':deepcopy(state['available_actions']),
            'current_observation':state['observation'],'remaining_actions':remaining,
            'semantics':RECEIPT_SEMANTICS}


def add_checkpoint_receipts(evidence,candidates):
    """Use one literal receipt table for all nested prefixes in a Red review."""
    if not candidates:raise ProtocolError('Red requires at least one cached candidate')
    by_index=dict(candidates)
    if len(by_index)!=len(candidates):raise ProtocolError('duplicate checkpoint index')
    longest=max((checkpoint['history'] for _,checkpoint in candidates),key=len)
    references=set()
    for row in evidence['candidates']:
        index=row['checkpoint_index']
        checkpoint=by_index[index]
        history=checkpoint['history']
        if len(history)!=index or history!=longest[:index]:
            raise ProtocolError('Red receipt views must be exact nested retained prefixes')
        state=checkpoint['environment']['public_state']
        last=history[-1]['after'] if history else checkpoint['initial']['after']
        if last!=state:raise ProtocolError('cached state differs from its retained prefix')
        view=receipt_references(history)
        row['public_receipt_refs']=view
        references.update(referenced_indices(view))
    evidence['public_receipt_table']=public_receipt_table(longest,references)
    evidence['public_receipt_semantics']=RECEIPT_SEMANTICS
