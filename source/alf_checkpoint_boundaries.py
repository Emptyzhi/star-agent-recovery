"""Compact prefix witnesses; neither interpret hidden state nor select a checkpoint."""
from alf_public_receipts import receipt_references
from icr_core import ProtocolError

COLUMNS=['checkpoint_index','last_completed_receipt','next_action_NOT_executed',
         'last_accepted_navigation','latest_inventory_receipt','accepted_transfers_after_inventory',
         'accepted_processing_receipts','accepted_container_changes','remaining_actions']


def checkpoint_boundaries(history,candidates,action_limit):
    rows=[];seen=set()
    for candidate in sorted(candidates,key=lambda row:row['checkpoint_index']):
        k=candidate['checkpoint_index']
        if type(k) is not int or not 0<=k<len(history) or k in seen:
            raise ProtocolError('checkpoint index must identify a distinct observed next action')
        seen.add(k)
        remaining=action_limit-k
        if candidate['remaining_actions']!=remaining or remaining<=0:
            raise ProtocolError('checkpoint budget does not match its retained prefix')
        refs=receipt_references(history[:k])
        rows.append([k,k-1 if k else None,history[k]['action'],
            refs['latest_accepted_navigation'],refs['latest_inventory_query'],
            refs['accepted_transfers_after_inventory'],refs['accepted_processing_receipts'],
            refs['accepted_container_changes'],remaining])
    return {'columns':COLUMNS,'rows':rows,
            'semantics':'A row is the state BEFORE its next_action_NOT_executed. Checkpoint k '
                'contains exactly receipts [0,k), ending at last_completed_receipt; receipt k '
                'has NOT happened in that state. For k=0, the current observation is the reset '
                'observation. Otherwise it is the observation of last_completed_receipt. '
                'All numeric evidence references point to the indexed FAILED_EXECUTION_HISTORY '
                'above; every such reference in a row is less than k. Read the exact action and '
                'observation at those references. null means no corresponding receipt, not empty '
                'inventory or absent objects. Accepted processing is a recorded operation, not '
                'proof of task completion. Full snapshots stay in the controller.'}
