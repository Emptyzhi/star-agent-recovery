"""Project previously unseen public views at the existing Red handoff."""
from icr_core import ProtocolError


def discarded_discoveries(path,checkpoint_index):
    if type(checkpoint_index) is not int or not 0<=checkpoint_index<len(path):
        raise ProtocolError('discovery projection needs an actual failed-path checkpoint')
    seen=set()
    selected=[]
    for index,node in enumerate(path):
        receipt=node.receipt
        # These native command families expose a view. Transfer, processing,
        # inventory and thought receipts must not become restored-state claims.
        view=(node.action=='look' or node.action.startswith(('go to ','open ','examine ')))
        if not receipt['native_accepted'] or not view:
            continue
        key=(node.action,receipt['observation'])
        if index>=checkpoint_index and key not in seen:
            selected.append(node.occurrence_id)
        seen.add(key)
    return tuple(selected)
