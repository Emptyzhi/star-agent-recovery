"""Exact public-interface references for Yellow, as already used by Red."""
from copy import deepcopy
from tree_history import canonical


def factor_path_interfaces(path):
    rows=deepcopy(path)
    interfaces=[]
    indices={}
    for row in rows:
        for side in ('before','after'):
            state=row[side]
            if 'public_interface_id' in state:
                raise ValueError('cannot factor an already referenced public state')
            interface={'available_actions':state.pop('available_actions')}
            key=canonical(interface)
            if key not in indices:
                indices[key]=len(interfaces)
                interfaces.append(interface)
            state['public_interface_id']=indices[key]
    return rows,interfaces
