"""State evidence derived only from accepted native transfer receipts."""
import re

from icr_core import ProtocolError


def restored_facts(history, state):
    if history and history[-1]['after'] != state:
        raise ProtocolError('restored facts require the exact retained prefix state')
    held = {}
    transfers = []
    navigation = None
    for index, receipt in enumerate(history):
        if type(receipt['native_accepted']) is not bool:
            raise ProtocolError('transfer evidence requires boolean acceptance')
        if not receipt['native_accepted']:
            continue
        action = receipt['action']
        if action.startswith('go to '):
            navigation = dict(receipt_index=index, action=action,
                              observation=receipt['observation'])
        if action.startswith('take '):
            match = re.fullmatch(r'take (.+) from (.+)', action)
            if match is None:
                raise ProtocolError('accepted take is outside the native transfer grammar')
            held[match[1]] = index
        elif action.startswith('put '):
            match = re.fullmatch(r'put (.+) in/on (.+)', action)
            if match is None:
                raise ProtocolError('accepted put is outside the native transfer grammar')
            held.pop(match[1], None)
        else:
            continue
        transfers.append({'receipt_index': index, 'action': action,
                          'observation': receipt['observation']})
    return {
        'retained_action_count': len(history),
        'known_held_objects': [{'object': item, 'successful_take_receipt': index}
                               for item, index in held.items()],
        'accepted_transfer_receipts': transfers,
        'latest_successful_navigation': navigation,
        'latest_successful_transfer': transfers[-1] if transfers else None,
        'latest_receipt': (dict(receipt_index=len(history)-1, action=history[-1]['action'],
            accepted=history[-1]['native_accepted'], observation=history[-1]['observation'])
            if history else None),
        'current_observation': state['observation'],
        'semantics': 'A successful take persists until a successful put of that object. '
                     'Rejected transfers do not change possession. These facts use only the retained '
                     'prefix; they do not assert an initially empty inventory or include discarded '
                     'future actions. An empty known-held list means no held object is established '
                     'by this transfer evidence. Observation and new receipts override advice.',
    }
