"""Native model-visible observations, distinct from full audit receipts."""
from copy import deepcopy


def model_feedback(receipt, dataset):
    result = {key: receipt[key] for key in ('observation', 'done', 'reward')}
    result['action_accepted'] = receipt['accepted']
    result['task_success'] = receipt['done'] and receipt['reward'] == 1
    if dataset == 'game24':
        # Game24 exposes grammar placeholders here, unlike ALFWorld's concrete
        # admissible commands. Native demonstrations already define the grammar.
        result['state'] = {key: deepcopy(value) for key, value in receipt['state'].items()
                           if key != 'available_actions'}
    elif dataset == 'alfworld':
        result['available_actions'] = deepcopy(receipt['state']['available_actions'])
    elif dataset != 'webshop':
        raise ValueError('unknown feedback dataset')
    return result
