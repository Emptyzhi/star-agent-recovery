"""Required resident capacity follows the selected author's dataset controller."""


def required_output_cap(dataset, method):
    if dataset not in ('alfworld', 'game24', 'webshop'):
        raise ValueError('unsupported dataset')
    if method == 'ga_rollback':
        return 1200 if dataset == 'alfworld' else 500
    return {'react': 100, 'reflexion': 256, 'tot': 1000, 'lats': 100}[method]
