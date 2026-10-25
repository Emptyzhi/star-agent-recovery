"""Public-trajectory value judgments with a single label on the first line."""
PROTOCOL = 'score_firstline_v7'
WEIGHTS = {'impossible': 0.001, 'likely': 1, 'sure': 20}
EXAMPLES = {
    'game24': [('Numbers left: 11 12', 'impossible'),
               ('Numbers left: 5 7 8', 'likely'),
               ('Numbers left: 10 14', 'sure')],
    'alfworld': [('The task has ended unsuccessfully; no further actions are available.', 'impossible'),
                 ('Goal: put an apple on the table. Observation: You take the apple from the fridge.', 'likely'),
                 ('Goal: put an apple on the table. Observation: You put the apple on the table. Task completed.', 'sure')],
    'webshop': [('The purchase has finished and the observed result says the task failed.', 'impossible'),
                ('The requested item has been found, but its requested size has not yet been selected.', 'likely'),
                ('The item, price and selected options match all visible requirements; Buy Now is available.', 'sure')],
}


def value_prompt(dataset, trajectory):
    goal = {'game24': 'Game24 goal: use every current number occurrence exactly once with +, -, *, / to reach 24. ',
            'alfworld': 'ALFWorld goal: complete the household task stated in the trajectory. ',
            'webshop': 'WebShop goal: buy an item and options matching all stated requirements. '}[dataset]
    examples = '\n\n'.join('Trajectory:\n' + text + '\nEvaluation: ' + label
                            for text, label in EXAMPLES[dataset])
    return (goal + 'Judge whether the task can be completed from the public trajectory. '
            'Use only observed information. Return exactly one lowercase label: '
            'impossible, likely, or sure. Output the label first and nothing else.\n\n'
            + examples + '\n\nTrajectory:\n' + trajectory
            + '\n\nReturn one label only.\nEvaluation:')


def parse_labels(outputs):
    if not outputs:
        raise ValueError('ToT evaluator returned no samples')
    total = 0
    for output in outputs:
        # Protocol permits outer whitespace and a terminal newline, never prose.
        label = output.strip()
        if label not in WEIGHTS:
            raise ValueError('ToT evaluator did not return one complete label: ' + repr(output))
        total += WEIGHTS[label]
    return total
