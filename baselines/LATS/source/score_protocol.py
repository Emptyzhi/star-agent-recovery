"""Public-trajectory correctness scores with one complete integer output."""
import re

PROTOCOL = 'score_firstline_v7'
MODEL_FAILURE_POLICY = 'empty_skip_unscored_v8'


class ScoreProtocolError(ValueError):
    """A completed model response violates the declared score format."""


def recorded_score(outputs, lower, upper, failures, label):
    try:
        return parse_score(outputs, lower, upper)
    except ScoreProtocolError as error:
        failures.append(dict(label=label, error=str(error), outputs=list(outputs),
                             ranking_value=-1, policy=MODEL_FAILURE_POLICY))
        # User policy: unscored candidates rank below valid scores; rewards
        # remain the original environment receipts, including terminal success.
        return -1
EXAMPLES = {
    'game24': [('The final expression is invalid and the task has ended unsuccessfully.', 0),
               ('The arithmetic steps are valid; several numbers remain and no final answer has been given.', 5),
               ('The final expression uses every original input once and the environment reports success.', 10)],
    'alfworld': [('The task has ended unsuccessfully.', 0),
                 ('Goal: put an apple on the table. Observation: You take the apple from the fridge.', 5),
                 ('Goal: put an apple on the table. Observation: You put the apple on the table. Task completed.', 10)],
    'webshop': [('The completed purchase is observed to match none of the requested requirements.', 1),
                ('The item matches some requirements, but the selected options do not match all of them.', 5),
                ('The purchased item and options match every requested requirement; the task reports success.', 10)],
}


def value_prompt(dataset, trajectory, memory=''):
    lower = 1 if dataset == 'webshop' else 0
    goal = {'game24': 'Game24 goal: use each original input occurrence exactly once to reach 24 with +, -, *, /. ',
            'alfworld': 'ALFWorld goal: complete the household task stated in the trajectory. ',
            'webshop': 'WebShop goal: buy the item and options matching every stated requirement. '}[dataset]
    examples = '\n\n'.join('Trajectory:\n' + text + '\nScore: ' + str(score)
                            for text, score in EXAMPLES[dataset])
    return (goal + 'Evaluate progress toward the stated task using only public observations. '
            f'Return exactly one integer from {lower} to 10; 10 means full correctness. '
            'Output the integer first and nothing else.\n\n' + examples
            + '\n\nEarlier failures and reflections:\n' + memory
            + '\n\nCurrent trajectory:\n' + trajectory
            + f'\n\nReturn one integer from {lower} to 10 only.\nScore:')


def parse_score(outputs, lower, upper):
    if not outputs:
        raise ScoreProtocolError('LATS evaluator returned no score samples')
    # Consume the first author sample; outer protocol whitespace is permitted.
    text = outputs[0].strip()
    if re.fullmatch(r'10|[0-9]', text) is None:
        raise ScoreProtocolError('LATS evaluator did not return one complete integer: ' + repr(outputs[0]))
    score = int(text)
    if not lower <= score <= upper:
        raise ScoreProtocolError('LATS evaluator score is outside the declared range')
    return score / 10
