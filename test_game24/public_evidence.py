"""Render the provenance of executed Game24 operations, never a candidate plan."""
from .game24_env import OPERATION_RE


def executed_operand_lineage(puzzle, history, state):
    """Expand receipt dependencies without evaluating or proposing arithmetic."""
    expressions = {
        'constant.' + str(index): (value, (index,))
        for index, value in enumerate(puzzle.split(), 1)
    }
    for receipt in history:
        if not receipt.accepted or receipt.produced is None:
            continue
        operation = OPERATION_RE.fullmatch(receipt.action)
        if operation is None or len(receipt.consumed) != 2:
            raise ValueError('executed operation lacks its binary provenance')
        left, right = (expressions[source] for source in receipt.consumed)
        inputs = left[1] + right[1]
        if len(inputs) != len(set(inputs)):
            raise ValueError('executed operand provenance reuses an input occurrence')
        expressions[receipt.produced] = (
            '(' + left[0] + ' ' + operation.group(2) + ' ' + right[0] + ')', inputs)
    lines = ['[EXECUTED_OPERAND_LINEAGE]',
             'Recorded expressions show input occurrences and accepted operation receipts, preserving operand order and parentheses. Their values are recorded environment facts, not proposed repairs. Task success still requires an accepted final answer.']
    for source, value in zip(state['sources'], state['numbers'], strict=True):
        expression, inputs = expressions[source]
        lines.append(source + ': recorded value ' + value + '; executed expression ' + expression
                     + '; original occurrences ' + ', '.join('constant.' + str(i) for i in inputs))
    return '\n'.join(lines)
