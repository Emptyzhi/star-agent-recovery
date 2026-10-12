"""Classical binary interaction designs and discrepancy traversal, not new theory."""

from dataclasses import dataclass
from enum import Enum
from itertools import combinations, product
import random


DEPTH = 4
ATTEMPT_CAP = 8


class ScheduleKind(str, Enum):
    COVERING = "COVERING"
    MATCHED_RANDOM = "MATCHED_RANDOM"
    ILDS = "ILDS"
    SINGLE_FORK = "SINGLE_FORK"


def validate_bits(bits, *, complete):
    if type(bits) is not tuple or any(type(bit) is not int or bit not in (0, 1) for bit in bits):
        raise ValueError("branch modes must be a tuple of integer zeroes and ones")
    if len(bits) > DEPTH or (complete and len(bits) != DEPTH):
        raise ValueError("branch mode length violates the four-slot contract")
    return bits


def _seed(value):
    if type(value) is not int or value < 0:
        raise ValueError("design seed must be a nonnegative integer")
    return value


def column_permutation(seed):
    columns = list(range(DEPTH))
    random.Random(_seed(seed)).shuffle(columns)
    return tuple(columns)


def _permute(rows, columns):
    if type(columns) is not tuple or sorted(columns) != list(range(DEPTH)):
        raise ValueError("columns must be a permutation of the four temporal positions")
    if any(type(column) is not int for column in columns):
        raise ValueError("column indices must be integers")
    return tuple(tuple(validate_bits(row, complete=True)[column] for column in columns) for row in rows)


def covering_rows(columns):
    rows = [(x, y, z, x ^ y) for x, y, z in product((0, 1), repeat=3)]
    rows.sort(key=lambda row: (sum(row), row))
    return _permute(rows, columns)


def matched_random_rows(seed, columns):
    """Uniform within fixed weight strata; good random coverage is never rejected."""
    rng = random.Random(_seed(seed))
    rows = []
    for weight, count in ((0, 1), (1, 1), (2, 3), (3, 3)):
        stratum = [row for row in product((0, 1), repeat=DEPTH) if sum(row) == weight]
        rows.extend(rng.sample(stratum, count))
    return _permute(rows, columns)


def ilds_rows():
    """Korf-style exact-discrepancy layers with preferred-first depth-first order."""
    def visit(prefix, remaining):
        free = DEPTH - len(prefix)
        if free == 0:
            if remaining == 0:
                yield prefix
            return
        if remaining < free:
            yield from visit(prefix + (0,), remaining)
        if remaining > 0:
            yield from visit(prefix + (1,), remaining - 1)

    return tuple(row for discrepancy in range(DEPTH + 1) for row in visit((), discrepancy))


def rows_for(kind, *, random_seed, columns):
    if not isinstance(kind, ScheduleKind):
        raise ValueError("unknown schedule kind")
    if kind is ScheduleKind.COVERING:
        return covering_rows(columns)
    if kind is ScheduleKind.MATCHED_RANDOM:
        return matched_random_rows(random_seed, columns)
    rows = ilds_rows()
    if kind is ScheduleKind.SINGLE_FORK:
        return tuple(row for row in rows if sum(row) <= 1)
    return rows


@dataclass(frozen=True)
class AttemptDesign:
    attempt: int
    replica: int
    row_index: int
    bits: tuple[int, ...]


def attempt_design(kind, attempt, *, master_seed, benchmark, task):
    if not isinstance(kind, ScheduleKind):
        raise ValueError("unknown schedule kind")
    if type(attempt) is not int or not 0 <= attempt < ATTEMPT_CAP:
        raise ValueError("attempt exceeds the frozen eight-attempt limit")
    for name, value, bound in (("benchmark", benchmark, 2), ("task", task, 500)):
        if type(value) is not int or not 0 <= value < bound:
            raise ValueError(name + " is outside the design address space")
    _seed(master_seed)
    size = {ScheduleKind.COVERING: 8, ScheduleKind.MATCHED_RANDOM: 8,
            ScheduleKind.ILDS: 16, ScheduleKind.SINGLE_FORK: 5}[kind]
    replica, row_index = divmod(attempt, size)
    task_address = benchmark * 500 + task
    base = (master_seed * 1000 + task_address) * 32
    columns = column_permutation(base)
    rows = rows_for(kind, random_seed=base + 1 + replica, columns=columns)
    return AttemptDesign(attempt, replica, row_index, rows[row_index])


def realized_pairs(consumed_bits):
    """Unreached slots never become implicit zeroes."""
    bits = validate_bits(consumed_bits, complete=False)
    return frozenset((left, right, bits[left], bits[right])
                     for left, right in combinations(range(len(bits)), 2))


def planned_pairs(rows):
    covered = set()
    for row in rows:
        covered.update(realized_pairs(validate_bits(row, complete=True)))
    return frozenset(covered)


PAIR_DENOMINATOR = 4 * len(tuple(combinations(range(DEPTH), 2)))
