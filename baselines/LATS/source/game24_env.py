"""Strict public Game24 environment used by STAR.

The environment owns arithmetic validity. The controller never evaluates a
candidate to guess intent or repair malformed output.
"""

from __future__ import annotations

import ast
import copy
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any


NUMBER = r"-?(?:\d+(?:\.\d+)?|\d+/\d+)"
OPERATION_RE = re.compile(
    rf"^\s*({NUMBER})\s*([+\-*/])\s*({NUMBER})\s*=\s*({NUMBER})\s*$"
)
ANSWER_RE = re.compile(r"^\s*answer\s*:\s*(.*?)\s*=\s*([^=]+)\s*$", re.I)


def parse_number(value: str) -> Fraction:
    value = str(value).strip()
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return Fraction(int(numerator), int(denominator))
    return Fraction(value)


def format_number(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def parse_operation(action: str) -> tuple[Fraction, str, Fraction, Fraction] | None:
    match = OPERATION_RE.fullmatch(str(action))
    if match is None:
        return None
    left, operator, right, result = match.groups()
    return parse_number(left), operator, parse_number(right), parse_number(result)


def _eval_expression(node: ast.AST) -> tuple[Fraction, list[Fraction]]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if isinstance(node.value, bool):
            raise ValueError("boolean literal is not arithmetic")
        value = Fraction(str(node.value))
        return value, [value]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value, inputs = _eval_expression(node.operand)
        return (-value if isinstance(node.op, ast.USub) else value), inputs
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
    ):
        left, left_inputs = _eval_expression(node.left)
        right, right_inputs = _eval_expression(node.right)
        if isinstance(node.op, ast.Add):
            value = left + right
        elif isinstance(node.op, ast.Sub):
            value = left - right
        elif isinstance(node.op, ast.Mult):
            value = left * right
        else:
            if right == 0:
                raise ValueError("division by zero")
            value = left / right
        return value, left_inputs + right_inputs
    raise ValueError("expression contains unsupported syntax")


def validate_answer(
    action: str, original_numbers: tuple[Fraction, ...]
) -> tuple[bool, str, Fraction | None]:
    match = ANSWER_RE.fullmatch(str(action))
    if match is None:
        return False, "final action must match answer: <expression> = <value>", None
    expression, declared = match.groups()
    try:
        value, inputs = _eval_expression(ast.parse(expression, mode="eval").body)
        declared_value = parse_number(declared)
    except (SyntaxError, ValueError, ZeroDivisionError) as error:
        return False, f"invalid final expression: {error}", None
    if sorted(inputs) != sorted(original_numbers):
        return False, "final expression must use every original number exactly once", value
    if value != declared_value:
        return False, "declared result does not match the expression", value
    if value != 24:
        return False, "final expression does not evaluate to 24", value
    return True, "Success!", value


@dataclass(frozen=True)
class Receipt:
    action: str
    before: dict[str, Any]
    after: dict[str, Any]
    accepted: bool
    done: bool
    reward: float | None
    detail: str
    consumed: tuple[str, ...] = ()
    produced: str | None = None
    parents: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "before": copy.deepcopy(self.before),
            "after": copy.deepcopy(self.after),
            "accepted": self.accepted,
            "done": self.done,
            "reward": self.reward,
            "detail": self.detail,
            "consumed": list(self.consumed),
            "produced": self.produced,
            "parents": list(self.parents),
        }


class Game24Environment:
    """One task with exact Fraction arithmetic and reversible checkpoints."""

    # The native Game24 task permits retries through a twelve-step interaction
    # window. Three accepted binary operations are still enough to consume the
    # pool; later calls can correct an invalid final expression.
    max_actions = 12

    def __init__(self, puzzle: str):
        tokens = puzzle.split()
        if any(re.fullmatch(NUMBER, value) is None for value in tokens):
            raise ValueError('puzzle must be four numeric tokens, not free text')
        values = tuple(parse_number(value) for value in tokens)
        if len(values) != 4:
            raise ValueError("Game24 puzzle must contain exactly four numbers")
        self.puzzle = str(puzzle).strip()
        self.original = values
        self._pool: list[tuple[str, Fraction, str, tuple[int, ...]]] = []
        self.history: list[Receipt] = []
        self.action_count = 0
        self.done = False
        self.reward: float | None = None
        self.reset()

    def reset(self) -> dict[str, Any]:
        self._pool = [
            (f"constant.{index}", value, format_number(value), ())
            for index, value in enumerate(self.original, start=1)
        ]
        self.history = []
        self.action_count = 0
        self.done = False
        self.reward = None
        return self.public_state()

    def checkpoint(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "pool": self._pool,
                "history": self.history,
                "action_count": self.action_count,
                "done": self.done,
                "reward": self.reward,
            }
        )

    def public_checkpoint(self, checkpoint: dict[str, Any]) -> dict[str, Any]:
        """Expose a checkpoint as public evidence without mutating the task."""
        return copy.deepcopy(self._public_from_checkpoint(checkpoint))

    def restore(self, checkpoint: dict[str, Any]) -> None:
        self._pool = copy.deepcopy(checkpoint["pool"])
        self.history = copy.deepcopy(checkpoint["history"])
        self.action_count = int(checkpoint["action_count"])
        self.done = bool(checkpoint["done"])
        self.reward = checkpoint["reward"]
        if self.public_state() != self._public_from_checkpoint(checkpoint):
            raise RuntimeError("checkpoint restore changed public Game24 state")

    def _public_from_checkpoint(self, checkpoint: dict[str, Any]) -> dict[str, Any]:
        pool = checkpoint["pool"]
        return {
            "puzzle": self.puzzle,
            "numbers": [format_number(item[1]) for item in pool],
            "sources": [item[0] for item in pool],
            "action_count": checkpoint["action_count"],
            "max_actions": self.max_actions,
            "accepted_operations": 4-len(pool),
            "done": checkpoint["done"],
            "reward": checkpoint["reward"],
            "available_actions": self._available_actions_for(pool, checkpoint["done"]),
        }

    @staticmethod
    def _available_actions_for(pool: list[Any], done: bool) -> list[str]:
        if done:
            return []
        actions = ["answer: <complete expression using the four original numbers> = 24"]
        if len(pool) >= 2:
            actions.insert(0, "<available binary operation using two current numbers>")
        return actions

    def public_state(self) -> dict[str, Any]:
        return {
            "puzzle": self.puzzle,
            "numbers": [format_number(item[1]) for item in self._pool],
            "sources": [item[0] for item in self._pool],
            "action_count": self.action_count,
            "max_actions": self.max_actions,
            "accepted_operations": 4-len(self._pool),
            "done": self.done,
            "reward": self.reward,
            "available_actions": self._available_actions_for(self._pool, self.done),
        }

    def _find_operand(
        self,
        value: Fraction,
        *,
        exclude_index: int | None = None,
    ) -> tuple[int, tuple[str, Fraction, str, tuple[int, ...]]] | None:
        for index, item in enumerate(self._pool):
            if index != exclude_index and item[1] == value:
                return index, item
        return None

    def _record(
        self,
        action: str,
        before: dict[str, Any],
        accepted: bool,
        detail: str,
        *,
        done: bool | None = None,
        reward: float | None = None,
        consumed: tuple[str, ...] = (),
        produced: str | None = None,
        parents: tuple[int, ...] = (),
    ) -> Receipt:
        after = self.public_state()
        if done is not None:
            after["done"] = done
            after["reward"] = reward
        receipt = Receipt(
            action=action,
            before=before,
            after=after,
            accepted=accepted,
            done=bool(after["done"]),
            reward=after["reward"],
            detail=detail,
            consumed=consumed,
            produced=produced,
            parents=parents,
        )
        self.history.append(receipt)
        return receipt

    def execute(self, action: str) -> Receipt:
        if self.done:
            raise RuntimeError("cannot execute an action after terminal state")
        action = str(action).strip()
        before = self.public_state()
        self.action_count += 1
        try:
            parsed = parse_operation(action)
        except (ValueError, ZeroDivisionError):
            return self._record(action, before, False, 'invalid arithmetic number')
        if parsed is None and action.lower().startswith("answer:"):
            # Native Game24 permits submitting a complete expression directly.
            # Its validity depends on the original inputs, not the number of prior attempts.
            valid, detail, _value = validate_answer(action, self.original)
            if valid:
                self.done = True
                self.reward = 1.0
                return self._record(action, before, True, detail, done=True, reward=self.reward)
            # A rejected final expression is public failure evidence, not a
            # terminal environment state. The model may submit another answer
            # within the native interaction window.
            return self._record(action, before, False, detail, done=False, reward=None)
        if parsed is None:
            return self._record(action, before, False, "invalid arithmetic action")
        if len(self._pool) == 1:
            return self._record(action, before, False, "final answer is required after three operations")
        left, operator, right, generated = parsed
        left_match = self._find_operand(left)
        right_match = self._find_operand(right, exclude_index=None if left_match is None else left_match[0])
        if left_match is None or right_match is None or left_match[0] == right_match[0]:
            return self._record(action, before, False,
                                "operand occurrence unavailable in the current pool; equal values are legal when two copies remain")
        if operator == "/" and right == 0:
            return self._record(action, before, False, "division by zero")
        if operator == "+":
            computed = left + right
        elif operator == "-":
            computed = left - right
        elif operator == "*":
            computed = left * right
        else:
            computed = left / right
        if computed != generated:
            return self._record(action, before, False, "declared result does not match the operation")
        left_item = self._pool[left_match[0]]
        right_item = self._pool[right_match[0]]
        parent_steps = tuple(sorted(set(left_item[3] + right_item[3])))
        source = f"operation.{self.action_count}"
        for index in sorted((left_match[0], right_match[0]), reverse=True):
            self._pool.pop(index)
        self._pool.append((source, generated, format_number(generated), parent_steps + (self.action_count,)))
        return self._record(
            action,
            before,
            True,
            "numbers updated",
            consumed=(left_item[0], right_item[0]),
            produced=source,
            parents=parent_steps,
        )

