"""GA-style action-only Game24 environment and task prompts."""
import ast
import copy
import csv
import json
import re
from fractions import Fraction


NUMBER = r"-?(?:\d+(?:\.\d+)?|\d+/\d+)"
OPERATION = re.compile(rf"^\s*({NUMBER})\s*([+\-*/])\s*({NUMBER})\s*=\s*({NUMBER})\s*$")
ANSWER = re.compile(r"^\s*answer\s*:\s*(.*?)\s*=\s*([^=]+)\s*$", re.I)


def parse_number(value):
    value = str(value).strip()
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return Fraction(int(numerator), int(denominator))
    return Fraction(value)


def format_number(value):
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def load_puzzles(path):
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "Puzzles" not in rows[0]:
        raise ValueError("Game24 CSV must contain a Puzzles column")
    return [str(row["Puzzles"]).strip() for row in rows]


def eval_expression(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        value = Fraction(str(node.value))
        return value, [value]
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left, left_inputs = eval_expression(node.left)
        right, right_inputs = eval_expression(node.right)
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
    raise ValueError("unsupported arithmetic expression")


def validate_answer(action, original):
    match = ANSWER.fullmatch(str(action))
    if match is None:
        return False, "final action must start with answer: and contain an expression", None
    expression, declared = match.groups()
    try:
        value, inputs = eval_expression(ast.parse(expression, mode="eval").body)
        declared_value = parse_number(declared)
    except (SyntaxError, ValueError, ZeroDivisionError) as error:
        return False, f"invalid final expression: {error}", None
    if sorted(inputs) != sorted(original):
        return False, "final expression must use every original number exactly once", value
    if value != declared_value or value != 24:
        return False, "final expression does not evaluate to 24", value
    return True, "Success!", value


class Game24Environment:
    """Stateful, exact arithmetic environment with the GA action protocol."""

    max_actions = 12

    def __init__(self, csv_path, task_id):
        self.puzzles = load_puzzles(csv_path)
        if task_id < 0 or task_id >= len(self.puzzles):
            raise ValueError("Game24 task ID is outside the CSV")
        self.task_id = task_id
        self.puzzle = self.puzzles[task_id]
        tokens = self.puzzle.split()
        if len(tokens) != 4 or any(re.fullmatch(NUMBER, token) is None for token in tokens):
            raise ValueError("Game24 puzzle must contain exactly four numeric tokens")
        self.original = tuple(parse_number(token) for token in tokens)
        self.reset()

    def reset(self):
        self.pool = list(self.original)
        self.action_count = 0
        self.done = False
        self.reward = 0
        return self._receipt("reset")

    def snapshot(self):
        return (tuple(format_number(value) for value in self.pool),
                self.action_count, self.done, self.reward)

    def restore(self, snapshot):
        values, self.action_count, self.done, self.reward = snapshot
        self.pool = [parse_number(value) for value in values]

    def _observation(self, detail):
        numbers = " ".join(format_number(value) for value in self.pool)
        return f"{detail}; numbers left: {numbers}"

    def _receipt(self, detail, reward=0, terminal=False):
        return dict(observation=self._observation(detail), reward=reward, terminal=terminal)

    def _operand_index(self, value, exclude=None):
        for index, current in enumerate(self.pool):
            if index != exclude and current == value:
                return index
        return None

    def step(self, action):
        if self.done:
            raise RuntimeError("cannot execute an action after a terminal Game24 state")
        action = str(action).strip()
        self.action_count += 1
        answer = validate_answer(action, self.original) if action.lower().startswith("answer:") else None
        if answer is not None:
            valid, detail, _ = answer
            if valid:
                self.done, self.reward = True, 1
                return self._receipt(detail, reward=1, terminal=True)
            return self._receipt(detail)
        match = OPERATION.fullmatch(action)
        if match is None:
            return self._receipt("invalid arithmetic action")
        left_text, operator, right_text, result_text = match.groups()
        left, right, declared = parse_number(left_text), parse_number(right_text), parse_number(result_text)
        left_index = self._operand_index(left)
        right_index = self._operand_index(right, exclude=left_index)
        if left_index is None or right_index is None:
            return self._receipt("operand occurrence unavailable in current numbers")
        if operator == "+":
            computed = left + right
        elif operator == "-":
            computed = left - right
        elif operator == "*":
            computed = left * right
        else:
            if right == 0:
                return self._receipt("division by zero")
            computed = left / right
        if computed != declared:
            return self._receipt("declared result does not match the operation")
        for index in sorted((left_index, right_index), reverse=True):
            self.pool.pop(index)
        self.pool.append(computed)
        return self._receipt("numbers updated")

    def close(self):
        return None


class Game24Task:
    """ToT task adapter using GA's Act n>/Obs n> action-only framing."""

    def __init__(self, world):
        self.world = world
        self.value_cache = {}

    def get_input(self, _idx):
        return self.world.cache[()]["observation"]

    def actions(self, y):
        return tuple(json.loads(line[len('[RAW_ACTION]'):])
                     if line.startswith('[RAW_ACTION]') else line
                     for line in y.splitlines() if line.strip())

    @staticmethod
    def serialize_action(action):
        # BFS stores one edge per line; JSON preserves empty/multiline rejected
        # proposals as one edge without rewriting their environment action.
        if not action or len(action.splitlines()) != 1 or action.startswith('[RAW_ACTION]'):
            return '[RAW_ACTION]' + json.dumps(action)
        return action

    @staticmethod
    def normalize_action(raw):
        if not isinstance(raw, str):
            raise ValueError('Game24 action output must be text')
        for line in raw.splitlines():
            marker = re.search(r'\bAction\s*:\s*(.*)$', line, flags=re.I)
            candidate = marker.group(1).strip() if marker else line.strip()
            candidate = re.sub(r'^(?:Act|Action)\s*\d*\s*[>:]\s*', '',
                               candidate, count=1, flags=re.I).strip()
            if OPERATION.fullmatch(candidate) or ANSWER.fullmatch(candidate):
                return candidate
        return raw.strip()

    def receipt(self, y):
        return self.world.at(self.actions(y))

    def trajectory(self, actions):
        text = self.world.cache[()]["observation"]
        for index, action in enumerate(actions, 1):
            receipt = self.world.at(actions[:index])
            text += f"\nAct {index}> {action}\nObs {index}> {receipt['observation']}"
        return text

    def cot_prompt_wrap(self, x, y):
        actions = self.actions(y)
        return (self.world.examples + "\n\n" + self.trajectory(actions) +
                f"\nAct {len(actions) + 1}>")

    def propose_prompt_wrap(self, x, y):
        actions = self.actions(y)
        return (self.world.examples + "\n\n" + self.trajectory(actions) +
                "\nPropose alternative next arithmetic actions, one per line. "
                "Do not include numbering, explanations, observations, or a sequence of actions.")

    def value_prompt_wrap(self, x, y):
        from .score_protocol import value_prompt
        return value_prompt('game24', self.trajectory(self.actions(y)))

    def value_outputs_unwrap(self, _x, _y, outputs):
        from .score_protocol import parse_labels
        return parse_labels(outputs)

    def test_output(self, _idx, y):
        return {"r": self.receipt(y)["reward"]}
