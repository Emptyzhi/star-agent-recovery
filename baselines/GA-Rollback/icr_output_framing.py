"""Declared output-message boundaries; no content reconstruction or token caps."""

import json


STOP_MODES = ("first_line", "first_nonempty_line", "lines_2", "lines_5", "json_object")


def frame_complete(text, mode):
    if mode is None:
        return False
    if mode == "first_line":
        return "\n" in text
    if mode == "first_nonempty_line":
        # Leading blank lines carry no diagnosis. Stop only after a complete
        # nonempty line; preserve all sampled text for the caller and audit.
        return any(line.strip() for line in text.split("\n")[:-1])
    if mode in {"lines_2", "lines_5"}:
        count = 2 if mode == "lines_2" else 5
        # Scope line protocols discard leading whitespace before parsing.
        parts = text.lstrip().split("\n", count)
        return len(parts) == count + 1 and not parts[-1].strip()
    if mode == "json_object":
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return False
        return type(value) is dict
    raise ValueError("unknown output framing mode")


def stop_reason(text, mode, *, eos, now, submitted_at, budget_seconds):
    if budget_seconds is not None and now >= submitted_at + budget_seconds:
        return "time"
    if eos:
        return "eos"
    if frame_complete(text, mode):
        return "frame"
    return None
