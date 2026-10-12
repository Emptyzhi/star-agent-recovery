"""Auditable input-window fitting for model-visible history and reflection text."""

from icr_core import ProtocolError


CONTEXT_WINDOW_TOKENS = 32768
REFLECTION_OUTPUT_TOKENS = 256


def _count(counter, text):
    if not callable(counter):
        raise ProtocolError("context fitting requires the actual tokenizer")
    value = counter(text)
    if type(value) is not int or value < 0:
        raise ProtocolError("context tokenizer returned an invalid count")
    return value


def _history_text(rows, omitted):
    lines = ["[ACTIVE_EXECUTION_HISTORY]\n"]
    if omitted:
        lines.append("[HISTORY_TRUNCATED omitted_records=%d policy=first_and_recent_complete_receipts]\n" % omitted)
    for receipt in rows:
        action, observation = receipt.get("action"), receipt.get("observation")
        if type(action) is not str or type(observation) is not str:
            raise ProtocolError("history action and observation must be actual text")
        lines.append(f"Action: {action}\nObservation: {observation}\n\n")
    lines.append("[ACTIVE_EXECUTION_HISTORY_END]")
    return "".join(lines)


def fit_history_prompt(history, build_prompt, token_counter, max_input_tokens):
    """Fit a prompt by dropping oldest complete receipts, retaining the first receipt."""
    if type(history) is not list or type(max_input_tokens) is not int or max_input_tokens < 1:
        raise ProtocolError("invalid history fitting arguments")
    full = build_prompt(_history_text(history, 0))
    if _count(token_counter, full) <= max_input_tokens:
        return full, {"truncated": False, "omitted_records": 0, "retained_records": len(history)}
    if not history:
        raise ProtocolError("fixed prompt exceeds the input window without history to truncate")
    for recent_count in range(len(history) - 1, -1, -1):
        rows = [] if recent_count == 0 else ([history[0]] if recent_count == 1
                else [history[0], *history[-(recent_count - 1):]])
        omitted = len(history) - len(rows)
        candidate = build_prompt(_history_text(rows, omitted))
        if _count(token_counter, candidate) <= max_input_tokens:
            return candidate, {"truncated": True, "omitted_records": omitted, "retained_records": len(rows)}
    candidate = build_prompt(_history_text([], len(history)))
    if _count(token_counter, candidate) > max_input_tokens:
        raise ProtocolError("fixed prompt exceeds the input window after history truncation")
    return candidate, {"truncated": True, "omitted_records": len(history), "retained_records": 0}


def fit_text_prompt(text, build_prompt, token_counter, max_input_tokens, *, label):
    """Fit free text by retaining both ends and marking omitted middle text."""
    if type(text) is not str or type(max_input_tokens) is not int or max_input_tokens < 1:
        raise ProtocolError(label + " fitting arguments are invalid")
    full = build_prompt(text)
    if _count(token_counter, full) <= max_input_tokens:
        return full, {"truncated": False, "omitted_characters": 0}
    marker = "\n[%s_TRUNCATED middle_text_omitted]\n" % label.upper()
    low, high, best = 0, len(text), None
    while low <= high:
        keep = (low + high) // 2
        if keep == 0:
            candidate_text = marker
        elif keep == len(text):
            candidate_text = text
        else:
            left = (keep + 1) // 2
            candidate_text = text[:left] + marker + text[-(keep - left):]
        candidate = build_prompt(candidate_text)
        if _count(token_counter, candidate) <= max_input_tokens:
            best = candidate, len(text) - keep
            low = keep + 1
        else:
            high = keep - 1
    if best is None:
        raise ProtocolError(label + " cannot fit the declared input window")
    return best[0], {"truncated": best[1] > 0, "omitted_characters": best[1]}
