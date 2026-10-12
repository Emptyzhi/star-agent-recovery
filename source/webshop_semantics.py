"""WebShop-specific public state and progress semantics.

The controller keeps the complete ETT. These helpers only classify public
receipts and build compact views for Red and Yellow; they never infer hidden
product facts or rewrite native actions.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Iterable


_PRICE = re.compile(r"(?im)^\s*price\s*:\s*\$\s*[0-9]+(?:\.[0-9]+)?\s*$")
_RATING = re.compile(r"(?im)^\s*rating\s*:\s*.+$")
_LOCAL_INFO_LABELS = frozenset({'Description', 'Features', 'Reviews', 'Attributes'})


def _bounded_values(values: Any, limit: int = 12) -> list[Any]:
    values = list(values or [])
    if len(values) <= limit:
        return values
    left = limit // 2
    right = limit - left
    return values[:left] + ["[VALUES_OMITTED count=%d]" % (len(values) - limit)] + values[-right:]


def _bounded_option_domains(domains: dict[str, Any], limit: int = 8) -> dict[str, dict[str, Any]]:
    """Project WebShop's label->category map without treating labels as lists.

    The native environment stores entries such as ``"35w x 36l": "size"``.
    Keeping the category as the outer key makes truncation explicit and avoids
    expanding every label into characters when the map is large.
    """
    grouped: dict[str, list[str]] = {}
    for label, category in (domains or {}).items():
        grouped.setdefault(str(category), []).append(str(label))
    result: dict[str, dict[str, Any]] = {}
    for category, labels in grouped.items():
        labels = sorted(set(labels))
        result[category] = {
            "count": len(labels),
            "labels": _bounded_values(labels, limit),
        }
    return result


def _product_title(observation: str, option_domains: dict[str, Any], page_type: str) -> str | None:
    """Extract only the rendered product title when the public page exposes it."""
    if page_type != "item":
        return None
    option_labels = {label for category in option_domains.values()
                     for label in category.get("labels", [])
                     if isinstance(label, str) and not label.startswith("[VALUES_OMITTED")}
    for line in observation.splitlines():
        text = line.strip()
        if text and not text.lower().startswith(("instruction:", "price:", "rating:", "[", "you have clicked")):
            if (text not in {"WebShop", "Description", "Features", "Reviews", "Attributes", "Invalid action!"}
                    and not any(label in text for label in option_labels)):
                return text
    return None


def public_semantic_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    """Return exact public fields useful for WebShop recovery decisions."""

    observation = str(state.get("observation", ""))
    option_domains = _bounded_option_domains(state.get("option_types", {}))
    return {
        "page_type": state.get("page_type"),
        "query": state.get("query", ""),
        "page_num": state.get("page_num", 1),
        "asin": state.get("asin", ""),
        "subpage": state.get("subpage", ""),
        "selected_options": deepcopy(state.get("selected_options", {})),
        "option_domains": option_domains,
        "visible_asins": _bounded_values(state.get("visible_asins", []), 10),
        "available_actions": list(state.get("available_actions", [])),
        "price_text": next((line.strip() for line in observation.splitlines()
                             if _PRICE.fullmatch(line)), None),
        "rating_text": next((line.strip() for line in observation.splitlines()
                              if _RATING.fullmatch(line)), None),
        "product_title": _product_title(observation, option_domains, state.get("page_type", "")),
        "visible_observation": observation,
    }


def yellow_local_commands(state: dict[str, Any]) -> tuple[str, ...]:
    """Return only current-item validation actions available to Yellow.

    Product selection, search, pagination, return-to-search and terminal
    submission remain Green decisions.  A variant option or an information
    subpage is a local observation/repair operation on the already-open item.
    """
    if state.get('page_type') != 'item':
        return ()
    option_labels = set((state.get('option_types') or {}).keys())
    allowed_labels = option_labels | _LOCAL_INFO_LABELS
    result = []
    for action in state.get('available_actions', ()):
        match = re.fullmatch(r'click\[(.*)\]', str(action))
        if match is not None and match.group(1) in allowed_labels:
            result.append(action)
    return tuple(result)


def semantic_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    """Expose the existing public state plus a typed WebShop projection."""

    result = deepcopy(state)
    result["webshop_semantics"] = public_semantic_snapshot(state)
    return result


def checkpoint_snapshot(state: dict[str, Any], observation_limit: int = 700) -> dict[str, Any]:
    """Compact public page state used in WebShop recovery prompts."""
    projection = public_semantic_snapshot(state)
    observation = projection["visible_observation"]
    if len(observation) > observation_limit:
        left = observation_limit // 2
        right = observation_limit - left
        observation = (observation[:left]
                       + "\n[MIDDLE_PAGE_TEXT_OMITTED]\n"
                       + observation[-right:])
    return {
        "page_type": projection["page_type"],
        "query": projection["query"],
        "page_num": projection["page_num"],
        "asin": projection["asin"],
        "subpage": projection["subpage"],
        "selected_options": projection["selected_options"],
        "option_domains": projection["option_domains"],
        "visible_asins": projection["visible_asins"],
        "price_text": projection["price_text"],
        "rating_text": projection["rating_text"],
        "product_title": projection["product_title"],
        "available_action_count": len(projection["available_actions"]),
        "visible_observation": observation,
    }


def progress_kind(before: dict[str, Any], after: dict[str, Any], action: str,
                  *, accepted: bool, done: bool) -> str:
    """Classify the public change caused by one native WebShop action."""

    del action
    if done:
        return "TERMINAL_SCORE_CHANGED"
    if not accepted:
        return "NO_PUBLIC_CHANGE"
    if before.get("query") != after.get("query") or before.get("page_num") != after.get("page_num"):
        return "QUERY_CHANGED"
    if before.get("asin") != after.get("asin"):
        return "PRODUCT_CHANGED"
    if before.get("selected_options") != after.get("selected_options"):
        return "OPTION_CHANGED"
    if before.get("subpage") != after.get("subpage"):
        return "SUBPAGE_VIEWED"
    if before.get("available_actions") != after.get("available_actions"):
        return "MENU_CHANGED"
    if before.get("observation") != after.get("observation"):
        return "TEXT_CHANGED"
    return "NO_PUBLIC_CHANGE"


def checkpoint_kind(occurrence: Any) -> str:
    """Classify the state immediately before one executed action."""

    receipt = occurrence.receipt
    action = occurrence.action
    if not receipt.get("native_accepted", False):
        return "REJECTED_EDGE"
    if action.startswith("think["):
        return "THOUGHT"
    if action.startswith("search["):
        return "SEARCH_SUBMIT"
    before = receipt.get("before") or {}
    after = receipt.get("after") or {}
    kind = progress_kind(before, after, action,
                         accepted=True, done=bool(receipt.get("done")))
    if kind == "PRODUCT_CHANGED":
        return "PRODUCT_OPEN"
    if kind == "OPTION_CHANGED":
        return "OPTION_CHANGE"
    if kind == "QUERY_CHANGED":
        return "QUERY_NAVIGATION"
    if action in {"click[Back to Search]", "click[< Prev]"}:
        return "NAVIGATION_BOUNDARY"
    if action == "click[Buy Now]":
        return "TERMINAL_SUBMIT"
    if kind == "SUBPAGE_VIEWED":
        return "INFORMATION_VIEW"
    if (before.get("page_type") == "item_sub"
            or after.get("page_type") == "item_sub"):
        return "INFORMATION_VIEW"
    return "NO_PUBLIC_CHANGE"


def key_checkpoint_indices(path: Iterable[Any]) -> tuple[int, ...]:
    """Return WebShop recovery candidates without deleting ETT nodes."""

    occurrences = list(path)
    selected: list[int] = []
    seen_searches: set[str] = set()
    seen_products: set[str] = set()
    for index, occurrence in enumerate(occurrences):
        kind = checkpoint_kind(occurrence)
        if kind == "SEARCH_SUBMIT":
            query = str(occurrence.action[7:-1])
            if query not in seen_searches:
                selected.append(index)
                seen_searches.add(query)
        elif kind == "PRODUCT_OPEN":
            asin = str(occurrence.receipt.get("after", {}).get("asin", ""))
            if asin not in seen_products:
                selected.append(index)
                seen_products.add(asin)
        elif kind in {"OPTION_CHANGE", "QUERY_NAVIGATION", "NAVIGATION_BOUNDARY",
                      "MENU_CHANGED", "TEXT_CHANGED",
                      "TERMINAL_SUBMIT", "REJECTED_EDGE"}:
            selected.append(index)
    if not selected and occurrences:
        selected.append(0)
    return tuple(dict.fromkeys(selected))


def checkpoint_view(path: Iterable[Any], candidates: Iterable[int]) -> str:
    """Build compact candidate evidence for Red without hiding raw ETT."""

    rows = []
    occurrences = list(path)
    for index in candidates:
        occurrence = occurrences[index]
        receipt = occurrence.receipt
        before = receipt.get("before") or {}
        after = receipt.get("after") or {}
        before_view = checkpoint_snapshot(before, observation_limit=160)
        after_view = checkpoint_snapshot(after, observation_limit=120)
        rows.append({
            "action_id": f"ACTION {index}",
            "checkpoint_kind": checkpoint_kind(occurrence),
            "action": occurrence.action,
            "before": before_view,
            "after_facts": {
                "page_type": after_view["page_type"],
                "asin": after_view["asin"],
                "selected_options": after_view["selected_options"],
                "price_text": after_view["price_text"],
                "option_domains": after_view["option_domains"],
            },
        })
    return "[WEBSHOP_KEY_CHECKPOINTS]\n" + repr(rows)


_FAULT_CUES = re.compile(
    r"\b(?:mismatch(?:es|ed)?|wrong|incorrect|failed|unwanted|unrequested|"
    r"does not match|doesn't match|not the requested|should not|instead of)\b",
    re.IGNORECASE,
)


def _mentioned_as_fault(text: str, value: Any) -> bool:
    """Require a public value to occur near an explicit negative cue."""

    value = str(value).strip()
    if len(value) < 2 or value.startswith("$"):
        return False
    for match in re.finditer(re.escape(value), text, re.IGNORECASE):
        left = max(0, match.start() - 96)
        right = min(len(text), match.end() + 96)
        if _FAULT_CUES.search(text[left:right]):
            return True
    return False


def public_fault_action_index(path: Iterable[Any], reflection: str) -> dict[str, Any] | None:
    """Map explicit public mismatch values to their first introducing action.

    This is a controller mapping, not a guessed checkpoint: it only accepts a
    value when the Red reflection names that exact public value next to an
    explicit negative cue, and the receipt shows that the action introduced
    the value.  Ambiguous or unsupported evidence remains available to the
    existing model selector instead of being silently resolved.
    """

    if not isinstance(reflection, str) or not reflection.strip():
        return None
    occurrences = list(path)
    matches: list[dict[str, Any]] = []
    for index, occurrence in enumerate(occurrences):
        receipt = occurrence.receipt
        if not receipt.get("native_accepted") or receipt.get("done"):
            continue
        before = receipt.get("before") or {}
        after = receipt.get("after") or {}
        kind = checkpoint_kind(occurrence)
        values: list[tuple[str, str]] = []
        if kind == "OPTION_CHANGE":
            before_options = before.get("selected_options") or {}
            for category, value in (after.get("selected_options") or {}).items():
                if before_options.get(category) != value:
                    values.append((f"option:{category}", value))
        elif kind == "PRODUCT_OPEN" and before.get("asin") != after.get("asin"):
            values.extend([
                ("asin", after.get("asin", "")),
                ("product_title", public_semantic_snapshot(after).get("product_title", "")),
            ])
        elif kind in {"SEARCH_SUBMIT", "QUERY_NAVIGATION"}:
            query = after.get("query", "")
            if query and query != before.get("query", ""):
                values.append(("query", query))
        for field, value in values:
            if _mentioned_as_fault(reflection, value):
                matches.append({"action_index": index, "field": field,
                                "value": str(value), "action": occurrence.action,
                                "checkpoint_kind": kind})
    if not matches:
        return None
    first = min(matches, key=lambda row: row["action_index"])
    return {"action_index": first["action_index"], "evidence": first,
            "candidates": matches, "source": "PUBLIC_MISMATCH_VALUE_TO_FIRST_TRANSITION"}
