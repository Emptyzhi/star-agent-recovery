"""Auditable execution-history tree and deterministic Actor-facing views."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from typing import Any

from icr_core import ProtocolError


TREE_BREADCRUMB_HIERARCHICAL = "TREE_BREADCRUMB_HIERARCHICAL"
ABLATE_NO_BREADCRUMB = "ABLATE_NO_BREADCRUMB"
ABLATE_NO_HIERARCHICAL_COMPRESSION = "ABLATE_NO_HIERARCHICAL_COMPRESSION"
TREE_ARMS = (
    TREE_BREADCRUMB_HIERARCHICAL,
    ABLATE_NO_BREADCRUMB,
    ABLATE_NO_HIERARCHICAL_COMPRESSION,
)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def columnar_graph(graph):
    """Lossless JSON node table: shared column names, explicit absent cells."""
    nodes = graph['nodes']
    columns = list(dict.fromkeys(key for node in nodes for key in node))
    rows, absent = [], {}
    for i, node in enumerate(nodes):
        rows.append([node[key] if key in node else None for key in columns])
        missing = [j for j, key in enumerate(columns) if key not in node]
        if missing:
            absent[str(i)] = missing
    return {**{k:v for k,v in graph.items() if k != 'nodes'},
            'schema':'execution_graph_table_v1', 'node_columns':columns,
            'node_rows':rows, 'node_absent':absent}


def _short_observation(value: Any) -> str:
    if not isinstance(value, str):
        raise ProtocolError("tree receipt observation must be text")
    return " ".join(value.split())


def _state_summary(snapshot: dict[str, Any]) -> str:
    if not isinstance(snapshot, dict):
        raise ProtocolError("tree state must be an object")
    if snapshot.get("benchmark") == "alfworld":
        return canonical(snapshot)
    required = ("page_type", "asin", "selected_options", "option_types")
    if any(key not in snapshot for key in required):
        raise ProtocolError("tree state lacks an explicit public field")
    page = snapshot["page_type"]
    asin = snapshot["asin"]
    selected = snapshot["selected_options"]
    option_types = snapshot["option_types"]
    if (type(page) is not str or type(asin) is not str
            or type(selected) is not dict or type(option_types) is not dict):
        raise ProtocolError("tree state public fields have invalid types")
    fields = {"page": page, "asin": asin, "selected": selected}
    if page == "item":
        field_names = sorted(set(option_types.values()))
        if any(type(name) is not str or not name for name in field_names):
            raise ProtocolError("tree option field names are not explicit")
        fields["fields"] = {name: selected[name] if name in selected else "UNSET"
                             for name in field_names}
    return canonical(fields)


def _receipt_fingerprint(receipt: dict[str, Any]) -> str:
    required = ("action", "observation", "before", "after", "done",
                "public_terminal_score", "invalid_action", "invalid_detail",
                "native_accepted")
    if any(key not in receipt for key in required):
        raise ProtocolError("tree receipt lacks its public evidence fields")
    return canonical({key: receipt[key] for key in required})


@dataclass
class TreeNode:
    node_id: int
    parent_id: int | None
    child_index: int
    action: str | None
    before: dict[str, Any] | None
    after: dict[str, Any]
    receipt: dict[str, Any] | None
    depth: int
    children: list[int] = field(default_factory=list)
    visits: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_root(self) -> bool:
        return self.parent_id is None

    @property
    def is_rejected(self) -> bool:
        return bool(self.receipt and self.receipt.get("invalid_action"))

    @property
    def is_terminal(self) -> bool:
        return bool(self.receipt and self.receipt.get("done"))

    @property
    def is_terminal_non_success(self) -> bool:
        return self.is_terminal and self.receipt.get("public_terminal_score") != 1.0

    @property
    def is_failure(self) -> bool:
        return self.is_rejected or self.is_terminal_non_success

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "child_index": self.child_index,
            "action": self.action,
            "before": deepcopy(self.before),
            "after": deepcopy(self.after),
            "receipt": deepcopy(self.receipt),
            "depth": self.depth,
            "children": list(self.children),
            "visits": deepcopy(self.visits),
        }


class ExecutionHistoryTree:
    """Full evidence tree plus three deterministic prompt representations.

    Nodes are shared only when the same parent has the same action and exact
    public receipt. Visits retain attempt/request identity and are never merged.
    """

    def __init__(self, root_state: dict[str, Any]):
        if not isinstance(root_state, dict):
            raise ProtocolError("tree root state must be an object")
        self.nodes: list[TreeNode] = [TreeNode(
            node_id=0, parent_id=None, child_index=0, action=None,
            before=None, after=deepcopy(root_state), receipt=None, depth=0)]
        self.current_id = 0
        self.tree_version = 0
        self.visit_serial = 0
        self.controller_events: list[dict[str, Any]] = []

    def node(self, node_id: int) -> TreeNode:
        if type(node_id) is not int or not 0 <= node_id < len(self.nodes):
            raise ProtocolError("tree node ID is outside the owned tree")
        return self.nodes[node_id]

    def reset_cursor(self) -> None:
        self.current_id = 0

    def _check_parent_state(self, parent: TreeNode, receipt: dict[str, Any]) -> None:
        before = receipt.get("before")
        if not isinstance(before, dict) or before != parent.after:
            raise ProtocolError("tree edge before-state differs from its parent state")

    def add_edge(self, parent_id: int, action: str, receipt: dict[str, Any], *,
                 visit_kind: str, attempt: int | str, request_id: str | None = None) -> int:
        parent = self.node(parent_id)
        if parent.is_root and action is None:
            raise ProtocolError("root cannot carry an action")
        if type(action) is not str or not action:
            raise ProtocolError("tree action must be nonempty text")
        if not isinstance(receipt, dict):
            raise ProtocolError("tree edge requires the actual receipt object")
        after = receipt.get("after")
        if not isinstance(after, dict):
            raise ProtocolError("tree receipt lacks its public after-state")
        self._check_parent_state(parent, receipt)
        fingerprint = _receipt_fingerprint(receipt)
        child = None
        for child_id in parent.children:
            candidate = self.node(child_id)
            if candidate.action == action and _receipt_fingerprint(candidate.receipt) == fingerprint:
                child = candidate
                break
        if child is None:
            child = TreeNode(
                node_id=len(self.nodes), parent_id=parent_id,
                child_index=len(parent.children), action=action,
                before=deepcopy(receipt["before"]), after=deepcopy(after),
                receipt=deepcopy(receipt), depth=parent.depth + 1)
            self.nodes.append(child)
            parent.children.append(child.node_id)
        self.visit_serial += 1
        child.visits.append({
            "visit_id": self.visit_serial,
            "visit_kind": visit_kind,
            "attempt": attempt,
            "request_id": request_id,
        })
        self.tree_version += 1
        self.current_id = child.node_id
        return child.node_id

    def begin_attempt(self, attempt: int | str, *, restored: bool) -> None:
        self.current_id = 0
        self.visit_serial += 1
        self.nodes[0].visits.append({
            "visit_id": self.visit_serial,
            "visit_kind": "ROOT_RESTORE" if restored else "ROOT_START",
            "attempt": attempt,
            "request_id": None,
        })
        self.tree_version += 1

    def note_action_error(self, parent_id: int, action: str, *, attempt: int | str,
                          request_id: str | None, error_type: str, error: str) -> None:
        if self.current_id != parent_id:
            raise ProtocolError("controller error parent differs from tree cursor")
        if any(type(value) is not str or not value for value in
               (action, error_type, error)):
            raise ProtocolError("controller action error evidence is incomplete")
        self.controller_events.append({
            "event": "ACTION_EXECUTION_ERROR",
            "parent_node_id": parent_id,
            "action": action,
            "attempt": attempt,
            "request_id": request_id,
            "error_type": error_type,
            "error": error,
        })
        self.tree_version += 1

    def seed_source(self, history: list[dict[str, Any]]) -> None:
        if not isinstance(history, list):
            raise ProtocolError("source history must be a list")
        parent_id = 0
        for ordinal, receipt in enumerate(history):
            action = receipt.get("action") if isinstance(receipt, dict) else None
            parent_id = self.add_edge(parent_id, action, receipt,
                                      visit_kind="SOURCE_EVIDENCE",
                                      attempt="source", request_id=None)
            self.current_id = parent_id
        self.reset_cursor()

    def ancestors(self, node_id: int) -> list[int]:
        result = []
        cursor = self.node(node_id)
        while True:
            result.append(cursor.node_id)
            if cursor.parent_id is None:
                break
            cursor = self.node(cursor.parent_id)
        result.reverse()
        return result

    def distance(self, left_id: int, right_id: int) -> int:
        left = self.ancestors(left_id)
        right = self.ancestors(right_id)
        common = 0
        for left_item, right_item in zip(left, right):
            if left_item != right_item:
                break
            common += 1
        return (len(left) - common) + (len(right) - common)

    def _subtree(self, node_id: int) -> list[TreeNode]:
        result = []
        stack = [node_id]
        while stack:
            current = self.node(stack.pop())
            result.append(current)
            stack.extend(reversed(current.children))
        return result

    def _stats(self, node_id: int) -> dict[str, Any]:
        nodes = self._subtree(node_id)
        terminals = [node for node in nodes if node.is_terminal]
        scores = [node.receipt["public_terminal_score"] for node in terminals
                  if isinstance(node.receipt.get("public_terminal_score"), (int, float))]
        return {
            "actions": max(0, len(nodes) - 1),
            "branches": len(self.node(node_id).children),
            "evaluated": len(terminals),
            "success": sum(node.receipt.get("public_terminal_score") == 1.0 for node in terminals),
            "non_success": sum(node.receipt.get("public_terminal_score") != 1.0 for node in terminals),
            "rejected": sum(node.is_rejected for node in nodes),
            "unknown": sum(not node.is_terminal and not node.children for node in nodes[1:]),
            "best_public_score": max(scores) if scores else None,
        }

    def _path_text(self, node_id: int) -> str:
        labels = ["Root"]
        for ancestor_id in self.ancestors(node_id)[1:]:
            action = self.node(ancestor_id).action
            labels.append(action if action is not None else "<missing-action>")
        return " -> ".join(labels)

    def _relative(self, node: TreeNode) -> str:
        if node.parent_id is None:
            return "parent=NONE sibling=0/1 depth=0"
        parent = self.node(node.parent_id)
        return (f"parent=N{node.parent_id} sibling={node.child_index + 1}/{len(parent.children)} "
                f"depth={node.depth}")

    def _failure_detail(self, node: TreeNode) -> str:
        receipt = node.receipt
        if receipt is None:
            raise ProtocolError("failure detail requested for the root")
        if node.is_rejected:
            status = "rejected"
        elif node.is_terminal:
            status = "terminal score=" + str(receipt.get("public_terminal_score"))
        else:
            raise ProtocolError("unknown failure type")
        observation = _short_observation(receipt.get("observation"))
        after = _state_summary(receipt["after"])
        return (f"{receipt['action']} -> [Receipt: {status}, Obs: {observation}, "
                f"After: {after}]")

    def _normal_detail(self, node: TreeNode) -> str:
        if node.receipt is None:
            return "Root: task start; no action"
        return f"{node.action} -> State: {_state_summary(node.after)}"

    def _node_line(self, node: TreeNode, *, breadcrumb: bool, current: bool,
                   full_receipt: bool, indent: str) -> str:
        marker = " [YOU ARE HERE]" if current else ""
        if node.is_root:
            return indent + "Root [TASK_START; NO_ACTION]" + marker
        if breadcrumb:
            location = (f"N{node.node_id} visits={len(node.visits)} "
                        f"Path: {self._path_text(node.node_id)}")
        else:
            location = (f"N{node.node_id} visits={len(node.visits)} "
                        f"{_self_relative(node, self)}")
        detail = (self._failure_detail(node)
                  if full_receipt and node.is_failure else self._normal_detail(node))
        return indent + location + marker + " -> " + detail

    def _failure_descendants(self, node_id: int) -> list[TreeNode]:
        return [node for node in self._subtree(node_id)
                if node.node_id != node_id and node.is_failure]

    def _render_group(self, parent: TreeNode, group: list[TreeNode], *,
                      breadcrumb: bool, indent: str) -> list[str]:
        group_id = "FG-" + str(parent.node_id) + "-" + str(group[0].child_index)
        lines = [indent + f"[FAIL_GROUP {group_id} parent=N{parent.node_id} "
                 f"members={[node.node_id for node in group]} observed_outcomes_only]"]
        for node in group:
            outcomes = [{"node": child.node_id,
                         "receipt": "terminal",
                         "score": child.receipt["public_terminal_score"],
                         "observation": child.receipt["observation"],
                         "submitted_state": json.loads(_state_summary(child.before))}
                        for child in self._subtree(node.node_id) if child.is_terminal]
            lines.append(indent + f"  N{node.node_id} {node.action} visits={len(node.visits)} "
                         + "outcomes=" + canonical(outcomes))
            for child in self._subtree(node.node_id):
                if child.is_rejected:
                    lines.append(self._node_line(child, breadcrumb=breadcrumb, current=False,
                                                 full_receipt=True, indent=indent + "    "))
        return lines

    def _failure_group_key(self, node: TreeNode):
        if node.is_terminal_non_success:
            return ("terminal",)
        if node.receipt is None or node.is_rejected or node.before.get("benchmark") == "alfworld" or node.before["page_type"] != "item":
            return None
        action = node.action
        if not action.startswith("click[") or not action.endswith("]"):
            return None
        label = action[6:-1]
        option_types = node.before["option_types"]
        if label not in option_types:
            return None
        descendants = self._subtree(node.node_id)
        terminals = [child for child in descendants if child.is_terminal]
        if (not terminals or any(not child.is_terminal_non_success for child in terminals)
                or any(not child.is_terminal and not child.children for child in descendants)):
            return None
        return ("option", option_types[label])

    def _render_children(self, parent: TreeNode, *, current_path: set[int],
                         breadcrumb: bool, compress: bool, indent: str) -> list[str]:
        lines: list[str] = []
        children = [self.node(child_id) for child_id in parent.children]
        grouped_ids = set()
        if compress:
            groups = {}
            for node in children:
                if node.node_id in current_path or self.distance(node.node_id, self.current_id) <= 2:
                    continue
                key = self._failure_group_key(node)
                if key is not None:
                    groups.setdefault(key, []).append(node)
            for group in groups.values():
                if len(group) >= 2:
                    lines.extend(self._render_group(parent, group, breadcrumb=breadcrumb, indent=indent))
                    grouped_ids.update(node.node_id for node in group)
        for node in children:
            if node.node_id in grouped_ids:
                continue
            distance = self.distance(node.node_id, self.current_id)
            far = compress and node.node_id not in current_path and distance > 2
            if far and not node.is_failure:
                stats = self._stats(node.node_id)
                failures = self._failure_descendants(node.node_id)
                lines.append(indent + f"N{node.node_id} [COLLAPSED] "
                             f"entry={node.action} "
                             f"actions={stats['actions']} branches={stats['branches']} "
                             f"evaluated={stats['evaluated']} success={stats['success']} "
                             f"non_success={stats['non_success']} unknown={stats['unknown']} "
                             f"best_public_score={stats['best_public_score']} "
                             f"visits={len(node.visits)} "
                             f"failure_evidence={[item.node_id for item in failures]}")
                for failure in failures:
                    lines.append(self._node_line(
                        failure, breadcrumb=breadcrumb, current=False,
                        full_receipt=True, indent=indent + "  ")
                        + " [FAR_FAILURE_EVIDENCE]")
                continue
            full = (not compress) or node.is_failure
            lines.append(self._node_line(node, breadcrumb=breadcrumb,
                                         current=node.node_id == self.current_id,
                                         full_receipt=full, indent=indent))
            lines.extend(self._render_children(node, current_path=current_path,
                                               breadcrumb=breadcrumb, compress=compress,
                                               indent=indent + "  "))
        return lines

    def render_compact(self) -> str:
        """One breadcrumb; full immediate children; distant outcomes as facts."""
        current = self.node(self.current_id)
        path = set(self.ancestors(self.current_id)) | {self.current_id}
        immediate = set(current.children)
        lines = [f"[TREE_VIEW version={self.tree_version}]",
                 f"[CURRENT_NODE N{self.current_id}] [YOU ARE HERE]",
                 "[BREADCRUMB] " + self._path_text(self.current_id),
                 "[CURRENT_STATE] " + _state_summary(current.after),
                 "[TREE] Root [TASK_START; NO_ACTION]"]

        def visit(node_id, indent):
            node = self.node(node_id)
            label = f"N{node_id} parent=N{node.parent_id} visits={len(node.visits)} action={node.action}"
            if node_id in immediate:
                lines.append(indent + label + " full_receipt=" + canonical(node.receipt))
            elif node_id in path:
                lines.append(indent + label + " state=" + _state_summary(node.after))
            else:
                outcomes = [{"node": item.node_id, "action": item.action,
                             "score": item.receipt["public_terminal_score"],
                             "invalid": item.receipt["invalid_action"],
                             "state": json.loads(_state_summary(item.after))}
                            for item in self._subtree(node_id) if item.is_failure or item.is_terminal]
                lines.append(indent + label + " subtree=" + canonical(self._stats(node_id))
                             + " observed_outcomes=" + canonical(outcomes))
                return
            for child in node.children:
                visit(child, indent + "  ")

        for child in self.node(0).children:
            visit(child, "  ")
        lines.append("[TREE_END]")
        return "\n".join(lines)

    def render_graph(self, *, scope_id: int, scope_kind: str, compact: bool = False, columnar=False) -> str:
        """Serialize parent links once; retain every node without recursive layout."""
        def node_ref(node_id):
            return f"s{scope_id}:n{node_id}"

        immediate = set(self.node(self.current_id).children)
        rows = []
        for node in self.nodes:
            visits = sorted({visit["visit_kind"] for visit in node.visits})
            kinds = set()
            for visit in node.visits:
                if visit["visit_kind"] == "YELLOW_MERGED":
                    kinds.add("YELLOW")
                elif visit["visit_kind"] == "SCOPED_EXECUTION":
                    kinds.add(scope_kind)
            row = {"id": node_ref(node.node_id),
                   "parent_id": None if node.parent_id is None else node_ref(node.parent_id),
                   "action": node.action, "execution_kinds": sorted(kinds),
                   "visit_count": len(node.visits), "visit_kinds": visits,
                   "state": json.loads(_state_summary(node.after)),
                   "terminal": node.is_terminal, "rejected": node.is_rejected,
                   "terminal_score": node.receipt["public_terminal_score"] if node.is_terminal else None}
            if node.receipt is not None and (node.node_id in immediate or (node.is_failure and not compact)):
                row["receipt"] = deepcopy(node.receipt)
            rows.append(row)
        graph = {"schema": "execution_graph_v1", "scope_id": scope_id, "scope_kind": scope_kind,
                 "root_id": node_ref(0), "current_node": node_ref(self.current_id),
                 "current_marker": "[YOU ARE HERE]",
                 "current_path": [node_ref(i) for i in self.ancestors(self.current_id)],
                 "nodes": rows, "controller_events": deepcopy(self.controller_events)}
        if columnar:
            graph = columnar_graph(graph)
        encoded = json.dumps(graph, ensure_ascii=not columnar, sort_keys=True, separators=(',', ':'), allow_nan=False)
        return "[EXECUTION_GRAPH]\n" + encoded + "\n[EXECUTION_GRAPH_END]"

    def render(self, *, variant: str) -> str:
        if variant not in TREE_ARMS:
            raise ProtocolError("unknown tree view variant")
        breadcrumb = variant != ABLATE_NO_BREADCRUMB
        compress = variant != ABLATE_NO_HIERARCHICAL_COMPRESSION
        current = self.node(self.current_id)
        current_path = set(self.ancestors(self.current_id))
        parent = self.node(current.parent_id) if current.parent_id is not None else None
        lines = [
            f"[TREE_VIEW version={self.tree_version}]",
            f"[CURRENT_NODE N{current.node_id} visits={len(current.visits)} "
            f"depth={current.depth} tree_nodes={len(self.nodes)}]",
        ]
        if breadcrumb:
            lines.append("[BREADCRUMB] " + self._path_text(self.current_id))
        lines.append("[RELATIVE_POSITION] " + self._relative(current))
        lines.append("[CURRENT_STATE] " + _state_summary(current.after))
        root_marker = " [YOU ARE HERE]" if self.current_id == 0 else ""
        lines.append("[TREE] Root [TASK_START; NO_ACTION]" + root_marker)
        lines.extend(self._render_children(self.node(0), current_path=current_path,
                                           breadcrumb=breadcrumb, compress=compress,
                                           indent="  "))
        if self.controller_events:
            lines.append("[CONTROLLER_EVENTS]")
            lines.extend("  " + canonical(event) for event in self.controller_events)
        lines.append("[TREE_END]")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tree_version": self.tree_version,
            "current_id": self.current_id,
            "visit_serial": self.visit_serial,
            "controller_events": deepcopy(self.controller_events),
            "nodes": [node.to_dict() for node in self.nodes],
        }


def _self_relative(node: TreeNode, tree: ExecutionHistoryTree) -> str:
    return tree._relative(node)
