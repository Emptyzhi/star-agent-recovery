"""Use the archived pure prompt functions without importing model/environment code."""

import ast
from pathlib import Path
from typing import Any, Dict, List

from icr_core import ProtocolError


def literal_assignment(path, name):
    module = ast.parse(Path(path).read_text(encoding="utf-8"))
    matches = [node for node in module.body if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
    if len(matches) != 1:
        raise ProtocolError("archived literal is not uniquely defined: " + name)
    value = ast.literal_eval(matches[0].value)
    if type(value) is not str or not value:
        raise ProtocolError("archived prompt must be a nonempty literal string")
    return value


def reflection_functions(source_path, examples_path):
    """Extract the two unmodified pure functions; never execute vendor imports."""
    module = ast.parse(Path(source_path).read_text(encoding="utf-8"))
    names = {"_get_scenario", "_generate_reflection_query"}
    nodes = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names or len(nodes) != len(names):
        raise ProtocolError("archived Reflection functions are missing or duplicated")
    namespace = {"List": List, "Dict": Dict, "Any": Any,
                 "FEW_SHOT_EXAMPLES": Path(examples_path).read_text(encoding="utf-8")}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["_generate_reflection_query"]


def web_memory_base(base, memories):
    if type(memories) is not tuple or any(type(memory) is not str for memory in memories):
        raise ProtocolError("memories must contain the literal generated strings")
    text = base
    if memories:
        text += "\nExternal CGCR public commitment ledger (machine-generated; not reflection):"
        for index, memory in enumerate(memories):
            text += f"\nConstraint entry {index}:\n{memory.strip()}"
    return text


def web_prompt(base, initial, history, memories):
    fixed = web_memory_base(base, memories)
    rolling = initial + "\n\nAction:"
    for row in history:
        rolling += f" {row['action']}\nObservation: {row['observation']}\n\nAction:"
    return fixed + rolling


def web_failed_log(initial, history):
    text = "Here is the task:\n" + initial + "\n"
    for row in history:
        text += f"Action: {row['action']}\nObservation: {row['observation']}\n"
    return text
