"""Execute the archived Table1 functions with an accounted model/env transport."""

import ast
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from icr_core import BudgetExhausted, ProtocolError, environment_call
from context_window import CONTEXT_WINDOW_TOKENS
from icr_identity import NodeAddress, Phase, Role
from icr_webshop import WebShopEpisode


def memory_base(base, memory):
    if memory:
        base += "\nYour memory for the task below:"
        for index, item in enumerate(memory):
            base += f"\nTrial {index}:\n{item.strip()}"
    return base


def delivered_actor_input(base, rolling, memory):
    fixed = memory_base(base, memory)
    return fixed + rolling


def load_native(vendor):
    source = Path(vendor) / "web_reflexion.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    names = {"Local_llm", "EnvironmentHistory", "extract_substring", "_get_scenario",
             "_generate_reflection_query", "_get_base_query", "webshop_run"}
    selected = [node for node in module.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in selected} != names:
        raise ProtocolError("archived Table1 functions are missing")
    scope = {"List": List, "Dict": Dict, "Any": Any, "Tuple": Tuple, "sys": sys,
             "FEW_SHOT_EXAMPLES": (Path(vendor) / "webshop_reflexion_examples.txt").read_text(encoding="utf-8")}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), scope)
    raw_run = scope["webshop_run"]
    local_llm = next(node for node in selected
                     if isinstance(node, ast.ClassDef) and node.name == "Local_llm")
    generate_method = next(node for node in local_llm.body
                           if isinstance(node, ast.FunctionDef) and node.name == "_generate")
    changed_limits = 0
    for node in ast.walk(generate_method):
        if isinstance(node, ast.keyword) and node.arg == "max_new_tokens":
            node.value = ast.Attribute(
                value=ast.Name(id="self", ctx=ast.Load()),
                attr="_output_limit", ctx=ast.Load())
            changed_limits += 1
    if changed_limits != 1:
        raise ProtocolError("native Local_llm must expose one output-limit binding")
    patched_scope = dict(scope)
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[deepcopy(local_llm)], type_ignores=[])), str(source) + ":output_limits", "exec"),
         patched_scope)
    patched_model_class = patched_scope["Local_llm"]
    corrected = deepcopy(next(node for node in selected if node.name == "webshop_run"))
    memory_windows = 0
    for node in ast.walk(corrected):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "EnvironmentHistory" and isinstance(node.args[2], ast.Subscript):
                node.args[2] = ast.Name(id="memory", ctx=ast.Load())
                memory_windows += 1
    if memory_windows != 1:
        raise ProtocolError("expected one archived Reflexion memory window")
    changes = 0
    for node in ast.walk(corrected):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_generate":
            node.args[0] = ast.Call(func=ast.Name(id="delivered_actor_input", ctx=ast.Load()),
                                    args=[ast.Name(id=name, ctx=ast.Load())
                                          for name in ("init_prompt", "prompt", "memory")], keywords=[])
            changes += 1
    if changes != 1:
        raise ProtocolError("memory-delivery variant needs exactly one native Actor input change")
    delivered_scope = dict(scope, delivered_actor_input=delivered_actor_input)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[corrected], type_ignores=[])),
                 str(source) + ":memory_delivery", "exec"), delivered_scope)
    return {"raw_run": raw_run, "delivered_run": delivered_scope["webshop_run"],
            "model_class": patched_model_class, "reflection_query": scope["_generate_reflection_query"]}


class NativeEpisode(WebShopEpisode):
    def __init__(self, website, session_id, task_id, base_prompt):
        super().__init__(website, session_id, task_id, base_prompt)
        self.prompt_token_counter = None
        self.prompt_context_limit = CONTEXT_WINDOW_TOKENS
        self.prompt_service_limit = CONTEXT_WINDOW_TOKENS
        self.last_history_window = {"truncated": False, "omitted_records": 0,
                                    "retained_records": 0}

    def prompt(self, memory):
        if self.initial is None:
            raise ProtocolError("Actor prompt requested before actual reset")
        if self.prompt_token_counter is None:
            return super().prompt(memory)
        def build(rows):
            base = memory_base(self.base_prompt, memory)
            rolling = self.initial["observation"] + "\n\nAction:"
            for row in rows:
                rolling += f" {row['action']}\nObservation: {row['observation']}\n\nAction:"
            return base + rolling
        max_input = min(self.prompt_context_limit - 100, self.prompt_service_limit)
        prompt = build(self.history)
        if self.prompt_token_counter(prompt) <= max_input:
            metadata = {"truncated": False, "omitted_records": 0,
                        "retained_records": len(self.history)}
        else:
            metadata = None
            for recent_count in range(len(self.history) - 1, -1, -1):
                rows = [] if recent_count == 0 else ([self.history[0]] if recent_count == 1
                        else [self.history[0], *self.history[-(recent_count - 1):]])
                candidate = build(rows)
                if self.prompt_token_counter(candidate) <= max_input:
                    prompt = candidate
                    metadata = {"truncated": True,
                                "omitted_records": len(self.history) - len(rows),
                                "retained_records": len(rows)}
                    break
            if metadata is None:
                raise ProtocolError("native Actor prompt exceeds the input window after history truncation")
        self.last_history_window = metadata
        return prompt

    def reset(self):
        # Table1 webshop_run deliberately calls reset twice at episode entry.
        self.initial = self._receipt(self.environment.step("reset"))
        self.history = []
        return deepcopy(self.initial)


class NativeEnvironment:
    def __init__(self, episode, ledger, emit):
        self.episode, self.ledger, self.emit = episode, ledger, emit
        self.receipts = []

    def step(self, index, action):
        if action == "reset":
            receipt = environment_call(self.episode, self.ledger, self.episode.reset)
        else:
            receipt = environment_call(self.episode, self.ledger, self.episode.step, action)
        self.receipts.append(receipt)
        self.emit("native_environment_receipt", receipt=receipt,
                  cumulative_cost=self.ledger.to_dict())
        return receipt["observation"], (receipt["public_terminal_score"] if receipt["done"] else 0.0), receipt["done"]


def model_adapter(native, oracle, ledger, *, task, attempt, role, emit, memories, actor_stop_mode=None):
    model = native["model_class"].__new__(native["model_class"])
    model.tokenizer = SimpleNamespace(pad_token_id=151645, eos_token_id=151645)
    model._output_limit = 256 if role is Role.REFLECTION else 100
    call_index = 0

    def pipeline(prompt, **kwargs):
        nonlocal call_index
        expected = {"temperature": 0.1, "max_new_tokens": model._output_limit,
                    "pad_token_id": 151645,
                    "eos_token_id": 151645, "return_full_text": False, "top_p": 1,
                    "do_sample": True}
        if kwargs != expected:
            raise ProtocolError("native Table1 generation arguments changed")
        phase = Phase.REFLEXION if role is Role.REFLECTION else Phase.TREE
        address = NodeAddress(0, task, phase, attempt, (), call_index)
        generation = oracle.request(address, role, 0, prompt, ledger,
                                    stop_mode=actor_stop_mode if role is Role.ORDINARY else None)
        emit("native_model_input", role=role.name, attempt=attempt, ordinal=call_index,
             requested_memory=list(memories), prompt=prompt,
             cumulative_cost=ledger.to_dict())
        call_index += 1
        return [{"generated_text": generation.text}]

    model.generation_pipe = pipeline
    return model


def reflection(native, oracle, ledger, *, task, attempt, failed_log, memories, emit):
    model = model_adapter(native, oracle, ledger, task=task, attempt=attempt,
                          role=Role.REFLECTION, emit=emit, memories=memories)
    prompt = native["reflection_query"](failed_log, list(memories))
    text = model._generate(prompt, "\n")
    emit("native_reflection_retained", attempt=attempt, text=text)
    return text


def execute_native(native, episode, oracle, ledger, *, task, attempt, memory, delivered, emit, actor_stop_mode=None):
    env = NativeEnvironment(episode, ledger, emit)
    model = model_adapter(native, oracle, ledger, task=task, attempt=attempt,
                          role=Role.ORDINARY, emit=emit, memories=memory if delivered else (), actor_stop_mode=actor_stop_mode)
    run = native["delivered_run"] if delivered else native["raw_run"]
    try:
        history, native_reward = run("fixed_%d" % task, env, episode.base_prompt,
                                     list(memory), model, to_print=False)
        last = env.receipts[-1]
        status = "TERMINAL" if last["done"] else "EPISODE_ACTION_LIMIT"
        score = last["public_terminal_score"] if last["done"] else None
        failed_log = str(history)
    except BudgetExhausted as exc:
        status, score, native_reward = str(exc), None, None
        failed_log = episode.failure_log() if episode.initial is not None else ""
    return {"status": status, "public_terminal_score": score, "success": score == 1.0,
            "native_return_reward": native_reward, "failed_log": failed_log,
            "reset_receipt": episode.initial, "history": list(episode.history),
            "executed_actions": [row["action"] for row in episode.history],
            "result_metadata": episode.result_metadata() if episode.initial is not None else None,
            "rebind_events": [], "rebind_passes": 1, "constraint_memory": [],
            "cumulative_cost": ledger.to_dict()}
