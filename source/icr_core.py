"""Shared proposal tree and real-execution contracts, independent of benchmark goals."""

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import time
from typing import Any, Callable, Protocol

from icr_design import DEPTH, realized_pairs, validate_bits
from icr_identity import (NodeAddress, Phase, Role, RequestAddressV2,
                          completion_request_seed, request_seed)


ORDINARY_OUTPUT_TOKEN_LIMIT = 100
REFLECTION_OUTPUT_TOKEN_LIMIT = 256
TOKEN_LIMIT = REFLECTION_OUTPUT_TOKEN_LIMIT
OPEN_PROPOSAL_LIMIT = 4


class ProtocolError(RuntimeError):
    pass


class BudgetExhausted(RuntimeError):
    pass


def nonnegative(value, name):
    if type(value) is not int or value < 0:
        raise ProtocolError(name + " must be a nonnegative integer")
    return value


def public_copy(value):
    """Persist only explicit JSON-compatible evidence; NaN is not a public state."""
    return json.loads(json.dumps(value, ensure_ascii=True, allow_nan=False))


class SupportKind(str, Enum):
    FINITE = "FINITE"
    OPEN = "OPEN"


@dataclass(frozen=True)
class ParsedAction:
    action: str
    grammar_valid: bool
    is_think: bool
    evidence: dict[str, Any]


@dataclass(frozen=True)
class ActionSupport:
    kind: SupportKind
    commands: tuple[str, ...]
    contract_id: str

    def __post_init__(self):
        if not isinstance(self.kind, SupportKind) or not self.contract_id:
            raise ProtocolError("action support requires a declared public contract")
        if type(self.commands) is not tuple or any(type(action) is not str for action in self.commands):
            raise ProtocolError("finite commands must be a tuple of complete strings")
        if len(set(self.commands)) != len(self.commands):
            raise ProtocolError("the action language is a set, not duplicate catalog entries")
        if self.kind is SupportKind.OPEN and self.commands:
            raise ProtocolError("open support must not pretend to enumerate its language")


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    prompt: str
    seed: int
    route: str
    commands: tuple[str, ...]
    output_limit: int = ORDINARY_OUTPUT_TOKEN_LIMIT
    stop_mode: str | None = None
    parameter_language: dict | None = None


@dataclass(frozen=True)
class Generation:
    request: GenerationRequest
    text: str
    input_tokens: int
    output_tokens: int
    raw_response: dict[str, Any]

    def __post_init__(self):
        if type(self.text) is not str:
            raise ProtocolError("generation text is required even for an empty model output")
        nonnegative(self.input_tokens, "input_tokens")
        nonnegative(self.output_tokens, "output_tokens")
        if (type(self.request.output_limit) is not int
                or self.request.output_limit < 1):
            raise ProtocolError("generation request has an invalid output limit")
        if self.output_tokens > self.request.output_limit:
            raise ProtocolError("generation exceeded its declared output limit")


@dataclass(frozen=True)
class Limits:
    environment_calls: int = 480
    model_requests: int | None = 96
    output_tokens: int | None = 96 * REFLECTION_OUTPUT_TOKEN_LIMIT
    input_tokens: int | None = 300_000
    elapsed_seconds: int = 3600

    def __post_init__(self):
        for name in ("environment_calls", "model_requests", "output_tokens",
                     "input_tokens", "elapsed_seconds"):
            if name in ('model_requests', 'output_tokens', 'input_tokens') and getattr(self, name) is None:
                continue
            if nonnegative(getattr(self, name), name) == 0:
                raise ProtocolError("resource limits must be positive")


@dataclass
class Ledger:
    limits: Limits
    started_at: float = field(default_factory=time.monotonic)
    environment_calls: int = 0
    backend_requests: int = 0
    model_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    physical_model_requests: int = 0
    physical_input_tokens: int = 0
    physical_output_tokens: int = 0
    cross_arm_cache_hits: int = 0
    own_cache_hits: int = 0
    owned_requests: set[str] = field(default_factory=set)
    unknown_generation_costs: list[str] = field(default_factory=list)
    model_queue_seconds: float = 0.0
    model_batch_seconds: float = 0.0
    timed_physical_model_requests: int = 0

    def reserve_time(self):
        if time.monotonic() - self.started_at >= self.limits.elapsed_seconds:
            raise BudgetExhausted("TIME_RESOURCE_LIMIT")

    def reserve_model(self, count, *, output_limit=ORDINARY_OUTPUT_TOKEN_LIMIT,
                      input_tokens=0, check_time=True):
        nonnegative(count, "reserved requests")
        if check_time:
            self.reserve_time()
        if ((self.limits.model_requests is not None and self.model_requests + count > self.limits.model_requests)
                or (self.limits.output_tokens is not None and self.output_tokens + count * output_limit > self.limits.output_tokens)
                or (self.limits.input_tokens is not None and self.input_tokens + input_tokens > self.limits.input_tokens)):
            raise BudgetExhausted("MODEL_RESOURCE_LIMIT")

    def reserve_environment(self):
        self.reserve_time()
        if self.environment_calls >= self.limits.environment_calls:
            raise BudgetExhausted("ENVIRONMENT_RESOURCE_LIMIT")

    def record_environment(self, backend_requests):
        nonnegative(backend_requests, "backend requests")
        self.reserve_environment()
        self.environment_calls += 1
        self.backend_requests += backend_requests

    def record_generation(self, generation, *, physical):
        identifier = generation.request.request_id
        if physical:
            self.physical_input_tokens += generation.input_tokens
            self.physical_output_tokens += generation.output_tokens
            timing = generation.raw_response
            if "queue_time_seconds" in timing and "wall_time_seconds" in timing:
                queue, batch = timing["queue_time_seconds"], timing["wall_time_seconds"]
                if any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in (queue, batch)):
                    raise ProtocolError("model timing must contain finite nonnegative durations")
                self.model_queue_seconds += queue
                self.model_batch_seconds += batch
                self.timed_physical_model_requests += 1
        if identifier in self.owned_requests:
            if physical:
                raise ProtocolError("an owned proposal must not be regenerated")
            self.own_cache_hits += 1
            return
        self.reserve_model(1, output_limit=generation.output_tokens,
                           input_tokens=generation.input_tokens, check_time=False)
        self.model_requests += 1
        self.input_tokens += generation.input_tokens
        self.output_tokens += generation.output_tokens
        self.owned_requests.add(identifier)
        if not physical:
            self.cross_arm_cache_hits += 1

    def to_dict(self):
        elapsed = time.monotonic() - self.started_at
        timing_complete = self.timed_physical_model_requests == self.physical_model_requests
        names = ("environment_calls", "backend_requests", "model_requests", "input_tokens",
                 "output_tokens", "physical_model_requests", "physical_input_tokens",
                 "physical_output_tokens", "cross_arm_cache_hits", "own_cache_hits")
        return {**{name: getattr(self, name) for name in names},
                "elapsed_seconds": elapsed,
                "model_queue_seconds": self.model_queue_seconds,
                "model_batch_seconds": self.model_batch_seconds,
                "queue_excluded_elapsed_seconds": elapsed - self.model_queue_seconds if timing_complete else None,
                "timing_accounting_complete": timing_complete,
                "unknown_generation_costs": list(self.unknown_generation_costs),
                "token_accounting_complete": not self.unknown_generation_costs}


class Episode(Protocol):
    action_limit: int
    backend_requests: int
    unused_final_proposal: bool

    def reset(self) -> dict[str, Any]: ...
    def snapshot(self) -> dict[str, Any]: ...
    def prompt(self, memory: tuple[str, ...]) -> str: ...
    def parse(self, text: str) -> ParsedAction: ...
    def support(self) -> ActionSupport: ...
    def public_physical(self, action: ParsedAction, support: ActionSupport) -> bool: ...
    def step(self, action: str) -> dict[str, Any]: ...
    def context(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class PairResult:
    ordinary: ParsedAction
    alternative: ParsedAction | None
    reason: str
    proposal_ids: tuple[str, ...]


class ProposalOracle:
    """Cache only submitted model proposals. Rewards and environment receipts are not returned."""

    def __init__(self, generate: Callable[[GenerationRequest], Generation], *, master_seed: int,
                 emit: Callable[..., None], token_counter=None, context_limit=None,
                 prompt_limit=None, request_namespace=None, generation_versions=None,
                 command_language_validator=None, ordinary_output_limit=ORDINARY_OUTPUT_TOKEN_LIMIT):
        self.generate = generate
        self.master_seed = master_seed
        self.emit = emit
        self.token_counter = token_counter
        self.context_limit = context_limit
        self.prompt_limit = prompt_limit
        self.request_namespace = request_namespace
        self.generation_versions = generation_versions
        self.command_language_validator = command_language_validator
        if type(ordinary_output_limit) is not int or ordinary_output_limit < 1:
            raise ProtocolError('ordinary output budget must be a positive integer')
        self.ordinary_output_limit = ordinary_output_limit
        # A scoped caller may track delivery separately from prompt rendering.
        # No observer preserves the existing non-scoped request contract.
        self.on_request_submitted = None
        if token_counter is not None and (
                not callable(token_counter) or type(context_limit) is not int
                or type(prompt_limit) is not int):
            raise ProtocolError("prompt counting requires declared model and service limits")
        self.contexts = {}
        self.generations = {}
        self.completed_pairs = {}
        self.completion_bindings = {}
        self.completion_seed_addresses = {}

    def bind(self, address, episode, prompt, memory):
        context = public_copy({"episode": episode.context(), "snapshot": episode.snapshot(),
                               "prompt": prompt, "memory": list(memory)})
        if address in self.contexts and self.contexts[address] != context:
            self.emit("node_context_mismatch", address=address.to_dict(),
                      expected=self.contexts[address], actual=context)
            raise ProtocolError("same node address has different actual history, state or prompt")
        if address not in self.contexts:
            self.contexts[address] = context
            self.emit("node_bound", address=address.to_dict(), context=context)

    def request(self, address, role, sample, prompt, ledger, *, commands=(),
                output_limit=None, stop_mode=None, parameter_language=None):
        seed = request_seed(address, role, sample, master_seed=self.master_seed)
        identifier = (f"b{address.benchmark}_t{address.task}_p{int(address.phase)}_"
                      f"r{address.replica}_n{address.binary_address}_s{address.native_ordinal}_"
                      f"role{int(role)}_q{sample}")
        if role is Role.REFLECTION and commands:
            raise ProtocolError("reflection requests cannot use a command language")
        if output_limit is None:
            output_limit = (REFLECTION_OUTPUT_TOKEN_LIMIT if role is Role.REFLECTION
                            else self.ordinary_output_limit)
        if type(output_limit) is not int or output_limit < 1:
            raise ProtocolError("request output limit is outside the declared protocol")
        route = ("reflection" if role is Role.REFLECTION
                 else "command_trie" if commands else "normal")
        if parameter_language is not None:
            from icr_parameter_language import validate_spec
            validate_spec(parameter_language)
            if commands or stop_mode is not None:
                raise ProtocolError('parameter language cannot mix with command trie or framing stops')
        request = GenerationRequest(identifier, prompt, seed, route, commands, output_limit, stop_mode,
                                    parameter_language)
        return self._submit_request(request, ledger)

    def _submit_request(self, request, ledger):
        identifier, prompt = request.request_id, request.prompt
        seed, route, commands = request.seed, request.route, request.commands
        output_limit = request.output_limit
        physical = identifier not in self.generations
        if physical:
            input_tokens = 0
            if self.token_counter is not None:
                input_tokens = self.token_counter(prompt)
                nonnegative(input_tokens, "complete prompt token count")
                self.emit("complete_prompt_counted", request_id=identifier,
                          input_tokens=input_tokens, output_limit=output_limit,
                          context_limit=self.context_limit, prompt_limit=self.prompt_limit)
                if input_tokens + output_limit > self.context_limit:
                    raise BudgetExhausted("MODEL_CONTEXT_LIMIT: complete prompt cannot be truncated")
                if input_tokens > self.prompt_limit:
                    raise BudgetExhausted("SERVICE_PROMPT_LIMIT: complete prompt cannot be truncated")
            ledger.reserve_model(1, output_limit=output_limit, input_tokens=input_tokens)
            ledger.physical_model_requests += 1
            self.emit("model_request_submitted", request_id=identifier, prompt=prompt,
                      seed=seed, route=route, commands=list(commands), parameter_language=request.parameter_language)
            if self.on_request_submitted is not None:
                self.on_request_submitted(request)
            try:
                generation = self.generate(request)
            except Exception:
                # The submitted request is real; missing usage remains unknown.
                ledger.model_requests += 1
                ledger.unknown_generation_costs.append(identifier)
                raise
            if generation.request != request:
                raise ProtocolError("model returned a generation for a different request")
            self.generations[identifier] = generation
            self.emit("model_request_completed", request_id=identifier, prompt=prompt,
                      seed=seed, route=route, commands=list(commands),
                      text=generation.text, input_tokens=generation.input_tokens,
                      output_tokens=generation.output_tokens, raw_response=generation.raw_response,
                      parameter_language=request.parameter_language)
        else:
            generation = self.generations[identifier]
            if generation.request != request:
                raise ProtocolError("proposal address was reused with different model input")
        ledger.record_generation(generation, physical=physical)
        if generation.raw_response.get("finish_reason") == "time":
            self.emit("model_request_time_limit", request_id=identifier, cost=ledger.to_dict())
            raise BudgetExhausted("MODEL_REQUEST_TIME_RESOURCE_LIMIT")
        self.emit("model_proposal_used", request_id=identifier, physical_request=physical,
                  standalone_cost=ledger.to_dict())
        return generation

    def choose_command(self, address, episode, memory, commands, *, state_revision,
                       no_good_version, contract_revision, horizon_remaining, ledger):
        if not isinstance(address, RequestAddressV2):
            raise ProtocolError("completion request requires its v2 address")
        if (not self.request_namespace or not isinstance(self.generation_versions, dict)
                or not self.generation_versions):
            raise ProtocolError("completion request lacks a namespace or generation versions")
        if (type(commands) is not tuple or not commands or len(commands) > 1024
                or len(set(commands)) != len(commands)
                or any(type(command) is not str for command in commands)):
            raise ProtocolError("completion commands must be a nonempty ordered set")
        if (type(no_good_version) is not int or no_good_version < 0
                or not contract_revision or type(horizon_remaining) is not int
                or horizon_remaining < 1):
            raise ProtocolError("completion request has invalid contract or budget revisions")
        support = episode.support()
        for command in commands:
            parsed = episode.parse(command)
            if (parsed.action != command or not parsed.grammar_valid or parsed.is_think
                    or not episode.public_physical(parsed, support)):
                raise ProtocolError("completion language contains a non-native public command")
        if not callable(self.command_language_validator):
            raise ProtocolError("completion request requires the actual command tokenizer validator")
        command_sequences = self.command_language_validator(commands, self.ordinary_output_limit)
        prompt = episode.prompt(memory)
        signature = public_copy({
            "public_state": episode.snapshot(), "state_revision": state_revision,
            "actor_context": episode.context(), "memory": list(memory), "prompt": prompt,
            "generation_versions": self.generation_versions, "route": "command_trie",
            "commands": list(commands), "no_good_version": no_good_version,
            "command_digest": hashlib.sha256(json.dumps(list(commands), ensure_ascii=True,
                separators=(",", ":")).encode("ascii")).hexdigest(),
            "command_token_sequences": command_sequences,
            "contract_revision": contract_revision, "horizon_remaining": horizon_remaining,
            "output_limit": self.ordinary_output_limit,
        })
        if address in self.completion_bindings and self.completion_bindings[address] != signature:
            self.emit("request_context_mismatch", address=address.to_dict(),
                      expected=self.completion_bindings[address], actual=signature)
            raise ProtocolError("REQUEST_CONTEXT_MISMATCH")
        self.completion_bindings[address] = signature
        seed = completion_request_seed(address, master_seed=self.master_seed)
        if seed in self.completion_seed_addresses and self.completion_seed_addresses[seed] != address:
            raise ProtocolError("COMPLETION_SEED_COLLISION")
        self.completion_seed_addresses[seed] = address
        identifier = self.request_namespace + "/" + address.identifier()
        self.emit("completion_request_bound", request_id=identifier, address=address.to_dict(),
                  seed=seed, signature=signature)
        generation = self._submit_request(GenerationRequest(
            identifier, prompt, seed, "command_trie", commands, self.ordinary_output_limit), ledger)
        raw = generation.raw_response
        tokens = raw["output_token_ids"]
        eos = raw["decoding"]["eos_token_id"]
        parsed = episode.parse(generation.text)
        if (raw["finish_reason"] not in {"eos", "eos_at_limit"}
                or not tokens or tokens[-1] != eos or eos in tokens[:-1]
                or raw["decoding"]["command_language_size"] != len(commands)
                or tuple(tokens[:-1]) not in command_sequences
                or generation.text not in commands or parsed.action != generation.text
                or not parsed.grammar_valid or parsed.is_think):
            self.emit("constrained_output_failure", request_id=identifier,
                      raw_response=raw, parser=parsed.evidence)
            raise ProtocolError("CONSTRAINED_OUTPUT_FAILURE")
        self.emit("completion_command_selected", request_id=identifier,
                  address=address.to_dict(), commands=list(commands), action=parsed.action,
                  parser=parsed.evidence)
        return parsed, identifier

    def ordinary(self, episode, address, memory, ledger):
        prompt = episode.prompt(memory)
        self.bind(address, episode, prompt, memory)
        support = episode.support()
        if (support.kind is SupportKind.FINITE
                and getattr(episode, 'ordinary_command_constrained', False)):
            commands = tuple(support.commands)
            generation = self.request(address, Role.ORDINARY, 0, prompt, ledger,
                                      commands=commands,
                                      output_limit=self.ordinary_output_limit)
            parsed = episode.parse(generation.text)
            if (generation.text not in commands or parsed.action != generation.text
                    or not parsed.grammar_valid or parsed.is_think
                    or not episode.public_physical(parsed, support)):
                self.emit('constrained_output_failure', request_id=generation.request.request_id,
                          raw_response=generation.raw_response, parser=parsed.evidence)
                raise ProtocolError('CONSTRAINED_OUTPUT_FAILURE')
            self.emit('ordinary_parsed', address=address.to_dict(),
                      request_id=generation.request.request_id, parser=parsed.evidence)
            return parsed, generation.request.request_id
        generation = self.request(address, Role.ORDINARY, 0, prompt, ledger,
                                  stop_mode=getattr(episode, "ordinary_stop_mode", None))
        parsed = episode.parse(generation.text)
        self.emit("ordinary_parsed", address=address.to_dict(),
                  request_id=generation.request.request_id, parser=parsed.evidence)
        return parsed, generation.request.request_id

    def pair(self, episode, address, memory, ledger):
        support = episode.support()
        previous = self.completed_pairs.get(address)
        if previous is None or any(identifier not in ledger.owned_requests for identifier in previous.proposal_ids):
            ledger.reserve_model(1 + (1 if support.kind is SupportKind.FINITE else OPEN_PROPOSAL_LIMIT))
        ordinary, ordinary_id = self.ordinary(episode, address, memory, ledger)
        ids = [ordinary_id]
        if ordinary.is_think:
            result = PairResult(ordinary, None, "A0_THINK", tuple(ids))
        elif not ordinary.grammar_valid:
            result = PairResult(ordinary, None, "A0_MALFORMED", tuple(ids))
        elif not episode.public_physical(ordinary, support):
            result = PairResult(ordinary, None, "A0_NOT_PUBLIC_QUALIFIED", tuple(ids))
        else:
            result = self._alternative(episode, address, memory, ledger, support, ordinary, ids)
        if previous is not None and previous != result:
            raise ProtocolError("same realized node produced a different candidate pair")
        self.completed_pairs[address] = result
        self.emit("pair_acquired", address=address.to_dict(), ordinary=result.ordinary.action,
                  alternative=None if result.alternative is None else result.alternative.action,
                  reason=result.reason, proposal_ids=list(result.proposal_ids))
        return result

    def _alternative(self, episode, address, memory, ledger, support, ordinary, ids):
        commands = ()
        if support.kind is SupportKind.FINITE:
            commands = tuple(command for command in support.commands if command != ordinary.action)
            for command in commands:
                parsed = episode.parse(command)
                if (parsed.action != command or parsed.is_think or not parsed.grammar_valid
                        or not episode.public_physical(parsed, support)):
                    raise ProtocolError("finite public command language contains an ineligible action")
            if not commands:
                return PairResult(ordinary, None, "ALT_UNAVAILABLE:NO_DISTINCT_PUBLIC_COMMAND", tuple(ids))
            count = 1
        else:
            count = OPEN_PROPOSAL_LIMIT
        prompt = episode.prompt(memory)
        for sample in range(count):
            generation = self.request(address, Role.ALTERNATIVE, sample, prompt, ledger, commands=commands)
            ids.append(generation.request.request_id)
            candidate = episode.parse(generation.text)
            accepted = (candidate.grammar_valid and not candidate.is_think
                        and candidate.action != ordinary.action
                        and episode.public_physical(candidate, support))
            self.emit("alternative_proposed", address=address.to_dict(), sample=sample,
                      request_id=generation.request.request_id, parser=candidate.evidence,
                      accepted=accepted)
            if accepted:
                return PairResult(ordinary, candidate, "ELIGIBLE_PAIR", tuple(ids))
        return PairResult(ordinary, None, "ALT_UNAVAILABLE:PROPOSAL_EXHAUSTED", tuple(ids))


def environment_call(episode, ledger, operation, *arguments):
    before = nonnegative(episode.backend_requests, "backend counter")
    ledger.record_environment(0)
    try:
        receipt = operation(*arguments)
    finally:
        after = nonnegative(episode.backend_requests, "backend counter")
        if after < before:
            raise ProtocolError("environment backend counter moved backwards")
        ledger.backend_requests += after - before
    if receipt["backend_requests"] != after - before:
        raise ProtocolError("receipt backend count disagrees with the actual call counter")
    return receipt


def _execute_attempt(episode, oracle, ledger, *, benchmark, task, phase, replica, bits, memory, emit):
    """Eligibility precedes reading a bit; every returned observation is freshly executed."""
    if bits is not None:
        validate_bits(bits, complete=True)
        if phase is not Phase.TREE:
            raise ProtocolError("only a declared shared-tree phase may consume script bits")
    if type(memory) is not tuple or any(type(value) is not str for value in memory):
        raise ProtocolError("memory must be the literal tuple declared for this tree")
    prefix = ()
    history = []
    slots = []
    status = "ACTIVE"
    terminal_score = None
    initial = None
    try:
        initial = environment_call(episode, ledger, episode.reset)
        if initial["done"]:
            raise ProtocolError("root checkpoint is already terminal")
        emit("attempt_reset", receipt=initial)
        for ordinal in range(episode.action_limit):
            ledger.reserve_environment()
            address = NodeAddress(benchmark, task, phase, replica, prefix, ordinal)
            if bits is not None and len(prefix) < DEPTH:
                pair = oracle.pair(episode, address, memory, ledger)
                if pair.alternative is None:
                    chosen, mode = pair.ordinary, None
                    emit("unary_decision", address=address.to_dict(), reason=pair.reason,
                         action=chosen.action, slot_consumed=False)
                else:
                    mode = bits[len(prefix)]
                    chosen = pair.ordinary if mode == 0 else pair.alternative
            else:
                chosen, _ = oracle.ordinary(episode, address, memory, ledger)
                mode = None
            emit("environment_action_submitted", address=address.to_dict(), action=chosen.action,
                 requested_mode=mode)
            receipt = environment_call(episode, ledger, episode.step, chosen.action)
            if receipt["action"] != chosen.action:
                raise ProtocolError("native wrapper changed the selected action")
            history.append(receipt)
            emit("environment_action_completed", address=address.to_dict(), receipt=receipt)
            if mode is not None:
                if not receipt["native_accepted"]:
                    raise ProtocolError("an asserted public-callable fork action was rejected")
                slots.append({"slot": len(prefix), "mode": mode, "action": chosen.action,
                              "address": address.to_dict(), "receipt_index": len(history) - 1})
                prefix += (mode,)
            if receipt["done"]:
                status, terminal_score = "TERMINAL", receipt["public_terminal_score"]
                break
        if status == "ACTIVE":
            status = "EPISODE_ACTION_LIMIT"
            if episode.unused_final_proposal:
                address = NodeAddress(benchmark, task, phase, replica, prefix, episode.action_limit)
                _, identifier = oracle.ordinary(episode, address, memory, ledger)
                emit("native_unused_final_proposal", request_id=identifier, executed=False)
    except BudgetExhausted as exc:
        status = str(exc)
    return {"status": status, "public_terminal_score": terminal_score,
            "success": terminal_score == 1.0, "reset_receipt": initial,
            "history": history, "script": None if bits is None else list(bits), "consumed_bits": list(prefix),
            "slots": slots, "realized_pairs": [list(pair) for pair in sorted(realized_pairs(prefix))],
            "unreached_slots": ([] if bits is None else [{"slot": index, "reason": status}
                                for index in range(len(prefix), DEPTH)]),
            "cumulative_cost": ledger.to_dict()}


def execute_script(episode, oracle, ledger, *, benchmark, task, replica, bits, memory, emit):
    return _execute_attempt(episode, oracle, ledger, benchmark=benchmark, task=task,
                            phase=Phase.TREE, replica=replica, bits=bits, memory=memory, emit=emit)


def execute_plain(episode, oracle, ledger, *, benchmark, task, phase, attempt, memory, emit):
    if phase not in (Phase.SOURCE, Phase.FIXED_RETRY, Phase.REFLEXION):
        raise ProtocolError("plain continuation requires its declared independent stream phase")
    return _execute_attempt(episode, oracle, ledger, benchmark=benchmark, task=task,
                            phase=phase, replica=attempt, bits=None, memory=memory, emit=emit)
