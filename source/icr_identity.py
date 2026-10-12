"""Explicit, bounded node/role seed addresses independent of scheduler order."""

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
from math import prod

from icr_design import validate_bits


class Phase(IntEnum):
    SOURCE = 0
    TREE = 1
    FIXED_RETRY = 2
    REFLEXION = 3


class Role(IntEnum):
    ORDINARY = 0
    ALTERNATIVE = 1
    REFLECTION = 2


RADICES = (2, 500, 4, 8, 32, 64, 3, 4)
ADDRESS_SPACE = prod(RADICES)
MAX_SERVICE_TASK_SEED = (2**64 - 1) // 1000


@dataclass(frozen=True)
class NodeAddress:
    benchmark: int
    task: int
    phase: Phase
    replica: int
    fork_prefix: tuple[int, ...]
    native_ordinal: int

    def __post_init__(self):
        validate_bits(self.fork_prefix, complete=False)
        if not isinstance(self.phase, Phase):
            raise ValueError("unknown node phase")
        for name, bound in (("benchmark", 2), ("task", 500), ("replica", 8),
                            ("native_ordinal", 64)):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < bound:
                raise ValueError(name + " exceeds its declared node address bound")

    @property
    def binary_address(self):
        result = 1
        for bit in self.fork_prefix:
            result = result * 2 + bit
        return result

    def to_dict(self):
        return {"benchmark": self.benchmark, "task": self.task, "phase": self.phase.name,
                "replica": self.replica, "fork_prefix": list(self.fork_prefix),
                "native_ordinal": self.native_ordinal}


def request_seed(address, role, sample, *, master_seed):
    if not isinstance(address, NodeAddress) or not isinstance(role, Role):
        raise ValueError("request needs a declared node and role")
    if type(sample) is not int or not 0 <= sample < 4:
        raise ValueError("proposal index exceeds the four-proposal limit")
    if role is not Role.ALTERNATIVE and sample != 0:
        raise ValueError("ordinary/reflection roles have exactly one request per address")
    if type(master_seed) is not int or master_seed < 0:
        raise ValueError("master seed must be a nonnegative integer")
    fields = (address.benchmark, address.task, int(address.phase), address.replica,
              address.binary_address, address.native_ordinal, int(role), sample)
    packed = 0
    for value, base in zip(fields, RADICES, strict=True):
        if not 0 <= value < base:
            raise ValueError("request field exceeds its mixed-radix base")
        packed = packed * base + value
    seed = master_seed * ADDRESS_SPACE + packed
    if seed > MAX_SERVICE_TASK_SEED:
        raise ValueError("request seed exceeds the existing model-service CUDA range")
    return seed


@dataclass(frozen=True)
class RequestAddressV2:
    benchmark: int
    task_instance: int
    outer_attempt: int
    branch_serial: int
    decision_serial: int
    purpose: str = "COMPLETE_COMMAND"
    sample: int = 0

    def __post_init__(self):
        for name in ("benchmark", "task_instance", "outer_attempt", "branch_serial",
                     "decision_serial"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(name + " must be a nonnegative integer")
        if self.benchmark != 0 or self.task_instance >= 500:
            raise ValueError("completion-v2 is scoped to the fixed WebShop task set")
        if self.purpose != "COMPLETE_COMMAND" or self.sample != 0:
            raise ValueError("completion-v2 has one declared purpose and sample")

    def to_dict(self):
        return {"schema": "completion-v2", "benchmark": self.benchmark,
                "task_instance": self.task_instance, "outer_attempt": self.outer_attempt,
                "branch_serial": self.branch_serial, "decision_serial": self.decision_serial,
                "purpose": self.purpose, "sample": self.sample}

    def identifier(self):
        return (f"completion-v2/b{self.benchmark}/t{self.task_instance}/"
                f"r{self.outer_attempt}/branch{self.branch_serial}/"
                f"decision{self.decision_serial}/{self.purpose}/q{self.sample}")


def completion_request_seed(address, *, master_seed):
    if not isinstance(address, RequestAddressV2):
        raise ValueError("completion request requires RequestAddressV2")
    if type(master_seed) is not int or master_seed < 0:
        raise ValueError("master seed must be a nonnegative integer")
    material = json.dumps(["completion-seed-v2", master_seed, address.to_dict()],
                          sort_keys=True, separators=(",", ":"), allow_nan=False)
    raw = hashlib.sha256(material.encode("ascii")).digest()
    return int.from_bytes(raw, "big") % (MAX_SERVICE_TASK_SEED + 1)
