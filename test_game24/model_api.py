"""Model API boundary for STAR Game24.

The controller depends on this small interface, not on a model family,
Transformers, vLLM, or a resident process implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Callable, Protocol, Sequence
from urllib.request import Request, urlopen
import math
from source.experiment_settings import Sampling


@dataclass(frozen=True)
class Generation:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    request_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    queue_seconds: float | None = None


class ModelClient(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        role: str,
        max_new_tokens: int,
        stop: Sequence[str] = (),
        commands: Sequence[str] = (),
    ) -> Generation: ...


class CallableModelClient:
    """Adapter for tests, resident clients, or an existing inference wrapper."""

    def __init__(self, generate_fn: Callable[..., Any]):
        self._generate_fn = generate_fn

    def generate(self, prompt: str, *, role: str, max_new_tokens: int, stop=(), commands=()):
        result = self._generate_fn(
            prompt,
            role=role,
            max_new_tokens=max_new_tokens,
            stop=tuple(stop),
            commands=tuple(commands),
        )
        if isinstance(result, Generation):
            return result
        if isinstance(result, str):
            return Generation(result)
        if not isinstance(result, dict) or not isinstance(result.get("text"), str):
            raise TypeError("model adapter must return Generation, string, or {text: string}")
        return Generation(
            text=result["text"],
            input_tokens=result.get("input_tokens"),
            output_tokens=result.get("output_tokens"),
            request_id=result.get("request_id"),
            raw=result,
            queue_seconds=result.get('queue_seconds'),
        )


class OpenAICompatibleClient:
    """Minimal HTTP client for an OpenAI-compatible resident endpoint."""

    def __init__(self, endpoint: str, *, model: str, timeout: float = 120.0, sampling=None):
        self.endpoint = endpoint.rstrip("/") + "/v1/completions"
        self.model = model
        self.timeout = timeout
        self.sampling = Sampling() if sampling is None else sampling

    def generate(self, prompt: str, *, role: str, max_new_tokens: int, stop=(), commands=()):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            **self.sampling.engine_parameters(),
            "stop": list(stop) or None,
        }
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        choice = body["choices"][0]
        usage = body.get("usage", {})
        return Generation(
            text=choice['text'],
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            request_id=body.get("id"),
            raw=body,
        )


class ResidentModelClient:
    """Translate role requests to the existing checked STAR resident protocol."""
    def __init__(self, client, tokenizer, *, prompt_limit, context_limit, prompt_format='plain'):
        self.client, self.tokenizer = client, tokenizer
        self.prompt_limit, self.context_limit = prompt_limit, context_limit
        self.prompt_format = prompt_format
        self.serial = 0

    def count_tokens(self, prompt):
        from model_backend import encode_prompt, format_model_prompt
        effective = format_model_prompt(self.tokenizer, prompt, self.prompt_format)
        return len(encode_prompt(self.tokenizer, effective)['input_ids'])

    def generate(self, prompt, *, role, max_new_tokens, stop=(), commands=()):
        from icr_core import GenerationRequest
        from model_backend import format_model_prompt
        from .star_controller import ResourceLimit
        tokens = self.count_tokens(prompt)
        if tokens > self.prompt_limit or tokens+max_new_tokens > self.context_limit:
            raise ResourceLimit('PROMPT_CONTEXT_LIMIT')
        effective = format_model_prompt(self.tokenizer, prompt, self.prompt_format)
        mode = ('first_nonempty_line' if role == 'yellow_diagnosis' else 'first_line') if stop else None
        route = 'command_trie' if commands else 'reflection' if role in ('red_reflection', 'red_checkpoint', 'yellow_diagnosis') else 'normal'
        self.serial += 1
        request = GenerationRequest(str(self.serial), effective, 1, route, tuple(commands), max_new_tokens, mode)
        reply = self.client.generate(request)
        queue = reply.raw_response['queue_time_seconds']
        if not isinstance(queue, (int, float)) or not math.isfinite(queue) or queue < 0:
            raise ValueError('invalid measured queue time')
        raw = dict(reply.raw_response, client_prompt_format=self.prompt_format, effective_prompt=effective)
        return Generation(reply.text, reply.input_tokens, reply.output_tokens,
                          reply.request.request_id, raw, queue)
