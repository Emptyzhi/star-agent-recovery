"""Persistent reverse-review frames and their bounded forward calls."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReviewSession:
    session_id: int
    red_scope_id: int
    source_failure_event_id: str
    frozen_path_occurrence_ids: tuple[int, ...]
    next_checkpoint_index: int
    status: str = "ACTIVE"
    current_checkpoint_index: int | None = None
    trial_ids: list[int] = field(default_factory=list)


@dataclass
class ForwardTrial:
    trial_id: int
    owner_review_session_id: int
    checkpoint_index: int
    prefix_occurrence_ids: tuple[int, ...]
    source_checkpoint_occurrence_id: int | None
    path_window_limit: int
    forward_actions_granted: int
    review_request_id: str
    advice: str
    start_cost: dict[str, Any]
    start_restore_count: int
    status: str = "ACTIVE"
    actual_suffix_occurrence_ids: list[int] = field(default_factory=list)
    forward_request_ids: list[str] = field(default_factory=list)
    handoff_request_id: str | None = None
    end_cost: dict[str, Any] | None = None
    end_restore_count: int | None = None
    score: float | None = None
    reason: str | None = None

    @property
    def forward_actions_used(self):
        return len(self.actual_suffix_occurrence_ids)
