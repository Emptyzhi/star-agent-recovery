"""Separate historical failure advice from a one-use bound repair command."""
from icr_core import ProtocolError


class RepairMemory:
    def __init__(self, lesson, checkpoint_index, first_action, facts=None):
        if not isinstance(lesson, str) or not lesson.strip():
            raise ProtocolError('repair memory requires the original nonempty model lesson')
        self.lesson = lesson
        self.checkpoint_index = checkpoint_index
        self.pending_instruction = first_action
        self.instruction_receipt = None
        self.facts = {} if facts is None else facts

    def observe_action(self, action, occurrence_id, accepted):
        if self.pending_instruction is None:
            return None
        if type(accepted) is not bool:
            raise ProtocolError('instruction retirement requires an actual acceptance receipt')
        instruction = self.pending_instruction
        status = ('SUPERSEDED' if action != instruction else
                  'EXECUTED_ACCEPTED' if accepted else 'EXECUTED_REJECTED')
        self.instruction_receipt = dict(instruction=instruction, action=action,
            occurrence_id=occurrence_id, accepted=accepted, status=status)
        self.pending_instruction = None
        return dict(self.instruction_receipt)

    def actor_context(self):
        return dict(controller_recovery_facts=self.facts,
                    historical_failure_lesson=self.lesson,
                    lesson_origin_checkpoint=self.checkpoint_index,
                    pending_instruction=self.pending_instruction,
                    semantics='The lesson describes a previous failed branch, not the current '
                    'inventory or a command sequence to repeat. Apply it only when its condition '
                    'matches current evidence. An executed or superseded instruction is retired; '
                    'use the latest receipts to choose the next unmet prerequisite.')
