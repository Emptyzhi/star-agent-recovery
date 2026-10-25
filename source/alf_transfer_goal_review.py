"""Bounded transfer reminders; task relevance remains a model decision."""
from alf_restored_facts import restored_facts
from tree_history import canonical

TRANSFER_GOAL_REVIEW_POLICY='FIRST_ACQUISITION_AND_FIRST_POST_PLACEMENT_ACQUISITION_PER_WINDOW'


def transfer_goal_review(controller):
    path=controller.lineage
    if len(path)<=controller.branch_start_ordinal or controller.episode.snapshot()['done']:
        return None
    current=path[-1]
    if not current.receipt['native_accepted'] or not current.action.startswith('take '):
        return None
    window=path[controller.branch_start_ordinal:]
    earlier_takes=[o for o in window[:-1] if o.receipt['native_accepted'] and o.action.startswith('take ')]
    earlier_puts=[o for o in window[:-1] if o.receipt['native_accepted'] and o.action.startswith('put ')]
    nodes=[current.occurrence_id]
    if not earlier_takes:
        slot='FIRST_ACQUISITION'
    elif earlier_puts and earlier_puts[-1].occurrence_id>earlier_takes[-1].occurrence_id:
        slot='ACQUISITION_AFTER_PLACEMENT'
        placement=earlier_puts[-1]
        nodes.insert(0,placement.occurrence_id)
        preceding=[o for o in earlier_takes if o.occurrence_id<placement.occurrence_id]
        if preceding:
            nodes.insert(0,preceding[-1].occurrence_id)
    else:
        return None
    key=canonical(['TRANSFER_GOAL_REVIEW',controller.cache_branch_id,slot])
    if key in controller.delivered_green_events:
        return None
    return dict(event_key=key,slot=slot,transfer_occurrence_ids=nodes,
                current_take_occurrence_id=current.occurrence_id,
                relevance_status='NOT_CLASSIFIED_BY_CONTROLLER')


def compact_event_state(controller):
    episode=controller.episode
    state=episode.snapshot()
    facts=restored_facts(episode.history,state)
    return dict(original_task_observation=episode.initial['observation'],
        current_observation=state['observation'],
        current_admissible_commands=list(state['available_actions']),
        known_held_objects=facts['known_held_objects'],
        latest_successful_transfer=facts['latest_successful_transfer'],
        latest_successful_navigation=facts['latest_successful_navigation'],
        remaining_actions=controller._forward_window()['new_actions_remaining'],
        semantics=facts['semantics']+' Only selected event receipts are shown; the full ETT stays in the controller.')
