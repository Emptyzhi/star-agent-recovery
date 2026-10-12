"""Readable failed-path evidence independent of checkpoint choice restrictions."""
from icr_core import ProtocolError


def failed_branch_event(history):
    observations = {}
    observation_ids = {}
    rows = []

    def reference(text):
        if not isinstance(text, str):
            raise ProtocolError('failed-branch observation must be literal public text')
        if text not in observation_ids:
            key = 'ob' + str(len(observation_ids))
            observation_ids[text] = key
            observations[key] = text
        return observation_ids[text]

    for index, receipt in enumerate(history):
        if (not isinstance(receipt['action'], str)
                or type(receipt['native_accepted']) is not bool
                or type(receipt['done']) is not bool):
            raise ProtocolError('failed-branch event requires native action and receipt types')
        rows.append(dict(receipt_index=index, action=receipt['action'],
            before_observation=reference(receipt['before']['observation']),
            result_observation=reference(receipt['observation']),
            accepted=receipt['native_accepted'], done=receipt['done'],
            terminal_score=receipt['public_terminal_score']))
    return dict(schema='ALFWORLD_FAILED_BRANCH_EVENT_V1', source_action_count=len(history),
        chronological_receipts=rows, observation_text=observations,
        semantics='This is the observed historical failed branch BEFORE rollback. All action '
        'and result rows are retained in order. Every observation reference resolves to literal '
        'text in observation_text; repeated text is stored once without dropping repeated actions. '
        'These are failure experiences, not actions already completed at the selected checkpoint. '
        'For current possession and completed operations use RESTORED_STATE_FACTS and '
        'RESTORED_BRANCH_RECEIPTS. A rejection does not establish its cause. No hidden state, '
        'expert plan, successful future continuation or model-written summary is included.')
