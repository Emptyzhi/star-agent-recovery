"""Project real Yellow events whose anchors remain on the current execution path."""
from copy import deepcopy


def recovery_events(controller, active):
    active_ids = set(active)
    rows, anchors, detail = [], [], set()
    for scope in controller.scopes.values():
        if scope.kind.value != 'YELLOW' or scope.event_anchor_occurrence_id not in active_ids:
            continue
        anchor = scope.event_anchor_occurrence_id
        anchors.append(anchor)
        detail.update(scope.occurrence_ids)
        decisions = []
        records = list(scope.correction_records)
        if (scope.correction_record is not None
                and all(r['request_id'] != scope.correction_record['request_id'] for r in records)):
            records.append(scope.correction_record)
        for record in records:
            decisions.append({key: deepcopy(record[key]) for key in (
                'request_id', 'conclusion', 'step_role', 'selected_action',
                'executed_occurrence_id', 'correction_completed')})
        rows.append({'scope_id': scope.scope_id, 'anchor_occurrence_id': anchor,
                     'incident_kind': scope.incident_kind, 'entry_reason': scope.entry_reason,
                     'exit_reason': scope.exit_reason, 'decisions': decisions,
                     'executed_occurrence_ids': list(scope.occurrence_ids),
                     'path_relation': ('ACTIVE_PREFIX' if set(scope.occurrence_ids) <= active_ids
                                       else 'HISTORICAL_BRANCH'),
                     'interpretation': 'Observed repair history and fallible model advice, not pending actions or proven task success.'})
    return rows, anchors, detail
