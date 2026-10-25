"""Phase-scoped guidance and concise operational memory for the staged variant."""
from .native_protocol import MEMORY_FIELDS, RUNTIME_PREFIX


ACTION_GUIDANCE = (
    'Solve the current task using the native actions and Act-only demonstrations. '
    'Return one command in the current controller format. Use only public feedback; '
    'do not invent observations or future turns. A concrete failure, repeated failed '
    'repair, or a wrong/stuck path can justify backtracking. Do not request recovery '
    'before trying a task action and observing failure evidence. When backtracking '
    'is warranted, call backtrack_candidates and wait for its actual result; checkpoint '
    'selection and reflection instructions are supplied in the recovery phase. '
    'An accepted action is not success; only terminal reward 1 is full success. '
    'A lower terminal score remains eligible for recovery. After a restore, use the '
    'notebook correction before repeating the abandoned strategy. Notebook feedback '
    'is historical and its root-cause diagnosis is a hypothesis; the current receipt '
    'defines the current state. Respect the notebook preconditions and red lines, '
    'unless new public evidence contradicts them. The current controller determines '
    'which commands are allowed, including periods when recovery is paused.')

MEMORY_ORDER = (
    'corrected_strategy', 'do_not_repeat', 'first_3_tool_calls',
    'must_satisfy_before_mutation', 'mutation_preflight', 'do_not_call_before',
    'success_invariant', 'failed_check', 'evidence_observed', 'root_cause',
    'assumptions_to_recheck', 'observed_loop', 'why_local_repair_failed',
    'rewind_target_reason', 'selected_candidate_safety_check')

COMPACT_MEMORY_GUIDANCE = (
    'Memory presentation for this benchmark adaptation: retain all 15 author fields, '
    'but put the executable correction first. Use this field order: '+
    ', '.join(MEMORY_ORDER)+'. '
    'Write concise field: value clauses as the memory body. Aim for '
    'the entire memory_summary to fit within 900 characters, the existing notebook '
    'item display limit; the combined response still has a 1200-token output cap. '
    'Name the concrete different next action and the action that must not be repeated '
    'before explaining the diagnosis. Keep supporting clauses brief; do not repeat '
    'the task, candidate list, tool schemas, or full history. Quote the latest actual '
    'failure feedback verbatim in evidence_observed, retain uncertain assumptions '
    'as uncertain, and use only available tools. Do not omit required fields, fabricate '
    'evidence, or replace a concrete correction with generic advice. This changes '
    'memory wording and field order only; the author validator and notebook limits '
    'are unchanged. No controller will shorten or repair the submitted note.')


def recovery_phase(allowances):
    if len(allowances) != 1 or not allowances[0].startswith(RUNTIME_PREFIX):
        raise ValueError('staged guidance requires exactly one current allowance')
    return '\nOutput phase: recovery-command.' in allowances[0]


def staged_guidance(original, allowances, *, actor_only):
    if set(MEMORY_ORDER) != set(MEMORY_FIELDS) or len(MEMORY_ORDER) != 15:
        raise ValueError('staged memory order must retain all author fields')
    if recovery_phase(allowances) and not actor_only:
        return original+'\n\n'+COMPACT_MEMORY_GUIDANCE
    return ACTION_GUIDANCE
