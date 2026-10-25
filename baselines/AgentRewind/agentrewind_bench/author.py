"""Use the pinned upstream candidate, commit, context and memory implementation."""
from pathlib import Path
import sys

UPSTREAM = Path(__file__).resolve().parents[1]/'vendor/upstream'
sys.path.insert(0, str(UPSTREAM))

from replay.backtrack import (BacktrackError, BacktrackRequested, backtrack_candidates, backtrack_commit,
                             backtrack_tool_definitions, backtrack_tool_instructions,
                             prepare_backtrack_replay, load_backtrack_memory,
                             render_backtrack_memory_notebook, is_backtrack_memory_message)
from replay.storage import JsonlWriter, make_record_uid, run_path
from replay.errors import InvalidBreakpointError
from replay.tools import tool_error_to_record


def tools():
    return [{'name': 'execute_action', 'description': 'Execute one native benchmark action.',
             'parameters': {'type': 'object', 'properties': {'action': {'type': 'string'}},
                            'required': ['action'], 'additionalProperties': False}},
            *backtrack_tool_definitions(selection_policy='agent-direct')]


def instruction(recovery_format='bracket-json',*,commit_boundary=None,json_tools=False):
    mapping = ('Benchmark-domain mapping for the unchanged author guidance below: '
            'workspace/filesystem means the native benchmark environment; verifier feedback '
            'means the actual task observation and terminal reward; mutation means a native '
            'action that changes the task state. Filesystem tools and artifacts in the '
            'guidance are examples from the original domain, not available tools here. '
            'Use only the native commands demonstrated for this benchmark. '
            'Verification means the existing benchmark action/feedback, never a new helper. '
            'Apply preconditions internally; output remains Act-only, without a reasoning turn. '
            'The benchmark text transport calls backtrack_commit with record_uid and '
            'memory_summary together in one JSON argument object inside brackets. '
            'Solve the current interactive task using its native actions and observed feedback. '
            'The demonstrations illustrate action syntax, not actions already executed in this task. '
            'For ordinary environment actions return one action line without explanation, '
            'Markdown or a JSON wrapper. Recovery tools use their advertised command syntax. '
            'Use only public observations. Follow the current action and recovery allowance. '
            'A recovery notebook records evidence from an abandoned branch: treat its diagnosis '
            'as a hypothesis and its observations as historical, not the current environment state. '
            'Feedback action_accepted means the environment accepted that command, not that the '
            'task succeeded. task_success is true only at a terminal reward of 1. '
            'A terminal reward below 1 is an unsuccessful attempt, even when positive; '
            'the remaining recovery allowance can still be used to improve it.')
    if recovery_format=='python':
        mapping=mapping.replace(
            'The benchmark text transport calls backtrack_commit with record_uid and '
            'memory_summary together in one JSON argument object inside brackets. ',
            'The benchmark tool transport uses the author function signature '
            'backtrack_commit(record_uid="<id>", memory_summary=r"""<recovery note>""") '
            'and backtrack_candidates(reason="<observed failure>"). Both commit '
            'arguments are submitted together in one reply. These are literal '
            'tool calls, not Python scripts; do not include Markdown fences. ')
    elif recovery_format=='lines':
        mapping=mapping.replace(
            'The benchmark text transport calls backtrack_commit with record_uid and '
            'memory_summary together in one JSON argument object inside brackets. ',
            'The benchmark text transport submits both author commit arguments in one reply: '
            'write backtrack_commit[<record_uid>] on the first line, then the complete '
            'memory_summary as the remaining text. Do not JSON-quote or Python-quote the '
            'recovery note. It is one combined tool call, not a separate memory request. '
            'Candidate listing uses backtrack_candidates[<observed failure>]. ')
    elif recovery_format!='bracket-json':
        raise ValueError('unknown recovery argument transport')
    if commit_boundary is not None:
        if recovery_format!='lines':
            raise ValueError('explicit commit boundary requires lines transport')
        old=('write backtrack_commit[<record_uid>] on the first line, then the complete '
            'memory_summary as the remaining text. ')
        if mapping.count(old)!=1:raise ValueError('lines transport mapping changed unexpectedly')
        mapping=mapping.replace(old,'write backtrack_commit[<record_uid>] on the first line, then the complete '
            'memory_summary, then '+commit_boundary+' alone on the final line. '
            'The final delimiter is outside the memory argument. ')
    if json_tools:
        mapping = ('Benchmark-domain mapping for the unchanged author guidance below: '
            'workspace/filesystem means the native benchmark environment; verifier feedback '
            'means actual task observations and terminal reward; mutation means a native '
            'action that changes task state. Filesystem tools are original-domain examples, '
            'not available tools. Use only the native benchmark actions and public observations. '
            'The demonstrations describe syntax, not actions executed in the current task. '
            'Input is plain text. Output one JSON object with name and arguments, using the '
            'current tool schema. execute_action carries one literal native action in action. '
            'backtrack_candidates carries reason. backtrack_commit carries record_uid and '
            'the complete memory_summary together. This serialization replaces bracket commands '
            'and function-call syntax in the guidance; the memory requirements are unchanged. '
            'Do not generate a reasoning turn outside the tool arguments. '
            'A notebook contains historical evidence and hypotheses, not the current state. '
            'action_accepted is not task success: full success requires terminal reward 1. '
            'A lower terminal reward remains an unsuccessful attempt eligible for recovery.')
    return mapping + '\n\n' + backtrack_tool_instructions(
        selection_policy='agent-direct', context_policy='rollback', workspace_policy='rollback')
