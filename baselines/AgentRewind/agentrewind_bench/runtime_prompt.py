"""Compact adapter instructions; author recovery guidance remains literal."""
import json

from .native_protocol import RUNTIME_PREFIX


PROTOCOL_PREFIX='[AGENTREWIND_PROTOCOL]\n'
PROTOCOL_INSTRUCTION=(PROTOCOL_PREFIX+
    'Native action budgets count the current retained trajectory, including rejected '
    'environment attempts. A restore retains its prefix depth; only the discarded suffix '
    'frees native action positions. Recovery commands use model calls but no native action '
    'positions. Costs never reset. Observations appear in the native transcript; the current '
    'receipt below adds status and any public structured state. Follow the current available '
    'commands and the author recovery and memory guidance.')


def compact_command_instruction(tools,output_limit,*,commit_end=None):
    names=[tool['name'] for tool in tools]
    if not names or len(set(names))!=len(names) or set(names)-{
            'execute_action','backtrack_candidates','backtrack_commit'}:
        raise ValueError('compact runtime requires a unique declared native/recovery command set')
    lines=['Available commands:']
    if 'execute_action' in names:
        lines.append('- One native environment action, using the demonstrated syntax.')
    if 'backtrack_candidates' in names:
        lines.append('- backtrack_candidates[reason]: request checkpoints using observed failure; wait for the real result.')
    if 'backtrack_commit' in names:
        lines.extend(['- backtrack_commit[<record_uid>]\n<complete memory_summary>',
            'Commit: use a returned top-level ID, with the header alone on its first line. '
            'Pass the following memory text verbatim, without JSON/Python quotes or Markdown fences. '
            'Follow the author memory fields. Limit: '+str(output_limit)+' output tokens.'])
        if commit_end is not None:
            lines.append('After the complete memory argument, write '+commit_end+' alone on the final line. '
                'It is a transport delimiter, not part of memory. No input/control blocks or later turns may occur in this response.')
    lines.append('Return one call only, with no additional command or simulated result.')
    return '\n'.join(lines)


def compact_runtime_instruction(*,tools,remaining,recoveries,blocked,candidates,paused,
                                public_receipt,output_limit,commit_end=None):
    allowance='unlimited' if recoveries is None else str(recoveries)
    lines=[RUNTIME_PREFIX.rstrip(),
        f'Native actions remaining: {remaining}. Recoveries remaining: {allowance}.']
    if paused:
        lines.append('Recovery is paused after two decisions without restore or an accepted action. '
            'Only a native action is allowed. An environment-accepted action reopens recovery. '
            'This current mode overrides earlier commit-only advice; no restore or budget refresh occurred.')
    elif candidates:
        lines.append('Candidates are already returned: select a checkpoint and commit with memory; do not list again.')
    elif blocked:
        lines.append('The native attempt has ended without full success. Request candidates before committing; no native action can run.')
    else:
        lines.append('Continue with a native action, or request recovery after concrete failure evidence as directed above.')
    lines.append('Output phase: recovery-command.' if blocked or candidates else
                 'Output phase: native-action-or-rewind-command.')
    if commit_end is not None and {t['name'] for t in tools}=={'backtrack_candidates'}:
        lines.append('Output framing: one-line candidate command.')
    lines.append(compact_command_instruction(tools,output_limit,commit_end=commit_end))
    # Preserve task status and domain-specific public state, not a second copy
    # of the literal observation that is already present in the native history.
    status={key:value for key,value in public_receipt.items() if key!='observation'}
    lines.append('Current receipt: '+json.dumps(status,ensure_ascii=False))
    return '\n'.join(lines)
