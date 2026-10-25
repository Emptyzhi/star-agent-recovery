"""Role prompts for STAR Game24."""
from pathlib import Path
import re

PROMPTS = Path(__file__).resolve().parents[1]/'prompts'/'game24'

TASK_GOAL = ('Task goal: use each of the four ORIGINAL input occurrences exactly once '
             'with +, -, *, / to obtain exactly 24. A complete final expression must equal 24.')

BASE_PROMPT = """Use four numbers and binary arithmetic (+, -, *, /) to obtain 24.
Use exactly two currently available numbers for each intermediate operation.
You may submit a complete final expression beginning with answer: when ready.
If using intermediate operations, three accepted binary operations combine the four numbers.
Use each original number exactly once in the final expression.

Return one public action on the first line and nothing else.
"""

DEMONSTRATIONS = (PROMPTS/'game24_base_actionly.txt').read_text(encoding='utf-8')
GREEN_PROMPT = DEMONSTRATIONS + """

Continue with exactly one action, with no introduction or Markdown.
Use two occurrences from the current remaining numbers for a binary operation.
Equal values may be used together when two copies remain; fractions and zero are allowed.
Write nonterminating division results as exact fractions, never rounded decimals.
The final answer is judged using each ORIGINAL input occurrence exactly once.
It may revise tentative intermediate operations that did not lead to 24.
Intermediate values are not additional input numbers.
{memory}
# Here is the task:
Input: {task}
{history}{events}{action_format}
{state_instruction}
Act {step}>"""

YELLOW_DIAGNOSIS_PROMPT = TASK_GOAL + """
You are Yellow, the bounded local-repair stage for Game24.
Diagnose the observed defect, not a hypothetical failure of an unexecuted plan.
Input: {task}
{history}
Current numbers: {numbers}
Attempt: {attempt}
Observed issue: {issue}
Exact controller protocol feedback: {protocol_feedback}
Focus from the recorded issue: {repair_focus}
Recovery context (unverified; actual receipts take precedence):
{recovery_memory}
Actions remaining in this window: {remaining_actions}. A final answer uses one slot.
Give one short line: observed defect; the part to change; a concrete correction and its reason.
Check the correction's arithmetic and original-input usage yourself. A final expression
uses each ORIGINAL occurrence once; an intermediate uses two CURRENT occurrences.
Rejected actions consume no operands. Equal inputs are separate occurrences.
Exact fractions, zero and negatives are legal. A valid intermediate need not equal 24.
If the CURRENT executed pool is 24, compare the failed final submission with its
recorded executed expression, including operators, operands and parentheses.
Diagnose the final transcription from those current receipts; an earlier abandoned
branch or unverified repair plan does not override the accepted current operations.
If an earlier accepted choice may need revision, call that an uncertain strategy judgment,
not an environment rejection or an executed repair.
Your judgment is stored in ETT. It cannot request or trigger Red; Green continues
until the ordinary failed-attempt boundary. Do not claim the correction was executed.
Diagnosis>"""

YELLOW_ACTION_PROMPT = DEMONSTRATIONS + '\n\n' + TASK_GOAL + """

Your local correction note (fallible advice):
{diagnosis}
Recovery context (unverified; actual receipts take precedence):
{recovery_memory}
Exact controller protocol feedback: {protocol_feedback}
Focus from the recorded issue: {repair_focus}
The last proposal was {attempt}. Observed issue: {issue}
Use the CURRENT remaining numbers below, not numbers already consumed.
{state_instruction}
Return one environment action, without Markdown, explanation or observation.
Use the actual feedback above to check your local note before choosing the correction.
The note is fallible: do not repeat its proposed expression merely because it calls it correct.
An intermediate is exactly a op b = c using two CURRENT occurrences and one operator.
Check a op b = c yourself; a compound intermediate is not an atomic action.
A complete final action is answer: <expression> = 24, using each ORIGINAL occurrence once.
When a supported correction is a complete final answer, submit that final answer rather
than replacing it with an intermediate operation.
It may revise the tentative prefix. Current intermediate values are not new original inputs.
Fractions must be exact; write a/b without parentheses or spaces inside the numeric literal.
Rejected actions did not consume operands. Accepted intermediates alone are not a solution.
When the CURRENT pool is the single value 24, write the final action from its
recorded executed expression, preserving its operands, operators and parentheses.
Correct the final transcription using current receipts, rather than an older
unverified plan. The model must still propose the action for environment acceptance.
Change the observed defect, check the replacement and use one available action slot.
Yellow cannot invoke Red, refresh the window or output BACKTRACK. Earlier-prefix
revision advice stays in ETT for the ordinary failed-attempt boundary.

# Here is the task:
Input: {task}
{history}Current numbers: {numbers}
Act {step}>"""

def red_demonstrations(demonstrations):
    """Project fixed public examples to their input and verbatim final action."""
    blocks = demonstrations.split('## Example ')[1:]
    if not blocks:
        raise ValueError('public examples have no declared example blocks')
    pairs = []
    for block in blocks:
        inputs = [line[len('Input: '):] for line in block.splitlines() if line.startswith('Input: ')]
        answers = [match.group(1) for line in block.splitlines()
                   if (match := re.fullmatch(r'Act [0-9]+>[ \t]*(answer: .+)', line))]
        if len(inputs) != 1 or len(answers) != 1 or len(inputs[0].split()) != 4:
            raise ValueError('public example needs one four-input task and one complete final action')
        pairs.append('Input: '+inputs[0]+'\nPlan: '+answers[0])
    return ('[PUBLIC_FINAL_ACTION_EXAMPLES]\n'+ '\n\n'.join(pairs)+
            '\n[/PUBLIC_FINAL_ACTION_EXAMPLES]')


RED_DEMONSTRATIONS = red_demonstrations(DEMONSTRATIONS)

RED_PROMPT = TASK_GOAL + '\n\n' + RED_DEMONSTRATIONS + """
You are Red, the first failure-reflection stage for this Game24 task.
Use the actual trajectory and environment receipts below to devise one revised plan.
Yellow advice is unverified. Inspect its executed correction and subsequent outcomes;
do not adopt a diagnosis that the actual feedback does not support.
Distinguish incorrect arithmetic, incorrect original-input usage, and an accepted
prefix that has not produced a solution. A non24 intermediate is not itself an error.
A revised plan must still reach 24. Ending a calculation at another number is not a completed repair.
The public input/final-action pairs above come from the same Actor examples.
They demonstrate a complete arithmetic repair, not a new executed trajectory.
Each accepted binary
operation consumes two CURRENT occurrences and creates one result. From four
inputs, exactly three accepted binary operations leave one value; then submit
the final expression. Do not append more binary operations or reuse consumed
original inputs to rescue a one-value pool that is not 24. Revise an earlier
choice, or propose a complete final expression using the four originals once.
Equal original values are distinct occurrences: when the task has two copies,
using that value twice is required, not itself a defect.
Distinguish a strategy problem from an incorrect final expression or a missing final submission.
If accepted operations reached 24, inspect their lineage and the failed final submission.
Otherwise state which earlier choice you would change, what prefix remains useful,
and one consistent continuation, checking its actual value and using each original occurrence once.
Do not insert new numeric literals.
A final expression may revise tentative operations; an intermediate replacement is
one binary operation over two available occurrences, not a compound expression.
Do not select a checkpoint here. Do not invent past observations or claim a suggested
repair has executed. State uncertainty if no supported correction is available.

Public failed-attempt evidence:
{trajectory}

[CURRENT_GAME24_TASK]
Input: {task}
[/CURRENT_GAME24_TASK]
Write at most 90 words of plain text for THIS task only:
Observed failure: cite the actual defect and the earlier choice that needs revision.
Plan: answer: <one complete proposed expression using all four original occurrences> = 24
The Plan line is fallible repair advice, not an executed action or a verified answer.
Check its arithmetic and original-input multiplicities yourself before stating it.
Do not use intermediate values as additional input literals. If you cannot find a
consistent repair, state that uncertainty rather than claiming an incorrect value is 24.
Give one proposed repair. Do not write an Act/Obs transcript or repeat the proposal.
New plan>
"""

RED_CHECKPOINT_PROMPT = TASK_GOAL + """
Choose the actual executed action before which to restore Game24.
Use the first reflection and the recorded operand occurrences to select a useful
prefix. The reflection is fallible advice; the candidate states are actual saved facts.
Root is a candidate, not a default. Do not retain an operation that
consumed an input the revised plan needs separately. A rejected final answer
leaves the pool unchanged, so restoring before it retains the preceding operations.
Repeated rejected actions can have identical saved operand states. Select according
to the prefix needed by the revised plan, not the largest action index.
Each ACTION n below denotes an actual action and restores its saved BEFORE state.
Output exactly one listed ACTION n. No reflection, review, explanation or Markdown.

Task: {task}
Failure: {failure}
First reflection:
{reflection}

Public candidates and their saved operand provenance (no future outcomes computed):
{candidates}

Choice:
"""


def render(template: str, **values: object) -> str:
    return template.format(**values)


def author_reflection(transcript, *, task):
    """Game24-specific reflection; public Actor demonstrations remain unchanged."""
    scenario = transcript.split('Here is the task:')[-1].strip()
    return render(RED_PROMPT, trajectory=scenario, task=task)
