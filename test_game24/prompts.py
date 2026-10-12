"""Role prompts for STAR Game24."""
from pathlib import Path
from source.alf_author_reflection import AUTHOR_INSTRUCTION

PROMPTS = Path(__file__).resolve().parents[1]/'prompts'/'game24'

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

YELLOW_DIAGNOSIS_PROMPT = """Review this rejected or repeated Game24 action using only the recorded observations.
Input: {task}
{history}
Current numbers: {numbers}
Attempt: {attempt}
Observed issue: {issue}
Exact controller protocol feedback: {protocol_feedback}
Recovery context (unverified; actual receipts take precedence):
{recovery_memory}
Actions remaining in this window: {remaining_actions}. Every executed action,
including a final answer, consumes one slot. Making the pool 24 is not a submitted answer.
Equal-valued numbers are separate available operands when two copies remain.
Exact fractions, zero and negative results are legal. Rounded decimals are not exact fractions.
Rejected proposals consume no operands. Do not invent an integer-only rule.
Distinguish a formatting error from unavailable operands or incorrect arithmetic.
In one short line identify the precise error and a locally executable correction.
State the observed defect, a proposed action, and why it addresses that defect
within the remaining slots. Check the proposed arithmetic yourself; changing
format or producing a legal value alone does not establish a solution.
If correcting a rejected final answer, consider correcting that complete answer
directly instead of spending a slot re-performing its intermediate arithmetic.
Do not turn a wrong numerical result into a formatting diagnosis.
If an answer uses the wrong inputs but the recorded operations already made 24,
repair the final expression by following those operations, not by replacing them.
If an operand is missing, name that missing occurrence and the actual current pool.
Do not call a consumed operand available or mistake a parenthesized fraction for a decimal.
If no local correction addresses the problem, explain which earlier choice needs revision.
Diagnosis>"""

YELLOW_ACTION_PROMPT = DEMONSTRATIONS + """

Your local correction note (fallible advice):
{diagnosis}
Recovery context (unverified; actual receipts take precedence):
{recovery_memory}
Exact controller protocol feedback: {protocol_feedback}
The last proposal was {attempt}. Observed issue: {issue}
Use the CURRENT remaining numbers below, not numbers already consumed.
{state_instruction}
Return one arithmetic action in the example format, without Markdown or code.
An intermediate correction must be one binary operation over two CURRENT occurrences;
do not submit a compound expression as one action. A nested expression is allowed
only after the mandatory answer: prefix in a complete final answer.
Fractions are legal: use exact fractions for nonterminating divisions.
A rejected action did not consume its operands. Correct its format or arithmetic locally.
A complete final answer may use the ORIGINAL four inputs regardless of the intermediate pool.
Repair the reason for failure, not just the syntax. Before choosing an operation,
consider how its result can be used in a complete expression with the other inputs.
Check the correction note against the actual operands and remaining slots. If
you endorse its complete final answer, submit that model-written answer rather
than replacing it with an intermediate operation that only makes the pool 24.
If the note is wrong, revise it yourself; it is not an instruction to copy an
invalid action. A legal intermediate is only locally valid, not proof of progress.
Rollback available: {rollback_available}. Return BACKTRACK only if a prior accepted
operation really needs revision and rollback is available. This is not the response
to a formatting error or an incorrect declared arithmetic result.

# Here is the task:
Input: {task}
{history}Current numbers: {numbers}
Act {step}>"""

RED_PROMPT = """You are Red, the Game24 failure reflection stage.
Review the actual actions and observations. Explain the observed failure and devise a concise revised plan.
An arithmetically valid step is not proof of a successful overall strategy. An unsuccessful suffix does not prove its entire prefix wrong.
Distinguish rejected arithmetic, an incorrect final expression, and a valid but unhelpful intermediate choice.
Do not choose a checkpoint in this first reflection. Any suggested action is unverified advice, not an executed fact.
Every proposed intermediate replacement must be one atomic binary operation over
two values present at that checkpoint. Do not write a compound expression as an
intermediate action; nested expressions belong only in the final answer: form.
If the arithmetic plan is uncertain, state the uncertainty rather than inventing
a transition or claiming that a compound action was executed.
Input: {task}
Failure: {failure}
Compact public trajectory:
{trajectory}
Write a concise reflection naming the failure, its public evidence and a plan for the next attempt.
"""

RED_CHECKPOINT_PROMPT = """You are Red in the second recovery reflection stage for Game24.
Use the first reflection and the public checkpoint candidates to choose the prefix that should be restored.
Choose a checkpoint before the action you would change according to the reflection. Root is a candidate, not a default.
Do not invent a checkpoint, action, state, or future result. The controller will validate your choice against the executed path.
Compare the operands needed by the proposed plan against each actual remaining pool.
Do not retain an operation that consumed an input the revised plan needs separately.
If only the final answer format was wrong and the current arithmetic pool is useful,
the current checkpoint is eligible. Select the boundary yourself; do not treat the
first reflection's action number as a reliable checkpoint index.
Respond in this exact two-field format, with no introduction or Markdown:
CHECKPOINT <integer>
REVIEW: <explain which failed action needs changing, why this pool permits it, and the next action, either an atomic operation or a complete answer>
Use at most 70 words. Both fields are required. Green receives this review as well
as the first reflection. When the observed operations already reached a single 24,
consider a checkpoint that retains those operations and repairs the final expression.
This is your decision, not a controller rule. A rejected final answer leaves the pool unchanged.
If the first plan is wrong or unfinished, say so explicitly;
do not claim a solution just because an expression ends with = 24.

Task: {task}
Failure: {failure}
First reflection:
{reflection}

Each candidate is the state BEFORE its next recorded action. A correct individual
operation can still be the wrong strategic choice. Select the boundary that lets
the new plan change the necessary operation; do not keep an operation your plan replaces.
Public checkpoint candidates (no future outcomes have been computed):
{candidates}

Choice:
"""


def render(template: str, **values: object) -> str:
    return template.format(**values)


def author_reflection(transcript):
    """Author reflection with an explicit arithmetic-plan objective, no external solver."""
    examples = (PROMPTS/'game24_reflexion_examples.txt').read_text(encoding='utf-8')
    scenario = transcript.split('Here is the task:')[-1].strip()
    objective = ('For THIS puzzle, write one concise recovery note from the actual receipts. '
                 'First distinguish a bad arithmetic strategy from an incorrect final expression or a missing final submission. '
                 'Do not blame an early operation merely because the attempt failed. '
                 'If the recorded operations reached a single 24, trace those operations back to the original inputs '
                 'and propose the complete answer; do not invent a replacement arithmetic strategy. '
                 'Otherwise work out a revised expression, checking its actual value and using each original occurrence once. '
                 'Use exact fractions and remember that equal-valued inputs are separate copies. '
                 'Name the operation or final submission to change and any useful operations to retain. '
                 'Finish with Plan: followed by your proposed operations or complete answer. '
                 'Do not repeat an explicitly unsuccessful plan as a solution. If unsolved, say so. '
                 'Stay under 120 words; proposals are not new observations. Do not select a checkpoint in this stage.')
    return AUTHOR_INSTRUCTION+'\n\n'+examples+'\n\n'+scenario+'\n\n'+objective+'\n\nNew plan>'
