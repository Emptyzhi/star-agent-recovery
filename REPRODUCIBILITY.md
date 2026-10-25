# Reproducing the reported STAR results

## Results and configuration identity

`reproducibility/runs.json` maps each released STAR result to its archived run
identifier, dataset, model, component setting, evaluation scope, and task CSV.
It covers the nine main-table STAR results, nine Qwen component removals, and
the Qwen WebShop recovery-allowance curve. The one-recovery curve point is the
Qwen WebShop main-table record. The other four curve prefixes are separate
records; these overlapping prefixes must not be counted as independent tasks.

The Qwen WebShop main-table record covers one recovery (258/500 successes).
The no-RLR component record contains 240/500 successes. Run identifiers in the
index determine which records belong to each comparison.

Each task CSV contains only task indices, outcomes, token counts, request
counts where recorded, and timing fields. Machine paths, account names,
scheduler identities, GPU UUIDs, and endpoints are omitted. A blank request
count in the compact Llama ALFWorld export means that this field was not
included in that export; it does not mean zero requests. No manuscript cost
comparison is computed from that missing field.

Archived settings come from execution requests, group run configurations, or
task identity records; their source type is stated in each configuration.
The repository revision identifies the released implementation; the result
index identifies historical runs. Use their recorded configuration for reruns.

## Task selection

- Game of 24: benchmark puzzle CSV rows 0 through 499.
- WebShop: instruction IDs 0 through 499 in the fixed benchmark installation.
- ALFWorld: the 134 unseen tasks listed in `reproducibility/alfworld_tasks.json`.
  Entries are relative to the externally installed `json_2.1.1` data root and
  retain their exact evaluation order. Resolve them against that root when
  building the runner's absolute-path manifest.

Dataset contents, model weights, and backend indexes remain external. Obtain
them under their upstream terms as described in `THIRD_PARTY.md`.

## Runtime and prompts

`reproducibility/recorded_runtime.json` records versions and GPU properties
from an archived resident health record, with its scope identified. The
requirements files contain compatible package constraints rather than a full
historical environment lock. Preserve the resolved environment for new runs.

After external dependency setup, benchmark demonstrations are under `prompts/`. Recovery instructions
and native history construction are implemented in the dataset adapters and
shared controller. The paper's prompt boxes are readable summaries. Reproduce
the actual implementation prompts when rerunning a reported configuration.

The unified runner resolves the 12/50/15 forward windows for Game of 24,
ALFWorld, and WebShop. Some preserved internal class defaults predate these
settings; use the unified runner with explicit configuration rather than
instantiating an internal controller with its legacy defaults.

## Scoring and cost boundaries

Run the standard-library-only aggregation script:

```bash
python reproducibility/summarize.py
```

It checks task coverage and reproduces SR, mean reward, and available mean
costs from the released task CSVs. SR is the mean binary outcome; WebShop reward
is the best actually executed purchase within the stated evaluation scope.
The script outputs unrounded values so display rounding can be applied last.
These task summaries reproduce result tables; they do not reconstruct the
decision-level failure labels without the original event trajectories.

WebShop reward is the highest reward from an executed purchase within the
evaluated recovery prefix, or zero without a purchase. Success requires reward
one. Unexecuted proposals and checkpoint scores do not contribute.
`benchmark_reward` and `success` report these outcomes; `terminal_rewards`
stores observed purchase rewards. The `final_public_reward`,
`native_final_benchmark_reward`, and `native_final_success` fields retain the
final environment outcome separately. The policy identifier is
`webshop_best_observed_purchase_v1`. Recovery-curve records apply this policy
through each successive recovery.

WebShop uses the configured action, recovery, context, per-call output, and
elapsed-time limits, with no additional total request or token cap by default.
Native environment acceptance determines whether an option click executes.
Accepted selected options are recorded even when their labels were absent
from the current public callable-action catalog; that catalog is not expanded.

Tokens include both model input and output across action, diagnosis, review,
and selection requests. Queue-excluded time is the task's measured controller
elapsed time minus accumulated tokenizer-lock and engine scheduling waits.
Active shared-batch processing, environment work, and restoration remain in
that time. Prefix records end at their evaluated recovery boundary. Neither
experiment throughput nor aggregate duration divided by task count is used.

## Restoration semantics

Game of 24 restores its saved operand pool and execution history. WebShop
restores the saved logical session and history. ALFWorld resets the same task
and replays the retained prefix, checking actions, observations, and the
available-command records against the saved checkpoint. The implementation
raises protocol errors for inconsistent restoration. A checkpoint identifies
the state before its recorded action; the next ordinary action is selected
only after restoration.

RLR records its diagnosis and the feedback produced by actual local actions.
TCR first reviews the failed continuation, then selects an existing eligible
checkpoint. Identification uses a public-evidence mapping when supported,
otherwise a constrained model choice. Empty actor output returns to the
pre-generation checkpoint. Unexecuted proposals
are incident evidence and do not become executed-action nodes.

## Component ablations

The paper's **w/o event subtree** variant uses the archived option
`without_event_subtree` and CSV identifier `no_event_subtree`. It retains
ordinary interaction history, recovery notes, and the recovery procedures,
while removing event-specific records and action guidance supplied through
the subtree. This is a component-removal comparison, not an equal-information
comparison between tree and flat serialization. STAR's full event views
include task-relevant reminders as well as execution and repair records.

## Complete main-table records

`check_table1.py` reconstructs all main-table rows, including unrounded
three-task means. Baseline CSVs retain task IDs, outcomes, recorded costs,
and outcome provenance.
The five archived GA episode-assertion outcomes in the Mistral WebShop run are
included with their original failure status and recorded costs. They are not
missing tasks or newly imputed outcomes. WebShop best-executed-purchase scores
follow the archived scoring records; one-recovery prefixes use recorded stage
outcomes and costs rather than final four-recovery totals.

`baseline_configurations.json` preserves selected fields from each run's
archived request or task identity. Supply local model paths, service endpoints,
and scheduler bindings when launching a new run.
