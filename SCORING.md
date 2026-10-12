# WebShop Reporting

The default task result uses the highest reward from an actually executed
purchase over the initial attempt and all recovery attempts. A task with no
purchase has reward 0. Success means this highest purchase reward equals 1.
Unexecuted proposals and checkpoint selection scores do not contribute.

`benchmark_reward` and `success` follow this rule. `terminal_rewards` contains
the observed purchase rewards. `final_public_reward`,
`native_final_benchmark_reward`, and `native_final_success` preserve the final
environment state separately. The scoring policy is
`webshop_best_observed_purchase_v1`.

Completed run reports include R and SR on a 0-100 scale. No separate rescoring
script is required. For the recovery-depth experiment, each stage reports the
cumulative highest purchase reward after 0, 1, 2, 3, or 4 actual recoveries.

WebShop has no model-request count limit by default and no derived cumulative
input/output token cap. Per-call token/context limits, action budgets, recovery
count and elapsed-time limits remain explicit settings. Native environment
acceptance determines whether an option click executes; no additional check
rejects an accepted option solely because its label was absent from the current
public option catalog. The actual selected options are recorded, without adding
that label to the public callable-action catalog.
