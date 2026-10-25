# Baseline adapters

These packages contain our environment, transport, accounting, and execution
adapters for the six comparison methods. Upstream algorithms, prompts, and
licenses are obtained separately at pinned revisions by `fetch_upstreams.py`.
The external source is not tracked in this artifact.

| Method | Adapter entry point | External implementation |
| --- | --- | --- |
| ReAct | `ReAct/run_baseline.py` | GA benchmark implementation |
| Reflexion | `Reflexion/run_baseline.py` | GA benchmark implementation |
| GA-Rollback | `GA-Rollback/run_baseline.py` | GA benchmark implementation |
| ToT | `ToT/run_tot.py` | Tree of Thoughts |
| LATS | `LATS/run_lats.py` | Language Agent Tree Search |
| AgentRewind | `AgentRewind/run_agentrewind.py` | AgentRewind library |

Use each entry point's `--help` to see explicit resource and model bindings.
The three `run_baseline.py` entry points consume a JSON request and the
arguments `--request`, `--output-dir`, `--model-path`, `--parent-job-id`,
`--node`, and `--gpu-index`. Run the two GPU shards with distinct output
directories and the corresponding resident endpoint. Recorded method settings
are available in `../reproducibility/baseline_configurations.json`; replace
external resource bindings with your own paths and service endpoints.

For AgentRewind, `--native-adapter-root` identifies the root STAR artifact used for shared
native benchmark resources. This supplies environments and demonstrations,
not STAR's local-repair or checkpoint-selection policy. ToT and LATS retain
their own search procedures and use dataset-specific environment adapters.

The ToT and LATS adapter sources use their archived Qwen ALFWorld 50-action
packages as the release snapshot.
The per-run configuration evidence remains authoritative for historical runs.
