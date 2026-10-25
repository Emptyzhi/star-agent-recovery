# STAR: State- and Trajectory-Aware Recovery

STAR records executed actions, environment feedback, and associated repair
events in an execution trajectory tree. Local recovery repairs an incident
without rollback. Task-level recovery reviews the failed execution, identifies
a checkpoint, and restores the environment and retained interaction history.

This anonymous artifact contains our implementation and baseline adapters,
plus the task-level records underlying every row of the manuscript's main
table. Third-party implementations and demonstrations are external dependencies.

## Verify the published results

No models or external packages are needed to recompute the main table:

```bash
python reproducibility/check_table1.py
python reproducibility/summarize.py
```

The main-table index covers 63 cohorts: seven methods, three models, and three
benchmarks. Each method/model combination includes 500 Game of 24 tasks,
134 ALFWorld tasks, and 500 WebShop tasks. The main-table records retain their
stated evaluation scopes, including the one-recovery WebShop prefixes.

`reproducibility/table1_index.json` identifies all 21 table rows and their task
records. `reproducibility/baseline_configurations.json` records configuration
evidence extracted from the selected runs. See `REPRODUCIBILITY.md` for scoring
and missing-field semantics, and `baselines/README.md` for adapter entry points.

## Obtain external dependencies

Install Python 3.10 or later and Git. Review `THIRD_PARTY.md`, then run:

```bash
python fetch_upstreams.py
python -m pip install -r requirements-test.txt
```

Exact upstream URLs and revisions are in `upstreams.json`. The helper retrieves
external code and applies the recorded compatibility edits. Downloaded code,
demonstrations, licenses, and the local cache are excluded from Git.
No benchmark data, model weights, or website indexes are bundled.

## Run STAR

Use the unified runner with explicit local resources and a resident model
service. Dataset action windows are 12 for Game of 24, 50 for ALFWorld, and
15 for WebShop. The default allows one checkpoint recovery. Sampling defaults
are temperature 0.1, top-p 1.0, top-k 20, and repetition penalty 1.05; the
environment initialization seed is 1. That seed does not reseed model sampling.

```bash
python run_star.py --dataset game24 --help
python run_star.py --dataset alfworld --help
python run_star.py --dataset webshop --help
python serve_model.py --help
```

Resident services and benchmark workloads run within an explicitly allocated
compute job. The runner requires the model path, resident endpoint, allocation,
node, GPU index, task IDs, output directory, and dataset resources. It validates
these bindings before execution. Install `requirements-vllm.txt` in the model
service environment and ALFWorld 0.3.5 in the environment worker installation.

Ordinary actions and local corrections use a 100-token output cap. STAR failure
review uses 256 tokens and checkpoint selection 100. ALFWorld permits up to four
local actions per entry; Game of 24 and WebShop permit one. All local actions
share the current execution window. Checkpoint restoration starts a new window.

## Recovery implementation

The tree stores executed actions and feedback. Diagnoses and repair events
refer to the corresponding execution nodes. Event views augment ordinary
interaction history; model advice remains separate from observed outcomes.
Checkpoint identification may map explicit public mismatch evidence to an
execution position or ask the model to select an available identifier. An
empty actor response restores its pre-generation checkpoint. Two recovery
stages therefore do not imply two model calls in every case.

The **w/o event subtree** ablation supplies all model roles with chronological
actions, observations, diagnoses, reflections, and actual restoration notices.
It does not supply retrieved event records, branch or operand-dependency links,
matched-repeat annotations, or separately selected recovery summaries. The model
chooses a restoration point from the available action indices using this history.
The full STAR path is unchanged. See `REPRODUCIBILITY.md` for configuration details.

## Layout

```text
source/                 STAR controller, event views, environment adapters
test_game24/            arithmetic adapter and controller
test_alfworld/          ALFWorld entry point
test_webshop/           WebShop entry point
vllm_service/          resident model service
baselines/              our six baseline adapter packages
adaptations/            edits to separately obtained upstream files
reproducibility/        selected task results and configuration evidence
tests/                  offline tests
upstreams.json          external revision and acquisition manifest
fetch_upstreams.py      explicit external dependency setup
```

After fetching dependencies, run offline checks with:

```bash
python -B -m unittest discover -s tests
```

The release has no model weights or running services. Offline checks do not
constitute a new benchmark run.
