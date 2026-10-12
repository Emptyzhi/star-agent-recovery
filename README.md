# STAR

STAR supports ALFWorld, Game24 and WebShop through dataset-specific adapters and
a shared model interface. This is the code-only export of the audited
`STAR_unified_20261002` implementation. It contains no experiment results,
trajectories, model weights, dataset catalogs or runtime caches.

## Repository layout

```text
LLMs/                  Model-service entry
test_alfworld/          ALFWorld entry
test_game24/            Game24 entry, arithmetic environment and controller
test_webshop/           WebShop entry
prompts/               Dataset demonstrations and reflection examples
source/                Shared CLI settings, clients, ETT and dataset runtime code
vendor/                Referenced author code and prompt sources
vllm_service/          Persistent inference server and token framing
site_override/         WebShop backend compatibility code and UI assets
tests/                 Offline regression tests
run_star.py            Unified --dataset entry
dispatch_star.py        Two-GPU queue dispatcher
serve_model.py         Model-service launcher
start_resident.sh      Existing-allocation resident queue loop
```

## Dataset entries

The dataset entries are thin aliases of the same CLI:

```bash
python test_alfworld/alf_star.py --help
python test_game24/g24_star.py --help
python test_webshop/web_star.py --help
python run_star.py --dataset game24 --help
```

Example command, inside an existing verified Slurm compute step:

```bash
python run_star.py \
  --dataset game24 --csv /shared/data/24.csv --task-ids 0,1,2,3 \
  --local_llm_path /shared/models/Qwen2.5-14B-Instruct \
  --endpoint http://127.0.0.1:41581 --dtype bfloat16 \
  --parent-job-id ACTUAL_PARENT --node ACTUAL_NODE --gpu-index 0 \
  --output-dir /shared/results/UNIQUE_RUN/group_0 --run-namespace UNIQUE_RUN_g0 \
  --workers-per-gpu 40 --environment-seed 1 \
  --temperature 0.1 --top-p 1 --top-k 20 --repetition-penalty 1.05
```

| Dataset | Input | Initial/recovery action defaults | Yellow actions per entry |
| --- | --- | --- | --- |
| alfworld | `--manifest task_manifest.json`, optional `--environment-config` | 49/49 | 4 |
| game24 | `--csv 24.csv` with a Puzzles column | 12/12 | 1 |
| webshop | `--website http://127.0.0.1:3001` | 15/15 | 1 |

ALFWorld defaults to 112 model requests per task. Its bundled environment
configuration allows 99 steps between resets so checkpoint prefix replay does
not consume the controller's separate 49-action recovery allowance. Model
requests include diagnosis and checkpoint selection; replay uses environment
steps without new model requests. An explicit `--model-request-limit` or
`--environment-config` overrides the corresponding default.

Task IDs refer to original CSV rows or manifest positions. No implicit task
selection/shuffle occurs. The ALFWorld manifest supplies `test`/`development`
lists of native gamefile identities. Dataset files are external to this repository.

## Model and configuration

Use `--local_llm_path` (alias `--model-path`) and `--endpoint` to select an
already-loaded model. Runners load only model metadata/tokenizer, not weights.
Sampling, concurrency, token caps, request/action budgets, timeouts and context
limits are CLI arguments. `--dry-run` prints resolved settings without contacting
a model. Act/ReAct is not an exposed mode; declared Act prompts are retained.

Game24 additionally accepts `--prompt-format plain|native_chat` (default
`plain`). Queue requests use the `prompt_format` field. `native_chat` applies
the selected tokenizer's chat template once, before token accounting and the
actual resident request. This option does not change task actions or budgets.

Game24 Red first reflects on the failed trajectory, then selects a checkpoint
and supplies a review. If that second response is malformed, omits the review,
or selects a checkpoint outside the restorable public path, the controller
explicitly restores checkpoint 0. It retains the raw invalid response and the
first reflection, without inventing a review or parsing a replacement selection.
Green then receives the full configured recovery window (12 actions by default).
Valid selections still restore the model-selected checkpoint. Empty first-stage
reflections, transport failures and invalid internal states remain errors.

The service entry is `python LLMs/serve_model.py --help`. It uses vLLM 0.6.6 and
must run inside a verified compute allocation. Model switching requires a
matching loaded service; a different runner argument does not reload weights.

The default model random stream is continuous, with no task/request reseeding.
Environment seed means WebShop initialization or ALFWorld general.random_seed.
Game24 arithmetic is deterministic; its recorded seed does not reseed the model.

All three datasets expose `--ablation full|without_yellow|without_red|without_event_subtree`.
ALFWorld without Yellow executes Green proposals directly, without local diagnosis
or repair calls. Without Red it ends after the initial 49-action window (or an
earlier terminal/empty Actor response), with no reflection, replay or second
window. Without the event subtree it omits sparse Green event projections and
historical recovery-event recall in Yellow. It preserves ordinary receipts,
Yellow's current local incident/feedback, Red reflection memory and the internal
ETT/checkpoints. Full mode retains the existing behavior. Budgets unused by a
disabled component are not reassigned. All arms use the unified CLI runner.

ALFWorld's native TextWorld registry and parsers are process-global. Its runtime
serializes environment creation/reset/step/replay/close while each task retains
its own environment state. Model requests stay outside that lock and execute
concurrently through the resident. Native temporary libraries should use local
compute storage (`TMPDIR=/tmp`), not an NFS temporary directory.

## Environment setup

Use Python 3.10 or newer for runner code. Install client requirements from
`requirements.txt`. Inference additionally uses `requirements-vllm.txt` in the
GPU runtime; this is not required for offline controller tests. ALFWorld needs
version 0.3.5, its data, and the native runtime dependencies. WebShop requires
the separately installed Princeton WebShop environment, full product data,
search index and Java 11; set `WEBSHOP_ROOT` and put `site_override` before the
installed WebShop package when starting its backend. The override preserves
query decoding and the public Attributes interface used in the baseline.

## Queue and accounting

`dispatch_star.py` accepts `dataset`, `run_id`, `package`, `task_ids`, two loaded
`endpoints`, dataset input paths, and optional experiment/sampling/capacity
settings. It partitions tasks across GPUs; each task has an independent client
and controller. `dispatch_webshop.py` retains compatibility with older requests
and routes dataset requests to the unified dispatcher.

Outputs are written only to explicit fresh run directories. Every model call,
including diagnosis/reflection/selection, is counted. Time excludes measured
model-service queue delay. Missing usage is unknown, not zero. Technical and
protocol failures are retained explicitly. No run has been launched by this export.

## Method provenance

WebShop uses model-independent r10_p12 r3 with r4 ablation switches. ALFWorld
uses the archived v16 controller, isolated from the WebShop-modified base.
Game24 keeps one initial window and one Red recovery; Red's second call selects
the checkpoint, and Green chooses freely after restoration. Its implementation
is synchronized with the reviewed Game24 v31 source, retaining the existing
module filenames and CLI entry. Both Red stages allow multiline output:
reflection uses 256 tokens, and checkpoint selection plus the required REVIEW
field uses 100 tokens. Both model-authored notes remain fallible recovery memory.

Game24 uses 12/12 action windows, one Yellow action per entry, two Yellow entries
per window, and 64 model requests. A rejected final expression leaves the pool
unchanged and can be retried only within the remaining controller budget.
Yellow may request the single Red recovery with BACKTRACK; this does not execute
a synthetic arithmetic action. Repeated malformed proposals receive explicit
protocol feedback and can end the window; no replacement answer is invented.

The Game24 event subtree projects two kinds of public evidence: changed operand
availability after an actual missing-operand rejection, and the unreplayed suffix
of an executed historical chain that reached 24 without a successful submission.
Operand-change notices are delivered once. Historical-chain evidence remains
visible only while the current executed provenance matches, shrinks as that
prefix is reexecuted, and retires after divergence or reaching 24 again. Boundary
source correspondences preserve historical operand identities. Raw ETT branches
remain stored; the controller does not search for or construct a solution.

Game24 uses a strict Fraction-based arithmetic adapter. It is not byte-identical
to GA's regex-based intermediate-number updater. Comparative evaluations must
declare a common environment contract or disclose this difference.

## Tests

```bash
python -B -m unittest discover -s tests
```

Tests using original archives are optional: set `STAR_BASELINE_ROOT` for WebShop
source parity, `STAR_ALFWORLD_BASELINE_ROOT` for ALFWorld parity, and
`STAR_TRAJECTORY_FIXTURES` for the two historical trajectory regressions.
These archives remain external because this export contains code only.

Controller logic is preserved. ALFWorld request and environment-step defaults
are aligned with its archived runs; packaging also replaces dated folder-name
imports and adds dataset/model entry aliases.
See `THIRD_PARTY.md` for author-source attribution.
