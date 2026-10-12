#!/usr/bin/env bash
set -euo pipefail

# Run this script inside the existing two-GPU Slurm step. These paths describe
# infrastructure only; model selection is the same CLI used by the runner.
: "${EXPECTED_PARENT:?set the verified Slurm parent}"
: "${EXPECTED_NODE:?set the verified compute hostname}"
: "${STATE:?set a fresh resident control directory}"
: "${RESULTS_ROOT:?set the WebShop result directory}"
: "${WEBSHOP_ROOT:?set the installed WebShop repository}"
: "${PYTHON_BIN:?set the Python interpreter with vLLM 0.6.6}"
PACKAGE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
test "${SLURM_JOB_ID}" = "$EXPECTED_PARENT"
test "$(hostname)" = "$EXPECTED_NODE"
test "${CUDA_VISIBLE_DEVICES}" = '0,1'
test -n "${SLURM_STEP_ID}"
test ! -e "$STATE"

# Resolve model, capacity and experiment options through shared CLI definitions.
export PYTHONPATH="$PACKAGE/source:$PACKAGE/vllm_service:$PACKAGE/site_override:$WEBSHOP_ROOT${STAR_RUNTIME_PYTHONPATH:+:$STAR_RUNTIME_PYTHONPATH}"
export WEBSHOP_BENCHMARK_ROOT="$WEBSHOP_ROOT"
mkdir -p "$STATE/queue" "$STATE/running" "$STATE/done" "$STATE/failed"
launch_values=$("$PYTHON_BIN" "$PACKAGE/source/resident_launch_config.py" --output "$STATE/settings.json" "$@")
mapfile -t config <<<"$launch_values"
MODEL=${config[0]}
DTYPE=${config[1]}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export CGCR_EXPECTED_PARENT="$EXPECTED_PARENT" CGCR_EXPECTED_NODE="$EXPECTED_NODE"
MODEL_PIDS=()
WEBSITE_PID=''
cleanup() {
  if [ -n "$WEBSITE_PID" ]; then kill "$WEBSITE_PID" 2>/dev/null || true; fi
  for pid in "${MODEL_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 143' TERM INT
for gpu in 0 1; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$PACKAGE/serve_model.py" \
    --local_llm_path "$MODEL" --dtype "$DTYPE" --state "$STATE/model$gpu" \
    --parent-job-id "$EXPECTED_PARENT" --node "$EXPECTED_NODE" --gpu-index "$gpu" \
    --port "$((41581+gpu))" --max-num-seqs "${config[3]}" --context-limit "${config[4]}" --prompt-limit "${config[5]}" \
    --max-output-tokens "${config[6]}" --gpu-memory-utilization "${config[7]}" --max-num-batched-tokens "${config[8]}" \
    "${config[9]}" --temperature "${config[10]}" --top-p "${config[11]}" --top-k "${config[12]}" --repetition-penalty "${config[13]}" \
    >"$STATE/model$gpu.log" 2>&1 &
  MODEL_PIDS+=("$!")
done
"$PYTHON_BIN" "$PACKAGE/source/serve_webshop.py" --port 3001 \
  --initialization-seed "${config[2]}" --show-attributes true >"$STATE/website.log" 2>&1 &
WEBSITE_PID=$!

# Preserve exact step and process identities for deliberate cancellation.
"$PYTHON_BIN" - "$STATE" "${MODEL_PIDS[@]}" "$WEBSITE_PID" <<'PY'
import json, os, socket, sys
from pathlib import Path
state = Path(sys.argv[1])
step = os.environ['SLURM_JOB_ID'] + '.' + os.environ['SLURM_STEP_ID']
(state/'control.json').write_text(json.dumps(dict(step=step, node=socket.gethostname(),
    model_pids=list(map(int, sys.argv[2:4])), website_pid=int(sys.argv[4]),
    cancel_step_command='scancel ' + step), indent=2))
PY
for gpu in 0 1; do
  for attempt in $(seq 1 1800); do
    kill -0 "${MODEL_PIDS[$gpu]}"
    if curl --silent --fail "http://127.0.0.1:$((41581+gpu))/health" >"$STATE/health$gpu.json"; then break; fi
    if [ "$attempt" = 1800 ]; then exit 1; fi
    sleep 2
  done
done
for attempt in $(seq 1 1800); do
  kill -0 "$WEBSITE_PID"
  if curl --silent --fail http://127.0.0.1:3001/fixed_0 >"$STATE/website_ready.txt"; then break; fi
  if [ "$attempt" = 1800 ]; then exit 1; fi
  sleep 2
done
"$PYTHON_BIN" - "$STATE" "$MODEL" "$DTYPE" <<'PY'
import json, sys
from pathlib import Path
from model_backend import load_model_bundle, check_model_health
model, _, _ = load_model_bundle(sys.argv[2], sys.argv[3])
state = Path(sys.argv[1])
settings = json.loads((state/'settings.json').read_text())
health = [json.loads((state/('health%d.json'%i)).read_text()) for i in (0,1)]
for row in health:
    check_model_health(row, model)
    if row['status'] != 'READY' or row['max_num_seqs'] != settings['max_num_seqs']:
        raise RuntimeError('resident is not ready at the declared capacity')
(state/'READY.json').write_text(json.dumps(dict(model=model, health=health), indent=2))
PY

while true; do
  for pid in "${MODEL_PIDS[@]}" "$WEBSITE_PID"; do kill -0 "$pid"; done
  if [ -f "$STATE/queue_paused.json" ]; then sleep 2; continue; fi
  request=$(find "$STATE/queue" -maxdepth 1 -type f -name '*.json' -print -quit)
  if [ -z "$request" ]; then sleep 2; continue; fi
  name=$(basename "$request")
  mv "$request" "$STATE/running/$name"
  set +e
  "$PYTHON_BIN" "$PACKAGE/dispatch_webshop.py" --request "$STATE/running/$name" \
    --local_llm_path "$MODEL" --dtype "$DTYPE" --results-root "$RESULTS_ROOT" \
    --defaults "$STATE/settings.json" \
    --parent-job-id "$EXPECTED_PARENT" --node "$EXPECTED_NODE" \
    >"$STATE/${name%.json}.log" 2>&1
  code=$?
  set -e
  if [ "$code" = 0 ]; then
    mv "$STATE/running/$name" "$STATE/done/$name"
  else
    mv "$STATE/running/$name" "$STATE/failed/$name"
    printf '{"failed_request":"%s","exit_code":%s}\n' "$name" "$code" >"$STATE/queue_paused.json"
  fi
done
