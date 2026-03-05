#!/bin/bash
set -euo pipefail

# K-sweep: find optimal retrieval k for each model using cached memories.
# Memories are already built — this only re-runs the QA answering step.

AMEM_PYTHON="/common/users/wx139/env/amem_env/bin/python3"
VLLM_PYTHON="/common/users/wx139/env/verl-agent-alfworld/bin/python"
WORKDIR="/common/users/wx139/code/opensource_all/A-mem_opensource"
DATASET="data/locomo10.json"

K_VALUES=(10 15 20 25 30 35 40 45 50)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

mkdir -p "$WORKDIR/logs" "$WORKDIR/results_k_sweep"

# ---------- vLLM models ----------
# Model configs: "model_name port eval_gpu"
VLLM_MODELS=(
    "meta-llama/Llama-3.2-3B-Instruct 30000 2"
    "meta-llama/Llama-3.2-1B-Instruct 30001 3"
    "Qwen/Qwen2.5-3B-Instruct 30002 2"
    "Qwen/Qwen2.5-1.5B-Instruct 30003 3"
)

launch_vllm() {
    local model=$1 port=$2 gpu=$3 logfile=$4
    log "Launching vLLM: model=$model port=$port gpu=$gpu"
    CUDA_VISIBLE_DEVICES=$gpu \
    VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=XFORMERS \
    $VLLM_PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --port "$port" --host 0.0.0.0 \
        --tensor-parallel-size 1 --trust-remote-code \
        --dtype float16 --enforce-eager \
        --gpu-memory-utilization 0.95 \
        --max-model-len 8192 \
        > "$logfile" 2>&1 &
    echo $!
}

wait_for_server() {
    local port=$1 name=$2 timeout=300
    log "Waiting for $name on port $port ..."
    for i in $(seq 1 $timeout); do
        if curl -s "http://localhost:$port/v1/models" > /dev/null 2>&1; then
            log "$name ready on port $port"
            return 0
        fi
        sleep 1
    done
    log "ERROR: $name failed to start on port $port"
    return 1
}

kill_port() {
    local port=$1
    local pids=$(lsof -ti :$port 2>/dev/null || true)
    [ -n "$pids" ] && echo "$pids" | xargs kill -9 2>/dev/null || true
}

run_k_sweep_for_model() {
    local backend=$1 model=$2 port=$3 eval_gpu=$4
    local model_short=$(echo "$model" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')

    for k in "${K_VALUES[@]}"; do
        local outfile="results_k_sweep/${model_short}_k${k}.json"
        log "Running k=$k for $model -> $outfile"
        CUDA_VISIBLE_DEVICES=$eval_gpu $AMEM_PYTHON test_advanced_robust.py \
            --backend "$backend" \
            --model "$model" \
            --dataset "$DATASET" \
            --output "$outfile" \
            --retrieve_k "$k" \
            --sglang_port "$port" \
            > "logs/ksweep_${model_short}_k${k}.log" 2>&1
        log "Done k=$k for $model"
    done
}

# =============================================================================
# Step 1: Launch all 4 vLLM servers (2 at a time to fit GPUs 0-3)
# =============================================================================
log "=== Phase 1: Llama models k-sweep ==="

# Launch Llama servers on GPUs 0,1
PID_LLAMA3B=$(launch_vllm "meta-llama/Llama-3.2-3B-Instruct" 30000 0 "logs/ksweep_vllm_llama3b.log")
PID_LLAMA1B=$(launch_vllm "meta-llama/Llama-3.2-1B-Instruct" 30001 1 "logs/ksweep_vllm_llama1b.log")

if wait_for_server 30000 "Llama-3B" && wait_for_server 30001 "Llama-1B"; then
    # Run sweeps in parallel (eval on GPUs 2,3)
    run_k_sweep_for_model vllm "meta-llama/Llama-3.2-3B-Instruct" 30000 2 &
    PID_SWEEP_LLAMA3B=$!
    run_k_sweep_for_model vllm "meta-llama/Llama-3.2-1B-Instruct" 30001 3 &
    PID_SWEEP_LLAMA1B=$!
    wait $PID_SWEEP_LLAMA3B || log "WARNING: Llama-3B sweep failed"
    wait $PID_SWEEP_LLAMA1B || log "WARNING: Llama-1B sweep failed"
else
    log "ERROR: Llama vLLM servers failed"
fi

# Shutdown Llama servers
log "Shutting down Llama servers"
kill $PID_LLAMA3B $PID_LLAMA1B 2>/dev/null || true
kill_port 30000; kill_port 30001
sleep 5

# =============================================================================
# Step 2: Qwen models k-sweep
# =============================================================================
log "=== Phase 2: Qwen models k-sweep ==="

PID_QWEN3B=$(launch_vllm "Qwen/Qwen2.5-3B-Instruct" 30000 0 "logs/ksweep_vllm_qwen3b.log")
PID_QWEN15B=$(launch_vllm "Qwen/Qwen2.5-1.5B-Instruct" 30001 1 "logs/ksweep_vllm_qwen15b.log")

if wait_for_server 30000 "Qwen-3B" && wait_for_server 30001 "Qwen-1.5B"; then
    run_k_sweep_for_model vllm "Qwen/Qwen2.5-3B-Instruct" 30000 2 &
    PID_SWEEP_QWEN3B=$!
    run_k_sweep_for_model vllm "Qwen/Qwen2.5-1.5B-Instruct" 30001 3 &
    PID_SWEEP_QWEN15B=$!
    wait $PID_SWEEP_QWEN3B || log "WARNING: Qwen-3B sweep failed"
    wait $PID_SWEEP_QWEN15B || log "WARNING: Qwen-1.5B sweep failed"
else
    log "ERROR: Qwen vLLM servers failed"
fi

log "Shutting down Qwen servers"
kill $PID_QWEN3B $PID_QWEN15B 2>/dev/null || true
kill_port 30000; kill_port 30001
sleep 3

# =============================================================================
# Step 3: Summary
# =============================================================================
log "=== K-sweep Summary ==="
$AMEM_PYTHON -c "
import json, os, glob

models = {}
for f in sorted(glob.glob('results_k_sweep/*.json')):
    base = os.path.basename(f).replace('.json','')
    # e.g. llama-3.2-3b-instruct_k10
    parts = base.rsplit('_k', 1)
    model_name = parts[0]
    k_val = int(parts[1])
    d = json.load(open(f))
    overall = d['aggregate_metrics']['overall']
    f1 = overall['f1']['mean']
    bleu1 = overall['bleu1']['mean']
    if model_name not in models:
        models[model_name] = []
    models[model_name].append((k_val, f1, bleu1))

print(f'{\"Model\":<30} {\"k\":>4} {\"F1\":>8} {\"BLEU-1\":>8}')
print('-' * 55)
for model_name in sorted(models):
    results = sorted(models[model_name], key=lambda x: x[0])
    best = max(results, key=lambda x: x[1])
    for k_val, f1, bleu1 in results:
        marker = ' <-- BEST' if k_val == best[0] else ''
        print(f'{model_name:<30} {k_val:>4} {f1:>8.4f} {bleu1:>8.4f}{marker}')
    print()
"

log "=== Done ==="
