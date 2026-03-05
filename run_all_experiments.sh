#!/usr/bin/env bash
# =============================================================================
# run_all_experiments.sh — Robust evaluation of A-MEM on 7 foundation models
#
# Models:
#   OpenAI:  gpt-4o-mini, gpt-4o, gpt-4
#   Llama:   Llama-3.2-3B-Instruct, Llama-3.2-1B-Instruct  (vLLM)
#   Qwen:    Qwen2.5-3B-Instruct, Qwen2.5-1.5B-Instruct    (vLLM)
#
# Uses vLLM 0.8.5 (verl-agent-alfworld env) for Turing GPU (sm_75) compat.
# SGLang 0.5+ dropped sm_75 support in sgl_kernel.
# =============================================================================
set -euo pipefail

WORKDIR="/common/users/wx139/code/opensource_all/A-mem_opensource"
AMEM_PYTHON="/common/users/wx139/env/amem_env/bin/python3"
VLLM_PYTHON="/common/users/wx139/env/verl-agent-alfworld/bin/python"
DATASET="data/locomo10.json"

PORT_A=30000
PORT_B=30001
TP_3B=1
TP_1B=1
GPUS_3B="0"
GPUS_1B="1"

HEALTH_TIMEOUT=300
HEALTH_INTERVAL=5

cd "$WORKDIR"
mkdir -p logs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

kill_port() {
    local port=$1
    local pids
    pids=$(lsof -ti :$port 2>/dev/null || true)
    if [ -n "$pids" ]; then
        log "Killing processes on port $port: $pids"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
}

wait_for_server() {
    local port=$1
    local name=$2
    local elapsed=0
    log "Waiting for $name on port $port ..."
    while [ $elapsed -lt $HEALTH_TIMEOUT ]; do
        if curl -s "http://localhost:$port/health" >/dev/null 2>&1; then
            log "$name is ready on port $port (took ${elapsed}s)"
            return 0
        fi
        if curl -s "http://localhost:$port/v1/models" >/dev/null 2>&1; then
            log "$name is ready on port $port via /v1/models (took ${elapsed}s)"
            return 0
        fi
        sleep $HEALTH_INTERVAL
        elapsed=$((elapsed + HEALTH_INTERVAL))
    done
    log "ERROR: $name failed to start on port $port after ${HEALTH_TIMEOUT}s"
    return 1
}

launch_vllm() {
    local model=$1 port=$2 tp=$3 gpus=$4 logfile=$5
    log "Launching vLLM: model=$model tp=$tp port=$port gpus=$gpus"
    CUDA_VISIBLE_DEVICES=$gpus \
    VLLM_USE_V1=0 \
    VLLM_ATTENTION_BACKEND=XFORMERS \
        $VLLM_PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --port "$port" \
        --host 0.0.0.0 \
        --tensor-parallel-size "$tp" \
        --trust-remote-code \
        --dtype float16 \
        --enforce-eager \
        --gpu-memory-utilization 0.90 \
        --max-model-len 4096 \
        > "$logfile" 2>&1 &
    echo $!
}

run_eval() {
    local backend=$1 model=$2 output=$3 port=${4:-30000} gpu=${5:-7}
    log "Starting eval: backend=$backend model=$model output=$output gpu=$gpu"
    CUDA_VISIBLE_DEVICES=$gpu \
    $AMEM_PYTHON test_advanced_robust.py \
        --backend "$backend" \
        --model "$model" \
        --dataset "$DATASET" \
        --output "$output" \
        --sglang_port "$port" \
        2>&1 | tee "logs/eval_$(basename "$output" .json).log"
    log "Finished eval: $output"
}

# =============================================================================
# STEP 0: Clean up
# =============================================================================
log "=== Step 0: Cleaning up existing servers ==="
kill_port $PORT_A
kill_port $PORT_B

# =============================================================================
# STEP 1: OpenAI models (no GPU needed, run in parallel)
# =============================================================================
log "=== Step 1: OpenAI models (gpt-4o-mini, gpt-4o, gpt-4) ==="

# OpenAI evals use SentenceTransformer on GPU; assign GPUs 5,6,7 to avoid vLLM clash
( run_eval openai gpt-4o-mini results_robust_gpt-4o-mini.json 30000 5 ) &
PID_GPT4O_MINI=$!

( run_eval openai gpt-4o results_robust_gpt-4o.json 30000 6 ) &
PID_GPT4O=$!

( run_eval openai gpt-4 results_robust_gpt-4.json 30000 7 ) &
PID_GPT4=$!

log "OpenAI evals backgrounded: gpt-4o-mini=$PID_GPT4O_MINI, gpt-4o=$PID_GPT4O, gpt-4=$PID_GPT4"

# =============================================================================
# STEP 2: Llama models (vLLM)
# =============================================================================
log "=== Step 2: Llama models ==="

PID_VLLM_LLAMA3B=$(launch_vllm \
    "meta-llama/Llama-3.2-3B-Instruct" $PORT_A $TP_3B "$GPUS_3B" \
    "logs/vllm_llama3.2-3b.log")

PID_VLLM_LLAMA1B=$(launch_vllm \
    "meta-llama/Llama-3.2-1B-Instruct" $PORT_B $TP_1B "$GPUS_1B" \
    "logs/vllm_llama3.2-1b.log")

log "vLLM servers launching: Llama-3B=$PID_VLLM_LLAMA3B, Llama-1B=$PID_VLLM_LLAMA1B"

if wait_for_server $PORT_A "Llama-3.2-3B" && wait_for_server $PORT_B "Llama-3.2-1B"; then
    ( run_eval vllm "meta-llama/Llama-3.2-3B-Instruct" \
        results_robust_llama3.2-3b.json $PORT_A 2 ) &
    PID_EVAL_LLAMA3B=$!

    ( run_eval vllm "meta-llama/Llama-3.2-1B-Instruct" \
        results_robust_llama3.2-1b.json $PORT_B 3 ) &
    PID_EVAL_LLAMA1B=$!

    wait $PID_EVAL_LLAMA3B || log "WARNING: Llama-3B eval failed"
    wait $PID_EVAL_LLAMA1B || log "WARNING: Llama-1B eval failed"
else
    log "ERROR: Llama vLLM servers failed to start."
fi

log "Shutting down Llama vLLM servers"
kill $PID_VLLM_LLAMA3B 2>/dev/null || true
kill $PID_VLLM_LLAMA1B 2>/dev/null || true
kill_port $PORT_A
kill_port $PORT_B
sleep 3

# =============================================================================
# STEP 3: Qwen models (vLLM)
# =============================================================================
log "=== Step 3: Qwen models ==="

PID_VLLM_QWEN3B=$(launch_vllm \
    "Qwen/Qwen2.5-3B-Instruct" $PORT_A $TP_3B "$GPUS_3B" \
    "logs/vllm_qwen2.5-3b.log")

PID_VLLM_QWEN1_5B=$(launch_vllm \
    "Qwen/Qwen2.5-1.5B-Instruct" $PORT_B $TP_1B "$GPUS_1B" \
    "logs/vllm_qwen2.5-1.5b.log")

log "vLLM servers launching: Qwen-3B=$PID_VLLM_QWEN3B, Qwen-1.5B=$PID_VLLM_QWEN1_5B"

if wait_for_server $PORT_A "Qwen2.5-3B" && wait_for_server $PORT_B "Qwen2.5-1.5B"; then
    ( run_eval vllm "Qwen/Qwen2.5-3B-Instruct" \
        results_robust_qwen2.5-3b.json $PORT_A 2 ) &
    PID_EVAL_QWEN3B=$!

    ( run_eval vllm "Qwen/Qwen2.5-1.5B-Instruct" \
        results_robust_qwen2.5-1.5b.json $PORT_B 3 ) &
    PID_EVAL_QWEN1_5B=$!

    wait $PID_EVAL_QWEN3B || log "WARNING: Qwen-3B eval failed"
    wait $PID_EVAL_QWEN1_5B || log "WARNING: Qwen-1.5B eval failed"
else
    log "ERROR: Qwen vLLM servers failed to start."
fi

log "Shutting down Qwen vLLM servers"
kill $PID_VLLM_QWEN3B 2>/dev/null || true
kill $PID_VLLM_QWEN1_5B 2>/dev/null || true
kill_port $PORT_A
kill_port $PORT_B

# =============================================================================
# STEP 4: Wait for OpenAI evals
# =============================================================================
log "=== Step 4: Waiting for OpenAI evals ==="
wait $PID_GPT4O_MINI || log "WARNING: gpt-4o-mini eval failed"
wait $PID_GPT4O || log "WARNING: gpt-4o eval failed"
wait $PID_GPT4 || log "WARNING: gpt-4 eval failed"

# =============================================================================
# Summary
# =============================================================================
log "=== All experiments complete ==="
log ""
log "Results files:"
for f in results_robust_*.json; do
    if [ -f "$f" ]; then
        log "  $f ($(wc -c < "$f") bytes)"
    fi
done

log ""
log "Quick metric summary (F1 and BLEU-1):"
$AMEM_PYTHON -c "
import json, glob, sys
files = sorted(glob.glob('results_robust_*.json'))
# skip test files
files = [f for f in files if 'test10' not in f]
if not files:
    print('  No result files found.')
    sys.exit(0)
print(f'  {\"Model\":<45} {\"F1\":>8} {\"BLEU-1\":>8}')
print('  ' + '-'*63)
for f in files:
    try:
        d = json.load(open(f))
        agg = d.get('aggregate_metrics', {})
        overall = agg.get('overall', {})
        f1 = overall.get('f1', {}).get('mean', -1)
        bleu1 = overall.get('bleu1', {}).get('mean', -1)
        model = d.get('model', f)
        print(f'  {model:<45} {f1:>8.4f} {bleu1:>8.4f}')
    except Exception as e:
        print(f'  {f}: error reading - {e}')
"

log "Done!"
